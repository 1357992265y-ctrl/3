import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

# --- 兼容性补丁: 修复 huggingface_hub 版本冲突 ---
import huggingface_hub

if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub import hf_hub_download

    huggingface_hub.cached_download = hf_hub_download
# -----------------------------------------------

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler)
                 # [已删除] build_scr

from utils import (save_args_to_yaml,
                   x0_from_epsilon,
                   reNormalize_img,
                   normalize_mean_std)

logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():
    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    if args.phase_2:
        # Phase 2: 仅加载 Phase 1 训练好的权重，不再加载 SCR
        if args.phase_1_ckpt_dir and os.path.exists(args.phase_1_ckpt_dir):
            unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
            style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
            content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))
            print("Loaded Phase 1 weights for Phase 2 training.")
        else:
            print(f"Warning: Phase 2 enabled but checkpoint dir not found: {args.phase_1_ckpt_dir}")

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptual Loss
    perceptual_loss = ContentPerceptualLoss()

    # [已删除] Load SCR module for supervision
    # 原有的 SCR 构建和加载逻辑已彻底移除

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size,
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size,
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution),
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    # [修改] 强制关闭 scr，无论 args.phase_2 是什么
    train_font_dataset = FontDataset(
        args=args,
        phase='train',
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=False)

    # 这里的 Batch Size 需要注意，如果显存不够可以调小
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate)

    # LR Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # ================= [新增] 自动计算 num_epochs =================<br/>    
# 如果 args 中没有 num_epochs，根据 max_train_steps 自动推算<br/>    
    if not hasattr(args, 'num_epochs') or args.num_epochs is None:
        # 计算每个 Epoch 有多少个更新步
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1  # 防止除以零
        
        # 向上取整计算需要的 Epoch 数
        import math
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # ================= [新增结束] =================

    if accelerator.is_main_process:
        print("Num examples:", len(train_font_dataset))
        print("Num Epochs:", args.num_epochs)
        print("Instantaneous batch size per device:", args.train_batch_size)
        print("Total train batch size (w. parallel, distributed & accumulation):",
              args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes)
        print("Total optimization steps:", args.max_train_steps)

    # Training Loop
    global_step = 0
    for epoch in range(0, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, samples in enumerate(train_dataloader):
            # Data
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            # 获取组件列表
            component_images = samples.get("component_images", None)

            with accelerator.accumulate(model):
                # 1. Sample noise
                latents = model.unet.config.sample_size
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=target_images.device).long()

                # 2. Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # 3. Predict the noise residual
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    component_list=component_images  # 传入组件
                )

                # 4. Compute Loss
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = args.offset_coefficient * offset_out_sum

                # Content Perceptual Loss
                pred_original_sample = x0_from_epsilon(
                    noise_scheduler, noise_pred, noisy_target_images, timesteps
                )
                pred_original_sample = reNormalize_img(pred_original_sample)
                norm_pred_original_sample = normalize_mean_std(pred_original_sample)
                target_images_norm = reNormalize_img(target_images)
                norm_target_images = normalize_mean_std(target_images_norm)
                perceptual_loss_output = perceptual_loss(norm_pred_original_sample, norm_target_images)

                total_loss = diff_loss + offset_loss + args.perceptual_coefficient * perceptual_loss_output

                # [已删除] Phase 2 SCR Loss
                # 原有的 SCR Loss 计算逻辑已移除

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "loss": total_loss.detach().item(),
                    "diff_loss": diff_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Save Checkpoint
                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    save_dir = f"{args.output_dir}/global_step_{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    # Unwarp model
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.unet.state_dict(), f"{save_dir}/unet.pth")
                    torch.save(unwrapped_model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                    torch.save(unwrapped_model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                    print(f"Saved checkpoint at step {global_step}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
