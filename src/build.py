from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# [修复] 从 .modules 直接导入，避免与 src/__init__.py 发生循环依赖
from .modules import (
    UNet,
    StyleEncoder,
    ContentEncoder
)


def build_unet(args):
    # ================= [修改开始] 安全获取参数 =================
    # 使用 getattr(args, '属性名', None) 防止 AttributeError

    # 1. 处理 down_block_types
    arg_down = getattr(args, 'down_block_types', None)
    down_block_types = tuple(arg_down) if arg_down else \
        ("DownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "DownBlock2D")

    # 2. 处理 up_block_types
    arg_up = getattr(args, 'up_block_types', None)
    up_block_types = tuple(arg_up) if arg_up else \
        ("MemoryAwareUpBlock2D", "MemoryAwareUpBlock2D", "MemoryAwareUpBlock2D", "UpBlock2D")

    # 3. 处理 block_out_channels
    arg_channels = getattr(args, 'block_out_channels', None)
    block_out_channels = tuple(arg_channels) if arg_channels else \
        (32, 64, 128, 256)
    # ================= [修改结束] =================

    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32)

    return unet


def build_style_encoder(args):
    # 简单的兼容处理，以防 args.style_image_size 是列表
    res = args.style_image_size
    if isinstance(res, (list, tuple)):
        res = res[0]

    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=res)
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    # 简单的兼容处理
    res = args.content_image_size
    if isinstance(res, (list, tuple)):
        res = res[0]

    # 默认开启 CCAM，使用 getattr 防止报错
    use_ccam = getattr(args, 'use_ccam', True)

    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=res,
        use_ccam=use_ccam)
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon")
    return ddpm_scheduler