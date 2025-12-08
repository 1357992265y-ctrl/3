#!/bin/bash

# 设置显卡 ID
#export CUDA_VISIBLE_DEVICES=0

# ================= 配置路径 =================
# 您的项目绝对路径
PROJECT_ROOT="/home/liqun/nmy/FontDiffuser-main"
cd "$PROJECT_ROOT" || exit
# ===========================================

# 启动第二阶段训练
# 注意：
# 1. --phase_1_ckpt_dir 应该指向第一阶段训练保存的某个 checkpoint 文件夹
#    (例如: ${PROJECT_ROOT}/outputs/FontDiffuser/FontDiffuser_training_phase_1_ComponentAware/global_step_440000)
#    这里我暂时将其指向 Phase 1 的通用输出目录，您可能需要根据实际生成的文件夹修改它。
# 2. --scr_ckpt_path 需要确保该文件存在于您的 ckpt 目录下

accelerate launch train.py \
    --seed=123 \
    --experience_name="FontDiffuser_training_phase_2_ComponentAware" \
    --data_root="${PROJECT_ROOT}/data_examples" \
    --output_dir="${PROJECT_ROOT}/outputs/FontDiffuser" \
    --report_to="tensorboard" \
    --phase_2 \
    --phase_1_ckpt_dir="${PROJECT_ROOT}/outputs/FontDiffuser/FontDiffuser_training_phase_1_ComponentAware" \
    --scr_ckpt_path="${PROJECT_ROOT}/ckpt/scr_210000.pth" \
    --sc_coefficient=0.01 \
    --num_neg=16 \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=16 \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=30000 \
    --ckpt_interval=5000 \
    --gradient_accumulation_steps=1 \
    --log_interval=50 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=1000 \
    --drop_prob=0.1 \
    --mixed_precision="no"