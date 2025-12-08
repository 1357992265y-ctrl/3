#!/bin/bash

# 设置显卡 ID
#export CUDA_VISIBLE_DEVICES=0

# ================= 配置路径 =================
# 您的项目绝对路径
PROJECT_ROOT="/home/liqun/nmy/FontDiffuser-main"
cd "$PROJECT_ROOT" || exit
# ===========================================

#!/bin/bash

# [新增] 优化显存碎片管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ... (你的环境激活代码，如果有的话) ...

accelerate launch --mixed_precision=fp16 train.py \
    --seed=123 \
    --experience_name=FontDiffuser_training_phase_1_ComponentAware \
    --data_root=/home/liqun/nmy/FontDiffuser-main/data_examples \
    --output_dir=/home/liqun/nmy/FontDiffuser-main/outputs/FontDiffuser \
    --report_to=tensorboard \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=4 \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=440000 \
    --ckpt_interval=10000 \
    --gradient_accumulation_steps=4 \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler=linear \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision=fp16
