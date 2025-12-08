#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ================= 配置路径 =================
# 您的项目绝对路径
PROJECT_ROOT="/home/liqun/nmy/FontDiffuser-main"
cd "$PROJECT_ROOT" || exit
# ===========================================

# 启动单字采样
# 注意：
# 1. --ckpt_dir: 权重文件夹路径
# 2. --style_image_path: 输入的风格图片路径
# 3. --character_input: 开启单字生成模式
# 4. --content_character: 输入的内容汉字 (例如 "隆")
# 5. --save_image_dir: 结果保存路径

python sample.py \
    --ckpt_dir="${PROJECT_ROOT}/ckpt" \
    --style_image_path="${PROJECT_ROOT}/data_examples/sampling/example_style.jpg" \
    --save_image \
    --character_input \
    --content_character="隆" \
    --save_image_dir="${PROJECT_ROOT}/outputs/sampling_result_char" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"