#!/bin/bash

# 激活 ms-swift 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ms-swift

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# 训练配置
MODEL_PATH="/data/shuang/models/Qwen2.5-7B-Instruct"
DATASET_PATH="/data/shuang/short-horizon/sft_train_messages.jsonl"
OUTPUT_DIR="/data/shuang/short-horizon/output"

# 使用 ms-swift 进行 SFT 训练 (LoRA 方式，节省显存)
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --max_length 7168 \
    --deepspeed zero3

echo "训练完成!"
