#!/bin/bash

# python3 download.py
# CUDA_VISIBLE_DEVICES=0,2,5,6 torchrun --nproc_per_node=4 fsdp_main_reclor.py --max_turns 1
# 定义要运行的命令及其对应的GPU
commands=(
    "CUDA_VISIBLE_DEVICES=0 python main_gsm8k.py --max_turns 1 --pretrained_model_path Qwen/Qwen2-7B-Instruct"
    "CUDA_VISIBLE_DEVICES=6 python main_gsm8k.py --max_turns 2 --pretrained_model_path Qwen/Qwen2-7B-Instruct"
    # "CUDA_VISIBLE_DEVICES=4 python main_reclor.py --max_turns 5 --pretrained_model_path Qwen/Qwen2-7B-Instruct"
    # "CUDA_VISIBLE_DEVICES=4 python main_reclor.py --max_turns 2 --pretrained_model_path Qwen/Qwen2-7B-Instruct"
    # "CUDA_VISIBLE_DEVICES=2 python main_reclor.py --max_turns 30"
    # "CUDA_VISIBLE_DEVICES=3 python main_reclor.py --max_turns 50"
    # "CUDA_VISIBLE_DEVICES=4 python main_reclor.py --max_turns 10"
)

# 循环启动所有命令，每个命令在不同的GPU上运行
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd &  # 使用 eval 来执行命令
done

# 等待所有后台任务完成
wait
