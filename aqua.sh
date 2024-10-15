#!/bin/bash

# CUDA_VISIBLE_DEVICES=6 python main_aqua.py --max_turns 1 --pretrained_model_path Qwen/Qwen2-7B-Instruct -e 2000
# CUDA_VISIBLE_DEVICES=6 python main_aqua.py --max_turns 2 --pretrained_model_path Qwen/Qwen2-7B-Instruct -e 2000
# CUDA_VISIBLE_DEVICES=6 python main_aqua.py --max_turns 3 --pretrained_model_path Qwen/Qwen2-7B-Instruct -e 2000
CUDA_VISIBLE_DEVICES=6 python main_aqua.py --max_turns 5 --pretrained_model_path Qwen/Qwen2-7B-Instruct