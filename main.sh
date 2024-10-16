DATASET='aqua'
CAT='math'
LEVEL='easy'
CACHE_DIR='/mnt/nvme/guoqing'
MODEL='meta-llama/Meta-Llama-3-70B-Instruct'
MAX_NEW_TOKENS=512

# 定义要运行的命令及其对应的GPU
export CUDA_VISIBLE_DEVICES=0,2,5,6

# echo "python main_${DATASET}.py \
#   --max_turns 5 \
#   --cat ${CAT} \
#   --level ${LEVEL} \
#   --cache_dir ${CACHE_DIR} \
#   --dataset ${DATASET} \
#   --pretrained_model_path ${MODEL} \
#   --max_new_tokens ${MAX_NEW_TOKENS} \
#   -s 1700 \
#   -e 3000"

python main_${DATASET}.py \
  --max_turns 5 \
  --cat ${CAT} \
  --level ${LEVEL} \
  --cache_dir ${CACHE_DIR} \
  --dataset ${DATASET} \
  --pretrained_model_path ${MODEL} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  -e 3000

