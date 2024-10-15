DATASET= 'reclor'
CAT='logic'
LEVEL='easy'
CACHE_DIR='/home/gluo/'
MODEL='Qwen2-7B-Instruct'


# 定义要运行的命令及其对应的GPU
export CUDA_VISIBLE_DEVICES=0,1
python main_${DATASET}.py \
--max_turns 5 \
--cat ${CAT} \
--level ${LEVEL} \
--cache_dir ${CACHE_DIR} \
--pretrained_model_path ${MODEL} \
-s 0
