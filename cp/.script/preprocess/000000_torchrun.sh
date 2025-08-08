source script/set_path.sh

#CONFIG_PATH="./config/.yaml"
#CUDA_VISIBLE_DEVICES=0 \
torchrun \
--standalone \
--nproc_per_node=1 \
main.py \
--stage preprocess \
--num_workers 10 \
#--config_path ${CONFIG_PATH}