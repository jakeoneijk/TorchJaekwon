source .script/set_path.sh

#CONFIG_PATH="./config/.yaml"
#CUDA_VISIBLE_DEVICES=0 \
python main.py \
--stage preprocess \
--num_workers 10 \
#--config_path ${CONFIG_PATH}