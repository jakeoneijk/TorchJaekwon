source script/set_path.sh

#CONFIG_PATH="./config/.yaml"
#CUDA_VISIBLE_DEVICES=0 \
python main.py \
--stage train \
--project_name "current_project_name" \
--log_tool wandb \
--debug_mode false \
#--config_path ${CONFIG_PATH} \
#-r