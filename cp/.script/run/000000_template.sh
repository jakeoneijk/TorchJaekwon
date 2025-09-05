set -euo pipefail

RUNNER=("python")
#RUNNER=("torchrun" "--standalone" "--nproc_per_node=1")
#CUDA_VISIBLE_DEVICES="0,1"

PROJECT_NAME="template_project"

CONFIG_PATH="config/000000_template.yaml"

STAGE="preprocess"
#STAGE="train"
#STAGE="inference"
#STAGE="evaluate"

if [[ "$STAGE" == "preprocess" ]]; then
  ARGS=(
    --num_workers 10
  )
elif [[ "$STAGE" == "train" ]]; then
  ARGS=(
    --project_name "$PROJECT_NAME"
    --log_tool wandb
    --debug_mode false
    #-r
  )
elif [[ "$STAGE" == "inference" ]]; then
  ARGS=(
    #--ckpt_name "step.pth"
  )
elif [[ "$STAGE" == "evaluate" ]]; then
  ARGS=(
    #--eval_dir_path_pred "artifacts/inference/"
  )
fi

source .script/set_path.sh
#CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"${RUNNER[@]}" main.py --stage "$STAGE" --config_path "$CONFIG_PATH" "${ARGS[@]}"