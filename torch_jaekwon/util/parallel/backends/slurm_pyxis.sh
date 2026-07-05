#!/bin/bash
# [backend] sbatch + pyxis/enroot · flow map: run_parallel_tasks.sh
# Submits one wave as an `sbatch --array` of independent 1-GPU jobs, each entering the
# container via run_in_container.sh. Selected by TJ_BACKEND=slurm_pyxis (or -b).
#
# Needs from the env file: TJ_PYTHON TJ_REPO TJ_PKG TJ_ACCOUNT TJ_PARTITION TJ_IMAGE
#   TJ_MOUNTS (may contain commas) · optional TJ_LOG_DIR (default $TJ_REPO/artifacts/slurm_log)

tj_submit_wave() {   # job_name njobs hours module [app args...]
  local job=$1 njobs=$2 hours=$3 module=$4; shift 4
  local log="${TJ_LOG_DIR:-$TJ_REPO/artifacts/slurm_log}/$job"; mkdir -p "$log"
  # image/mounts/worker/cmdline go POSITIONALLY to run_in_container.sh (never --export:
  # SLURM splits --export on commas, and TJ_MOUNTS has commas).
  sbatch \
    --array="1-$njobs" --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
    --partition="$TJ_PARTITION" --account="$TJ_ACCOUNT" \
    --time="$hours:00:00" --job-name="$job" \
    --output="$log/%A_%a.out" --error="$log/%A_%a.err" \
    "$TJ_PKG/util/parallel/worker/run_in_container.sh" \
    "$TJ_IMAGE" "$TJ_MOUNTS" "$TJ_PKG/util/parallel/worker/run_one_worker.sh" \
    "-m $module -p $TJ_PYTHON -r $TJ_REPO -- $*"
}
