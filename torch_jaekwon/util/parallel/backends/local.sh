#!/bin/bash
# Local backend for run_parallel_tasks.sh -- NO job scheduler. Runs the wave as plain
# background processes on the current machine (one worker per visible GPU, round-robin),
# and blocks until they finish. A drop-in alternative to a cluster's sbatch-based
# tj_submit_wave: source this instead and the driver + Python run unchanged.
#
# Provides the contract the driver expects:
#   tj_submit_wave <job_name> <njobs> <hours> <module>   # 'hours' (walltime) ignored
# plus:
#   TJ_WAVE_BLOCKS=1   # submission is synchronous, so the driver loops wave->wave
#                      # until 0 leftover (one command runs to completion).
#
# Requires (set before sourcing, e.g. by the project's env_setup.sh):
#   TJ_PYTHON  TJ_PKG  TJ_REPO

export TJ_WAVE_BLOCKS=1

# Spawn the wave locally and wait. Each worker drains the shared claim list, so the
# worker COUNT is just the parallelism width -- we cap it at the number of GPUs (the
# driver's njobs is sized for the scheduler's many-small-jobs model). Leftover beyond
# one wave is mopped up by the driver's blocking loop.
tj_submit_wave() {   # job_name njobs hours module   (hours ignored: no walltime locally)
  local module=$4 njobs=$2
  local worker="$TJ_PKG/util/parallel/run_one_worker.sh"
  local ngpu=1 pids=() rc=0 i p

  # GPU count, default 1 (CPU / single device). Guard nvidia-smi: under the driver's
  # `set -o pipefail` a bare `nvidia-smi | wc -l` would abort the whole run on a box
  # without it (the pipeline inherits nvidia-smi's 127 exit).
  if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l) || ngpu=1
    (( ngpu >= 1 )) || ngpu=1
  fi
  (( njobs > ngpu )) && njobs=$ngpu
  echo "[local] launching $njobs worker(s) across $ngpu GPU(s) for module=$module"

  for (( i=0; i<njobs; i++ )); do
    CUDA_VISIBLE_DEVICES=$(( i % ngpu )) TJ_WORKER_ID="$i" TJ_WORLD_SIZE="$njobs" \
      bash "$worker" "$module" "$TJ_PYTHON" "$TJ_REPO" &
    pids+=($!)
  done

  # Block until every worker exits; surface a non-zero status if any failed.
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
