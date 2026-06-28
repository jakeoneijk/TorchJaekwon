#!/bin/bash
# Cluster-agnostic driver for ANY ParallelTaskProcessor subclass (run on the LOGIN
# node). All dataset logic lives in the Python subclass; all cluster specifics live
# in a sourced "contract" file (the project's env_setup.sh) -- so this script has
# ZERO cluster-specific literals and runs unchanged across projects and clusters.
#
# Contract the sourced $TJ_CLUSTER_ENV must provide:
#   TJ_PYTHON                                   # python interpreter (login + jobs)
#   tj_submit_wave <job_name> <njobs> <hours> <module> [app args...]
#       # submit <njobs> INDEPENDENT 1-GPU jobs, each running:
#       #   run_one_worker.sh -m <module> -p <TJ_PYTHON> -r <TJ_REPO> -- [app args...]
#
# Optional per-run app args: everything after `--` on this driver's CLI is forwarded
# verbatim to `python -m <module> {run,count,wipe} <app args>` (so a module can take
# argparse config instead of hardcoding it). The generic layer never interprets them.
# CONTRACT RULE: pass only values with NO spaces/commas -- they word-split through the
# scheduler; keep space/comma-bearing constants hardcoded in the launch script.
set -euo pipefail

log()   { echo "[driver] $*"; }
usage() { echo "Usage: $0 -M <python.module> [-j job_name] [-t hours] [-m max_tasks] [-- <app args>]" >&2; }

parse_args() {
  MODULE="" JOB_NAME="parallel_tasks" HOURS=4 MAX_TASKS=40
  local OPTIND=1 opt
  while getopts "M:j:t:m:" opt; do
    case $opt in
      M) MODULE="$OPTARG" ;;  j) JOB_NAME="$OPTARG" ;;
      t) HOURS="$OPTARG" ;;   m) MAX_TASKS="$OPTARG" ;;
      *) usage; exit 1 ;;
    esac
  done
  shift $((OPTIND - 1))
  [[ "${1:-}" == "--" ]] && shift   # everything after -- is forwarded to the module
  APP_ARGS=("$@")                    # opaque per-run args (e.g. --config ... --ckpt ...)
  [[ -n "$MODULE" ]] || { echo "ERROR: -M <python.module> is required (e.g. src.preprocess.fisher)" >&2; exit 1; }
}

load_cluster_contract() {
  : "${TJ_CLUSTER_ENV:?set TJ_CLUSTER_ENV=/path/to/cluster/env_setup.sh}"
  # shellcheck disable=SC1090
  source "$TJ_CLUSTER_ENV"   # provides TJ_PYTHON + tj_submit_wave
}

# Remove orphan temp dirs from prior waves. Safe ONLY between waves: a live claim
# and a dead worker's orphan look identical on disk. The `wipe` subcommand itself
# is implemented by ParallelTaskProcessor (torch_jaekwon/util/parallel/parallel_task_processor.py).
wipe_orphan_temps() { "$TJ_PYTHON" -m "$MODULE" wipe "${APP_ARGS[@]}"; }

# Leftover (not-yet-done) task count. Prints ONLY the number to stdout (the caller
# captures it), so this must never log to stdout. The `count` subcommand itself is
# implemented by ParallelTaskProcessor (torch_jaekwon/util/parallel/parallel_task_processor.py).
count_leftover() { "$TJ_PYTHON" -m "$MODULE" count "${APP_ARGS[@]}"; }

# One wave = min(leftover, MAX_TASKS) independent 1-GPU workers. The backend packs
# them onto free GPUs; each worker races the list via atomic claims and exits when
# work runs out (no allocated-idle).
submit_one_wave() {
  local left="$1" njobs=$(( left < MAX_TASKS ? left : MAX_TASKS ))
  log "submitting one wave of $njobs independent 1-GPU worker(s)"
  tj_submit_wave "$JOB_NAME" "$njobs" "$HOURS" "$MODULE" "${APP_ARGS[@]}"
}

# Async backends (e.g. sbatch): submission returns immediately, so do ONE wave and
# let the user rerun this script after it finishes (until 0 leftover).
run_async() {
  wipe_orphan_temps                       # 1. clear prior-wave orphans
  local left; left="$(count_leftover)"    # 2. how much work is left?
  log "module=$MODULE leftover=$left"
  (( left > 0 )) || { log "nothing left to process -- done."; return 0; }
  submit_one_wave "$left"                  # 3. submit
  log "wave submitted. Rerun this script to mop up leftovers (exits when 0)."
}

# Synchronous backends (e.g. local): each wave blocks until its workers exit, so loop
# wave->wave here until nothing is left -- one command runs to completion. Aborts if a
# wave makes no progress, so a backend that fails fast can't spin forever.
run_blocking() {
  local left prev=""
  while :; do
    wipe_orphan_temps                     # safe: between waves, nothing is running
    left="$(count_leftover)"
    log "module=$MODULE leftover=$left"
    (( left == 0 )) && { log "all tasks done."; return 0; }
    if [[ -n "$prev" ]] && (( left >= prev )); then
      log "ERROR: no progress in the last wave (leftover stuck at $left) -- aborting."
      return 1
    fi
    prev=$left
    submit_one_wave "$left"
  done
}

main() {
  parse_args "$@"
  load_cluster_contract                   # may set TJ_WAVE_BLOCKS (synchronous backend)
  if [[ "${TJ_WAVE_BLOCKS:-0}" == "1" ]]; then
    run_blocking
  else
    run_async
  fi
}

main "$@"
