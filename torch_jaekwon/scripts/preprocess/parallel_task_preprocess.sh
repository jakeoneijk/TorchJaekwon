#!/bin/bash
# Cluster-agnostic driver for ANY ParallelTaskPreprocessor subclass (run on the LOGIN
# node). All dataset logic lives in the Python subclass; all cluster specifics
# live in a sourced "contract" file (the project's env_setup.sh). This script
# itself contains ZERO cluster-specific literals, so it is reusable across
# projects AND clusters -- clone torch_jaekwon elsewhere, write that cluster's
# contract, and this runs unchanged.
#
# Contract the sourced $TJ_CLUSTER_ENV must provide:
#   TJ_PYTHON                                   # python interpreter (login + jobs)
#   tj_submit_wave <job_name> <njobs> <hours> <module>
#       # submit <njobs> INDEPENDENT 1-node/1-GPU jobs, each running the worker
#       # parallel_task_worker.sh with (module, TJ_PYTHON, TJ_REPO) as positional args.
#
# CONTRACT RULE: tj_submit_wave must NOT pass comma-containing values as args or
# via --export -- SLURM splits on commas (even --export=ALL fails on some
# clusters). Keep comma-bearing constants (e.g. container MOUNTS) hardcoded in
# the cluster's own launch script.
#
# Strategy (see ParallelTaskPreprocessor docstring for the full rationale):
#   1. `python -m M wipe`  -> remove orphan temp dirs from prior waves (safe only
#      between waves: a live claim and a dead worker's orphan look identical).
#   2. `python -m M count` -> leftover tasks (login-safe). Exit if zero.
#   3. tj_submit_wave -> one wave of min(leftover, MAX_TASKS) independent 1-GPU
#      jobs; the scheduler packs them onto free GPUs; each worker races the list
#      via atomic claims and exits when work runs out (no allocated-idle).
#   4. Rerun this script until it reports 0 leftover.
set -euo pipefail

: "${TJ_CLUSTER_ENV:?set TJ_CLUSTER_ENV=/path/to/cluster/env_setup.sh}"
# shellcheck disable=SC1090
source "$TJ_CLUSTER_ENV"

MODULE=""
JOB_NAME="parallel_task_preprocess"
HOURS=4
MAX_TASKS=40

OPTIND=1
while getopts "M:j:t:m:" opt; do
  case $opt in
    M) MODULE="$OPTARG" ;;
    j) JOB_NAME="$OPTARG" ;;
    t) HOURS="$OPTARG" ;;
    m) MAX_TASKS="$OPTARG" ;;
    *) echo "Usage: $0 -M <python.module> [-j job_name] [-t hours] [-m max_tasks]"; exit 1 ;;
  esac
done
[[ -n "$MODULE" ]] || { echo "ERROR: -M <python.module> is required (e.g. src.preprocess.fisher)"; exit 1; }

# 1. wipe orphan temps + 2. count leftover (both dataset-aware + login-safe)
"$TJ_PYTHON" -m "$MODULE" wipe
LEFT="$("$TJ_PYTHON" -m "$MODULE" count)"
echo "[driver] module=$MODULE leftover=$LEFT"
if [[ "$LEFT" -eq 0 ]]; then
  echo "[driver] nothing left to process -- done."
  exit 0
fi

# 3. submit one wave (cluster does the actual sbatch/srun/container launch)
NJOBS=$(( LEFT < MAX_TASKS ? LEFT : MAX_TASKS ))
echo "[driver] submitting one wave of $NJOBS independent 1-GPU job(s)"
tj_submit_wave "$JOB_NAME" "$NJOBS" "$HOURS" "$MODULE"
echo "[driver] wave submitted. After it finishes, rerun this script to mop up leftovers (exits when 0)."
