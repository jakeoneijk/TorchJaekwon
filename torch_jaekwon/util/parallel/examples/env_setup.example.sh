#!/bin/bash
# EXAMPLE cluster contract for run_parallel_tasks.sh. Copy this into your project,
# replace the PLACEHOLDER cluster values, and point TJ_CLUSTER_ENV at your copy.
# It must provide TJ_PYTHON + tj_submit_wave (see ../README.md). Sourced by the
# driver (login node) and used to launch each worker.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TJ_PYTHON="${TJ_PYTHON:-python}"              # interpreter (login node + jobs)
export TJ_REPO="$HERE"                                # repo root; here = this examples dir
# Make the task module importable on the login node (count/wipe) regardless of CWD.
export PYTHONPATH="$TJ_REPO:${PYTHONPATH:-}"
# torch_jaekwon install dir, so "$TJ_PKG/util/parallel/..." resolves.
export TJ_PKG="${TJ_PKG:-$("$TJ_PYTHON" -c 'import torch_jaekwon, os; print(os.path.dirname(torch_jaekwon.__file__))')}"

# --- backend selection ------------------------------------------------------
# TJ_BACKEND=local -> run on this machine (no scheduler); anything else -> the
# cluster's sbatch backend defined below.
if [[ "${TJ_BACKEND:-slurm}" == "local" ]]; then
  # shellcheck disable=SC1091
  source "$TJ_PKG/util/parallel/backends/local.sh"
else
  # PLACEHOLDER SLURM backend -- replace partition/account and the submit wrapper with
  # your cluster's. Submits <njobs> independent 1-GPU jobs, each running the worker with
  # (module, python, repo) as space-separated args (NO commas -- SLURM splits on them).
  tj_submit_wave() {   # job_name njobs hours module
    local job=$1 njobs=$2 hours=$3 module=$4
    local worker="$TJ_PKG/util/parallel/run_one_worker.sh"
    sbatch \
      --array="1-$njobs" --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
      --partition="PLACEHOLDER_PARTITION" --account="PLACEHOLDER_ACCOUNT" \
      --time="$hours:00:00" --job-name="$job" \
      your_sbatch_wrapper.sh "$module $TJ_PYTHON $TJ_REPO"
  }
fi
