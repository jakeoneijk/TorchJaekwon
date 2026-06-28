#!/bin/bash
# Backend-agnostic inner worker for ANY ParallelTaskProcessor subclass. One of these
# runs per GPU -- launched by srun inside the container (cluster backend) or as a bare
# background process (local backend). It just runs `python -m <module> run`, which
# races the shared task list via atomic claims (see ParallelTaskProcessor).
#
# Everything it needs arrives as POSITIONAL ARGS (space-separated, NO commas):
#   $1 = python module       (e.g. src.preprocess.fisher)
#   $2 = python interpreter   (e.g. /.../envs/ntd/bin/python)
#   $3 = repo root            (added to PYTHONPATH; the module resolves from here)
# We deliberately do NOT rely on env propagation into the container: SLURM splits
# --export on commas (even --export=ALL fails on some clusters), so the cluster's
# tj_submit_wave passes these as a space-separated arg string instead.
set -euo pipefail

MODULE="${1:?usage: run_one_worker.sh <module> <python> <repo>}"
PYTHON="${2:?missing python interpreter arg}"
REPO="${3:?missing repo root arg}"
cd "$REPO"

# Default to HF offline: from_pretrained() calls (parakeet / semamba / voxcpm /
# tokenizers) then use the warm cache instead of stalling on hub network retries on a
# compute node with no/firewalled internet (a silent slow-import, not an error).
# Overridable -- set HF_HUB_OFFLINE=0 for a run that must download (e.g. first-time local).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Worker identity. Prefer a backend-neutral TJ_WORKER_ID (set by the local backend);
# fall back to SLURM's per-task rank, then this shell's PID -- so the SAME worker runs
# unchanged under any backend (scheduler or bare background processes).
WORKER_ID="${TJ_WORKER_ID:-${SLURM_PROCID:-$$}}"
JOB_ID="${SLURM_JOB_ID:-local}"

# Per-worker LOCAL caches so concurrent workers don't contend on a shared filesystem
# lock (Triton/Inductor/Numba/HF jit caches were a multi-node stall source). Keyed on
# (job, worker) so every worker -- on any backend -- gets its own.
CACHE_BASE="/tmp/parallel_task_cache_${JOB_ID}_${WORKER_ID}"
mkdir -p "$CACHE_BASE"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/inductor"
export NUMBA_CACHE_DIR="$CACHE_BASE/numba"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"

# Identity for the Python side (informational; the atomic-claim race needs no ranks).
# Each prefers its neutral override, then SLURM, then a safe default.
export LOCAL_RANK="${TJ_WORKER_ID:-${SLURM_LOCALID:-0}}"
export RANK="${TJ_WORKER_ID:-${SLURM_PROCID:-0}}"
export WORLD_SIZE="${TJ_WORLD_SIZE:-${SLURM_NTASKS:-1}}"

echo "[worker] host=$(hostname) module=$MODULE WORKER_ID=$WORKER_ID RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE"
"$PYTHON" -m "$MODULE" run
