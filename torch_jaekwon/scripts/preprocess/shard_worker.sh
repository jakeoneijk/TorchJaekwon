#!/bin/bash
# Cluster-agnostic inner worker for ANY ShardPreprocessor subclass. srun runs one
# of these per GPU (inside the container); it just runs `python -m <module> run`,
# which races the shared shard list via atomic claims (see ShardPreprocessor).
#
# Everything it needs arrives as POSITIONAL ARGS (space-separated, NO commas):
#   $1 = python module       (e.g. src.preprocess.fisher)
#   $2 = python interpreter   (e.g. /.../envs/ntd/bin/python)
#   $3 = repo root            (added to PYTHONPATH; the module resolves from here)
# We deliberately do NOT rely on env propagation into the container: SLURM splits
# --export on commas (even --export=ALL fails on some clusters), so the cluster's
# tj_submit_wave passes these as a space-separated arg string instead.
set -euo pipefail

MODULE="${1:?usage: shard_worker.sh <module> <python> <repo>}"
PYTHON="${2:?missing python interpreter arg}"
REPO="${3:?missing repo root arg}"
cd "$REPO"

# Stay offline on compute nodes (avoid HF network retries that stall imports).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Per-rank LOCAL caches so many ranks don't contend on a shared filesystem lock
# (Triton/Inductor/Numba/HF jit caches were a multi-node stall source).
CACHE_BASE="/tmp/shard_cache_${SLURM_JOB_ID:-0}_${SLURM_PROCID:-0}"
mkdir -p "$CACHE_BASE"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/inductor"
export NUMBA_CACHE_DIR="$CACHE_BASE/numba"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"

# GPU index for this rank on its node + identity (workers read these).
export LOCAL_RANK="${SLURM_LOCALID:-0}"
export RANK="${SLURM_PROCID:-0}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"

echo "[shard_worker] node=$(hostname) module=$MODULE RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE"
"$PYTHON" -m "$MODULE" run
