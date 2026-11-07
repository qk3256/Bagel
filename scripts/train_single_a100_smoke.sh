#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PYTHON="${ENV_PYTHON:-$HOME/miniconda3/envs/bagel/bin/python}"
TORCHRUN="${TORCHRUN:-$HOME/miniconda3/envs/bagel/bin/torchrun}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
RUN_NAME="${RUN_NAME:-single-a100-smoke}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$RUN_NAME}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BAGEL_DATA_ROOT="${BAGEL_DATA_ROOT:-$REPO_ROOT/data/bagel_example}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=1 \
  train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/single_a100_smoke.yaml \
  --llm_path models/Qwen2.5-0.5B-Instruct \
  --vit_path models/siglip-so400m-patch14-384 \
  --vae_path models/BAGEL-7B-MoT/ae.safetensors \
  --layer_module Qwen2MoTDecoderLayer \
  --use_flex True \
  --max_latent_size 16 \
  --vit_max_num_patch_per_side 16 \
  --num_workers 1 \
  --prefetch_factor 2 \
  --expected_num_tokens 1024 \
  --max_num_tokens_per_sample 1024 \
  --max_num_tokens 2048 \
  --prefer_buffer_before 512 \
  --max_buffer_size 10 \
  --sharding_strategy FULL_SHARD \
  --num_shard 1 \
  --total_steps "$TOTAL_STEPS" \
  --warmup_steps 0 \
  --log_every 1 \
  --save_every 999999 \
  --wandb_offline True \
  --results_dir "$RESULTS_DIR" \
  --checkpoint_dir "$RESULTS_DIR/checkpoints" \
  --wandb_name "$RUN_NAME"
