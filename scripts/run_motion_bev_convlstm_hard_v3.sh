#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
NAME="motion_bev_convlstm_static3000_dynamic3000_mixed50_seed72_${STAMP}"
mkdir -p "$ROOT/logs/background"
exec "$PYTHON" "$ROOT/_launcher.py" train \
    --task Instinct-Nav-G1-SRU-MotionBEVConvLSTM-HardV3-OfficialParkour-v0 \
    --num_envs 4096 --max_iterations 6000 --run_name "$NAME" \
    --headless --seed 72 --wandb --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project ablation_sru > "$ROOT/logs/background/${NAME}.log" 2>&1
