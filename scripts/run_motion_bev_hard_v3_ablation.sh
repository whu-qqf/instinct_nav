#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs/background"
COMMON=(--num_envs 4096 --max_iterations 6000 --headless --seed 72
        --wandb --wandb_entity qqfqqf0612-wuhan-university --wandb_project ablation_sru)

run_case() {
    local task="$1" name="${2}_${STAMP}"
    "$PYTHON" "$ROOT/_launcher.py" train --task "$task" --run_name "$name" \
        "${COMMON[@]}" > "$ROOT/logs/background/${name}.log" 2>&1
}

run_case Instinct-Nav-G1-SRU-NoBEVRewardMatched-HardV3-OfficialParkour-v0 nobev_reward_matched_seed72
run_case Instinct-Nav-G1-SRU-MotionBEVQuery-HardV3-OfficialParkour-v0 motion_bev_query_seed72
