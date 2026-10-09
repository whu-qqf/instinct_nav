#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
LAUNCHER="$ROOT/_launcher.py"
LOG_DIR="$ROOT/logs/background"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --num_envs 4096
  --max_iterations 5000
  --headless
  --wandb
  --wandb_entity qqfqqf0612-wuhan-university
  --wandb_project ablation_sru
  --seed 72
)

run_experiment() {
  local task="$1"
  local run_name="$2"
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date --iso-8601=seconds)] START $run_name" | tee -a "$log_file"
  "$PYTHON" "$LAUNCHER" train \
    --task "$task" \
    --run_name "$run_name" \
    "${COMMON_ARGS[@]}" >>"$log_file" 2>&1
  echo "[$(date --iso-8601=seconds)] COMPLETE $run_name" | tee -a "$log_file"
}

run_experiment \
  Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV2-OfficialParkour-v0 \
  hard_v2_nobev_static1000_dynamic4000_seed72

run_experiment \
  Instinct-Nav-G1-SRU-Ablation-CurrentRiskBEV-HardEncounterV2-OfficialParkour-v0 \
  hard_v2_current_risk_static1000_dynamic4000_seed72

run_experiment \
  Instinct-Nav-G1-SRU-Ablation-SectorRiskBEV-HardEncounterV2-OfficialParkour-v0 \
  hard_v2_sector_risk_static1000_dynamic4000_seed72

