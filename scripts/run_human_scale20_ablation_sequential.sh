#!/usr/bin/env bash

set -uo pipefail

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
LAUNCHER="$ROOT/_launcher.py"
STATE_DIR="$ROOT/logs/human_scale20_ablation_sequence"
SEQUENCE_LOG="$STATE_DIR/sequence.log"

mkdir -p "$STATE_DIR"

log_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$SEQUENCE_LOG"
}

run_training() {
    local label=$1
    local task=$2
    local run_name=$3
    local output_log="$STATE_DIR/${label}.log"

    log_status "START ${label}: task=${task} run=${run_name}"
    "$PYTHON" "$LAUNCHER" train \
        --task "$task" \
        --num_envs 4096 \
        --max_iterations 5000 \
        --run_name "$run_name" \
        --headless \
        --wandb \
        --wandb_entity qqfqqf0612-wuhan-university \
        --wandb_project ablation_sru \
        --seed 72 \
        >"$output_log" 2>&1
    local exit_code=$?
    log_status "END ${label}: exit_code=${exit_code}"
    return "$exit_code"
}

log_status "HUMAN SCALE 20x20 SEQUENCE START"

if ! run_training \
    nobev \
    Instinct-Nav-G1-SRU-Ablation-NoBEV-HumanScale20-OfficialParkour-v0 \
    sru_nobev_human_scale20_static3000_mixed50_v1_seed72; then
    log_status "SEQUENCE ABORT: NoBEV training failed; BEV training was not started"
    exit 1
fi

if ! run_training \
    bev \
    Instinct-Nav-G1-SRU-Ablation-BEV-HumanScale20-OfficialParkour-v0 \
    sru_bev_human_scale20_static3000_mixed50_v1_seed72; then
    log_status "SEQUENCE ABORT: BEV training failed"
    exit 1
fi

log_status "HUMAN SCALE 20x20 SEQUENCE COMPLETE"
