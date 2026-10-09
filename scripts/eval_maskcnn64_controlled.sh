#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
NEW_RUN="$ROOT/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-09-15_11-24-02_g1_sru_mask_cnn64_gated_add_hard_allenv_height16_v1"
OLD_RUN="$ROOT/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-09-11_15-37-00_mask_cnn_encounter_restored_rewards_4096_20260911_153005"
NEW_CHECKPOINT="$NEW_RUN/model_5999.pt"
OLD_CHECKPOINT="$OLD_RUN/model_5999.pt"
OUT="$ROOT/logs/eval_maskcnn64_controlled_20260916"
NEW_TASK=Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-v0
OLD_TASK=Instinct-Nav-G1-SRU-MaskCNN-Encounter-RestoredRewards-Banks-OfficialParkour-v0

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [training_pid]\n' "$0" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    printf 'Waiting for training PID %s before starting serial Isaac evaluations.\n' "$1"
    while kill -0 "$1" 2>/dev/null; do
        sleep 60
    done
    # Allow Isaac Sim to release its GPU and carb resources after trainer exit.
    sleep 30
fi

if [[ ! -s "$NEW_CHECKPOINT" ]]; then
    printf 'Final checkpoint missing; training may have stopped early: %s\n' "$NEW_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -s "$OLD_CHECKPOINT" ]]; then
    printf 'Comparison checkpoint missing: %s\n' "$OLD_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUT"
COMMON=(
    --num_envs 4096 --num_episodes 8192 --seed 72 --headless
    --sru_ref_success --print_every 1024
    --force_obstacle_difficulty 1.0
    --obstacle_env_fraction 1.0
    --obstacle_height_m 1.6
)

run_eval() {
    local name="$1" task="$2" checkpoint="$3" condition="$4" mask_mode="$5"
    local condition_args=(--disable_dynamic_obstacles)
    local mask_args=()
    if [[ "$condition" == hard_dynamic ]]; then
        condition_args=(--force_dynamic_obstacles)
    fi
    if [[ "$mask_mode" == zero ]]; then
        mask_args=(--zero_dynamic_mask_input)
    fi
    while pgrep -f '/workspace/instinct_nav/scripts/train_senior.py|/workspace/instinct_nav/source/instinct_nav/scripts/eval.py' >/dev/null; do
        printf '%s Waiting for another Isaac train/eval process before %s\n' "$(date -Is)" "$name"
        sleep 60
    done
    printf '%s Starting %s\n' "$(date -Is)" "$name"
    "$PYTHON" "$ROOT/_launcher.py" eval \
        --task "$task" --checkpoint "$checkpoint" \
        "${COMMON[@]}" "${condition_args[@]}" "${mask_args[@]}" \
        --json_out "$OUT/$name.json" > "$OUT/$name.stdout.log" 2>&1
    printf '%s Completed %s\n' "$(date -Is)" "$name"
}

run_eval new_static "$NEW_TASK" "$NEW_CHECKPOINT" static normal
run_eval old_static "$OLD_TASK" "$OLD_CHECKPOINT" static normal
run_eval new_hard_dynamic "$NEW_TASK" "$NEW_CHECKPOINT" hard_dynamic normal
run_eval new_hard_dynamic_zero_mask "$NEW_TASK" "$NEW_CHECKPOINT" hard_dynamic zero
run_eval old_hard_dynamic "$OLD_TASK" "$OLD_CHECKPOINT" hard_dynamic normal

printf 'All evaluation summaries are in %s\n' "$OUT"
