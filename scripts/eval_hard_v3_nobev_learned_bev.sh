#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
OUT=${1:?Usage: eval_hard_v3_nobev_learned_bev.sh OUTPUT_DIRECTORY}
RUN_ROOT="$ROOT/logs/rsl_rl/g1_sru_hard_encounter_v3_cylinder"
NOBEV="$RUN_ROOT/2026-09-30_09-56-26_hard_v3_cylinder_nobev_static3000_dynamic3000_mixed50_seed72/model_5999.pt"
BEV="$RUN_ROOT/2026-10-02_18-33-19_hard_v3_cylinder_learned_bev_query_static3000_dynamic3000_mixed50_seed72/model_5999.pt"

test -s "$NOBEV"
test -s "$BEV"
mkdir -p "$OUT"

run_case() {
    local name="$1" task="$2" checkpoint="$3" condition="$4"
    local condition_args=(--disable_dynamic_obstacles)
    if [[ "$condition" == dynamic ]]; then
        condition_args=(--force_dynamic_obstacles --force_obstacle_difficulty 1.0 --obstacle_env_fraction 1.0)
    fi
    local common=(--task "$task" --checkpoint "$checkpoint" --seed 72 --headless --sru_ref_success)
    printf '%s START %s statistics\n' "$(date -Is)" "$name"
    "$PYTHON" "$ROOT/_launcher.py" eval "${common[@]}" "${condition_args[@]}" \
        --num_envs 4096 --num_episodes 19200 --print_every 1024 \
        --json_out "$OUT/${name}.json" > "$OUT/${name}.stdout.log" 2>&1
    test -s "$OUT/${name}.json"
    printf '%s START %s failure GIF collection\n' "$(date -Is)" "$name"
    "$PYTHON" "$ROOT/_launcher.py" eval "${common[@]}" "${condition_args[@]}" \
        --num_envs 64 --num_episodes 512 --print_every 128 \
        --save_failure_gifs --failure_gif_folder "$OUT/${name}_failure_gifs" \
        --max_failure_gifs 10 --max_success_gifs 0 --no_failure_rgb_include_successes \
        --failure_rgb_full_episode --failure_rgb_gif_frame_ms 200 \
        --json_out "$OUT/${name}_gif_collection.json" > "$OUT/${name}_gifs.stdout.log" 2>&1
    test -s "$OUT/${name}_gif_collection.json"
    "$PYTHON" -c 'from pathlib import Path; import sys; files=list(Path(sys.argv[1]).glob("*.gif")); print("GIF count:", len(files)); assert len(files)==10, f"Expected 10 failure GIFs, found {len(files)}"' "$OUT/${name}_failure_gifs"
    printf '%s COMPLETE %s\n' "$(date -Is)" "$name"
}

NOBEV_TASK=Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV3Cylinder-OfficialParkour-v0
BEV_TASK=Instinct-Nav-G1-SRU-LearnedBEVQuery-HardEncounterV3Cylinder-OfficialParkour-v0
run_case nobev_static "$NOBEV_TASK" "$NOBEV" static
run_case learned_bev_static "$BEV_TASK" "$BEV" static
run_case nobev_dynamic100 "$NOBEV_TASK" "$NOBEV" dynamic
run_case learned_bev_dynamic100 "$BEV_TASK" "$BEV" dynamic
printf '%s ALL COMPLETE %s\n' "$(date -Is)" "$OUT"
