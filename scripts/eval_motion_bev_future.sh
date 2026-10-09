#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/workspace/instinct_nav
PYTHON=/workspace/instinctlab51-env/bin/python
OUT=${1:?Usage: eval_motion_bev_future.sh NEW_OUTPUT_DIRECTORY}
TASK=Instinct-Nav-G1-SRU-MotionBEVFuture-HardV3-OfficialParkour-v0
CHECKPOINT="$ROOT/logs/rsl_rl/g1_sru_motion_bev_hard_v3/2026-10-05_17-40-58_motion_bev_future_static3000_dynamic3000_mixed50_seed72_20261005_173708/model_5999.pt"

test -s "$CHECKPOINT"
mkdir "$OUT"
common=(--task "$TASK" --checkpoint "$CHECKPOINT" --seed 72 --terrain_seed 72 --headless --sru_ref_success)

run_statistics() {
    local name="$1"
    shift
    printf '%s START %s statistics\n' "$(date -Is)" "$name"
    "$PYTHON" "$ROOT/_launcher.py" eval "${common[@]}" "$@" \
        --num_envs 4096 --num_episodes 19200 --print_every 1024 \
        --no_failure_rgb_include_observations \
        --json_out "$OUT/${name}.json" > "$OUT/${name}.stdout.log" 2>&1
    test -s "$OUT/${name}.json"
}

run_gifs() {
    local name="$1"
    shift
    printf '%s START %s complete failure GIFs\n' "$(date -Is)" "$name"
    "$PYTHON" "$ROOT/_launcher.py" eval "${common[@]}" "$@" \
        --num_envs 64 --num_episodes 512 --print_every 128 \
        --save_failure_gifs --failure_gif_folder "$OUT/${name}_failure_gifs" \
        --max_failure_gifs 10 --max_success_gifs 0 --no_failure_rgb_include_successes \
        --failure_rgb_full_episode --failure_rgb_gif_frame_ms 200 \
        --json_out "$OUT/${name}_gif_collection.json" > "$OUT/${name}_gifs.stdout.log" 2>&1
    test -s "$OUT/${name}_gif_collection.json"
    "$PYTHON" -c 'from pathlib import Path; import sys; files=list(Path(sys.argv[1]).glob("*.gif")); print("GIF count:", len(files)); assert len(files)==10, f"Expected 10 failure GIFs, found {len(files)}"' "$OUT/${name}_failure_gifs"
}

dynamic=(--force_dynamic_obstacles --force_obstacle_difficulty 1.0 --obstacle_env_fraction 1.0)
run_statistics motion_bev_future_static --disable_dynamic_obstacles
run_statistics motion_bev_future_dynamic100 "${dynamic[@]}"
run_gifs motion_bev_future_static --disable_dynamic_obstacles
run_gifs motion_bev_future_dynamic100 "${dynamic[@]}"
printf '%s ALL COMPLETE %s\n' "$(date -Is)" "$OUT"
