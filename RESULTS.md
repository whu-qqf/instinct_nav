# Results

This file records the current frozen baseline result for `instinct_nav`. Treat this checkpoint and evaluation as the project reference point before starting new ablations.

## Golden Baseline

- Run: `2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch`
- Checkpoint: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt`
- Evaluation JSON: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999.json`
- Evaluation episodes: `500`
- Evaluation envs: `256`
- Overall success rate: `0.902` (`451 / 500`)

The JSON was generated before the final task-id cleanup, so its `task` field still says `Isaac-Nav-PPO-G1-Instinct-Play-v0`. The current project task id for reproducing the same line is `Instinct-Nav-G1-SRU-Play-v0`.

## Terrain Results

| Terrain | Episodes | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| `maze` | 151 | 134 | 0.8874 |
| `non_maze` | 96 | 84 | 0.8750 |
| `pits` | 106 | 92 | 0.8679 |
| `stairs` | 147 | 141 | 0.9592 |

## Termination Rates

| Termination | Count | Rate |
| --- | ---: | ---: |
| `early_termination` | 444 | 0.888 |
| `base_contact` | 32 | 0.064 |
| `large_pitch_angle` | 24 | 0.048 |
| `time_out` | 5 | 0.010 |
| `terrain_fall` | 0 | 0.000 |

In this task, `early_termination` is mostly the normal success termination path: the episode ends once the robot reaches the navigation goal.

## Level Results

| Level | Episodes | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| `level_0` | 79 | 66 | 0.8354 |
| `level_1` | 77 | 70 | 0.9091 |
| `level_2` | 90 | 87 | 0.9667 |
| `level_3` | 86 | 74 | 0.8605 |
| `level_4` | 75 | 67 | 0.8933 |
| `level_5` | 93 | 87 | 0.9355 |

## Artifacts

- Failure GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/failure_gifs/`
- Success/failure eval GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/eval_gifs_success_and_failure/`
- Stairs success GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_success_gifs/`
- Play video: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/play/rl-video-step-0.mp4`

## Reproduce Training

Foreground training:

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name instinct_nav_g1_sru_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav
```

Background training:

```bash
nohup /workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name instinct_nav_g1_sru_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    > /workspace/instinct_nav/logs/instinct_nav_g1_sru_independent_v1.out 2>&1 &
```

## Reproduce Evaluation

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 256 \
    --num_episodes 500 \
    --headless \
    --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999_reproduce.json \
    --save_failure_gifs \
    --max_failure_gifs 10
```

## Record Success And Failure GIFs

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 64 \
    --num_episodes 100 \
    --headless \
    --save_success_gifs \
    --save_failure_gifs \
    --max_success_gifs 10 \
    --max_failure_gifs 10 \
    --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_success_failure_gifs_reproduce.json
```

## Record Stairs Success GIFs

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 1 \
    --steps 0 \
    --headless \
    --bev_follow \
    --bev_height_m 22.0 \
    --require_terrain_name stairs \
    --min_initial_distance 6.0 \
    --max_resample_attempts 1000 \
    --gif_count 5 \
    --gif_length 500 \
    --gif_frame_ms 80 \
    --gif_dir /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_success_gifs
```

## Next Work

Do not change the main training path before saving a clean comparison baseline. The next work should be:

- Keep `model_2999.pt` as the frozen golden checkpoint.
- Export a compact table and several representative GIFs for reports.
- Add a method document explaining the high-level SRU policy, frozen low-level Instinct policy, observations, depth encoder, height encoder, action bridge, reward, and termination design.
- Run new experiments only as named ablations, not as silent edits to the golden baseline.
