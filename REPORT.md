# Project Report

This report is a compact presentation version of the current `instinct_nav` baseline. For implementation details, see [METHOD.md](./METHOD.md) and [CodeReading.md](./CodeReading.md). For exact commands and full metrics, see [RESULTS.md](./RESULTS.md).

## 1. Goal

`instinct_nav` aims to build a G1 humanoid autonomous navigation system by combining:

- SRU-style high-level visual navigation.
- Frozen InstinctLab G1 parkour low-level locomotion.

The high-level policy learns where and how to move. The frozen low-level policy executes the commanded SE(2) velocity with full-body humanoid control.

## 2. System

```text
Depth camera + proprioception + goal command
        |
        v
High-level ActorCriticSRU, 5 Hz
        |
        v
SE(2) velocity command [vx, vy, wz]
        |
        v
Frozen InstinctLab G1 parkour policy, 50 Hz
        |
        v
29D joint position target
```

Important design choices:

- High-level action is only `[vx, vy, wz]`, not joint actions.
- Low-level G1 parkour checkpoint is frozen.
- Actor uses noisy/delayed proprioception, goal and encoded depth.
- Critic uses privileged height scan and clean depth.
- Depth feature shape is fixed as `[64, 5, 8]`.
- Height feature shape is fixed as `[64, 7, 7]`.
- Visibility map/query attention is off for the current frozen baseline.

## 3. Current Baseline

Golden run:

```text
2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch
```

Golden checkpoint:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt
```

Evaluation:

```text
episodes: 500
envs: 256
successes: 451
success_rate: 0.902
```

## 4. Terrain Results

| Terrain | Episodes | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| `maze` | 151 | 134 | 0.8874 |
| `non_maze` | 96 | 84 | 0.8750 |
| `pits` | 106 | 92 | 0.8679 |
| `stairs` | 147 | 141 | 0.9592 |

The stairs result is currently the strongest terrain-specific result. The lower success rates on `pits`, `non_maze`, and some maze cases suggest the remaining failures are mostly terrain-stability and recovery issues rather than a complete navigation failure.

## 5. Failure Modes

| Termination | Count | Rate |
| --- | ---: | ---: |
| `base_contact` | 32 | 0.064 |
| `large_pitch_angle` | 24 | 0.048 |
| `time_out` | 5 | 0.010 |
| `terrain_fall` | 0 | 0.000 |

`early_termination` is not listed as a failure mode here because reaching the goal also triggers early termination in this task.

Main residual problems:

- `base_contact`: torso touches terrain, usually after unstable foot placement or collision.
- `large_pitch_angle`: robot tilt exceeds the safe threshold.
- `time_out`: rare; indicates the robot did not reach the goal within the episode horizon.

## 6. Representative Artifacts

Success and failure examples:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/eval_gifs_success_and_failure/
```

Stairs success examples:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_success_gifs/
```

Failure examples:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/failure_gifs/
```

Recommended clips for presentation:

- `videos/stairs_success_gifs/stairs_000.gif`
- `videos/stairs_success_gifs/stairs_001.gif`
- `videos/stairs_success_gifs/stairs_002.gif`
- `videos/eval_gifs_success_and_failure/ep_0005__success__maze__level_4__type_1__obsdelay_lv1_av1_pg2_tp2_d1.gif`
- `videos/eval_gifs_success_and_failure/ep_0007__success__pits__level_1__type_27__obsdelay_lv0_av1_pg0_tp1_d1.gif`
- `videos/failure_gifs/ep_0021__base_contact__stairs__level_2__type_18__obsdelay_lv2_av2_pg0_tp1_d1.gif`
- `videos/failure_gifs/ep_0007__large_pitch_angle__pits__level_5__type_28__obsdelay_lv1_av0_pg0_tp0_d0.gif`

## 7. Why This Version Works

The final baseline improved after aligning the whole execution chain, not after changing one isolated reward term.

Important fixes:

- High-level actor/critic observation order and dimensions were made consistent with SRU splitting logic.
- `projected_gravity` was kept in both high-level and low-level observation paths.
- High-level depth was encoded into the expected `[64, 5, 8]` feature format.
- Critic privileged height scan was encoded into `[64, 7, 7]`.
- G1 low-level observation group was restored with InstinctLab-style history.
- High-level action was routed through `base_velocity.vel_command_b`.
- Frozen low-level policy checkpoint and action backend were moved into the independent project.
- Runner, SRU network, dropout mask handling, recurrent hidden state handling and task registration were unified inside `instinct_nav`.

The key lesson is that a humanoid navigation stack is highly sensitive to interface distributions. A checkpoint can be correct, but if observation history, command injection, camera preprocessing or recurrent-state handling differs, training behavior changes substantially.

## 8. Next Experiments

Do not modify the frozen baseline silently. New work should use separate run names and be reported as ablations.

Recommended order:

1. Multi-seed evaluation of the current implementation.
2. One-camera vs two-camera depth input.
3. Visibility-map fusion ablation: `Early`, `Late`, and `Query`.
4. Low-pass filter alpha sweep.
5. Goal noise and observation delay ablation.
6. Terrain-specific analysis for pits and hard maze cases.
7. Fine-tuning experiments only after the baseline comparison table is fixed.

## 9. Current Status

The project has reached a reproducible working baseline:

```text
overall success_rate: 0.902
stairs success_rate: 0.9592
```

The next priority is documentation, visualization and controlled ablation, not more untracked mainline changes.
