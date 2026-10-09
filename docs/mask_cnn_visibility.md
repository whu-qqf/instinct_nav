# Mask CNN with visibility memory

The MaskCNN task retains the inherited 64 x 5 visibility BEV patches and
DynamicMapCross/query attention. Dynamic segmentation is not projected into
these patches. Both actor and critic receive the same cached external mask.

Dynamic input: delayed mask and masked normalized depth, 2 x 36 x 64.
CNN -> 4 x 8 spatial features -> positional self-attention -> 128D -> SRU.
Actor/critic observation sizes: 7504/10641. SRU inputs: 208/272.
Segmentation auxiliary loss is disabled. The encoder learns through PPO.
PPO exploration settings inherit the baseline (initial std 1.0, entropy 0.005).

Proximity, approach, centerline and unobserved shaping are removed; TTC cost
is -0.25. Collision termination and navigation rewards remain enabled.

Encounter spawning is isolated behind obstacle_encounter_spawn. Five slots
remain available. Candidates are sampled at activation using current forward
velocity, with an estimated crossing time. A minimum 1.5-second spacing,
nearby-obstacle capacity (one below difficulty 0.6, two thereafter), goal
clearance and a conservative lateral height-field screen gate activation.
Rejected candidates stay parked and are retried. Accepted paths remain fixed.
This first version uses crossing paths only; mixed behaviors are future work.
The height screen is not a proof of free space or full route feasibility.
An episode can finish before all five slots activate.

This changes the MaskCNN architecture and requires fresh training. Neither
the earlier token-free MaskCNN smoke checkpoint nor a RecurrentFusion
checkpoint matches. Existing RecurrentFusion task behavior is unchanged.

## Formal run (2026-09-07)

Run name: `mask_cnn_visibility_encounter_4096_scratch_20260907_111612`.
Started from scratch with 4096 environments, 6000 iterations, seed 72,
W&B project `qqfqqf0612-wuhan-university/instinct_nav`.
Detached launcher PID at startup: 910011; trainer PID: 910012.
Stdout: `logs/mask_cnn_visibility_encounter_4096_scratch_20260907_111612.stdout.log`.
Preflight: three CPU tests passed, py_compile and git diff --check passed,
and the preceding 32-env full PPO smoke produced model_0.pt.

## Old-baseline mask ablation (2026-09-11)

Task: `Instinct-Nav-G1-SRU-MaskCNN-OldBaseline-OfficialParkour-v0`, with
matching `-Dev-v0` and `-Play-v0` variants. This task inherits the old
DynamicMapCross RichTerrain Centerline Unobserved configuration, not the
recent encounter/banks task, and adds only the existing mask/depth image
observation and its CNN/attention/SRU feature branch.

- Keep terrain-only critic height scan and the 5D visibility patches.
- Restore baseline centerline +0.5 and unobserved -0.2 rewards; retain
  baseline proximity -0.2, approach -0.1 and TTC 0.0.
- Keep baseline four obstacle slots, 2-8 s spawn delay, no stagger,
  fixed enable at iteration 4504, 50% participating environments, no banks.
- Use four PPO mini-batches, 16 steps per environment, seed 72.
- Formal run starts from scratch: 4096 environments, 6000 iterations.
  Do not load the old model into the larger recurrent input.

This controls configuration relative to the baseline available in the current
codebase; it is not an exact replay of historical source or simulator state.
The existing encounter MaskCNN tasks are unchanged. No new critic risk-map
input is added, to avoid changing another architectural factor.

Validation: py_compile and git diff --check passed. The 32-env smoke at
`2026-09-11_13-03-23_old_baseline_mask_smoke` completed PPO iteration 0 and
saved a CPU-loadable `model_0.pt` with finite parameters; actor/critic SRU
inputs are 208D/272D. Formal launch name:
`mask_cnn_old_baseline_4096_scratch_20260911_130439`.
Stdout: `logs/mask_cnn_old_baseline_4096_scratch_20260911_130439.stdout.log`.
Launcher PID at startup: 165949.

### User correction: obstacles from iteration 0

The user rejected the initial static phase. All three OldBaseline MaskCNN
task variants now set `obstacle_max_switch_iteration = 0`. The preceding
4504-iteration schedule describes only the superseded launch, not the current
task. Restart from scratch in a new run; preserve its existing artifacts.
Keep four slots, 50% participating environments, and the per-episode 2-8 s
spawn delay. Network, rewards, PPO settings and terrain remain unchanged.
This experiment is therefore baseline rewards + mask CNN + immediate
obstacle curriculum, not a mask-only ablation of the historical training run.

### Final comparison setup: preserve the completed encounter run

The user requires obstacle logic identical to the completed September 7/8
MaskCNN encounter experiment. The intervening OldBaseline launches are
superseded and their artifacts retained. The new task is
`Instinct-Nav-G1-SRU-MaskCNN-Encounter-RestoredRewards-Banks-OfficialParkour-v0`
(matching Dev and Play variants). It subclasses the completed experiment's
environment and reuses its runner; only two reward terms change:
`stair_centerline=+0.5`, `velocity_into_unobserved=-0.2`.

Keep five obstacle slots, first delay 1-2 s, minimum interval 1.5 s,
encounter-gated crossing, immediate curriculum enable, success threshold 0.90,
difficulty step 0.2, four terrain banks switching every 1000 iterations,
eight PPO mini-batches, TTC -0.25, no proximity/approach rewards, and the
unchanged mask CNN / visibility / terrain-only critic-height architecture.
Start fresh with 4096 environments, 6000 iterations, seed 72. This isolates
the navigation-reward pair rather than the network architecture.

Validation: `2026-09-11_15-23-56_restored_rewards_encounter_smoke` completed
32-env PPO iteration 0 with a finite, CPU-loadable checkpoint. Saved YAML
comparison against the completed September 8 run found identical obstacle
fields, reset/update obstacle event parameters, policy and PPO configuration;
only the two intended reward entries differ. Formal launcher PID: 267793.
Run name: `mask_cnn_encounter_restored_rewards_4096_20260911_153005`.
Stdout: `logs/mask_cnn_encounter_restored_rewards_4096_20260911_153005.stdout.log`.

## September 15 mask-zero inference ablation

The completed restored-rewards checkpoint `model_5999.pt` was evaluated with
4096 environments and 8192 episodes per condition at seed 72. The ablation
executes the complete GT-mask, corruption, and masked-depth pipeline to preserve
its random-number consumption, then replaces the final 4608D observation block
with zeros. Network shapes and all other observations remain unchanged.

| condition | normal mask | zero mask | delta (zero - normal) |
| --- | ---: | ---: | ---: |
| static success | 81.5186% | 78.3936% | -3.1250 pp |
| dynamic-d0 success | 79.6997% | 81.7017% | +2.0020 pp |
| dynamic obstacle collision | 0.4761% | 1.0742% | +0.5981 pp |
| dynamic base contact | 7.5928% | 7.1899% | -0.4028 pp |
| dynamic orientation failure | 12.2070% | 10.0098% | -2.1973 pp |

Interpretation: the mask branch is behaviorally active and reduces obstacle
collisions, but its direct 128D input to the main SRU also changes the underlying
navigation/pose behavior. In this aggregate evaluation, the collision benefit
is outweighed by orientation failures with the normal mask. Do not conclude
that mask perception is useless, or that zero-mask is a deployable policy.
Static and dynamic conditions are independent seeded aggregate runs rather than
case-by-case paired replays.

The first run in `eval_mask_zero_pair_20260914_204610` is invalid for causal
comparison because its early zero return skipped mask-corruption RNG draws.
The corrected result is in
`eval_mask_zero_rng_matched_pair_20260915_004418`. Both corrected conditions
have `missing_record=multiple_records=stale_success=disagreements=0` across all
8192 episodes.

Recommended next architecture: keep dynamic memory, but isolate it from the
navigation recurrent state. Encode the mask stream into its own recurrent
dynamic context and apply a bounded, zero-initialized residual action correction
or gated late fusion. This preserves a stable depth/proprio navigation backbone
while allowing motion history to reduce collision risk.

## MaskCNN64 gated-add ablation (2026-09-15)

Paper-style architecture figure:
[`mask_cnn64_gated_add_paper_architecture.svg`](figures/mask_cnn64_gated_add_paper_architecture.svg)
(also exported as PDF and 4K PNG).

Task: `Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-v0`
with matching `-Dev-v0` and `-Play-v0` variants. Observations, rewards, terrain
banks, five-obstacle timing and mask corruption retain the completed
restored-rewards setup. In addition to the network ablation, this task fixes the
dynamic-obstacle difficulty at its hardest tier from startup, assigns obstacles
to every environment, and raises obstacle height to 1.6 m.

| dynamic-obstacle setting | completed 128D run | MaskCNN64 gated-add |
| --- | ---: | ---: |
| initial/fixed difficulty | 0.0, promoted by success | 1.0 |
| participating environments | 25%-90% by difficulty | 100% |
| obstacle geometry | 0.5 x 0.5 x 0.8 m | 0.5 x 0.5 x 1.6 m |
| obstacles per environment | 5 | 5 |
| activation timing | first 1-2 s, then every 1.5 s | unchanged |

The shared mask CNN now emits 64D. Independent actor/critic `64 -> 64`
adapters align that context with their different depth-attention feature spaces.
Each branch predicts a scalar sigmoid gate, initialized to 0.1, and applies
`LayerNorm(depth64 + gate * mask_adapter64)`. The mask is no longer concatenated
to the recurrent input. Actor and critic gates are logged as
`Policy/mask_gate_actor` and `Policy/mask_gate_critic`.

| component | sru_ref | MaskCNN 128 concat | MaskCNN64 gated-add |
| --- | ---: | ---: | ---: |
| actor raw observation | 2576 | 7504 | 7504 |
| critic raw observation | 5713 | 10641 | 10641 |
| raw mask block | 0 | 4608 | 4608 |
| mask CNN output | 0 | 128 | 64 |
| depth-attention output | 64 | 64 | 64 |
| mask/depth fusion | none | concat | gated residual add |
| actor SRU input | 80 | 208 | 80 |
| critic SRU input | 144 | 272 | 144 |
| actor/critic SRU hidden | 512 | 512 | 512 |
| late scene-query output | 0 | 64 | 64 |
| actor post-SRU input | 512 | 576 | 576 |
| critic post-SRU input incl. time | 520 | 584 | 584 |

This architecture is checkpoint-incompatible with the 128D concat model and
must train from scratch. The raw observation sizes remain unchanged because the
full-resolution two-channel mask block is still stored and encoded; this
ablation isolates recurrent fusion rather than input-resolution/storage effects.

Validation: five pure-PyTorch SRU tests passed, including recurrent dimensions,
forward/backward gradients, initial gate value and legacy concat preservation.
The final 32-env one-iteration PPO smoke at
`2026-09-15_09-49-09_mask_cnn64_gated_add_final_smoke_20260915` confirmed
observations `7504/10641`, actor/critic SRU inputs `80/144`, hidden size 512,
finite checkpoint parameters and `model_0.pt` output. TensorBoard recorded
learned actor/critic gates `0.0896/0.0911` after the first update (initialized
at `0.1`).

After enabling the fixed hardest obstacle setup, a second 32-env one-iteration
smoke at `2026-09-15_10-21-41_mask_cnn64_hard_allenv_height16_smoke_20260915`
recorded `difficulty=1.0` and `curriculum_env_frac=1.0`; its saved environment
configuration contains five cuboids sized `0.5 x 0.5 x 1.6 m` and it produced a
finite `model_0.pt`.

### Controlled evaluation queued (2026-09-16)

The 128D completed run and the 64D gated-add run are not directly comparable
from their training curves: the old run used difficulty 0, a partial obstacle-env
fraction, and 0.8 m cuboids, while the new run uses difficulty 1, every env, and
1.6 m cuboids. `scripts/eval_maskcnn64_controlled.sh` therefore runs five serial
8192-episode evaluations after the current training process exits: both policies
in static and matched hard-dynamic conditions, plus the new policy with its
dynamic-mask input zeroed. Each run uses 4096 envs, seed 72, the sru_ref success
metric, and the same 1.6 m obstacle geometry. The hard-dynamic condition pins
difficulty 1 and 100% obstacle participation before scene creation. Summaries
include the actual obstacle conditions and are written under
`logs/eval_maskcnn64_controlled_20260916/`. These are seeded independent runs,
not case-by-case paired replays. No Isaac evaluation runs concurrently with
the ongoing trainer. The terrain-bank switch occurs every 1000 PPO iterations;
these short evaluation passes therefore exercise bank 0 only. Treat this first
batch as a controlled bank-0 diagnosis, not an aggregate over all four banks.

### MaskCNN32 concatenation ablation (2026-09-16)

The MaskCNN64 gated-add run was stopped at approximately iteration 3310 at the
user's request. Its latest periodic checkpoint is `model_3000.pt`; the queued
`model_5999.pt` evaluations were cancelled. This run did not establish a final
evaluation result.

`Instinct-Nav-G1-SRU-MaskCNN32-Concat-Encounter-RestoredRewards-Banks-OfficialParkour-v0`
keeps the same observations, rewards, terrain banks, five-obstacle timing,
difficulty 1 from iteration 0, 100% obstacle environments, and 1.6 m cuboids.
The mask/masked-depth CNN and spatial self-attention now produce 32D, which is
concatenated after the 64D depth-attention output and before actor/critic SRU.
There is no mask/depth addition, scalar gate, or fusion LayerNorm. Actor/critic
SRU inputs are `112/176`, between gated-add `80/144` and the original 128D
concat `208/272`; raw observations remain `7504/10641`.

The 32-env one-iteration Isaac smoke confirmed these dimensions, difficulty
`1.0`, obstacle-env fraction `1.0`, and a saved `model_0.pt`. The new task must
train from scratch because its SRU and CNN weights differ from both earlier
architectures. The 4096-env, 6000-iteration, seed-72 run started at
`logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-09-16_23-13-27_g1_sru_mask_cnn32_concat_hard_allenv_height16_v1/`.
Its first update completed and saved finite `model_0.pt` parameters; this is
startup verification, not an efficacy result.

### Matched NoMask control (2026-09-18)

`Instinct-Nav-G1-SRU-NoMaskControl-Encounter-RestoredRewards-Banks-OfficialParkour-v0`
inherits the MaskCNN32 environment, removes only the actor/critic dynamic-mask
observation, and disables the policy mask encoder. The five-obstacle encounter
schedule, 1.6 m geometry, difficulty 1 from iteration 0, 100% obstacle
participation, rewards, terrain banks, seed, and eight PPO mini-batches remain
the same. Actor/critic observations return to `2896/6033`; SRU inputs return
to `80/144`. Both still use the 5D visibility BEV and SRU-after scene query.
This is a from-scratch architecture control, not a compatible resume of the
MaskCNN32 checkpoint. Dev and Play variants use the corresponding NoMask task
suffixes.

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-NoMaskControl-Encounter-RestoredRewards-Banks-OfficialParkour-v0 \
    --num_envs 4096 --max_iterations 6000 \
    --run_name g1_sru_nomask_control_hard_allenv_height16_v1 \
    --headless --wandb --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav --seed 72
```

The pure-PyTorch recurrent/scene-path tests passed. The 32-env, one-iteration
IsaacLab smoke at `2026-09-18_10-16-02_nomask_control_hard_smoke_20260918`
confirmed actor/critic observations `2896/6033`, SRU inputs `80/144`, obstacle
difficulty `1.0`, 100% obstacle participation, and a finite `model_0.pt`.

The MaskCNN32 process had already exited at iteration 3204 without a normal
completion record; its `model_3000.pt` remains untouched. The independent
NoMask control started from scratch at
`logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-09-18_10-22-50_g1_sru_nomask_control_hard_allenv_height16_v1/`.
Its first PPO update saved `model_0.pt`. A structured comparison of saved
configs found the environment identical to the MaskCNN32 run after replacing
only actor/critic `dynamic_mask` terms with null. The agent config differs
only in `use_dynamic_mask=false` and the run name. Early training metrics are
startup checks, not an efficacy result.
