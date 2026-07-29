# Real Robot Deployment Code Review

Date: 2026-07-29

Scope:
- `deploy/scripts/g1_nav_onboard_onnx.py`
- `deploy/scripts/g1_nav_onboard.py`
- `deploy/instinct_nav_onboard/onnx_high_level_runtime.py`
- `deploy/scripts/export_high_level_sru_onnx.py`
- `deploy/instinct_nav_onboard/obs_builder.py`
- `deploy/scripts/goal_manager.py`
- related deployment notes in `deploy/README.md`
- training-side observation definitions in `source/instinct_nav/instinct_nav/navigation/`

Assumption:
- The production deployment path must use ONNX because `instinct_onboard` only supports ONNX for the low-level stack.
- Therefore, the ONNX high-level path is the only deployment path that should be treated as authoritative for real robot use.

## Executive Summary

The current deployment architecture is directionally correct:

```text
robot sensors
-> high-level SRU ONNX runtime
-> base_velocity_cmd
-> instinct_onboard low-level parkour ONNX stack
-> joint targets
```

This is the right control split. The main issues are not architectural. They are in:

- observation parity with training
- ONNX contract hard-coding
- safety fallback behavior
- state-machine robustness
- exporter/runtime consistency validation

The most important conclusion is:

1. Keep only the ONNX deployment path as the real robot production path.
2. Treat PyTorch checkpoint runtime only as a debug/reference path.
3. Before any serious robot trial, close the observation-parity and safety gaps listed below.

## Findings

### 1. `base_lin_vel_b` is hard-coded to zeros in real deployment input

Severity: High

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:40)
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:76)

Details:
- Both deployment entry paths set `base_lin_vel_b=np.zeros(3, dtype=np.float32)`:
  - ONNX path: `deploy/scripts/g1_nav_onboard_onnx.py`
  - Python checkpoint path: `deploy/scripts/g1_nav_onboard.py`
- The actor observation builder includes `base_lin_vel_b` as part of the 16D proprio block.
- Training did not use a zero placeholder here. The high-level actor observation is defined with
  `mdp.base_lin_vel_delayed` plus additive noise, and the underlying source is
  `asset.data.root_lin_vel_b`.

Risk:
- If the trained policy relies on base linear velocity feedback, deployment is feeding a materially different observation distribution than training.
- This is not a cosmetic mismatch. It can change action quality, stability, and memory behavior.

Recommendation:
- Prefer wiring a real body-frame linear velocity estimate into `SensorPacket.base_lin_vel_b`.
- The most practical first source is a robot odometry / state-estimator topic that already publishes
  `nav_msgs/Odometry` or equivalent twist.
- Convert to body frame only if the odometry source is world-frame. If the published twist is already
  in `child_frame_id=base_link`, do not rotate it again.
- If reliable linear velocity is not available, run an explicit simulation ablation with actor-side `base_lin_vel` removed or zeroed and compare performance against the deployment policy.
- Do not proceed to serious robot trials while assuming this mismatch is harmless.

Additional note:
- The repository already has an odometry consumer in `deploy/scripts/goal_manager.py`, but that odom
  stream is currently used only for world-goal to body-goal conversion, not for high-level policy
  observation.

### 2. High-level ONNX runtime hard-codes hidden-state shape `(1, 1, 512)`

Severity: High

Files:
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:41)
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:42)
- [deploy/scripts/export_high_level_sru_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/export_high_level_sru_onnx.py:114)

Details:
- The runtime initializes:
  - `self.h = np.zeros((1, 1, 512), dtype=np.float32)`
  - `self.c = np.zeros((1, 1, 512), dtype=np.float32)`
- The exporter prints these dimensions, but the runtime does not read them from the ONNX graph or a manifest.

Risk:
- Any future checkpoint change to SRU hidden size or number of layers will silently break deployment assumptions.
- This creates a fragile coupling between exporter internals and runtime code.

Recommendation:
- Export a manifest alongside `policy.onnx`, for example:

```json
{
  "obs_dim": 2960,
  "num_actions": 3,
  "rnn_num_layers": 1,
  "rnn_hidden_size": 512,
  "scene_token_dim": 384
}
```

- Make `OnnxHighLevelRuntime` read that manifest and allocate hidden state from it.
- Reject startup if manifest and ONNX IO contract disagree.

### 3. No explicit PyTorch-vs-ONNX parity check is built into the export flow

Severity: High

Files:
- [deploy/scripts/export_high_level_sru_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/export_high_level_sru_onnx.py:121)
- [deploy/scripts/export_high_level_sru_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/export_high_level_sru_onnx.py:124)

Details:
- The exporter runs one PyTorch forward pass on dummy zeros and exports ONNX.
- It does not reload the exported ONNX and compare:
  - `actions`
  - `h_out`
  - `c_out`
  against the PyTorch exporter module on the same inputs.

Risk:
- A successful export does not prove deployment correctness.
- Numerical or graph-level divergence can survive until robot-side testing.

Recommendation:
- Extend the export script to:
  1. export ONNX
  2. load ONNXRuntime session
  3. run the same dummy and several randomized inputs
  4. compare `actions`, `h_out`, `c_out` with tolerances
- Fail export if parity is out of bounds.

Minimum tests:
- single-step zero input
- single-step random finite input
- multi-step recurrent rollout with non-zero hidden state
- reset case

### 4. Safety fallback only covers timing drift, not runtime exceptions

Severity: High

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:178)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:185)

Details:
- `_update_sru_velocity_command_if_due()` zeros `base_velocity_cmd` only when the high-level update is late.
- If `self.sru_runtime.step(packet)` raises, there is no local exception handling and no guaranteed zero-command fallback before the node unwinds.

Risk:
- The robot can keep the last non-zero command cached in `base_velocity_cmd`.
- In `--nodryrun` mode this is a real safety issue.

Recommendation:
- Wrap the SRU update path in `try/except`.
- On exception:
  - set `self.base_velocity_cmd = np.zeros(3, dtype=np.float32)`
  - call `self.sru_runtime.reset()`
  - optionally switch `current_agent_name` back to `stand`
  - emit an error log with throttling

Suggested behavior:

```text
SRU step error
-> zero command immediately
-> reset SRU hidden state
-> stop autonomous progression
-> require operator intervention or explicit re-arm
```

### 5. Deployment state machine is better than the Python path, but still too implicit for field use

Severity: Medium

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:153)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:205)

Details:
- Goal receipt can trigger `stand -> parkour`.
- Goal reached can trigger `parkour -> stand`.
- This is correct as a first version.

Remaining issue:
- There is still no explicit deployment mode model such as:
  - `idle`
  - `armed`
  - `autonomous_active`
  - `fault`
- The ONNX path supports goal-driven `stand -> parkour` and `parkour -> stand` transitions.
- The Python checkpoint path is less complete here: it supports `auto_start_parkour`, but it does not
  currently mirror the ONNX path's `_goal_start_requested` transition behavior.

Risk:
- As more operator behaviors are added, implicit booleans like `_goal_start_requested` and `auto_start_parkour` will become hard to reason about.
- Maintaining two deployment entry points with slightly different autonomy semantics increases the
  chance of testing one behavior and deploying another.

Recommendation:
- Introduce a small explicit autonomy mode enum/state variable.
- Only allow non-zero autonomous commands when:
  - low-level buffers are ready
  - robot is in the expected agent state
  - autonomy mode is armed and active
- Either keep the ONNX path as the only authoritative production entry point, or make the Python
  checkpoint path match the same state transitions exactly.

### 6. Scene-token construction is intentionally approximate and must be treated as a known sim-to-real gap

Severity: Medium

Files:
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:89)
- [deploy/instinct_nav_onboard/obs_builder.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/obs_builder.py:77)
- [deploy/README.md](/abs/path/D:/instinct_nav/deploy/README.md:164)

Details:
- Deployment-side scene tokens are built from current-frame torso depth plus local visibility accumulation.
- Training-side token construction used IsaacLab-side camera/root-pose information and delayed BEV warping.

Risk:
- Even if ONNX export is perfect, token semantics may differ from training.
- This is likely to show up as poor behavior around occlusion, exploration memory, or turn planning.

Recommendation:
- Treat this as a declared parity gap, not a hidden assumption.
- Before serious robot trials, compare token statistics between:
  - IsaacLab eval dump
  - robot dry-run dump
- The next parity improvement should be odom/root-pose based visibility warping on the robot.

### 7. There is no contract file tying exporter output to runtime expectations

Severity: Medium

Files:
- [deploy/scripts/export_high_level_sru_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/export_high_level_sru_onnx.py:142)
- [deploy/instinct_nav_onboard/config.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/config.py:29)

Details:
- Runtime expects a fixed actor observation layout from local Python constants.
- Exporter expects a matching checkpoint architecture from `SceneTokenSruPolicyConfig`.
- No artifact bundle records that these two came from the same model/export step.

Risk:
- It is easy to deploy mismatched combinations of:
  - checkpoint
  - depth encoder ONNX
  - high-level policy ONNX
  - runtime constants

Recommendation:
- Export a deployment bundle:

```text
policy.onnx
depth_encoder.onnx
deploy_manifest.json
git_revision.txt
checkpoint_info.json
```

- Make `g1_nav_onboard_onnx.py` validate the bundle before startup.

### 8. Command limiting is present, but command-shaping is still too thin for first real robot trials

Severity: Medium

Files:
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:147)
- [deploy/instinct_nav_onboard/config.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/config.py:46)

Details:
- Current deployment clamps `[vx, vy, wz]` to configured min/max.
- There is no explicit slew-rate limiter, deadband, or first-order smoothing in the ONNX runtime path.

Risk:
- Even with correct action values, step-to-step jumps can be too aggressive for early field trials.

Recommendation:
- Add deployment-side shaping before writing to `base_velocity_cmd`:
  - per-axis deadband
  - per-axis slew rate limit
  - optional low-pass filter
- Keep these settings in deployment config, not embedded in model logic.

### 9. The runtime writes commands twice through two layers

Severity: Low

Files:
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:70)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:189)

Details:
- `OnnxHighLevelRuntime.step()` already writes to the command sink.
- `_update_sru_velocity_command_if_due()` then also writes `self.base_velocity_cmd = command.as_array()`.

Risk:
- Not currently wrong because both target the same buffer, but this is redundant and weakens ownership clarity.

Recommendation:
- Pick one owner:
  - either the runtime writes to sink and the node does not reassign
  - or the runtime returns command and the node owns the final write

The second option is usually clearer for robot safety state machines.

## Recommended Modifications

### Priority 0: Required before any non-dryrun test

1. Add exception-safe zeroing in `_update_sru_velocity_command_if_due()`.
2. Add exporter parity verification against ONNXRuntime.
3. Stop hard-coding hidden state shape in runtime.
4. Freeze a deployment bundle format and reject mismatched artifacts.
5. Decide and document the single authoritative production entry point: ONNX path only, or strict
   state-machine parity across both paths.

### Priority 1: Required before serious navigation trials

1. Replace zero `base_lin_vel_b` with a real estimate, or re-train / validate without it.
   Recommended first source: existing odom / state-estimator twist.
2. Add deployment-side command shaping:
   - deadband
   - slew-rate limiting
   - optional smoothing
3. Make autonomy mode explicit instead of relying only on booleans.

### Priority 2: Required for robust sim-to-real iteration

1. Add IsaacLab vs robot dump comparison tooling.
2. Improve scene-token parity with odom/root-pose warping.
3. Unify PyTorch `obs_builder` logic and ONNX runtime `build_actor_obs` / `build_scene_tokens` through a shared test suite.

## Recommended Test Sequence

### Stage 1: Export contract verification

Run on workstation:

1. Export `policy.onnx` and `depth_encoder.onnx`.
2. Re-load both in ONNXRuntime.
3. Compare PyTorch vs ONNX:
   - `actions`
   - `h_out`
   - `c_out`
4. Save a deployment manifest.

Exit criterion:
- deterministic parity within agreed tolerance

### Stage 2: Robot-PC dry run

Run only with dry-run enabled.

Checks:
- `/lowstate` and IMU are valid
- depth frames update
- actor obs dim is exactly `2960`
- encoded depth is finite and shaped correctly
- scene tokens are finite and numerically sane
- `base_velocity_cmd` updates at 5 Hz
- low-level joint targets remain finite

Exit criterion:
- at least 30 minutes stable runtime without SRU or runtime exceptions

### Stage 3: Constrained non-dryrun

Use:
- flat terrain
- short body-frame goals
- low command limits
- operator ready on e-stop

Suggested temporary tighter limits than current config:

```text
vx: +/- 0.3
vy: +/- 0.2
wz: +/- 0.4
```

Exit criterion:
- no unstable command bursts
- clean goal start/goal reached transitions
- reliable zeroing on fault or timeout

### Stage 4: Expanded trials

Only after prior stages pass:
- turning
- repeated goals
- cluttered local geometry
- stairs or more difficult terrain

## Suggested Work Split

### Track A: Export and contract

Owner tasks:
- add bundle manifest
- add ONNX parity check
- add runtime contract validation

### Track B: Runtime safety

Owner tasks:
- exception fallback
- zero command on failure
- explicit autonomy mode
- command shaping

### Track C: Observation parity

Owner tasks:
- replace zero `base_lin_vel_b`
- dump comparison against IsaacLab eval
- scene-token parity audit

## Bottom Line

The deployment path is viable, but it is not yet reliable enough for serious real robot navigation trials without further tightening.

The most important next step is not another model change. It is to convert the ONNX deployment path from "runs end-to-end" into "has a verified export contract, verified observation parity, and verified fail-safe behavior."

That is the threshold for calling the current real robot deployment path dependable.