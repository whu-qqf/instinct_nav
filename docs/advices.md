# Real-Robot Deploy Fix Checklist

Date: 2026-07-29

Scope:
- `deploy/scripts/g1_nav_onboard_onnx.py`
- `deploy/instinct_nav_onboard/onnx_high_level_runtime.py`
- `deploy/scripts/export_high_level_sru_onnx.py`
- `deploy/scripts/goal_manager.py`

Purpose:
- Turn the current deployment review into an executable fix list.
- Focus on the remaining safety, contract, and observation-parity issues after the latest ONNX deployment updates.

## P0: Fix Before Any Non-Dryrun Retry

### 1. Latch autonomy faults instead of allowing automatic re-entry

Severity: High

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:191)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:283)

Problem:
- `_enter_autonomy_fault()` clears `_goal_start_requested`, zeros the command, resets SRU, and switches back to `stand`.
- But `stand -> parkour` still triggers whenever `--auto_start_parkour` is enabled.
- Therefore a depth timeout or SRU exception can produce:
 - `parkour -> fault -> stand -> parkour -> fault`

Required change:
- Add a latched fault state such as:
 - `_autonomy_fault_latched: bool`
 - or an explicit mode enum: `idle / armed / active / fault`
- In `_enter_autonomy_fault()`:
 - set the fault latch
 - clear goal-start intent
 - keep the robot in `stand`
- In the `stand` branch:
 - refuse `stand -> parkour` if the fault latch is active
- Add one explicit recovery path:
 - operator reset command
 - manual method call
 - or node restart

Acceptance:
- With `--auto_start_parkour` enabled, forcing a camera timeout must not re-enter `parkour` automatically.
- After fault, the robot must remain in `stand` until explicit re-arm.

### 2. Add odom freshness watchdog for `base_lin_vel_b`

Severity: High

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:235)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:212)

Problem:
- `base_lin_vel_b` now comes from odometry, which is the correct direction.
- But there is no freshness check for odom updates.
- If odom stalls, the actor continues receiving stale linear velocity.

Required change:
- Track last odom receipt time, for example:
 - `_last_odom_timestamp_s`
 - `_odom_timeout_s`
- In `_on_odom()`:
 - update freshness timestamp each message
- Before SRU step:
 - verify odom is still fresh when `use_odom_velocity=True`
- On timeout choose one policy and document it clearly:
 - preferred: enter autonomy fault
 - acceptable fallback: force `base_lin_vel_b = 0` and log a warning

Acceptance:
- If `/dog_odom` stops updating, deployment must not silently keep using a frozen linear velocity vector.
- Fault or explicit degrade behavior must be visible in logs.

## P1: Fix Before Wider Robot Trials

### 3. Correct manifest field semantics for `scene_token_dim`

Severity: Medium

Files:
- [deploy/scripts/export_high_level_sru_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/export_high_level_sru_onnx.py:242)

Problem:
- The exporter currently writes:
 - `"scene_token_dim": num_actor_scene_token_features`
- For the current v3/v4 policy this is `384`, not the per-token dimension `6`.
- The field name implies per-token dimensionality, not flattened total size.

Required change:
- Replace:
 - `num_actor_scene_token_features`
- With:
 - `actor_critic.scene_token_dim`
- If useful, add separate explicit fields:
 - `scene_token_num_tokens`
 - `scene_token_dim`
 - `scene_token_features_total`

Acceptance:
- The exported manifest must report:
 - `scene_token_num_tokens = 64`
 - `scene_token_dim = 6`
 - `scene_token_features_total = 384`

### 4. Extend runtime manifest validation beyond basic ONNX IO shapes

Severity: Medium

Files:
- [deploy/instinct_nav_onboard/onnx_high_level_runtime.py](/abs/path/D:/instinct_nav/deploy/instinct_nav_onboard/onnx_high_level_runtime.py:86)

Problem:
- Runtime currently validates:
 - `obs_dim`
 - `num_actions`
 - `rnn_num_layers`
 - `rnn_hidden_size`
- But scene-token-related metadata is not checked for semantic consistency.

Required change:
- Validate manifest fields against runtime config where applicable:
 - `scene_token_dim`
 - `scene_token_num_tokens`
 - `encoded_depth_dim`
 - `proprio_dim`
- Fail fast at startup if bundle metadata and runtime layout disagree.

Acceptance:
- A mismatched manifest must stop startup before robot control begins.

### 5. Record and throttle autonomy fault reasons

Severity: Medium

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:283)

Problem:
- Current fault path logs errors, but repeated conditions may spam logs.
- There is no persistent explicit state for diagnosis.

Required change:
- Store:
 - `_last_autonomy_fault_reason`
 - `_last_autonomy_fault_time_s`
- Throttle repeated identical fault messages.
- Log fault entry and recovery separately.

Acceptance:
- Repeated stale-camera faults should not flood logs every cycle.
- Operators should be able to identify the last fault cause quickly.

## P2: Recommended Hardening

### 6. Make `odom_twist_frame=auto` assumptions explicit in docs and CLI usage

Severity: Low

Files:
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:240)
- [deploy/README.md](/abs/path/D:/instinct_nav/deploy/README.md)

Problem:
- `auto` infers body-vs-world twist from `child_frame_id`.
- This is a reasonable heuristic, but it is still a heuristic.

Required change:
- Document the exact inference rule in `deploy/README.md`.
- Recommend explicit operator selection in field runs:
 - `--odom_twist_frame body`
 - or `--odom_twist_frame world`

Acceptance:
- Operators should not need to inspect code to understand how `auto` behaves.

### 7. Decide whether the Python checkpoint deploy path remains supported

Severity: Low

Files:
- [deploy/scripts/g1_nav_onboard.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard.py:1)
- [deploy/scripts/g1_nav_onboard_onnx.py](/abs/path/D:/instinct_nav/deploy/scripts/g1_nav_onboard_onnx.py:1)

Problem:
- The latest safety and odom changes are concentrated in the ONNX path.
- The Python checkpoint path still has weaker deployment semantics.

Required change:
- Choose one:
 - declare ONNX as the only production deployment path
 - or port the same state-machine and safety fixes into `g1_nav_onboard.py`

Acceptance:
- Documentation and code must agree on which entry point is production-safe.

## Validation Checklist

### Safety

1. Start with `--auto_start_parkour`.
2. Force camera frame timeout.
3. Confirm:
  - command is zeroed
  - node returns to `stand`
  - autonomy does not automatically re-enter `parkour`

### Odom

1. Run with odom enabled.
2. Stop odom updates while depth still updates.
3. Confirm:
  - fault or explicit degrade behavior occurs
  - stale `base_lin_vel_b` is not silently reused forever

### Manifest

1. Re-export ONNX bundle.
2. Inspect manifest JSON.
3. Confirm:
  - `scene_token_dim == 6`
  - `scene_token_num_tokens == 64`
  - total token features are represented separately if included

### Stability

1. Dry-run for 10 to 30 minutes.
2. Confirm:
  - no repeated fault loops
  - no uncontrolled command bursts
  - goal reached still transitions cleanly to `stand`

## Bottom Line

The latest deployment updates fixed the most important original gap by replacing zero `base_lin_vel_b` with odometry-derived linear velocity and by adding SRU fault fallback.

The next critical step is to make fault handling truly operator-safe:
- fault must latch
- stale odom must be detected
- manifest semantics must be cleaned up before further tooling depends on them