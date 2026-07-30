# InstinctNav Deployment Memory

Last updated: 2026-07-30 (Asia/Shanghai)

## Current pause state

- Work paused because the G1 battery ran out.
- No `g1_parkour.py` or InstinctNav dry-run process was intentionally left running.
- Do not start autonomous `--nodryrun` immediately after power is restored. Re-run the checks below first.

## Low-level `instinct_onboard` issue resolved

The original `g1_parkour.py --nodryrun` process did not react because CycloneDDS
was bound to `docker0` instead of the robot Ethernet interface `enP8p1s0`.

Changes:

- `/workspace/unitree-jetson-container/unitree_ros2/setup.sh`
  - Detect the active interface whose IPv4 address is in `192.168.123.0/24`.
  - Fail explicitly when no robot interface is found.
- `/workspace/unitree-jetson-container/env.sh`
  - Use `ROS_DOMAIN_ID=0`.
  - Source the correct `unitree-jetson-container/unitree_ros2/setup.sh` path.

Verified before battery depletion:

- Robot interface: `enP8p1s0`, `192.168.123.164/24`.
- `/lowstate`: about 1050 Hz.
- `/secondary_imu`: about 1050 Hz.
- `/wirelesscontroller`: about 20 Hz after the controller wakes up.
- Low-level parkour loop: about 50 Hz.
- Correct low-level state sequence: cold start -> `R1` stand -> `L1` parkour.
- `R2` or `L2` turns motors off and exits.

## InstinctNav deployment issues found and fixed

Production entry point:

```text
deploy/scripts/g1_nav_onboard_onnx.py
```

The production script targets the newer API in:

```text
instinct_nav/external/instinct_onboard
```

It is not compatible with the older sibling package at
`/workspace/unitree-jetson-container/instinct_onboard` unless the external path
is placed first in `PYTHONPATH`.

Implemented changes:

1. Added `UnitreeJoyStick` to the production ONNX node.
   - Waits for the first `/wirelesscontroller` sample.
   - `R1` cancels autonomy and returns to stand.
   - `R2` or `L2` uses the existing motor-off emergency shutdown.
2. Fixed dry-run state-machine validation.
   - `--auto_start_parkour` skips cold start only in dry-run.
   - The auto-start request is consumed once, so `R1` no longer causes repeated stand/parkour transitions.
   - Real `--nodryrun` startup still performs cold start normally.
3. Restored training/deployment action parity.
   - The trained Gaussian action term applies `tanh(raw_action)` before producing velocity commands.
   - ONNX deployment now applies the same `tanh` transform before safety clamps.
   - The raw action remains in the `last_action` observation, matching training semantics.
4. Corrected deployment documentation.
   - Source `/workspace/unitree-jetson-container/env.sh` first.
   - Preserve ROS Python paths when setting `PYTHONPATH`.
   - Use the separate stand and parkour checkpoint directories; do not pass the parkour model as `--standdir`.
5. Updated production dependencies.
   - `deploy/requirements.txt` is the torch-free ONNX runtime set.
   - `deploy/requirements-debug.txt` adds torch/torchvision for legacy debug tools.

## Verification completed

- High-level manifest and policy SHA256 validation passed.
- ONNX contract passed:
  - actor observation: 2960
  - action: 3
  - SRU hidden/cell state: `(1, 1, 512)`
  - encoded depth: 2560
  - scene tokens: `64 x 6 = 384`
- CPU ONNX inference returned finite outputs.
- Real robot dry-run loaded:
  - high-level SRU ONNX
  - high-level depth encoder ONNX
  - parkour low-level ONNX
  - stand low-level ONNX
- Real dry-run received lowstate, torso IMU, joystick, odometry and RealSense data.
- Camera ran at about 60 Hz.
- Main loop ran at about 49.9 Hz.
- Dry-run auto-start reached parkour and generated 20 high-level debug records at:

```text
/tmp/instinct_nav_dryrun_fixed/index.jsonl
```

- All recorded depth, encoded depth, scene-token and actor-observation values were finite.

## Important unresolved validation risk

Real-robot dry-run produced large raw policy actions, for example approximately:

```text
[+2.1, -8.5, +0.27]
```

Safety processing bounded the outgoing command, often to approximately:

```text
[+0.3, -0.2, ...]
```

The deployment action transform is now consistent with training, but repeated
command saturation may indicate a sim-to-real observation distribution mismatch.
Do not begin autonomous real motion until the robot dump is compared with a
matching IsaacLab evaluation dump. Check especially:

- `projected_gravity_b` sign/order
- `/dog_odom` velocity frame (`body` versus `world`)
- RealSense depth invalid values and range
- encoded-depth mean/std
- scene-token distribution
- raw-action and processed-command saturation rates

The scene-token deployment still uses current-frame torso depth and local
visibility accumulation without full odometry-based BEV warping. This is a known
training/deployment parity limitation.

## Resume checklist for the next session

1. Charge and safely support the robot; clear the area and prepare emergency stop.
2. Confirm no stale control process:

   ```bash
   pgrep -af 'g1_parkour.py|g1_nav_onboard_onnx.py'
   ```

3. Prepare the environment:

   ```bash
   source /workspace/unitree-jetson-container/env.sh
   source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
   cd /workspace/unitree-jetson-container/instinct_nav
   export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH
   ```

4. Verify interface and topics:

   ```bash
   echo "$CYCLONEDDS_URI"
   ros2 topic hz /lowstate
   ros2 topic hz /secondary_imu
   ros2 topic hz /wirelesscontroller
   ros2 topic hz /dog_odom
   ```

5. Repeat the end-to-end dry-run with `--auto_start_parkour`; do not add `--nodryrun`.
6. Save a fresh robot debug dump and compare it with the matching IsaacLab evaluation dump.
7. Only after distribution parity is acceptable, perform a stand-only `--nodryrun` test without `--auto_start_parkour` and without publishing a navigation goal.

See `README.md` and `deploy/README.md` for the corrected full commands.

## 2026-07-30 first no-dry-run attempt

- The first real launch loaded all ONNX models and received low state and torso
  IMU successfully.
- On the first `/wirelesscontroller` sample it immediately reported
  `Safety shutdown triggered via joystick`, before all startup buffers were
  ready or the 50 Hz control loop began. Consequently the robot correctly did
  not move and no normal stand transition was attempted.
- This means the controller message had the R2 (`0x10`) or L2 (`0x20`) bit set.
  Release both triggers before retrying; do not bypass this emergency stop.
- The Unitree joystick handler now logs the raw `keys` value and decoded L2/R2
  state when this happens again.
- Startup is now included in the node's cleanup `try/finally`, so an emergency
  stop while waiting for initial buffers also terminates the RealSense child
  process and releases shared memory.
- `/dog_odom` was measured at about 51 Hz with `frame_id=odom` and
  `child_frame_id=robot_center`. ROS Odometry twist is expressed in the child
  frame, so `robot_center` was added to the `--odom_twist_frame auto` body-frame
  aliases. Existing processes must be restarted to pick up this change; for
  field runs, `--odom_twist_frame body` remains the clearest explicit setting.
- RealSense occasionally missed the former 32 ms `wait_for_frames` deadline.
  That killed and restarted the camera subprocess, then latched the autonomy
  depth-stale fault. The minimum wait is now 100 ms; an isolated RealSense
  frame timeout is treated as a dropped frame, while the existing 0.25 s
  freshness watchdog still latches a real sustained outage. The subprocess
  also releases its ctypes/NumPy shared-memory views before closing SHM, which
  removes the secondary `BufferError: cannot close exported pointers exist`.
- During a later long forward-goal real-robot test, the robot reportedly fell
  and continued twitching. Real autonomous motion must remain suspended until
  the fall logs and hardware condition are reviewed. No onboard/goal-manager
  process was found afterward, and `/tmp/instinct_nav_real_debug` did not
  contain a usable incident dump. The current stack has no verified fall-state
  motor-off guard, so do not resume by simply relaunching the previous command.
- A fall guard and persistent incident recorder were subsequently added to the
  production ONNX entry. Defaults are 55 degrees sustained for 0.15 seconds;
  invalid projected gravity fails closed. The check runs before each 50 Hz
  agent/actuator step and uses the existing zero-gain motor-off path. Real mode
  now requires `--safety_log_dir`. The JSONL telemetry includes torso gravity
  and tilt, angular/linear velocity, shaped and raw high-level commands, goal,
  joint state and last target, camera frame counter, odom age, agent state,
  autonomy fault state, and torque-clipping counters. Logs and ONNX dumps must
  be placed under `/workspace`, never `/tmp`.
