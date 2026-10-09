# instinct_nav onboard deployment skeleton

This folder is the lightweight real-robot deployment side of `instinct_nav`.
It is intended to be copied to the robot PC / user workstation and connected to
`instinct_onboard` or the robot SDK. It should not depend on IsaacLab or Isaac Sim.

## Runtime architecture

```text
real robot sensors
  IMU / joints / depth / odom / goal
        |
        v
SensorAdapter -> HighLevelObsBuilder @ 5 Hz
        |
        v
SRU navigation policy
  obs: proprio(16) + encoded depth(2560) + scene tokens(384)
  action: raw high-level velocity command
        |
        v
Command bridge / clamp / low-pass
        |
        v
instinct_onboard low-level policy @ 50 Hz
  obs: base_ang_vel / projected_gravity / velocity_commands history
       joint_pos / joint_vel / actions / depth history
  action: G1 joint command
        |
        v
robot actuator interface
```

## What can be reused from instinct_onboard

Use `instinct_onboard` for the frozen G1 low-level policy path:

```text
velocity command -> Instinct low-level policy -> joint targets -> robot SDK
```

`instinct_nav` adds the high-level SRU navigation policy in front of it:

```text
depth + proprio + goal + scene tokens -> SRU -> velocity command
```

So the integration target is not to replace `instinct_onboard`, but to provide a
`VelocityCommandSink` that sends SRU output into the command input already used by
`instinct_onboard`.

## Current important limitation

The current trained policy uses actor scene tokens:

```text
policy obs dim = 2960 = 16 proprio/goal/action + 2560 encoded depth + 384 scene tokens
```

The existing `ActorCriticSRU.export_jit()` in `source/rsl_rl` still uses the old
single-camera exporter that assumes the tail of the observation is only depth. It
therefore should not be used blindly for the scene-token policy until the exporter
is updated to include the scene-token query branch.

The production robot path is now the ONNX entry point:

```text
deploy/scripts/g1_nav_onboard_onnx.py
```

The Python checkpoint path is retained only as a debug/reference path. It should
not be treated as production-safe unless the same state-machine, watchdog, and
manifest checks are ported to it.

## Files

```text
instinct_nav_onboard/config.py                 Shared constants from the training task
instinct_nav_onboard/packets.py                Typed sensor/command packets
instinct_nav_onboard/obs_builder.py            High-level observation builder interface
instinct_nav_onboard/runtime.py                5 Hz SRU runtime loop wrapper
instinct_nav_onboard/bridges.py                Abstract robot/instinct_onboard bridges
instinct_nav_onboard/sru_checkpoint_policy.py  Python checkpoint loader for scene-token ActorCriticSRU
scripts/run_onboard_stub.py                    Dry-run entry point for local integration
scripts/g1_nav_onboard_onnx.py                 Production ROS2/instinct_onboard ONNX entry point
scripts/g1_nav_onboard.py                      Debug-only Python checkpoint integration entry point
```

## Deployment checklist

1. Copy this repository to the robot PC or user workstation.
2. Use the robot-side `instinct_onboard/instinct_venv` with `onnxruntime`, NumPy, OpenCV, PyYAML and `pyrealsense2`. PyTorch is not required for production ONNX deployment.
3. Bring the exported high-level ONNX bundle and low-level Instinct artifacts:

```text
artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx
artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx
artifacts/high_level/sru_scene_token/deploy_manifest.json
data&model/checkpoints/stand_onboard/exported/actor.onnx
data&model/checkpoints/stand_onboard/exported/0-depth_encoder.onnx
data&model/checkpoints/stand_onboard/params/env.yaml
data&model/checkpoints/parkour_onboard_preview_stair/exported/actor.onnx
data&model/checkpoints/parkour_onboard_preview_stair/exported/0-depth_encoder.onnx
data&model/checkpoints/parkour_onboard_preview_stair/params/env.yaml
```

These are the official InstinctLab deployment policies. The stand and stair
policies have the same ONNX input/output shapes but different weights. Always
pass `stand_onboard` as `--standdir` and `parkour_onboard_preview_stair` as
`--logdir`; never point both arguments at one directory. The `data&model`
bundle is intentionally local-only and is not committed to this repository.

The robot-tested `instinct_onboard` source is vendored directly under
`external/instinct_onboard`. It includes the joystick emergency-stop diagnostics,
torque-clipping counters, RealSense dropped-frame handling, and shared-memory
cleanup fixes. A normal clone contains these changes; no submodule initialization
or patch application is required.

4. Run dry mode first: policy outputs commands, but robot actuators are not enabled.
5. Then enable the low-level policy with strict command clamps and watchdogs.

For the torch-free production path install `requirements.txt`. The legacy Python
checkpoint/debug tools additionally require `requirements-debug.txt`.

## Recommended first integration target

Start with high-level-only dry run:

```text
real sensors -> HighLevelObsBuilder -> SRU -> print/log velocity command
```

Only after the command stream is numerically sane should it be connected to
`instinct_onboard`.

## instinct_onboard integration point confirmed

After inspecting `external/instinct_onboard`, the correct low-level command hook is:

```text
scripts/g1_parkour.py::G1ParkourNode.base_velocity_cmd
instinct_onboard/agents/parkour_agent.py::ParkourAgent._get_base_velocity_cmd_obs()
```

`G1ParkourNode.main_loop_callback()` currently overwrites `base_velocity_cmd` from
the Unitree joystick every 20 ms:

```text
self.base_velocity_cmd = self._compute_velocity_from_joystick()
```

For autonomous navigation, replace that assignment with the latest SRU command,
or subclass the node so the command source is configurable. The low-level
`ParkourAgent` will then consume the same buffer without changing its ONNX actor
or joint-command path.

This repository provides:

```text
instinct_nav_onboard.instinct_onboard_bridge.InstinctOnboardVelocityCommandSink
```

which writes a `VelocityCommand` into `node.base_velocity_cmd`.

Recommended integration on the robot PC:

```text
1. Bring up instinct_onboard G1ParkourNode in dryrun.
2. Disable joystick overwrite of base_velocity_cmd, or only use joystick as e-stop/mode switch.
3. Run SRU high-level runtime at 5 Hz.
4. Sink SRU command into G1ParkourNode.base_velocity_cmd.
5. Let ParkourAgent keep running at 50 Hz and sending joint targets.
```

## Legacy Python checkpoint entry script

`deploy/scripts/g1_nav_onboard.py` is a legacy debug/reference entry point. It
reuses instinct_onboard's G1 parkour low-level stack and loads the SRU checkpoint
through PyTorch. It does not currently carry the full production ONNX path's
fault latch, odom watchdog, and manifest validation semantics. Do not use it as
the production real-robot entry point.

Debug-only dry-run wiring command on the robot PC:

```bash
PYTHONPATH=/path/to/instinct_nav/deploy:/path/to/instinct_onboard:$PYTHONPATH \
python /path/to/instinct_nav/deploy/scripts/g1_nav_onboard.py \
  --standdir /path/to/parkour_agent_logdir \
  --logdir /path/to/parkour_agent_logdir \
  --auto_start_parkour \
  --goal_pose_topic /instinct_nav/goal_pose
```

This command defaults to dry-run. It publishes to instinct_onboard's dry-run
lowcmd topic and uses `ZeroHighLevelPolicy` if `--sru_policy_path` is omitted.
Use this only to validate ROS topics, RealSense frames, low-level ONNX loading,
and the `base_velocity_cmd` bridge.

Do not pass `--nodryrun` until all of these are verified on the robot PC:

```text
- /lowstate and /secondary_imu update normally
- RealSense depth frames update normally
- ParkourAgent loads exported low-level ONNX models
- base_velocity_cmd is updated by SRU runtime at 5 Hz
- dry-run /lowcmd_* contains finite joint targets
```

Debug-only Python SRU checkpoint path:

```bash
PYTHONPATH=/path/to/instinct_nav/deploy:/path/to/instinct_nav/source/rsl_rl:/path/to/instinct_onboard:$PYTHONPATH \
python /path/to/instinct_nav/deploy/scripts/g1_nav_onboard.py \
  --standdir /path/to/parkour_agent_logdir \
  --logdir /path/to/parkour_agent_logdir \
  --sru_checkpoint_path /path/to/model_2999.pt \
  --depth_encoder_path /path/to/instinct_nav/artifacts/sru/depth_encoder/vae_pretrain_new.pth \
  --debug_dump_dir /path/to/instinct_nav_onboard_debug/$(date +%Y%m%d_%H%M%S) \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 1000 \
  --sru_device cpu \
  --auto_start_parkour \
  --goal_pose_topic /instinct_nav/goal_pose
```

`--sru_checkpoint_path` directly loads the trained Python `ActorCriticSRU`
checkpoint. Use this only for debugging observation construction against the
training-time Python modules. For real robot production trials, use
`deploy/scripts/g1_nav_onboard_onnx.py`.

Important: this debug path can help inspect the policy checkpoint, depth encoder,
scene-token builder and command bridge in one Python path. Before trusting any
deployment path, compare logged real-robot observations against IsaacLab eval
samples, especially:

```text
- projected_gravity_b sign/order
- base_ang_vel_b scale/order
- real depth range and invalid-value convention
- encoded_depth mean/std and shape 64x5x8
- scene token [x, y, visible_ratio, depth_mean, depth_edge, frontier] ranges
- SRU command clamp and low-level base_velocity_cmd history
```

The deploy scene-token builder currently uses current-frame torso depth and local
visibility accumulation. Training used IsaacLab camera/root poses for delayed
BEV warping, so odom/root-pose warping is the next parity item before serious
robot trials.

## Observation debug dumps

Use `--debug_dump_dir` during dry-run before any real robot trial. The runtime
writes one `index.jsonl` plus optional per-step `.npz` files. Each dumped step
contains:

```text
base_lin_vel_b, base_ang_vel_b, projected_gravity_b
joint_pos, joint_vel
depth_m                         shape 36x64
goal_b                          shape 4
encoded_depth                   shape 2560
scene_tokens                    shape 384
actor_obs                       shape 2960
raw_action                      shape 3, unclamped SRU output
command                         shape 3, clamped velocity command sent to instinct_onboard
```

For long runs, use stats-only mode to avoid large files:

```bash
--debug_dump_dir /path/to/dump --debug_dump_stats_only --debug_dump_every_steps 1 --debug_dump_max_steps 2000
```

The first comparison target is not success rate; it is distribution parity with
IsaacLab eval samples. Check sign/order and numeric range before passing
`--nodryrun`.


## Production ONNX entry point

Use `deploy/scripts/g1_nav_onboard_onnx.py` for real robot deployment. Important
field behavior:

```text
--camera_frame_timeout_s   depth frame watchdog; timeout latches autonomy fault
--odom_timeout_s           odom velocity watchdog; timeout latches autonomy fault
--odom_twist_frame auto    body if child_frame_id is base_link/base/body/torso_link/robot_center, otherwise world->body rotation
--fall_tilt_threshold_deg  torso tilt threshold for fall motor-off (default 55 deg)
--fall_tilt_duration_s     required sustained tilt before motor-off (default 0.15 s)
--safety_log_dir           persistent 10 Hz incident JSONL directory; required with --nodryrun
```

The production node also requires `/wirelesscontroller`: `R1` returns autonomous
navigation to stand, while `R2` or `L2` triggers the motor-off emergency stop.
For end-to-end dry-run validation, `--auto_start_parkour` skips cold-start only
when actuator output is disabled. It waits with a zero command and does not run
the SRU until a real goal arrives; real-robot startup still performs cold-start.

For field tests, prefer explicit `--odom_twist_frame body` or
`--odom_twist_frame world` after inspecting `/dog_odom`. The `auto` mode is only
a convenience heuristic. If any autonomy fault is latched, the node stays in
`stand` and ignores new navigation goals until the node is restarted.

Real-robot mode now refuses to start without `--safety_log_dir`. Store both
that directory and `--debug_dump_dir` under `/workspace`, not `/tmp`, so a
power cycle does not erase incident evidence. The fall monitor checks torso
projected gravity before every 50 Hz actuator step. If tilt exceeds the
configured threshold continuously, or the IMU gravity vector becomes invalid,
it writes a critical event, publishes the existing motor-off command, and
exits. Motor-off makes the robot limp: a support harness and a spotter are still
required and this software guard is not a substitute for the physical E-stop.

## DynamicMapMask YOLO11 segmentation

Training continues to use the IsaacLab GT dynamic-obstacle mask with corruption.
Deployment uses a single-stage pipeline: registered RealSense RGB -> YOLO11-seg
instance masks -> depth-aligned 64x64 BEV -> current-only `dynamic_presence`.

The local model assets are intentionally ignored by Git:

```text
artifacts/deploy_models/yolo11_seg/yolo11s-seg.pt
```

Default weight provenance:

```text
yolo11s-seg.pt  sha256 1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3
```

Install the segmentation dependency in the robot/deployment environment, not in
the Isaac Sim training environment (their `filelock` requirements conflict):

```bash
pip install -r deploy/requirements-yolo.txt
```

Export a DynamicMapMask checkpoint with explicit token semantics:

```bash
python deploy/scripts/export_high_level_sru_onnx.py \
  --checkpoint_path /path/to/model.pt \
  --output_dir /path/to/export \
  --scene_token_layout dynamic_map_mask \
  --require_onnx_parity
```

Add the backend to the production command for dry-run validation:

```bash
--scene_token_layout dynamic_map_mask \
--dynamic_mask_backend yolo11_seg \
--yolo11_seg_weights artifacts/deploy_models/yolo11_seg/yolo11s-seg.pt \
--yolo_dynamic_class_ids 0 \
--yolo_confidence 0.25 \
--yolo_iou 0.45 \
--yolo_device 0
```

Class IDs refer to the segmentation checkpoint. The official YOLO11s-seg weights
use COCO classes; `0` selects people. A semantic class does not prove that an
instance is moving, so deployments that must distinguish stationary people or
vehicles need temporal tracking.

YOLO11-seg removes the expensive SAM ViT-H image encoder. A real-weight CPU smoke
on the development EPYC host measured 25.62 ms per 480x270 frame after warm-up
(39.03 FPS), including Ultralytics preprocessing and mask union. This is not a
target-computer deadline result. The provider remains dry-run-only until it is
measured on the robot computer or perception is moved out of the 50 Hz control
callback. Debug dumps include `color_rgb` and `dynamic_mask` for checking
registration and mask quality.
