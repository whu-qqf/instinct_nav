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

For now, deploy the current policy by loading the PyTorch checkpoint and the local
`source/rsl_rl` `ActorCriticSRU` module directly in Python. JIT/ONNX export should
be treated as a follow-up task.

## Files

```text
instinct_nav_onboard/config.py                 Shared constants from the training task
instinct_nav_onboard/packets.py                Typed sensor/command packets
instinct_nav_onboard/obs_builder.py            High-level observation builder interface
instinct_nav_onboard/runtime.py                5 Hz SRU runtime loop wrapper
instinct_nav_onboard/bridges.py                Abstract robot/instinct_onboard bridges
instinct_nav_onboard/sru_checkpoint_policy.py  Python checkpoint loader for scene-token ActorCriticSRU
scripts/run_onboard_stub.py                    Dry-run entry point for local integration
scripts/g1_nav_onboard.py                      ROS2/instinct_onboard integration entry point
```

## Deployment checklist

1. Copy this repository to the robot PC or user workstation.
2. Install a lightweight Python env with PyTorch, NumPy and the robot SDK.
3. Add `source/rsl_rl` to `PYTHONPATH` so `ActorCriticSRU` can be constructed.
4. Bring the trained SRU checkpoint and low-level Instinct checkpoint:

```text
logs/rsl_rl/.../model_2999.pt
artifacts/low_level/g1_parkour/model_30000.pt
artifacts/sru/depth_encoder/vae_pretrain_new.pth
```

5. Implement a concrete `SensorSource` for the robot sensors.
6. Implement a concrete `VelocityCommandSink` that writes to `instinct_onboard`.
7. Run dry mode first: policy outputs commands, but robot actuators are not enabled.
8. Then enable the low-level policy with strict command clamps and watchdogs.

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

## G1 navigation onboard entry script

`deploy/scripts/g1_nav_onboard.py` is the first real integration entry point.
It reuses instinct_onboard's G1 parkour low-level stack and replaces the joystick
velocity source with an SRU command cache when the node is in `parkour` mode.

Dry-run wiring command on the robot PC:

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

Current real SRU policy path:

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
checkpoint. This is the correct route for the current scene-token policy. The
older `--sru_policy_path` is reserved for a future scene-token-capable
TorchScript export.

Important: this proves the policy checkpoint, depth encoder, scene-token builder
and command bridge can run in one Python path. Before enabling `--nodryrun`, still
compare logged real-robot observations against IsaacLab eval samples, especially:

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

