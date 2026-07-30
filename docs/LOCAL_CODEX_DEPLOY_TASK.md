# Local Codex Task: instinct_nav Onboard Dry-Run Deployment

Date: 2026-07-28
Executor: Local Codex on QQF workstation / robot PC
Project: `instinct_nav`

> Status note (2026-07-30): this document records the original workstation
> integration task. The production robot entry point is now
> `deploy/scripts/g1_nav_onboard_onnx.py`. Follow `deploy/README.md` and
> `DEPLOYMENT_MEMORY.md` for the current artifact layout, safety requirements,
> and robot-computer commands. Keep the Python checkpoint flow below for
> observation/debug comparison only.

## Objective

Set up and validate the lightweight onboard deployment path for `instinct_nav` on the local Ubuntu 24.04 + ROS2 Jazzy machine, reusing the existing `ros_venv` and `colcon_ws` environment from the previous BeyondMimic reproduction.

The goal is **not** to run real robot closed-loop immediately. The goal is to run **dry-run only**, collect onboard observation/action debug dumps, and compare them with the IsaacLab eval reference dump generated on the server.

Do not pass `--nodryrun` unless the user explicitly confirms after dump comparison.

## Existing Local Environment

The user previously ran BeyondMimic with:

```bash
cd ~/colcon_ws/
source ~/ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash

# MuJoCo sim
ros2 launch motion_tracking_controller mujoco.launch.py   policy_path:=/home/qqf/2025-12-13_11-00-23_tennis_retarget.onnx

# Real robot
ros2 launch motion_tracking_controller real.launch.py   network_interface:=enp0s31f6   policy_path:=/home/qqf/2025-12-12_15-52-51_run_retarget.onnx
```

Assume ROS2 Jazzy, `colcon_ws`, Unitree-related packages, and `ros_venv` may already be partially configured.

## Important Compatibility Note

`instinct_onboard` upstream says it was tested on:

```text
Ubuntu 22.04 + ROS2 Humble + Unitree G1 Jetson Orin NX, 29DoF
```

The user machine is Ubuntu 24.04 + ROS2 Jazzy. Continue with Jazzy first because the user already has a working sim2real environment, but treat ROS API/package errors as expected compatibility issues.

## Repository Layout Expected Locally

Recommended location:

```text
~/instinct_nav
~/instinct_nav/external/instinct_onboard
~/colcon_ws
~/ros_venv
```

Clone if missing:

```bash
cd ~
git clone https://github.com/whu-qqf/instinct_nav.git
cd ~/instinct_nav
mkdir -p external
git clone https://github.com/project-instinct/instinct_onboard.git external/instinct_onboard
```

## Required Artifacts

Copy these from the training server to the local machine:

```text
SRU high-level checkpoint:
  model_2999.pt

SRU depth encoder:
  artifacts/sru/depth_encoder/vae_pretrain_new.pth

Instinct low-level G1 parkour export/logdir:
  stand agent logdir
  parkour agent logdir/exported/actor.onnx
  parkour agent logdir/exported/0-depth_encoder.onnx
```

Suggested local placement:

```text
~/instinct_nav/artifacts/checkpoints/model_2999.pt
~/instinct_nav/artifacts/sru/depth_encoder/vae_pretrain_new.pth
~/instinct_nav/artifacts/low_level/g1_parkour/
```

Do not invent paths. If artifacts are missing, ask the user where they are.

## Environment Setup Commands

Use the existing ROS virtual environment:

```bash
cd ~/colcon_ws
source ~/ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash
```

Install deploy dependencies:

```bash
cd ~/instinct_nav
pip install -r deploy/requirements.txt
pip install opencv-python
```

Install `instinct_onboard` into the same venv. Start with CPU ONNXRuntime for dry-run stability:

```bash
cd ~/instinct_nav/external/instinct_onboard
FORCE_CPU=1 pip install -e .[all]
```

If OpenCV conflicts with the user's system OpenCV, retry:

```bash
FORCE_CPU=1 pip install -e .[noopencv]
```

## Environment Sanity Checks

Run these in a fresh terminal:

```bash
cd ~/colcon_ws
source ~/ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash

export PYTHONPATH=~/instinct_nav/deploy:~/instinct_nav/source/rsl_rl:~/instinct_nav/external/instinct_onboard:$PYTHONPATH

lsb_release -a
echo $ROS_DISTRO
python --version
python -c "import rclpy; print('rclpy ok')"
python -c "import cv2; print('cv2 ok')"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import instinct_nav_onboard; print('instinct_nav_onboard ok')"
python -c "import instinct_onboard; print('instinct_onboard ok')"
ros2 pkg list | grep -i unitree || true
ros2 pkg list | grep -i ros2_numpy || true
```

Record failures exactly. Do not hide traceback.

## First Runtime Test: No ROS / No Robot Stub

Run:

```bash
cd ~/instinct_nav
source ~/ros_venv/bin/activate
export PYTHONPATH=~/instinct_nav/deploy:$PYTHONPATH

python ~/instinct_nav/deploy/scripts/run_onboard_stub.py --steps 3
```

Expected output resembles:

```text
[instinct_nav_onboard] t=... vx=+0.000 vy=+0.000 wz=+0.000
```

This only validates the deploy Python package and runtime wiring.

## Second Runtime Test: G1 Onboard Dry-Run

Do not add `--nodryrun`.

Use the actual local artifact paths. Example:

```bash
cd ~/colcon_ws
source ~/ros_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash

export PYTHONPATH=~/instinct_nav/deploy:~/instinct_nav/source/rsl_rl:~/instinct_nav/external/instinct_onboard:$PYTHONPATH

python ~/instinct_nav/deploy/scripts/g1_nav_onboard.py   --standdir ~/instinct_nav/artifacts/low_level/g1_parkour   --logdir ~/instinct_nav/artifacts/low_level/g1_parkour   --sru_checkpoint_path ~/instinct_nav/artifacts/checkpoints/model_2999.pt   --depth_encoder_path ~/instinct_nav/artifacts/sru/depth_encoder/vae_pretrain_new.pth   --debug_dump_dir ~/instinct_nav/onboard_debug_dump/$(date +%Y%m%d_%H%M%S)   --debug_dump_every_steps 1   --debug_dump_max_steps 1000   --sru_device cpu   --auto_start_parkour \
  --goal_pose_topic /instinct_nav/goal_pose
```

Expected behavior:

- No real actuator command should be sent because `--nodryrun` is omitted.
- SRU checkpoint loads.
- SRU depth encoder loads.
- Low-level parkour ONNX exports load.
- ROS node starts.
- Debug dump directory is created.
- `index.jsonl` and `step_*.npz` files appear after high-level SRU steps.

If it fails before dump generation, capture the full traceback.

## Debug Dump Fields

Each `.npz` should contain:

```text
depth_m                         shape 36x64
encoded_depth                   shape 2560
scene_tokens                    shape 384
actor_obs                       shape 2960
raw_action                      shape 3
command                         shape 3
base_lin_vel_b                  shape 3
base_ang_vel_b                  shape 3
projected_gravity_b             shape 3
goal_b                          shape 4
joint_pos                       shape 29
joint_vel                       shape 29
```

`index.jsonl` contains per-step min/max/mean/std summaries.

## Reference Eval Dump From Server

The server-side IsaacLab reference dump was generated at:

```text
/workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-26_21-39-31_g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1/eval_debug_dump
```

It had:

```text
episodes: 50
successes: 45
success_rate: 0.90
saved debug steps: 178
actor_obs: 2960
encoded_depth: 2560
scene_tokens: 384
depth_m: 36x64
```

Local Codex should not need this absolute server path unless the dump is copied locally. Ask the user for the local reference dump path if comparing locally.

## What To Compare Before Real Robot Closed Loop

Compare onboard dry-run dump against IsaacLab eval reference dump:

```text
projected_gravity_b
base_ang_vel_b
depth_m
encoded_depth
scene_tokens
actor_obs
raw_action
command
```

Critical checks:

```text
projected_gravity_b standing should be close to [0, 0, -1]
base_ang_vel_b sign/order/scale should be plausible
raw depth shape should be 36x64 after resize
invalid depth convention should match expectations
encoded_depth should not be all zero and should have similar mean/std to eval
scene_tokens should not be all zero and should be in a similar range
actor_obs should be exactly 2960-D
raw_action should not be persistently exploding
command should be clamp-limited before entering base_velocity_cmd
```

## Do Not Do Yet

Do not do these until the user and remote Codex confirm dump parity:

```text
Do not pass --nodryrun.
Do not run full real robot autonomous navigation.
Do not modify low-level ONNX policy.
Do not change action scaling unless evidence shows mismatch.
Do not replace ROS2 Jazzy/Humble compatibility code blindly.
```

## Likely Failure Points On Ubuntu 24.04 / ROS2 Jazzy

Expect possible issues in:

```text
ros2_numpy package availability
Unitree msg package names or generated Python module paths
rclpy API differences from Humble
pyrealsense2 / RealSense permissions
network_interface handling in instinct_onboard
crc_module.so lookup
ONNXRuntime CPU/GPU package conflicts
OpenCV ABI conflicts
```

When a failure occurs, collect:

```bash
echo $ROS_DISTRO
python --version
pip list | grep -E 'torch|onnx|opencv|numpy|ros|unitree|instinct'
ros2 topic list -t
ros2 node list
```

Then send the full traceback to the remote Codex.

## Success Criteria For This Task

The task is complete when:

1. `run_onboard_stub.py --steps 3` works.
2. `g1_nav_onboard.py` starts in dry-run without `--nodryrun`.
3. A local onboard debug dump folder is produced.
4. The dump includes `index.jsonl` and at least several `step_*.npz` files.
5. The `.npz` shapes match the expected fields above.
6. No real robot closed-loop command has been enabled.
