# SRU + InstinctLab G1 训练复现文档

更新时间：2026-07-13

本文档记录当前 `/workspace/sru` 中 SRU navigation + InstinctLab G1 frozen low-level locomotion policy 的完整链路。目标是让后续复现时能明确：用哪个环境、哪个 checkpoint、哪些接口、哪些配置、怎么启动训练、怎么诊断问题。

## 当前有效链路

```text
SRU high-level navigation policy
  -> 3D SE(2) velocity action [vx, vy, wz]
  -> InjectedVelocityCommand(base_velocity)
  -> frozen InstinctLab G1 low-level policy observation
  -> EncoderMoEActorCritic frozen policy
  -> 29D raw action
  -> InstinctLab JointPositionActionCfg(scale=beyondmimic_action_scale, use_default_offset=True)
  -> Isaac Sim G1 29DoF robot
```

当前训练任务：`Isaac-Nav-PPO-G1-Instinct-v0`。

当前活跃训练进程：

```text
PID: 521178
cmd: /workspace/sru/.venv-isaacsim510/bin/python scripts/train.py --task Isaac-Nav-PPO-G1-Instinct-v0 --num_envs 4096 --max_iterations 15000 --run_name sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_20260712_1100 --headless
```

当前 run 目录：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-12_11-05-27_sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_20260712_1100
```

当前 TensorBoard：

```text
http://0.0.0.0:6006/
log file: /workspace/sru/logs/tensorboard_g1_instinct_6006.log
```

截至 2026-07-13 10:34 左右，训练到约 iteration 3417，最新已保存 checkpoint 为 `model_3000.pt`。日志片段显示：

```text
success_rate ~= 0.90
base_contact ~= 0.07
large_pitch_angle ~= 0.067
terrain_fall ~= 0.000-0.001
steps/s ~= 2700
GPU memory ~= 29.5 GB
```

## 版本和路径

主工作目录：`/workspace/sru`。

Python / Isaac Sim 环境：

```text
/workspace/sru/.venv-isaacsim510
python: 3.11.0rc1
torch: 2.12.0.dev20260408+cu128
isaacsim: 5.1.0.0
```

辅助 Python 环境：

```text
/workspace/sru/.venv-sru311
python: 3.11.0rc1
```

源码 checkout：

```text
/workspace/sru/external/IsaacLab                  commit 0f00ca2b
/workspace/sru/external/IsaacLab-instinct-v232     commit 37ddf626
/workspace/sru/external/InstinctLab                commit ba28d3d
/workspace/sru/external/instinct_rl                commit ba45ed2
/workspace/sru/external/GR00T-WholeBodyControl     commit f90ed64
```

当前链路已经放弃 SONIC/GR00T low-level controller。`GR00T-WholeBodyControl` 不是当前训练链路的核心依赖。

GPU/driver 当前状态示例：

```text
GPU: NVIDIA RTX PRO 6000 Blackwell
Driver: 580.105.08
CUDA driver: 13.0
GPU memory: 97,887 MiB
```

## 关键权重和数据

Frozen low-level G1 InstinctLab checkpoint：

```text
/workspace/sru/logs/instinct_rl_isaacsim510/20260709_223509/model_30000.pt
```

同目录包含原始 InstinctLab 训练输出：

```text
/workspace/sru/logs/instinct_rl_isaacsim510/20260709_223509/
  model_5000.pt
  model_10000.pt
  model_15000.pt
  model_20000.pt
  model_25000.pt
  model_30000.pt
  params/
  git/
```

可覆盖的环境变量：

```bash
export SRU_G1_INSTINCT_CHECKPOINT=/path/to/model.pt
export SRU_INSTINCT_RL_ROOT=/path/to/instinct_rl
export INSTINCTLAB_DATASET_ROOT=/workspace/sru/external/datasets/instinctlab
```

当前 SRU frozen low-level policy 执行不依赖 AMASS motion-reference dataset；原始 InstinctLab parkour 训练/诊断如果保留 motion reference，则需要 dataset。

## 任务注册

任务注册文件：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/__init__.py
```

注册任务：

```text
Isaac-Nav-PPO-G1-Instinct-v0       正式训练
Isaac-Nav-PPO-G1-Instinct-Dev-v0   诊断/简化地形
Isaac-Nav-PPO-G1-Instinct-Play-v0  play/可视化
```

正式训练入口：

```text
env_cfg_entry_point: G1InstinctNavigationEnvCfg
rsl_rl_cfg_entry_point: G1InstinctNavPPOVisMapRunnerCfg
```

## 环境配置

主配置文件：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/navigation_env_cfg.py
```

核心类：

```text
G1InstinctNavigationEnvCfg
G1InstinctNavigationEnvCfg_DEV
G1InstinctNavigationEnvCfg_PLAY
```

机器人配置直接复用 InstinctLab G1：`G1_with_shoe_CFG`。

URDF：

```text
/workspace/sru/external/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf
```

关键机器人设置：

```text
merge_fixed_joints = True
init_state.pos = (0.0, 0.0, 0.9)
contact sensor active
self_collision = true in saved env params
```

当前 run 的仿真频率：

```text
sim.dt = 0.005
NavigationEnvCfg.decimation = 40
high-level SRU env step = 0.2 s = 5 Hz
low_level_decimation = 4
low-level Instinct policy call cadence = 4 * 0.005 = 0.02 s = 50 Hz
```

也就是说，SRU 高层每 0.2s 输出一次 `[vx, vy, wz]`，wrapper 内部在 0.2s 内以 50Hz 多次运行 frozen low-level policy。

### 传感器

SRU policy 仍使用导航任务的 policy/critic observation；frozen low-level policy 使用单独的 `instinct_low_level_policy` observation group。

Instinct low-level camera：

```text
self.scene.camera = deepcopy(InstinctParkourSceneCfg(...).camera)
self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
mesh_prim_paths += get_link_prim_targets(G1_29DOF_LINKS)
```

Low-level depth observation term：

```python
depth_image = ObsTerm(
    func=instinct_mdp.delayed_visualizable_image,
    params={
        "data_type": "distance_to_image_plane_noised_history",
        "sensor_cfg": SceneEntityCfg("camera"),
        "history_skip_frames": 5,
        "num_output_frames": 8,
        "delayed_frame_ranges": (0, 1),
        "debug_vis": False,
    },
)
```

SRU navigation raycast camera 仍存在，用于 SRU depth/visibility/eval 可视化；不要把它和 Instinct low-level 的 `scene.camera` 混淆。

### Reset / events

G1 Instinct 配置覆盖了继承自 quadruped/SRU 的若干 reset/event：

```text
reset_base pose_range yaw: (-0.1, 0.1)
reset_base velocity_range x/y/z/roll/pitch/yaw: (-0.2, 0.2)
reset_robot_joints = reset_joints_by_offset
reset_robot_joints position_range=(0.0, 0.0)
reset_robot_joints velocity_range=(0.0, 0.0)
randomize_camera_height = None
randomize_action_scale = None
reset_delay_buffer = None
randomize_low_pass_filter_alpha = None
push_robot = None
```

这些随机化不是 frozen InstinctLab G1 low-level policy 在 SRU wrapper 中稳定运行的必要条件，早期大量跌倒和假终止与这些继承配置有关。

### Termination

当前 G1-specific termination：

```text
base_contact body_names = "torso_link", threshold = 1.0
large_pitch_angle = nav_mdp.bad_orientation(limit_angle=1.0)
```

早期继承的 navigation Euler-angle termination 会把几乎直立的 G1 误判为 large pitch，是已确认 bug。

### 地形

当前正式训练已经切回原始 SRU 地形难度：

```python
self.scene.terrain.max_init_terrain_level = 10
self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
self.scene.terrain.terrain_generator.curriculum = False
```

历史上曾用更温和地形启动过一条 run：

```python
max_init_terrain_level = 5
difficulty_range = [0.0, 0.5]
curriculum = False
```

但当前活跃 run 是：

```text
sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_20260712_1100
```

不要把它和 09:34/09:48 的 `historyfix_terrainstart` run 混用。

## Frozen low-level policy 接口

相关文件：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_frozen_backend.py
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_navigation_actions.py
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_navigation_actions_cfg.py
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/injected_velocity_command.py
```

SRU 高层 actor 输出 3D action：`[vx, vy, wz]`。

当前 action term：`InstinctFrozenNavigationSE2Action`。

高层 action 处理流程：

```text
raw SRU action
  -> if use_raw_actions=True, copy as processed action
  -> tanh(processed action) because policy_distr_type="gaussian"
  -> low-pass filter, alpha=0.5
  -> write to InjectedVelocityCommand.vel_command_b
```

因此 SRU actor 的实际速度命令范围被 tanh 限制在约 `[-1, 1]`。

`InjectedVelocityCommand` 是简单 command buffer：

```text
vel_command_b: [num_envs, 3]
command property returns vel_command_b
_resample_command resets selected envs to zero
_update_command is no-op
```

Frozen low-level observation term `velocity_commands` 从 `command_name="base_velocity"` 读取该 buffer。

### Low-level obs 格式

Frozen backend 期望原始 InstinctLab `EncoderMoEActorCritic` 的 flat obs：

```text
base_ang_vel        24
projected_gravity   24
velocity_commands   24
joint_pos          232
joint_vel          232
actions            232
depth_image       8 x 18 x 32 = 4608
-----------------------------------
total             5376
```

`instinct_low_level_policy` observation group 以 dict 形式计算，wrapper 按 backend 的 ordered obs_format 拼接。

Frozen backend 构建：

```python
modules.build_actor_critic(
    "EncoderMoEActorCritic",
    policy_cfg,
    obs_format,
    num_actions=29,
    num_rewards=1,
)
```

Policy 架构：

```text
actor_hidden_dims = [256, 128, 64]
critic_hidden_dims = [256, 128, 64]
activation = elu
num_moe_experts = 4
depth encoder output = 128
```

Checkpoint 加载后 `eval()` 并冻结所有参数。

### 29D low-level action 应用

Frozen policy 输出是 29D raw action，不是绝对 joint target。必须走 InstinctLab 的 JointPositionAction 语义：

```python
JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=beyondmimic_action_scale,
    use_default_offset=True,
)
```

不要把 frozen policy 输出直接写成绝对关节角。

### 最关键的历史修复

`ObservationManager.compute_group(...)` 默认 `update_history=False`。SRU wrapper 内部每个 high-level step 会多次调用 low-level policy；如果 low-level obs history 不更新，policy 会在 50Hz 子步中收到陈旧历史，导致快速跌倒。

正确 low-level call：

```python
observations = self._env.observation_manager.compute_group(
    group_name=self.cfg.observation_group,
    update_history=True,
)
```

这个修复是 SRU wrapper 与原始 InstinctLab policy 行为对齐的关键。

## SRU PPO 配置

配置文件：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/agents/rsl_rl_cfg.py
```

当前 run 的 `params/agent.yaml`：

```text
seed = 72
num_steps_per_env = 16
max_iterations = 15000
logger = tensorboard
experiment_name = g1_instinct_navigation_ppo_vismap
run_name = sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_20260712_1100
save_interval = 500
```

Policy：

```text
class_name = ActorCriticSRU
rnn_type = lstm_sru
rnn_hidden_size = 512
rnn_num_layers = 1
dropout = 0.2
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
num_cameras = 1
image_input_dims = (64, 5, 8)
height_input_dims = (64, 7, 7)
use_actor_visibility_map = False
use_actor_visibility_query_attention = False
```

PPO：

```text
num_learning_epochs = 5
num_mini_batches = 4
learning_rate = 0.001
schedule = adaptive
gamma = 0.995
lam = 0.95
entropy_coef = 0.005
clip_param = 0.2
value_loss_coef = 0.1
max_grad_norm = 1.0
desired_kl = 0.01
```

4096 env、16 steps/env 时，每 iteration 的 rollout samples：`4096 * 16 = 65536 transitions`。

`num_mini_batches=4` 意味着每个 PPO epoch 的 minibatch 大小约 16384 transitions。之前讨论过改成 8，但当前有效 run 已改回 4。

## 启动训练

推荐脚本：

```text
/workspace/sru/scripts/instinct_sru_train.sh
```

脚本会设置：

```bash
ISAACLAB_DIR=/workspace/sru/external/IsaacLab-instinct-v232
INSTINCTLAB_DIR=/workspace/sru/external/InstinctLab
NAV_TASK_DIR=/workspace/sru/external/IsaacLab/source/isaaclab_nav_task
PYTHON_BIN=/workspace/sru/.venv-isaacsim510/bin/python
TASK=Isaac-Nav-PPO-G1-Instinct-v0
NUM_ENVS=4096
MAX_ITERATIONS=15000
WANDB_MODE=offline
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

关键 PYTHONPATH：

```bash
/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab
/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_assets
/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_tasks
/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_rl
/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_mimic
/workspace/sru/external/InstinctLab/source/instinctlab
/workspace/sru/external/instinct_rl
/workspace/sru/external/IsaacLab/rsl_rl
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task
```

启动示例：

```bash
cd /workspace/sru
RUN_NAME=sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_$(date +%Y%m%d_%H%M) NUM_ENVS=4096 MAX_ITERATIONS=15000 nohup /workspace/sru/scripts/instinct_sru_train.sh   > /workspace/sru/logs/sru_g1_instinct_train_$(date +%Y%m%d_%H%M).log 2>&1 &
```

检查进程：

```bash
pgrep -af "scripts/train.py --task Isaac-Nav-PPO-G1-Instinct-v0"
ps -fp <PID>
nvidia-smi
```

停止训练：

```bash
kill <PID>
```

不要用 `kill -9`，除非普通 `kill` 无法结束；正常退出更利于写完日志。

## TensorBoard

当前 logger 是 tensorboard。启动方式示例：

```bash
/workspace/sru/.venv-sru311/bin/python -m tensorboard.main   --logdir /workspace/sru/external/IsaacLab/source/isaaclab_nav_task/logs/rsl_rl/g1_instinct_navigation_ppo_vismap   --host 0.0.0.0   --port 6006
```

重点看：

```text
Metrics/robot_goal/success_rate
Episode_Termination/base_contact
Episode_Termination/large_pitch_angle
Episode_Termination/terrain_fall
Episode_Termination/time_out
Metrics/robot_goal/velocity_toward_goal
Metrics/robot_goal/velocity_magnitude
Mean episode length
Mean reward
Value function loss
Surrogate loss
Mean action noise std
```

判断训练是否正常：早期不应大量 `base_contact` / `large_pitch_angle`；`terrain_fall` 应接近 0；如果 episode_length 极短且 base_contact/large_pitch_angle 同时很高，优先怀疑 low-level wrapper 或 reset/termination。

## 诊断脚本

SRU wrapper 内诊断 frozen low-level：

```text
/workspace/sru/scripts/diagnose_g1_instinct_low_policy.py
```

零命令诊断：

```bash
cd /workspace/sru/external/IsaacLab/source/isaaclab_nav_task
OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_assets:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_tasks:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_rl:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_mimic:/workspace/sru/external/InstinctLab/source/instinctlab:/workspace/sru/external/instinct_rl:/workspace/sru/external/IsaacLab/rsl_rl:/workspace/sru/external/IsaacLab/source/isaaclab_nav_task /workspace/sru/.venv-isaacsim510/bin/python /workspace/sru/scripts/diagnose_g1_instinct_low_policy.py   --task Isaac-Nav-PPO-G1-Instinct-v0 --num_envs 32 --steps 120 --vx 0 --vy 0 --wz 0 --headless
```

固定前进命令诊断：

```bash
cd /workspace/sru/external/IsaacLab/source/isaaclab_nav_task
OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_assets:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_tasks:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_rl:/workspace/sru/external/IsaacLab-instinct-v232/source/isaaclab_mimic:/workspace/sru/external/InstinctLab/source/instinctlab:/workspace/sru/external/instinct_rl:/workspace/sru/external/IsaacLab/rsl_rl:/workspace/sru/external/IsaacLab/source/isaaclab_nav_task /workspace/sru/.venv-isaacsim510/bin/python /workspace/sru/scripts/diagnose_g1_instinct_low_policy.py   --task Isaac-Nav-PPO-G1-Instinct-v0 --num_envs 32 --steps 120 --vx 0.45 --vy 0 --wz 0 --processed_command --headless
```

`--processed_command` 表示把输入速度当作 tanh 后的期望命令，脚本内部会反算 `atanh` 作为 raw action 喂给 wrapper。

原始 InstinctLab 环境 parity 诊断：

```text
/workspace/sru/scripts/diagnose_original_instinct_g1_parkour.py
```

关键参数：

```text
--task Instinct-Parkour-Target-Amp-G1-Play-v0
--policy_checkpoint /workspace/sru/logs/instinct_rl_isaacsim510/20260709_223509/model_30000.pt
--override_obs_command
```

历史结论：原始 InstinctLab env 中同一 checkpoint 的零命令 120 step 诊断稳定，`resets_mean=0`，这证明 checkpoint 本身不是坏的。SRU wrapper 早期不稳定的决定性原因是 low-level observation history 没有以 50Hz 更新。

## 关键历史问题和修复

1. step-0 大量 `large_pitch_angle`：继承的 navigation Euler-angle termination 不适合 G1。修复为 `nav_mdp.bad_orientation(limit_angle=1.0)`。
2. `reset_joints_by_scale` 不适合 G1：改为 `reset_joints_by_offset`，位置/速度 offset 都为 0。
3. low-level history 不更新：low-level policy 子步必须用 `compute_group(..., update_history=True)`。
4. robot root 与 spawn origin 不同步：G1 config 启用 `RobotNavigationGoalCommandCfg.sync_robot_root_to_spawn_origin=True`。

## Evaluation / GIF 可视化

主 eval 脚本：

```text
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/scripts/eval.py
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/scripts/eval_depth_viz.py
```

当前 GIF 面板顺序：

```text
BEV | rear-view RGB | noisy/delayed depth | processed depth | height | steep | local map | local gradient | visibility
```

后方相机当前定义：

```text
从机器人后方看向机器人；
位置默认：后退 4m、右偏 1.5m、上方 5m；
target 默认：robot root 上方 1.0m；
orientation: gravity_aligned_look_at；
up_axis = "Z"，画面竖直方向与重力方向一致，不跟随机器人 roll/pitch 翻转。
```

smoke 验证输出：

```text
/workspace/sru/.cache/query_attention_rgb_obs_gifs/g1_instinct_model1000_rear_gravity_smoke
/workspace/sru/.cache/logs/eval-g1-instinct-model1000-rear-gravity-smoke-summary.json
```

summary 中应看到：

```json
"rear_view_orientation": "gravity_aligned_look_at"
```

注意：可视化 depth panel 已改为 noisy/delayed depth，而不是 raw raycast depth。

## 复现检查清单

训练前：

```text
1. nvidia-smi 确认 GPU 可见。
2. pgrep -af scripts/train.py / scripts/eval.py 确认没有旧进程占 GPU。
3. 确认 checkpoint 存在：/workspace/sru/logs/instinct_rl_isaacsim510/20260709_223509/model_30000.pt。
4. Python 使用 /workspace/sru/.venv-isaacsim510/bin/python。
5. PYTHONPATH 使用 IsaacLab-instinct-v232 + InstinctLab + instinct_rl + local nav task。
6. 先跑 diagnose_g1_instinct_low_policy.py，确认 fixed command 不会大量 reset。
7. 再启动 4096-env 训练。
```

训练中：

```text
1. iteration time 正常约 24s；steps/s 约 2700。
2. GPU memory 约 29.5GB。
3. base_contact / large_pitch_angle 不应突然接近 1。
4. terrain_fall 应接近 0。
5. success_rate 应维持或提升；当前 3000+ iteration 已约 0.9。
6. 每 500 iteration 保存一次 model_*.pt。
```

训练后：

```text
1. 保存 run dir 和 log 文件。
2. 用 eval.py 对 checkpoint 生成 summary 和 GIF。
3. 检查 termination breakdown，不只看 success_rate。
4. 如果失败率异常，先跑 low-level fixed-command 诊断，而不是直接改 PPO。
```

## 常见坑

1. 不要用 SONIC/GR00T 的底层 controller 路径复现当前结果；当前底层是 InstinctLab G1 checkpoint。
2. 不要直接把 frozen policy 的 29D 输出当绝对 joint target。
3. 不要删除 `update_history=True`，这是 wrapper 稳定的关键。
4. 不要把 `scene.camera` 和 `raycast_camera` 混为一谈；Instinct low-level depth 用 `scene.camera`。
5. 不要用旧的温和地形 run 当作当前正式训练配置；当前正式 run 是原始 SRU 地形难度。
6. 不要只看 success_rate；同时看 base_contact、large_pitch_angle、terrain_fall、episode_length。
7. 如果 Isaac Sim import 触发 EULA，设置 `OMNI_KIT_ACCEPT_EULA=YES`。
8. 如果 shell 里 `CMAKE_PREFIX_PATH: unbound variable`，避免在 `set -u` 下直接引用未定义变量，或先 `export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}`。
9. 如果 headless render 报 GLFW/display warning，通常不是致命错误；看是否继续进入 env setup 和 evaluation/training。
10. 如果 TensorBoard 报 TensorFlow reduced feature set，通常仍可用。

## 当前文件索引

核心训练配置：

```text
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/navigation_env_cfg.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/agents/rsl_rl_cfg.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/config/g1_instinct/__init__.py
```

Action / command bridge：

```text
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_frozen_backend.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_navigation_actions.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/actions/instinct_navigation_actions_cfg.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/injected_velocity_command.py
external/IsaacLab/source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/injected_velocity_command_cfg.py
```

启动/诊断脚本：

```text
scripts/instinct_sru_train.sh
scripts/diagnose_g1_instinct_low_policy.py
scripts/diagnose_original_instinct_g1_parkour.py
```

Eval / GIF：

```text
external/IsaacLab/source/isaaclab_nav_task/scripts/eval.py
external/IsaacLab/source/isaaclab_nav_task/scripts/eval_depth_viz.py
external/IsaacLab/source/isaaclab_nav_task/tests/test_eval_depth_viz.py
```

当前训练输出：

```text
/workspace/sru/logs/sru_g1_instinct_train_20260712_1100_orig_sru_terrain_minibatch4.log
/workspace/sru/external/IsaacLab/source/isaaclab_nav_task/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-12_11-05-27_sru_g1_instinct_frozen_4096env_orig_sru_terrain_minibatch4_20260712_1100
```