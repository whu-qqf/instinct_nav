# Method

本文档记录 `instinct_nav` 当前冻结基线的方法设计。它和 [CodeReading.md](./CodeReading.md) 的侧重点不同：`CodeReading.md` 按代码执行顺序带读实现，本文档按算法系统说明，方便后续写论文、组会汇报和 ablation 设计。

当前冻结基线结果见 [RESULTS.md](./RESULTS.md)。

## 1. System Overview

`instinct_nav` 使用两层控制结构：

```text
High-level SRU navigation policy, 5 Hz
  input: proprioception + goal command + encoded depth
  output: SE(2) velocity command [vx, vy, wz]

Frozen InstinctLab G1 parkour low-level policy, 50 Hz
  input: locomotion proprio history + command history + visual history
  output: 29D joint position target
```

高层策略只学习导航，不直接学习关节运动。低层策略来自 InstinctLab G1 parkour checkpoint，并在导航训练中冻结。这样做的核心目的，是把困难问题拆成：

- 高层：根据目标和视觉地形输出可执行的速度命令。
- 低层：把速度命令稳定地转成 G1 人形机器人的全身运动。

当前主线 task id：

```text
Instinct-Nav-G1-SRU-v0
Instinct-Nav-G1-SRU-Dev-v0
Instinct-Nav-G1-SRU-Play-v0
```

任务注册在 [g1_instinct/__init__.py](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/__init__.py)。

## 2. Environment And Timing

基础导航环境配置在 [navigation_env_cfg.py](./source/instinct_nav/instinct_nav/navigation/navigation_env_cfg.py)，G1/InstinctLab 适配配置在 [g1_instinct/navigation_env_cfg.py](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)。

关键频率：

```text
sim.dt = 0.005 s             -> physics 200 Hz
low_level_decimation = 4     -> low-level action 50 Hz
PLANNING_FREQ = 5 Hz         -> high-level SRU action 5 Hz
high-level decimation = 40
episode_length_s = 60.0
```

高层每 0.2 s 输出一次 `[vx, vy, wz]`，底层在这 0.2 s 内持续以 50 Hz 执行冻结 locomotion policy。

G1 配置中几个关键对齐点：

- robot asset 使用 InstinctLab 的 `G1_with_shoe_CFG`。
- robot 初始高度为 `z = 0.9`。
- 高层 command 写入 `base_velocity`，低层从该 command group 读取速度目标。
- raycast camera 和 height scanner 都挂到 `torso_link`。
- 训练地形难度固定为 `difficulty_range = [0.5, 1.0]`，关闭 terrain curriculum。

运行时环境类在 [navigation_env.py](./source/instinct_nav/instinct_nav/navigation/navigation_env.py)。它的主要作用是先创建 `ObservationDelayManager`，再加载 IsaacLab manager，因此 delayed observation 在 observation manager 第一次计算前已经可用。

## 3. Observation Design

高层 actor 和 critic 使用不同 observation group。

Actor policy observation：

```text
base_lin_vel          3
base_ang_vel          3
projected_gravity     3
last_action           3
target_position       4
depth_image        2560
-------------------------
total              2576
```

Critic observation：

```text
base_lin_vel             3
base_ang_vel             3
projected_gravity        3
last_action              3
target_position          4
time_normalized          1
height_scan_critic    3136
depth_image           2560
----------------------------
total                 5713
```

其中 16D proprioception 是：

```text
base_lin_vel       3
base_ang_vel       3
projected_gravity  3
last_action        3
target_position    4
```

`projected_gravity` 是世界重力方向投影到机器人 base/body 坐标系后的 3D 向量。它提供 roll/pitch 姿态信息，对 G1 低层平衡和高层判断失稳都很关键。

Actor 使用 delayed/noisy observation：

- `base_lin_vel_delayed`
- `base_ang_vel_delayed`
- `projected_gravity_delayed`
- `generated_commands_reshaped_delayed`
- `depth_image_noisy_delayed`

Critic 使用 cleaner/privileged observation：

- non-delayed proprioception
- clean goal command
- `time_normalized`
- privileged height scan feature
- perfect encoded depth

观测函数集中在 [observations.py](./source/instinct_nav/instinct_nav/navigation/mdp/observations.py)。

## 4. Goal Command

导航目标由 `robot_goal` command 生成。高层看到的目标不是世界坐标点，而是 SRU 风格的 body-frame goal observation：

```text
[dir_x, dir_y, dir_z, log(distance + 1)]
```

训练中 actor 侧目标加入噪声和 delay，critic 侧保持 clean。到达目标后 episode 终止，不在同一个 episode 内继续重采样目标。

当前 success 的语义是：episode 内机器人到达目标半径。由于到达目标会触发终止，日志里的 `early_termination` 大多数情况下代表成功终止，而不是异常失败。

## 5. Depth And Height Encoding

高层 depth 来自 `raycast_camera`。原始 camera 输出 `distance_to_image_plane`，shape 为 `40 x 64`：

```text
raw depth [B, 40, 64]
  -> nan/inf handling
  -> reshape [B, 1, 40, 64]
  -> depth encoder
  -> encoded depth [B, 64, 5, 8]
  -> flatten [B, 2560]
```

Actor 使用 `depth_image_noisy_delayed`，critic 使用 `depth_image_prefect`。G1 初始化时调用：

```python
initialize_depth_noise_generator(robot_name="g1", use_jit_precompiled=False)
```

这样 depth encoder 和 G1 camera config 在环境初始化阶段被固定。

Critic 额外使用 height scan：

```text
height scanner ray hits
  -> 51 x 51 height grid
  -> clamp [-5, 5]
  -> heightscan encoder
  -> encoded height [B, 64, 7, 7]
  -> flatten [B, 3136]
```

这部分只进入 value function，不进入 actor，所以推理时不需要 privileged height scan。

## 6. Action Bridge To Frozen Low-Level Policy

高层 action term 是 `InstinctFrozenNavigationSE2Action`，实现位于 [instinct_navigation_actions.py](./source/instinct_nav/instinct_nav/navigation/mdp/navigation/actions/instinct_navigation_actions.py)。

高层 actor 输出 3D action：

```text
[vx, vy, wz]
```

处理流程：

```text
SRU action
  -> tanh if gaussian policy distribution
  -> policy_scaling [1.0, 1.0, 1.0]
  -> low-pass filter alpha = 0.5
  -> write to base_velocity.vel_command_b
  -> compute Instinct low-level observation group
  -> frozen Instinct policy
  -> 29D joint position action
```

低层 checkpoint：

```text
artifacts/low_level/g1_parkour/model_30000.pt
```

低层 observation group 是 `instinct_low_level_policy`，定义在 [g1_instinct/navigation_env_cfg.py](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)。它保留 InstinctLab parkour policy 需要的历史观测：

```text
base_ang_vel        8-frame history
projected_gravity   8-frame history
velocity_commands   8-frame history
joint_pos           8-frame history
joint_vel           8-frame history
actions             8-frame history
depth_image         8 frames, history_skip_frames = 5
```

低层 policy 在训练中不更新参数。高层训练只改变 SRU policy，使其输出更适合 frozen G1 locomotion backend 的速度命令。

## 7. Actor-Critic SRU Network

网络实现位于 [actor_critic_sru.py](./source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)，SRU cell 位于 [lstm_sru.py](./source/rsl_rl/rsl_rl/networks/sru_memory/lstm_sru.py)。

关键网络参数：

```text
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
rnn_hidden_size = 512
rnn_num_layers = 1
dropout = 0.2
num_cameras = 1
image_input_dims = (64, 5, 8)
height_input_dims = (64, 7, 7)
use_actor_visibility_map = False
use_actor_visibility_query_attention = False
```

Actor forward：

```text
actor obs [B, 2576]
  -> split proprio [B, 16] and depth [B, 2560]
  -> depth reshape [B, 64, 5, 8]
  -> CrossAttentionFuseModule(depth, proprio)
  -> image feature [B, 64]
  -> concat [image feature 64 + proprio 16] = [B, 80]
  -> LSTM_SRU recurrent memory
  -> hidden [B, 512]
  -> LinearConstDropout
  -> actor MLP [512, 256, 128]
  -> action mean [B, 3]
  -> Normal(mean, std)
```

Critic forward：

```text
critic obs [B, 5713]
  -> split proprio [B, 16], time [B, 1], height [B, 3136], depth [B, 2560]
  -> height reshape [B, 64, 7, 7]
  -> depth reshape [B, 64, 5, 8]
  -> CrossAttentionFuseModule(height, proprio) -> [B, 64]
  -> CrossAttentionFuseModule(depth, proprio)  -> [B, 64]
  -> concat [height 64 + depth 64 + proprio 16] = [B, 144]
  -> LSTM_SRU recurrent memory
  -> hidden [B, 512]
  -> time embedding [B, 8]
  -> concat [B, 520]
  -> LinearConstDropout
  -> critic MLP [512, 256, 128]
  -> value [B, 1]
```

SRU cell 的核心是带 transform gate 的 recurrent update：

```text
concat [x_t, h_t]
  -> gates i, f, o, g
  -> transform_gate(x_t)
  -> refined forget gate
  -> c_next
  -> h_next
```

这使高层 policy 能利用历史观测，处理 partial observability、depth delay、目标遮挡和人形运动滞后。

## 8. Reward And Termination

当前 reward 保持导航 sparse/dense 混合风格：

```text
joint_acc_l2_joint          weight -1e-7
lateral_movement           weight -0.1
rot_movement               weight -1e-5
action_rate_l1             weight -0.1
episode_termination        weight -50.0
reach_goal_xy_soft         weight 0.25
reach_goal_xy_tight        weight 1.5
backward_movement_penalty  weight -0.0
```

主要 termination：

```text
time_out
base_contact
large_pitch_angle
early_termination
terrain_fall
```

G1 配置中 `base_contact` 只检查 `torso_link`，`large_pitch_angle` 使用 G1 的 bad orientation 逻辑，阈值为 `limit_angle = 1.0` rad。

评估结果显示主要失败来自：

- `base_contact`: 32 / 500
- `large_pitch_angle`: 24 / 500
- `time_out`: 5 / 500

这说明当前瓶颈主要是少量高难地形下的人形稳定性，而不是大规模目标采样失败。

## 9. PPO Training Configuration

Runner 配置位于 [rsl_rl_cfg.py](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/agents/rsl_rl_cfg.py)，继承 Go2 SRU runner 并覆盖 G1 相关项。

关键训练参数：

```text
seed = 72
num_steps_per_env = 16
num_mini_batches = 4
num_learning_epochs = 5
learning_rate = 1.0e-3
schedule = adaptive
desired_kl = 0.01
gamma = 0.995
lam = 0.95
entropy_coef = 0.005
value_loss_coef = 0.1
clip_param = 0.2
max_grad_norm = 1.0
empirical_normalization = False
save_interval = 500
```

训练使用 4096 env、3000 iterations：

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

## 10. Evaluation Protocol

当前 frozen baseline 使用：

```text
checkpoint: model_2999.pt
episodes: 500
num_envs: 256
overall success_rate: 0.902
```

分 terrain 结果：

| Terrain | Episodes | Success Rate |
| --- | ---: | ---: |
| `maze` | 151 | 0.8874 |
| `non_maze` | 96 | 0.8750 |
| `pits` | 106 | 0.8679 |
| `stairs` | 147 | 0.9592 |

评估命令和 GIF 录制命令见 [RESULTS.md](./RESULTS.md)。

## 11. Why The Final Baseline Improved

早期失败的主要原因不是单个 reward 项，而是多个接口细节没有同时对齐：

- 高层 actor/critic observation shape、顺序和 noisy/delayed 语义需要与 SRU policy 的切分逻辑一致。
- G1 低层 policy 不能只“接上 checkpoint”，还必须复刻 InstinctLab 的 low-level observation group、history、depth history 和 command 写入方式。
- 高层 action 需要经过与低层命令接口一致的 tanh、scaling 和 low-pass filter。
- runner 入口、rsl_rl、ActorCriticSRU、dropout mask、hidden state reset、dict obs flatten 顺序都需要保持一致。
- camera/depth encoder 的输出必须是网络预期的 `[64, 5, 8]` feature，而不是任意 depth tensor。
- task registration 和 env inheritance 需要稳定，避免旧 alias、旧 wrapper 或旧 normalization 逻辑混入。

最终 `0.902` baseline 是在这些链路被统一后得到的，因此后续不应继续无记录修改主线。任何新变化都应该作为单独 ablation 运行，并与 `model_2999.pt` 结果对比。

## 12. Recommended Ablations

冻结当前 baseline 后，可以做以下独立实验：

- Two-camera vs one-camera depth input.
- Visibility map / query attention on vs off.
- Low-pass filter alpha sweep.
- Goal noise and observation delay ablation.
- Reward weight sweep for `action_rate_l1` and `episode_termination`.
- Terrain-specific fine-tuning for pits or high-level stairs.
- Different random seeds with the same final implementation.

每个 ablation 都应使用新的 run name，并在 [RESULTS.md](./RESULTS.md) 中追加结果，不覆盖当前 golden baseline。
