# instinct_nav

`instinct_nav` 是当前用于 **G1 人形机器人高层导航 + 冻结 InstinctLab parkour 低层 locomotion** 的主实验项目。项目目标不是继续修补 `sru_new`，而是把 SRU 导航思想适配到 G1/InstinctLab 体系中，并保持底层运动能力复用、上层导航策略独立训练。

当前冻结基线结果、checkpoint、评估表格和复现命令见 [RESULTS.md](./RESULTS.md)。
方法设计说明见 [METHOD.md](./METHOD.md)，代码阅读路线见 [CodeReading.md](./CodeReading.md)。
汇报版摘要见 [REPORT.md](./REPORT.md)。

当前主线：

```text
High-level SRU navigation policy @ 5 Hz
  obs: proprio + goal_command + encoded_depth_image
  action: [vx, vy, wz]

Frozen InstinctLab G1 parkour locomotion policy @ 50 Hz
  obs: parkour proprio history + velocity_commands history + depth image history
  action: 29D joint position target
```

## 当前状态

已经完成的核心能力：

- 冻结低层 G1 parkour policy 接入，checkpoint 固定为 `artifacts/low_level/g1_parkour/model_30000.pt`。
- 高层只输出速度命令 `[vx, vy, wz]`，不直接输出关节动作。
- 高层 5 Hz，底层 50 Hz；一个高层 step 内执行多次冻结低层控制。
- SRU 高层网络已接入 depth CNN encoder、SRU recurrent stream、self/cross/query attention、TC-Dropout 风格 dropout。
- `visibility_map` 已接入，并支持 `early fusion`、`late fusion`、`query fusion` 三种融合方式。
- MDPO/DML 已接入，包含双策略 env split、互蒸馏 KL、双 storage 和 TC-Dropout mask replay。
- critic 已接入 privileged `encoded_height_scan=(64,7,7)` 和 `time_normalized`。
- SRU maze procedural terrain 已迁移到本项目。
- goal/spawn 采样已对齐 SRU mask 语义：有 terrain mask 时使用 `_height_field_valid_mask` 和 `_height_field_spawn_mask`。
- goal observation 已统一为 SRU 风格 `[dir_x, dir_y, dir_z, log(distance + 1)]`。
- policy 侧 goal observation 已加入 SRU 风格噪声：旋转、平移、概率扰动；critic 侧保持 clean。
- 到达目标后 episode 终止，不再重采样目标。
- 训练日志已修复 `success / goal_reached_count` 统计顺序问题。
- 已加入 `obstacle_proximity`、`terrain_drop_proximity`、`stuck_time`、`terrain_fall`、`large_tilt_angle` 等稳定性相关 reward/termination。
- 已支持 wandb 记录，并修复 step 非单调导致数据被忽略的问题。



## 关键文件

- 任务配置: [frozen_env_cfg.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/config/g1/frozen_env_cfg.py)
- task 注册: [__init__.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/config/g1/__init__.py)
- SRU 网络: [actor_critic_sru.py](/workspace/instinct_nav/instinct_nav/models/sru/actor_critic_sru.py)
- MDPO/DML: [mdpo.py](/workspace/instinct_nav/instinct_nav/algorithms/mdpo.py)
- goal command / metrics / terrain mask sampling: [commands.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/commands.py)
- rewards: [rewards.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/rewards.py)
- terminations: [terminations.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/terminations.py)
- visual observations: [visual_observations.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/visual_observations.py)
- goal observation noise: [custom_noise.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/custom_noise.py)
- 冻结低层 action bridge: [frozen_low_level_actions.py](/workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/frozen_low_level_actions.py)
- play 脚本: [scripts/play.py](/workspace/instinct_nav/scripts/play.py)
- eval 脚本: [scripts/evaluate.py](/workspace/instinct_nav/scripts/evaluate.py)
- launcher: [_launcher.py](/workspace/instinct_nav/_launcher.py)

## 推荐训练命令

当前正式基线优先使用 **SRU maze + PPO + no visibility map + sparse reward**。这是最接近 `docs/attention.md` 中师兄高成功率链路的配置；MDPO 和 QueryFusion 暂时作为 ablation，不作为默认路线。

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --run_name sru_maze_ppo_sparse_aligned_lowlevel_backend_v1 \
  --headless \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav
```

如果只做短程验证，可以先跑 1000 iteration，但不要用 1000 iteration 的 sparse reward 结果判断最终上限。

最小 smoke test：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Dev-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --run_name ppo_sparse_novis_smoke \
  --headless
```

对照 fusion / MDPO ablation：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-MDPO-EarlyFusion-v0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --run_name sru_maze_early_fusion_v1 \
  --headless

/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-MDPO-LateFusion-v0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --run_name sru_maze_late_fusion_v1 \
  --headless

/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-QueryFusion-SparseReward-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name sru_maze_ppo_query_sparse_ablation_v1 \
  --headless
```

## Play / Eval

播放训练好的 checkpoint：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Play-v0 \
  --checkpoint /workspace/instinct_nav/logs/instinct_nav/<run_dir>/model_1000.pt \
  --num_envs 1 \
  --steps 1200 \
  --log_interval 20
```

大地图 play：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
  --task Instinct-Nav-G1-Frozen-SruMaze-MDPO-QueryFusion-PlayLarge-v0 \
  --checkpoint /workspace/instinct_nav/logs/instinct_nav/<run_dir>/model_1000.pt \
  --num_envs 1 \
  --steps 2000 \
  --log_interval 20
```

录视频时使用 `--video --video_length N`，不要写成 `--video N`：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Play-v0 \
  --checkpoint /workspace/instinct_nav/logs/instinct_nav/<run_dir>/model_1000.pt \
  --num_envs 1 \
  --steps 1200 \
  --headless \
  --video \
  --video_length 1200
```

评估并保存失败/成功 GIF，默认使用跟随机器人俯视视角：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Play-v0 \
  --checkpoint /workspace/instinct_nav/logs/instinct_nav/<run_dir>/model_1000.pt \
  --num_envs 2 \
  --steps 3000 \
  --eval_episodes 50 \
  --save_failure_gifs \
  --save_success_gifs \
  --camera_mode topdown_follow \
  --camera_follow_height 12.0
```

## 当前独立项目任务 ID

正式 baseline：

```text
Instinct-Nav-G1-SRU-v0
Instinct-Nav-G1-SRU-Dev-v0
Instinct-Nav-G1-SRU-Play-v0
```

Visibility-map fusion ablation：

```text
Instinct-Nav-G1-SRU-Visibility-Early-v0
Instinct-Nav-G1-SRU-Visibility-Early-Dev-v0
Instinct-Nav-G1-SRU-Visibility-Early-Play-v0
Instinct-Nav-G1-SRU-Visibility-Late-v0
Instinct-Nav-G1-SRU-Visibility-Late-Dev-v0
Instinct-Nav-G1-SRU-Visibility-Late-Play-v0
Instinct-Nav-G1-SRU-Visibility-Query-v0
Instinct-Nav-G1-SRU-Visibility-Query-Dev-v0
Instinct-Nav-G1-SRU-Visibility-Query-Play-v0
```

这些是当前 `source/instinct_nav` 独立项目内注册的任务。下面早期 `Frozen-SruMaze` 任务名只作为历史迁移记录，不作为当前主线使用。

## 当前主要任务 ID

SRU maze 正式基线：

```text
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Dev-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Play-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-PlayLarge-v0
```

SRU maze fusion / MDPO variants：

```text
Instinct-Nav-G1-Frozen-SruMaze-MDPO-EarlyFusion-v0
Instinct-Nav-G1-Frozen-SruMaze-MDPO-LateFusion-v0
Instinct-Nav-G1-Frozen-SruMaze-MDPO-QueryFusion-v0
```

Parkour / hiking 相关实验任务：

```text
Instinct-Nav-G1-Frozen-ParkourTerrain-MDPO-v0
Instinct-Nav-G1-Frozen-ParkourTerrainSparseWall-MDPO-QueryFusion-v0
Instinct-Nav-G1-Frozen-ParkourTerrainNoWall-MDPO-QueryFusion-v0
Instinct-Nav-G1-Frozen-ParkourSruStairsSparseWall-MDPO-QueryFusion-v0
```

当前正式建议先用 `SruMaze-PPO-SparseReward` 复现稳定基线，再把 QueryFusion、MDPO/DML 和 hybrid terrain 作为独立 ablation。

## 训练日志怎么看

重点指标：

```text
Mean reward
Mean episode length
Metrics/robot_goal/success
Metrics/robot_goal/goal_reached_count
Metrics/robot_goal/goal_reached_steps
Metrics/robot_goal/goal_progress
Metrics/robot_goal/velocity_toward_goal
Metrics/base_velocity/error_vel_xy
Metrics/base_velocity/error_vel_yaw
Episode_Termination/base_contact
Episode_Termination/early_termination
Episode_Termination/terrain_fall
Episode_Termination/large_tilt_angle
distill_kl
```

指标解释：

- `success`：episode 级导航成功率，一个 episode 内至少到达目标一次即为成功。
- `goal_reached_count`：episode 内新到达目标的次数；当前到达目标即终止，所以通常应与 `success` 接近或相同。
- `goal_reached_steps`：处在目标半径内的 step 数；因为到达后很快终止，当前也经常与 `success` 接近。
- `early_termination`：当前语义下主要表示成功到达目标触发终止。
- `base_contact`：torso 等非法接触终止，高说明机器人摔倒或动作不稳定。
- `large_tilt_angle`：姿态倾斜过大终止，高说明底层稳定性或高层命令有问题。
- `terrain_fall`：掉下地形或高度异常终止。
- `goal_progress`：距离目标减少量，越大说明导航方向越正确。
- `velocity_toward_goal`：机器人速度在目标方向上的投影。
- `obstacle_proximity`：前方障碍太近的惩罚，来自深度图。
- `terrain_drop_proximity`：前方地形落差/坑边风险惩罚，来自 critic height scan。
- `stuck_time`：长时间低速度、低进展且目标仍较远时的惩罚。
- `distill_kl`：MDPO 双策略互蒸馏 KL，过高说明两个策略分歧较大。

注意：`Metrics/robot_goal/goal_distance` 是 episode 内累积/平均日志口径，不等同于当前瞬时距离；大数值不一定直接表示目标采样错误。

## 与原 SRU 的对齐情况

已经尽量对齐的部分：

- 高层 SRU 控制速度命令，不直接控制关节。
- 高层频率保持 5 Hz。
- 视觉输入经过 CNN encoder。
- 使用 self/cross/query attention 形式融合视觉、本体、目标和 visibility 信息。
- actor 使用深度视觉，critic 使用 privileged height scan。
- goal observation 使用 `[dir_xyz, log(distance + 1)]`。
- policy 侧加入 goal observation 噪声。
- 使用 SRU maze procedural terrain 和 mask-based goal/spawn sampling。
- 加入 MDPO/DML 双策略训练。

有意不同的部分：

- 机器人从原 SRU 的轮足机器狗换成 G1 人形。
- 低层从旧 SRU locomotion 换成冻结 InstinctLab G1 parkour policy。
- 低层深度图、proprio history、动作尺度和命令接口以 InstinctLab parkour 为准。
- PPO/runner 使用 `instinct_rl` 适配层，不直接复用旧 `rsl_rl` 运行时。
- 当前 reward 在 SRU 原始奖励基础上加入了 G1 稳定性、障碍接近、落差接近和卡住惩罚。

## 目标采样验证

当前 SRU maze 目标采样已做过独立检查：

```text
sampler_ok=True
valid mask shape=[4,299,299]
spawn mask shape=[4,299,299]
goal_counts min/mean/max=37526/51663.8/78555
spawn_counts min/mean/max=33362/37997.5/40787
distance quantiles min/p05/p25/p50/p75/p95/max=[0.14, 3.37, 9.90, 14.80, 21.45, 28.91, 35.64]
mean_distance=15.63
too_close(<0.5)=0.003
very_far(>30)=0.031
local spawn+goal selected tile ratio=1.000
```

结论：SRU maze 下采样逻辑能正确使用 terrain mask；如果训练失败，优先看稳定性、奖励尺度、低层命令和终止分布，不要先假设目标采样坏了。

## 资产状态

已经迁入本项目的关键资产：

```text
artifacts/low_level/g1_parkour/model_30000.pt
instinct_nav/assets/g1/g1_parkour_cfg.py
instinct_nav/assets/g1/unitree_g1.py
instinct_nav/assets/g1/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf
instinct_nav/assets/g1/resources/unitree_g1/
```

仍保留为外部依赖：

- IsaacLab / IsaacSim 运行时。
- `instinct_rl` runner/algorithm 基础设施。
- InstinctLab 的部分 parkour terrain generator、camera helper、locomotion MDP helper。

当前目标不是完全脱离 InstinctLab，而是先把会影响复现实验的机器人资产和预训练模型固定到 `instinct_nav` 内。

## 当前风险与下一步

当前主要风险：

- G1 在楼梯、坑边、墙角附近稳定性仍可能不足，重点看 `base_contact` 和 `large_tilt_angle`。
- parkour/hiking hybrid 地形的目标可达性仍比 SRU maze 更复杂，需要在 SRU maze 稳定后再切回去。
- `visibility_map` 当前是局部可见性信息，不是全局 SLAM 记忆，不能单独解决全局不可达目标。
- 冻结底层不擅长后退，因此 play/recovery 不应依赖倒退脱困；训练中应通过障碍/卡住惩罚让高层少进入死角。

建议顺序：

1. 优先跑 `SruMaze-PPO-SparseReward`，确认 `success_rate`、`base_contact`、`large_tilt_angle` 和 `time_out` 的趋势。
2. 正式对齐实验使用 `4096 env + 16 steps/env + minibatch 4 + 15000 iterations`；1000 iteration 只作为早期诊断。
3. 如果稳定性仍差，先检查 reset/event 是否按 `docs/attention.md` 生效，再考虑动作范围和 termination 阈值。
4. PPO/no-vis 基线稳定后，再做 `PPO-QueryFusion-SparseReward`、MDPO/DML 和 fusion variants。
5. SRU maze 稳定后，再回到 `ParkourSruStairsSparseWall` 这类 hybrid terrain；若出现封闭墙体不可达目标，再做 connected-component navigable mask。

## 与 sru_new 的关系

`sru_new` 现在只作为历史参考和实现对照。长期维护和训练入口应放在 `instinct_nav`，SRU 的网络结构、视觉处理、fusion 方式、DML 和地形语义都应在本项目内完成适配，而不是继续依赖旧项目运行时。

## 2026-07-15/16 进展记录

本阶段目标从“尝试多种 MDPO/visibility/hybrid terrain 组合”收敛为：先复现师兄 G1 成功链路，再做 ablation。当前主线是：

```text
PPO + SRU maze + no visibility map + sparse reward + frozen G1 Instinct low-level
4096 env + 16 steps/env + minibatch 4 + 15000 iterations
rnn_hidden_size=512, dropout=0.2, MLP=[512,256,128]
```

推荐主线命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name sru_maze_ppo_sparse_novis_sru_obs_aligned_v1 \
  --headless \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav
```

### 已完成的关键对齐

- reset/event 已按 `docs/attention.md` 收紧，并通过 dev smoke 的 `params/env.yaml` 验证：yaw `[-0.1,0.1]`，root velocity `[-0.2,0.2]`，`reset_joints_by_offset` 且 joint offset 为 0，`push_robot=None`，`base_external_force_torque=None`，camera/action scale/delay/low-pass 随机化关闭。
- sparse reward 已去掉此前的导航稠密项：`goal_progress`、`velocity_toward_goal`、`obstacle_proximity`、`terrain_drop_proximity`、`stuck_time` 不再属于 sparse baseline reward。
- `reach_goal_xyz` 已对齐 SRU 风格的稀疏/延迟触发：接近 timeout、小概率随机检查、或 `time_at_goal > 0` 后才开放 shaped goal reward。
- goal 终止已从“进半径立即 done”改为 SRU 风格驻留终止：到达目标后累计 `required_time_at_goal_s=4.0`，超过驻留步数才成功终止。
- success metric 已修正为 rolling episode success，不再把 `goal_reached_steps/count/success` 混成同一个瞬时量。
- high-level observation 已新增 SRU-aligned 版本：actor 使用 `base_lin_vel_delayed`、`base_ang_vel_delayed`、`projected_gravity_delayed`、`last_action`、`goal_command`、`encoded_depth_image`；critic 使用非 delayed proprioception、`last_action`、`goal_command`、`time_normalized`、depth 和 height scan。
- frozen low-level observation 保持 InstinctLab parkour policy 所需的 joint/action/depth history，不混入 high-level actor 语义。
- action bridge 的有效动作范围已明确：实际映射只由 `command_min/command_max + tanh + low_pass_filter_alpha` 决定；`policy_scaling/policy_bias` 已标为 legacy unused，避免误读实验记录。

### 当前 reward 状态

当前主线 sparse reward 激活项是：

```text
termination_penalty = is_terminated * -50
flat_orientation_l2
lin_vel_z_l2
action_rate_l2
action_l2
reach_goal_xy_soft = 0.25
reach_goal_xy_tight = 1.5
```

注意：`docs/new_attention.md` 中记录的 SRU 风格 reward 组还包括：

```text
action_rate_l1
lateral_movement
rot_movement
joint_acc_l2_joint
```

这些尚未并入当前默认主线。后续若要验证 reward 对齐，应单独开 ablation，不要覆盖 `PPO-SparseReward-v0` 主线，否则无法区分 observation、action 和 reward 的影响。

建议实验顺序：

```text
A. LegacyG1FlatObs control + current reward
B. SRU-aligned obs + current reward
C. SRU-aligned obs + new_attention reward ablation
```

### 任务分组

当前应优先使用：

```text
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Dev-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Play-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-PlayLarge-v0
```

为复现 2026-07-15 正在跑的旧 obs 对照组，新增 legacy task：

```text
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-LegacyG1FlatObs-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-LegacyG1FlatObs-Dev-v0
Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-LegacyG1FlatObs-Play-v0
```

legacy task 的差异是 high-level actor/critic 仍继承 G1Flat proprioception，包括 joint/action 相关输入。它只用于对照，不作为推荐路线。

MDPO、Early/Late/QueryFusion、ParkourTerrain、SparseWall、NoWall、ParkourSruStairsSparseWall 等任务目前都视为 ablation 或历史探索，不作为当前默认训练入口。

### 2026-07-15 legacy control 训练观察

当前 legacy control run：

```text
sru_maze_ppo_sparse_novis_attention_reset_sparse_strict_v1
```

在 iteration 2720-2730 附近，日志显示：

```text
success_rate ~= 0.70-0.72
early_termination ~= 0.71
time_out ~= 0.06
base_contact ~= 0.19-0.20
large_tilt_angle ~= 0.11-0.12
value_loss ~= 550-625
```

结论：旧 obs 对照组虽然 value loss 高，但 success 持续上升，timeout 显著下降，base_contact/large_tilt 相比早期回落，已经具备对照价值。value loss 高主要来自 sparse reward、终止分布和 return 方差，不应单独作为停止训练的理由。后续需要用 SRU-aligned obs 主线对比最终成功率、稳定性和 play 泛化表现。

### 2026-07-16 补充结论

昨天的主要判断是：当前不应再把问题归因到 PPO 超参数或目标点采样。目标采样、PPO runner、reset/event、success 统计和 low-level history 更新都已经做过对齐或修正；后续最有价值的对照是 high-level observation、action bridge 和 reward family。

训练日志中出现“success_rate 上升但 Mean reward 不一定同步上升”是合理的。当前 reward 同时包含稀疏到达奖励、termination penalty 和动作/姿态稳定项；一个 batch 里如果成功更多但摔倒、接触、动作变化或 episode 长度分布也变化，平均奖励可能低于另一个 success 略低的 batch。因此判断训练进展时优先看：

```text
Metrics/robot_goal/success_rate
Episode_Termination/time_out
Episode_Termination/base_contact
Episode_Termination/large_tilt_angle
play/eval 实际轨迹
```

`value_loss` 目前只能作为 critic 拟合难度参考，不能单独代表策略变差。sparse reward、成功/失败 return 差异和终止分布变化都会显著抬高 value target 方差。

后续实验原则：

1. 保留 legacy obs run 作为对照，不要覆盖它。
2. 新开 SRU-aligned obs baseline，用同一套 reward 和 action bridge 对比 observation 影响。
3. 如果 SRU-aligned obs 仍然 base_contact/large_tilt 高，再单独做 action bridge ablation。
4. reward 对齐 `new_attention.md` 中的 `action_rate_l1/lateral_movement/rot_movement/joint_acc_l2_joint` 应作为最后一组独立 ablation。
5. 不要同时修改 observation、action、reward、termination，否则无法定位是哪一项带来收益或退化。

## 2026-07-17 对齐记录

今天的目标是继续对齐师兄 `sru_ref` 中的 G1 SRU 成功链路，重点排查“版本之外的代码实现差异”。结论是：相机配置大体已经接近，最值得先改的是 frozen low-level backend 和 env/sim 关键运行项。

### 已修改

- 新增师兄同款 raw-action frozen backend：`instinct_nav/tasks/navigation/mdp/instinct_frozen_backend.py`。
  - 按师兄 `InstinctFrozenPolicyBackend` 语义构建 `EncoderMoEActorCritic`。
  - 固定 low-level obs format：`base_ang_vel / projected_gravity / velocity_commands / joint_pos / joint_vel / actions / depth_image`。
  - 输出 29D raw action，而不是 absolute joint target；后续仍交给 `JointPositionActionCfg(scale=beyondmimic_action_scale, use_default_offset=True)`。
  - 默认 `instinct_rl_root=/workspace/sru_ref/external/instinct_rl`，用于 frozen low-level policy 构建。
- 替换 frozen action bridge：`instinct_nav/tasks/navigation/mdp/frozen_low_level_actions.py`。
  - 不再走本项目 `load_parkour_low_level_policy()` 包装。
  - 改为 `InstinctFrozenPolicyBackend.compute_raw_actions()`。
  - 保持高层 action 处理逻辑：`tanh(raw) -> policy_scaling -> low-pass filter -> ExternalVelocityCommand`。
- 补齐 low-level action history 观测：`instinct_nav/tasks/navigation/mdp/low_level_observations.py`。
  - 增加 `last_low_level_action(...)` 兼容师兄命名。
  - `InstinctFrozenLowLevelObsCfg.actions` 改为使用 `nav_mdp.last_low_level_action`，参数为 `action_term="velocity_command"`。
- 补齐 SRU `NavigationEnvCfg` 中关键 sim/env 项：`instinct_nav/tasks/navigation/config/g1/frozen_env_cfg.py`。
  - `is_finite_horizon = True`
  - `sim.disable_contact_processing = True`
  - `sim.physics_material = scene.terrain.physics_material`
  - 保持 `decimation=40`、`episode_length_s=60.0`、`sim.dt=0.005`、sensor update period 和 terrain difficulty/curriculum 对齐。
- PPO/runner 小项保持源码对齐：`seed=72`、`reward_shifting_value=0.05`、`empirical_normalization=False`、`normalizers={}`、`value_clip_param=0.2`。

### 验证结果

已通过静态编译：

```bash
python3 -m py_compile \
  /workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/instinct_frozen_backend.py \
  /workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/frozen_low_level_actions.py \
  /workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/frozen_low_level_actions_cfg.py \
  /workspace/instinct_nav/instinct_nav/tasks/navigation/mdp/low_level_observations.py \
  /workspace/instinct_nav/instinct_nav/tasks/navigation/config/g1/frozen_env_cfg.py
```

已通过 4 env / 1 iter smoke test：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-Frozen-SruMaze-PPO-SparseReward-Dev-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --run_name smoke_backend_align \
  --headless
```

smoke 中确认：

- `robot_goal: RobotNavigationGoalCommand`
- `base_velocity: ExternalVelocityCommand`
- low-level policy obs format 为 `base_ang_vel/projected_gravity/velocity_commands/joint_pos/joint_vel/actions/depth_image`
- high-level actor obs shape 为 `(2576,)`
- critic obs 包含 `depth_image=(2560,)` 和 `height_scan_critic=(3136,)`
- low-level backend 成功构建并完成 1 iteration，无 shape/backend 报错。

### 当前训练观察

当前推荐 run：

```text
sru_maze_ppo_sparse_aligned_lowlevel_backend_v1
```

早期日志需要这样读：

- iteration 0 没有 `Mean reward` 是正常的，因为还没有 episode 结束。
- `Episode_Termination/*` 不是“本轮终止原因占比”，不能相加成 1。
  - IsaacLab 这里记录的是 `TerminationManager._last_episode_dones` 全 env buffer 均值。
  - `time_out=True` 表示 truncated/time-out 类别；成功到达目标的 `early_termination` 当前也是 `time_out=True` term。
- `Metrics/robot_goal/success` 来自 goal command 的 rolling success tracker，不能和 `Episode_Termination/time_out` 直接互斥比较。
- iteration 20 观察到：`success≈0.26`、`reach_goal_xy_tight≈0.11`、`base_contact≈0.03`、`large_tilt≈0.02`，说明新 low-level backend 训练链路已经能产生成功 episode；但 `velocity_toward_goal` 仍为负，方向策略还没稳定。

### 暂未修改但已对比

- 高层 `ActorCriticSRU`：当前文件不是师兄 patched `rsl_rl` 原文件，但在主线配置 `visibility off + single camera` 下核心拓扑等价：`proprio+depth attention -> SRU -> TC-Dropout -> MLP`，critic 为 `height+depth+proprio -> SRU -> time embedding -> TC-Dropout -> MLP`。当前实现是为了适配 `instinct_rl` runner 的 `obs_format` 和 hidden-state 接口，暂不建议直接替换。
- Env 继承链：师兄是 `G1InstinctNavigationEnvCfg(NavigationEnvCfg)`，当前仍是 `InstinctNavG1FrozenEnvCfg(G1FlatEnvCfg)` 后手动补齐。关键 sim/sensor/terrain 项已经补齐；直接改继承链会牵扯 G1 scene、InstinctLab camera/action 和 low-level obs，暂不建议硬切。
- 版本栈仍未完全一致：当前 IsaacLab 是 `v2.3.2`，师兄参考是 `v2.2.1-dirty`；high-level runner/PPO 仍跑当前 `/workspace/instinct_rl`，frozen low-level backend 构建时默认使用 `/workspace/sru_ref/external/instinct_rl`。


### 2026-07-17 追加：主线任务去 G1Flat Env 继承

- 将 `InstinctNavG1FrozenEnvCfg` 从 `G1FlatEnvCfg` 改为继承 `InstinctLabRLEnvCfg`，避免继续隐式继承 G1Flat 的 env 默认终止、decimation、episode 设置。
- 在 `frozen_env_cfg.py` 内新增本项目自己的 `InstinctNavFrozenTerminationsCfg` 和 `InstinctNavFrozenCurriculumCfg`，显式声明 `time_out/base_contact/early_termination/terrain_fall/large_tilt_angle`。
- 将主线 `scene/events/rewards` 从 G1Flat 子类拆出为本地类：保留 G1 contact sensor、sky light、startup friction randomization、reset_base、reset_joints_by_offset 这些与师兄 G1 配置一致的必要项。
- 目前仅 legacy G1FlatObs 对照任务还保留 `G1FlatPolicyObsCfg/G1FlatCriticObsCfg` 作为观测基类；主线 SparseReward 任务不再通过 G1FlatEnvCfg 获取隐藏默认项。
- 已通过 `py_compile` 静态检查。完整 Isaac smoke 需要重启/另启仿真进程；当前训练进程仍在运行时，新代码不会影响该进程。

### 2026-07-17 追加：SRU actor/critic 输入顺序修正

- 训练 80 iter 仍表现为低成功率、高 time_out、低 `velocity_toward_goal`，主要不是摔倒问题，而是高层导航学习没有形成有效推进。
- 修正 `ActorCriticSRU._process_actor_stream()` 的 recurrent 输入顺序：从原来的 `proprio + visual_features` 改为师兄 SRU 的 `visual_features + proprio`。这个顺序会直接影响 LSTM_SRU 的输入语义。
- 修正 critic 观测顺序为师兄配置：`base/proprio + time_normalized + height_scan_critic + depth_image`，并将 critic group 设置为 `concatenate_terms=True`，贴近 patched rsl_rl 的 flat tensor 假设。
- 已通过 `py_compile` 静态检查。该改动必须重启训练才生效。

### 2026-07-17 追加：视觉初始化与 success 指标对齐

- 增加本地 `initialize_depth_noise_generator(robot_name="g1", use_jit_precompiled=False)` 入口，并在 G1 frozen env `__post_init__()` 中调用，贴近师兄 G1 配置的 depth encoder 初始化流程。
- 修正 command reset 日志语义：`Metrics/robot_goal/success` 记录当前 reset episode 是否成功，`success_rate` 保持 rolling buffer，避免两者都显示 success rate。
- 已通过 `py_compile` 静态检查。该改动必须重启训练才生效。

### 2026-07-17 追加：SRU strict flat tensor 前向

- 主线无 visibility 的 `ActorCriticSRU` 前向改为师兄 patched rsl_rl 的 flat tensor split 语义，不再通过 named obs 拆分主线 actor/critic 输入。
- Actor 按尾部 `depth_image` 切分：`proprio + depth_tail -> CrossAttention(depth, proprio) -> [depth_feature, proprio] -> LSTM_SRU`。
- Critic 按尾部 `time + height + depth` 切分：`proprio + time + height_scan_critic + depth_image -> [height_feature, depth_feature, proprio] -> LSTM_SRU -> time embedding`。
- `reset()` 不再隐式重置 TC-dropout mask，只重置 actor/critic SRU hidden state，贴近师兄 SRU module。
- 已用纯 PyTorch dummy 输入验证真实主线维度：actor `(B, 2576) -> (B, 3)`，critic `(B, 5713) -> (B, 1)`；并通过 `py_compile`。

### 2026-07-17 追加：失败后全链路 strict 对齐

- 训练 420 iter 后仍只有 `success≈0.12`、`goal_progress<0`、`velocity_toward_goal≈0.03`，判定当前主线没有复现师兄早期学习趋势，开始一次性大改对齐。
- 重写 frozen low-level action bridge：直接向 `base_velocity` command term 写入 senior 语义的 `vel_command_b`，移除本地 `set_manual_command/clear_manual_command` 和逐 env/action 维度兼容逻辑，低通滤波按单一 `low_pass_filter_alpha` 执行。
- 将 navigation goal command/reset/metrics 改为师兄语义：补 `metrics/in_goal` observation group；reset 时从 `observation_manager.compute_group("metrics")` 读取成功信号；success rolling buffer 由 termination 更新，不再使用本地 `episode_has_reached_goal/episode_result_recorded` 状态机。
- 将 termination 链路改为 SRU/G1 语义：`time_out/at_goal/base_contact/terrain_fall` 写入 `goal_reached_buffer`；主线大姿态终止改为师兄 G1 文档中的 `isaac_mdp.bad_orientation(limit_angle=1.0)`。
- 视觉链路对齐非 JIT G1 depth encoder：`DepthNoiseEncoder` 支持 `CameraConfig/get_camera_config("g1")`，返回 `(encoded_depth, noisy_depth)`；high-level depth observation 继续输出 `(64,5,8)` flatten，并走 delayed encoded depth。
- Scene 配置补齐师兄 `height_scanner`，并移除 high-level `raycast_camera` 上本地额外的 `ray_alignment="base"/depth_clipping_behavior="none"` 字段；`require_terrain_sampling` 默认改为 True，terrain mask 缺失时不再静默 fallback。
- 已通过 `py_compile`；纯 PyTorch depth encoder 检查输出 `(2, 64, 5, 8)` 和 noisy depth `(2, 1, 40, 64)`。裸 Python 配置导入因环境未初始化 `pxr` 失败。
- 已通过 4 env / 1 iter launcher smoke：`metrics` group、policy `(2576,)`、critic `(5713,)`、low-level obs、SRU actor/critic 初始化和第一轮 PPO collection 均正常。

### 2026-07-17 追加：高层 SRU action std 改为 log_std

- 训练早期观察到 `Mean action noise std` 在 20 iteration 降到约 `0.38`，明显快于师兄 G1 日志中从 `1.0` 缓慢下降到 `0.9/0.8` 的趋势，说明高层策略探索过早塌缩。
- 定位到当前 `instinct_rl` 基类直接优化 `std` 参数，而师兄 patched `rsl_rl` 使用 `log_std` 参数。直接优化 std 会让 PPO 更容易快速压低动作噪声。
- 在 `ActorCriticSRU` 内局部覆盖 action distribution：删除基类直接 `std` 参数，新增 `log_std` 参数；`update_distribution()` 使用 `Normal(mean, exp(log_std))`；`action_std` 和 `clip_std()` 保持 runner/PPO 接口兼容。
- 不修改全局 `/workspace/instinct_rl`，影响范围只限高层 SRU policy。旧 checkpoint 中的 `std` 参数名与新 `log_std` 不兼容，建议此修改后从头启动训练。
- 已通过 `py_compile`、纯 PyTorch SRU 形状/参数检查和 4 env / 1 iter launcher smoke；smoke iteration 0 显示 `Mean action noise std: 1.00`。

### 2026-07-17 追加：新增师兄原链路 train-senior 入口

- 新增 `scripts/train_senior.py`，训练时直接导入 `/workspace/sru_ref/external/IsaacLab/source/isaaclab_nav_task` 的师兄任务注册，不再走本项目的 `Instinct-Nav-*` task registry。
- 新入口使用师兄 `Isaac-Nav-PPO-G1-Instinct-v0` / `Isaac-Nav-PPO-G1-Instinct-Dev-v0`，读取 `rsl_rl_cfg_entry_point`，runner 改回 `rsl_rl.runners.OnPolicyRunner`，不是 `instinct_rl.runners.OnPolicyRunner`。
- 显式接入师兄 patched `rsl_rl` 扩展目录：`/workspace/sru_ref/external/IsaacLab/rsl_rl/rsl_rl`，包括原 `ActorCriticSRU`、`sru_memory` 和 rsl_rl PPO 行为；同时保留师兄脚本里的 legacy dict obs adapter。
- 默认设置 `SRU_G1_INSTINCT_CHECKPOINT=/workspace/sru_ref/logs/instinct_rl_isaacsim510/20260709_223509/model_30000.pt`、`SRU_INSTINCT_RL_ROOT=/workspace/sru_ref/external/instinct_rl`，避免继续落到 `/workspace/sru/...` 的旧绝对路径。
- `_launcher.py` 新增 `train-senior` 命令；该入口用于 strict reproduction，不支持当前 `train` 入口的 wandb 参数，日志按师兄配置写入 `logs/rsl_rl/<experiment_name>/...`。
- 已通过 `py_compile` 和 launcher help 检查；真实 Isaac smoke 未运行，因为当前仍有旧训练进程在占用训练链路。

### 2026-07-17 追加：train-senior 支持 wandb

- `train-senior` 增加 `--wandb`、`--wandb_entity`、`--wandb_project` 参数。默认仍保持师兄配置的 `tensorboard`，只有显式传 `--wandb` 时才覆盖为 `agent_cfg.logger = "wandb"`。
- 师兄 patched `rsl_rl` 的 `WandbSummaryWriter` 从环境变量 `WANDB_USERNAME` 读取 entity，因此 `--wandb_entity` 会同步写入 `WANDB_USERNAME`。
- `--wandb_project` 会覆盖 `agent_cfg.wandb_project`；不传时沿用师兄 G1 配置里的 `isaaclab_nav_g1_instinct`。
- 已通过 `py_compile` 和 `scripts/train_senior.py --help` 检查。

### 2026-07-18 追加：train-senior 修正 IsaacLab 源码导入路径

- 发现 `train-senior` 虽然已经使用师兄 `rsl_rl`，但 `isaaclab/isaaclab_tasks/isaaclab_rl` 仍可能从 `/workspace/IsaacLab-2.3.2/source/...` 导入，形成半师兄半本地链路。
- 修正 `scripts/train_senior.py` 的 `sys.path`：逐个优先插入师兄 `source/isaaclab`、`source/isaaclab_tasks`、`source/isaaclab_rl`、`source/isaaclab_assets`、`source/isaaclab_nav_task`，而不是只插入 `/source` 根目录。
- 保留师兄 `rsl_rl` 优先路径 `/workspace/sru_ref/external/IsaacLab/rsl_rl`，并继续只在 fallback 到非师兄 rsl_rl 时注册 dict-observation adapter。
- 已通过 `py_compile`；非仿真导入检查确认模块解析进入师兄源码路径。当前工具环境无 CUDA，无法完成 Isaac smoke。

### 2026-07-18 追加：train-senior 启动硬检查和诊断落盘

- `scripts/train_senior.py` 增加 strict import path check：`isaaclab`、`isaaclab_tasks`、`isaaclab_rl`、`isaaclab_nav_task`、`rsl_rl`、`ActorCriticSRU`、`PPO` 必须来自 `/workspace/sru_ref/external/IsaacLab/...`，否则训练启动阶段直接报错退出。
- 每次训练在 runner 创建前写入 `params/senior_diagnostics.json`，记录实际 import 文件、task、seed、runner/PPO/policy 配置、actor/critic obs shape、action dim、相机/terrain/action 配置、G1 low-level checkpoint sha256、depth encoder sha256。
- 启动日志会打印 diagnostics summary，用于快速确认是否仍存在半师兄半本地链路、obs 维度不一致、checkpoint/depth 权重路径错误。
- 已通过 `py_compile` 和 `train_senior.py --help` 检查；当前工具环境无 CUDA，未跑 Isaac smoke。

### 2026-07-18 追加：train-senior AppLauncher 兼容边界修复

- 发现强制优先导入师兄 `source/isaaclab` 会让 `AppLauncher` 加载师兄旧 `isaaclab.python.headless-2.2.1` Kit app；该 app 依赖 cp310 的 `omni.physx`，在当前 IsaacSim 5.1 / Python 3.11 环境下无法解析扩展。
- 修正为显式兼容边界：IsaacLab core/AppLauncher/isaaclab_tasks/isaaclab_rl wrapper 使用当前 `/workspace/IsaacLab-2.3.2`，师兄 `isaaclab_nav_task` 和 patched `rsl_rl` 继续优先使用 `/workspace/sru_ref/external/IsaacLab/...`。
- strict diagnostics 的 import 校验同步更新：core 模块期望 current IsaacLab，navigation task 和 rsl_rl 期望 senior 路径，避免再次加载旧 Kit app。
- 已通过 `py_compile` 和 `train_senior.py --help`，help 阶段不再触发旧 `isaaclab.python.headless-2.2.1` 扩展依赖解析。

### 2026-07-18 追加：train-senior 禁用 rsl_rl GitPython 代码快照

- 师兄 `rsl_rl` 的 `OnPolicyRunner.learn()` 在首个 iteration 会调用 `store_code_state(self.log_dir, self.git_status_repos)`，默认包含 `rsl_rl.__file__`。
- 当前 `/workspace/sru_ref/external/IsaacLab/rsl_rl` 挂载仓库触发 GitPython `dubious ownership / safe.directory`，导致训练在第一轮日志阶段中断。
- `train_senior.py` 创建 runner 后设置 `runner.git_status_repos = []`，跳过 GitPython 快照；代码来源、import 路径、checkpoint/depth hash 已由 `params/senior_diagnostics.json` 记录。
- 已通过 `py_compile`。

### 2026-07-18 追加：本轮效果改善的实际原因复盘

- 本轮 `senior_fullcopy_g1_instinct_v4` 到 348 iteration 时已达到 `success_rate≈0.82`、`time_out≈0.06`、`base_contact≈0.09`、`reach_goal_xy_tight≈0.63`，明显好于 2026-07-17 晚上最后一轮本地主线 strict SRU run。
- 2026-07-17 晚上 `run-20260717_213535` 在 100 iteration 附近为：`Mean reward≈-3.99`、`reach_goal_xy_tight≈0.115`、`success_rate≈0.144`、`time_out≈0.635`、`action_noise_std≈0.39`。当前 `run-20260718_091329` 在 100 iteration 附近为：`Mean reward≈12.73`、`reach_goal_xy_tight≈0.362`、`success_rate≈0.237`、`time_out≈0.263`、`action_noise_std≈0.57`；到 120 iteration 已到 `success_rate≈0.370`、`reach_goal_xy_tight≈0.462`。
- 改善的核心不是单独调 reward，而是把训练主链路从本项目 `Instinct-Nav-*` 任务切到师兄的 `Isaac-Nav-PPO-G1-Instinct-v0` 任务和 patched `rsl_rl` runner。也就是说，env cfg、reward manager、command term、termination term、observation groups、low-level action bridge、SRU ActorCritic 和 PPO rollout/update 行为都由师兄代码路径主导。
- 本轮高层 policy 使用的是师兄 patched `rsl_rl` 的 `ActorCriticSRU`，不再使用本项目重写的 `instinct_nav.models.sru.actor_critic_sru.ActorCriticSRU`。这避免了 flat tensor split、depth tail 切分、critic time/height/depth 切分、LSTM_SRU hidden state、dropout mask 和 action distribution 细节与师兄实现不一致。
- 本轮 runner 使用师兄 patched `rsl_rl.runners.OnPolicyRunner` 和 `PPO`，`reward_shifting_value=0.05`、MDPO/SRU runner 里的 reward logging、rollout storage、action std/log_std 更新路径都来自师兄实现。之前本地主线虽然手动补过 `log_std`，但仍经过本地 `instinct_rl` runner/API 适配层，探索噪声在早期更容易塌到 `0.39`；当前 100 iteration 仍有 `action_noise_std≈0.57`，探索保留更接近师兄曲线。
- 本轮 low-level checkpoint、`instinct_rl_root`、depth encoder 权重、`isaaclab_nav_task` 任务包都从 `/workspace/sru_ref` 显式指定，并在 `senior_diagnostics.json` 中记录 sha256，避免继续使用本项目或旧 `/workspace/sru` 绝对路径下的权重/代码。当前 checkpoint sha256 为 `caf6e5bec6eab8621573b1b5a11c03189934559710e2eb7cd6eff8ea48025e4b`，depth encoder sha256 为 `65bc47e9173b5256385bc2c24280764f54e404dc1017f3e927fda85d21e29e4a`。
- 当前仍不是 100% 原环境复现：`isaaclab.app`、`isaaclab.envs`、`isaaclab_tasks`、`isaaclab_rl` 仍来自 `/workspace/IsaacLab-2.3.2`，而师兄文档使用 `/workspace/sru/external/IsaacLab-instinct-v232`；Python venv 也仍是 `/workspace/instinctlab51-env`，不是 `/workspace/sru/.venv-isaacsim510`。因此本轮变好主要证明“算法/任务/runner/权重链路”已经基本回到师兄路径，剩余差异集中在 IsaacLab core/runtime。
- 结论：这轮效果提升的关键原因是绕过本项目移植版训练链路，直接使用师兄任务包和 patched `rsl_rl` 作为主训练链路；本项目只保留启动、路径修正、wandb 和诊断外壳。当前 `success_rate≈0.82` 说明主要算法链路已经接近师兄成功配置，后续若还要追 0.9+，优先看 IsaacLab-instinct-v232/Python venv 运行环境是否能进一步一致。

### 2026-07-18 追加：为什么移植版会和师兄链路产生差异

- 差异的根本原因不是某一个 yaml 参数，而是本项目最初采用的是“移植/重写/适配”路线，不是直接运行师兄原工程。只要训练主链路不是同一套代码路径，ActorCritic、runner、env、action bridge、depth pipeline、runtime 任意一层的实现细节都可能改变 PPO 的 rollout 分布。
- 本地重写的 `ActorCriticSRU` 即使网络参数看起来一致，也可能在实现细节上不同：actor/critic flat tensor split 顺序、depth tail 切分、critic 的 `time + height + depth` 拼接、LSTM_SRU hidden state reset、dropout mask 生命周期、action distribution 使用 `std` 还是 `log_std`。这些细节不会体现在 yaml 里，但会直接改变动作分布和梯度更新。
- runner/API 适配层也会产生差异。师兄使用 patched `rsl_rl.runners.OnPolicyRunner` 和 `PPO`，而本地移植版曾经过 `instinct_rl` runner/API 或 adapter。runner 会影响 rollout storage、`reward_shifting_value` 是否真正生效、GAE/value target、action std 更新、`dones/time_out` 处理、`extras["observations"]["critic"]` 传递和日志统计。
- task/env 注册路径不同也会产生差异。本地 `Instinct-Nav-*` 任务是手动拼接 SRU navigation 逻辑和 InstinctLab G1 配置；师兄是 `Isaac-Nav-PPO-G1-Instinct-v0`。这意味着 command term、reward manager、termination term、observation groups、scene/sensor 初始化、reset 顺序都可能有细节差异。当前效果提升后，主链路已经改为直接加载师兄 `isaaclab_nav_task`。
- low-level action bridge 是高风险差异点。高层输出速度命令后，是否写入正确的 `base_velocity` command term、是否使用师兄语义的 `vel_command_b`、是否走 processed command、low-pass filter、action scaling、low-level obs group 名称，都会影响“同一个高层动作”在 G1 低层 policy 中实际执行出的运动。
- depth/camera pipeline 也是高风险差异点。即使最终输入 shape 都是 `(64, 5, 8)`，raw depth 范围、invalid depth 处理、noise 注入、delay buffer、encoder JIT/non-JIT、camera config、raycast update period 等差异都会改变 SRU 高层看到的视觉分布。视觉分布差一点，导航成功率可能差很多。
- 运行环境仍可能造成剩余差异。当前 `train-senior` 的算法链路已经主要来自师兄代码，但 `isaaclab.app`、`isaaclab.envs`、`isaaclab_tasks`、`isaaclab_rl` 仍来自 `/workspace/IsaacLab-2.3.2`，师兄文档使用 `/workspace/sru/external/IsaacLab-instinct-v232`；Python venv 也不同。这可能影响 reset、raycaster、contact/physics、manager env step、termination 统计和 sensor update，因此能解释当前 100 iteration 曲线仍慢于师兄，但 350 iteration 后已能追到 `success_rate≈0.82`。
- 结论：之前差异来自“看起来参数对齐，但实际执行路径没有完全对齐”。本轮有效改善说明，直接使用师兄 `isaaclab_nav_task + patched rsl_rl + ActorCriticSRU + checkpoint/depth encoder` 后，主要算法链路已经接近师兄；剩余要追求严格复现时，应优先处理 IsaacLab-instinct-v232/Python venv 运行环境差异，而不是继续随意改 reward。

### 2026-07-18 追加：train-senior 支持从 checkpoint 续训

- 停电中断后确认 `senior_fullcopy_g1_instinct_v4` 已保存 `model_500.pt`，路径为 `/workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_09-13-26_senior_fullcopy_g1_instinct_v4/model_500.pt`。
- `scripts/train_senior.py` 新增 `--resume_checkpoint` 参数，创建 runner 后调用师兄 patched `rsl_rl.runners.OnPolicyRunner.load(path, load_optimizer=True)`，恢复 actor-critic、optimizer 和 `current_learning_iteration`。
- `_launcher.py train-senior` 同步透传 `--resume_checkpoint` 和 `--no_load_optimizer`；默认加载 optimizer，只有显式传 `--no_load_optimizer` 时才只加载模型权重。
- 注意师兄 runner 的 `learn(num_learning_iterations=...)` 语义是“从当前 checkpoint iteration 再训练多少轮”，不是“训练到多少总轮”。例如从 `model_500.pt` 继续到约 3000 总轮，应传 `--max_iterations 2500`；如果传 `--max_iterations 3000`，最终会跑到约 3500 总轮。
- 已通过 `py_compile` 和 `scripts/train_senior.py --help` 检查。

### 2026-07-19 追加：senior_fullcopy_g1_instinct_v5_scratch 训练完成

- 训练 run：`/workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch`。
- 训练已完成 `3000` iterations，保存 checkpoint：`model_0.pt`、`model_500.pt`、`model_1000.pt`、`model_1500.pt`、`model_2000.pt`、`model_2500.pt`、`model_2999.pt`。
- 最重要 checkpoint：`/workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt`。
- 最终 `2995-2999` iteration 附近指标稳定在：`success_rate≈0.898-0.909`、`Mean reward≈39-43`、`reach_goal_xy_tight≈0.740-0.764`、`time_out≈0.038`、`base_contact≈0.061-0.064`、`large_pitch_angle≈0.057-0.059`、`terrain_fall≈0.0005-0.0007`、`Mean action noise std≈0.32`。
- 最后一轮 `2999/3000`：`success_rate=0.8979`、`Mean reward=40.34`、`reach_goal_xy_tight=0.7435`、`time_out=0.0375`、`base_contact=0.0610`、`large_pitch_angle=0.0579`。
- 本轮达到并稳定超过师兄要求的 `80%+` 成功率，接近 `90%`，可作为当前 `instinct_nav` 的 golden baseline。后续任何独立化、重构、替换本地任务/runner 的修改，都应和该 run 的训练曲线、checkpoint evaluation、diagnostics 对齐比较。
- 当前下一步不应继续随意改 reward/PPO，而应先做独立 evaluation/play 验证，再逐步把 `/workspace/sru_ref` 中的成功链路 正式吸收到 `instinct_nav/source`，保持该 run 作为回归对照。

### 2026-07-19 追加：接入师兄 play/eval 和失败 GIF 导出

- `_launcher.py` 新增 `play-senior` 与 `eval-senior` 命令，直接调用 `/workspace/sru_ref/external/IsaacLab/source/isaaclab_nav_task/scripts/play.py` 和 `scripts/eval.py`，并复用当前 `train-senior` 的 senior task / patched `rsl_rl` / senior InstinctLab 路径设置。
- `play-senior` 默认 task 为 `Isaac-Nav-PPO-G1-Instinct-Play-v0`，支持 `--checkpoint`、`--num_envs`、`--steps`、`--video`、`--bev_follow` 等常用参数，其余师兄 play.py 参数可继续通过 passthrough 传入。
- `eval-senior` 默认 task 为 `Isaac-Nav-PPO-G1-Instinct-Play-v0`，支持 `--checkpoint`、`--num_envs`、`--num_episodes`、`--json_out`、`--seed`、`--save_failure_gifs`。其中 `--save_failure_gifs` 会映射到师兄 eval.py 的 `--failure_rgb_dir`、`--failure_rgb_limit`、`--failure_rgb_export_mode gif`、`--failure_rgb_history_frames`，用于保存失败案例 GIF。
- 推荐最终 checkpoint eval 命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval-senior     --task Isaac-Nav-PPO-G1-Instinct-Play-v0     --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt     --num_envs 256     --num_episodes 1000     --headless     --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999.json     --save_failure_gifs     --max_failure_gifs 10     --failure_gif_folder /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/failure_gifs
```

- 推荐 play/可视化命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play-senior     --task Isaac-Nav-PPO-G1-Instinct-Play-v0     --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt     --num_envs 1     --steps 1000     --video     --video_length 600     --bev_follow
```

- 已通过 `_launcher.py` 的 `py_compile` 检查。实际 eval/play 需要启动 IsaacSim，尚未在本次记录中运行。

### 2026-07-19 追加：项目独立性迁移第一版完成

- 已将当前成功训练链路从 `/workspace/sru_ref` 迁入项目内 `source/` 和 `artifacts/`，`train-senior`、`eval-senior`、`play-senior` 默认不再读取师兄目录。
- 新增/更新的项目内依赖：
  - `source/isaaclab_nav_task`：师兄 `Isaac-Nav-PPO-G1-Instinct-v0` 任务包、navigation mdp、play/eval/train 脚本和 SRU depth/camera pipeline，复制时排除了 `.git`、`logs`、`wandb`、`__pycache__`。
  - `source/rsl_rl`：师兄 patched `rsl_rl`，包含 `ActorCriticSRU`、SRU PPO/runner 相关实现。
  - `source/instinct_rl`：低层 frozen G1 policy 构建依赖。
  - `source/instinctlab`：G1 parkour robot config、action scale、low-level observation functions 等依赖。
  - `artifacts/low_level/g1_parkour/model_30000.pt`：替换为成功训练真正使用的低层 checkpoint。
  - `artifacts/sru/depth_encoder/vae_pretrain_new.pth` 与 `artifacts/sru/heightscan_encoder/vae_heightscan3.pth`：补齐本地 SRU encoder 权重。
- 关键权重 hash：
  - G1 low-level checkpoint：`caf6e5bec6eab8621573b1b5a11c03189934559710e2eb7cd6eff8ea48025e4b`。
  - SRU depth encoder：`65bc47e9173b5256385bc2c24280764f54e404dc1017f3e927fda85d21e29e4a`。
  - SRU heightscan encoder：`3b07961f836e59bf03cb7031560d5bed330bb5957e5ae06bbb3ef303c3c5ca26`。
- `_launcher.py` 路径解析已改为项目优先：
  - `SENIOR_NAV_TASK_ROOT` 默认 `/workspace/instinct_nav/source/isaaclab_nav_task`。
  - `SENIOR_RSL_RL_ROOT` 默认 `/workspace/instinct_nav/source/rsl_rl`。
  - `SENIOR_INSTINCTLAB_SOURCE` 默认 `/workspace/instinct_nav/source/instinctlab`。
  - `SENIOR_INSTINCT_RL_ROOT` 默认 `/workspace/instinct_nav/source/instinct_rl`。
  - `SENIOR_G1_CHECKPOINT` 默认 `/workspace/instinct_nav/artifacts/low_level/g1_parkour/model_30000.pt`。
- `scripts/train_senior.py` 同步改为项目内 source 默认路径；保留 `INSTINCT_NAV_SENIOR_*`、`SRU_G1_INSTINCT_CHECKPOINT`、`SRU_INSTINCT_RL_ROOT`、`SRU_REF_ROOT` 作为显式 override，但不再内置 `/workspace/sru_ref` 默认值。
- 清理了运行源码中的旧绝对路径默认值：本地 `g1_parkour_cfg.py`、`instinct_frozen_backend.py`、`frozen_low_level_actions_cfg.py`、`models/sru/depth_encoder.py`、`models/sru/heightscan_encoder.py`、`scripts/test_low_level_bridge.py`，以及 source 内 `g1_instinct/g1_sonic` config、Instinct/Sonic action cfg/backend、showcase/eval/paper_eval 辅助脚本。
- 当前仍保留的外部运行前提：IsaacSim 5.1 兼容环境和 IsaacLab core 仍通过 `INSTINCT_NAV_ISAACLAB_ROOT` 默认 `/workspace/IsaacLab-2.3.2` 提供。原因是师兄旧 IsaacLab app 在当前 Python 3.11 / IsaacSim 5.1 环境下会触发 `isaaclab.python.headless-2.2.1` / `omni.physx` 扩展依赖问题。也就是说，算法任务链路已经项目内置，但仿真 runtime 仍是外部基础设施。
- 验证结果：
  - 源码扫描：`instinct_nav/`、`scripts/`、`source/` 的可执行源码中已无 `/workspace/sru_ref`、`/workspace/sru_new`、`/workspace/sru/`、`/workspace/hiking_artifacts` 默认路径。
  - `py_compile` 已通过：`_launcher.py`、`scripts/train_senior.py`、`scripts/test_low_level_bridge.py`、本地 SRU encoder/backend/cfg、source train/eval/play/showcase/paper_eval、source g1 config 和 action backend。
  - 轻量入口检查已通过：`/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/scripts/train_senior.py --help`。
  - launcher 默认路径检查显示全部解析到 `/workspace/instinct_nav/source` 和 `/workspace/instinct_nav/artifacts`。
- 迁移后的训练命令仍保持不变，只是现在默认读本项目内置链路：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train-senior \
    --task Isaac-Nav-PPO-G1-Instinct-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name senior_fullcopy_g1_instinct_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav
```

### 2026-07-19 追加：清理废弃本地移植代码

- 当前主线已经切到项目内 `source/isaaclab_nav_task + source/rsl_rl + source/instinctlab + source/instinct_rl`，因此删除早期本地移植版 `Instinct-Nav-*` 代码，避免后续误跑旧链路。
- `_launcher.py` 只保留 4 个命令：`doctor`、`train-senior`、`play-senior`、`eval-senior`。已移除废弃入口：`train`、`play`、`eval`、`test-low-level`。
- 删除的旧代码/脚本：
  - `scripts/train.py`、`scripts/play.py`、`scripts/evaluate.py`、`scripts/debug_frozen_task.py`、`scripts/debug_timing.py`、`scripts/test_low_level_bridge.py`。
  - `instinct_nav/algorithms`、`instinct_nav/assets`、`instinct_nav/low_level`、`instinct_nav/models`、`instinct_nav/tasks`、`instinct_nav/terrains`。
  - `instinct_nav/checks.py`、`instinct_nav/config.py`。
- 删除的运行产物/缓存：`.cache`、`outputs`、`__pycache__`、`debug_frozen_report.txt`、`debug_timing_report.txt`，以及 source 中已有的 `__pycache__`。
- 保留内容：
  - `scripts/train_senior.py` 和 `_launcher.py`，作为当前训练/评估/播放入口。
  - `instinct_nav/runtime.py`，用于统一 runtime env / Python / PYTHONPATH。
  - `source/`，作为当前正式源码树。
  - `artifacts/`，保存低层 G1 checkpoint 和 SRU encoder 权重。
  - `logs/`，保留 golden baseline 训练结果和 checkpoint，没有清理。
  - `docs/` 与历史 README 记录，用于追溯迁移过程。
- 验证结果：
  - `py_compile` 通过：`_launcher.py`、`instinct_nav/__init__.py`、`instinct_nav/runtime.py`、`scripts/train_senior.py`。
  - `_launcher.py doctor` 通过。
  - `_launcher.py --help` 现在只显示 `{doctor,eval-senior,play-senior,train-senior}`。
  - `scripts/train_senior.py --help` 通过。
  - 实际源码扫描无 `instinct_nav.tasks`、`instinct_nav.models`、`instinct_nav.low_level`、`Instinct-Nav-*`、旧 `scripts/train.py/play.py/evaluate.py` 引用。

### 2026-07-19 追加：当前独立主线常用命令

以下命令基于清理后的独立项目入口，只使用 `_launcher.py` 中保留的 `train-senior`、`eval-senior`、`play-senior`。

前台训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train-senior \
    --task Isaac-Nav-PPO-G1-Instinct-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name senior_fullcopy_g1_instinct_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav
```

后台训练命令：

```bash
mkdir -p /workspace/instinct_nav/logs/terminal
nohup /workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train-senior \
    --task Isaac-Nav-PPO-G1-Instinct-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name senior_fullcopy_g1_instinct_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    > /workspace/instinct_nav/logs/terminal/senior_fullcopy_g1_instinct_independent_v1.out 2>&1 &
```

后台训练日志查看：

```bash
tail -f /workspace/instinct_nav/logs/terminal/senior_fullcopy_g1_instinct_independent_v1.out
```

最终 checkpoint 评估并保存失败 GIF：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval-senior \
    --task Isaac-Nav-PPO-G1-Instinct-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 256 \
    --num_episodes 1000 \
    --headless \
    --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999.json \
    --save_failure_gifs \
    --max_failure_gifs 10 \
    --failure_rgb_history_frames 10 \
    --failure_rgb_gif_frame_ms 250 \
    --failure_gif_folder /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/failure_gifs
```

GIF 帧设置按师兄 `eval.py` 默认值显式写出：`--failure_rgb_history_frames 10` 表示每个失败 GIF 保存最近 10 个终止前/pre-impact control-step 帧，不是前后各 10 帧；`--failure_rgb_gif_frame_ms 250` 表示每帧 250 ms。若该值 `<=1`，师兄代码会退化为只保存最后一个 pre-impact 帧。

如果需要把成功 episode 也纳入 failure-rgb 导出通道用于对比，可额外加 `--failure_rgb_include_successes`；如果只想保存 base contact 失败，可额外加 `--failure_rgb_base_contact_only`。如果需要把 depth/height observation 也拼进 GIF，可额外加 `--failure_rgb_include_observations`。

使用和录制视频一致的俯视跟随视角 play 命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play-senior \
    --task Isaac-Nav-PPO-G1-Instinct-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 1 \
    --steps 1000 \
    --video \
    --video_length 600 \
    --bev_follow \
    --bev_height_m 22.0
```

只采样楼梯地形并连续保存多段 GIF：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play-senior \
    --task Isaac-Nav-PPO-G1-Instinct-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 1 \
    --steps 0 \
    --headless \
    --bev_follow \
    --bev_height_m 22.0 \
    --require_terrain_name stairs \
    --max_resample_attempts 500 \
    --gif_count 5 \
    --gif_length 300 \
    --gif_frame_ms 80 \
    --gif_dir /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_gifs
```

`--gif_count` 控制保存几个 GIF，输出文件形如 `stairs_000.gif`、`stairs_001.gif`。每段 GIF 在 env0 到达终止条件或达到 `--gif_length` 后保存，然后自动 reset 并重新采样楼梯地形；`--steps 0` 表示不额外限制总 play 步数。

只检查运行环境和当前入口命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py doctor
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py --help
```

### 2026-07-19 追加：将外部成功链路正式吸收到 source 源码树

- 为避免项目结构看起来像“搬运代码”，已将旧内置源码目录正式吸收到 `source/` 源码树。
- 当前正式源码结构：
  - `source/isaaclab_nav_task`：G1 SRU navigation task、mdp、train/eval/play 脚本、depth/camera pipeline。
  - `source/rsl_rl`：SRU patched runner/PPO/ActorCriticSRU。
  - `source/instinctlab`：G1 parkour robot config、action scale、low-level observation functions。
  - `source/instinct_rl`：低层 frozen policy 构建依赖。
- `_launcher.py` 和 `scripts/train_senior.py` 默认路径已改为 `source/*`。
- `pyproject.toml` 已将 `isaaclab_nav_task*`、`rsl_rl*`、`instinctlab*`、`instinct_rl*` 纳入包发现范围，使这些代码成为 `instinct_nav` 项目的正式源码组成部分，而不是外部依赖目录。
- 保留原包名 `isaaclab_nav_task`、`rsl_rl`、`instinctlab`、`instinct_rl`，原因是这些包名被 gym task registry、rsl_rl 内部 import、InstinctLab 配置和 checkpoint 构建逻辑直接依赖；现在先正式吸收到项目源码树，不做高风险大规模重命名。
- 验证结果：active code 中已无旧内置源码目录命名、旧根变量或旧绝对路径命中，`py_compile`、`_launcher.py --help`、`scripts/train_senior.py --help` 通过。

### 2026-07-20 追加：正式 Isaac 项目结构整理

- 将主任务包从旧命名 `source/isaaclab_nav_task/isaaclab_nav_task` 迁移为正式项目同名包：`source/instinct_nav/instinct_nav`。
- 保留轻量兼容包：`source/instinct_nav/isaaclab_nav_task`。旧任务 id `Isaac-Nav-PPO-G1-Instinct-v0`、`Isaac-Nav-PPO-G1-Instinct-Play-v0` 暂时继续可用，避免立刻破坏历史 checkpoint、README 命令和评估脚本。
- 将外部底层依赖从 `source/` 移出：
  - `external/instinctlab`：G1 parkour robot config、action scale、low-level observation functions。
  - `external/instinct_rl`：低层 frozen policy loader / network builder。
- 保留 `source/rsl_rl` 作为当前 SRU patched runner/PPO/ActorCriticSRU fork。当前训练直接依赖包名 `rsl_rl`，暂不重命名，避免影响 runner、checkpoint 和策略构建。
- 当前正式源码职责：
  - `source/instinct_nav/instinct_nav/navigation`：G1 SRU navigation task、MDP、terrain、camera/depth pipeline、任务注册。
  - `source/instinct_nav/scripts`：项目内 train/eval/play/showcase 脚本。
  - `source/rsl_rl`：SRU 算法实现。
  - `external/*`：外部接口与底层依赖。
  - `artifacts/*`：低层 G1 checkpoint 和 SRU encoder 权重。
- `_launcher.py` 不再通过普通 `import instinct_nav.runtime` 加载 runtime，改为按文件路径加载 `source/instinct_nav/instinct_nav/runtime.py`。原因是正式 `instinct_nav` 包导入时会注册 Isaac task 并 patch terrain，不能在 AppLauncher 启动前触发。
- `pyproject.toml` 包发现已改为：`source/instinct_nav`、`source/rsl_rl`、`external/instinctlab`、`external/instinct_rl`。
- `runtime.py` 默认优先使用项目内 `external/instinctlab`，并把 `source/instinct_nav`、`source/rsl_rl`、`external/instinctlab`、`external/instinct_rl` 加入 `PYTHONPATH`。
- 验证结果：
  - `_launcher.py doctor` 通过，显示 InstinctLab root 已解析到 `/workspace/instinct_nav/external/instinctlab`。
  - `_launcher.py --help` 通过。
  - `scripts/train_senior.py --help`、`source/instinct_nav/scripts/play.py --help`、`source/instinct_nav/scripts/eval.py --help` 通过。
  - `compileall` 已通过：`source/instinct_nav`、`source/rsl_rl`、`external/instinctlab`、`external/instinct_rl`、`scripts/train_senior.py`、`_launcher.py`。
- 注意：一次最小 Isaac headless task registry smoke test 在当前已有训练进程占用 GPU 时启动过慢，已手动停止该 smoke test；没有停止用户训练进程。后续建议在 GPU 空闲时再跑完整 task registry + env create smoke test。

### 2026-07-20 追加：迁移后真实训练入口 smoke test 通过

- 在暂停正式训练后，使用真实入口完成了小规模 smoke test：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train-senior     --task Isaac-Nav-PPO-G1-Instinct-v0     --num_envs 32     --max_iterations 1     --run_name migration_smoke_test     --headless
```

- smoke test 成功创建环境、加载低层 frozen policy、加载 depth encoder / heightscan encoder、初始化 runner，并保存 `model_0.pt`。
- 产物目录：`logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-20_10-40-06_migration_smoke_test`。
- `senior_diagnostics.json` 关键路径确认：
  - `instinct_nav`：`/workspace/instinct_nav/source/instinct_nav/instinct_nav/__init__.py`。
  - `instinct_nav.navigation.config.g1_instinct.navigation_env_cfg`：`/workspace/instinct_nav/source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py`。
  - `rsl_rl`：`/workspace/instinct_nav/source/rsl_rl/rsl_rl/__init__.py`。
  - `instinctlab`：`/workspace/instinct_nav/external/instinctlab/instinctlab/__init__.py`。
  - `instinct_rl_root`：`/workspace/instinct_nav/external/instinct_rl`。
  - low-level checkpoint：`/workspace/instinct_nav/artifacts/low_level/g1_parkour/model_30000.pt`，hash 仍为 `caf6e5bec6eab8621573b1b5a11c03189934559710e2eb7cd6eff8ea48025e4b`。
  - depth encoder：`/workspace/instinct_nav/source/instinct_nav/instinct_nav/navigation/assets/data/Policies/depth_encoder/vae_pretrain_new.pth`，hash 仍为 `65bc47e9173b5256385bc2c24280764f54e404dc1017f3e927fda85d21e29e4a`。
- 运行时 shape 也正确：`policy_obs_shape=(32, 2576)`，`critic_obs_shape=(32, 5713)`，`num_actions=3`。
- GPU 已释放，smoke test 没有残留训练进程。

### 2026-07-20 追加：独立项目命名收尾

- 新增正式任务 id：`Instinct-Nav-G1-SRU-v0`、`Instinct-Nav-G1-SRU-Dev-v0`、`Instinct-Nav-G1-SRU-Play-v0`。后续训练、评估、play 默认使用这些 `instinct_nav` 自有 id。
- 旧 `Isaac-Nav-PPO-G1-Instinct-*` task id 已从当前注册表删除，不再作为兼容 alias；历史日志中的旧命令仅作为记录保留。
- `_launcher.py` 新增正式短命令：`train`、`eval`、`play`。`train-senior`、`eval-senior`、`play-senior` 暂时保留为兼容入口，执行链路相同。
- 移除 `_launcher.py` 和 `scripts/train_senior.py` 中的 `/workspace/sru_ref` fallback 默认路径。当前默认只读取项目内：`source/instinct_nav`、`source/rsl_rl`、`external/instinctlab`、`external/instinct_rl`、`artifacts/low_level/g1_parkour/model_30000.pt`。
- `source/instinct_nav/setup.py` 改为 `find_packages(include=["instinct_nav*", "isaaclab_nav_task*"])`，避免正式安装时只安装顶层包而漏掉 `instinct_nav.navigation.*` 子包。
- 新 task id smoke test 已通过：`_launcher.py train --task Instinct-Nav-G1-SRU-v0 --num_envs 32 --max_iterations 1 --run_name task_id_smoke_test --headless` 成功保存 `model_0.pt`；`senior_diagnostics.json` 中 `task=Instinct-Nav-G1-SRU-v0`，核心 import 均解析到项目内路径。
- 2026-07-20 更新：任务注册文件中已删除旧 `Isaac-Nav-PPO-G1-Instinct-*` alias，只保留 `Instinct-Nav-G1-SRU-*` 三个正式 task id。

正式训练命令：

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

正式评估命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /path/to/model_2999.pt \
    --num_envs 256 \
    --num_episodes 1000 \
    --headless \
    --json_out /path/to/eval_summary.json \
    --save_failure_gifs \
    --max_failure_gifs 10
```

正式 play/GIF 命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /path/to/model_2999.pt \
    --num_envs 1 \
    --headless \
    --bev_follow \
    --gif_count 5 \
    --gif_length 500 \
    --gif_frame_ms 80 \
    --gif_dir /path/to/videos/play_gifs
```



### 2026-07-23 追加：G1 低层视觉与观测入口继续本地化

- 低层运控相机不再从 InstinctLab parkour task 的 `SceneCfg.camera` deep-copy；已在 `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py` 中显式定义 `NoisyGroupedRayCasterCameraCfg`。
- 本地显式保留低层 checkpoint 期望的相机参数：
  - parent link: `{ENV_REGEX_NS}/Robot/torso_link`
  - ray alignment: `yaw`
  - raw depth resolution: `64 x 36`
  - FOV: `89.51 deg x 58.29 deg`
  - update period: `0.02 s`
  - camera offset: `pos=(0.0487988662332928, 0.01, 0.4378029937970051)`，`rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0)`
  - noise pipeline: `crop_and_resize(crop_region=(18, 0, 16, 16))`、`gaussian_blur(kernel_size=3, sigma=1)`、`depth_normalization(depth_range=(0.0, 2.5), output_range=(0.0, 1.0))`
  - history buffer: `data_histories={"distance_to_image_plane_noised": 37}`
- 新增 `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/g1_asset_cfg.py`，在 instinct_nav 内部构造 `G1_WITH_SHOE_CFG`，避免从 `instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg` 导入整个 parkour task 配置。
- G1-with-shoe URDF 路径现在指向项目内 external checkout：`/workspace/instinct_nav/external/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf`。
- `InstinctLowLevelPolicyCfg` 的本体观测和 command/action cfg 已切到 `nav_mdp` 入口：`base_ang_vel`、`projected_gravity`、`generated_commands`、`joint_pos_rel`、`joint_vel_rel`、`JointPositionActionCfg`。
- 已将 InstinctLab 的低层图像历史读取器本地化为 `nav_mdp.delayed_visualizable_image`，用于从 `NoisyGroupedRayCasterCamera` 的 `distance_to_image_plane_noised_history` 中按延迟抽取 `8 x 18 x 32` depth history。
- 静态检查结果：`g1_instinct/navigation_env_cfg.py` 和本地 `observations.py` 不再直接引用 `instinctlab.tasks.parkour.mdp`、`InstinctParkourSceneCfg` 或 `G1_with_shoe_CFG`。
- 语法检查通过：

```bash
/workspace/instinctlab51-env/bin/python -m compileall -q \
    /workspace/instinct_nav/source/instinct_nav/instinct_nav/navigation/config/g1_instinct/g1_asset_cfg.py \
    /workspace/instinct_nav/source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py \
    /workspace/instinct_nav/source/instinct_nav/instinct_nav/navigation/mdp/observations.py
```
