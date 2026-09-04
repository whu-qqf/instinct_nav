# instinct_nav 开发日志 (LOG)

本文件按时间顺序记录 `instinct_nav` 的训练 / 对齐 / 部署进展，内容从原 README 迁出并保持原样。

> 注意：早期条目会引用已被取代的配置（如 scene-token v1–v3、`40 x 40 m` 地形、checkpoint 版低层、旧速度契约 `vy∈[-0.6,0.6]`）。**当前主线以 [README.md](./README.md) 为准**：DynamicMapCross + RichTerrain v4 + 官方 Parkour ONNX 低层。评估结果基线见本文件末尾「评估结果基线快照」章节。

---

## 2026-08/09 动态障碍课程与分割 mask 工作

**动态障碍避障课程（AvoidReact / Demote）**：在 `-Centerline-Unobserved-` 基础上叠加重平衡避障奖励（proximity −2.0 / approach −1.0 / 前瞻 TTC −0.5）+ 能力驱动分级课程（`dynamic_obstacles.py`）。

- **`-AvoidReact-`**（P2 可见接近生成 `reactive_spawn=5m` + P3 碰撞写失败 `record_failure`）：障碍从 ≥5m 处 pop-in 接近、碰撞计入 success 失败。注册 `-AvoidReact-{Banks}` 系列。
- **`-Demote-` 双向课程**：`demote_success=0.73 / advance_success=0.80` 死带。**结果：负**——满难度 eval 与 monotonic 基线打平（碰撞 0.073 vs 0.088、success 0.735 vs 0.745），双向降档未带来实质提升；唯一作用是让模型在扛不住满难度时退回 d≈0.6。已否决，不再投入。
- 关键陷阱：`--force_dynamic_obstacles` 只 `enable()` 不设难度，且课程的 `_maybe_advance_difficulty` 会覆盖 `--force_obstacle_difficulty`（demote 模型会被降档）；要跑真·静态需 `--disable_dynamic_obstacles`（`eval.py` 新增，把 `auto_switch_success=2.0` + `max_switch_iteration=10^9`）。**并行跑 eval 会撞 carb mutex 卡死，eval 必须串行。**

**动态障碍分割 mask（DynamicMask Phase 1，网络观测，替代"不增加网络观测"的旧约束）**：诊断发现 visibility map 是"只增不减"的 seen/unseen 记忆（`observations.py` `torch.maximum`），先天表示不了移动障碍物 → 动态避障失明。因此新增轻量分割头：

- 观测：`raw_depth_delayed`（原始 36×64 延迟深度，供 mask 头）+ `dynamic_obstacle_mask_label`（训练期真值：障碍物包围盒投影到延迟相机帧 + 深度遮挡校验；用 manager 的 `positions_w`+`spawned` 状态历史按 visual lag 精确对齐）。
- 网络：`actor_critic_sru.py` 新增 `_DynamicMaskHead`（raw depth → 64D bottleneck + sigmoid mask），64D feature 注入 actor/critic SRU 输入。
- 训练：`rollout_storage` 存 mask label；`ppo.py` 加 `lambda_mask` BCE+Dice 辅助 loss；`on_policy_runner` 记录 `Loss/mask`、`Loss/mask_dice_loss`、`Loss/mask_positive_rate`。
- 新 task `-AvoidReact-DynamicMask-Banks-{v0,Dev,Play}`（`lambda_mask=0.2`），默认关、旧 task 逐字节不变。
- 注意：原始 depth 是 **36×64**（共享低层相机 `height=36`），不是 40×64；`mask_input_dims=(1,36,64)`。旧 checkpoint `strict=True` 加载会因新增 mask 头 mismatch → DynamicMask 需从头训。
- `train_senior.py`/`train.py` 的 `_to_legacy_obs` 已改为透传 `mask_label` 观测组（此前被丢弃导致 mask loss 恒 0）。

---

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


### 2026-07-23 追加：Scene-token query fusion 原型

- 新增第一版 `scene_tokens` observation，用现有 local visibility memory 生成固定场景 token，作为后续接入 PQ3D token 的接口原型。
- 当前 token 不是 PQ3D 输出，而是从 `64 x 64` visibility map patchify 得到：
  - 逻辑 shape：`[num_envs, 64, 6]`
  - observation manager 中 flatten 后 shape：`[num_envs, 384]`
  - 每个 token 特征：`[x_center, y_center, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal]`
- 新增 `nav_mdp.scene_tokens_from_visibility_map`，内部复用 `local_visibility_map` 的 noisy/delayed depth 更新逻辑，然后把 BEV map 聚合成 `8 x 8 = 64` 个 patch tokens。
- `ActorCriticSRU` 新增 actor scene-token query attention 分支：
  - baseline `proprio + depth` 仍先进入 SRU memory。
  - SRU recurrent feature + proprio feature 作为 query。
  - `scene_tokens` 经 projector 后作为 key/value。
  - attention 输出 concat 到 actor MLP 前。
- 新增 runner/task：
  - `Instinct-Nav-G1-SRU-SceneToken-Query-v0`
  - `Instinct-Nav-G1-SRU-SceneToken-Query-Dev-v0`
  - `Instinct-Nav-G1-SRU-SceneToken-Query-Play-v0`
- smoke test 通过：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-SceneToken-Query-Dev-v0 \
    --num_envs 32 \
    --max_iterations 1 \
    --run_name scene_token_query_smoke_test_v2 \
    --headless
```

- smoke test 观察到 policy obs shape 为 `(2960,)`：原 `(2576,)` + `scene_tokens (384,)`。低层运控 observation 仍为 `depth_image (8, 18, 32)`，未改低层输入。
- 保存产物：`logs/rsl_rl/g1_instinct_navigation_ppo_scene_token_dev/2026-07-23_20-39-46_scene_token_query_smoke_test_v2/model_0.pt`。


### 2026-07-24 追加：Scene-token v2 actor/critic risk-token ablation

- 新增 `nav_mdp.scene_tokens_risk_from_visibility_map`，保留 v1 的 6D token，不覆盖旧任务。
- v2 token 从 `64 x 64` local visibility memory 和同一路 actor raycast depth 派生：
  - 逻辑 shape：`[num_envs, 64, 9]`
  - observation manager flatten 后 shape：`[num_envs, 576]`
  - 每个 token 特征：`[x_center, y_center, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal, depth_mean, depth_edge, frontier]`
- `depth_mean/depth_edge/frontier` 都来自 actor 可用的 raycast camera / visibility memory，没有使用 critic-only `height_scanner_critic`，避免把 privileged height scan 泄漏给 actor。
- v2 环境同时把同一份 scene token 加入 policy 和 critic observation：
  - actor: `proprio + depth + scene_tokens`
  - critic: `proprio + time + height + depth + scene_tokens`
- token 函数内部加入 step-level cache，避免同一个仿真步里 policy/critic 分别读取 observation 时重复 update visibility map。
- `ActorCriticSRU` 新增 critic scene-token query attention 分支：critic SRU recurrent feature + critic proprio 作为 query，scene tokens 作为 key/value，attention 输出 concat 到 critic MLP 前。
- 新增 runner/task：
  - `Instinct-Nav-G1-SRU-SceneTokenRisk-Query-v0`
  - `Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Dev-v0`
  - `Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Play-v0`
- 本机当前没有可见 CUDA，Isaac dev 初始化被 `No CUDA GPUs are available` 阻止；已完成 Python 语法检查和 CPU tensor forward 检查。
- CPU forward 检查结果：actor feature shape `[..., 576]`，critic feature shape `[..., 584]`，`act_inference` 输出 `[..., 3]`，`evaluate` 输出 `[..., 1]`。

推荐训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-SceneTokenRisk-Query-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name g1_sru_scene_token_risk_query_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    --seed 72
```


### 2026-07-26 追加：Scene-token v3 no-goal 评估结果

- 新增并完成 `SceneTokenRiskNoGoal` ablation，token 去掉 per-token goal-relative 特征，只保留局部场景/风险：
  - token shape：`[num_envs, 64, 6]`
  - token features：`[x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier]`
  - actor/critic 都接收同一份 scene token。
- 训练 run：`logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-25_17-29-52_g1_sru_scene_token_risk_no_goal_query_v1/`
- checkpoint：`model_2999.pt`
- 1000 episode 普通评估：`eval_summary_model_2999_1000ep.json`
- 评估结果：
  - overall success：`0.916` (`916 / 1000`)
  - maze：`0.9183`
  - non_maze：`0.9167`
  - pits：`0.8900`
  - stairs：`0.9310`
  - base_contact：`0.048`
  - large_pitch_angle：`0.057`
- 对比结论：
  - v1 scene-token：`0.875`
  - v2 risk-token：`0.892`
  - v3 risk-no-goal：`0.916`
  - 去掉 `dist_to_goal/cos_to_goal/sin_to_goal` 后，楼梯和整体成功率明显提升，说明 scene token 更适合作为局部场景/风险描述，目标信息应由原始 goal observation 和 query 分支使用。
- fixed-case manifest 评估暂不作为结论依据：v1 fixed-case 评估出现异常早期成功率，但普通 v1 重评估正常，后续需要单独修复 `case_manifest_in` 后的 visibility/depth cache reset 与 sensor sync。

### 2026-07-26 追加：Scene-token v4 shared-lowcam 单相机任务

- 新增 shared-lowcam ablation：高层 SRU 复用 frozen Instinct 低层的同一个 `camera`，不再创建额外 `raycast_camera`。
- 新任务：
  - `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0`
  - `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Dev-v0`
  - `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Play-v0`
- 高层 observation 改动：
  - actor `depth_image`：从 `SceneEntityCfg("raycast_camera")` 改为 `SceneEntityCfg("camera")`。
  - critic `depth_image`：从 `SceneEntityCfg("raycast_camera")` 改为 `SceneEntityCfg("camera")`。
  - actor/critic `scene_tokens`：从同一个 `camera` 的 raw `distance_to_image_plane`、pose、intrinsics 构建。
- 低层 observation 不变：`instinct_low_level_policy.depth_image` 仍使用 `camera` 的 `distance_to_image_plane_noised_history`，即 8 帧、crop/resize 后的 `(8, 18, 32)` 历史输入。
- 关键验证：当前 SRU depth encoder 可直接接受底层相机 raw `64x36`，输出仍是 `64x5x8=2560`，所以 `ActorCriticSRU.image_input_dims=(64,5,8)` 和网络结构不需要改。
- smoke test 已通过：
  - command：`/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Dev-v0 --num_envs 32 --max_iterations 1 --run_name shared_lowcam_smoke_test_32env --headless`
  - policy obs：`2960 = proprio/goal/action 16 + depth 2560 + scene_tokens 384`
  - critic obs：`6097 = proprio/goal/action 16 + time 1 + height 3136 + depth 2560 + scene_tokens 384`
  - low-level policy depth：`(8, 18, 32)`
  - output：`logs/rsl_rl/g1_instinct_navigation_ppo_scene_token_risk_no_goal_shared_lowcam_dev/2026-07-26_17-28-53_shared_lowcam_smoke_test_32env/model_0.pt`

正式训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    --seed 72
```

## Onboard Deployment Skeleton

2026-07-28 新增轻量真机部署骨架：[`deploy/`](/workspace/instinct_nav/deploy)。

当前部署路线不是把 IsaacLab 训练容器直接连机器人，而是在本机/机器人随身工控机运行轻量推理包：

```text
real sensors -> SRU high-level nav policy @ 5 Hz -> velocity command -> instinct_onboard low-level policy @ 50 Hz -> G1 joints
```

关键文件：

- [`deploy/README.md`](/workspace/instinct_nav/deploy/README.md)：部署架构、instinct_onboard 复用方式和 checklist。
- [`deploy/instinct_nav_onboard/config.py`](/workspace/instinct_nav/deploy/instinct_nav_onboard/config.py)：shared-lowcam 相机参数、obs layout、频率和默认 artifact 路径。
- [`deploy/instinct_nav_onboard/packets.py`](/workspace/instinct_nav/deploy/instinct_nav_onboard/packets.py)：真机 sensor packet 与 velocity command 数据结构。
- [`deploy/instinct_nav_onboard/obs_builder.py`](/workspace/instinct_nav/deploy/instinct_nav_onboard/obs_builder.py)：高层 actor obs 拼接接口和低层 depth history 预处理参考。
- [`deploy/instinct_nav_onboard/runtime.py`](/workspace/instinct_nav/deploy/instinct_nav_onboard/runtime.py)：5 Hz SRU runtime loop，输出速度命令，不直接发关节。
- [`deploy/instinct_nav_onboard/bridges.py`](/workspace/instinct_nav/deploy/instinct_nav_onboard/bridges.py)：`SensorSource` / `VelocityCommandSink` 抽象接口，用于后续对接 robot SDK 或 instinct_onboard。
- [`deploy/scripts/run_onboard_stub.py`](/workspace/instinct_nav/deploy/scripts/run_onboard_stub.py)：无 IsaacLab dry-run，验证部署包能独立启动。

已验证：

```bash
PYTHONPATH=/workspace/instinct_nav/deploy /workspace/instinctlab51-env/bin/python -m py_compile \
  /workspace/instinct_nav/deploy/instinct_nav_onboard/*.py \
  /workspace/instinct_nav/deploy/scripts/run_onboard_stub.py

PYTHONPATH=/workspace/instinct_nav/deploy /workspace/instinctlab51-env/bin/python \
  /workspace/instinct_nav/deploy/scripts/run_onboard_stub.py --steps 3
```

重要限制：当前 scene-token policy 的 `ActorCriticSRU.export_jit()` 仍是旧的 depth-tail 导出器，不包含 scene-token query branch；部署当前最佳 checkpoint 时优先走 Python checkpoint runtime。后续若要 JIT/ONNX 上机器人，需要先补 scene-token exporter。

### IsaacLab 对照 observation dump

2026-07-28 追加：`source/instinct_nav/scripts/eval.py` 已支持和 onboard deploy 相同格式的 observation/action dump，用于部署前分布对齐。

推荐先用当前最佳 shared-lowcam checkpoint 生成一小批仿真对照样本：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
  --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Play-v0 \
  --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-26_21-39-31_g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1/model_2999.pt \
  --num_envs 64 \
  --num_episodes 50 \
  --seed 72 \
  --headless \
  --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-26_21-39-31_g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1/eval_debug_dump_summary.json \
  --debug_dump_dir /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-26_21-39-31_g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1/eval_debug_dump \
  --debug_dump_env_id 0 \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 200
```

输出内容和 deploy dump 对齐：

```text
index.jsonl
step_00000001.npz
  depth_m, encoded_depth, scene_tokens, actor_obs, raw_action, command
  base_lin_vel_b, base_ang_vel_b, projected_gravity_b, goal_b, joint_pos, joint_vel
```

如果只想看统计量，减少磁盘占用，可追加：

```bash
--debug_dump_stats_only
```

部署前比较顺序：先对比 `index.jsonl` 里的 min/max/mean/std，再打开 `.npz` 看具体数组。重点检查 `projected_gravity_b` 符号、`base_ang_vel_b` 量纲、`depth_m` 无效值约定、`encoded_depth` 分布、`scene_tokens` 范围和 `raw_action/command` 是否异常。


## 2026-07-28 真机 ONNX 部署进展

今天完成了 G1 机器人端的轻量部署链路整理，目标是让机器人随身 NX 容器不依赖 IsaacLab、不依赖 torch，只用 `onnxruntime + instinct_onboard + RealSense` 运行导航。

已完成内容：

- 高层 SRU actor 已导出为 ONNX：`artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx`。
- 高层 SRU depth encoder 已导出为 ONNX：`artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx`。
- 低层 parkour ONNX 已整理到：`artifacts/low_level/g1_parkour/exported/`。
- 新增 torch-free 高层推理 runtime：`deploy/instinct_nav_onboard/onnx_high_level_runtime.py`。
- 新增机器人端 ONNX onboard 入口：`deploy/scripts/g1_nav_onboard_onnx.py`。
- 新增目标管理器：`deploy/scripts/goal_manager.py`，输入指定 odom/world frame 坐标点，持续发布机器人 base 坐标系下的目标，到点后连续发布停止目标让机器人回到 stand。
- 新增部署 manifest：`artifacts/high_level/sru_scene_token/deploy_manifest.json`，记录 obs 维度、动作维度、SRU hidden size、checkpoint iteration、ONNX sha256、scene-token 数量/单 token 维度/总维度。
- ONNX runtime 启动时会读取/校验 manifest、ONNX sha256、ONNX IO contract 和 scene-token/encoded-depth/proprio 语义字段，不再硬编码 hidden state shape。
- ONNX onboard 入口已接入 `/dog_odom` 线速度作为高层 actor 的 `base_lin_vel_b`，支持用 `--odom_twist_frame auto/body/world` 显式选择 twist 坐标系，避免部署观测中该项固定为 0 或坐标系误判。
- ONNX onboard 入口已加入 SRU 异常清零、回退 stand、命令 deadband / slew-rate / low-pass shaping。
- ONNX onboard 入口已加入深度相机帧新鲜度检查和 odom 新鲜度检查，深度/odom 超时会 latch autonomy fault、清零自主命令并回退 stand。fault latch 后不会因为 `--auto_start_parkour` 或新目标自动重入 parkour，需要重启节点显式 re-arm。
- 新增 dump 对比工具：`deploy/scripts/compare_debug_dumps.py`，用于比较 IsaacLab eval dump 和机器人 dry-run dump 的观测/动作分布。
- 修复 `deploy/instinct_nav_onboard/__init__.py`，避免 ONNX 部署路径因为包初始化提前导入 torch。

机器人端已验证：

```text
RealSense D435i 可枚举，深度流 480x270@60Hz 可读。
ROS2 lowstate / secondary_imu / lowcmd 话题可见。
高层 SRU ONNX 可被 onnxruntime 加载。
高层 deploy manifest 已生成，并记录 obs_dim=2960、rnn_hidden_size=512。
低层 parkour ONNX 可被 instinct_onboard 加载。
dry-run 下相机约 60Hz，主循环约 50Hz。
```

注意：dry-run 不会真实下发关节命令，所以 `ColdStartAgent` 可能一直卡在默认姿态误差检查，这是正常的。真正让机器人调整到 parkour 初始站姿时需要加 `--nodryrun`，但必须先确认机器人处于安全调试/运动控制状态。

### 完整上机运行顺序

下面顺序是当前推荐的真机测试流程。第一次上机只做 `dry-run -> stand -> 短目标`，不要一开始测试楼梯或长距离。

#### 0. 进入机器人容器并准备环境

在机器人 NX 上打开第一个终端：

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH
```

确认 ROS2 和相机基础状态：

```bash
ros2 topic list | grep -Ei "lowstate|secondary_imu|lowcmd|dog_odom"
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
```

#### 1. 检查 ONNX bundle 能加载

```bash
python -c "import onnxruntime as ort; d='artifacts/high_level/sru_scene_token'; print(ort.InferenceSession(d+'/sru_high_level_scene_token.onnx').get_inputs()); print(ort.InferenceSession(d+'/sru_high_level_depth_encoder.onnx').get_inputs())"
```

#### 2. 启动 onboard dry-run

这个命令不会真实发布 `/lowcmd`，只会发布 dry-run topic，用于确认相机、odom、SRU、低层 ONNX 都能跑通：

```bash
python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/sru_scene_token/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/dryrun \
  --debug_dump_every_steps 5 \
  --debug_dump_max_steps 100
```

预期现象：

```text
Loaded high-level SRU ONNX policy
Loaded high-level SRU depth encoder ONNX
Low state data received
Torso IMU data received
All necessary buffers received
Camera process running at about 60 Hz
Actual main loop frequency about 50 Hz
Dry-run auto-start enabled; starting parkour without cold start
```

`--auto_start_parkour` 在 dry-run 下会跳过无法实际改变关节的 cold-start 并进入 parkour 待命；未收到真实目标时保持零速度且不运行 SRU，收到目标后才验证高层 ONNX、相机、odom 和命令链。非 dry-run 首次测试不要添加该参数。如果这里出现 `Autonomy fault latched`，先不要进入非 dry-run。需要根据日志原因检查相机、odom 或 ONNX bundle。

#### 3. 启动目标管理器

另开第二个机器人容器终端，重复第 0 步的环境准备，然后启动：

```bash
python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.5 \
  --stop_publish_count 10
```

#### 4. dry-run 下发布短目标

在第三个终端发布一个很短的 odom 目标。第一次建议 `0.5m`，不是 `1m` 或更远：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
"{header: {frame_id: 'odom'}, point: {x: 0.5, y: 0.0, z: 0.0}}"
```

此时仍然是 dry-run，所以机器人不会真实运动。主要看 onboard 终端是否进入 `parkour`、SRU 是否产生有限 command、是否没有 fault。

#### 5. 非 dry-run 站姿测试

> **当前暂停执行：** 真机长距离测试曾发生跌倒。在真机 dump 与匹配的
> IsaacLab eval dump 完成分布对比、跌倒日志和硬件状态完成复核之前，不得执行本节
> 或下一节。以下命令仅保留为后续受控站姿验收记录。

恢复测试必须由用户明确确认。机器人需使用吊架/保护绳，现场操作员保持物理急停，
停止第 2 步 dry-run 进程后再启动 onboard。首次只做站姿，不启动目标管理器且不发布目标：

```bash
python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/sru_scene_token/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --safety_log_dir /workspace/instinct_nav_logs/safety \
  --debug_dump_dir /workspace/instinct_nav_logs/stand_test \
  --debug_dump_stats_only \
  --nodryrun
```

预期是机器人从当前状态调整到 parkour 默认站姿，然后停在 `stand` 等目标。不要加 `--auto_start_parkour`。

#### 6. 非 dry-run 短目标测试

本节同样处于暂停状态。只有站姿验收、安全日志和 dump 对比均通过后，才可由用户明确
批准恢复。届时启动第 3 步目标管理器，并只发布 0.5m 短目标：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
"{header: {frame_id: 'odom'}, point: {x: 0.5, y: 0.0, z: 0.0}}"
```

测试顺序建议：

```text
0.5m 前进 -> 0.5m 左/右偏移 -> 轻微转向目标 -> 1.0m 前进
```

任意阶段如果出现 `Autonomy fault latched`，节点会保持 `stand` 并拒绝新目标。处理方式是先查明 fault 原因，然后重启 onboard 节点重新 re-arm。

无线手柄必须保持连接。`R1` 会停止自主导航并回到 stand，`R2` 或 `L2` 会触发电机急停并退出进程。

实机模式还要求显式传入 `--safety_log_dir`，并建议与
`--debug_dump_dir` 一起放到 `/workspace` 下，不能再放 `/tmp`。部署节点会以
10 Hz 持久化 IMU 倾角、关节状态、低层目标、高层原始动作、速度命令、相机帧计数、
odom age 和扭矩裁剪计数。默认躯干倾角超过 `55°` 持续 `0.15 s`，或 IMU
重力向量失效时，会记录 `fall_safety_shutdown`、发布电机关闭命令并退出。
电机关闭后机器人会失去支撑，因此仍必须使用吊架/保护绳和现场急停人员。

### 补充说明

`--odom_twist_frame auto` 的规则是：如果 `/dog_odom.child_frame_id` 是 `base_link/base/body/torso_link/robot_center`，直接把 twist 当 body-frame 速度；否则把 twist 当 world-frame 速度，并用 odom pose quaternion 旋转到 body frame。现场正式测试建议显式指定 `--odom_twist_frame body` 或 `--odom_twist_frame world`，不要长期依赖 auto heuristic。

如果现场确认 `/dog_odom.twist` 不可靠，可以临时加 `--disable_odom_velocity` 回退到旧的零线速度观测，但这会重新引入训练/部署观测不一致。

手动停止导航可以直接向 onboard 内部目标接口发布零目标：

```bash
ros2 topic pub --once /instinct_nav/goal_pose geometry_msgs/msg/PoseStamped \
"{header: {frame_id: 'base_link'}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

### 对比仿真和真机 Debug Dump

先在 IsaacLab eval 和机器人 dry-run 各保存一组 debug dump，然后运行：

```bash
python deploy/scripts/compare_debug_dumps.py \
  --left /path/to/isaaclab_eval_debug_dump \
  --right /workspace/instinct_nav_logs/dryrun \
  --left_label isaaclab \
  --right_label robot \
  --max_files 200 \
  --output_json /workspace/instinct_nav_logs/dump_compare.json
```

优先看 `depth_m`、`encoded_depth`、`scene_tokens`、`actor_obs`、`raw_action/command` 的均值、方差、finite ratio 和 `mean_abs_delta`。如果这些分布差异很大，真机行为异常时应先修观测链路，而不是继续改网络。

当前状态机：

```text
cold_start 调整到 parkour 默认初始姿态
-> stand 等待目标点
-> 收到目标点后进入 parkour/SRU 导航
-> 到点后清零速度并切回 stand
```

不要在第一次测试时加 `--auto_start_parkour`，否则站好后会自动进入 parkour。

### 当前部署文件清单

```text
artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx
artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx
artifacts/high_level/sru_scene_token/deploy_manifest.json
artifacts/low_level/g1_parkour/exported/actor.onnx
artifacts/low_level/g1_parkour/exported/0-depth_encoder.onnx
artifacts/low_level/g1_parkour/params/agent.yaml
artifacts/low_level/g1_parkour/params/env.yaml
deploy/scripts/g1_nav_onboard_onnx.py
deploy/scripts/goal_manager.py
deploy/instinct_nav_onboard/onnx_high_level_runtime.py
```

## 2026-07-30 机器人端最终运行命令

以下命令使用已经在真机验证稳定的 InstinctLab 官方双策略：

```text
stand    : /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard
parkour  : /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair
```

`--standdir` 与 `--logdir` 不得指向同一目录。仓库内旧的
`artifacts/low_level/g1_parkour` 不用于正式真机部署。

### 1. 进入容器并准备环境

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH
```

### 2. 首次运行：dry-run

该命令不会向真实 `/lowcmd` 输出关节控制。`--auto_start_parkour` 仅用于绕过
dry-run 中无法完成的 cold-start 姿态调整并进入 parkour 待命。未收到目标时保持零速度且不运行 SRU；发布目标后才开始保存 SRU 和动作 dump。

```bash
mkdir -p /workspace/instinct_nav_logs/dryrun

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/sru_scene_token/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/dryrun \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 100
```

### 3. 启动目标管理器

在第二个容器终端重复第 1 步环境准备，然后运行：

```bash
python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.5 \
  --stop_publish_count 10
```

### 4. 发布 0.5 m 测试目标

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
"{header: {frame_id: 'odom'}, point: {x: 0.5, y: 0.0, z: 0.0}}"
```

检查 `/workspace/instinct_nav_logs/dryrun/index.jsonl` 中的 `reset`、
`policy_saturation_mask`、`command_clip_mask`、`scene_tokens` 和 `command`。
连续更新同一个目标时，`reset` 不应持续为 `true`。

### 5. 真机受控运行

仅在 dry-run、官方 stand 策略和急停均验证通过后执行。必须使用吊架或保护绳，
现场人员保持遥控器急停。真机命令不添加 `--auto_start_parkour`，并必须提供持久化
安全日志目录：

```bash
mkdir -p /workspace/instinct_nav_logs/safety /workspace/instinct_nav_logs/real

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/sru_scene_token/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --safety_log_dir /workspace/instinct_nav_logs/safety \
  --debug_dump_dir /workspace/instinct_nav_logs/real \
  --debug_dump_every_steps 5 \
  --debug_dump_max_steps 100 \
  --nodryrun
```

启动后先完成 cold-start 和 stand 验证，再启动目标管理器并发布 0.5 m 目标。
`R1` 返回 stand；`R2` 或 `L2` 触发急停并退出。

## 2026-07-30 训练匹配低层策略真机命令

> **已停用，禁止真机执行本节命令。**
>
> 2026-07-30 MuJoCo 对照发现，`artifacts/low_level/g1_parkour` 在零速度命令下
> 仍会产生明显后退和大姿态摆动，在 `vx=0.2 m/s` 时反向运动并跌倒。ONNX
> 与 checkpoint 数值匹配只能证明导出正确，不能证明策略本身可部署。当前完整
> SRU sim2sim 和后续真机测试必须使用已验证稳定的官方
> `data&model/checkpoints/parkour_onboard_preview_stair` 底层。

本节覆盖上一节的 parkour 低层选择。数值检查确认，高层 SRU 训练使用的
`artifacts/low_level/g1_parkour/model_30000.pt` 与仓库内导出的低层 ONNX
一致：

```text
actor ONNX 最大绝对误差        : 1.19e-6
depth encoder ONNX 最大绝对误差: 4.77e-7
```

官方 `parkour_onboard_preview_stair` 是另一套低层权重，不再作为当前高层 SRU
的匹配低层。站立阶段仍使用已验证的官方 `stand_onboard`：

```text
stand   : /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard
parkour : /workspace/unitree-jetson-container/instinct_nav/artifacts/low_level/g1_parkour
```

### 1. 启动目标管理器

```bash
python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.1 \
  --stop_publish_count 10
```

### 2. 启动训练匹配的真机控制

首次测试必须使用吊架或保护绳，并由现场人员持有遥控器。不要添加
`--auto_start_parkour`。启动后先完成 cold-start 和 stand，未确认站稳前不要
发布目标。

```bash
mkdir -p /workspace/instinct_nav_logs/safety /workspace/instinct_nav_logs/nodryrun_ours

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/instinct_nav/artifacts/low_level/g1_parkour \
  --sru_policy_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/sru_scene_token/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/sru_scene_token/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --safety_log_dir /workspace/instinct_nav_logs/safety \
  --debug_dump_dir /workspace/instinct_nav_logs/nodryrun_ours \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 100 \
  --nodryrun
```

### 3. 发布首次小目标

确认机器人已稳定站立后，在目标管理器使用的终端发布 `0.3 m` 正前方目标：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
"{header: {frame_id: 'odom'}, point: {x: 0.3, y: 0.0, z: 0.0}}"
```

`R1` 返回 stand 并锁存暂停，后续持续目标消息不会再次启动 parkour；需要重启
onboard 节点才能重新启用自主导航。`R2` 或 `L2` 触发急停并退出。

## 2026-07-30 MuJoCo SRU 完整导航 Sim2Sim

完整闭环使用以下频率和数据流：

```text
MuJoCo 200 Hz -> 动态深度相机 60 Hz -> SRU 5 Hz
              -> 官方 parkour 50 Hz -> 29 维关节 PD
```

默认目标是 MuJoCo 世界坐标系中的 `(2.0, 0.0, 0.0)`，到达阈值为 `0.5 m`。
底层默认使用已验证稳定的官方 `parkour_onboard_preview_stair`，不要替换为
`artifacts/low_level/g1_parkour`。

### Headless 验证

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python \
  deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --goal 2.0 0.0 0.0 \
  --duration 20 \
  --json_out artifacts/mujoco/sru_navigation_2m.json
```

### 实时可视化

在远程桌面终端中执行：

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python \
  deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --goal 2.0 0.0 0.0 \
  --duration 20 \
  --viewer \
  --json_out artifacts/mujoco/sru_navigation_2m_viewer.json
```

可通过 `--goal X Y Z` 修改世界坐标目标，通过
`--high_command_limit 0.3 0.2 0.4` 修改高层速度限幅。默认高层 ONNX、
深度编码器和 manifest 均读取 `artifacts/high_level/sru_scene_token/`。

当前平地验证结果：

```text
goal=(2.0, 0.0): 4.13 s 到达，未跌倒，最终距离 0.498 m
goal=(2.0, 1.0): 4.31 s 到达，未跌倒，最终距离 0.499 m
```

已知现象：当前 SRU 的 `vy` 在上述测试中持续达到 `-0.2 m/s` 限幅，主要依靠
偏航转向完成目标跟踪。这与 IsaacLab 和实机 dry-run 的横向动作偏置一致，
后续加入障碍物前必须继续分析，不能据此恢复真机自主导航。


## 2026-07-31 基于官方 Parkour ONNX 低层重新训练 SRU

Hiking Parkour checkpoint 的行走流畅性不理想，当前正式训练改用已验证稳定的官方 `parkour_onboard_preview_stair` 低层。任务、SRU、PPO、观测和奖励保持不变，只将冻结低层替换为官方 ONNX：

```text
actor:
data&model/checkpoints/parkour_onboard_preview_stair/exported/actor.onnx

depth encoder:
data&model/checkpoints/parkour_onboard_preview_stair/exported/0-depth_encoder.onnx

训练任务:
Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0
```

训练时 ONNX 权重会加载到等价的 PyTorch 模块，在 GPU 上对所有 Isaac 环境批量推理；所有低层参数均冻结。高层保持 2960 维 actor observation、3 维导航动作、512 维 SRU 和 `64 x 6` scene tokens。该任务默认从零训练高层，不要使用旧 SRU checkpoint 续训。

### 单轮 Smoke Test

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Dev-v0 \
  --num_envs 32 \
  --max_iterations 1 \
  --seed 72 \
  --run_name g1_sru_scene_token_risk_no_goal_shared_lowcam_official_onnx_smoke \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav \
  --headless
```

已验证该命令可完成一次 PPO 更新并生成 `model_0.pt`。

### 正式训练

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 72 \
  --run_name g1_sru_scene_token_risk_no_goal_shared_lowcam_official_onnx_v1 \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav \
  --headless
```

训练产物位于：

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/
```

## 2026-08-02 官方 Parkour 低层 SRU 训练完成与 Sim2Sim

正式训练已完成。训练使用固定的官方 Parkour ONNX 低层、4096 个 IsaacLab
并行环境和 seed 72；高层 SRU 从零训练，没有从旧 checkpoint 续训。

```text
run:
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/
2026-08-01_16-04-56_g1_sru_scene_token_risk_no_goal_shared_lowcam_official_onnx_v1/

final checkpoint:
model_3000.pt
```

不要将 `--num_envs` 直接提高到 8192。实测该配置在完成数轮更新后触发
PhysX `Patch buffer overflow`，随后以 CUDA `invalid configuration argument`
退出；4096 是当前已经完整验证的配置。

### 导出高层部署 ONNX

高层 SRU policy 使用最终 checkpoint 导出，并执行 PyTorch/ONNX 数值一致性检查：

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python \
  deploy/scripts/export_high_level_sru_onnx.py \
  --checkpoint_path logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-01_16-04-56_g1_sru_scene_token_risk_no_goal_shared_lowcam_official_onnx_v1/model_3000.pt \
  --output_dir artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801 \
  --require_onnx_parity

/workspace/instinctlab51-env/bin/python \
  deploy/scripts/export_high_level_depth_encoder_onnx.py \
  --weights_path source/instinct_nav/instinct_nav/navigation/assets/data/Policies/depth_encoder/vae_pretrain_new.pth \
  --output_dir artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801
```

部署目录包含：

```text
artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/
├── deploy_manifest.json
├── sru_high_level_depth_encoder.onnx
└── sru_high_level_scene_token.onnx
```

导出检查已通过，接口保持为 2960 维观测、3 维动作以及一层 512 维 SRU
hidden/cell state。

### MuJoCo 闭环结果

测试链路为动态深度相机、高层 SRU、官方 Parkour ONNX 和 29 维关节 PD
闭环。当前平地 headless 结果：

```text
goal=(2.0, 0.0, 0.0): 3.85 s 到达，未跌倒，最终距离 0.500 m
goal=(4.0, 0.0, 0.0): 8.85 s 到达，未跌倒，最终距离 0.499 m
```

4 米可视化命令：

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python \
  deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --high_depth_encoder artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --manifest artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --goal 4.0 0.0 0.0 \
  --duration 20 \
  --viewer \
  --json_out artifacts/mujoco/g1_sru_official_onnx_v1_20260801_4m_viewer.json
```

当前测试中前进指令持续达到 `0.3 m/s` 上限，侧移和旋转也出现瞬时限幅。
MuJoCo 已验证能够到达目标且不跌倒，但真机 `--nodryrun` 前仍需先完成
不同方向目标的 Sim2Sim 和机器人端 dry-run 动作分布检查。

## 2026-08-02 最新机器人部署命令

以下命令在机器人 Jetson 的既有 `unitree.unitree-jetson-container` 容器内执行。
高层使用本轮 `model_3000.pt` 导出的 ONNX，stand 和 parkour 使用机器人上已经
验证稳定的官方 checkpoint。首次启动必须先运行 dry-run。

### 1. 准备终端环境

每个新终端进入容器后都必须完整执行以下命令。先激活师兄已有的
`instinct_venv`，再加载 ROS 2 和 Unitree overlay，最后加入本项目、
`instinct_onboard` 与 CRC 扩展的 Python 路径：

```bash
cd /workspace/unitree-jetson-container
source instinct_onboard/instinct_venv/bin/activate
source unitree_ros2/setup.sh
mkdir -p /tmp/instinct_nav_crc
ln -sf /workspace/unitree-jetson-container/instinct_onboard/third_party/g1_crc/crc_module.so /tmp/instinct_nav_crc/crc_module.so
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:/tmp/instinct_nav_crc:$PYTHONPATH
cd /workspace/unitree-jetson-container/instinct_nav
```

不要跳过 import 检查：

```bash
python -c "import rclpy, crc_module; import unitree_hg.unitree_hg_s__rosidl_typesupport_c; from unitree_go.msg import WirelessController; import instinct_nav_onboard, instinct_onboard; print('deployment imports and ROS type support ok')"
```

如果 `source unitree_ros2/setup.sh` 没有成功加载 overlay，可显式执行：

```bash
source /opt/ros/humble/setup.bash
source /opt/unitree_ros2/install/setup.bash
```

然后重新建立 `/tmp/instinct_nav_crc`、执行 `export PYTHONPATH=...` 并进行
import 检查。只有输出 `deployment imports and ROS type support ok` 后才能启动节点。

确认相机、机器人状态和里程计均可用：

```bash
rs-enumerate-devices | head
ros2 topic list | grep -E 'lowstate|secondary_imu|lowcmd|dog_odom|wirelesscontroller'
```

### 2. 启动目标管理器

在终端 A 中运行：

```bash
cd /workspace/unitree-jetson-container/instinct_nav

python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.1 \
  --stop_publish_count 10
```

### 3. 最新模型 Dry-run

在终端 B 中运行。该命令只发布到随机命名的 `/lowcmd_dryrun_*`，不会控制真实
关节；`--auto_start_parkour` 仅用于 dry-run 跳过无法真实完成的 cold-start。

```bash
cd /workspace/unitree-jetson-container/instinct_nav
mkdir -p /workspace/instinct_nav_logs/dryrun_official_v1

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/dryrun_official_v1 \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 100
```

### 4. 查看机器人坐标并发布目标

待终端 B 显示相机约 60 Hz、主循环约 50 Hz，且 low state、IMU、joystick、
odom 均已收到后，在终端 C 查看机器人当前 odom 位姿：

```bash
ros2 topic echo /dog_odom --once
```

其中 `pose.pose.position.x/y` 是机器人在 odom 世界坐标系中的平面坐标，
`orientation` 是机器人朝向四元数。持续查看位置可运行：

```bash
ros2 topic echo /dog_odom --field pose.pose.position
```

目标管理器接收的是 odom **绝对坐标**，不是相对位移。若当前机器人约位于
`(1.478, 0.121)`，朝向约 `19.6 deg`，沿当前朝向前进 `0.8 m` 的目标约为
`(2.232, 0.389)`，发布示例为：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
"{header: {frame_id: 'odom'}, point: {x: 2.232, y: 0.389, z: 0.0}}"
```

实际测试必须先读取当前 odom，并替换示例中的 `x/y`。平面目标的 `z` 保持
`0.0`。检查日志中无 `Autonomy fault latched`、相机/odom 超时或持续异常
动作后，使用 `Ctrl+C` 结束 dry-run。

### 5. 受保护条件下运行真机

首次真机测试必须使用吊架或保护绳，现场人员持续持有遥控器。机器人先按官方
流程退出 sport mode 并复位关节。命令中不要添加 `--auto_start_parkour`；节点会
依次执行 cold-start、stand，收到目标后才进入 parkour。`R1` 回到 stand，`L2`
或 `R2` 触发安全退出。

```bash
cd /workspace/unitree-jetson-container/instinct_nav
mkdir -p /workspace/instinct_nav_logs/safety_official_v1 /workspace/instinct_nav_logs/nodryrun_official_v1

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.1 -0.05 -0.15 \
  --command_max 0.1 0.05 0.15 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.1 0.05 0.15 \
  --command_lowpass_alpha 0.4 \
  --safety_log_dir /workspace/instinct_nav_logs/safety_official_v1 \
  --debug_dump_dir /workspace/instinct_nav_logs/nodryrun_official_v1 \
  --debug_dump_every_steps 1 \
  --debug_dump_max_steps 100 \
  --nodryrun
```

确认机器人独立稳定保持 stand 后，先读取最新 odom 坐标，再发布正前方短距离
绝对目标。目标距离应明显大于 `0.1 m` 到达阈值；测试区域必须平整且无人。
首次真机保持上述低限速，不要直接恢复 `0.3/0.2/0.4`。

## MuJoCo 地形与导航可视化

MuJoCo 导航脚本支持 `flat`、`stairs` 和 `ramp` 三种地形。使用 `--viewer`
时默认显示以下调试标记：

- 黄色线框：机器人深度相机的视锥覆盖范围。
- 红、绿、蓝箭头：机器人躯干上方的 X、Y、Z 坐标轴。
- 绿色球体和竖线：世界坐标系中的目标点。

楼梯场景：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --high_depth_encoder artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --manifest artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --terrain stairs --terrain_start_x 1.5 --stair_height 0.12 \
  --stair_depth 0.50 --stair_steps 4 \
  --goal 4.5 0.0 0.48 --duration 30 --viewer
```

斜坡场景：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --high_depth_encoder artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --manifest artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --terrain ramp --terrain_start_x 1.5 --ramp_angle_deg 12 \
  --ramp_length 3.0 --goal 4.5 0.0 0.6 --duration 30 --viewer
```

`--terrain_width` 调整通道宽度，`--camera_frustum_length` 只调整黄色视锥的
显示长度，不改变策略实际使用的相机参数。需要隐藏所有标记时添加
`--no_debug_visuals`。不使用 `--viewer` 即可进行无界面批量 sim2sim 测试。

### 双终端交互目标模式

交互模式下，窗口不会因到达目标或 `--duration` 到期而关闭。机器人启动后保持
站立并等待目标；到达后停止，再等待下一个世界坐标目标。关闭 viewer 才会结束。

终端 1 启动持续运行的楼梯场景：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_scene_token.onnx \
  --high_depth_encoder artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/sru_high_level_depth_encoder.onnx \
  --manifest artifacts/high_level/g1_sru_shared_lowcam_official_onnx_v1_20260801/deploy_manifest.json \
  --terrain stairs --terrain_start_x 1.5 --stair_height 0.12 --stair_depth 0.50 --stair_steps 4 \
  --goal_threshold 0.5 --interactive_goal_port 5555 --viewer
```

终端 2 随时发送 MuJoCo 世界坐标系中的 `x y z`：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/send_goal.py 4.5 0.0 0.48
```

再次运行 `send_goal.py` 会立即替换当前目标并重置高层 SRU 状态。例如返回平地
上的另一个位置：`send_goal.py 0.5 -1.0 0.0`。自定义端口时，两边都使用
相同的 `--port`/`--interactive_goal_port` 数值。

### 默认丰富地形

默认文件为 `deploy/mujoco/terrains/rich_course.json`。其中包含宽楼梯、高台、
错位落脚台、走廊边界、矮障碍和终点平台，可直接手工修改对象参数。

在上面的持续交互命令中，将地形参数替换为：

```bash
--terrain custom --terrain_file deploy/mujoco/terrains/rich_course.json
```

每个对象的 `position` 和 `size` 使用米，`rotation_rpy_deg` 使用角度。当前支持
可碰撞的 `box`，所有对象同时进入深度相机射线检测和 MuJoCo 接触计算。

## 连续可见性得分（ScoreVis）

日期：2026-08-03；执行者：Codex。

原任务继续使用二值 visibility map，不改变已有 checkpoint 的输入语义。新增
`ScoreVis` 任务使用连续置信度：当前帧命中的栅格逐步向 1 累积，未命中的
历史栅格按 `0.995` 弱衰减。更新式为：

```text
decayed_t = 0.995 * score_(t-1)
score_t = clamp(decayed_t + 0.15 * hit_t * (1 - decayed_t), 0, 1)
```

scene token 的维度仍为 `64 x 6`；第三维从二值 patch 可见比例变为平均置信度，
`frontier` 改为连续置信度的空间梯度。该变化需要重新训练，不能直接复用旧
二值 visibility checkpoint。

训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-v0 \
  --num_envs 4096 --max_iterations 6000 --seed 72 \
  --run_name g1_sru_scene_token_score_vis_official_parkour_v1 \
  --wandb --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav --headless
```

导出新 checkpoint 后，MuJoCo 必须显式启用相同衰减：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy /path/to/score_vis/sru_high_level_scene_token.onnx \
  --high_depth_encoder /path/to/score_vis/sru_high_level_depth_encoder.onnx \
  --manifest /path/to/score_vis/deploy_manifest.json \
  --visibility_decay 0.995 --terrain custom \
  --terrain_file deploy/mujoco/terrains/rich_course.json --viewer
```

真机部署 ScoreVis 权重时同样添加 `--visibility_decay 0.995`。旧 ONNX 不添加该
参数，默认值 `1.0` 会严格保留原二值地图行为。

### Visibility 生成链路

1. 复用底层 parkour 与高层 SRU 共用的深度相机，读取带噪声和延迟的深度图。
2. 使用与深度帧同一时刻的相机位姿、四元数和内参，将有效深度像素反投影为 3D 点。
3. 将点转换到机器人 yaw 对齐的局部坐标系，并投影到机器人周围
   `6.4 m x 6.4 m`、分辨率为 `64 x 64` 的 BEV 网格。
4. 根据机器人连续两帧的根位姿变化，对历史地图进行平移和旋转补偿；ScoreVis
   模式保留双线性插值后的连续值，旧二值模式仍重新阈值化。
5. 当前帧命中的网格增加置信度，未命中的历史网格执行弱衰减。episode reset
   时地图、历史位姿和缓存一起清零。
6. 将 `64 x 64` 地图划分为 `8 x 8` 个 patch，得到 64 个 scene token。

每个 token 保持 6 维：

```text
[x_center, y_center, visibility_score, depth_mean, depth_edge, frontier]
```

- `visibility_score`：patch 内连续可见性置信度的平均值。
- `depth_mean`：当前深度图在该区域的归一化平均深度。
- `depth_edge`：深度图水平与垂直梯度的区域平均值。
- `frontier`：连续 visibility score 的空间梯度，表示已充分观测区域与低置信区域的边界。

最终 scene-token observation 仍为 `64 x 6 = 384` 维，因此 SRU 的网络输入尺寸
没有变化；变化的是这 384 维中 visibility/frontier 的数值分布，所以必须重新训练。

### 新增任务

```text
Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-v0
Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-Dev-v0
Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-Play-v0
```

实现同步覆盖以下链路：

- IsaacLab 训练 observation 与任务注册。
- Torch onboard observation builder。
- ONNX 高层运行时及 visibility 位姿补偿。
- MuJoCo sim2sim，通过 `--visibility_decay 0.995` 启用。
- 真机 ONNX 部署，通过同名参数启用。

### 今日验证

- 所有修改文件通过 `py_compile`。
- `deploy/tests` 全量测试通过：`18 passed`。
- 连续模式回归测试确认：重复命中时得分升高，停止命中时得分衰减。
- 兼容性测试确认：默认 `visibility_decay=1.0` 时旧地图仍为二值且不会衰减。
- ScoreVis Dev 任务已在 IsaacLab 中成功注册、解析并初始化。


## 动态障碍课程（Dynamic Obstacle Curriculum）

日期：2026-08-04；执行者：Codex。

在 ScoreVis 基础上新增阶段三：动态障碍避障课程。低层 parkour 保持冻结，只往场景里加脚本化运动的刚体障碍，并接入相机射线、避障奖励与碰撞终止。

### 新增内容

- `mdp/dynamic_obstacles.py`：`DynamicObstacleManager`，支持 `patrol`（折线往返）与 `random_walk`（随机转向）两种运动模式；reset 时在机器人周围环形区域采样并避开目标点；高度由地形高度场实时查询，保证障碍贴着地面移动。
- env cfg：每个 env 生成 4 个 0.5×0.5×0.8 m 的橙色箱子（`RigidObjectCfg`），共享相机 `mesh_prim_paths` 加入障碍（`track_mesh_transforms=True`，保证移动网格实时参与 raycast）。
- 事件：`dynamic_obstacles_reset`（reset 模式）、`dynamic_obstacles_update`（interval 0.0s，即每个 env step 更新一次）。
- 奖励/终止：`obstacle_proximity`（-0.5，指数 shaping，penalty_radius=1.2m、falloff=0.3）、`obstacle_approach_penalty`（-0.3，接近速度惩罚）、`obstacle_collision`（0.6m 内终止）。

### 新增任务

```text
Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-v0
Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-Dev-v0
Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-Play-v0
```

### 训练命令

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train   --task Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-v0   --num_envs 4096 --max_iterations 6000 --seed 72   --run_name g1_sru_scene_token_score_vis_dynamic_official_parkour_v1   --wandb --wandb_entity qqfqqf0612-wuhan-university   --wandb_project instinct_nav --headless
```

### 参数（可调课程项）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `num_obstacles` | 4 | 每 env 障碍数 |
| `obstacle_speed_range` | (0.3, 0.6) | 运动速度 m/s |
| `obstacle_pattern` | patrol | patrol / random_walk |
| `obstacle_half_extent_m` | 2.0 | 巡逻盒半宽 |
| `obstacle_min/max_start_dist` | 1.5 / 3.2 | 初始环形采样半径 |
| `obstacle_collision_radius` | 0.6 | 碰撞终止半径 |
| `obstacle_penalty_radius` | 1.2 | 近障惩罚半径 |
| `obstacle_auto_switch_success` | 0.9 | 普通阶段窗口成功率达标后自动开启动态 |
| `obstacle_max_switch_iteration` | 3000 | 最迟在第 3000 个 PPO iteration 强制开启动态（保证动态阶段训练量） |
| `obstacle_env_steps_per_iteration` | 16 | 每 PPO iteration 的 env step 数，用于把轮数换算成 env step（3000 × 16 = 48000） |
| `obstacle_switch_check_interval` | 100 | 每 100 env step 检查一次切换条件 |

### 课程切换机制（单次训练内自动切换）

一次训练共 6000 轮，不需要 resume_checkpoint：

1. **静态阶段**（约前 0~3000 轮）：障碍停在离机器人 40m 远的驻车位置（超出 30m 地形块，机器人永远走不到）、重力关闭、速度为 0，深度/可见性看不到它们；且奖励/终止函数在未启用时**显式返回 0/False**，静态阶段完全等价于普通 ScoreVis 训练；
2. 每 100 env step 检查一次窗口成功率（`robot_goal` 的 success buffer 均值，约 4096×10 个 episode 的滚动窗口）；
3. **切换触发**：窗口成功率 ≥ 0.9 **或** PPO 迭代数达到 3000（硬上限 = 3000 × 16 = 48000 env step，保证动态阶段至少 ~3000 轮训练量）→ 障碍自动传送到机器人周围环形区域并开始运动（避开当前机器人位置和目标点）；
4. **动态阶段**（切换到第 6000 轮）：正常避障训练，proximity/approach 惩罚与碰撞终止生效。

日志会打印 `[DynamicObstacleManager] Switching curriculum at env step ... (~iteration ...) ...`。新增 TensorBoard/WandB 指标：`Metrics/dynamic_obstacles/enabled`（0/1）与 `Metrics/dynamic_obstacles/min_dist_mean`（平均最近障碍距离，静态阶段为 40m）。


### 今日验证

- 所有新增/修改文件通过 `py_compile`。
- Dev smoke test 通过：`--num_envs 32 --max_iterations 1` 成功创建 32 env（含 4×32 个动态障碍）、事件/奖励/终止全部激活、产出 `model_0.pt`。
- 单任务课程切换 smoke test 通过：静态阶段障碍驻车在远处、奖励/终止保持 ~0，自动切换逻辑（窗口成功率 ≥ 0.9 或第 3000 轮）已接入并验证不报错。
- 待办（未做）：动态场景 eval manifest（统计 success + collision frequency）、MuJoCo 移动 box 扩展、真机侧同步。

## 2026-08-12 动态障碍评估与部署导出

动态障碍课程已完成 `model_5999.pt` 训练。评估结果如下：

| 场景 | 回合数 | 成功率 | 关键终止率 |
| --- | ---: | ---: | --- |
| 静态（关闭动态障碍） | 500 | 0.924 | `base_contact=0.038`，`large_pitch_angle=0.048` |
| 动态（强制开启障碍） | 500 | 0.840 | `obstacle_collision=0.060`，`base_contact=0.098`，`large_pitch_angle=0.064` |

评估文件：

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-09_21-13-05_g1_sru_scene_token_score_vis_dynamic_official_parkour_v2/eval_static_no_dynamic_model_5999_500ep.json
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-09_21-13-05_g1_sru_scene_token_score_vis_dynamic_official_parkour_v2/eval_dynamic_forced_model_5999_500ep.json
```

### 高层 ONNX 导出

使用最新动态障碍训练 checkpoint 导出，PyTorch/ONNX parity 检查已通过：

```text
artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/
├── sru_high_level_scene_token.onnx
├── sru_high_level_depth_encoder.onnx
└── deploy_manifest.json
```

MuJoCo 部署时必须保持 ScoreVis 的可见性衰减：

```bash
/workspace/instinctlab51-env/bin/python deploy/mujoco/g1_sru_navigation_sim2sim.py \
  --low_policy_dir 'data&model/checkpoints/parkour_onboard_preview_stair' \
  --high_policy artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_scene_token.onnx \
  --high_depth_encoder artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_depth_encoder.onnx \
  --manifest artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/deploy_manifest.json \
  --visibility_decay 0.995 --terrain custom \
  --terrain_file deploy/mujoco/terrains/rich_course.json --viewer
```

真机 ONNX 部署同样传入 `--visibility_decay 0.995`，不能省略，否则会退回旧的无衰减可见性行为。

### 真机 ROS2 ONNX 命令

在机器人容器内完成 ROS2、相机和 `instinct_venv` 初始化后运行。先不加
`--nodryrun` 做 dry-run 检查；确认传感器、低层策略和高层策略正常后，再加
`--nodryrun` 发布真实 `/lowcmd`：

```bash
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/deploy_manifest.json \
  --visibility_decay 0.995 \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom \
  --odom_twist_frame body \
  --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 \
  --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 \
  --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/score_vis_dynamic_v2 \
  --debug_dump_every_steps 5 \
  --debug_dump_max_steps 100
```

真实控制前，在上述命令末尾追加：

```bash
--nodryrun
```

### 最新模型三终端完整流程

以下命令均在机器人 NX 容器中执行，路径使用本次导出的
`g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812` bundle。

**终端 1：进入容器、准备环境并检查设备**

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH

ros2 topic list | grep -Ei "lowstate|secondary_imu|lowcmd|dog_odom"
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
python -c "import onnxruntime as ort; d='artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812'; print(ort.InferenceSession(d+'/sru_high_level_scene_token.onnx').get_inputs()); print(ort.InferenceSession(d+'/sru_high_level_depth_encoder.onnx').get_inputs())"
```

**终端 2：启动最新高层 ONNX onboard（先 dry-run）**

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH
mkdir -p /workspace/instinct_nav_logs/score_vis_dynamic_v2_dryrun

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/deploy_manifest.json \
  --visibility_decay 0.995 \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom --odom_twist_frame body --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/score_vis_dynamic_v2_dryrun \
  --debug_dump_every_steps 1 --debug_dump_max_steps 100
```

dry-run 确认正常后，停止终端 2 的进程。首次真实测试必须使用吊架/保护绳和现场急停，
然后在终端 2 执行下面的真实控制命令。与 dry-run 命令相比，该命令增加了
`--nodryrun` 和持久化安全日志目录，并且不使用 `--auto_start_parkour`：

```bash
mkdir -p /workspace/instinct_nav_logs/score_vis_dynamic_v2_safety

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_scene_token_score_vis_dynamic_official_parkour_v2_20260812/deploy_manifest.json \
  --visibility_decay 0.995 \
  --camera_frame_timeout_s 0.25 \
  --goal_pose_topic /instinct_nav/goal_pose \
  --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom --odom_twist_frame body --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --safety_log_dir /workspace/instinct_nav_logs/score_vis_dynamic_v2_safety \
  --debug_dump_dir /workspace/instinct_nav_logs/score_vis_dynamic_v2_real \
  --debug_dump_every_steps 5 --debug_dump_max_steps 100 \
  --nodryrun
```

启动后先等待机器人完成 cold-start 并稳定停在 stand，确认站姿和安全日志正常后，
再由终端 3 发布 `0.5m` 短目标。不要在首次真实测试中加入 `--auto_start_parkour`。

**终端 3：启动目标管理器并发布短目标**

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH

python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.5 \
  --stop_publish_count 10
```

目标管理器启动后，在同一终端或另一个 shell 发布短目标：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'odom'}, point: {x: 0.5, y: 0.0, z: 0.0}}"
```

建议顺序为 `0.5 m` 前进、左右 `0.5 m` 偏移，再进行 `1.0 m` 测试。确认站姿、
odom、相机和 dump 均正常后才发布目标；按下 `R1` 返回 stand，`R2/L2` 用于急停。

## 2026-08-12 Dynamic Map Cross-Attention Ablation

为增强动态障碍变化的表达，新增独立 ablation，不修改现有 ScoreVis 6D scene-token
任务。历史 ScoreVis 保持 `decay=0.995`，当前帧从同一 noisy/delayed shared-lowcam
深度流提取 patch 特征。

每个 `8 x 8` patch 输出 5 个原始量：

```text
[history_score, current_hit, current_depth, current_edge, visibility_delta]
```

网络不再把这些量直接当作手工 scene-risk token。当前帧 4D 分支作为 query，历史
ScoreVis 1D 分支作为 key/value，经过带二维位置 embedding 的 map-to-map cross
attention 后生成 64 个 learned dynamic tokens。现有 SRU hidden + proprio query
再对 learned tokens 做第二层 cross attention，得到 actor/critic scene feature。

新增任务：

```text
Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-v0
Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-Dev-v0
Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-Play-v0
```

正式训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-v0 \
  --num_envs 4096 --max_iterations 6000 --seed 72 \
  --run_name g1_sru_dynamic_map_cross_corridor_v1 \
  --wandb --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav --headless
```

验证结果：`py_compile`、纯 PyTorch actor/critic forward/backward 和 32-env PPO smoke
均通过。smoke observation 为 actor `2896 = 16 + 2560 + 64*5`、critic `6033`，
并成功产出：

```text
logs/rsl_rl/g1_instinct_navigation_dynamic_map_cross_dev/2026-08-12_22-23-37_dynamic_map_cross_smoke/model_0.pt
```

该任务必须从零训练，不能加载旧 ScoreVis `64x6` checkpoint。高层 ONNX exporter
和真机 observation builder 需在新 checkpoint 训练完成后按 `64x5` 契约同步扩展；
现有 `64x6` 部署 bundle 保持不变。

### 2026-08-12 动态障碍课程行为升级

本节覆盖上方 2026-08-04 版本的随机 `patrol/random_walk` 行为描述。DynamicMapCross
正式任务现在采用机器人当前位置到目标点的导航走廊生成障碍交互轨迹：

```text
横穿 30% / 迎面 25% / 同向 15% / 斜向汇入 20% / 走停 10%
```

每条轨迹在走廊 `25%~75%` 位置附近采样，最多 rejection sampling 12 次，并约束：

- 障碍起点距机器人至少 `1.5m`；
- 轨迹起点和终点避开目标至少 `1.0m`；
- 多障碍起点间距至少 `0.9m`；
- 采样失败时使用确定性的分隔横穿轨迹，不接受重叠位置；
- 轨迹到端点后往返，走停行为在运动和 `0.5~1.5s` 暂停之间切换；
- 每个环境 reset 时根据新的机器人与目标位置重新生成交互轨迹。

课程不再因为窗口成功率提前开启。正式任务固定在第 `3003` 个 PPO iteration 切换：

```text
3003 iterations * 16 env steps/iteration = 48048 env steps
```

Dev task 仅为 smoke test 将切换阈值设为 0，以实际覆盖动态轨迹代码；正式/Play task
仍使用 `3003`。动态路径 smoke 已通过并产出：

```text
logs/rsl_rl/g1_instinct_navigation_dynamic_map_cross_dev/2026-08-12_22-35-53_dynamic_map_cross_trajectory_smoke/model_0.pt
```

### VisScore + 当前帧 Cross-Attention 网络流程

这次改进的核心是保留长期可见性记忆 `VisScore`，同时显式加入当前帧。网络不再只依赖
经过衰减的历史图判断动态变化，也不再把手工 patch 特征直接作为最终 scene token。

#### 1. VisScore 历史图

深度图使用与高层 policy 相同的 noisy/delayed shared-lowcam 输入。每帧深度投影到机器人
局部坐标系中的 `8 x 8 = 64` 个 patch，并更新局部可见性记忆：

```text
history_score[t] = max(current_visibility[t], decay * history_score[t-1])
decay = 0.995
```

`history_score` 表示最近一段时间内该局部区域是否被看见，而不是当前帧中是否仍然可见。
原有 ScoreVis `64 x 6` 任务保持不变；本次新增的是独立 DynamicMapCross ablation。

#### 2. 当前帧可见图

每个 patch 从当前深度帧提取四个量：

```text
current_hit       当前帧是否有有效投影/占用
current_depth     当前帧归一化平均深度
current_edge      patch 内深度边缘强度
visibility_delta  current_hit 与 history_score 的变化差
```

因此 observation manager 输出的逻辑 patch 为：

```text
[history_score, current_hit, current_depth, current_edge, visibility_delta]
```

总 shape 为 `[num_envs, 64, 5]`，flatten 后为 `320` 维。`visibility_delta` 使刚出现、
刚消失或发生遮挡变化的区域可以直接进入网络，而不必让 SRU 从单张衰减图中反推。

#### 3. 第一级 Cross-Attention：当前帧查询 VisScore

网络先把历史图和当前帧拆成两个分支：

```text
history_score [B,64,1] -> history projection -> K/V
current map   [B,64,4] -> current projection -> Q
2D patch position      -> learned position embedding
Q(current) x K/V(history) -> map cross-attention -> 64 learned dynamic tokens
```

二维 patch position embedding 同时加入 query 和 key/value，使 attention 保留局部空间对应关系。
输出 token 已经是当前观测相对于历史可见性记忆的学习结果，不再等同于原始 5D 手工特征。

#### 4. 第二级 Cross-Attention：导航状态查询动态 token

随后沿用 query-token 架构，让导航 recurrent state 决定当前应该关注哪些动态区域：

```text
proprio + encoded depth -> SRU -> hidden state
hidden state + proprio  -> navigation query
navigation query x learned dynamic tokens -> scene feature
scene feature + SRU feature -> actor/critic MLP
```

目标信息仍只通过普通 `target_position` observation 和 navigation query 影响注意力，
不会写入每个 patch token。这延续了 v3 no-goal token 的结论，避免在局部场景描述中重复
混入目标方向和距离。

完整数据流如下：

```text
shared noisy/delayed depth
  |-- DepthNoiseEncoder --------------------------> encoded depth (2560)
  |-- local projection -> VisScore memory --------> history [64,1] -- K/V --+
  `-- current-frame patch extraction -------------> current [64,4] -- Q ----+--> map cross-attention
                                                                              |
                                                                              v
                                                               learned dynamic tokens [64,D]
                                                                              |
proprio + encoded depth -> SRU -> hidden + proprio -> navigation query -------+--> token cross-attention
                                                                                     |
                                                                                     v
                                                                            actor / critic output
```

对应的最终 observation 维度为：

```text
actor  = 16 + 2560 + 64*5 = 2896
critic = 17 + 3136 + 2560 + 64*5 = 6033
```

#### 5. 动态障碍课程与正式训练

动态障碍轨迹围绕机器人到目标的走廊生成，包括横穿、迎面、同向、斜向汇入和走停。
正式任务不会根据成功率提前开启动态障碍，日志已确认在 iteration `3003` 精确切换：

```text
[DynamicObstacleManager] Switching curriculum at env step 48048 (~iteration 3003) (fixed iteration 3003).
[DynamicObstacleManager] Dynamic obstacles ENABLED (step 48048).
Learning iteration 3003/6000
```

后台正式训练命令：

```bash
setsid /workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-v0 \
  --num_envs 4096 \
  --max_iterations 6000 \
  --seed 72 \
  --run_name g1_sru_dynamic_map_cross_corridor_v1 \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav \
  --headless \
  </dev/null > /workspace/instinct_nav/logs/train_dynamic_map_cross_corridor_v1.out 2>&1 &
```

实际 run 目录：

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/
```

截至 2026-08-15，该次 `6000` iteration 训练已经完成。TensorBoard 最后 step 为
`5999`，最终 checkpoint 为 `model_5999.pt`，checkpoint 内记录的 `iter` 也为 `5999`。
iteration 5999 的训练窗口指标为：

| metric | value |
| --- | ---: |
| `Train/mean_reward` | `32.1969` |
| `Metrics/robot_goal/success_rate` | `0.8854` |
| `Episode_Reward/reach_goal_xy_tight` | `0.7591` |
| `Episode_Termination/base_contact` | `0.0854` |
| `Episode_Termination/large_pitch_angle` | `0.0506` |
| `Episode_Termination/obstacle_collision` | `0.0507` |
| `Policy/mean_noise_std` | `0.4521` |
| `Metrics/dynamic_obstacles/enabled` | `1.0000` |

训练进程已经退出。退出时 Isaac Sim 在 `simulation_app.close()` 阶段报告 Vulkan
`ERROR_OUT_OF_HOST_MEMORY` 并发生 segmentation fault；该异常发生在 step 5999 的事件和
`model_5999.pt` 完整写入之后，不影响最终 checkpoint，但说明仿真清理阶段仍有资源释放
问题。以上是训练窗口指标，不替代后续普通 eval。

检查最终产物和日志：

```bash
ls -lh logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/model_5999.pt
tail -100 /workspace/instinct_nav/logs/train_dynamic_map_cross_corridor_v1.out
```

### 2026-08-15 model_5999 静态/动态障碍普通评估

使用相同 checkpoint、Play task、`256` 个环境、`1000` episodes 和 `seed=72` 完成两组
普通评估。静态组保持课程关闭；动态组使用 `--force_dynamic_obstacles`，日志确认从 step 0
启用并为全部环境重新采样走廊交互轨迹。

| condition | episodes | success | maze | non-maze | pits | stairs | base contact | large pitch | obstacle collision | time out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 静态障碍 | 1000 | **0.9260** | 0.9135 | 0.9320 | 0.9216 | 0.9369 | 0.0480 | 0.0510 | 0.0000 | 0.0170 |
| 动态障碍 | 1000 | **0.7570** | 0.7562 | 0.7663 | 0.7385 | 0.7664 | 0.1410 | 0.0630 | 0.1030 | 0.0100 |

动态障碍使总体成功率下降 `0.169`（16.9 个百分点），显式障碍碰撞率为 `0.103`，
base contact 也从 `0.048` 上升到 `0.141`。这说明 policy 在静态场景保持较强能力，但动态
交互仍是主要短板。该结果属于普通 eval；不同 episode 完成顺序会使两组 terrain episode
数量略有差异，因此不是严格的逐案例配对比较。

评估结果文件：

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/eval_static_model_5999_1000ep.json
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/eval_dynamic_model_5999_1000ep.json
```

## 2026-08-17: Remove Unused Sonic Integration

- Removed the unused `g1_sonic` task registrations and configuration.
- Removed the Sonic action, backend, command adapter, and their tests.
- Removed the stale Instinct contract test that loaded the legacy `InstinctFrozenG1Backend`
  embedded in `sonic_backend.py`; current training uses the independent
  `instinct_frozen_backend.py` implementation.
- The active `g1_instinct` tasks and frozen official G1 parkour low-level policy are unchanged.

## 2026-08-17: Fix Evaluation Episode Outcome Accounting

- Evaluation no longer reads the rolling success tracker as the result of a
  finished episode; that tracker could be stale for `bad_orientation` and dynamic
  obstacle terminations, and multiple termination terms could write it in one step.
- Both evaluation passes now derive success from the current episode's
  `early_termination` term and classify each episode exactly once in
  `episode_outcomes`.
- Physical failures take precedence when multiple terms fire together. The
  existing `termination_rates` remain available as overlapping trigger diagnostics.
- `py_compile` and `git diff --check` pass. A real evaluation should be rerun after
  the current GPU/Isaac process is available.

## 2026-08-18: v4 Model 3000 Evaluation With Correct Episode Outcomes

Checkpoint:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-16_16-28-39_g1_sru_dynamic_map_cross_rich_terrain_v4/model_3000.pt
```

Both passes used 256 environments, 1000 episodes, seed 72, and the corrected
`episode_outcomes` accounting. The static pass left dynamic obstacles disabled;
the dynamic pass used `--force_dynamic_obstacles` and confirmed obstacles enabled
from step 0.

| Condition | Success | Base contact | Orientation | Obstacle collision | Timeout | Outcome total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0.802 | 0.071 | 0.114 | 0.000 | 0.013 | 1000 |
| Dynamic | 0.703 | 0.098 | 0.071 | 0.123 | 0.005 | 1000 |

Dynamic obstacles reduce success by 9.9 percentage points. The main new failure
mode is obstacle collision (12.3%), followed by base contact (9.8%). The exact
JSON outputs are:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-16_16-28-39_g1_sru_dynamic_map_cross_rich_terrain_v4/eval_static_model_3000_1000ep_fixed.json
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-16_16-28-39_g1_sru_dynamic_map_cross_rich_terrain_v4/eval_dynamic_model_3000_1000ep_fixed.json
```

## 2026-08-17：统一视觉延迟与 Current-BEV DynamicMapCross

本轮只修改 observation/geometry，不修改 SRU 网络或课程：

- encoded depth、noisy raw depth、capture-time camera pose/quaternion 和 intrinsics 统一进入
  `camera_visual_packet` delay buffer，同一步 actor/critic/map 复用同一个 packet 与 lag。
- 当前 robot pose 不延迟；capture-time camera pose 仅用于将延迟深度正确投影到当前机器人局部坐标。
- DynamicMapCross 当前帧由 delayed raw depth 投影为 robot-local `64x64` Current BEV，不再从
  相机图像坐标直接池化。
- 5 维 patch 保持 `64x5=320`，使用更新前 History、Current BEV hit/depth/edge 和 signed delta。
- reset 同步清空 packet cache、delay buffer 和 visibility map。

当前 v4 进程使用修改前代码，未停止或覆盖；新链路属于后续 v5。`py_compile`、
`git diff --check` 已通过。32-env Isaac smoke 因 v4 占用唯一 CUDA context 暂未执行。

动态组的关键启动证据：

```text
[DynamicObstacleManager] Dynamic obstacles ENABLED (step 0).
[INFO] Dynamic obstacles forced ENABLED for evaluation.
```


### 2026-08-15 DynamicMapCross model_5999 ONNX 部署 bundle

最终 checkpoint 已转换为包含两级 cross-attention 的高层 ONNX。该 exporter 会自动从
checkpoint 识别 `scene_token_dim=5` 和 `use_dynamic_map_cross_attention=true`，不会再按
旧 ScoreVis `64x6` 网络错误导出。

```text
artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/
├── deploy_manifest.json
├── sru_high_level_depth_encoder.onnx
└── sru_high_level_scene_token.onnx
```

导出命令：

```bash
cd /workspace/instinct_nav

/workspace/instinctlab51-env/bin/python deploy/scripts/export_high_level_sru_onnx.py   --checkpoint_path logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/model_5999.pt   --output_dir artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815   --require_onnx_parity

/workspace/instinctlab51-env/bin/python deploy/scripts/export_high_level_depth_encoder_onnx.py   --weights_path source/instinct_nav/instinct_nav/navigation/assets/data/Policies/depth_encoder/vae_pretrain_new.pth   --output_dir artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815
```

部署契约：

```text
policy input obs:          [B, 2896]
policy input h/c:          [1, B, 512]
policy output actions:     [B, 3]
encoded depth:             2560 = 64*5*8
dynamic map patches:       320 = 64*5
patch layout:              [history_score, current_hit, current_depth, current_edge, visibility_delta]
visibility_decay:          0.995
checkpoint iteration:      5999
ONNX opset:                17
```

文件哈希：

```text
1b9d0dcb1b0a9b6cd64cd9bff3f199116cfce1044f19798ad84082b79a2e2fae  sru_high_level_depth_encoder.onnx
79d4879bd14c08bb791abe11dca7db801e173b8ae9f0ffdc2bc0edd556152cfb  sru_high_level_scene_token.onnx
```

验证结果：PyTorch/ONNX zero、random 和连续四步 recurrent parity 均通过；ONNX-only
runtime 也完成一帧 `depth -> encoder -> 64x5 patches -> SRU policy -> command` 端到端
测试，command、hidden 和 cell 均为有限值。

真机部署时必须使用包含本节修改的最新项目代码。旧版 onboard runtime 固定生成
ScoreVis `64x6` observation，不能与该 bundle 混用。真机命令中的三个路径应改为：

```bash
--sru_policy_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_scene_token.onnx --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_depth_encoder.onnx --deploy_manifest_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/deploy_manifest.json --visibility_decay 0.995
```


### DynamicMapCross 真机部署：三个终端完整命令与遥操作

使用最新 bundle：`artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/`。
机器人端必须使用包含 DynamicMapCross `64x5` onboard runtime 的最新代码。首次真实控制
必须使用吊架/保护绳、现场监护人员和物理急停。

#### 终端 1：环境准备和设备检查

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH

ARTIFACT_DIR=artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815
ls -lh "$ARTIFACT_DIR"
sha256sum "$ARTIFACT_DIR"/sru_high_level_scene_token.onnx
sha256sum "$ARTIFACT_DIR"/sru_high_level_depth_encoder.onnx
python -c "import json; d=json.load(open('$ARTIFACT_DIR/deploy_manifest.json')); print(json.dumps(d,indent=2))"
python -c "import onnxruntime as ort; d='$ARTIFACT_DIR'; p=ort.InferenceSession(d+'/sru_high_level_scene_token.onnx'); e=ort.InferenceSession(d+'/sru_high_level_depth_encoder.onnx'); print([(x.name,x.shape) for x in p.get_inputs()]); print([(x.name,x.shape) for x in p.get_outputs()]); print([(x.name,x.shape) for x in e.get_inputs()]); print([(x.name,x.shape) for x in e.get_outputs()])"

ros2 topic list | grep -Ei "lowstate|secondary_imu|lowcmd|dog_odom|wirelesscontroller"
ros2 topic hz /dog_odom
```

`/dog_odom` 有稳定频率后按 `Ctrl-C`，继续检查：

```bash
ros2 topic echo /dog_odom --once
ros2 topic echo /wirelesscontroller --once
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
```

期望 policy 输入为 `obs [batch,2896]`、`h/c [1,batch,512]`，depth encoder 输出
`[batch,64,5,8]`。正确哈希：

```text
79d4879bd14c08bb791abe11dca7db801e173b8ae9f0ffdc2bc0edd556152cfb  sru_high_level_scene_token.onnx
1b9d0dcb1b0a9b6cd64cd9bff3f199116cfce1044f19798ad84082b79a2e2fae  sru_high_level_depth_encoder.onnx
```

#### 终端 2：先运行 dry-run

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH
mkdir -p /workspace/instinct_nav_logs/dynamic_map_cross_model5999_dryrun

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/deploy_manifest.json \
  --visibility_decay 0.995 --sru_high_level_hz 5.0 \
  --camera_frame_timeout_s 0.25 --joystick_topic /wirelesscontroller \
  --goal_pose_topic /instinct_nav/goal_pose --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom --odom_twist_frame body --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --fall_tilt_threshold_deg 55 --fall_tilt_duration_s 0.15 \
  --auto_start_parkour \
  --debug_dump_dir /workspace/instinct_nav_logs/dynamic_map_cross_model5999_dryrun \
  --debug_dump_every_steps 1 --debug_dump_max_steps 100
```

dry-run 默认不发布真实 `/lowcmd`。确认没有 contract mismatch/watchdog timeout，且
`actor_obs=2896`、scene patch 为 320 维、action/command 均为有限值。然后在终端 2
按 `Ctrl-C` 停止；不要同时运行 dry-run 和真实节点。

#### 终端 2：确认 dry-run 后运行真实控制

真实命令不使用 `--auto_start_parkour`。节点自动执行 `cold_start -> stand`，收到目标后
才进入 parkour。真实模式必须有 `--nodryrun` 和 `--safety_log_dir`。

```bash
mkdir -p /workspace/instinct_nav_logs/dynamic_map_cross_model5999_safety
mkdir -p /workspace/instinct_nav_logs/dynamic_map_cross_model5999_real

python deploy/scripts/g1_nav_onboard_onnx.py \
  --standdir /workspace/unitree-jetson-container/data/data/checkpoints/stand_onboard \
  --logdir /workspace/unitree-jetson-container/data/data/checkpoints/parkour_onboard_preview_stair \
  --sru_policy_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_scene_token.onnx \
  --sru_depth_encoder_onnx_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/sru_high_level_depth_encoder.onnx \
  --deploy_manifest_path artifacts/high_level/g1_sru_dynamic_map_cross_corridor_model5999_20260815/deploy_manifest.json \
  --visibility_decay 0.995 --sru_high_level_hz 5.0 \
  --camera_frame_timeout_s 0.25 --joystick_topic /wirelesscontroller \
  --goal_pose_topic /instinct_nav/goal_pose --goal_reached_threshold 0.1 \
  --odom_topic /dog_odom --odom_twist_frame body --odom_timeout_s 0.5 \
  --command_min -0.3 -0.2 -0.4 --command_max 0.3 0.2 0.4 \
  --command_deadband 0.02 0.02 0.02 --command_slew_rate 0.3 0.2 0.4 \
  --command_lowpass_alpha 0.4 \
  --fall_tilt_threshold_deg 55 --fall_tilt_duration_s 0.15 \
  --safety_log_dir /workspace/instinct_nav_logs/dynamic_map_cross_model5999_safety \
  --safety_telemetry_hz 10 \
  --debug_dump_dir /workspace/instinct_nav_logs/dynamic_map_cross_model5999_real \
  --debug_dump_every_steps 5 --debug_dump_max_steps 100 \
  --nodryrun
```

等待终端 2 打印 cold-start 完成并进入 stand，确认站姿稳定和 safety log 正常，再发布目标。

#### 终端 3：目标管理器

```bash
sudo docker exec -it unitree.unitree-jetson-container bash
source /workspace/unitree-jetson-container/env.sh
source /workspace/unitree-jetson-container/instinct_onboard/instinct_venv/bin/activate
cd /workspace/unitree-jetson-container/instinct_nav
export PYTHONPATH=/workspace/unitree-jetson-container/instinct_nav/deploy:/workspace/unitree-jetson-container/instinct_nav/external/instinct_onboard:$PYTHONPATH

python deploy/scripts/goal_manager.py \
  --odom_topic /dog_odom \
  --world_goal_topic /instinct_nav/world_goal_point \
  --body_goal_topic /instinct_nav/goal_pose \
  --goal_frame odom \
  --goal_reached_threshold 0.5 \
  --stop_publish_count 10 \
  --publish_hz 10
```

该进程保持运行，持续把世界坐标目标转换为 body-frame 目标。

#### 遥操作：复用终端 1 查看坐标并发布目标

先查看机器人绝对坐标：

```bash
ros2 topic echo /dog_odom --once
```

读取 `pose.pose.position.x/y`。下面的目标是 `odom` 下的绝对坐标，不是相对位移。
若当前位置接近 `x=0,y=0`，首次发布前方 0.5m 目标：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'odom'}, point: {x: 0.5, y: 0.0, z: 0.0}}"
```

左右目标示例：

```bash
ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'odom'}, point: {x: 0.0, y: 0.5, z: 0.0}}"

ros2 topic pub --once /instinct_nav/world_goal_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'odom'}, point: {x: 0.0, y: -0.5, z: 0.0}}"
```

每次只发布一个目标，顺序建议为前进 0.5m、横向 0.5m、1.0m，再测试动态障碍。

#### 手柄和停止方式

- 不需要按 `L1`；收到目标后自动从 stand 切换到 parkour。
- `R1`：清零自主命令并返回 stand；该停止会锁存，必须重启终端 2 才能重新接受目标。
- `R2` 或 `L2`：电机断电急停并退出，机器人会变软，必须有吊架。
- `Ctrl-C`：正常停止进程，不代替物理急停。
- 到达目标后自动停止并返回 stand。
- watchdog/姿态保护触发后会锁存 fault，排除原因后必须重启节点。

测试后保留：

```text
/workspace/instinct_nav_logs/dynamic_map_cross_model5999_safety/
/workspace/instinct_nav_logs/dynamic_map_cross_model5999_real/
```

## 2026-08-15：DynamicMapCross RichTerrain 第一版

在现有导航高度场生成器上增量扩展，不替换原始 `MAZE_TERRAIN_CFG`，因此
`model_5999.pt` 对应的原任务和地形仍可复现。新增配置为 `RICH_MAZE_TERRAIN_CFG`：

- 单个 tile 从 `30 x 30 m` 扩大到 `40 x 40 m`，网格由 `15 x 15` 扩大到 `20 x 20`，单格仍为 `2 m`。
- 楼梯从固定 5 阶、固定 0.2 m 阶高，扩展为每个结构独立采样 `3-7` 阶和 `0.10-0.20 m` 阶高；阶高上限随 terrain difficulty 增长。
- 参考 InstinctLab parkour terrain 的 stairs-up/down、slope 和 discrete-obstacle 思路，在当前生成器中加入宽缓坡、低矮离散障碍和 `14-18 cm` 单级长条凸台，继续复用导航专用 valid/platform/spawn mask。
- 缓坡不再占用独立 tile，而是混入开放障碍和离散方块地形；单级凸台也混入这两类地形，每块生成 `2-5` 根，长度 `2-5 m`、宽度 `0.3-0.6 m`，方向为横向、纵向或 `±45°`。
- 地形比例：迷宫 20%、开放混合障碍 20%、随机楼梯 25%、方块混合障碍 20%、坑/沟槽 15%。
- RichTerrain episode 时长从 60 s 增加到 80 s，以适应扩大后的 start-goal 路径。
- DynamicMapCross 观测、网络、动态障碍课程和 frozen official parkour 低层策略保持不变。

新增任务：

```text
Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-v0
Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Dev-v0
Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Play-v0
```

dev smoke test：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Dev-v0 \
  --num_envs 32 \
  --max_iterations 1 \
  --run_name rich_terrain_smoke_test \
  --headless
```

可视化检查：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Play-v0 \
  --num_envs 20 \
  --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/model_5999.pt
```

正式训练（新 run，不覆盖原 baseline）：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-v0 \
  --num_envs 4096 \
  --max_iterations 6000 \
  --run_name g1_sru_dynamic_map_cross_rich_terrain_v1 \
  --headless \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav \
  --seed 72
```

已完成 32 env / 1 iteration smoke test，policy observation 为 `2896`、critic observation 为
`6033`，成功产出 `model_0.pt`。首次创建 16 个 40 m dev tile 时，terrain 随机生成耗时约
`0.91 s`，完整 scene 创建约 `59.25 s`。

### 交互进入 RichTerrain

只加载真实训练地形 mesh，不创建机器人和策略。打开全部六类地形：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py inspect-terrain \
  --difficulty 0.8 \
  --seed 72
```

单独检查楼梯或混合障碍：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py inspect-terrain \
  --terrain variable_stairs --difficulty 0.8 --seed 72

/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py inspect-terrain \
  --terrain blocks --difficulty 0.8 --seed 72
```

可选地形名：`maze`、`non_maze`、`variable_stairs`、`blocks`、`pits`。
在 Isaac Sim viewport 中按住鼠标右键并使用 `W/A/S/D` 移动，滚轮调整速度；关闭窗口退出。
该命令需要图形桌面，不要追加 `--headless`。

## 2026-08-16：RichTerrain v4、官方低层与新速度范围

本节取代上面的“RichTerrain 第一版”作为当前正式训练配置。第一版记录继续保留用于追踪
设计演化，但其中的 `40 x 40 m`、混入旧类别的斜坡/横条以及 80 s episode 均已撤销。

### 地形最终设计

- 原始 `MAZE_TERRAIN_CFG` 完全不变。
- RichTerrain 恢复原尺寸：每个 tile 为 `30 x 30 m`、内部网格为 `15 x 15`，仍使用 `6 x 30` 个 tile；episode 时长恢复继承原配置。
- 保留五个旧类别及其原有生成参数：`maze`、`non_maze`、`variable_stairs`、`blocks`、`pits`。
- 新增独立类别 `traversal_mix`，复杂穿越元素不再混入其他旧类别。
- 旧五类保持原来的 `20:20:25:20:15` 相对采样比例，统一缩放后占比为 `16% / 16% / 20% / 16% / 12%`；`traversal_mix` 独立占 `20%`。
- `traversal_mix` 同时生成楼梯/平台、宽缓坡、凸起横条和横条形洼地。
- 每组楼梯独立随机采样 `3-7` 阶；单阶高度受 difficulty 控制，difficulty=0.5 时约为 `0.10-0.15 m`，difficulty=1.0 时为 `0.10-0.20 m`；布局也从三种组合中随机选择。
- 凸起横条每块请求生成 `6-10` 根，高 `0.14-0.18 m`；横条洼地每块请求生成 `4-7` 根，深 `0.08-0.18 m`。二者方向均可为横向、纵向或 `±45°`。

地形俯视预览：

```text
artifacts/terrain/rich_terrain_traversal_mix_preview.png
```

交互检查新增类别：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py inspect-terrain \
  --terrain traversal_mix \
  --difficulty 0.8 \
  --seed 72
```

### 高层速度命令范围

高层 3 维网络输出先经过 `tanh`，再仿射映射到与官方 Parkour 低层训练分布一致的范围：

```text
vx: 0.0 to 1.0 m/s
vy: fixed 0.0 m/s
wz: -1.0 to 1.0 rad/s
```

因此高层原始零动作 `[0, 0, 0]` 映射为 `[0.5, 0, 0]`。训练 action bridge、部署
runtime 和 ONNX runtime helper 已同步采用该 contract。旧 checkpoint 的动作语义不同，v4 必须从零训练，
不能续接 RichTerrain v1/v2/v3。

### 实际使用的官方冻结低层

正式 task 的继承链最终设置 `backend_type="official_onnx"`，运行时实例化
`OfficialOnnxFrozenPolicyBackend`。它从官方 onboard ONNX 文件读取参数，重建为等价的 GPU
PyTorch 网络以支持 4096 环境批量推理，并将全部参数设置为 `requires_grad=False`：

```text
81b1af17d36ac1191e2654c35fb902d89c1efd96eea612c0a27d77c1edb5b61a  data&model/checkpoints/parkour_onboard_preview_stair/exported/actor.onnx
6da9f70e7f1cb3b37f02cac73fa16f6698afd427d9e44d833d345277a33b5b69  data&model/checkpoints/parkour_onboard_preview_stair/exported/0-depth_encoder.onnx
```

`senior_diagnostics.json` 中仍会打印 `artifacts/low_level/g1_parkour/model_30000.pt`，但它只是
通用 action cfg 的兼容字段。只有 `backend_type="checkpoint"` 才会加载该 PT 文件；当前
OfficialParkour task 不走该分支。

### Smoke test 与正式训练

32 env / 1 iteration smoke 已完成，RichTerrain mesh cooking、动态障碍物、官方冻结低层和一次
PPO 更新均正常，`terrain_fall=0`，并产出：

```text
logs/rsl_rl/g1_instinct_navigation_dynamic_map_cross_dev/2026-08-16_16-24-18_rich_terrain_speed_v4_smoke/model_0.pt
```

正式 v4 已从零启动。动态障碍课程保持固定在 iteration `3003` 切换：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-v0 \
  --num_envs 4096 \
  --max_iterations 6000 \
  --run_name g1_sru_dynamic_map_cross_rich_terrain_v4 \
  --headless \
  --wandb \
  --wandb_entity qqfqqf0612-wuhan-university \
  --wandb_project instinct_nav \
  --seed 72
```

当前 run 目录与启动日志：

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-16_16-28-39_g1_sru_dynamic_map_cross_rich_terrain_v4/
logs/launch/g1_sru_dynamic_map_cross_rich_terrain_v4.log
```

查看训练：

```bash
tail -f /workspace/instinct_nav/logs/launch/g1_sru_dynamic_map_cross_rich_terrain_v4.log
ps -ef | grep g1_sru_dynamic_map_cross_rich_terrain_v4 | grep -v grep
```


---

## 附:评估结果基线快照(原 RESULTS.md)

> 本节由原 RESULTS.md 于文档重构时并入,作为 v4 训练完成后的对比基线。
> 内容为迁移时点的原样快照;当前主线以 [README.md](./README.md) 为准。

# Results

This file records the current frozen baseline result for `instinct_nav`. Treat this checkpoint and evaluation as the project reference point before starting new ablations.

## Golden Baseline

- Run: `2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch`
- Checkpoint: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt`
- Evaluation JSON: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999.json`
- Evaluation episodes: `500`
- Evaluation envs: `256`
- Overall success rate: `0.902` (`451 / 500`)

The JSON was generated before the final task-id cleanup, so its `task` field still says `Isaac-Nav-PPO-G1-Instinct-Play-v0`. The current project task id for reproducing the same line is `Instinct-Nav-G1-SRU-Play-v0`.

## Terrain Results

| Terrain | Episodes | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| `maze` | 151 | 134 | 0.8874 |
| `non_maze` | 96 | 84 | 0.8750 |
| `pits` | 106 | 92 | 0.8679 |
| `stairs` | 147 | 141 | 0.9592 |

## Termination Rates

| Termination | Count | Rate |
| --- | ---: | ---: |
| `early_termination` | 444 | 0.888 |
| `base_contact` | 32 | 0.064 |
| `large_pitch_angle` | 24 | 0.048 |
| `time_out` | 5 | 0.010 |
| `terrain_fall` | 0 | 0.000 |

In this task, `early_termination` is mostly the normal success termination path: the episode ends once the robot reaches the navigation goal.

## Level Results

| Level | Episodes | Successes | Success Rate |
| --- | ---: | ---: | ---: |
| `level_0` | 79 | 66 | 0.8354 |
| `level_1` | 77 | 70 | 0.9091 |
| `level_2` | 90 | 87 | 0.9667 |
| `level_3` | 86 | 74 | 0.8605 |
| `level_4` | 75 | 67 | 0.8933 |
| `level_5` | 93 | 87 | 0.9355 |

## Artifacts

- Failure GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/failure_gifs/`
- Success/failure eval GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/eval_gifs_success_and_failure/`
- Stairs success GIFs: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_success_gifs/`
- Play video: `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/play/rl-video-step-0.mp4`

## Reproduce Training

Foreground training:

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

Background training:

```bash
nohup /workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name instinct_nav_g1_sru_independent_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    > /workspace/instinct_nav/logs/instinct_nav_g1_sru_independent_v1.out 2>&1 &
```

## Reproduce Evaluation

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 256 \
    --num_episodes 500 \
    --headless \
    --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_model_2999_reproduce.json \
    --save_failure_gifs \
    --max_failure_gifs 10
```

## Record Success And Failure GIFs

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 64 \
    --num_episodes 100 \
    --headless \
    --save_success_gifs \
    --save_failure_gifs \
    --max_success_gifs 10 \
    --max_failure_gifs 10 \
    --json_out /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/eval_summary_success_failure_gifs_reproduce.json
```

## Record Stairs Success GIFs

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
    --task Instinct-Nav-G1-SRU-Play-v0 \
    --checkpoint /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/model_2999.pt \
    --num_envs 1 \
    --steps 0 \
    --headless \
    --bev_follow \
    --bev_height_m 22.0 \
    --require_terrain_name stairs \
    --min_initial_distance 6.0 \
    --max_resample_attempts 1000 \
    --gif_count 5 \
    --gif_length 500 \
    --gif_frame_ms 80 \
    --gif_dir /workspace/instinct_nav/logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-18_18-37-50_senior_fullcopy_g1_instinct_v5_scratch/videos/stairs_success_gifs
```

## Next Work

Do not change the main training path before saving a clean comparison baseline. The next work should be:

- Keep `model_2999.pt` as the frozen golden checkpoint.
- Export a compact table and several representative GIFs for reports.
- Add a method document explaining the high-level SRU policy, frozen low-level Instinct policy, observations, depth encoder, height encoder, action bridge, reward, and termination design.
- Run new experiments only as named ablations, not as silent edits to the golden baseline.


## Scene Token Ablations

These results compare the current G1 SRU navigation baseline against the scene-token ablations. The token ablations use 1000-episode normal evaluation with `seed=72`; the original golden baseline is the earlier 500-episode senior/fullcopy evaluation.

| Run | Episodes | Success Rate | Maze | Non Maze | Pits | Stairs | Base Contact | Large Pitch | Time Out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `senior_fullcopy_g1_instinct_v5_scratch` | 500 | 0.902 | 0.8874 | 0.8750 | 0.8679 | 0.9592 | 0.064 | 0.048 | 0.010 |
| `g1_sru_scene_token_query_v1` | 1000 | 0.875 | 0.8620 | 0.8621 | 0.9028 | 0.8768 | 0.085 | 0.086 | 0.014 |
| `g1_sru_scene_token_risk_query_v1` | 1000 | 0.892 | 0.9009 | 0.8911 | 0.8995 | 0.8776 | 0.065 | 0.078 | 0.023 |
| `g1_sru_scene_token_risk_no_goal_query_v1` | 1000 | 0.916 | 0.9183 | 0.9167 | 0.8900 | 0.9310 | 0.048 | 0.057 | 0.023 |

Current best token checkpoint:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-25_17-29-52_g1_sru_scene_token_risk_no_goal_query_v1/model_2999.pt
```

Current best token evaluation summary:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-25_17-29-52_g1_sru_scene_token_risk_no_goal_query_v1/eval_summary_model_2999_1000ep.json
```

Main conclusion: scene tokens should describe local scene/risk rather than duplicate goal-relative information in every token. The v3 no-goal risk token improves overall success rate and reduces torso contact relative to v1/v2.

Important caveat: fixed-case manifest evaluation is currently not trusted for scene-token policies. A v1 fixed-case run showed an abnormal early success rate, while normal v1 re-evaluation returned to the expected range. Use normal evaluation for the current comparison until `case_manifest_in` reset/cache handling is audited.

## DynamicMapCross Static vs Dynamic Obstacles

The final `g1_sru_dynamic_map_cross_corridor_v1/model_5999.pt` checkpoint was evaluated for
1000 episodes with 256 environments and seed 72. The dynamic run force-enabled corridor
obstacles from environment step 0; the static run left the obstacle curriculum disabled.

| Condition | Success | Maze | Non Maze | Pits | Stairs | Base Contact | Large Pitch | Obstacle Collision | Time Out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static obstacles | 0.9260 | 0.9135 | 0.9320 | 0.9216 | 0.9369 | 0.0480 | 0.0510 | 0.0000 | 0.0170 |
| Dynamic obstacles | 0.7570 | 0.7562 | 0.7663 | 0.7385 | 0.7664 | 0.1410 | 0.0630 | 0.1030 | 0.0100 |

Dynamic obstacles reduce success by 16.9 percentage points. Obstacle collision (10.3%) and
the increase in base contact (4.8% to 14.1%) are the main observed regressions. These are
normal evaluations rather than paired fixed-case evaluations, so terrain episode counts differ
slightly between conditions.

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/eval_static_model_5999_1000ep.json
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-08-12_22-42-54_g1_sru_dynamic_map_cross_corridor_v1/eval_dynamic_model_5999_1000ep.json
```
