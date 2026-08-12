# Dynamic Obstacle + ScoreVis 最新问题汇总

日期：2026-08-04

范围：当前最新动态避障训练链路，重点检查 `ScoreVis` 分数图、动态障碍课程、奖励/终止、训练与部署语义一致性。

更新：2026-08-08 复查最新提交 `df59210 将训练前期物体移出场外`。该提交已经修复“静态阶段障碍可能污染训练”的核心问题：未启用动态障碍时，`obstacle_proximity`、`obstacle_approach_penalty`、`obstacle_collision` 分别显式返回 `0/0/False`，障碍也从 `12m` 外移动到 `40m` 外。

## 当前主线

当前动态避障任务：

```bash
Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-v0
```

该任务继承自：

```text
ScoreVis + shared lowcam + official frozen parkour
```

也就是说：

- 低层 parkour policy 仍然冻结。
- 高层 SRU 仍然输出导航速度 command。
- 高层 scene token 仍为 `64 x 6 = 384` 维。
- 新增变化是 visibility map 从二值图变为连续分数图。
- 动态避障阶段新增脚本运动刚体障碍、避障 reward 和碰撞 termination。

ScoreVis token 语义：

```text
[x_center, y_center, visibility_score, depth_mean, depth_edge, frontier]
```

网络输入维度没有变，但数值语义变了，因此旧二值 visibility checkpoint 不能直接复用，必须重新训练。

## 问题 1：动态障碍接近速度符号疑似反了

文件：

```text
source/instinct_nav/instinct_nav/navigation/mdp/dynamic_obstacles.py
```

位置：

```python
def closing_speed_to_robot(self) -> torch.Tensor:
    ...
    to_obs = self.positions_w[:, :, :2] - root_xy
    unit = to_obs / dist.unsqueeze(-1)

    obs_vel = self.dirs * self.speeds.unsqueeze(-1)
    robot_vel = robot.data.root_lin_vel_w[:, :2].unsqueeze(1)
    rel_vel = obs_vel - robot_vel
    closing = torch.sum(rel_vel * unit, dim=-1)
    return closing.clamp_min(0.0).max(dim=-1).values
```

### 问题解释

`unit` 的方向是：

```text
robot -> obstacle
```

如果障碍物正朝机器人运动，那么相对速度 `rel_vel = obs_vel - robot_vel` 在这个方向上的投影通常是负数，因为障碍物在往 `-unit` 方向移动。

但当前代码使用：

```python
closing = dot(rel_vel, unit)
```

这会导致真正接近机器人时可能得到负值，然后被：

```python
clamp_min(0.0)
```

清零。

### 影响

`obstacle_approach_penalty` 可能没有惩罚真正危险的接近行为。

训练时策略主要依赖 `obstacle_proximity` 和 `obstacle_collision`，但缺少对“正在快速靠近”的提前惩罚，动态避障学习会更被动。

### 建议修改

将接近速度改成距离缩短速度：

```python
closing = -torch.sum(rel_vel * unit, dim=-1)
```

或者显式写成：

```python
closing = torch.sum((robot_vel - obs_vel) * unit, dim=-1)
```

建议同时加一个最小单元测试：

- 障碍在机器人前方。
- 障碍朝机器人移动。
- `closing_speed_to_robot()` 应返回正数。
- 障碍远离机器人移动。
- `closing_speed_to_robot()` 应返回 0 或接近 0。

### 当前状态

截至提交 `df59210`，该问题仍存在。最新代码仍为：

```python
closing = torch.sum(rel_vel * unit, dim=-1)
```

建议优先修改。

## 问题 2：课程切换参数名和实际单位不一致

文件：

```text
source/instinct_nav/instinct_nav/navigation/mdp/dynamic_obstacles.py
source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py
README.md
```

当前配置：

```python
obstacle_max_switch_iteration: int = 3000
obstacle_switch_check_interval: int = 100
```

实际判断：

```python
step = int(self.env.common_step_counter)
...
if mean_success >= self.auto_switch_success or step >= self.max_switch_iteration:
    ...
```

### 问题解释

代码比较的是：

```text
env.common_step_counter
```

这通常是环境 step，不是 PPO iteration。

当前 runner 继承自 Go2/G1 PPO 配置：

```python
num_steps_per_env = 16
```

所以：

```text
3000 env steps ≈ 187.5 PPO iterations
```

但 README 当前描述是“最迟第 3000 轮强制开启动态”，这更像 PPO iteration。

### 影响

如果原计划是前 3000 PPO iterations 先训练静态 ScoreVis，再训练后 3000 PPO iterations 动态避障，那么当前代码会过早切换，大约在第 188 个 PPO iteration 就打开动态障碍。

这可能导致：

- 静态基础导航还没稳定，动态障碍提前加入。
- success 曲线前期震荡更大。
- 误以为“3000 轮后才切换”，但实际训练很早进入动态阶段。

### 建议修改

有两种合理方案，二选一即可。

方案 A：承认该参数单位是 env step，重命名并更新文档。

```python
obstacle_max_switch_step: int = 3000
obstacle_switch_check_interval_steps: int = 100
```

方案 B：如果确实想按 PPO iteration 切换，则换算成 env step。

例如希望第 3000 PPO iteration 切换：

```text
max_switch_step = 3000 * num_steps_per_env = 3000 * 16 = 48000
```

此时配置应接近：

```python
obstacle_max_switch_iteration: int = 48000
```

但更建议直接改名为 `obstacle_max_switch_step`，避免再次误解。

### 当前状态

截至提交 `df59210`，该问题仍存在。

当前 README 仍写：

```text
obstacle_max_switch_iteration = 3000
最迟在第 3000 轮强制开启动态
```

但代码仍使用：

```python
step = int(self.env.common_step_counter)
```

因此当前实际语义仍是：

```text
3000 env steps
```

不是：

```text
3000 PPO iterations
```

如果当前训练意图是“先静态训练约 3000 PPO iterations”，需要把阈值改到约 `48000`，或者把参数和 README 明确改成 step 单位。

## 问题 3：静态阶段障碍污染训练的问题已基本修复

文件：

```text
source/instinct_nav/instinct_nav/navigation/mdp/dynamic_obstacles.py
source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py
```

原问题：

早期实现中，动态障碍课程未启用时，障碍只是停在机器人附近较远位置，奖励/终止函数仍会根据障碍位置计算。即使障碍速度为 0，也可能带来以下风险：

- 静态阶段 `obstacle_proximity` 不完全为 0。
- 如果机器人走到驻车障碍附近，可能触发 `obstacle_collision`。
- 静态阶段不再严格等价于普通 ScoreVis 训练。

最新提交 `df59210` 已做如下修复：

```python
if not manager.enabled:
    return torch.zeros(env.num_envs, device=env.device)
```

```python
if not manager.enabled:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
```

并且：

```python
self.park_offset_m = 40.0
```

同时静态阶段不再根据地形高度查询障碍 z，而是固定：

```python
positions[:, :, 2] = 0.5
```

配置侧还设置了：

```python
disable_gravity=True
```

### 当前状态

该问题已经基本修复。

仍建议做一次 dev smoke 验证：

- 动态障碍未启用时，`obstacle_proximity` 均值为 0。
- 动态障碍未启用时，`obstacle_approach_penalty` 均值为 0。
- 动态障碍未启用时，`obstacle_collision` 为 0。
- 切换后这些指标开始正常变化。

## 问题 4：README 当前有编码显示风险

文件：

```text
README.md
```

现象：

在 Windows PowerShell 用部分命令读取 README 最新段落时，ScoreVis 和 Dynamic Obstacle 部分可能显示乱码。

### 影响

这可能只是本地 PowerShell 编码问题，也可能是文件内容曾经被错误编码写入。需要以 GitHub 页面或 UTF-8 编辑器显示为准。

### 建议检查

在 GitHub 页面或 VS Code UTF-8 模式下打开 README。

如果 GitHub 显示正常，可以不处理。

如果 GitHub 也乱码，应恢复或重写 README 中 2026-08-03/2026-08-04 两段中文记录。

## 问题 5：ScoreVis 训练、部署必须显式保持同一个 decay

涉及文件：

```text
source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py
deploy/instinct_nav_onboard/config.py
deploy/instinct_nav_onboard/onnx_high_level_runtime.py
deploy/instinct_nav_onboard/obs_builder.py
deploy/mujoco/g1_sru_navigation_sim2sim.py
```

训练侧 ScoreVis 当前设置：

```python
self.observations.policy.scene_tokens.params["decay"] = 0.995
self.observations.critic.scene_tokens.params["decay"] = 0.995
```

部署侧默认值：

```python
visibility_decay: float = 1.0
```

### 问题解释

`visibility_decay=1.0` 是为了兼容旧二值 visibility checkpoint。

但 ScoreVis checkpoint 训练时使用的是：

```text
decay = 0.995
```

因此部署 ScoreVis 权重时，必须显式传：

```bash
--visibility_decay 0.995
```

### 影响

如果部署 ScoreVis checkpoint 时忘记传 `--visibility_decay 0.995`，网络输入维度不会错，但 token 数值分布会变回近似二值图，造成训练/部署 observation distribution mismatch。

这类错误不会直接报 shape error，但会导致真机或 MuJoCo 表现明显变差。

### 建议修改

导出的 ONNX manifest 里应记录训练时的 visibility decay，并让 runtime 在加载 ScoreVis checkpoint 时自动使用 manifest 中的值。

更稳妥的方式：

- manifest 写入 `visibility_decay=0.995`。
- runtime 如果命令行没有显式覆盖，则使用 manifest 值。
- runtime 打印最终生效值。
- 如果 manifest 是 ScoreVis，但最终 `visibility_decay=1.0`，给出 warning 或直接报错。

## 问题 6：动态障碍课程缺少独立日志指标

当前代码会打印：

```text
[DynamicObstacleManager] Switching curriculum at step ...
[DynamicObstacleManager] Dynamic obstacles ENABLED ...
```

但训练指标中不一定有明确 scalar 表示：

- 当前是否已启用动态障碍。
- 当前 obstacle min distance。
- 当前 obstacle collision rate。
- 当前 obstacle approach penalty 均值。
- 静态阶段和动态阶段 success 是否分开统计。

### 影响

只看 overall success 容易误判。

如果动态阶段失败增加，可能不知道是：

- 障碍看不到。
- 接近速度 reward 无效。
- 碰撞 termination 太严。
- 动态课程切换太早。
- 低层 parkour 对突然减速/侧移不稳定。

### 建议增加指标

建议至少记录：

```text
Metrics/dynamic_obstacles/enabled
Metrics/dynamic_obstacles/min_dist_mean
Metrics/dynamic_obstacles/collision_rate
Episode_Reward/obstacle_proximity
Episode_Reward/obstacle_approach_penalty
Episode_Termination/obstacle_collision
```

如果 logger 已经自动记录 reward/termination，仍建议补一个 `enabled` 或从日志中明确标出切换点。

## 问题 7：动态障碍 eval 还需要单独设计

当前 README 记录里已经说明：

```text
待办：动态场景 eval manifest、MuJoCo 移动 box 扩展、真机侧同步。
```

### 影响

动态避障不能只用原来的静态导航 eval 判断。

需要同时看：

- 到点成功率。
- 与动态障碍碰撞率。
- 近障最小距离。
- 是否因为绕障导致 timeout 上升。
- 是否为了避障牺牲楼梯/复杂地形稳定性。

### 建议 eval 指标

动态避障 eval 建议单独输出：

```text
success_rate
obstacle_collision_rate
mean_min_obstacle_distance
timeout_rate
base_contact_rate
large_pitch_angle_rate
goal_distance_final_mean
```

建议固定几类场景：

- 正面横穿障碍。
- 侧向接近障碍。
- 障碍靠近目标点。
- 多障碍交叉移动。
- 楼梯/高低地形上动态障碍。

## 问题 8：动态障碍是否真实进入相机 raycast 需要继续验证

配置侧已经做了：

```python
track_mesh_transforms=True
self.scene.camera.mesh_prim_paths = list(self.scene.camera.mesh_prim_paths) + extra_targets
```

这说明设计上希望 shared lowcam 能看到移动障碍。

### 风险

动态刚体位置每 step 更新后，相机 raycast 是否使用最新 mesh transform，需要在 IsaacLab 运行中确认。

如果 transform 同步滞后或 mesh target 没生效，策略在深度图/ScoreVis 中看不到动态障碍，只能通过碰撞/奖励被动学习。

### 建议验证

做一个小规模 play/smoke：

- 放一个障碍在相机正前方。
- 导出或打印 depth min。
- 障碍移动时 depth min 应同步变化。
- ScoreVis token 中 `visibility_score/depth_mean/depth_edge/frontier` 应随障碍移动变化。

## 优先级建议

P0：

- 修正 `closing_speed_to_robot()` 的接近速度符号。
- 明确 `obstacle_max_switch_iteration` 的单位，避免课程过早切换。
- 保持 `df59210` 已修复的静态阶段 reward/termination gating，不要回退。

P1：

- 给动态障碍加最小单元测试或 smoke test。
- 增加动态障碍关键 TensorBoard/WandB 指标。
- 确认 README 编码是否只是在 PowerShell 读取时乱码。
- 验证 `disable_gravity=True` 的动态阶段行为是否符合训练假设。

P2：

- manifest 自动记录并应用 `visibility_decay`。
- 建立动态避障专用 eval manifest。
- MuJoCo 和真机侧同步动态障碍/移动 box 验证流程。

## 建议下一步

先不要急着长时间训练。建议先做两个小修：

1. 修接近速度符号。
2. 把课程切换参数单位改清楚。

然后用 dev 任务跑 1 到 5 个 iteration，检查：

```text
obstacle_proximity
obstacle_approach_penalty
obstacle_collision
DynamicObstacleManager Switching curriculum
```

这些指标正常后，再继续正式 6000 iteration 动态避障训练。