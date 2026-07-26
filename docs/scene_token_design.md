# Scene Token Design Notes

本文档整理 instinct_nav 当前 scene token 路线：为什么引入 token、当前 v1/v2 怎么生成、token 如何进入 SRU actor/critic、训练现象如何解释，以及后续 Temporal / Mamba-SSM 方向。

核心实现文件：

- [observations.py](../source/instinct_nav/instinct_nav/navigation/mdp/observations.py)
- [navigation_env_cfg.py](../source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)
- [rsl_rl_cfg.py](../source/instinct_nav/instinct_nav/navigation/config/g1_instinct/agents/rsl_rl_cfg.py)
- [actor_critic_sru.py](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [task registration](../source/instinct_nav/instinct_nav/navigation/config/g1_instinct/__init__.py)

## 1. Motivation

原 SRU navigation 的核心是：

```text
proprio + depth feature -> attention encoder -> SRU memory -> MLP -> high-level velocity command
```

它依赖 SRU 的时间记忆和 depth encoder 的视觉特征完成导航。我们在 G1 人形机器人上观察到，单纯复现 senior baseline 后，整体成功率可以很高，但如果要进一步做场景理解，直接把所有信息压进 depth feature 和 SRU hidden state 不够显式。

因此引入 scene token 的目的不是替代 SRU，而是给高层策略一个显式的局部场景接口：

```text
local scene memory / geometric cues -> fixed scene tokens -> query attention -> actor / critic
```

这个接口将来可以接入更强的 3D token，例如 PQ3D / object token / traversability token，但当前阶段先用低成本、actor 可观测的 depth / visibility 信息构造 token，保证训练开销可控。

## 2. Current Scene Token Source

当前 token 不是 PQ3D token，也不是 3D segmentation token。

它来自：

```text
raycast_camera depth
    -> local_visibility_map
    -> 64 x 64 robot-centric BEV visibility memory
    -> 8 x 8 patch tokens
```

也就是说，每个 token 对应局部 BEV 网格里的一个 patch。默认：

```text
visibility map: 64 x 64
token grid:     8 x 8
num tokens:     64
```

`local_visibility_map` 是机器人坐标系下的局部显式记忆。它表示哪些区域被当前/历史 depth ray 看见过，不是 occupancy map，也不是完整高度图。

## 3. SceneToken v1

v1 函数：

- [scene_tokens_from_visibility_map](../source/instinct_nav/instinct_nav/navigation/mdp/observations.py)

v1 token 逻辑 shape：

```text
[num_envs, 64, 6]
```

ObservationManager 拼接前会 flatten 成：

```text
[num_envs, 384]
```

每个 token 的 6 维为：

```text
[x_center, y_center, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal]
```

含义：

- `x_center`: token patch 中心在机器人局部坐标系下的前后位置，归一化到约 `[-1, 1]`。
- `y_center`: token patch 中心在机器人局部坐标系下的左右位置，归一化到约 `[-1, 1]`。
- `visible_ratio`: 该 patch 内可见栅格比例。
- `dist_to_goal`: 该 patch 中心到目标方向的距离，按 map size 归一化。
- `cos_to_goal`: patch 指向目标方向的 cos 分量。
- `sin_to_goal`: patch 指向目标方向的 sin 分量。

v1 task：

```text
Instinct-Nav-G1-SRU-SceneToken-Query-v0
Instinct-Nav-G1-SRU-SceneToken-Query-Dev-v0
Instinct-Nav-G1-SRU-SceneToken-Query-Play-v0
```

训练/评估现象：

```text
scene-token v1 1000ep eval:
success_rate      0.875
base_contact      0.085
large_pitch_angle 0.086
stairs            0.877

senior baseline 500ep eval:
success_rate      0.902
base_contact      0.064
large_pitch_angle 0.048
stairs            0.959
```

v1 的关键信号是：pits 表现有提升，但 stairs / 姿态稳定性变差。说明 token 信息可能让 actor 更积极地朝目标走，但没有足够表达地形风险。

## 4. SceneToken v2: Risk Token + Critic Token

v2 目标是同时处理两个问题：

1. actor 有 token、critic 无 token，导致 PPO value 估计可能信息不足。
2. v1 token 偏导航方向，缺少局部地形风险信息。

v2 函数：

- [scene_tokens_risk_from_visibility_map](../source/instinct_nav/instinct_nav/navigation/mdp/observations.py)

v2 token 逻辑 shape：

```text
[num_envs, 64, 9]
```

ObservationManager flatten 后：

```text
[num_envs, 576]
```

每个 token 的 9 维为：

```text
[x_center, y_center,
 visible_ratio,
 dist_to_goal, cos_to_goal, sin_to_goal,
 depth_mean, depth_edge, frontier]
```

新增的 3 维都是 actor-safe 信息，来自 actor 同一路 `raycast_camera` / visibility memory，不使用 critic-only `height_scanner_critic`。

### 4.1 depth_mean

来源：当前 noisy depth sample。

处理流程：

```text
depth image
    -> valid depth mask
    -> normalize to [0, 1]
    -> adaptive_avg_pool2d to 8 x 8
```

它表示每个 patch 的平均远近。无效 depth 会填成 `1.0`，也就是偏远/未知。

### 4.2 depth_edge

来源：归一化 depth 图的局部差分。

处理流程：

```text
grad_x = abs(depth[:, :, 1:] - depth[:, :, :-1])
grad_y = abs(depth[:, 1:, :] - depth[:, :-1, :])
depth_edge = avg_pool(grad_x + grad_y) -> 8 x 8
```

它近似表达局部深度突变。台阶边缘、坑边、障碍物边缘通常会让这个值变大。

### 4.3 frontier

来源：visibility map，而不是 depth。

处理流程：

```text
visibility_binary = visibility > 0.5
unknown = 1 - visibility_binary
frontier = abs spatial difference of unknown map
frontier -> avg_pool to 8 x 8
```

它表示已知区域和未知区域的边界。frontier 高的地方通常意味着信息边界，可能对应尚未观测完整的区域。

v2 task：

```text
Instinct-Nav-G1-SRU-SceneTokenRisk-Query-v0
Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Dev-v0
Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Play-v0
```

## 5. Token Enters ActorCriticSRU

当前 v1/v2 都是 query fusion，不是简单拼接进 SRU recurrent input。

actor 路径：

```text
actor obs
    -> split proprio / depth / scene_tokens
    -> depth attention encoder
    -> concat(depth_feature, proprio)
    -> actor SRU memory
    -> h_t
    -> query = MLP([h_t, proprio])
    -> scene tokens -> token projector + position embedding
    -> cross attention(query, tokens)
    -> scene feature
    -> concat(h_t, scene feature)
    -> actor MLP
    -> velocity command action
```

关键代码：

- [_extract_actor_scene_tokens](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [_query_actor_scene_tokens](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [process_actor_input](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)

critic v2 路径：

```text
critic obs
    -> split proprio / time / height / depth / scene_tokens
    -> height attention encoder
    -> depth attention encoder
    -> concat(height_feature, depth_feature, proprio)
    -> critic SRU memory
    -> h_t^critic
    -> query = MLP([h_t^critic, proprio])
    -> scene tokens -> shared token projector + position embedding
    -> critic cross attention(query, tokens)
    -> scene feature
    -> concat(h_t^critic, scene feature, time_embedding)
    -> critic MLP
    -> value
```

关键代码：

- [_extract_critic_scene_tokens](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [_query_critic_scene_tokens](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [process_critic_input](../source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)

## 6. Why Critic Token Matters

PPO 中：

```text
actor: 选择动作
critic: 估计状态价值
advantage = return - value
```

如果 actor 看得到 scene token，而 critic 看不到，会出现：

```text
critic 看来是同一个状态
actor 因 token 不同采取不同动作
未来回报差异很大
```

这会让 critic 的 value 估计更 noisy，从而让 advantage 也更 noisy。

需要注意：asymmetric actor-critic 本身不一定坏。常见的 privileged critic 是：

```text
critic 看得比 actor 多
actor 看得少
```

但 v1 的方向是：

```text
actor 看 token
critic 不看 token
```

这属于 critic 信息不足。v2 让 critic 也看 token，目的是让 value function 能解释 token 造成的策略/回报差异。

## 7. Goal-relative Token Features: Open Question

v1/v2 当前都包含：

```text
dist_to_goal, cos_to_goal, sin_to_goal
```

这三个特征有争议。

保留它们的理由：

- token 不只是 scene patch，还带有当前任务目标关系。
- query attention 更容易知道哪些 patch 与目标方向相关。

可能的副作用：

- actor 本体观测中已经有目标位置/goal command，token 内再编码目标可能重复。
- token 带强 goal bias，可能鼓励策略更激进地朝目标走。
- 对 stairs 这类地形，最近目标方向不一定是动态上最稳定的方向。
- 如果将来接 PQ3D，scene token 更应该表达 scene 本身，而不是把 task goal 混进 token。

因此建议后续单独做一个 no-goal token ablation：

```text
SceneTokenRiskNoGoal
[x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier]
```

对应逻辑：

```text
scene tokens: 描述局部场景是什么
SRU/proprio/goal query: 决定当前目标下应该关注哪里
```

## 8. Visibility Map vs Scene Token

Visibility-map query fusion：

```text
visibility map -> CNN -> learned tokens -> cross attention
```

Scene-token query fusion：

```text
visibility/depth-derived hand-crafted tokens -> MLP projector -> cross attention
```

二者区别：

- visibility map 分支让 CNN 自己从二值 map 学 token。
- scene token 分支显式给出 patch 坐标、可见比例、目标关系、风险特征。
- visibility map 原分支是 actor-only，critic 看不到 visibility map。
- scene token v2 是 actor + critic 都看 token。

## 9. Temporal Token Direction

当前 token 是每步当前局部 memory 的静态 token。后续可以做 temporal token：

```text
current tokens X_t
previous tokens X_{t-1}
robot delta pose DeltaT
```

最简单版本：

```text
[X_t, X_{t-1}] -> 128 tokens
```

更合理版本：

```text
M_{t-1}
    -> SE(2) warp by delta pose
    -> aligned memory Mprime_{t-1}
    -> gated update with X_t
    -> M_t
```

这里要注意：当前 `local_visibility_map` 本身已经做了机器人坐标系下的局部 memory / warp / update，因此如果直接把前后两个 accumulated visibility token 都喂给网络，可能信息冗余。更有价值的是区分：

```text
instant observation token
persistent memory token
frontier / risk token
```


## 10. Current Recommendation

当前优先级：

1. 先完整评估 `SceneTokenRisk-Query-v0`，重点看 `success_rate / stairs / base_contact / large_pitch_angle`。
2. 如果 v2 仍然让 `large_pitch_angle` 和 `base_contact` 高于 senior baseline，下一版优先做 `SceneTokenRiskNoGoal`。
3. 如果 no-goal token 稳定性改善，再考虑 temporal token。
4. Mamba-SSM 更适合作为后续论文创新模块，不建议在当前复现主线还未稳定前替换 SRU 主干。


## 12. SceneTokenRiskNoGoal v3

v3 是针对 v2 中 per-token goal-relative features 的 ablation。

v3 函数：

- [scene_tokens_risk_no_goal_from_visibility_map](../source/instinct_nav/instinct_nav/navigation/mdp/observations.py)

v3 token 逻辑 shape：

```text
[num_envs, 64, 6]
```

ObservationManager flatten 后：

```text
[num_envs, 384]
```

每个 token 的 6 维为：

```text
[x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier]
```

相比 v2，去掉：

```text
dist_to_goal, cos_to_goal, sin_to_goal
```

目标信息仍保留在正常的 `target_position / goal_command_body` observation 中，并通过 `h_t + proprio` query 去决定关注哪些 scene tokens。这样 token 本身只描述局部场景和风险，减少 per-token goal bias 对楼梯稳定性的潜在干扰。

v3 task：

```text
Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-v0
Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-Dev-v0
Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-Play-v0
```

训练命令：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --run_name g1_sru_scene_token_risk_no_goal_query_v1 \
    --headless \
    --wandb \
    --wandb_entity qqfqqf0612-wuhan-university \
    --wandb_project instinct_nav \
    --seed 72
```


## 13. V3 Evaluation Result

`SceneTokenRiskNoGoal` finished training and normal 1000-episode evaluation.

Run directory:

```text
logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-25_17-29-52_g1_sru_scene_token_risk_no_goal_query_v1/
```

Evaluation summary:

```text
eval_summary_model_2999_1000ep.json
```

Result:

```text
episodes:          1000
successes:         916
success_rate:      0.916
base_contact:      0.048
large_pitch_angle: 0.057
time_out:          0.023
```

Terrain success:

```text
maze:     0.9183
non_maze: 0.9167
pits:     0.8900
stairs:   0.9310
```

Ablation interpretation:

```text
v1 scene-token:    0.875
v2 risk-token:     0.892
v3 risk-no-goal:   0.916
```

The result supports the current design choice: scene tokens should encode local geometry, visibility, and risk, while target information should remain in the normal goal observation and enter the scene-token branch through the query. Duplicating goal-relative features inside every token appears to add a harmful bias for humanoid navigation, especially on stairs.
