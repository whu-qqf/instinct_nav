# instinct_nav Code Reading Guide

本文档按实际执行链路梳理 `instinct_nav` 的核心代码：从观测值获取、预处理、拼接，到 `ActorCriticSRU`、SRU 循环和 MLP 输出。

## 1. 总体执行链路

训练入口：

```text
scripts/train_senior.py
  -> load task cfg
  -> gym.make(env)
  -> LegacyRslRlVecEnvWrapper
  -> OnPolicyRunner
  -> PPO rollout
  -> ActorCriticSRU.act / ActorCriticSRU.evaluate
```

核心文件：

- [task 注册](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/__init__.py)
- [G1 env 配置](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)
- [NavigationEnvCfg / ObservationsCfg](./source/instinct_nav/instinct_nav/navigation/navigation_env_cfg.py)
- [观测函数 observations.py](./source/instinct_nav/instinct_nav/navigation/mdp/observations.py)
- [OnPolicyRunner](./source/rsl_rl/rsl_rl/runners/on_policy_runner.py)
- [PPO](./source/rsl_rl/rsl_rl/algorithms/ppo.py)
- [ActorCriticSRU](./source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)
- [LSTM_SRU / SRU cell](./source/rsl_rl/rsl_rl/networks/sru_memory/lstm_sru.py)

## 2. Task 注册

正式 task 注册在：

[task 注册](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/__init__.py)

正式 task id：

```text
Instinct-Nav-G1-SRU-v0
Instinct-Nav-G1-SRU-Dev-v0
Instinct-Nav-G1-SRU-Play-v0
```

旧 `Isaac-Nav-PPO-G1-Instinct-*` task id 已删除，不再作为兼容 alias 注册。

训练时推荐使用：

```bash
--task Instinct-Nav-G1-SRU-v0
```

## 3. 观测配置在哪里定义

G1 专用配置在：

[G1 env 配置](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)

其中 `G1InstinctNavigationEnvCfg` 继承：

[NavigationEnvCfg](./source/instinct_nav/instinct_nav/navigation/navigation_env_cfg.py)

高层 actor / critic 观测主要定义在父类：

[ObservationsCfg.PolicyCfg / CriticCfg](./source/instinct_nav/instinct_nav/navigation/navigation_env_cfg.py)

低层 InstinctLab policy 观测定义在：

[InstinctLowLevelPolicyCfg](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)

## 4. Actor policy 观测

`ObservationsCfg.PolicyCfg` 是高层 actor 输入。

当前 actor policy obs：

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

对应配置：

```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel_delayed, noise=Unoise(n_min=-0.2, n_max=0.2))
base_ang_vel = ObsTerm(func=mdp.base_ang_vel_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
projected_gravity = ObsTerm(func=mdp.projected_gravity_delayed, noise=Unoise(n_min=-0.1, n_max=0.1))
last_action = ObsTerm(func=mdp.last_action)
target_position = ObsTerm(func=mdp.generated_commands_reshaped_delayed, ...)
depth_image = ObsTerm(func=mdp.depth_image_noisy_delayed, ...)
```

要点：

- actor 看到的是 delayed / noisy 的导航观测。
- `projected_gravity` 是机器人 base 坐标系下的重力方向，相当于姿态水平仪。
- `target_position` 来自 `robot_goal` command，并经过 delay/noise。
- `depth_image` 不是 raw depth，而是 depth encoder 后的 feature。

## 5. Critic 观测

`ObservationsCfg.CriticCfg` 是 value function 输入。

当前 critic obs：

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

对应配置：

```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
projected_gravity = ObsTerm(func=mdp.projected_gravity)
last_action = ObsTerm(func=mdp.last_action)
target_position = ObsTerm(func=mdp.generated_commands_reshaped, ...)
time_normalized = ObsTerm(func=mdp.time_normalized, ...)
height_scan_critic = ObsTerm(func=mdp.height_scan_feat, ...)
depth_image = ObsTerm(func=mdp.depth_image_prefect, ...)
```

要点：

- critic 使用 cleaner / privileged 信息。
- critic 有 `height_scan_critic`，actor 没有。
- critic 用 perfect depth，actor 用 noisy delayed depth。
- actor 和 critic 的 proprioceptive 部分都保持 16 维。

## 6. Depth 预处理

主要函数在：

[observations.py](./source/instinct_nav/instinct_nav/navigation/mdp/observations.py)

Actor 使用：

```python
depth_image_noisy_delayed(env, sensor_cfg)
```

Critic 使用：

```python
depth_image_prefect(env, sensor_cfg)
```

流程：

```text
raycast_camera 原始 depth: [B, H, W], H=40, W=64
  -> nan/inf 处理
  -> reshape [B, 1, 40, 64]
  -> depth encoder
  -> encoded depth [B, 64, 5, 8]
  -> flatten [B, 2560]
```

因此：

```text
64 * 5 * 8 = 2560
```

Actor 的 noisy delayed depth 还会经过：

```python
env.delay_manager.compute_delayed_depth(...)
```

## 7. Height Scan 预处理

Critic 的 `height_scan_critic` 来自：

```python
height_scan_feat(env, sensor_cfg)
```

流程：

```text
height scanner ray hits
  -> 51 x 51 height grid
  -> clamp [-5, 5]
  -> heightscan VAE encoder
  -> encoded height [B, 64, 7, 7]
  -> flatten [B, 3136]
```

因此：

```text
64 * 7 * 7 = 3136
```

## 8. Runner 如何把 obs 交给网络

Runner 在这里读取 obs：

[OnPolicyRunner](./source/rsl_rl/rsl_rl/runners/on_policy_runner.py)

关键逻辑：

```python
obs, extras = self.env.get_observations()
num_obs = obs.shape[1]
num_critic_obs = extras["observations"]["critic"].shape[1]

actor_critic = ActorCriticSRU(
    num_obs,
    num_critic_obs,
    self.env.num_actions,
    **policy_cfg,
)
```

当前实际 shape：

```text
num_obs = 2576
num_critic_obs = 5713
num_actions = 3
```

PPO rollout 时调用：

[PPO](./source/rsl_rl/rsl_rl/algorithms/ppo.py)

关键逻辑：

```python
self.transition.actions = self.actor_critic.act(obs).detach()
self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
```

也就是：

```text
actor obs  -> ActorCriticSRU.act      -> action
critic obs -> ActorCriticSRU.evaluate -> value
```

## 9. ActorCriticSRU 初始化

核心文件：

[ActorCriticSRU](./source/rsl_rl/rsl_rl/modules/actor_critic_sru.py)

关键维度：

```python
image_input_dims = (64, 5, 8)
height_input_dims = (64, 7, 7)
num_image_features = 64 * 5 * 8 = 2560
num_height_features = 64 * 7 * 7 = 3136
```

Actor proprio dim：

```text
2576 - 2560 = 16
```

Critic proprio dim：

```text
5713 - 3136 - 2560 - 1 = 16
```

这 16 维是：

```text
base_lin_vel       3
base_ang_vel       3
projected_gravity  3
last_action        3
target_position    4
--------------------
total             16
```

当前 G1 网络配置来自：

[G1InstinctNavPPOVisMapRunnerCfg](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/agents/rsl_rl_cfg.py)

关键参数：

```text
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
rnn_hidden_size = 512
dropout = 0.2
num_cameras = 1
use_actor_visibility_map = False
use_actor_visibility_query_attention = False
```

## 10. Actor 前向流程

函数：

```python
ActorCriticSRU.process_actor_input(...)
```

流程：

```text
actor obs [B, 2576]
  -> other_obs = first 16 dims
  -> image_obs = last 2560 dims
  -> image_obs reshape [B, 64, 5, 8]
  -> CrossAttentionFuseModule(image_obs, other_obs)
  -> image_features [B, 64]
  -> concat [image_features 64 + proprio 16] = [B, 80]
  -> MemorySRU / LSTM_SRU
  -> recurrent hidden [B, 512]
  -> LinearConstDropout
  -> Actor MLP [512, 256, 128]
  -> action mean [B, 3]
  -> Normal(mean, std)
  -> sample action [B, 3]
```

关键代码：

```python
visual_tail_features = self.num_image_features * self.num_cameras + self.num_visibility_map_features
other_obs = observations[..., : -visual_tail_features]
image_list = self._extract_image_observations(...)
image_features = self.attn_image_net(image_input, other_obs)
combined_features = torch.cat([image_features, other_obs], dim=-1)
combined_features = self.memory_a(combined_features, masks, hidden_states)
```

注意：actor 不是直接把 2576 维丢进 MLP，而是先将 depth feature 和 proprio 分开处理。

## 11. Critic 前向流程

函数：

```python
ActorCriticSRU.process_critic_input(...)
```

流程：

```text
critic obs [B, 5713]
  -> other_obs = 16 dims
  -> time_obs = 1 dim
  -> height_obs = 3136 dims reshape [B, 64, 7, 7]
  -> image_obs = 2560 dims reshape [B, 64, 5, 8]
  -> CrossAttentionFuseModule(height_obs, other_obs) -> height_features [B, 64]
  -> CrossAttentionFuseModule(image_obs, other_obs) -> image_features [B, 64]
  -> concat [height 64 + image 64 + proprio 16] = [B, 144]
  -> MemorySRU / LSTM_SRU
  -> recurrent hidden [B, 512]
  -> time embedding [B, 8]
  -> concat [512 + 8] = [B, 520]
  -> LinearConstDropout
  -> Critic MLP [512, 256, 128]
  -> value [B, 1]
```

关键代码：

```python
time_offset = self.num_height_features + self.num_image_features * self.num_cameras + 1
other_obs = observations[..., :-time_offset]
time_obs = observations[..., -time_offset].unsqueeze(-1)
height_obs = observations[..., -(height_start + self.num_height_features) : -height_start]
image_list = self._extract_image_observations(observations)

height_features = self.attn_height_net(height_obs, other_obs)
image_features = self.attn_critic_image_net(image_input, other_obs)
combined_features = torch.cat([height_features, image_features, other_obs], dim=-1)
combined_features = self.memory_c(combined_features, masks, hidden_states)
time_embed = self.time_layer(time_obs)
return torch.cat([combined_features.squeeze(0), time_embed], dim=-1)
```

## 12. SRU 循环

SRU 实现在：

[LSTM_SRU / LSTMSRUCell](./source/rsl_rl/rsl_rl/networks/sru_memory/lstm_sru.py)

核心 cell：

```python
class LSTMSRUCell(nn.Module):
```

每个时间步：

```text
x_t, h, c
  -> concat [x_t, h]
  -> linear_all 得到 i, f, o, g
  -> transform_gate(x_t)
  -> g_t = tanh(tx * g)
  -> refined forget gate
  -> c_next = f * c + (1 - f) * g_t
  -> h_next = o * tanh(c_next)
```

外层循环：

```python
for t in range(seq_len):
    x_t = x[t]
    for layer_idx, cell in enumerate(self.cells):
        h_t, c_t = cell(x_t, h[layer_idx], c[layer_idx])
        x_t = h_t
```

训练时 recurrent batch 通常是：

```text
obs_batch: [T, B, obs_dim]
masks_batch: episode reset mask
hidden_states: rollout 中保存的 h/c
```

推理/play 时通常是单步：

```text
obs: [B, obs_dim]
seq_len = 1
hidden state 持续滚动
```

## 13. MLP 和动作分布

Actor 输出：

```python
combined_features = self.linear_dropout_actor(combined_features, dropout_masks)
mean = self.actor(combined_features)
std = self.log_std.exp().expand_as(mean)
self.distribution = Normal(mean, std)
action = self.distribution.sample()
```

当前 action 是 3 维高层速度命令：

```text
vx, vy, yaw_rate
```

这个 action 会进入：

```text
InstinctFrozenNavigationSE2Action
```

再由冻结的 InstinctLab G1 low-level policy 转成 29 DoF 关节动作。

## 14. Low-level policy 观测

低层 policy 不直接使用高层 actor 的 2576 维 obs，而是使用单独 observation group：

```text
InstinctLowLevelPolicyCfg
```

定义在：

[InstinctLowLevelPolicyCfg](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py)

低层 obs：

```text
base_ang_vel        8 frame history -> 24
projected_gravity   8 frame history -> 24
velocity_commands   8 frame history -> 24
joint_pos           8 frame history -> 232
joint_vel           8 frame history -> 232
actions             8 frame history -> 232
depth_image         8 frames, 18 x 32
```

低层 policy 是 frozen 的，不训练。高层 SRU 学的是如何输出 3 维速度命令，让低层 policy 去执行。

## 15. 推荐阅读顺序

建议按这个顺序读代码：

1. [navigation_env_cfg.py](./source/instinct_nav/instinct_nav/navigation/navigation_env_cfg.py) 只看 `ObservationsCfg`。
2. [observations.py](./source/instinct_nav/instinct_nav/navigation/mdp/observations.py) 看每个观测函数到底取了什么数据。
3. [on_policy_runner.py](./source/rsl_rl/rsl_rl/runners/on_policy_runner.py) 看 obs / critic_obs 如何传给网络。
4. [ppo.py](./source/rsl_rl/rsl_rl/algorithms/ppo.py) 看 actor / critic 被调用的位置。
5. [actor_critic_sru.py](./source/rsl_rl/rsl_rl/modules/actor_critic_sru.py) 看观测切分、attention、SRU、MLP。
6. [lstm_sru.py](./source/rsl_rl/rsl_rl/networks/sru_memory/lstm_sru.py) 看 SRU cell 的循环和门控。
7. [g1_instinct/navigation_env_cfg.py](./source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py) 看 G1 低层 frozen policy 的接口。
