# instinct_nav

`instinct_nav` 是 **G1 人形机器人高层自主导航** 项目：复用 InstinctLab 冻结的 Parkour 低层运动策略，在其上按 SRU 思想训练高层导航策略。高层只输出速度命令，低层负责把速度命令变成稳定的关节动作。

本 README 只记录 **项目概况、网络结构、运行命令**。

- 时间顺序的训练/对齐/部署进展 → [LOG.md](./LOG.md)
- 评估结果与对比表 → [LOG.md](./LOG.md)（历史基线快照并入其末尾章节）
- 代理/开发操作手册 → [AGENTS.md](./AGENTS.md)
- 真机部署 → [deploy/README.md](./deploy/README.md)、修复清单 [docs/advices.md](./docs/advices.md)

---

## 一、项目概况

### 双层控制架构

```text
High-level SRU navigation policy @ 5 Hz
  obs   : proprio + goal_command + encoded_depth + dynamic-map tokens
  action: [vx, vy, wz]  (SE(2) 速度命令)
        │  仿射映射到低层训练分布
        ▼
Frozen InstinctLab G1 Parkour low-level policy @ 50 Hz  (官方 ONNX，requires_grad=False)
  obs   : parkour proprio history + velocity_commands history + depth history
  action: 29D joint position target
```

- 频率：物理 200 Hz（`sim.dt=0.005`）；低层 50 Hz（`low_level_decimation=4`）；高层 5 Hz（每个高层 step 内执行多次冻结低层控制）。
- 高层**只输出速度命令**，不直接输出关节动作；低层 checkpoint 全程冻结、只做推理。
- 高层视觉复用低层同一物理相机（shared-lowcam），不额外创建高层相机。

### 当前主线（RichTerrain v4）

| 项 | 值 |
| --- | --- |
| 主任务 | `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-v0` |
| 高层网络 | DynamicMapCross 两级 cross-attention（见下） |
| 冻结低层 | `backend_type="official_onnx"`，官方 Parkour ONNX 权重对 |
| 速度契约 | `vx∈[0,1]`、`vy≡0`、`wz∈[-1,1]`（tanh→仿射，对齐低层训练分布） |
| 地形 | RichTerrain v4：`30 x 30 m` tile、`15 x 15` 网格、6 类子地形 |
| 动态障碍 | 走廊导航轨迹障碍，课程固定在 iteration `3003` 开启 |

> v4 与旧 checkpoint 的动作语义/网络维度不同，**必须从零训练**，不能续接 RichTerrain v1–v3、ScoreVis 或 scene-token checkpoint。历史演化见 [LOG.md](./LOG.md)。

### 关键文件

| 用途 | 路径 |
| --- | --- |
| 任务配置 | `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/navigation_env_cfg.py` |
| 任务注册 | `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/__init__.py` |
| runner / PPO cfg | `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/agents/rsl_rl_cfg.py` |
| SRU actor/critic | `source/rsl_rl/rsl_rl/modules/actor_critic_sru.py` |
| PPO 算法 | `source/rsl_rl/rsl_rl/algorithms/ppo.py` |
| 高层动作桥（SE(2)→低层命令） | `source/instinct_nav/instinct_nav/navigation/mdp/navigation/actions/instinct_navigation_actions.py`（+ `..._cfg.py`） |
| 冻结低层 backend | `.../navigation/mdp/navigation/actions/instinct_frozen_backend.py` |
| observation（VisScore / dynamic-map patch / goal / token） | `source/instinct_nav/instinct_nav/navigation/mdp/observations.py` |
| 动态障碍 | `source/instinct_nav/instinct_nav/navigation/mdp/dynamic_obstacles.py` |
| 地形生成 | `source/instinct_nav/instinct_nav/terrains/{maze_config.py, hf_terrains_maze.py, hf_terrains_maze_cfg.py}` |
| 训练入口 | `_launcher.py` → `scripts/train_senior.py` |
| 部署 | `deploy/instinct_nav_onboard/`、`deploy/scripts/g1_nav_onboard_onnx.py` |

### 资产与依赖

- 冻结低层权重：`data&model/checkpoints/parkour_onboard_preview_stair/exported/{actor.onnx, 0-depth_encoder.onnx}`（官方 onboard 版）。运行时由 `OfficialOnnxFrozenPolicyBackend` 重建为等价 GPU PyTorch 网络以支持 4096 环境批量推理。
- `artifacts/low_level/g1_parkour/model_30000.pt` 仅为 `backend_type="checkpoint"` 的兼容字段，官方 task 不加载它。
- 外部依赖保持不变：IsaacLab / IsaacSim 运行时、`external/instinctlab`（sensor / 低层接口 / terrain helper）、`external/instinct_rl`。

### 与原 SRU 的关系

高层沿用 SRU 思想（5 Hz 速度命令、CNN depth encoder、SRU 记忆、attention 融合、privileged critic、mask-based goal/spawn 采样、`[dir_xyz, log(dist+1)]` 目标观测 + policy 侧噪声）；有意不同的是：机器人换成 G1 人形，低层换成冻结的官方 InstinctLab Parkour 策略，速度命令范围严格对齐低层训练分布。`sru_new` 仅作历史参考。

---

## 二、网络结构

### 高层数据流

```text
actor : proprio+goal+action(16) + encoded_depth(2560) + dynamic-map tokens(64x5=320) = 2896
critic: 上述 + time_normalized + encoded height_scan(64x7x7=3136) = 6033
```

1. Depth（shared-lowcam raw metric depth）经 SRU `DepthNoiseEncoder` → `64 x 5 x 8 = 2560`。
2. proprio + depth 进入 attention / SRU recurrent memory。
3. dynamic-map tokens 经两级 cross-attention 生成，再由 SRU hidden + proprio 作为 query 融合。
4. actor 输出 SE(2) 速度命令；critic 额外吃 `time_normalized` 与 privileged height scan，输出 value。

### DynamicMapCross 两级 cross-attention

保留长期可见性记忆 `VisScore` 的同时显式引入当前帧，避免网络只能从单张衰减图反推动态变化。

**VisScore 历史图**（与高层 policy 同一路 noisy/delayed shared-lowcam 深度）：

```text
history_score[t] = max(current_visibility[t], decay * history_score[t-1]),  decay = 0.995
```

**当前帧每个 `8 x 8` patch 提取 4 个量**，与历史分共同构成逻辑 patch：

```text
[history_score, current_hit, current_depth, current_edge, visibility_delta]   # [N, 64, 5] → flatten 320
```

- 第 1 级（map→map）：当前帧 4D 分支作 **query**，历史 VisScore 1D 分支作 **key/value**，加二维位置 embedding，输出 **64 个 learned dynamic tokens**。
- 第 2 级（query→token）：SRU hidden + proprio 作 query，对 64 个 learned tokens 做 cross-attention，得到 actor/critic 的 scene feature。

`visibility_delta`（当前是否命中与历史分之差）让刚出现 / 刚消失 / 遮挡变化的区域直接进入网络。

### 高层速度契约（v4）

3 维网络输出先 `tanh`，再仿射映射到与官方 Parkour 低层训练分布一致的范围：

```text
vx: 0.0 → 1.0 m/s
vy: 固定 0.0 m/s        # 低层从未见过横向速度，保持 unicycle 语义
wz: -1.0 → 1.0 rad/s
```

因此高层原始零动作 `[0,0,0]` 映射为 `[0.5, 0, 0]`。训练 action bridge、部署 runtime、ONNX runtime helper 三处已同步该 contract。

### RichTerrain v4 地形

- 原始 `MAZE_TERRAIN_CFG` 完全不变。
- RichTerrain：每 tile `30 x 30 m`、内部 `15 x 15` 网格、`6 x 30` 个 tile，episode 时长继承原配置。
- 采样占比：`maze 16% / non_maze 16% / variable_stairs 20% / blocks 16% / pits 12% / traversal_mix 20%`。
- `traversal_mix`（复杂穿越元素集中于此，不再混入旧类别）：楼梯/平台（每组随机 `3–7` 阶）、宽缓坡、凸起横条（每块 `6–10` 根，高 `0.14–0.18 m`）、横条形洼地（每块 `4–7` 根，深 `0.08–0.18 m`），方向可横/纵/`±45°`。

---

## 三、运行命令

统一入口：`/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py <子命令>`。

### 正式训练（当前 v4 主线）

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

动态障碍课程固定在第 `3003` 个 PPO iteration 开启（`3003 * 16 steps = 48048 env steps`）。

### Smoke test（Dev 任务，跑通即可）

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Dev-v0 \
  --num_envs 32 \
  --max_iterations 1 \
  --run_name rich_terrain_v4_smoke \
  --headless
```

Dev 任务把动态障碍切换阈值设为 0 以覆盖动态轨迹代码路径；确认产出 `model_0.pt`、obs shape（actor `2896` / critic `6033`）、`terrain_fall=0`。不要用 `num_envs=2` 做 PPO smoke（recurrent mini-batch 可能空 batch）。

### Play / Eval（Play 任务）

```bash
# play 可视化
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py play \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Play-v0 \
  --checkpoint logs/rsl_rl/g1_instinct_navigation_ppo_vismap/<run_dir>/model_XXXX.pt \
  --num_envs 1 --steps 1200 --log_interval 20

# eval 统计成功率 + 保存失败 GIF
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py eval \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Play-v0 \
  --checkpoint logs/rsl_rl/g1_instinct_navigation_ppo_vismap/<run_dir>/model_XXXX.pt \
  --num_envs 256 --num_episodes 1000 --headless \
  --json_out logs/.../eval_summary.json \
  --save_failure_gifs --max_failure_gifs 10
```

录视频用 `--video --video_length N`（不要写成 `--video N`）。headline 结果用普通 eval，fixed-case manifest 对 token/记忆型策略当前不完全可信。

### 地形检查

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py inspect-terrain \
  --terrain traversal_mix --difficulty 0.8 --seed 72
```

俯视预览：`artifacts/terrain/rich_terrain_traversal_mix_preview.png`。

### 看训练进度

```bash
tail -f /workspace/instinct_nav/logs/launch/g1_sru_dynamic_map_cross_rich_terrain_v4.log
ps -ef | grep _launcher.py | grep -v grep
```

重点指标：`Metrics/robot_goal/success`（episode 级成功率）、`goal_progress`、`velocity_toward_goal`；稳定性看 `Episode_Termination/{base_contact, large_pitch_angle, terrain_fall, time_out}`；探索看 `Policy/mean_noise_std`（塌缩过快需警惕）；动态障碍看 `obstacle_collision`。`early_termination` 在本任务中主要是到达目标的正常终止路径。

> 无 stdout 文件时从 TensorBoard event 读标量：
> ```bash
> /workspace/instinctlab51-env/bin/python -c 'from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; p="PATH_TO_EVENTS"; ea=EventAccumulator(p); ea.Reload(); print(ea.Tags()["scalars"])'
> ```

## 2026-08-17：统一视觉延迟与 BEV DynamicMapCross

本次只修改 DynamicMapCross 的 observation/geometry 链路，不修改 SRU、Depth
Self/Cross-Attention、Actor/Critic、速度映射或课程。当前 v4 进程继续使用启动时加载的旧代码；
以下改动用于后续从零训练的 v5。

- 每个相机每个高层 step 只生成一个 `VisualPacket`：encoded depth、noisy raw depth、拍摄时刻
  camera position/quaternion 和 intrinsics。
- actor depth、critic depth、Current BEV 和 History BEV 共用一个 visual packet delay buffer
  和同一个 per-environment lag（当前为 `0-2` 个高层 step）。
- packet 按 `(camera_name, common_step_counter)` 缓存，同一步重复计算 observation 不会推进
  delay buffer 两次；reset 时清空 packet cache、delay buffer 和 visibility map。
- 当前 robot pose 保持准确；随 packet 延迟的 camera pose 只用于深度帧的正确反投影。
- DynamicMapCross 当前帧不再直接池化相机图像，而是投影为与 History 对齐的 robot-local
  `64x64` Current BEV，再池化为 `8x8`、共 64 个 patch。

每个 patch 仍为 5 维，总 observation 仍为 `320` 维：

```text
[history_before_update,
 current_bev_hit_ratio,
 current_bev_depth,
 current_bev_depth_edge,
 current_bev_hit_ratio - history_before_update]
```

History K/V 使用融合当前帧之前的状态；本步 patch 构造后才 decay/融合 Current BEV。网络侧
current-to-history Cross-Attention 和 SRU scene query 均保持不变。

验证：`py_compile` 与 `git diff --check` 已通过。普通 pytest 环境没有初始化 Isaac Sim `pxr`，
新增 geometry test 会显式跳过；32-env smoke 因 v4 正在占用唯一 CUDA context 无法并发启动，
需在 v4 结束后补跑。
