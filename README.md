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
          + dynamic-mask BEV presence (DynamicMapMask variant)
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
| 动态障碍 | 走廊导航轨迹障碍，课程固定在 iteration `4504` 开启 |

> v4 与旧 checkpoint 的动作语义/网络维度不同，**必须从零训练**，不能续接 RichTerrain v1–v3、ScoreVis 或 scene-token checkpoint。历史演化见 [LOG.md](./LOG.md)。

### 当前实验：纯 SRU 与 visibility BEV 受控消融（训练中）

| 项 | 值 |
| --- | --- |
| NoBEV | `Instinct-Nav-G1-SRU-Ablation-NoBEV-Static3000-Mixed50-OfficialParkour-v0`；无 scene token，obs `2576/5713` |
| BEV | `Instinct-Nav-G1-SRU-Ablation-BEV-Static3000-Mixed50-OfficialParkour-v0`；`64 x 5` token，obs `2896/6033` |

### HardEncounterV3：Learned BEV Query Fusion

V3 使用 `3000 static + 3000 dynamic` 课程；动态阶段固定且精确覆盖 50% 环境，不再升级到
hard tier。每个动态环境最多维持 3 个活跃圆柱，速度为 `0.20-0.50 m/s`。障碍资产为直径
`0.5 m`、高度 `1.6 m` 的圆柱，并保留几何碰撞终止。
配对实验只改变 BEV 输入与网络：

| 项 | NoBEV control | LearnedBEVQuery |
| --- | --- | --- |
| task | `Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV3Cylinder-OfficialParkour-v0` | `Instinct-Nav-G1-SRU-LearnedBEVQuery-HardEncounterV3Cylinder-OfficialParkour-v0` |
| actor / critic obs | `2576 / 5713` | `8720 / 11857` |
| SRU input | `80 / 144` | `80 / 144`，不把 BEV 写入循环状态 |
| BEV | 无 | `6 x 32 x 32` |
| fusion | 无 | CNN -> `8 x 8 x 64` learned tokens -> actor/critic 独立 SRU query attention |

BEV 六通道依次为静态 visibility history、当前静态 depth hit、当前归一化 depth、当前动态
occupancy、前一帧动态 occupancy、前两帧动态 occupancy。历史动态层先按机器人位姿 warp，且不使用
障碍物 GT 速度；网络从三帧空间差异中隐式学习运动。两组顺序训练脚本为
`scripts/run_hard_encounter_v3_learned_bev_ablation_sequential.sh`；每组训练 6000 轮。

### HumanScale20 地形消融

为适配体型小于原轮足机器狗的 G1，HumanScale20 保持每块地形 `30 m x 30 m`，将拓扑从
`15 x 15 @ 2.0 m` 加密为 `20 x 20 @ 1.5 m`，坑区桥宽从两个单元缩为一个单元。
台阶高度仍为 `0.10-0.20 m`；可变楼梯限制为 3-5 级、`0.28-0.30 m` 踏面，避免把
高层导航消融变成 frozen 低层步态极限测试。

| variant | task |
| --- | --- |
| NoBEV | `Instinct-Nav-G1-SRU-Ablation-NoBEV-HumanScale20-OfficialParkour-v0` |
| BEV | `Instinct-Nav-G1-SRU-Ablation-BEV-HumanScale20-OfficialParkour-v0` |

两者继续使用相同的 3000 轮静态 + 2000 轮 50% 动态调度、奖励、seed 和网络配置；
仅 visibility BEV 输入不同。顺序训练脚本为 `scripts/run_human_scale20_ablation_sequential.sh`。
| 控制变量 | 两者 SRU 输入均为 `80/144`，相同奖励、地形、障碍、PPO、4096 env 和 seed 72 |
| 动态风险（仅 BEV） | 同帧 delayed mask + depth 投影到 `64 x 64` BEV，膨胀 3 cells，current-only；动态深度不写入持久 visibility |
| 障碍 | 动态阶段严格固定50% env（4096中2048）；5 个、1.6 m 高、difficulty 1、分时 encounter 激活 |
| 训练阶段 | 每个任务连续5000轮：前3000纯静态；后2000固定50% env动态、50% env静态 |
| 地形 | 单个30-column RichTerrain bank，不切换 |
| 执行 | NoBEV 从零训练，完成后自动从零启动 BEV；W&B project `ablation_sru` |

BEV 任务的 mask-to-BEV 管线从 iteration 0 起始终运行。静态阶段 GT mask 为零且关闭 false positives，所以不改变 visibility。动态阶段仅非 mask depth 更新持久 visibility，mask depth 只构建当前风险层，从而避免移动物体拖影。NoBEV 对照不创建或消费任何 visibility-map observation。

### 历史实验：DynamicMapMask（6D token）

| 项 | 值 |
| --- | --- |
| 任务 | `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-v0` |
| 目的 | 将训练期仿真动态物体 mask 投影进 visibility-map BEV；部署端以后可替换为 YOLO segmentation |
| scene tokens | `64 x 6 = 384`，`[history_score, current_hit, current_depth, current_edge, visibility_delta, dynamic_presence]` |
| actor / critic obs | `2960 / 6097`（基线 `2896 / 6033` + 每个 token 1 个动态通道） |
| 网络 | history 通道与 5 个 current 通道分别投影后做 DynamicMapCross；不使用 mask CNN 或辅助 loss |
| 鲁棒化 | 8% 整帧 dropout、5% 正像素 dropout、误检 blob、腐蚀/膨胀、最多 3 像素位移 |
| 奖励 | `stair_centerline=0`；其余 Unobserved/AvoidReact 奖励保持 |
| 障碍课程 | iteration 0 开启；success >= 0.90 时难度增加 0.2；不降级 |
| 地形 banks | 4 个预生成 bank，每 1000 PPO iteration 切换 |
| 训练要求 | 从零训练；DynamicMapMask checkpoint 与 5D DynamicMapCross、DirectMask、旧 DynamicMask 均不兼容 |

mask 真值使用与 visual packet 相同 lag 的障碍位置、spawn 与 curriculum-active 历史，并与同帧 delayed depth 一起经过 camera -> world -> robot-local BEV 投影。`dynamic_presence` 只描述当前帧，不写入持久 visibility history，避免移动障碍留下轨迹残影。rasterizer 与 BEV scatter 保持 GPU 批量实现。

`DynamicMapMask-RecurrentFusion` 是新增网络消融：从相同 `64 x 6` token 中对动态单元做 presence-weighted attention pooling，形成 64D context 输入 actor/critic SRU，同时保留原有 SRU 后 scene query attention。observation 仍为 `2960/6097`，但 SRU 权重形状改变，必须从零训练。

`MaskCNN64-GatedAdd` 是基于已完成 restored-rewards MaskCNN run 的新消融：完整 mask + masked-depth 仍经 CNN/self-attention，但输出从 128D 降为 64D，经 actor/critic 独立 adapter 和初始 0.1 的 sigmoid gate，分别残差加到对应 depth-attention 64D。mask 不再 concat 到 recurrent input，因此 actor/critic SRU 输入从 `208/272` 恢复为 sru_ref 的 `80/144`；raw obs 仍为 `7504/10641`，必须从零训练。该任务的动态障碍从 iteration 0 固定难度 1，覆盖 100% 环境，5 个障碍按原时间间隔激活，几何尺寸为 `0.5 x 0.5 x 1.6 m`。完整维度表见 `docs/mask_cnn_visibility.md`。

`MaskCNN32-Concat` 是后续独立消融：同一 mask CNN 输出 32D，直接拼接到 actor/critic SRU 输入，不使用 gated-add 或融合 LayerNorm。SRU 输入为 `112/176`，raw obs、奖励、五障碍时序、难度 1、100% 环境和 1.6 m 高度与 MaskCNN64-GatedAdd 相同；从零训练，不能加载 64D 门控模型的 checkpoint。

最新论文式网络图：[SVG](docs/figures/mask_cnn64_gated_add_paper_architecture.svg) / [PDF](docs/figures/mask_cnn64_gated_add_paper_architecture.pdf) / [4K PNG](docs/figures/mask_cnn64_gated_add_paper_architecture_4k.png)。

该 RecurrentFusion 任务固定创建 5 个动态障碍，不再由 difficulty 改变数量。每个 episode 首个障碍在 `1–2 s` 后激活，其余按 `1.5 s` 间隔依次激活；difficulty 只继续调节速度、生成距离、行为类型和参与动态课程的环境比例。每个障碍的 activation state 独立写入 delayed mask history。

`DirectMask`（额外追加 `9 x 16` mask，`3040/6177` obs）保留为输入位置消融；旧 `DynamicMask`（raw depth -> CNN mask head，`5200/8337` obs，BCE+Dice）保留为感知学习消融。训练仍使用带扰动的 GT mask；部署端使用 YOLO11-seg 生成实例分割并合并投影到同一 `dynamic_presence` 通道，配置与性能见 `deploy/README.md`。

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
5. DynamicMapMask 任务把同帧动态物体 mask 用 delayed depth 反投影到 BEV，作为每个 DynamicMapCross token 的第 6 个 current-only 通道，不含 mask CNN 或辅助 loss。
6. RecurrentFusion 消融将含动态目标的 BEV token 汇聚为 64D context，actor SRU 输入为 `16+64+64=144`，critic SRU 输入为 `16+64+64+64=208`；SRU 后仍保留 64D scene attention。
7. DirectMask 保留为独立输入消融；旧 DynamicMask ablation 才会用 `36 x 64` raw delayed depth 和 privileged label 训练分割头。

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

### Terrain bank 实验任务

在不改变网络、观测维度和官方低层 ONNX 的前提下，bank 任务启动时预生成 4 组
RichTerrain，并每 1000 个 PPO iteration 切换一组。环境只在 episode reset 时进入
新 bank，不会在 episode 中途移动。完整任务 ID：

    Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Banks-OfficialParkour-v0

默认时间线为 0-999: bank 0、1000-1999: bank 1、2000-2999: bank 2、
3000-3999: bank 3，之后循环。TensorBoard 指标
Metrics/robot_goal/terrain_bank_id 用于确认当前 reset episode 所在 bank。

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

动态障碍课程固定在第 `4504` 个 PPO iteration 开启（`4504 * 16 steps = 72064 env steps`）。

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

DynamicMapMask smoke（当前推荐）：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-Dev-v0 \
  --num_envs 32 --max_iterations 1 --run_name dynamic_map_mask_smoke --headless
```

确认 scene token 为 `384`、obs shape 为 actor `2960` / critic `6097`，产出 `model_0.pt`，并监控 `Metrics/dynamic_mask/bev_token_positive_rate`。

RecurrentFusion ablation smoke：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-RecurrentFusion-Banks-OfficialParkour-Dev-v0 \
  --num_envs 32 --max_iterations 1 --run_name dynamic_map_mask_recurrent_fusion_smoke --headless
```

obs shape 仍为 actor `2960` / critic `6097`；网络应打印 recurrent fusion 64D，以及 actor/critic SRU input `144/208`。

MaskCNN64 gated-add smoke：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-Dev-v0 \
  --num_envs 32 --max_iterations 1 --run_name mask_cnn64_gated_add_smoke --headless
```

确认 obs shape 为 actor `7504` / critic `10641`，mask feature 为 64D，actor/critic SRU input 为 `80/144`，并监控 `Policy/mask_gate_{actor,critic}`。

DirectMask ablation smoke：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DirectMask-Banks-OfficialParkour-Dev-v0 \
  --num_envs 32 --max_iterations 1 --run_name direct_mask_smoke --headless
```

确认无 `mask_label` observation group、obs shape 为 actor `3040` / critic `6177`，网络打印 `Dynamic mask input mode: direct_mask`，并产出 `model_0.pt`。

旧 DynamicMask ablation smoke：

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
  --task Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DynamicMask-Banks-OfficialParkour-Dev-v0 \
  --num_envs 32 --max_iterations 1 --run_name dynamic_mask_vectorized_smoke --headless
```

除 checkpoint 和 obs shape（actor `5200` / critic `8337`）外，还需确认 `Loss/mask*` 均为有限值，且障碍出现后的 `mask_positive_rate` 非零。

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

重点指标：`Metrics/robot_goal/success`（episode 级成功率）、`goal_progress`、`velocity_toward_goal`；稳定性看 `Episode_Termination/{base_contact, large_pitch_angle, terrain_fall, time_out}`；探索看 `Policy/mean_noise_std`（塌缩过快需警惕）；动态障碍看 `obstacle_collision`。DynamicMapMask 监控 `Metrics/dynamic_mask/{clean_positive_rate,input_positive_rate,bev_token_positive_rate}`，且 `Loss/mask` 应恒为 0；只有旧 DynamicMask ablation 需要监控 `Loss/{mask,mask_dice_loss,mask_positive_rate,mask_pred_positive_rate,mask_soft_iou}`，防止全背景塌缩。`early_termination` 在本任务中主要是到达目标的正常终止路径。

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

### MotionBEV Query (2026-10-05)

New task: `Instinct-Nav-G1-SRU-MotionBEVQuery-HardV3-OfficialParkour-v0`.
Control: `Instinct-Nav-G1-SRU-NoBEVRewardMatched-HardV3-OfficialParkour-v0`.
Both provide `-Dev-v0` and `-Play-v0` variants. Previous tasks are unchanged.

| Component | Completed LearnedBEV | MotionBEV |
| --- | --- | --- |
| BEV input | 6 x 32 x 32 | 6 x 32 x 32 |
| Dynamic history offsets | 0.2 / 0.4 s | 0.4 / 0.8 s |
| Native dynamic dilation | 7 x 7 | 3 x 3 |
| Query CNN spatial path | 32 -> 16 -> 8 -> 4 -> pool 8 | 32 -> 16 -> 8 -> 8 |
| Query feature / tokens | 64D / 64 | 64D / 64 |
| Actor / critic SRU input | 80 / 144 | 80 / 144 |
| Actor / critic flat observation | 8720 / 11857 | 8720 / 11857 |
| velocity_into_unobserved | configured -0.2 | removed in treatment and control |

SRU stays one layer, hidden size 512; query fusion stays after the SRU. Dynamic
history is ego-motion compensated and reset per episode. Other rewards, terrain,
physical cylinders and curriculum remain inherited from HardV3: 3000 static +
3000 dynamic iterations with an exact 50% dynamic cohort, three active cylinders
at 0.20-0.50 m/s and height 1.6 m. No future occupancy prediction loss is implemented
in this task. This changes perception semantics and should be trained from scratch.

The paired training script is prepared but has not been started:

```bash
screen -dmS motion_bev_hard_v3 bash /workspace/instinct_nav/scripts/run_motion_bev_hard_v3_ablation.sh
```

It runs 4096 environments, seed 72, with WandB project `ablation_sru`, sequentially
training the reward-matched control and then MotionBEV into timestamped new runs.

### MotionBEV Future Auxiliary Task

`Instinct-Nav-G1-SRU-MotionBEVFuture-HardV3-OfficialParkour-v0` extends MotionBEV
Query with training-only future occupancy supervision (`-Dev-v0` / `-Play-v0`
also registered). Actor/critic observations and SRU dimensions stay unchanged.

- Shared actor BEV CNN -> 8x8x64 -> two transpose convolutions -> two 32x32
  future occupancy logits, for 0.4 and 0.8 seconds. The decoder adds 41,042
  parameters. Forecasts are auxiliary, not new action inputs; query fusion stays.
- A separate 27D teacher group stores actual robot x/y/yaw and six obstacle
  slots' x/y/active/source-visible states. It is never fed to actor or critic.
  PPO constructs targets from later states in the same rollout, projected into
  the source robot frame, using radius-0.25 m cylinder footprints. These are
  actual recorded centers, not constant-velocity extrapolation.
- Targets include only objects observed at the source step. Reset-crossing
  horizons and tracked-slot despawn/recycle are invalidated. The final four
  source steps of each rollout have no complete future pair and are omitted.
  Fully static/unobserved sources skip the loss rather than train all-background.
- Loss: `0.05 * (weighted BCE + Dice)`, positive BCE weight 20, at most 256
  sampled source frames per PPO minibatch. Labels are generated on demand,
  avoiding dense full-rollout future-map storage. Existing PPO return signatures
  and disabled-task storage layout are preserved.
- Logging: `Loss/future_occupancy`, `Loss/future_bce`, `Loss/future_dice`, plus
  `Metrics/future/{soft_iou,label_positive_rate,pred_positive_rate,valid_fraction}`.
  Valid fraction is conditional on sampled source-visible windows. A two-update
  32-env PPO smoke completed with positive labels and nonzero forecast gradients;
  it does not establish navigation benefit.

This task completed 6000 updates on 2026-10-08. Deployment does
not require teacher states or execution of the forecast decoder. Export of the
new checkpoint has not yet been tested.

```bash
screen -dmS motion_bev_future_hard_v3 bash /workspace/instinct_nav/scripts/run_motion_bev_future_hard_v3.sh
```

The script uses the unchanged HardV3 3000-static/3000-mixed50 curriculum,
4096 envs, seed 72, WandB `ablation_sru`, and timestamped outputs. Compare against
MotionBEVQuery to isolate the auxiliary task, and reward-matched NoBEV to assess
the full BEV treatment. Earlier LearnedBEV eval statistics still need rerunning
with the read-only visualization fix before being used as headline comparisons.

### MotionBEV ConvLSTM Ablation (2026-10-08)

New task: `Instinct-Nav-G1-SRU-MotionBEVConvLSTM-HardV3-OfficialParkour-v0`,
with Dev/Play variants. Formal training started from scratch on 2026-10-08 in
detached screen `motion_bev_convlstm_20261008`, with 4096 envs, 6000 updates,
seed 72 and WandB project `ablation_sru` (run `maf3gdnr`). The initial PPO update
completed and saved model_0.pt; dynamic obstacles are off in the static phase.
Launch script: `scripts/run_motion_bev_convlstm_hard_v3.sh`.

- Static S0/S1/S2 -> existing spatial CNN -> 64x8x8 features. The dynamic inputs
  are zeroed for this actor static encoder; the critic retains its previous CNN.
- Dynamic D0/D1/D2 -> small CNN -> single ConvLSTM, hidden/cell 32x8x8 ->
  1x1 projection to 64x8x8. Add spatial features to the static branch, then keep
  the existing SRU-conditioned query fusion and actor MLP. Navigation SRU is
  still one 512D layer with actor/critic input dimensions 80/144.
- The forecast decoder consumes projected dynamic memory only, not static CNN
  features. Predict 0.4/0.8 s occupancy with the existing recorded-future teacher,
  source-visibility filtering, reset/recycle validity and 0.05 weighted BCE+Dice.
- ConvLSTM hidden/cell maps are warped into the current robot frame before each
  update. Three actor-only odometry values (x/y/yaw) are used exclusively by the
  warp, never by depth attention, navigation SRU or the query. Actor observations
  grow from 8720 to 8723; critic remains 11857. Simulation supplies exact odometry;
  real deployment will need measured odometry and separate export validation.
- PPO stores initial spatial states with recurrent trajectories. Navigation
  updates replay padded trajectories; forecasting replays continuous rollout
  sequences, clearing spatial states at episode boundaries. Independent-frame
  forecasting is explicitly rejected. Auxiliary replay never mutates online states.
- Additional parameters: dynamic CNN 15,104; ConvLSTM 73,856; projection 2,112,
  total 91,072 relative to MotionBEVFuture. Prior tasks/checkpoints are untouched;
  this new task must train from scratch.
- Reward, terrain, cylinder logic and formal curriculum remain unchanged:
  3000 static updates, then 3000 mixed updates with 50% dynamic environments.
  Cylinders use geometric collision termination, not physical contact forces.

Validation: 51 focused tests passed; 32-env PPO smoke saved model_0.pt/model_1.pt
and updated the dynamic CNN, ConvLSTM and decoder. Spatial initial states are
stored only on the actor side; an additional smoke verified independent
actor/critic state-list lengths without dummy critic allocations.

The completed previous-task eval in
`logs/motion_bev_future_eval_retry_20261008/` contains 19,200 episodes per
condition and ten complete failure GIFs each: SRU-reference success is 83.74%
static and 61.82% dynamic100. Older unseeded-terrain evaluations are references,
not strictly paired comparisons. ConvLSTM is a hypothesis, not a proven gain.
