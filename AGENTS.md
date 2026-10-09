# AGENTS.md - Codex 工作操作手册

本文件面向后续 Codex/开发代理，定义在 `instinct_nav` 项目中的职责边界、工作流程、项目上下文和常用命令。

## 0. 角色定位与职责边界

| instruction | notes |
| --- | --- |
| Codex 负责任务规划、代码阅读、代码修改、文档生成、测试验证、质量审查等完整开发流程 | 默认推进到可验证结果 |
| 工作对象是 `/workspace/instinct_nav`，这是一个独立 IsaacLab-style 人形自主导航项目 | 不要回退成依赖 `/workspace/sru_ref` 或旧外部源码的实验脚本 |
| 技术目标是 SRU 高层导航 + frozen InstinctLab G1 parkour 低层运控 | 高层学习导航，低层 checkpoint 保持冻结 |
| 默认自主决策实现路径，仅在删除核心文件、破坏性 git 操作、远程 push、连续失败等场景询问用户 | 不因普通实现细节频繁打断用户 |
| 所有结论必须基于代码、日志、配置或评估文件证据 | 不凭记忆猜测训练配置 |
| 不得覆盖用户正在训练的 run、checkpoint 或已有未提交改动 | 工作区可能是 dirty 状态 |

## 1. 项目结构

| path | purpose |
| --- | --- |
| `source/instinct_nav/instinct_nav/` | 主 IsaacLab task package |
| `source/instinct_nav/instinct_nav/navigation/` | navigation env、MDP、commands、observations、rewards、terminations |
| `source/instinct_nav/instinct_nav/navigation/config/g1_instinct/` | G1 + frozen Instinct 低层的主要任务配置 |
| `source/rsl_rl/rsl_rl/` | 项目内 SRU-capable rsl_rl 实现 |
| `external/instinctlab/` | InstinctLab 外部代码，本项目复用其 sensor/低层接口 |
| `external/instinct_rl/` | frozen policy runtime 相关外部代码 |
| `artifacts/low_level/g1_parkour/model_30000.pt` | frozen G1 parkour 低层 checkpoint |
| `logs/` | 训练/评估输出，通常不要提交 |
| `README.md` | 项目概况、网络结构、运行命令 |
| `LOG.md` | 时间顺序的训练/对齐/部署进展记录 |
| `docs/advices.md` | 真机部署修复清单 |
| `docs/paper_inspiration_design.md` | 论文启发与设计备忘 |

## 2. 主要任务 ID

### 2.1 Baseline / senior-aligned

| task id | usage |
| --- | --- |
| `Instinct-Nav-G1-SRU-v0` | G1 SRU 主 baseline |
| `Instinct-Nav-G1-SRU-Dev-v0` | 小规模 smoke/dev |
| `Instinct-Nav-G1-SRU-Play-v0` | play 可视化 |

### 2.2 Scene-token variants

| task id | meaning |
| --- | --- |
| `Instinct-Nav-G1-SRU-SceneToken-Query-v0` | v1，actor query token，含 goal-relative token feature |
| `Instinct-Nav-G1-SRU-SceneTokenRisk-Query-v0` | v2，actor/critic risk token，9D，含 goal-relative feature |
| `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-v0` | v3，actor/critic risk token，6D，去掉 per-token goal feature |
| `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0` | v4，v3 token + 高层复用低层 `camera` 单相机 |

对应 dev/play 任务按 `-Dev-v0`、`-Play-v0` 后缀查找。

### 2.3 当前 DynamicMap / DynamicMask 主线

| task id | usage |
| --- | --- |
| `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Banks-OfficialParkour-v0` | 当前 DynamicMapCross terrain-bank baseline |
| `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-v0` | 当前推荐：GT mask 通过 delayed depth 投影进 visibility-map BEV token |
| `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-RecurrentFusion-Banks-OfficialParkour-v0` | 新网络消融：动态 BEV context 同时进入 SRU 与后期 scene attention |
| `Instinct-Nav-G1-SRU-DynamicRiskBEV-Encounter-RestoredRewards-SuccessBanks-OfficialParkour-v0` | 已停止消融：无 mask 网络；动态 mask 在原 5D visibility BEV 中渲染为黑色高风险区，90% success 后切 bank |
| `Instinct-Nav-G1-SRU-DynamicRiskBEV-Static5000-Mixed50-SingleTerrain-OfficialParkour-v0` | 最新待训：单地形、原5D SRU、5000静态+2000混合，动态mask仅作current-only BEV风险覆盖 |
| `Instinct-Nav-G1-SRU-Ablation-NoBEV-Static3000-Mixed50-OfficialParkour-v0` | 当前消融A：纯SRU，无BEV；3000静态+2000混合 |
| `Instinct-Nav-G1-SRU-Ablation-BEV-Static3000-Mixed50-OfficialParkour-v0` | 当前消融B：5D visibility BEV；与消融A同调度，NoBEV完成后自动训练 |
| `Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-v0` | MaskCNN 64D 门控残差加到 actor/critic depth 64D；SRU `80/144`；动态难度 1、100% env、障碍高 1.6 m |
| `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DirectMask-Banks-OfficialParkour-v0` | compact mask 独立输入 ablation |
| `Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DynamicMask-Banks-OfficialParkour-v0` | 旧 depth-to-mask CNN ablation |
| 对应 `-Dev-v0` | 32-env smoke；动态障碍从 iteration 0 开启 |
| 对应 `-Play-v0` | play / 普通 eval |

## 3. 当前重要结果

### 3.1 当前最好完成结果：v3 no-goal scene token

| item | value |
| --- | --- |
| run dir | `logs/rsl_rl/g1_instinct_navigation_ppo_vismap/2026-07-25_17-29-52_g1_sru_scene_token_risk_no_goal_query_v1/` |
| checkpoint | `model_2999.pt` |
| eval summary | `eval_summary_model_2999_1000ep.json` |
| overall success | `0.916` |
| stairs success | `0.931` |
| base_contact | `0.048` |
| large_pitch_angle | `0.057` |

结论：去掉 token 内的 `dist_to_goal/cos_to_goal/sin_to_goal` 后，v3 明显优于 v1/v2。scene token 应描述局部场景/风险，目标信息保留在普通 goal observation，并通过 query 影响 attention。

### 3.2 当前正在验证：v4 shared-lowcam

| item | value |
| --- | --- |
| task | `Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0` |
| run name | `g1_sru_scene_token_risk_no_goal_shared_lowcam_query_v1` |
| purpose | 高层 SRU 与 frozen 低层共用同一个物理 depth sensor `camera` |
| early observation | 约 190 iter 时 learning 明显快于 v3 同轮次 |

不要把 v4 早期曲线当成最终结论。最终仍需普通 eval 对比 v3。

### 3.3 当前开发阶段：DynamicRiskBEV 单地形课程

| item | value |
| --- | --- |
| branch | `feature/dynamic-obstacle-avoidance-curriculum` |
| current experiment | `Static5000-Mixed50-SingleTerrain`，已实现未训练 |
| actor / critic obs | `2896 / 6033` |
| scene token | `[num_envs, 64, 5]`，与旧 DynamicMapCross 相同 |
| SRU input | actor/critic `80/144`；无 MaskCNN、无额外 recurrent feature |
| obstacles | 前5000轮关闭；之后固定50% env启用5个、1.6 m、difficulty 1障碍 |
| perception | 训练期 GT mask；当前不实现 YOLO，不训练 depth-to-mask CNN |
| auxiliary loss | 无，`Loss/mask=0` |
| terrain | 单个30-column bank，不切换；总计7000轮 |
| checkpoint rule | 从零训练；网络 shape 与旧 5D DynamicMapCross 相同，但观测语义不同 |

AvoidReact-Demote 已完成验证且没有实质收益。DynamicMapMask headline 结论必须来自普通 eval，不得用 dev smoke 或训练窗口替代。

## 4. 相机与视觉链路

### 4.1 非 shared 任务的高层 SRU 相机

| item | value |
| --- | --- |
| sensor name | `raycast_camera` |
| type | IsaacLab `RayCasterCameraCfg` |
| link | G1 中挂到 `{ENV_REGEX_NS}/Robot/torso_link` |
| output | raw `distance_to_image_plane` |
| effective image size | `64 x 40` |
| SRU encoder output | `64 x 5 x 8` |
| flattened depth feature | `2560` |

### 4.2 frozen Instinct 低层相机

| item | value |
| --- | --- |
| sensor name | `camera` |
| type | `NoisyGroupedRayCasterCameraCfg` |
| link | `{ENV_REGEX_NS}/Robot/torso_link` |
| raw image size | `64 x 36` |
| FOV | about `89.51 deg x 58.29 deg` |
| update period | `0.02 s` |
| raw output | `distance_to_image_plane` |
| low-level processed stream | `distance_to_image_plane_noised_history` |
| low-level policy input | `(8, 18, 32)` |

重要事实：已做 shape check，当前 SRU `DepthNoiseEncoder` 接收低层 raw `64 x 36` 后仍输出 `64 x 5 x 8 = 2560`，所以 shared-lowcam 不需要修改 `ActorCriticSRU.image_input_dims=(64, 5, 8)`。

### 4.3 shared-lowcam 规则

| instruction | notes |
| --- | --- |
| 高层 actor/critic depth 使用 `SceneEntityCfg("camera")` | 读取 raw `distance_to_image_plane`，不直接吃低层 noised history |
| high-level scene tokens 也使用 `SceneEntityCfg("camera")` | token 投影使用该 sensor 自身 `image_shape/pos_w/quat_w_world/intrinsic_matrices` |
| 低层 policy 输入保持不变 | 继续使用 `distance_to_image_plane_noised_history` |
| shared-lowcam task 中 `self.scene.raycast_camera = None` | 避免创建第二个高层相机 |

## 5. Observation 到网络的数据流

### 5.1 高层 actor

1. Observation manager 收集 policy group。
2. senior-aligned G1 policy proprio/goal/action 共 `16` 维。
3. `depth_image_noisy_delayed` 读取 raw metric depth。
4. depth 经过 SRU depth noise/encoder，输出 `2560` 维 encoded depth。
5. scene-token 任务从同一路 noisy/delayed depth 构建 local visibility memory。
6. v3/v4 scene token 为 `[num_envs, 64, 6]`，flatten 后 `384` 维。
7. `ActorCriticSRU` 将 flat obs 拆成 proprio、depth、scene token。
8. proprio + depth 先进入 attention/SRU recurrent memory。
9. token query branch 使用 SRU hidden state + proprio 作为 query，对 scene tokens 做 cross attention。
10. attention 输出 concat 到 actor MLP 前，actor 输出高层 SE(2) velocity command。
11. RecurrentFusion 消融额外在 SRU 前对 `dynamic_presence` 非零 token 做加权 attention pooling，把 64D context 写入循环记忆；原后期 query branch 保留。

### 5.2 高层 critic

1. Critic 接收对应 proprio/goal/action。
2. Critic 额外接收 `time_normalized`。
3. Critic 接收 privileged `height_scan_critic` encoded feature。
4. Critic 接收 encoded depth。
5. v2/v3/v4 critic 也接收 scene token。
6. Critic SRU + critic token attention 输出 value estimate。

### 5.3 frozen 低层 policy

1. 高层 action 注入 `base_velocity` command。
2. `InstinctFrozenNavigationSE2Action` 构造低层 observation。
3. frozen Instinct policy 消费 proprio/action history 和 `(8,18,32)` depth history。
4. frozen policy 输出 G1 joint position action。

## 6. Scene Token 设计规范

| item | value |
| --- | --- |
| preferred version | v3 no-goal risk token |
| logical shape | `[num_envs, 64, 6]` |
| flattened shape | `[num_envs, 384]` |
| features | `[x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier]` |

默认不要恢复 per-token goal-relative features。已知 v1/v2 的 `dist_to_goal/cos_to_goal/sin_to_goal` 会把目标信息混入 token 本身，效果不如 v3。目标信息应通过普通 `target_position` observation 进入网络，再由 query branch 使用。

### 6.1 DynamicMapMask 实现规范

| instruction | notes |
| --- | --- |
| 当前推荐把 mask 融入 DynamicMapCross token | 不额外追加 standalone mask，不训练 depth-to-mask CNN |
| mask 必须与 delayed depth 同时间、同视场 | 使用同一 visual packet 与障碍状态 lag history |
| 使用 depth 的 camera -> world -> robot-local BEV 投影 | mask 和 depth 必须共用 pixel stride、valid-depth 与 cell index |
| token 为 `64 x 6` | 前 5 维保持旧 DynamicMapCross，第 6 维为 `dynamic_presence` |
| dynamic presence 只属于当前帧 | 不写入 persistent visibility history，防止移动残影 |
| actor/critic 同一步必须复用相同 token | 使用 `_instinct_nav_dynamic_map_patch_cache` |
| 训练需保留分割扰动 | frame/pixel dropout、false positive、morphology、pixel shift |
| DynamicMapMask 专用课程 | `stair_centerline=0`；障碍 iteration 0 开启；success >= 0.90 晋级 0.2 |
| terrain banks | 4 个 bank，每 1000 PPO iteration 切换 |
| 正式训练监控三项正率 | `Metrics/dynamic_mask/{clean_positive_rate,input_positive_rate,bev_token_positive_rate}` |
| obs 与 checkpoint | actor `2960`、critic `6097`；新架构从零训练 |

DirectMask（standalone `9 x 16` mask，`3040/6177` obs）保留为 ablation，不是当前推荐输入布局。

### 6.2 旧 DynamicMask 实现规范

| instruction | notes |
| --- | --- |
| label 必须使用与 visual packet 相同 lag 的 `positions_w/spawned/curriculum_envs` 历史 | 不得用当前速度反推历史位置 |
| history 容量必须由 `max_delay_depth + 1` 推导 | 不得硬编码为 3 |
| mask rasterizer 必须对 env/pixel 做 GPU 批量广播 | 禁止 `for env` + CUDA `.item()` |
| 正式训练同时监控 label/pred positive rate 与 soft IoU | 防止稀疏标签下全背景塌缩 |
| DynamicMask obs 为 actor `5200`、critic `8337` | 旧 DynamicMapCross 必须仍为 `2896/6033` |

## 7. 工作流程

### 阶段 0：需求理解与上下文收集

| instruction | notes |
| --- | --- |
| 先确认用户是要检查、解释、给命令、还是直接改代码 | 用户常明确说“先不要修改”“你来做”“开始修改” |
| 优先用 `rg` / `sed` / `git diff` / TensorBoard event 读取证据 | 不凭记忆回答配置 |
| 涉及训练效果时同时看 reward、success、termination、noise std | 单看 success 容易误判 |
| 涉及相机/obs/网络时必须串起完整数据流 | sensor -> observation -> concat -> ActorCriticSRU split -> SRU/MLP |

### 阶段 1：任务规划

| instruction | notes |
| --- | --- |
| 对较大改动先形成 3-6 步短计划 | 用当前可用 plan tool 或直接在回复中说明 |
| 优先新增 ablation task，不覆盖已验证 task | 方便可复现实验对比 |
| 先定义验收标准 | 至少包括 py_compile/smoke test/obs shape/关键日志指标 |

### 阶段 2：代码执行

| instruction | notes |
| --- | --- |
| 优先使用 `apply_patch` 编辑 | 当前容器可能因 bwrap 限制失败，失败后可用受控 Python 写入 |
| 不要用破坏性 git 命令 | 不要 `git reset --hard`、不要回退用户改动 |
| 不要删除 logs/checkpoints，除非用户明确要求 | 训练产物可能有价值 |
| 改 token 维度时同步改 env observation 和 runner policy config | 否则 checkpoint/obs split 会错 |
| 改相机源时同步改 actor depth、critic depth、scene token sensor_cfg | 避免 actor/critic 视觉分布不一致 |

### 阶段 3：验证

| instruction | notes |
| --- | --- |
| 轻量代码改动至少跑 `py_compile` | 如果 sandbox bwrap 失败，用已授权 Python 非 sandbox 执行 |
| 新 task 必须跑 dev smoke test | 推荐 `num_envs=32 --max_iterations 1` |
| 不要用 `num_envs=2` 做 PPO smoke test | recurrent mini-batch 可能出现空 batch |
| smoke test 需要确认 obs shape 和是否产出 `model_0.pt` | 只看环境创建成功不够 |
| 训练中没有 stdout 文件时从 TensorBoard event 读标量 | 见常用命令 |

## 8. 常用命令

### 8.1 当前 shared-lowcam 正式训练

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

### 8.2 smoke test

```bash
/workspace/instinctlab51-env/bin/python /workspace/instinct_nav/_launcher.py train \
    --task Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Dev-v0 \
    --num_envs 32 \
    --max_iterations 1 \
    --run_name shared_lowcam_smoke_test \
    --headless
```

### 8.3 查看训练进程

```bash
ps -ef | grep _launcher.py
```

### 8.4 查看 event scalar tags

```bash
/workspace/instinctlab51-env/bin/python -c 'from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; p="PATH_TO_EVENTS"; ea=EventAccumulator(p); ea.Reload(); print(ea.Tags()["scalars"])'
```

### 8.5 关键训练指标

| metric | meaning |
| --- | --- |
| `Metrics/robot_goal/success_rate` | 原始 SRU rolling 成功率；保留旧 termination 分别写入语义 |
| `Metrics/robot_goal/strict_success_rate` | 每个 episode 恰好写一次、物理失败优先的严格成功率 |
| `Metrics/robot_goal/success` | 当前 episode/window 成功统计，可能和 success_rate 口径不同 |
| `Episode_Reward/reach_goal_xy_tight` | 到点奖励，训练进展关键指标 |
| `Episode_Termination/base_contact` | 身体接触终止，反映低层执行稳定性 |
| `Episode_Termination/large_pitch_angle` | 大俯仰终止，反映姿态稳定性 |
| `Episode_Termination/time_out` | 超时比例 |
| `Policy/mean_noise_std` | 探索噪声，过快塌缩需警惕 |

## 9. 评估与 play 注意事项

| instruction | notes |
| --- | --- |
| headline 结果优先使用普通 eval | fixed-case manifest 当前对 token policy 不完全可信 |
| fixed-case manifest eval 需要先修 cache reset/sensor sync | v1 fixed-case 曾出现异常早期成功率 |
| play/eval 要使用匹配的 `-Play-v0` task 和同架构 checkpoint | no-token/v1/v2/v3/v4 checkpoint 不能随意混用 |
| 保存 GIF 时确认是否需要成功样本、失败样本、楼梯样本或完整 episode | 用户经常需要具体 terrain 类型 |

## 10. 已知 caveats

| caveat | details |
| --- | --- |
| 随机种子不保证 IsaacLab/GPU RL 完全确定 | 相同配置曲线可能不同 |
| early curve 不能替代最终 eval | 最终至少跑足够 episodes 的普通 eval |
| `success_rate` 与 strict/termination 指标口径不同 | legacy 是每项分别写入；strict 是每 episode 一次；termination 是最近一次终止项 |
| `early_termination` 不一定全是失败 | 需结合 termination group 和 success 逻辑解释 |
| shared-lowcam 是新视觉分布 | 网络维度不变，但必须重新训练 |

## 11. 行为准则

| instruction | notes |
| --- | --- |
| 先读代码，再下结论 | 特别是 actor/critic 维度、obs 顺序、相机参数 |
| 先保护已验证结果，再新增实验 | 不覆盖 v3 no-goal 和已有 checkpoint |
| 修改后及时记录 README/RESULTS/相关 docs | 保持时间顺序，最新记录写在后面 |
| 发现自己误操作要立刻说明并检查副作用 | 例如 shell quoting 导致命令替换 |
| 最终回复要短，说明改了什么、验证了什么、还剩什么风险 | 用户更关心可执行结论 |

## 12. MotionBEV Preparation (2026-10-05)

- New tasks: `Instinct-Nav-G1-SRU-MotionBEVQuery-HardV3-OfficialParkour-v0` and
  `Instinct-Nav-G1-SRU-NoBEVRewardMatched-HardV3-OfficialParkour-v0`.
- MotionBEV uses the same six-channel 32x32 map and 64D late query, with dynamic
  lags 0.4/0.8 s, native dilation 3x3, and a CNN preserving an actual 8x8 feature
  grid. SRU dimensions stay 80/144, hidden size 512, one layer.
- Both remove `velocity_into_unobserved`; other HardV3 settings are inherited.
  Formal schedule stays 3000 static + 3000 dynamic, exact 50% cohort.
- `scripts/run_motion_bev_hard_v3_ablation.sh` is prepared, not running. Preserve
  prior tasks/checkpoints. MotionBEVQuery alone has no forecast loss. The separate
  MotionBEVFuture task adds the auxiliary loss described below.
- Eval GIFs must read policy BEV caches without calling map update functions.
  Earlier LearnedBEV statistics need read-only reevaluation before comparison.

### MotionBEV Future Auxiliary (Confirmed 2026-10-05)

- Task: `Instinct-Nav-G1-SRU-MotionBEVFuture-HardV3-OfficialParkour-v0`, with
  Dev/Play variants. Actor/critic inputs stay 8720/11857 and SRU stays 80/144.
- Prediction uses only actor BEV features. Its training-only decoder has 41,042
  parameters and two occupancy outputs (0.4/0.8 s); it does not enlarge SRU input.
- The 27D future_teacher observation is privileged supervision only. Training
  wrappers must forward it to extras, never concatenate it into actor/critic.
- PPO targets use actual future positions of source-observed cylinder slots in
  the source robot frame. Skip reset/recycle windows, rollout tails and sources
  with no observed active object. Pure static training has no forecast loss.
- Coefficient 0.05, weighted BCE + Dice, BCE positive weight 20, sampled batch
  256; log auxiliary loss, soft IoU, positive rates and conditional validity.
- A 32-env, two-update smoke saved model_0.pt/model_1.pt with nonzero labels and
  updated head weights. Formal training started from zero on 2026-10-05 in
  detached screen `motion_bev_future_20261005_fixed`; 4096 envs, 6000 iterations,
  seed 72, WandB `ablation_sru`, run ID `0ayx5lay`. Iteration 0 saved model_0.pt.
- Completed 6000 updates on 2026-10-08; final checkpoint model_5999.pt. Seeded
  static/dynamic100 eval completed in logs/motion_bev_future_eval_retry_20261008:
  19200 episodes per condition, SRU-reference success 0.837396 / 0.618229,
  plus ten full-episode failure GIFs per condition. No eval remains running.
- Seeded startup exited without summary; preserve its logs. Statistics-only eval
  must not render/cache image panels, and progress must use threshold crossings.
- Future-task eval must keep use_future_occupancy=True for strict weight loading
  while disabling lambda_future and future_teacher. Use explicit --terrain_seed
  for paired comparisons; --seed alone historically did not control terrain.
- Active run directory: `logs/rsl_rl/g1_sru_motion_bev_hard_v3/2026-10-05_17-40-58_motion_bev_future_static3000_dynamic3000_mixed50_seed72_20261005_173708/`.
- Cylinders inherit the previous run's geometric collision termination, NOT
  physical contact forces (`obstacle_collision_enabled=False`). Do not describe
  this setup as physical cylinder collision.
- Initial startup `...seed72_20261005_173010` was stopped to fix WandB same-step
  scalar drops; retain its files, but use the restarted run for formal results.
- Prior tasks, frozen low-level weights and PPO return signatures remain intact.
  Validate deployment export separately before claiming new-checkpoint support.

### MotionBEV ConvLSTM (2026-10-08, Training)

- New MotionBEVConvLSTM-HardV3-OfficialParkour task family uses
  ActorCriticBEVConvLSTM. Preserve the old MotionBEVFuture family.
- Actor-only dynamic CNN -> 32x8x8 ConvLSTM -> 64x8x8 projection joins the
  static spatial CNN before the original SRU query. Critic remains unchanged.
- Actor obs 8723: original 8720 followed by x/y/yaw odometry, used only for ego
  warp. Critic 11857, SRU inputs 80/144, one 512D layer. Adds 91,072 parameters.
- Store/replay spatial hidden/cell/pose with PPO trajectories. Actor and critic
  state lists may have different lengths; do not allocate fake critic states.
  Do not supervise temporal
  forecasting with independent frames or mutate online state during PPO updates.
- Forecast decoder reads dynamic memory only. Supervision, losses, rewards,
  terrain, obstacles and 3000-static/3000-mixed50 schedule are inherited unchanged.
- Exact simulation odometry is not deployment-ready localization. New checkpoint
  export/runtime support has not been validated. Smoke validation is not a
  performance result. User authorized formal training on 2026-10-08.
- Validation: 51 focused tests and 32-env PPO smoke passed, saved model_0.pt /
  model_1.pt, with nonzero forecast labels/loss and updated temporal weights.
- Formal training started from zero in detached screen
  `motion_bev_convlstm_20261008`: 4096 envs, 6000 updates, seed 72, WandB
  `ablation_sru`, run ID `maf3gdnr`. First update and model_0.pt verified;
  obstacles disabled during the initial static phase.
- Run: `logs/rsl_rl/g1_sru_motion_bev_hard_v3/2026-10-08_15-28-36_motion_bev_convlstm_static3000_dynamic3000_mixed50_seed72_20261008_152406/`.
- Stdout: `logs/background/motion_bev_convlstm_static3000_dynamic3000_mixed50_seed72_20261008_152406.log`.
