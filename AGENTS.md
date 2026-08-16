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
| `Metrics/robot_goal/success_rate` | 成功率主指标之一 |
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
| `success` 与 `success_rate` 可能口径不同 | 需要查 command/eval logger 具体实现 |
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
