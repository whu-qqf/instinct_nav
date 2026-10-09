# Visibility Query 与 Scene Token Query 对比

## 对比对象

- `2026-07-20_21-14-26_g1_sru_visibility_query_v1`
- `2026-07-23_20-49-31_g1_sru_scene_token_query_v1`

结论来自各 run 保存的 `params/env.yaml`、`params/agent.yaml`、
`params/senior_diagnostics.json`、checkpoint state dict 和 TensorBoard event，
而不是由当前默认配置反推。

## 配置差异

| 项目 | Visibility Query | Scene Token Query v1 |
| --- | --- | --- |
| task | `Instinct-Nav-G1-SRU-Visibility-Query-v0` | `Instinct-Nav-G1-SRU-SceneToken-Query-v0` |
| actor obs | `6672 = 16 + 2560 + 4096` | `2960 = 16 + 2560 + 384` |
| critic obs | `5713` | `5713` |
| BEV 输入 | 完整 `1 x 64 x 64` visibility map | `64 x 6` scene tokens |
| 环境侧压缩 | 无 | 每个 `8 x 8` BEV patch 汇聚成 1 个 token |
| token 特征 | 网络从二值图学习 | `[x, y, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal]` |
| 网络侧编码 | 3 层 stride-2 CNN，得到 `64 x 8 x 8` | token MLP `6 -> 64 -> 64` |
| attention | 512D SRU hidden + 16D proprio 生成 64D query，对 64 个 CNN token 做 4-head cross-attention | 同一 query，对 64 个 scene token 做 4-head cross-attention |
| 场景分支输出 | 64D | 64D |
| Actor SRU | `80 -> 512` | `80 -> 512` |
| actor 后融合 | `[512,64] -> 576 -> 512` | `[512,64] -> 576 -> 512` |
| 场景分支参数 | `99,264` | `80,320` |
| checkpoint 总模型参数 | `3,856,023` | `3,837,079` |

## 保持一致的部分

- 4096 environments、3000 iterations、seed 72。
- PPO：5 epochs、4 mini-batches、learning rate `1e-3`、`gamma=0.995`、`lam=0.95`。
- actor proprio/goal/action 共 16D。
- depth observation `64 x 5 x 8 = 2560`，经相同 depth attention 得到 64D。
- Actor SRU 输入 80D、hidden 512D；scene/visibility 都是 SRU 后期 query fusion，不进入循环记忆。
- critic observation 5713D；critic 使用 proprio、depth、privileged height scan 和 time，不使用 visibility map 或 scene token。
- actor head、critic head、冻结 G1 parkour 低层控制器相同。

## 需要注意的非网络差异

保存的 `env.yaml` 还显示低层 depth history observation 的函数入口发生了变化：

- Visibility Query：`instinctlab.envs.mdp.observations.exteroception:delayed_visualizable_image`
- Scene Token Query：`instinct_nav.navigation.mdp.observations:delayed_visualizable_image`

两次 runtime diagnostics 的低层相机 shape 都是 `36 x 64`，配置 diff 中没有其他环境差异，
但由于保存目录没有源码快照，不能仅凭配置证明这两个函数在当时逐元素完全等价。

## 训练记录

- Visibility Query event 记录到 iteration 2134，最后训练窗口 success `0.8802`，峰值 `0.9041`；没有对应正式 eval 文件。
- Scene Token Query 完成 iteration 2999，最后训练窗口 success `0.9095`，峰值 `0.9225`；保存的 1000-episode eval success 为 `0.875`。
- 因为前者没有训练满 3000 且没有同口径 eval，不能仅用这两个数字断言 Scene Token 结构一定更优。

## 图

- [SVG](figures/visibility_vs_scene_token_query_architecture.svg)
- [PDF](figures/visibility_vs_scene_token_query_architecture.pdf)
- [高分辨率 PNG](figures/visibility_vs_scene_token_query_architecture.png)

