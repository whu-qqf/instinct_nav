# 组会 PPT 结构（G1 高层导航 + 动态障碍避障 暑假进展）

> 故事线：**问题 → 成果 → 瓶颈 → 诊断 → 新方向 → 待讨论**
> 约 15 页；每页给「内容 bullet」+「配图/表格建议」。

---

## 一、开场

### P1 封面
- 题目：《G1 人形机器人高层导航与动态障碍避障——暑假进展》
- 报告人 / 日期
- （可选）放一张机器人 + 移动障碍的示意 GIF/截图

### P2 目标 & 一句话总结
- 目标：让 G1 用「冻结 parkour 低层 + 高层 SRU 导航」，在富地形（楼梯/台）里、有移动障碍的情况下到达目标且不撞
- 一句话总结：**静态导航已做成（0.91），当前卡点是"感知移动障碍"，正在满难度做感知×控制解耦消融**
- 配图：一条「进度条」或「时间线缩略图」暗示全貌

---

## 二、背景与系统

### P3 任务背景
- 两层控制：高层 SRU 导航（5Hz 出 SE(2) 速度）+ 冻结 InstinctLab parkour 低层（50Hz，官方 ONNX）
- 三个难点：富地形（楼梯/高台）+ 动态障碍（移动 cuboid）+ 观测延迟（0–2 步）
- 配图：一张「高层 → 低层」的简洁示意

### P4 系统架构
- 用 `docs/figures/mask_cnn64_gated_add_paper_architecture*` 那套图（depth backbone → SRU → actor；critic 特权输入；frozen 低层）
- 要点：观测走「延迟视觉 packet」，depth/mask 同帧对齐
- 配图：已有的论文风架构图，标红「本暑假改动的模块」

---

## 三、进展时间线

### P5 五阶段总览（时间线表）
| 阶段 | 时间 | 一句话 |
|---|---|---|
| ① 复现+独立化 | 7/15–19 | 对齐师兄链路，项目独立 |
| ② 视觉融合消融 | 7/20–26 | scene-token query 成主视觉方案 |
| ③ 导航配方 v3 | 8/12–20 | **静态导航 0.91** |
| ④ 动态避障诊断 | 8/29–30 | 定位"感知表示不了移动物体"根因 |
| ⑤ Mask 感知方案 | 9/5–20 | 感知×控制解耦，满难度消融收敛中 |

- 配图：横向时间线，标注每个阶段的「产出」

---

## 四、核心成果（重点）

### P6 阶段③ 导航配方 v3 ——「静态导航做成了」
- v3 配方 = tanh 速度映射 + 线积分 `velocity_into_unobserved` 惩罚 + `stair_centerline` + DynamicMapCross + 分级动态障碍课程
- 数字（run `...tanh_9000_v3`，9000 iter）：
  - success **0.9105**（含 50% env 动态障碍）
  - base_contact 0.041 / large_pitch 0.082 / obstacle_collision 0.155
- checkpoint：`model_10499.pt`

### P7 阶段③ 关键消融（对比图）
- tanh（✓ 0.91）vs exp_gate（✗ 0.60，base_contact 冲到 34%）vs 半球 unobserved（✗ 0.60，"前进税"）
- 结论：这几个负结果是配方收敛的关键，讲出「为什么是这套」
- 配图：success / base_contact 双柱状图，tanh 高亮

### P8 阶段④ 动态避障引入 → 暴露问题
- 满难度（difficulty=1.0）下 Avoid 基线 collision **0.484**、success 0.90（虚高）
- 两个问题：① 贴脸必撞（无接近预警）② success 虚高（碰撞不计失败）
- 配图：一张「障碍突然出现在正前方」的失败 GIF 截图

### P9 阶段④ 根因诊断（全场转折，多停 1 分钟）
- reactive spawn（≥5m 可见接近）→ collision 48%→21%；record_failure → success 0.90→0.72（诚实化）
- **根因**：visibility map 是「只增不减」的 seen/unseen 记忆（`torch.maximum`），**移动障碍进入已看区域不会变黑** → `velocity_into_unobserved` 对动态避障失明
- Demote 双向课程：**负结果**（满难度 eval 与 monotonic 打平），已否决
- 配图：一张「障碍移动进已看区域、map 却不变黑」的示意图（对比 depth 已看到障碍 vs vis map 仍白）

### P10 阶段⑤ 感知方案设计（感知×控制解耦）
- 三模式：
  - **DirectMask**（oracle 上限）：直接喂真值 mask，模拟 YOLO 部署契约
  - **MaskCNN**（learned encoder）：CNN+self-attention 从 depth/mask 学编码
  - **NoMask**（基线）
- 配图：三模式架构对比（或复用 P4 图改三份）

### P11 阶段⑤ 满难度消融（数据表）
- 满难度 hard + allenv + 1.6m 高障碍：
  - MaskCNN64 gated_add **0.615**
  - MaskCNN32 concat **0.698**
  - NoMask control（跑中）
- 收敛点：**DynamicRiskBEV**——砍掉独立 MaskCNN，把 GT mask 并进 visibility map 作「黑色高风险覆盖区」，恢复原始 SRU 宽度
- 配图：success / collision 双指标对比表

---

## 五、当前状态 & 下一步

### P12 当前正在跑
- NoMask control（补齐 `Δmask = MaskCNN − NoMask` 对比）+ DynamicRiskBEV 新方向
- 重评估计划：5 个关键 checkpoint 用松口径（`--sru_ref_success`）串行重跑

### P13 下一步计划
1. NoMask 跑完 → 算出 mask 真实贡献
2. 决定：回 MaskCNN 还是走 DynamicRiskBEV
3. 满难度 + 多 bank 的正式评估，出论文口径数字

---

## 六、待讨论

### P14 三个待讨论问题
1. 「visibility map 表示不了移动物体」这个诊断完整吗？还有没有别的根因（如 5Hz 对快速障碍的物理反应上限）？
2. 感知解耦的 oracle（DirectMask）上限能到多少？这个实验设计站不站得住？
3. DynamicRiskBEV vs 回到 MaskCNN，哪个是更合理的收敛点？

### P15 总结 / 致谢
- 回顾故事线一句收尾
- 列出下一步想从组会得到的反馈

---

## 附：排版提醒
- **P5 / P7 / P11** 信息密度高，用图/表，别纯文字
- **P9 根因诊断** 是转折点，值得配一张「map 不变黑」示意图
- **负结果（exp_gate / Demote）如实讲**，体现"系统性收敛"而非"试错"
- 口径统一说明：本文 success 均为 **SuccessRateTracker 松口径（`--sru_ref_success`）**，不用严格排他口径
