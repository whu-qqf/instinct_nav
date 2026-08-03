# 论文启发与设计备忘:FocusNav / GuideWalk → instinct_nav

> 日期:2026-08-03
> 状态:设计讨论稿,未进入实现。所有代码改动需用户明确批准后执行。

本文档汇总两篇 arXiv 论文(FocusNav、GuideWalk)对 `instinct_nav` 项目的启发,以及围绕这些启发展开的架构讨论结论。文中涉及本项目代码的地方均已核对过实际实现(`ActorCriticSRU`、`observations.py`、`g1_instinct/navigation_env_cfg.py` 等)。

---

## 0. 项目现状速览(讨论基准)

| 项目 | 状态 |
| --- | --- |
| 冻结低层 | G1 parkour checkpoint `artifacts/low_level/g1_parkour/model_30000.pt`,50 Hz,消费速度命令 + 深度历史 |
| 高层 SRU | 5 Hz,输出 `[vx, vy, wz]`,SRU 记忆 + scene token cross-attention |
| 当前最好结果 | v3 no-goal scene token:普通 eval overall success 0.916(1000 ep) |
| 当前主线 | ScoreVis 连续可见性置信度(2026-08-03 启动,检查时 ~iter 1063:success_rate 0.875、base_contact 0.082、large_pitch 0.077、noise_std 0.314) |
| 已部署链路 | ONNX 高层导出、MuJoCo sim2sim、真机 onboard runtime |

相机与地图事实(已核对):

- 相机 `NoisyGroupedRayCasterCameraCfg`:64×36,FOV 89.51°×58.29°,挂 torso;
- 高层可见性地图:6.4m×6.4m、64×64 格(0.1m/格)、深度有效 0.5–8m、pixel_stride 2;
- scene token:8×8 个 token,每 token 0.8m×0.8m patch,6 维特征 `[x_center, y_center, visibility_score, depth_mean, depth_edge, frontier]`,展平 384 维;
- 相机 raycast `mesh_prim_paths=["/World/ground"]`(目前只有地面);
- `base_contact` 终止只检测 `torso_link` 接触;`large_pitch_angle` 阈值为 1.0 rad;
- 当前 `rewards.py` 仅有 action_rate / lateral / rot / reach_goal / backward,`obstacle_proximity` 已不在。

---

## 1. 两篇论文要点

### 1.1 FocusNav(arXiv:2601.12790)

定位:Unitree G1 人形局部导航,空间选择性注意力框架。训练范式:先用特权信息(完整地图+目标)训 GuideOracle,视觉策略再通过 BC 学习。

核心模块:

- **WGSCA(Waypoint-Guided Spatial Cross-Attention)**:预测一串无碰撞 waypoint,以 waypoint 为 query 对 BEV 地图特征做 spatial-aligned cross-attention。近 waypoint 注意力落在脚边,远 waypoint 沿预期轨迹延伸,实现"预测性避障"。
- **SASG(Stability-Aware Selective Gating)**:proprio 过 MLP 出 Gumbel-Softmax 二值门,稳定性度量 `S_m = 1/(1 + k1|φ_xy|² + k2|ω_xy|²)`,cross-entropy 监督。门关闭时只保留近场 waypoint 特征(`m_h = m_1 + g·Σ_{k=2..N} m_k`)。
- **Backward 预测范式**:waypoint 从目标倒推回机器人(q_N → … → q_1),减少自回归复合误差,保证 goal-conditioned 推理链。
- **Traversability 解码器**:从 BEV 特征重建可通行图,BCE 监督,使视觉表征 task-relevant。

结果(自报):动态+非结构化场景 success 87.02%,对比 Gallant 50.32%、PGCA 63.67%。注意 tradeoff:动态非结构化下完整版碰撞频率略高于 WGSCA-Only(SASG 优先保稳)。

### 1.2 GuideWalk(arXiv:2606.10449)

定位:导航 + 运动学统一端到端框架。核心是**两阶段训练 + 复合 teacher**。

- **DWA 速度引导**:经典动态窗口法,对候选 (v, ω) 做 rollout,按 heading/dist/velocity 加权打分,取 `V_r = V_s ∩ V_d ∩ V_a` 中最优命令,作为显式速度引导;
- **复合 teacher**:DWA 导航引导 + 特权 locomotion teacher,蒸馏成一个学生策略;
- **阶段 1:DAgger 蒸馏**(学生状态分布下模仿 teacher 动作);
- **阶段 2:PPO + 辅助 BC loss**:`L = L_PPO + λ·L_BC`,在线轨迹上对齐 teacher 动作;
- **差分驱动假设**:只用 (v, ω) 而非全向 [vx, vy, wz],因为 locomotion 对前向/转向跟踪好、横向差;
- 奖励示例:0.5m 内到达给常数 1,否则朝向与目标夹角 <105° 时给 `4.5·v_forward`。

关键消融(自报):

| 消融 | 现象 |
| --- | --- |
| w/o 导航引导 | 障碍场景 NSR 明显下降,轨迹振荡、绕路 |
| w/o locomotion teacher | 复杂地形稳定性崩 |
| w/o 阶段 1(纯 RL) | 最终性能相近但训练时间显著变长(单次 update 0.24s vs 0.08s) |
| w/o 阶段 2(只蒸馏) | 障碍场景 NSR 从 99.0% 掉到 18.3%(分布偏移) |

---

## 2. 对 instinct_nav 的四条启发

### 2.1 SASG 稳定性门控

**动机**:当前终止统计中 base_contact 0.082、large_pitch 0.077,稳定性是主要失败来源。

**落地形式**(单相机可用,不需要 LiDAR):

- 门控的"近/远"来自你地图本身:6.4m 地图、8×8 token、每 token 0.8m,token 中心从机器人附近铺到 ±3.2m;
- 不稳时只保留机器人周围 ~1–1.6m 的 token 环(或 waypoint 中的 q1),其余 token/waypoint 特征置零;
- 监督:stability metric `S_m = 1/(1 + k1|φ_xy|² + k2|ω_xy|²)`,门与 S_m 做 cross-entropy。

**额外理由**:远端 token 多为历史帧累积的 memory,姿态不稳时 ego-motion warp 本身不可靠,截断远端同时丢弃了最不可信的信息。

**注意**:FocusNav 自报在动态场景下 SASG 会略增碰撞频率。若避障是硬目标,SASG 应排在避障能力之后,或与 waypoint query 配合。

### 2.2 Waypoint-guided query(当前讨论最深入的一条)

见第 4 节完整设计。

### 2.3 DWA 速度引导 + BC 两阶段训练

**价值**:显式速度引导作为"时间一致、朝向目标"的中间表示,正则化策略行为,减少轨迹抖动;蒸馏 warm start 加快收敛;在线 BC 项防止纯 RL 走偏。

**落地形式**:

- 仿真内用 DWA(深度栅格 + 差分运动学 rollout)每步生成专家 (v*, ω*);
- 阶段 1:高层策略 BC 模仿;
- 阶段 2:现有 PPO + `λ·‖a − a*‖²`(在线轨迹上算)。

**与现有 MDPO/DML 的关系**:互补。MDPO 是策略间互蒸馏,这里增加"经典规划器 → 策略"蒸馏。

**可选实验**:高层动作去掉 vy 或加横向速度惩罚(对应 GuideWalk 的差分驱动假设)。

### 2.5 DAgger 在线蒸馏(与 2.3 配套,需单独说明)

GuideWalk 的“阶段 1”不是离线 BC,而是 **DAgger**——这是它能在学生自身状态分布下学好的关键,值得单独拎出来。

**DAgger 是什么**:

- 学生策略(高层 SRU)在线与环境交互,收集**自己诱导出的状态分布**下的轨迹;
- 对每个访问到的状态,查询冻结的复合 teacher 得到目标动作;
- 新样本不断聚合进数据集(逐步累积),再迭代优化 min ||π_s(o) − π_t(o~)||²;
- 与离线 BC 的本质区别:BC 用 teacher 的静态数据集,存在分布偏移(学生动作偏差 → 到达 teacher 没见过的状态 → 错误放大);DAgger 用学生自己的状态分布采样,天然覆盖学生实际会遇到的场景。

**对 instinct_nav 的映射**:

| 角色 | 本项目对应 |
| --- | --- |
| 学生策略 | 高层 SRU 策略(输出 [vx, vy, wz]) |
| teacher(速度层) | DWA 规划器(便宜、可并行、无需训练) |
| teacher(复合层,可选) | DWA → 冻结低层 parkour → 关节动作;蒸馏到高层时主要用速度层输出 |
| 状态分布 | 高层 5Hz rollout 中采集 |

**为什么适合本项目**:

- teacher 是经典规划器/冻结策略,**无需训练、便宜稳定**,可以逐 step 在线调用;
- 项目已有 MDPO/DML(策略间互蒸馏),DAgger 的不同点是 teacher 为规划器,两者可共存;
- 配合低层冻结设计:蒸馏目标只到高层速度命令,不触碰冻结低层,风险可控。

**风险与注意**:

- teacher 上限决定蒸馏上限:DWA 是局部规划器,对复杂死胡同/全局路径可能不优(A* waypoint 可以作为全局引导补偿);
- 动态障碍下 DWA 的 rollout 模型决定其响应速度;
- 建议 DAgger 只做 warm start 初始化,后续仍用 PPO + BC 微调(对应 GuideWalk 的 w/o Stage 2 → 18.3% 教训:只蒸馏不微调会因分布偏移/无碰撞反馈而退化)。

### 2.4 Traversability 辅助解码头

**是什么**:训练时给 scene token 特征加一个轻量解码头,输出"哪里能走"的逐格图(0~1 可通行概率);推理时摘掉,零部署成本。

**标签免费**:仿真里地形已知,`_height_field_valid_mask` / `_height_field_spawn_mask`(goal 采样在用)就是现成标签;或按高度跳变/坡度阈值构造。按机器人当前位姿采样到局部 BEV 坐标、与地图对齐。

**为什么有用**:visibility map 记录的是"射线打到哪里"(地面、墙、障碍都会打中),分不清可通行地面与不可通行障碍;辅助任务逼 encoder 学会区分,让 token 特征更有判别力。

---

## 3. 动态障碍避障

### 3.1 FocusNav 的做法(已确认)

- 训练环境包含密集静态柱林 + **随机轨迹移动的红机器人实体** + 人群密集场景;
- 评测单独分 "Dynamic Obstacle Environment",动态+非结构化下 success 87.02%(对比 Gallant 50.32%、PGCA 63.67%);
- 论文未公开移动障碍的具体实现(轨迹生成、速度、避碰、重生机制),需要自行设计。

### 3.2 落地需要的四件事

1. **训练侧加移动障碍实体(IsaacLab)**:动态 prim(圆柱/胶囊/简化人形)+ 脚本化运动(随机 waypoint 折线巡逻、正弦摆动、go-to-point);**相机 `mesh_prim_paths` 必须加入障碍 prim**(目前只有 `/World/ground`);障碍要有碰撞体,并在接触传感器上单独识别"与障碍碰撞"。
2. **奖励/终止补齐**:与障碍距离惩罚(或接近速率惩罚)、撞上障碍的终止/强惩罚、绕开障碍仍朝目标的 shaping。当前 rewards.py 无 obstacle_proximity,需重新加。
3. **高低层联动验证**:高层 5Hz + 观测延迟,动态障碍快时可能反应不过来;低层冻结 parkour 有自己的 8 帧深度 history,但训练域无移动障碍,反应未知。先 MuJoCo 验证(rich_course.json 已支持 box 进深度射线与碰撞,扩展成移动 box 即可)。
4. **MuJoCo / 部署侧同步**:移动障碍、碰撞检测、指标(碰撞频率、障碍最小距离)对齐。

### 3.3 架构级坑:幽灵残留

可见性地图是"历史累积 + 仅 ego-motion 补偿"的显式记忆:

- 二值版本对移动障碍永不衰减,会留下旧位置残影;
- ScoreVis 0.995 衰减很慢(命中后约 64 帧掉到一半);
- FocusNav 用当前帧点云 BEV,无此问题。

候选对策(均需用户拍板):

- scene token 增加"当前帧可见"通道;
- 动态场景使用更快的衰减;
- waypoint 预测头直接吃当前帧深度(不依赖残影地图)。

---

## 4. Waypoint-guided query 设计(讨论定稿)

### 4.1 生成方式:训练时 A* 特权标签 + 推理时学习预测

训练时(标签来源):

1. 从地形高度场构造占用网格(高度跳变/坡度超阈值 → 不可通行),复用 `_height_field_valid_mask` / `_height_field_spawn_mask`;
2. 网格上跑 A*(64×64,5Hz 开销可接受),从机器人到目标(世界坐标已知)得无碰撞路径;
3. 沿路径等距/变距采样 N 个点,**从目标倒推回机器人**(backward),转换到机器人 yaw 对齐局部坐标系——与 scene token 的 `x_center/y_center` 同一坐标系。

推理时:

- 小预测头,输入 `[SRU hidden, proprio(含 goal), 深度特征]`,输出 N×2 局部坐标;
- 训练监督:MSE vs A* waypoint;可选 FocusNav 式 latent consistency 正则;
- 部署:预测头随 ONNX 导出(额外输出或单独网络)。

备选(不学习):DWA 在深度栅格上 rollout,取最优轨迹采样点为 waypoint,训练/推理同源,但无 teacher-student gap、上限取决于简单规划器。

**避免退路**:waypoint 必须来自真正无碰撞规划,不能只是"目标方向+距离"的几何插值(会退化成 v1/v2 失败的 per-token goal feature)。

### 4.2 数量与间距

**推荐:N=4,近密远疏,`0.7 / 1.4 / 2.2 / 3.0` 米。**

理由:

- 覆盖刚好匹配 ±3.2m 地图,4 个以 0.8~1.0m 间隔铺满;
- SASG 门控粒度合适(q1 近场安全点,q2~q4 中远端规划点);
- backward 自回归链短、复合误差小;
- 预测头仅 8 维输出,开销可忽略;
- 5 query(4 waypoint + 原 1 query)对 64 token,attention 计算量几乎不变。

近密远疏的原因:近处是立刻决策 + SASG 保留区;远处是趋势锚。

验证:做成 task 配置参数(如 `scene_token_query_num_waypoints=4`),先跑通链路,再 ablation(3 vs 4 vs 6)。

### 4.3 query 组成:不只是坐标

推荐 `waypoint_query = 坐标 PE + 预测头 latent + 序数 embedding`,对应 FocusNav 的 `m_k = Attn(x̂_k + PE(q̂_k), F_bev + PE, F_bev)`:

1. **坐标 PE(空间定位)**:sinusoidal PE,(x_k, y_k) 与 token 同坐标系,让 attention 分数反映真实物理近邻。注意现有 `scene_token_position_embedding` 是固定可学习参数 [1,64,dim],**不能用于连续变化的 waypoint**,waypoint 侧需坐标驱动 PE;
2. **预测头 latent(语义)**:waypoint 生成网络内部逐点 latent,天然编码目标条件(距离/朝向残差)、路径结构(曲率/转弯)、记忆上下文(用了 SRU hidden);这是 query 的"内容"部分;
3. **序数 embedding(层级)**:第 k 个 waypoint 的序数编码,供 attention 分层(近点细节/远点趋势)与 SASG 截断(q2~q4)。

**避免的坑**:不要把 waypoint 附近 token 特征采样进 query("看图说话"),会让 attention 自指、破坏预测头存在的意义。

### 4.4 与 SRU 记忆的关系(已核对代码)

实际数据流:`depth → attn_image_net → [image, proprio] → memory_a(SRU) → [hidden, proprio] → scene_query_proj(单 query)→ cross-attn(token projector + 位置编码)→ concat → actor MLP`。

结论:

- **结构性无影响**:scene attention 在 SRU 之后,输出只拼接进 actor MLP,不回流入 `memory_a`,waypoint query 不会改变 SRU hidden/cell 更新;
- **唯一影响通道是共享 encoder 梯度**:waypoint loss 梯度回流进 `attn_image_net` 等共享层,间接改变 SRU 消费的图像特征分布。对策:waypoint loss 先 stop-gradient / 小权重 / 独立网络;
- **不要替换,要追加**:保留现有 `[SRU hidden, proprio]` query,waypoint 作为附加 query;记忆继续驱动 attention(FocusNav 的 GRU 也是"门控截断 waypoint 时靠记忆兜底");
- **分工**:SRU 记忆 = 时间性(去过哪、见过什么、运动历史);waypoint = 空间规划(当前到目标的无碰撞路径),互补不冗余;
- **可选增强**:waypoint 头输入含 SRU hidden,让记忆参与 waypoint 生成,提高跨步稳定性、减少抖动。

---

## 5. 优先级与执行顺序(当前判断)

1. **动态/静态障碍环境基础设施**(移动 prim + raycast 接入 + 碰撞检测 + 避障奖励 + MuJoCo 验证)—— 地基,先 dev task 验证低层对突然障碍的真实反应;
2. **Waypoint-guided query**(第 4 节设计)—— 与 v3 结论一脉相承,是预测性避障的关键;
3. **Traversability 辅助解码头** —— 表征增强,成本低;
4. **SASG 稳定性门控** —— 压 base_contact/large_pitch 有用,但动态场景有碰撞 tradeoff,排在避障能力之后;
5. **DWA 速度引导 + DAgger 蒸馏 + PPO+BC 两阶段**(2.3 / 2.5)—— 收益大但改动训练管线,最后做。

不做的事:不动正在跑的 ScoreVis 训练;不动冻结低层;新想法一律开新 ablation task,先 smoke test 再正式训练;不改 token 语义(幽灵残留对策)需用户拍板。

---

## 6. 待用户决策的点

- [ ] waypoint 走"预测头(A* 标签)"还是"DWA 确定性"路线;
- [ ] waypoint loss 是否先用 stop-gradient 隔离;
- [ ] 蒸馏用离线 BC 还是 DAgger 在线蒸馏(warm start 阶段);
- [ ] 动态障碍是否作为第一优先级投入;
- [ ] 幽灵残留对策(当前帧可见通道 / 更快衰减 / 交给 waypoint 头);
- [ ] 是否接受高层动作空间收敛(去 vy / 横向惩罚)实验;
- [ ] 每个想法的实现范围与顺序。

---

## 7. 修改纪律

按用户要求:没有明确要求修改的功能一律不修改;没有要求添加的功能一律不加入;涉及修改/添加但不确定的地方先询问,得到答复后再行动。本文档仅为设计备忘,不构成任何代码改动。
