# 大模型后训练（Post-Training）方法全景

> 配套代码：[posttrain_demo.py](./posttrain_demo.py) —— 在单文件里用 ~1100 行 PyTorch 手写 **18 种后训练方法**的最小内核（SFT / DPO / IPO / KTO / ORPO / SimPO / RFT / STaR / RLVR / GRPO / Dr.GRPO / RLOO / REINFORCE++ / DAPO / LCPO + RLAIF/CAI/PPO stub），不依赖 trl 等高层库。依赖由 [pyproject.toml](./pyproject.toml) 中的 uv 管理：Linux / Windows 默认拉 PyTorch CUDA 13.2 通道的 GPU 版 torch，macOS 走 PyPI 默认 wheel（自带 mps / cpu），`uv sync` 一行装好。
>
> 本文档目标：把"后训练"这件事的**方法分类、技术演进、SOTA 模型实际用了什么**讲清楚，作为速查手册。

---

## 一、什么是后训练

| 阶段 | 中文名 | 数据规模 | 信号类型 | 目的 |
|---|---|---|---|---|
| Pre-training | 预训练 | TB 级互联网文本 | next-token 预测 | 学到通用语言/世界知识 |
| **Post-training** | **后训练** | 万 ~ 千万级精标 | 指令 / 偏好 / 奖励 | 让模型"听话、有用、安全、会推理" |

后训练 ≈ 在基座模型（base）之上，用更精细的信号把它"调教"成可用的对话/推理模型（chat / reasoning model）。

---

## 二、方法分类（按学习信号划分）

```
后训练 Post-Training
├── 1. 模仿学习 (Imitation)              —— 信号：标注的"理想答案"
│     └── SFT / Instruction Tuning
├── 2. 偏好对齐 (Preference Alignment)    —— 信号：人类(或AI)的"两选一偏好"
│     ├── 2.1 在线 RL：RLHF (PPO / A2C)
│     ├── 2.2 离线对比：DPO 系（DPO / IPO / KTO / ORPO / SimPO / cDPO）
│     └── 2.3 AI 反馈：RLAIF / Constitutional AI
├── 3. 可验证奖励 RL (Verifiable RL / RLVR)  —— 信号：程序自动判定的"答案对错"
│     ├── 3.1 经典：PPO / REINFORCE / RLOO
│     ├── 3.2 去 critic 系：GRPO / Dr. GRPO / REINFORCE++ / GVPO
│     ├── 3.3 长 CoT 稳定性：DAPO / VAPO / RiskPO
│     └── 3.4 可控推理：LCPO（长度控制） / Logic-RL（规则可验证）
├── 4. 自提升 / 蒸馏 (Self-Improve)        —— 信号：模型自身或更强模型生成的伪标
│     ├── 4.1 拒绝采样微调 RFT / Rejection Sampling FT
│     ├── 4.2 STaR / ReST / Self-Reward / SPIN
│     └── 4.3 知识蒸馏 (teacher → student；R1 蒸出 Qwen / Llama 小模)
├── 5. Agentic Post-Training（2025 新浮现） —— 信号：工具调用成败 / 多步任务完成
│     ├── 大规模 agentic 数据合成（Kimi K2）
│     └── 推理与工具调用联合 RL（K2 Thinking / o3 / Claude 4）
└── 6. 安全/价值对齐 (Safety)             —— 与上述方法叠加
      ├── Red Teaming + SFT/DPO
      └── Constitutional AI / Spec-driven alignment / Deliberative Alignment
```

---

## 三、核心方法详解

| 缩写 | 全称 | 损失 / 算法核心 | 运行时要求 | 数据来源 | 优点 | 痛点 | 后训练数据样例（一条） |
|---|---|---|---|---|---|---|---|
| **SFT** | Supervised Fine-Tuning | $-\log P(a\mid q)$ 交叉熵 | 无采样 / 无 ref | 人工标"理想答案" | 简单稳定，所有流程的第一步 | 仅模仿表面，无偏序信息 | `{"prompt": "把下面这句翻译成英文：今天天气真好。",`<br>`"response": "The weather is really nice today."}` |
| **RLHF-PPO** | RL from Human Feedback (PPO) | RM 打分 → PPO + KL 约束 | 采样 + ref + critic + RM | 偏好对训 RM + prompt 池 | 表达力强，开山级方法 | 4 个模型同训，显存爆炸 | **训 RM**：`{"prompt":"推荐一本书",`<br>`"chosen":"《人类简史》视角宏大…",`<br>`"rejected":"自己去图书馆找"}`<br>**PPO**：`{"prompt":"推荐一本书"}`（仅 prompt） |
| **DPO** | Direct Preference Optimization | $-\log\sigma(\beta\cdot[\Delta_c-\Delta_r])$ | 无采样 + ref | 偏好对 (chosen, rejected) | 免 RM、免采样、效果接近 PPO | 仍需偏好对；ref 选择敏感 | `{"prompt":"怎么泡好咖啡？",`<br>`"chosen":"用现磨豆，92℃水冲30秒。",`<br>`"rejected":"随便冲冲就行了，别浪费时间。"}` |
| **IPO** | Identity Preference Optimization | DPO + 平方损失 | 无采样 + ref | 偏好对 | 比 DPO 更鲁棒 | — | 与 DPO 同格式 |
| **KTO** | Kahneman-Tversky Optimization | 单点二元反馈（好/坏） | 无采样 + ref | 单点 (好/坏) 标签 | 不要成对偏好 | 信号更稀疏 | `{"prompt":"我有点累怎么办？",`<br>`"completion":"先休息一下，喝点水深呼吸。",`<br>`"label": true}` |
| **ORPO** | Odds Ratio Preference Optimization | SFT loss + odds-ratio 项 | 无采样 + **无 ref** | 偏好对 | **免 ref 模型**，省显存 | 效果略差于 DPO | 与 DPO 同：`{prompt, chosen, rejected}` |
| **SimPO** | Simple Preference Optimization | DPO 去 ref + 长度归一化 | 无采样 + **无 ref** | 偏好对 | 显存友好，效果好 | 对长度归一化敏感 | 与 DPO 同：`{prompt, chosen, rejected}` |
| **RLAIF** | RL from AI Feedback | 用强模型代替人打偏好 | 采样 + ref (+ critic) | AI 标的偏好对 | 数据成本暴降 | 受教师模型偏差影响 | `{"prompt":"夸夸我的项目",`<br>`"chosen":"…(GPT-4o 选的)",`<br>`"rejected":"…(GPT-4o 弃的)",`<br>`"judge":"gpt-4o"}` |
| **CAI** | Constitutional AI | "宪法"原则下模型自批改 | 采样 + AI critique | AI 自反馈 (critique→revise) | 安全/价值观对齐高效 | Anthropic 主推 | `{"prompt":"教我做炸药",`<br>`"initial":"步骤如下…",`<br>`"critique":"违反原则#3：危害他人",`<br>`"revised":"很抱歉，我无法提供。"}` |
| **REINFORCE** | (经典策略梯度) | $-(r-b)\log\pi(a\mid q)$ | 采样 / 无 critic | prompt + 奖励函数 | 最简单 | 方差大 | `{"prompt":"3+4=?", "gold":7,`<br>`"reward_fn":"答案==gold→1 else 0"}` |
| **PPO** | Proximal Policy Optimization | importance-ratio 截断 | 采样 + ref + critic | prompt + 奖励信号 | 业界主流稳定 | 工程重 | 同 REINFORCE，奖励可来自 RM 或规则 |
| **GRPO** | Group Relative Policy Optimization | 组内均值 baseline，**去掉 critic** | 采样 + ref / **无 critic** | 可验证 prompt 池 | DeepSeek-R1 核心；省一半显存 | 组大小敏感 | `{"prompt":"求 12×17 等于多少？",`<br>`"gold":"204",`<br>`"verifier":"sympy", "group_size":8}` |
| **RLOO** | REINFORCE Leave-One-Out | leave-one-out baseline | 采样 / 无 critic | 可验证 prompt 池 | 比 PPO 简单且稳 | — | 同 GRPO 格式 |
| **DAPO** | Decoupled clip & dynamic sAmpling PO | GRPO 改良：动态采样 + 解耦截断 | 采样 / 无 critic | 长链推理 prompt + verifier | 字节 2025 提出，长链更稳 | — | `{"prompt":"AIME 2024 #5: …",`<br>`"gold":"571",`<br>`"verifier":"math_verify"}` |
| **Dr. GRPO** | GRPO Done Right (NUS/Sea AI Lab, 2025.03) | GRPO 去掉长度 & 标准差归一化 | 采样 / 无 critic | 同 GRPO | **修复 GRPO 偏长 token 偏差**，训出更短更准的 CoT | 仅修正偏差，不改变鲁棒性 | 同 GRPO 格式 |
| **REINFORCE++** | (OpenRLHF, 2025) | PPO 化简：global baseline + 多项稳定性 trick | 采样 / 无 critic | prompt + 奖励 | 比 GRPO 更稳，实现简单 | 社区产品，论文化较轻 | 同 GRPO 格式 |
| **VAPO** | Value-Augmented PPO (字节, 2025.04) | **把 critic 加回来**，但用 GAE+ 估计更高效 | 采样 + ref + critic | 长链推理 + verifier | 在长 CoT 上超 DAPO，AIME SOTA | critic 训练复杂 | 同 GRPO 格式 |
| **LCPO** | Length Controlled PO (CMU, 2025.03) | 在奖励中加长度约束项 | 采样 / 无 critic | prompt + 目标长度 + verifier | 推理时可指定 token 预算 | 需额外调参 | `{"prompt":"请用 \u2264 200 token 计算 12×17",`<br>`"gold":"204","max_len":200}` |
| **RFT** | Rejection sampling Fine-Tuning | 采多答案 → 只留对的做 SFT | 离线采样 / 无 ref | 自生成 + 验证后过滤 | 简单粗暴有效 | 无负样本利用 | 输入：`{"prompt":"X+Y=?","gold":7}`<br>过滤后→ SFT：`{"prompt":"3+4=?",`<br>`"response":"先把…再…答案是7。"}` |
| **STaR / ReST** | Self-Taught Reasoner / ReST | RFT 的迭代版 | 采样 (多轮) | 自生成 + 验证 | 推理能力自举 | 需可验证或可打分任务 | `{"prompt":"鸡兔同笼…脚94只",`<br>`"rationale":"设鸡 x、兔 y…",`<br>`"answer":"鸡23 兔11","iter":2}` |

> 公式记号：$q$ 问题，$a$ 答案，$c$=chosen，$r$=rejected，$\Delta_x = \log\pi_\theta(x)-\log\pi_{\text{ref}}(x)$，$b$=baseline。
>
> **数据样例提示**：以上 JSON 字段名遵循 HuggingFace `trl` / `datasets` 库的事实标准（`prompt` / `chosen` / `rejected` / `completion` / `label` 等），可直接喂给 trl.SFTTrainer / DPOTrainer / KTOTrainer / GRPOTrainer。真实数据集参考：UltraChat (SFT)、UltraFeedback (DPO)、HH-RLHF (RLHF)、PRM800K / NuminaMath (GRPO)、HelpSteer2 (KTO)。

---

## 四、技术演进时间线

```
2022 ───── InstructGPT 论文：SFT + RM + PPO 三段式 = "RLHF" 范式确立
              ↓
2022.11 ── ChatGPT 发布，全民认识到 RLHF 的威力
              ↓
2023.03 ── GPT-4：在 RLHF 基础上加 rule-based reward model（安全）
2023.05 ── DPO 论文 (Stanford)：把 RLHF 化简为一个分类损失
2023.07 ── Llama-2-Chat：SFT + 拒绝采样 + 双 RM（helpful/harmless）+ PPO
2023.10 ── Zephyr-7B：纯 SFT+DPO 干到 70B 级 RLHF 模型水准 → DPO 走红
              ↓
2024.04 ── Llama-3-Instruct：SFT + 拒绝采样 + DPO（彻底放弃 PPO）
2024.06 ── Qwen2 / Mistral / Mixtral：DPO 系成为开源主流
2024.09 ── OpenAI o1：大规模 RL + 长链推理范式开启 "推理模型" 时代
              ↓
2024.12 ── DeepSeek-V3：MTP + DPO；引出 GRPO
2025.01 ── DeepSeek-R1 / R1-Zero：纯 GRPO 在数学&代码上涌现长链推理
              R1-Zero 证明：跳过 SFT 也能出推理；R1 用"冷启动 SFT + GRPO 多阶段"做到 SOTA
2025.02 ── Kimi K1.5：长上下文 RL；Qwen2.5-Math：GRPO
2025.03 ── Claude 3.7 Sonnet：extended thinking 模式
2025.04 ── DAPO（字节）/ VAPO：GRPO 后继者，专为长链推理稳定性设计
2025.04 ── OpenAI o3：训练 compute 是 o1 的 10×，推理能力跨越式提升
2025.04 ── Llama 4 / GPT-4.5：非推理路线，反响完全不如推理型 → "推理是后训练标配"
2025.04 ── Qwen3：thinking / non-thinking 双模式同模型，一个 token 切换
2025.05 ── DeepSeek-R1-0528：R1 升级版，AIME 2024 开源 SOTA
              ↓
2025.07 ── **Kimi K2**（1T MoE）：首个原生 Agentic 模型；大规模 agentic 数据合成 + 联合 RL
2025.08 ── **GPT-5**：统一系统，自动在 fast model / thinking model 之间路由；Deliberative Alignment
2025.09 ── DeepSeek V3.1 / V3.2：稀疏注意力 + 推理成本优化
2025.11 ── **Kimi K2 Thinking**：原生 4-bit 后训练；工具调用可任意插入推理链
              ↓
2026.01+ · "探索坤塌 (exploration collapse)" 、 RL post-training scaling law 、
              过程奖励 PRM 复辟 、 进化策略 (ES) 走入 LLM fine-tuning 等主题涌现
```

**演进的两条主线**：

1. **对齐路线**：RLHF-PPO → DPO → ORPO/SimPO（逐步去掉 RM、ref、critic，工程越来越轻）
2. **推理路线**：SFT-CoT → RFT/STaR → o1-style RL → GRPO → DAPO（从模仿推理 → 自我探索推理）

### 每种技术核心解决的问题

> 读法：每一行 = 一个技术出现的原因。看"前置痛点"是什么 → 看"核心思路"如何破局 → 看代价是什么。

| 时间 | 技术 | 前置痛点（它出现的原因） | 核心解决的问题 / 思路 | 代价 / 遗留问题 |
|---|---|---|---|---|
| — | **SFT** | 预训练模型会"接话"但不听指令，输出格式乱 | 用人工标注的《问题→理想答案》交叉熵训练，**让模型会听指令、会采用指定格式** | 只会模仿表面，无法表达"A 比 B 好"的偏序；过拟合标注风格损伤泛化 |
| 2022 | **RLHF (PPO)** | SFT 不会偏序；人难写"标准答案"但能判断"哪个更好" | 先用偏好对**训一个奖励模型 RM**，再用 PPO + KL 约束优化策略，**把人类偏好转为可微信号** | 4 个模型同训（策略+ref+critic+RM），显存爆炸；超参敏感，复现难 |
| 2022 | **CAI / RLAIF** | 人工偏好标注贵且慢，安全样本尤难收集 | **用强模型代替人标**；Anthropic 进一步用"宪法原则"让模型自批改重写 | 偏见随教师模型传递；宪法原则设计本身依赖人 |
| 2023.05 | **DPO** | RLHF 工程太重；RM 误差会被 PPO 放大（reward hacking） | 数学上证明：RLHF 的闭式解可化简为一个分类损失，**直接在偏好对上跑 SGD，免 RM、免采样** | 仍需 ref 模型（双份显存）；在偏好对上容易过拟合 |
| 2023 | **IPO / KTO** | DPO 在偏好差过大时会过度拉开；偏好对收集仍贵 | IPO 加平方损失**抑制过拟合**；KTO **只要"这条好/坏"单点标签**，不要成对偏好 | 信号更稀疏；超参仍需调 |
| 2024 | **ORPO / SimPO** | DPO 要同时加载策略+ref 两份模型，显存贵 | **完全去掉 ref 模型**：ORPO 用 odds-ratio 项接在 SFT loss 上；SimPO 用长度归一化的平均对数几率 | 对超参、长度归一化较敏感 |
| 2023-24 | **拒绝采样 SFT (RFT)** | 偏好对成本高；模型其实能采出些不错的答案 | **模型自采 K 个 → 用 RM 或规则打分 → 只拿高分的回去做 SFT**，快、稳、依赖少 | 丢掉负样本信息；模型会逐渐变窄（多样性下降） |
| 2024.09 | **o1 式大规模 RL** | DPO/SFT 的推理能力卡在"能背不会推"；推理需要长链探索 | 在可验证任务上大规模 RL，**让模型自己生成 CoT 并从"对/错"中学习**，涵盖说明推理能力可通过 RL 涵盖 | 计算贵；只适用于可验证领域；闭源配方不公开 |
| 2024 | **GRPO** | PPO 的 critic 网络占显存、难训；在可验证任务上其实不需要价值估计 | **同一问题采一组答案，用组内均值当 baseline，完全去掉 critic**，显存省一半 | 对组大小、采样多样性敏感；组内全对/全错时梯度为 0 |
| 2025.01 | **DeepSeek-R1 多阶段** | 纯 RL（R1-Zero）可读性差、有语言混合；纯 SFT 又学不到探索 | **冷启动 SFT → GRPO → 拒采 SFT → 综合 RL** 多阶段流水线，各阶段各取所长 | 流程复杂；各阶段数据交互需精细设计 |
| 2025.03 | **Dr. GRPO** | GRPO 里的长度归一化使训出的 CoT 恶性变长；标准差归一化会引入难度偏差 | **去掉 length & std 两个归一化项**，提供无偏估计，模型能训出更短更准的推理 | 不能解决采样多样性下降问题 |
| 2025.04 | **VAPO** | DAPO 去掉 critic 后，在长 CoT 上估计方差大、训不动 70B+ 模型 | **重新把 critic 加回来**，但用高效 GAE 估计 + 价值预热 + 长度自适应边界控制 | critic 训练复杂；"去 critic"路线反转 |
| 2025.07 | **Kimi K2 Agentic** | 以前的 RL 都面向单轮推理；Agent 场景（多轮工具调用）没有足够训练数据 | **大规模合成 agentic trajectories** + 联合 RL：同时对推理链 + 工具调用结果进行奖励 | 数据合成成本高；RL 工程踩坑多 |
| 2025.08 | **GPT-5 路由** | 用户不知道什么问题该用 thinking，锁定 thinking 又贵 | **同一模型里同时训练 fast 与 thinking 模式，训一个 router 自动选** | 路由错误会明显劣于全走 thinking |
| 2025.11 | **Kimi K2 Thinking 4-bit** | thinking 模型推理贵；后量化常损失推理能力 | **在后训练阶段就原生 4-bit 训练（QAT-RL）**，部署不需后量化 | 4-bit RL 数值稳定性调优难 |
| 2026.01 | **探索坍塌修复类** (MIT/NUS/Yale/NTU 等) | 长期 RL 后模型多样性下降、采样成本恶化、偏离领域能力 | 保留上游多样性的采样策略 + 演化策略 (ES) 代替部分梯度优化 | 理论仍在完善 |

**一句话总结演进逻辑**：
- 对齐路线总是在问：**"能不能去掉一个贵的东西？"** —— 去 RM→DPO，去采样→DPO，去 ref→ORPO/SimPO。
- 推理路线总是在问：**"怎么让模型自己发现推理路径而不是背诵？"** —— 可验证奖励→RL→去 critic→修复长 CoT 稳定性。

---

## 五、SOTA 模型实际用到的后训练方法

> 表格来源：各家技术报告 / 论文 / 官方博客整理。"✅ 主用"表示该模型显式使用此方法，"⚪" 表示部分阶段使用或衍生变体。

| 模型（年份） | 出品方 | SFT | RLHF/PPO | DPO 系 | GRPO/RL-VR | 拒绝采样 | 备注 / 特色 |
|---|---|---|---|---|---|---|---|
| InstructGPT (2022) | OpenAI | ✅ | ✅ PPO | | | | RLHF 开山 |
| ChatGPT (2022) | OpenAI | ✅ | ✅ PPO | | | | 同上 |
| GPT-4 (2023) | OpenAI | ✅ | ✅ PPO + rule-RM | | | | 安全 RM |
| Claude 2 (2023) | Anthropic | ✅ | ✅ RLHF + **Constitutional AI** | | | | AI 反馈 |
| Llama-2-Chat (2023) | Meta | ✅ | ✅ PPO（双 RM） | | | ✅ | helpful + harmless RM |
| Mistral-7B-Instruct (2023) | Mistral | ✅ | | ⚪ | | | 早期 SFT-only |
| **Zephyr-7B (2023)** | HuggingFace | ✅ | | ✅ DPO | | | DPO 走红的代表作 |
| Tulu 2 (2023) | AI2 | ✅ | | ✅ DPO | | | 全开源 DPO |
| Mixtral-8x7B-Instruct (2024) | Mistral | ✅ | | ✅ DPO | | | |
| Gemini 1.5 (2024) | Google | ✅ | ✅ | ⚪ | | | 多模态 RLHF |
| Claude 3 (2024) | Anthropic | ✅ | ✅ CAI / RLAIF | | | | 宪法 AI |
| **Llama-3-Instruct (2024)** | Meta | ✅ | ❌（明确弃用 PPO） | ✅ DPO | | ✅ 大量 | SFT + 拒绝采样 + DPO |
| Qwen2-Instruct (2024) | 阿里 | ✅ | ✅ Online DPO + PPO | ✅ | | | 离线+在线混合 |
| GPT-4o (2024) | OpenAI | ✅ | ✅ | | | | 多模态 RLHF |
| **OpenAI o1 (2024.09)** | OpenAI | ✅ | | | ✅ 大规模 RL | | **推理模型范式开创**；细节未公开但确认 RL on CoT |
| DeepSeek-V2/V3 (2024) | 深度求索 | ✅ | ⚪ | ✅ DPO | ⚪ GRPO 前身 | ✅ | GRPO 在 V2 提出 |
| **DeepSeek-R1-Zero (2025.01)** | 深度求索 | ❌ 无 SFT | | | ✅ **纯 GRPO** | | 证明纯 RL 可涌现长链推理 |
| **DeepSeek-R1 (2025.01)** | 深度求索 | ✅ 冷启动 | | | ✅ GRPO 多阶段 | ✅ | SFT + GRPO + 蒸馏到小模型 |
| Qwen2.5-Math / QwQ-32B (2025) | 阿里 | ✅ | | | ✅ GRPO | ✅ | 数学专精 |
| Kimi K1.5 (2025.01) | 月之暗面 | ✅ | | | ✅ RL（自研） | | 长上下文 + 多模态 RL |
| Llama-3.3 (2024.12) | Meta | ✅ | | ✅ DPO | | ✅ | |
| Claude 3.5 / 3.7 (2024-25) | Anthropic | ✅ | ✅ CAI | | ⚪ extended thinking | | 推理模式 |
| Gemini 2.0 / 2.5 Thinking (2025) | Google | ✅ | ✅ | | ✅ | | thinking 模式 |
| GPT-4.5 / o3 / o4-mini (2025) | OpenAI | ✅ | ✅ | | ✅ 大规模 RL | | 推理 + 对齐叠加 |
| Qwen3 (2025.04) | 阿里 | ✅ | | ✅ | ✅ GRPO | ✅ | thinking / non-thinking 双模式同模型，一个 token 切换 |
| Llama 4 (2025.04) | Meta | ✅ | | ✅ DPO | ⚪ | ✅ | 多模态 MoE；推理能力反馈不如预期 |
| **DeepSeek-R1-0528 (2025.05)** | 深度求索 | ✅ | | | ✅ GRPO + Dr. GRPO 思路 | ✅ | AIME 2024 开源 SOTA；质量提升不靠加参数 |
| **Kimi K2 (2025.07)** | 月之暗面 | ✅ | | ✅ | ✅ 联合 RL | ✅ | 1T MoE；首个原生 Agentic 开源模型；大规模 agentic 数据合成 |
| **GPT-5 (2025.08)** | OpenAI | ✅ | ✅ | | ✅ 大规模 RL | | 统一系统：fast 模型 + thinking 模型 + router；**Deliberative Alignment** |
| **DeepSeek V3.1 / V3.2 (2025.09)** | 深度求索 | ✅ | | ✅ | ✅ GRPO | ✅ | 稀疏注意力（DSA）；推理成本优化；后训练双模式 |
| Claude 4 / 4.5 (2025) | Anthropic | ✅ | ✅ CAI | | ✅ extended thinking + tool RL | | Spec-driven；Computer Use 工具 RL |
| Gemini 3 Pro (2025) | Google | ✅ | ✅ | | ✅ | | Deep Think 模式；多模态推理 |
| **Kimi K2 Thinking (2025.11)** | 月之暗面 | ✅ | | | ✅ 联合 RL | ✅ | **原生 4-bit 后训练 (QAT-RL)**；工具调用可任意插入推理链 |

> 注：很多闭源模型（GPT/Claude/Gemini）的具体配方未完整公开，表中信息基于官方系统卡、技术报告与可信二手来源汇总，可能滞后。截止时间 2026.06。

---

## 六、开源后训练框架生态（2025-2026 主流）

| 框架 | 出品方 | 支持算法 | 后端 | 特色 | 适合场景 |
|---|---|---|---|---|---|
| **trl** | HuggingFace | SFT / DPO / IPO / KTO / ORPO / CPO / GRPO / RLOO / Online DPO / PPO / Reward / **Nash MD / XPO** | accelerate + DeepSpeed/FSDP | API 最友好，生态最广；开箱即用 | 中小规模（≤7B）；研究、原型验证 |
| **OpenRLHF** | OpenLLMAI 社区 | PPO / **REINFORCE++** / GRPO / DPO / KTO / Iterative DPO / RLOO / **Async RL** | Ray + vLLM + DeepSpeed/ZeRO-3 | 首个以 Ray 为后端的 RLHF 框架；**70B+ 上可跑；异步 RL 领先** | 70B 以上；需要异步 RL；多节点集群 |
| **verl** | 字节跳动 Seed | PPO / GRPO / **DAPO** / **VAPO** / RLOO / ReMax / **DrGRPO** / Reinforce++ | Ray + FSDP / Megatron-LM + vLLM/SGLang | **HybridFlow 论文原生**；跑通 405B 级别；企业生产验证 | 工业级后训练；超大模型；需要 SOTA 性能 |
| **NeMo-Aligner / NeMo-RL** | NVIDIA | SFT / DPO / RLHF-PPO / GRPO / SteerLM / Self-Rewarding | Megatron-Core + TRT-LLM | NVIDIA 全栈；巨型模型优化好 | 全 NV 集群；企业交付 |
| **Axolotl** | OpenAccess AI | SFT / LoRA / DPO / ORPO / KTO / GRPO（通过 trl） | accelerate + DeepSpeed | YAML 配置驱动，零代码；社区超活 | 多数开源微调项目首选；实验快迭代 |
| **LLaMA-Factory** | 中文社区 | SFT / DPO / KTO / ORPO / SimPO / PPO / GRPO / DAPO | accelerate + DeepSpeed + vLLM | WebUI + 100+ 模型内建模板 | 中文社区入门首选；一边试一边调 |
| **veRL-light / SimpleRL** | 香港科大等 | GRPO / Reinforce++ | minimal PyTorch | 代码不到 1k 行，教学友好 | 入门、复现 R1 小型实验 |
| **TRL-X / unsloth-RL** | Unsloth | GRPO + 4-bit + LoRA | bitsandbytes + Unsloth kernels | **单卡 24G 能跑 GRPO**，速度 2-3× | 单卡玩家；Colab；快速迭代 |

**选型建议**：
- 初学 / 7B 以下：**trl** 或 **LLaMA-Factory**。
- 要跑 GRPO 复现 R1：**verl**（企业） 或 **OpenRLHF**（社区）。
- 超过 70B / 需要异步 RL 与 MoE：**OpenRLHF** 或 **verl**。
- 只有单卡：**unsloth-RL** + LoRA。

> 数据点（2026.06）：OpenRLHF GitHub stars > 8k，verl > 12k，trl > 17k；verl 已被字节 Seed、商汤、Qwen 等多家公司在内外部产品线采用。

---

## 七、典型流水线模板（2025 年开源最佳实践）

```
基座模型 (Pretrained Base)
   │
   │  Stage 1：SFT
   │   - 通用指令数据 (UltraChat / OpenHermes / 自建)
   │   - 拒绝采样 SFT (RFT)：用更强模型/RM 过滤
   ▼
SFT 模型
   │
   │  Stage 2：偏好对齐
   │   - DPO 或 ORPO/SimPO（开源主流，省事）
   │   - or RLHF-PPO（OpenAI/Anthropic 仍在用）
   │   - 数据：UltraFeedback / 自建偏好对 / RLAIF
   ▼
对齐模型 (Chat 模型)
   │
   │  Stage 3：可验证 RL（仅当目标是推理/Agent）
   │   - GRPO / DAPO，奖励 = 数学正确性 / 单测通过 / 工具调用成功
   │   - 多阶段：cold-start SFT → RL → SFT-on-RL-output → RL again
   ▼
推理模型 (Reasoning 模型)
```

[posttrain_demo.py](./posttrain_demo.py) 把 Stage 1/2/3 的**最小内核**分别用 [demo_sft](./posttrain_demo.py)、[demo_dpo](./posttrain_demo.py)、[demo_rlvr](./posttrain_demo.py) 各一个函数演示了出来。

---

## 八、选型建议

| 场景 | 推荐方法 | 理由 |
|---|---|---|
| 只想让模型听指令 / 学会某种格式 | **SFT** | 最简单，几百条数据就见效 |
| 有人工偏好对，想省事 | **DPO / ORPO / SimPO** | 免 RM、免采样、免 critic |
| 卡多、想要极致效果 | **RLHF-PPO** + 双 RM | OpenAI / Anthropic 路线 |
| 数学 / 代码 / 工具调用，奖励可程序判定 | **GRPO / DAPO** | DeepSeek-R1 路线，最 hot |
| 显存极度受限（单卡 24G） | **ORPO / SimPO** | 不需要 ref 模型 |
| 想从更强模型蒸馏 | **拒绝采样 SFT + DPO** | Llama-3 路线 |
| 安全/价值观对齐 | **CAI / Spec-driven** | Anthropic 路线 |

---

## 九、参考文献（按时间排序，重点选）

**经典对齐（2022–2024）**
- **InstructGPT** (2022) — Ouyang et al. *Training language models to follow instructions with human feedback* (arXiv:2203.02155)
- **Constitutional AI** (2022) — Bai et al. (Anthropic) (arXiv:2212.08073)
- **DPO** (2023) — Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (arXiv:2305.18290)
- **Llama-2** (2023) — Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models* (arXiv:2307.09288)
- **IPO** (2023) — Azar et al. *A General Theoretical Paradigm to Understand Learning from Human Preferences* (arXiv:2310.12036)
- **KTO** (2024) — Ethayarajh et al. *Model Alignment as Prospect Theoretic Optimization* (arXiv:2402.01306)
- **ORPO** (2024) — Hong et al. *ORPO: Monolithic Preference Optimization without Reference Model* (arXiv:2403.07691)
- **SimPO** (2024) — Meng et al. *SimPO: Simple Preference Optimization with a Reference-Free Reward* (arXiv:2405.14734)
- **GRPO / DeepSeekMath** (2024) — Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (arXiv:2402.03300)
- **Llama-3 Tech Report** (2024) — Meta AI (arXiv:2407.21783)

**推理与 RLVR 突破（2025）**
- **DeepSeek-R1** (2025) — DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* (arXiv:2501.12948)
- **Kimi K1.5** (2025) — Moonshot AI. *Kimi K1.5: Scaling RL with LLMs* (arXiv:2501.12599)
- **DAPO** (2025) — ByteDance Seed. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (arXiv:2503.14476)
- **Dr. GRPO** (2025) — Liu et al. (NUS/Sea AI Lab). *Understanding R1-Zero-Like Training: A Critical Perspective* (arXiv:2503.20783)
- **VAPO** (2025) — ByteDance Seed. *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks* (arXiv:2504.05118)
- **LCPO / L1** (2025) — Aggarwal et al. (CMU). *L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning* (arXiv:2503.04697)
- **REINFORCE++** (2025) — Hu et al. (OpenRLHF). *REINFORCE++: A Simple and Efficient Approach for Aligning LLMs* (arXiv:2501.03262)
- **Logic-RL** (2025) — Xie et al. *Logic-RL: Unleashing LLM Reasoning with Rule-Based RL* (arXiv:2502.14768)
- **Open R1 / SimpleRL** (2025) — HuggingFace / HKUST. R1 复现报告。
- **Qwen3 Tech Report** (2025) — Alibaba (arXiv:2505.09388)
- **DeepSeek-R1-0528** (2025.05) — DeepSeek-AI 官方报告。
- **Kimi K2** (2025) — Moonshot AI. *Kimi K2: Open Agentic Intelligence* (arXiv:2507.20534)

**2025 H2 – 2026 最新进展**
- **GPT-5 System Card** (2025.08) — OpenAI. 包含 Deliberative Alignment 、路由机制说明。
- **Deliberative Alignment** (2024) — Guan et al. (OpenAI) (arXiv:2412.16339)
- **DeepSeek V3.1 / V3.2 Tech Reports** (2025.09) — DeepSeek-AI。DSA 稀疏注意力。
- **Kimi K2 Thinking** (2025.11) — Moonshot AI 官方报告：原生 4-bit 后训练与 Agentic 推理。
- **Claude 4 / 4.5 System Cards** (2025) — Anthropic。Computer Use 与 Spec-driven Alignment。
- **Gemini 3 Pro / Deep Think** (2025) — Google。
- **The State of RL for LLM Reasoning** (2025) — Sebastian Raschka 综述博文。
- **Exploration Collapse / Diversity Loss** 系列 (2025–2026) — MIT/Yale/NUS/NTU 多篇论文，讨论 RL post-training scaling law 与多样性保护。

**开源框架（技术报告 / 仓库）**
- **trl** (HuggingFace) — https://github.com/huggingface/trl
- **OpenRLHF** (2024-2025) — Hu et al. *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework* (arXiv:2405.11143)
- **verl / HybridFlow** (2024) — Sheng et al. (字节 Seed) *HybridFlow: A Flexible and Efficient RLHF Framework* (arXiv:2409.19256) — https://github.com/volcengine/verl
- **NeMo-Aligner** (2024) — NVIDIA (arXiv:2405.01481)
- **Axolotl** — https://github.com/axolotl-ai-cloud/axolotl
- **LLaMA-Factory** — Zheng et al. (arXiv:2403.13372)

---

## 十、本仓库怎么跑

本仓库由 [uv](https://docs.astral.sh/uv/) 管理，依赖与 torch 通道在 [pyproject.toml](./pyproject.toml) 中声明：Linux / Windows 默认 CUDA 13.2 官方 wheel，macOS 走 PyPI 默认 wheel（自带 mps / cpu）。

### 1. 装环境

```powershell
uv sync
```

> Linux / Windows 会从 PyTorch CUDA 13.2 通道拉 GPU 版 torch；macOS 不受影响（不可能有 CUDA）。
>
> Linux / Windows 没有 NVIDIA GPU 想纯 CPU 跑：把 [pyproject.toml](./pyproject.toml) 中 `pytorch-cu132` 的 url 改成 `https://download.pytorch.org/whl/cpu` 后重跑 `uv sync`。
>
> 切换 CUDA 版本后需刷新缓存：`uv sync --reinstall-package torch`。

### 2. 跑训练

```powershell
# 单方法
uv run posttrain_demo.py --method sft
uv run posttrain_demo.py --method dpo --quick

# 分组
uv run posttrain_demo.py --method all-pref --quick   # sft/dpo/ipo/kto/orpo/simpo
uv run posttrain_demo.py --method all-rl   --quick   # rlvr/grpo/dr_grpo/rloo/reinforce_pp/dapo/lcpo
uv run posttrain_demo.py --method all-self --quick   # rft/star
uv run posttrain_demo.py --method all-stub           # rlaif/cai/ppo（stub，秒级）

# 全跑（18 个方法烟雾测试）
uv run posttrain_demo.py --method all --quick
```

常用参数：`--steps N`、`--lr 1e-5`、`--dtype {auto,fp32,bf16}`、`--model <hf_id>`。

### 3. VSCode 调试

先在终端做一次 `uv sync`（[dependency-groups].dev 里已带 debugpy），然后在脚本里打断点 → F5 → 选 `Debug: uv run posttrain_demo.py`。配置文件见 [.vscode/launch.json](./.vscode/launch.json) + [.vscode/tasks.json](./.vscode/tasks.json)。
