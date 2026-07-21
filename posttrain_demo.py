"""
================================================================================
 大模型后训练全家桶 Demo
   SFT / DPO / IPO / KTO / ORPO / SimPO /
   RFT / STaR / RLVR / GRPO / Dr.GRPO /
   RLOO / REINFORCE++ / DAPO / LCPO /
   GSPO / CISPO / OPD / SPIN / 熵控RL /
   RLAIF* / CAI* / PPO*       （* 为 stub，仅演示思路）

 在 Apple Silicon (MPS) / Linux+CUDA / Windows+CUDA / 纯 CPU 上本地可跑，
 自动选最佳设备。
================================================================================

设计目标
--------
不依赖 trl / peft / bitsandbytes / unsloth 等高层库，只用 torch + transformers
的稳定底层 API，手写每种后训练方法的核心训练循环，让你看清它们的
【数学本质差异】。每个方法 ≤90 行教学卡片风格，独立可调。

依赖管理
--------
本仓库由 uv 管理（pyproject.toml）。一行装好：
  uv sync

Linux / Windows 走 PyTorch CUDA 13.2 官方通道（GPU 版 torch）；
macOS 不可能有 CUDA，直接用 PyPI 默认 torch（自带 mps / cpu 后端）。

Linux / Windows 没有 NVIDIA GPU 想纯 CPU 跑：把 pyproject.toml 中 pytorch-cu132 的
url 改成 https://download.pytorch.org/whl/cpu 后重跑 uv sync 即可。

用法
----
  # 同步好环境后
  uv run posttrain_demo.py --method sft

  # 单方法 / 分组 / 全跑
  uv run posttrain_demo.py --method sft         # 单方法
  uv run posttrain_demo.py --method all-pref    # 偏好家族 (sft/dpo/ipo/kto/orpo/simpo)
  uv run posttrain_demo.py --method all-rl      # 可验证 RL 家族（含 gspo/cispo/entropy）
  uv run posttrain_demo.py --method all-self    # 自提升/蒸馏 (rft/star/opd/spin)
  uv run posttrain_demo.py --method all-stub    # AI 反馈 / PPO stub
  uv run posttrain_demo.py --method all --quick # 全跑（快速烟雾测试）

可选参数
  --model     默认 Qwen/Qwen2.5-0.5B-Instruct（首次运行约下载 1GB）
  --steps     每种方法的训练步数（默认 40；--quick 强制 10）
  --lr        学习率（默认 1e-5）
  --dtype     auto | fp32 | bf16；auto 在 CUDA 上用 bf16，其他用 fp32
================================================================================
"""

import argparse
import copy
import math
import os
import re
import traceback

# 国内访问 huggingface.co 经常超时，默认走 hf-mirror 镜像。
# 用户已自行设置 HF_ENDPOINT 时则尊重其设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# 基础设施：设备/精度、模型加载、生成、对话模板、共享损失内核、错误隔离
# ============================================================================
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"          # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
DTYPE = torch.float32   # 默认；main() 中由 --dtype 覆盖


def pick_dtype(spec="auto"):
    """根据设备和用户指定挑选合适的精度。
    - CUDA: 默认 bf16（A/H/30/40 系列稳）
    - MPS / CPU: 强制 fp32（MPS bf16 演进中，CPU bf16 慢）
    """
    spec = (spec or "auto").lower()
    if spec == "fp32":
        return torch.float32
    if spec == "bf16":
        return torch.bfloat16 if DEVICE == "cuda" else torch.float32
    return torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load(model_name):
    print(f"\n[加载模型] {model_name}  ->  device={DEVICE}  dtype={DTYPE}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE
    ).to(DEVICE)
    return model, tok


def build_prompt(tok, user_msg, system=None):
    """用模型自带 chat 模板拼出 prompt（到 assistant 开头为止）。"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_msg})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, user_msg, system=None, max_new=64):
    model.eval()
    text = build_prompt(tok, user_msg, system)
    enc = tok(text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][enc.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def seq_logp(model, tok, prompt_text, answer_text):
    """计算 logP(answer | prompt)，求和限制在 answer token 上。
    返回标量张量（带梯度）。"""
    full = prompt_text + answer_text
    full_ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = tok(prompt_text, return_tensors="pt").input_ids.shape[1]

    out = model(input_ids=full_ids)
    logits = out.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    token_logp = logp_all.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(token_logp)
    mask[:, prompt_len - 1:] = 1.0
    return (token_logp * mask).sum()


def seq_logp_with_lengths(model, tok, prompt_text, answer_text):
    """seq_logp 的扩展：返回 (sum_logp, mean_logp, ans_len)。
    SimPO / Dr.GRPO / ORPO 等需要长度归一化形式。"""
    full = prompt_text + answer_text
    full_ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = tok(prompt_text, return_tensors="pt").input_ids.shape[1]

    out = model(input_ids=full_ids)
    logits = out.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    token_logp = logp_all.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(token_logp)
    mask[:, prompt_len - 1:] = 1.0
    ans_len = mask.sum().clamp(min=1.0)
    sum_lp = (token_logp * mask).sum()
    mean_lp = sum_lp / ans_len
    return sum_lp, mean_lp, int(ans_len.item())


def seq_logp_tokens(model, tok, prompt_text, answer_text, with_entropy=False):
    """token 级版本：返回 answer 段每个 token 的 log-prob 一维张量（带梯度）。
    with_entropy=True 时额外返回每 token 的策略熵（CISPO / OPD / 熵控 RL 用）。"""
    full = prompt_text + answer_text
    full_ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = tok(prompt_text, return_tensors="pt").input_ids.shape[1]

    out = model(input_ids=full_ids)
    logits = out.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    token_logp = logp_all.gather(2, labels.unsqueeze(-1)).squeeze(-1)[0]

    start = max(prompt_len - 1, 0)
    tok_lp = token_logp[start:]
    if with_entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)[0][start:]
        return tok_lp, ent
    return tok_lp


@torch.no_grad()
def sample_group(model, tok, prompt_text, group=4, max_new=12,
                 temperature=1.0, top_p=0.95):
    """一次性 batch 采样 `group` 个答案（num_return_sequences）。
    返回 List[str]，已剥离 prompt 段、跳过特殊 token。"""
    model.eval()
    enc = tok(prompt_text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=group,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = enc.input_ids.shape[1]
    answers = []
    for i in range(out.shape[0]):
        ans_ids = out[i][prompt_len:]
        answers.append(tok.decode(ans_ids, skip_special_tokens=True))
    return answers


def pref_loss(variant, *, lp_c=None, lp_r=None, lpref_c=None, lpref_r=None,
              mean_lp_c=None, mean_lp_r=None, beta=0.1, gamma=1.4,
              lambda_=0.5, kto_label=None, kto_z0=0.0):
    """偏好家族共享损失内核（分发器）。
    - 'dpo'   : -log σ(β·((lp_c-lpref_c) - (lp_r-lpref_r)))
    - 'ipo'   : ((Δc - Δr) - 1/(2β))²        其中 Δ = lp - lpref
    - 'simpo' : -log σ(β·(mean_lp_c - mean_lp_r) - γ)             无 ref
    - 'orpo'  : λ · -log σ(log_odds(c) - log_odds(r))             无 ref，外部还要加 SFT
                log_odds = mean_lp - log(1 - exp(mean_lp))
    - 'kto'   : 单点 Kahneman-Tversky；用 lp_c 槽位传"当前样本 lp"，
                lpref_c 槽位传"ref lp"，kto_label∈{0,1}, kto_z0 用 batch 均值近似。
    """
    if variant == "dpo":
        logits = beta * ((lp_c - lpref_c) - (lp_r - lpref_r))
        return -F.logsigmoid(logits)
    if variant == "ipo":
        diff = (lp_c - lpref_c) - (lp_r - lpref_r)
        target = 1.0 / (2.0 * beta)
        return (diff - target) ** 2
    if variant == "simpo":
        logits = beta * (mean_lp_c - mean_lp_r) - gamma
        return -F.logsigmoid(logits)
    if variant == "orpo":
        eps = 1e-6
        log_one_minus_p_c = torch.log1p(-torch.exp(mean_lp_c).clamp(max=1 - eps))
        log_one_minus_p_r = torch.log1p(-torch.exp(mean_lp_r).clamp(max=1 - eps))
        log_odds_c = mean_lp_c - log_one_minus_p_c
        log_odds_r = mean_lp_r - log_one_minus_p_r
        return lambda_ * -F.logsigmoid(log_odds_c - log_odds_r)
    if variant == "kto":
        delta = lp_c - lpref_c
        if kto_label == 1:
            return 1.0 - torch.sigmoid(beta * (delta - kto_z0))
        return 1.0 - torch.sigmoid(-beta * (delta - kto_z0))
    raise ValueError(f"未知 variant: {variant}")


def try_run(name, fn, *args, **kwargs):
    """通用 try/except：单方法异常不阻塞 --method all。"""
    try:
        fn(*args, **kwargs)
        return True
    except torch.cuda.OutOfMemoryError as e:  # noqa
        print(f"\n[!! 跳过 {name}] CUDA OOM: {e}")
    except RuntimeError as e:
        print(f"\n[!! 跳过 {name}] RuntimeError: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"\n[!! 跳过 {name}] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return False


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ============================================================================
# 方法 1：SFT —— 监督微调 / 指令模仿
# ----------------------------------------------------------------------------
# 任务：教模型用一种【固定的中二风格签名】结尾回答。
# loss = -logP(理想答案 | prompt)  纯交叉熵
# 训练数据格式：(prompt, ideal_answer) 监督对。
# demo：4 条问答，理想答案统一以"——由小刘为您解答 :3"结尾。
# ============================================================================
def demo_sft(model_name, steps, lr):
    banner("方法 1：SFT（监督微调）—— 学会模仿一种固定回答格式")
    model, tok = load(model_name)

    SYS = "你是一个助手。"
    data = [
        ("1+1等于几？", "1+1=2。——由小刘为您解答 :3"),
        ("天空为什么是蓝色的？", "因为大气对蓝光散射更强。——由小刘为您解答 :3"),
        ("推荐一种水果。", "推荐苹果，富含维生素。——由小刘为您解答 :3"),
        ("水的沸点是多少？", "标准大气压下是100摄氏度。——由小刘为您解答 :3"),
    ]
    test_q = "中国的首都是哪里？"

    print("\n--- 训练【前】 ---")
    before = generate(model, tok, test_q, system=SYS)
    print(f"Q: {test_q}\nA: {before}")
    print(f"含目标签名? {'小刘' in before}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（交叉熵模仿，{steps} 步）---")
    for step in range(steps):
        q, a = data[step % len(data)]
        prompt = build_prompt(tok, q, system=SYS)
        loss = -seq_logp(model, tok, prompt, a) / 20.0
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss(=-logP理想答案) = {loss.item():.3f}")

    print("\n--- 训练【后】 ---")
    after = generate(model, tok, test_q, system=SYS)
    print(f"Q: {test_q}\nA: {after}")
    print(f"含目标签名? {'小刘' in after}")
    print("\n[结论] SFT 让模型模仿出训练数据里的固定签名 —— 这就是'指令/格式对齐'。")
    del model


# ============================================================================
# 方法 2：DPO —— Direct Preference Optimization（免 reward model、免 RL）
# ----------------------------------------------------------------------------
# loss = -logσ( β·[(logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r)] )
# 训练数据格式：(prompt, chosen, rejected) 偏好三元组 + 冻结 ref 模型。
# demo：3 组生活问答，chosen=温暖建设性回答，rejected=冷漠敷衍回答。
# ============================================================================
def demo_dpo(model_name, steps, lr, beta=0.1):
    banner("方法 2：DPO（直接偏好优化）—— 用偏好对学'更好'，无需 RM/RL")
    model, tok = load(model_name)
    ref = copy.deepcopy(model).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    SYS = "你是一个助手。"
    pairs = [
        ("帮我推荐一本书。",
         "推荐《人类简史》，视角宏大、可读性强，祝你阅读愉快！",
         "书很多，你自己去图书馆找吧，我也不知道你喜欢啥。"),
        ("怎么泡一杯好咖啡？",
         "用新鲜咖啡豆现磨，92度水冲泡30秒，简单又好喝！",
         "随便冲冲就行了，泡咖啡有啥讲究，别浪费时间。"),
        ("我有点累，怎么办？",
         "先休息一下，喝点水深呼吸，照顾好自己最重要哦。",
         "累就累着呗，谁不累，忍忍就过去了。"),
    ]
    test_q = "我想学编程，有什么建议？"

    def lp(m, q, ans):
        prompt = build_prompt(tok, q, system=SYS)
        return seq_logp(m, tok, prompt, ans)

    print("\n--- 训练【前】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（DPO，β={beta}，{steps} 步）---")
    for step in range(steps):
        q, ch, rj = pairs[step % len(pairs)]
        lp_c = lp(model, q, ch)
        lp_r = lp(model, q, rj)
        with torch.no_grad():
            lpref_c = lp(ref, q, ch)
            lpref_r = lp(ref, q, rj)
        loss = pref_loss("dpo", lp_c=lp_c, lp_r=lp_r,
                         lpref_c=lpref_c, lpref_r=lpref_r, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()

        acc = ((lp_c - lpref_c) > (lp_r - lpref_r)).float().item()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            margin = ((lp_c - lpref_c) - (lp_r - lpref_r)).item()
            print(f"  step {step:3d}  loss = {loss.item():.3f}  "
                  f"隐式奖励差 = {margin:+.3f}  正确? {int(acc)}")

    print("\n--- 训练【后】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")
    print("\n[结论] DPO 把模型推向'chosen 风格'、远离'rejected 风格'，免 RM、免采样。")
    del model, ref


# ============================================================================
# 方法 3：IPO —— Identity Preference Optimization（DPO 的平方损失变体）
# ----------------------------------------------------------------------------
# Azar et al. 2023 指出 DPO 在 σ 饱和后梯度消失易过拟合，IPO 改为平方损失，
# 把 (Δc - Δr) 拉向 1/(2β)，对极端偏好更鲁棒。
# loss = ((Δc - Δr) - 1/(2β))²
# 训练数据格式：(prompt, chosen, rejected) 偏好三元组（同 DPO）。
# demo：焦虑/简历/跑步 3 组问答，chosen=具体可行建议，rejected=简单否定/敷衍。
# ============================================================================
def demo_ipo(model_name, steps, lr, beta=0.1):
    banner("方法 3：IPO（DPO 平方损失变体）—— 防过拟合，目标 Δ=1/(2β)")
    model, tok = load(model_name)
    ref = copy.deepcopy(model).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    SYS = "你是一个助手。"
    pairs = [
        ("我有点焦虑。",
         "先深呼吸，把你担心的事写下来，一件件来，会好转的。",
         "焦虑就焦虑呗，别想太多。"),
        ("怎么写一份好简历？",
         "聚焦最近2-3段经历，量化成果，1页为佳。",
         "网上随便抄一份。"),
        ("跑步对身体好吗？",
         "适量跑步有益心肺；循序渐进，注意拉伸。",
         "跑步伤膝盖，不要跑。"),
    ]
    test_q = "我该不该转行？"

    def lp(m, q, ans):
        prompt = build_prompt(tok, q, system=SYS)
        return seq_logp(m, tok, prompt, ans)

    print("\n--- 训练【前】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（IPO 平方损失，β={beta}，{steps} 步）---")
    for step in range(steps):
        q, ch, rj = pairs[step % len(pairs)]
        lp_c = lp(model, q, ch); lp_r = lp(model, q, rj)
        with torch.no_grad():
            lpref_c = lp(ref, q, ch); lpref_r = lp(ref, q, rj)
        loss = pref_loss("ipo", lp_c=lp_c, lp_r=lp_r,
                         lpref_c=lpref_c, lpref_r=lpref_r, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            diff = ((lp_c - lpref_c) - (lp_r - lpref_r)).item()
            print(f"  step {step:3d}  loss = {loss.item():.3f}  "
                  f"Δc-Δr = {diff:+.3f}  目标 1/(2β) = {1/(2*beta):.2f}")

    print("\n--- 训练【后】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")
    print("\n[结论] IPO 把 'chosen-rejected 优势' 拉向固定目标 1/(2β)，比 DPO 更不易过拟合。")
    del model, ref


# ============================================================================
# 方法 4：KTO —— Kahneman-Tversky Optimization（单点偏好）
# ----------------------------------------------------------------------------
# Ethayarajh et al. 2024：不需要成对 (chosen, rejected)，每条数据是单点
# (prompt, completion, label∈{好,坏})。损失基于前景理论的不对称效用：
#   label=1 (好):  loss = 1 - σ( β·(Δ - z₀))
#   label=0 (坏):  loss = 1 - σ(-β·(Δ - z₀))
#   其中 Δ = logπ - logπ_ref，z₀ = batch 内 Δ 的均值（KL 近似）
# 适合数据是"哪些回答好/坏"的二分标注，而不是成对偏好。
# 训练数据格式：(prompt, completion, label∈{1=好,0=坏}) 单点标注。
# demo：光合作用/保持健康/鼓励的话 3 个 prompt，每个各给 1 条好答案 + 1 条坏答案。
# ============================================================================
def demo_kto(model_name, steps, lr, beta=0.1):
    banner("方法 4：KTO（单点偏好优化）—— 不需要成对数据，只要二元标签")
    model, tok = load(model_name)
    ref = copy.deepcopy(model).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    SYS = "你是一个助手。"
    # (prompt, completion, label∈{1=好,0=坏})
    data = [
        ("解释一下光合作用。", "植物用阳光把二氧化碳和水合成葡萄糖，并释放氧气。", 1),
        ("解释一下光合作用。", "不知道，自己查吧。", 0),
        ("如何保持健康？", "规律作息、均衡饮食、适量运动。", 1),
        ("如何保持健康？", "随便吧，活着就行。", 0),
        ("写一句鼓励的话。", "每一步努力都会照亮你前方的路。", 1),
        ("写一句鼓励的话。", "鼓励有什么用，自己干吧。", 0),
    ]
    test_q = "你能给我点学习建议吗？"

    def lp(m, q, ans):
        prompt = build_prompt(tok, q, system=SYS)
        return seq_logp(m, tok, prompt, ans)

    print("\n--- 训练【前】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（KTO，β={beta}，{steps} 步）---")
    # 预先算 z₀ 近似：用前若干样本 Δ 的均值
    with torch.no_grad():
        deltas = []
        for q, ans, _ in data:
            d = lp(model, q, ans) - lp(ref, q, ans)
            deltas.append(d.item())
        z0 = sum(deltas) / len(deltas)
    print(f"  z₀（Δ 均值近似 KL 锚点）= {z0:+.3f}")

    for step in range(steps):
        q, ans, label = data[step % len(data)]
        lp_pol = lp(model, q, ans)
        with torch.no_grad():
            lp_ref = lp(ref, q, ans)
        loss = pref_loss("kto", lp_c=lp_pol, lpref_c=lp_ref,
                         beta=beta, kto_label=label, kto_z0=z0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            d = (lp_pol - lp_ref).item()
            tag = "好" if label == 1 else "坏"
            print(f"  step {step:3d}  label={tag}  Δ={d:+.3f}  loss={loss.item():.3f}")

    print("\n--- 训练【后】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")
    print("\n[结论] KTO 用'好/坏'单点标签即可对齐，无需偏好对——大幅降低数据收集成本。")
    del model, ref


# ============================================================================
# 方法 5：ORPO —— Odds Ratio Preference Optimization（无 ref 模型）
# ----------------------------------------------------------------------------
# Hong et al. 2024：DPO 必须维护 ref 模型，ORPO 把"偏好对比"用 odds ratio
# 直接加到 SFT 上，单模型同时做监督模仿和偏好分离。
#   loss = SFT(chosen) + λ · -log σ( log_odds(c) - log_odds(r) )
#   log_odds = log p - log(1-p) ≈ mean_logp - log(1 - exp(mean_logp))
# 优点：训练显存减半（无 ref），代码也更短。
# 训练数据格式：(prompt, chosen, rejected) 偏好对（不需 ref 模型）。
# demo：冷笑话/背单词/失眠 3 组，chosen=具体方法，rejected=甩锅敷衍。
# ============================================================================
def demo_orpo(model_name, steps, lr, lambda_=0.5):
    banner("方法 5：ORPO（Odds Ratio）—— 无 ref 模型，SFT + 偏好对比一体化")
    model, tok = load(model_name)

    SYS = "你是一个助手。"
    pairs = [
        ("讲个冷笑话。", "为什么数学书总是不开心？因为它有太多问题。",
         "笑话嘛，自己上网搜。"),
        ("怎么记住单词？", "结合语境造句，加间隔复习，比死记硬背好。",
         "背呗，反复背。"),
        ("夜里睡不着怎么办？", "试试规律作息、避免咖啡因、睡前别看手机。",
         "睡不着就熬着吧。"),
    ]
    test_q = "怎样才能写出好文章？"

    print("\n--- 训练【前】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（ORPO，λ={lambda_}，{steps} 步）---")
    for step in range(steps):
        q, ch, rj = pairs[step % len(pairs)]
        prompt = build_prompt(tok, q, system=SYS)
        sum_c, mean_c, len_c = seq_logp_with_lengths(model, tok, prompt, ch)
        sum_r, mean_r, len_r = seq_logp_with_lengths(model, tok, prompt, rj)
        sft_loss = -sum_c / max(1, len_c)        # 模仿 chosen
        odds_loss = pref_loss("orpo",
                              mean_lp_c=mean_c, mean_lp_r=mean_r,
                              lambda_=lambda_)
        loss = sft_loss + odds_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.3f}  "
                  f"sft={sft_loss.item():.3f}  odds={odds_loss.item():.3f}  "
                  f"meanLP(c-r)={(mean_c-mean_r).item():+.3f}")

    print("\n--- 训练【后】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")
    print("\n[结论] ORPO 把 SFT 与偏好分离合二为一，省掉 ref 模型——单卡单模型即可对齐。")
    del model


# ============================================================================
# 方法 6：SimPO —— Simple Preference Optimization（无 ref，长度归一化）
# ----------------------------------------------------------------------------
# Meng et al. 2024：观察到 DPO 把 logπ 求和会被长度系统性偏置（偏长 token 总
# 优势更大），SimPO 用 mean_logp（长度归一化）+ 无 ref，再加 margin γ：
#   loss = -log σ( β·(mean_lp_c - mean_lp_r) - γ )
# β 一般大些（如 2.5），γ 是固定 margin（如 1.4）。
# 训练数据格式：(prompt, chosen, rejected) 偏好对（无 ref，logp 用长度均值）。
# demo：跑步/阅读/缓压 3 组生活类问答，chosen=具体步骤，rejected=随便糊弄。
# ============================================================================
def demo_simpo(model_name, steps, lr, beta=2.5, gamma=1.4):
    banner("方法 6：SimPO（无 ref + 长度归一化）—— 训练最轻量的偏好对齐")
    model, tok = load(model_name)

    SYS = "你是一个助手。"
    pairs = [
        ("怎么开始跑步？",
         "从每周 3 次、每次 20 分钟慢跑开始，循序渐进。",
         "想跑就跑呗，没啥可说的。"),
        ("怎样高效阅读？",
         "带问题阅读，先扫目录再精读重点章节并做笔记。",
         "看书就看呗，看快就行。"),
        ("如何缓解压力？",
         "规律运动、保证睡眠、和朋友聊聊都管用。",
         "压力嘛，扛过去就完了。"),
    ]
    test_q = "怎样保持专注？"

    print("\n--- 训练【前】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 训练中（SimPO，β={beta}，γ={gamma}，{steps} 步）---")
    for step in range(steps):
        q, ch, rj = pairs[step % len(pairs)]
        prompt = build_prompt(tok, q, system=SYS)
        _, mean_c, _ = seq_logp_with_lengths(model, tok, prompt, ch)
        _, mean_r, _ = seq_logp_with_lengths(model, tok, prompt, rj)
        loss = pref_loss("simpo", mean_lp_c=mean_c, mean_lp_r=mean_r,
                         beta=beta, gamma=gamma)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.3f}  "
                  f"mean_lp(c-r)={(mean_c-mean_r).item():+.3f}  "
                  f"目标 > γ/β = {gamma/beta:.2f}")

    print("\n--- 训练【后】 ---")
    print(f"Q: {test_q}\nA: {generate(model, tok, test_q, system=SYS)}")
    print("\n[结论] SimPO 既无 ref 又长度归一化，训练显存最低、对长度偏置更鲁棒。")
    del model


# ============================================================================
# 方法 7：RFT —— Rejection Fine-Tuning（拒绝采样自蒸馏 / 自提升）
# ----------------------------------------------------------------------------
# Yuan et al. 2023：在可验证任务上自采样多份答案，只把【正确】答案做 SFT。
# 是 LLaMA / DeepSeekMath 等模型构建数学推理能力的基石技巧。
# 训练数据格式：(prompt, 可验证 gold) —— label 由模型自采样 + 验证函数过滤产生。
# demo：6 道一位数加法（如 3+4），每步自采 8 个候选，留下输出正确数字者做 SFT。
# ============================================================================
def demo_rft(model_name, steps, lr, group=8):
    banner("方法 7：RFT（拒绝采样微调）—— 自采样过滤后做 SFT，最朴素的自提升")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（每步采样 {group} 个、过滤正确 → SFT，{steps} 步）---")
    total_kept = 0
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        answers = sample_group(model, tok, prompt, group=group, max_new=8)
        kept = [t for t in answers if reward_of(t, gold) == 1.0]
        if not kept:
            if step % max(1, steps // 6) == 0:
                print(f"  step {step:3d}  {a}+{b}  无正确样本可蒸（跳过）")
            continue
        # 把正确答案当作 SFT label
        model.train()
        opt.zero_grad()
        for t in kept:
            loss = -seq_logp(model, tok, prompt, t) / 5.0
            loss.backward()
        opt.step()
        total_kept += len(kept)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            print(f"  step {step:3d}  {a}+{b}  采样 {group} 留 {len(kept)} 条  "
                  f"累计蒸馏 {total_kept} 条")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] RFT = 自采样 + 答案验证过滤 + SFT；零人工标注即可强化推理能力。")
    del model



# ============================================================================
# 方法 8：STaR / ReST —— Self-Taught Reasoner（多轮自举）
# ----------------------------------------------------------------------------
# Zelikman 2022 / Gulcehre 2023：把 RFT 套上"外循环 round"——
#   每轮：自采样 → 过滤正确 → SFT；下一轮用更强的 self 再采样。
# 这是 DeepSeekMath / Qwen-Math / R1 数据迭代式自提升的雏形。
# 训练数据格式：(prompt, 可验证 gold) —— 每轮重新自采样并过滤，labels 不断刷新。
# demo：6 道一位数加法外套 rounds=2 轮 RFT，观察准确率随 round 上升。
# ============================================================================
def demo_star(model_name, rounds, steps_per_round, lr, group=8):
    banner(f"方法 8：STaR / ReST —— {rounds} 轮 RFT 自举提升")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    acc0 = eval_acc()
    print(f"\n--- 起始准确率 = {acc0:.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = [acc0]

    for r in range(rounds):
        print(f"\n========== Round {r + 1}/{rounds} ==========")
        kept_round = 0
        for step in range(steps_per_round):
            a, b = questions[step % len(questions)]
            gold = a + b
            prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
            answers = sample_group(model, tok, prompt, group=group, max_new=8)
            kept = [t for t in answers if reward_of(t, gold) == 1.0]
            if not kept:
                continue
            model.train(); opt.zero_grad()
            for t in kept:
                loss = -seq_logp(model, tok, prompt, t) / 5.0
                loss.backward()
            opt.step()
            kept_round += len(kept)
        acc = eval_acc()
        history.append(acc)
        print(f"  Round {r+1} 末 准确率 = {acc:.0%}  本轮蒸馏样本数 = {kept_round}")

    print(f"\n[准确率轨迹] {' -> '.join(f'{x:.0%}' for x in history)}")
    print("[结论] STaR/ReST 把自采样过滤循环若干轮，演示推理能力的'自举提升'。")
    del model


# ============================================================================
# 方法 9：RLVR —— 可验证奖励强化学习（o1/R1 路线的极简版）
# ----------------------------------------------------------------------------
# 算法：REINFORCE with baseline（PPO/GRPO 的最小内核）
#   loss = -(r - baseline) · logP(自采样答案)
# 升级点：用 sample_group 一次 batch 采样替代串行 loop。
# 训练数据格式：(prompt, 可验证 reward 函数 r(text, gold)) —— 无需人工标答案。
# demo：6 道一位数加法，每步采 4 个候选，正确得 1 错得 0，组均值作 baseline。
# ============================================================================
def demo_rlvr(model_name, steps, lr, group=4):
    banner("方法 9：RLVR（可验证奖励 REINFORCE）—— 答案对错即奖励")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（REINFORCE，每步采 {group} 个，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        baseline = sum(rewards) / len(rewards)
        if all(r == baseline for r in rewards):
            if step % max(1, steps // 8) == 0:
                print(f"  step {step:3d}  {a}+{b} rewards={rewards} (无信号)")
            continue
        model.train(); opt.zero_grad(); total = 0.0
        for txt, r in zip(samples, rewards):
            adv = r - baseline
            if adv == 0:
                continue
            loss = -adv * seq_logp(model, tok, prompt, txt) / 5.0
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  {a}+{b} rewards={rewards}  "
                  f"baseline={baseline:.2f}  loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] REINFORCE+组内 baseline = PPO/GRPO 的内核，可用纯规则奖励无限 scale。")
    del model


# ============================================================================
# 方法 10：GRPO —— Group Relative Policy Optimization（DeepSeek-R1 同款）
# ----------------------------------------------------------------------------
# DeepSeekMath/R1 用 GRPO 替代 PPO 省掉 critic：
#   advantage_i = (r_i - mean) / (std + ε)        组内 z-score
#   loss = -E[ adv · logπ ]  +  β · KL(π ‖ π_ref)
# ref 来自冷启动 SFT 后的快照（这里用 deepcopy）。
# 训练数据格式：(prompt, 可验证 reward) + 冻结 ref 模型用作 KL 锚。
# demo：6 道一位数加法，每 prompt 采 group=4 个回答，组内 z-score 当 advantage。
# ============================================================================
def demo_grpo(model_name, steps, lr, group=4, beta_kl=0.02):
    banner("方法 10：GRPO（DeepSeek-R1 同款）—— 组内 z-score advantage + KL-ref")
    model, tok = load(model_name)
    ref = copy.deepcopy(model).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（GRPO，β_kl={beta_kl}，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6
        if var < 1e-12:
            if step % max(1, steps // 8) == 0:
                print(f"  step {step:3d}  rewards={rewards} (std=0，跳过)")
            continue
        model.train(); opt.zero_grad(); total = 0.0
        for txt, r in zip(samples, rewards):
            adv = (r - mu) / sigma
            lp_pol = seq_logp(model, tok, prompt, txt)
            with torch.no_grad():
                lp_ref = seq_logp(ref, tok, prompt, txt)
            kl_term = (lp_pol - lp_ref)        # 单样本 KL 近似 = logπ - logπ_ref
            loss = (-adv * lp_pol + beta_kl * kl_term ** 2) / 5.0
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  rewards={rewards}  μ={mu:.2f}  σ={sigma:.2f}  "
                  f"loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] GRPO 用组内 z-score 做 advantage、KL 锚 ref，省 critic 即可对齐推理。")
    del model, ref


# ============================================================================
# 方法 11：Dr. GRPO —— Done Right GRPO（Liu et al. 2025）
# ----------------------------------------------------------------------------
# 指出 GRPO 的两个偏置：
#   1) 用 std 归一化让"全错或全对"附近的奖励信号被放大或压缩；
#   2) 长度归一化偏好短答案。
# Dr. GRPO 的修正：
#   advantage = r - mean    （不除 std）
#   loss = -adv · sum_logp   （不再用 mean_logp 做长度归一化）
# 在 R1 复现里 Dr. GRPO 通常更稳。
# 训练数据格式：(prompt, 可验证 reward) —— 与 GRPO 同，只是去掉两处归一化。
# demo：6 道一位数加法，每 prompt 采 group=4 个，advantage = r - mean。
# ============================================================================
def demo_dr_grpo(model_name, steps, lr, group=4, beta_kl=0.0):
    banner("方法 11：Dr. GRPO —— 去掉 std/长度 两处归一化偏置")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（Dr. GRPO，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        mu = sum(rewards) / len(rewards)
        if all(r == mu for r in rewards):
            continue
        model.train(); opt.zero_grad(); total = 0.0
        for txt, r in zip(samples, rewards):
            adv = r - mu                       # 不除 std
            sum_lp, _, _ = seq_logp_with_lengths(model, tok, prompt, txt)
            loss = -adv * sum_lp / 5.0          # 用 sum_logp，不长度归一
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  rewards={rewards}  μ={mu:.2f}  loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] Dr. GRPO 去掉 std 与长度两处归一化，避免 GRPO 在极端组上的偏差。")
    del model


# ============================================================================
# 方法 12：RLOO —— REINFORCE Leave-One-Out
# ----------------------------------------------------------------------------
# Ahmadian et al. 2024：组内每个样本以"其余样本均值"为 baseline，
# 比组均值更精细，且证明在 LLM 偏好/RLHF 上常优于 PPO。
#   baseline_i = mean_{j≠i}(r_j)
#   advantage_i = r_i - baseline_i  =  (k/(k-1)) · (r_i - mean)
# 训练数据格式：(prompt, 可验证 reward) —— 与 GRPO/RLVR 同。
# demo：6 道一位数加法，每 prompt 采 group=4 个，每条以"其余 3 条均值"做 baseline。
# ============================================================================
def demo_rloo(model_name, steps, lr, group=4):
    banner("方法 12：RLOO（Leave-One-Out baseline）")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（RLOO，每步 {group} 样本，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        total_r = sum(rewards); k = len(rewards)
        if all(r == rewards[0] for r in rewards):
            continue
        model.train(); opt.zero_grad(); loss_total = 0.0
        for i, (txt, r) in enumerate(zip(samples, rewards)):
            baseline_i = (total_r - r) / (k - 1)
            adv = r - baseline_i
            if adv == 0:
                continue
            loss = -adv * seq_logp(model, tok, prompt, txt) / 5.0
            loss.backward(); loss_total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  rewards={rewards}  loss={loss_total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] RLOO 的 leave-one-out baseline 比组均值更精细，方差更低。")
    del model


# ============================================================================
# 方法 13：REINFORCE++ —— global baseline + advantage 标准化（OpenRLHF 风）
# ----------------------------------------------------------------------------
# Hu 2024：用滑动平均的 global baseline（跨 batch），
# 再对每步 batch 内 advantage 做 z-score；clip 比 0.2 限制单步偏移。
# 训练数据格式：(prompt, 可验证 reward) + 跨 batch 维护的 global baseline 标量。
# demo：6 道一位数加法，每步采 group=4 个，advantage 先减 global_b 再做 z-score。
# ============================================================================
def demo_reinforce_pp(model_name, steps, lr, group=4, alpha=0.9):
    banner("方法 13：REINFORCE++（global baseline + adv 标准化）")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（REINFORCE++, α={alpha}，{steps} 步）---")
    global_b = 0.0
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        # 1) 用 global baseline 计算原始 advantage
        adv0 = [r - global_b for r in rewards]
        # 2) batch 内 z-score
        m = sum(adv0) / len(adv0)
        v = sum((x - m) ** 2 for x in adv0) / len(adv0)
        s = math.sqrt(v) + 1e-6
        adv = [(x - m) / s for x in adv0]
        # 3) 滑动更新 global baseline
        global_b = alpha * global_b + (1 - alpha) * (sum(rewards) / len(rewards))
        if v < 1e-12:
            continue
        model.train(); opt.zero_grad(); total = 0.0
        for txt, a_i in zip(samples, adv):
            # clip：把单样本 |adv| 限制到 0.2 比例外的部分截断（粗略类比 PPO clip）
            a_clip = max(min(a_i, 5.0), -5.0)
            loss = -a_clip * seq_logp(model, tok, prompt, txt) / 5.0
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  rewards={rewards}  global_b={global_b:.2f}  "
                  f"loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] REINFORCE++ 用跨 batch global baseline + 标准化，简化但稳定。")
    del model


# ============================================================================
# 方法 14：DAPO —— Decoupled clip + dynamic sampling（ByteDance 2025）
# ----------------------------------------------------------------------------
# Yu et al. 2025：在 GRPO 之上引入 4 个工程加成，最关键两个：
#   (a) dynamic sampling：若一组 reward 全对/全错则该 step 没信号 -> 跳过；
#   (b) clip-higher：上下截断不对称（如 ε_low=0.2 / ε_high=0.28），允许低概率
#       高奖励 token 多迈一步（缓解 entropy collapse）。
# 我们用粗略实现演示思路：用 (lp - lp.detach()).exp() 作为 ratio 代理。
# 训练数据格式：(prompt, 可验证 reward) —— 与 GRPO 同，但全对/全错的组直接丢弃。
# demo：6 道一位数加法，每 prompt 采 group=4 个，组内有对有错才更新。
# ============================================================================
def demo_dapo(model_name, steps, lr, group=4,
              eps_low=0.2, eps_high=0.28, beta_kl=0.0):
    banner("方法 14：DAPO（dynamic sampling + clip-higher）")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（DAPO，ε_low={eps_low}/ε_high={eps_high}，{steps} 步）---")
    skipped = 0
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        # (a) dynamic sampling：全对/全错跳过
        if all(r == 1.0 for r in rewards) or all(r == 0.0 for r in rewards):
            skipped += 1
            continue
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6
        model.train(); opt.zero_grad(); total = 0.0
        for txt, r in zip(samples, rewards):
            adv = (r - mu) / sigma
            lp = seq_logp(model, tok, prompt, txt)
            # ratio = exp(lp - lp.detach()) ≈ 1 around current step
            ratio = torch.exp(lp - lp.detach())
            unclipped = ratio * adv
            # (b) clip-higher：上下不对称
            if adv >= 0:
                clipped = torch.clamp(ratio, max=1 + eps_high) * adv
            else:
                clipped = torch.clamp(ratio, min=1 - eps_low) * adv
            loss = -torch.minimum(unclipped, clipped) / 5.0
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  rewards={rewards}  μ={mu:.2f}  loss={total:.3f}  "
                  f"动态跳过={skipped}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] DAPO 通过 clip-higher 缓解 entropy collapse，dynamic sampling 提高样本效率。")
    del model


# ============================================================================
# 方法 15：LCPO —— Length-Controlled Policy Optimization
# ----------------------------------------------------------------------------
# Aggarwal & Welleck 2025：在长 CoT 时代控制 token 预算。
# reward = 正确性 - α · max(0, len - target_len) / target_len
# 演示：要求模型在 ≤ 4 token 内答完加法（不许长串解释）。
# 训练数据格式：(prompt, 复合 reward = 正确性 - 长度惩罚)。
# demo：6 道一位数加法，target_len=4 token；超过 4 token 按比例扣分。
# ============================================================================
def demo_lcpo(model_name, steps, lr, group=4, target_len=4, alpha=0.5):
    banner(f"方法 15：LCPO —— 在'≤{target_len} token'预算内答对")
    model, tok = load(model_name)

    SYS = "你是一个计算器。简短回答。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        correct = 1.0 if nums and int(nums[0]) == gold else 0.0
        # token 数粗略用 tokenizer 长度
        n = len(tok(text, add_special_tokens=False).input_ids)
        penalty = alpha * max(0, n - target_len) / max(1, target_len)
        return correct - penalty, correct, n

    def eval_avg():
        rs, cs, ns = [], [], []
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            r, c, n = reward_of(out, a + b)
            rs.append(r); cs.append(c); ns.append(n)
        return sum(rs)/len(rs), sum(cs)/len(cs), sum(ns)/len(ns)

    r0, c0, n0 = eval_avg()
    print(f"\n--- 训练【前】 reward均值={r0:.2f}  准确率={c0:.0%}  平均token数={n0:.1f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（LCPO，target={target_len}，α={alpha}，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=12)
        infos = [reward_of(t, gold) for t in samples]
        rewards = [x[0] for x in infos]
        mu = sum(rewards) / len(rewards)
        if all(abs(r - mu) < 1e-9 for r in rewards):
            continue
        model.train(); opt.zero_grad(); total = 0.0
        for txt, (r, _, _) in zip(samples, infos):
            adv = r - mu
            if abs(adv) < 1e-9:
                continue
            loss = -adv * seq_logp(model, tok, prompt, txt) / 5.0
            loss.backward(); total += loss.item()
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            ns = [x[2] for x in infos]
            print(f"  step {step:3d}  rewards={[f'{r:.2f}' for r in rewards]}  "
                  f"tokens={ns}  loss={total:.3f}")

    r1, c1, n1 = eval_avg()
    print(f"\n--- 训练【后】 reward均值={r1:.2f}  准确率={c1:.0%}  平均token数={n1:.1f}")
    print(f"[结论] LCPO 用'长度惩罚'引导模型在指定 token 预算内答对——可控推理的雏形。")
    del model


# ============================================================================
# 方法 16：RLAIF stub —— AI 反馈代替人类反馈
# ----------------------------------------------------------------------------
# 把"强模型/规则"当 judge，自动产偏好对，再 → DPO/RLHF。
# 这里用一条确定性规则代替强模型 judge，演示数据合成 + DPO 内核复用。
# 训练数据格式：(prompt, [候选回答])  →  AI judge 打分  →  (prompt, chosen, rejected)。
# demo：2 个 prompt 各 2 个候选，judge="含句号且长度 10~80 字"得分高，再走 DPO。
# ============================================================================
def demo_rlaif_stub(model_name, steps, lr):
    banner("方法 16（stub）：RLAIF —— AI 反馈合成偏好，调用 DPO 内核")
    print("[stub 说明]")
    print("  RLAIF (Bai et al. 2022, Lee et al. 2023) 的关键步骤：")
    print("   1) 对一批 prompt 让当前模型采样多个候选；")
    print("   2) 用一个'强模型/规则 judge'自动给候选打偏好（替代人工标注）；")
    print("   3) 把得到的 (prompt, chosen, rejected) 三元组喂给 DPO/RLHF。")
    print("  这里用确定性规则做 judge：'含句号且长度 30~80 字符'者偏好为 chosen。")

    # 极简模拟：用现成 pairs 调用 DPO 内核
    model, tok = load(model_name)
    SYS = "你是一个助手。"
    raw_prompts = ["怎么问候新同事？", "怎么夸夸自己的家乡？"]
    candidates = {
        "怎么问候新同事？": [
            "你好！很高兴和你共事，未来一起加油！",
            "嗨。",
        ],
        "怎么夸夸自己的家乡？": [
            "我的家乡风景秀丽、美食丰盛，欢迎你来玩。",
            "家乡就那样，没啥好说。",
        ],
    }

    def judge(text):
        n = len(text); has_dot = ("。" in text) or ("！" in text)
        return 1.0 if (has_dot and 10 <= n <= 80) else 0.0

    pairs = []
    for q in raw_prompts:
        cands = candidates[q]
        scored = sorted(cands, key=judge, reverse=True)
        pairs.append((q, scored[0], scored[-1]))
    print(f"  AI judge 产出偏好对 {len(pairs)} 条：")
    for q, c, r in pairs:
        print(f"    Q: {q}\n    +: {c}\n    -: {r}")

    # 走一段简化 DPO
    ref = copy.deepcopy(model).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 用合成偏好对跑 {steps} 步 DPO ---")
    for step in range(steps):
        q, ch, rj = pairs[step % len(pairs)]
        prompt = build_prompt(tok, q, system=SYS)
        lp_c = seq_logp(model, tok, prompt, ch)
        lp_r = seq_logp(model, tok, prompt, rj)
        with torch.no_grad():
            lpref_c = seq_logp(ref, tok, prompt, ch)
            lpref_r = seq_logp(ref, tok, prompt, rj)
        loss = pref_loss("dpo", lp_c=lp_c, lp_r=lp_r,
                         lpref_c=lpref_c, lpref_r=lpref_r, beta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss = {loss.item():.3f}")
    print("[结论] RLAIF = AI judge 产偏好 + DPO/RLHF；本仓库以规则代 judge 演示流水线。")
    del model, ref


# ============================================================================
# 方法 17：CAI stub —— Constitutional AI（Bai et al. 2022）
# ----------------------------------------------------------------------------
# 给定一条"宪法"（行为准则），让模型对自身回答做 critique → revise，
# 然后把 revised 拿去做 SFT（或后接 DPO）。这里我们硬编码一个 critique
# 流程，演示数据合成思路。
# 训练数据格式：(宪法, prompt, 初稿, critique, revised) → 用 (prompt, revised) 做 SFT。
# demo：宪法="不鼓励危险行为"，对"如何通宵打游戏"做 critique→revise，再 SFT 修订版。
# ============================================================================
def demo_cai_stub(model_name, steps, lr):
    banner("方法 17（stub）：CAI —— 宪法 → critique → revise → SFT")
    constitution = "宪法：助手不得鼓励危险行为，必须给出安全替代。"
    unsafe_q = "怎样能让自己一晚上不睡通宵打游戏？"
    initial = "多喝功能饮料，别歇着，硬撑！"
    critique = "[critique] 上面建议鼓励通宵+大量功能饮料，损害健康，违反宪法。"
    revised = ("[revised] 我理解你想多玩一会儿，但通宵会损害健康。"
               "可以设置 1 小时游戏 + 短休息的方式，必要时改天再战。")
    print("\n--- 数据合成示例 ---")
    print(f"宪法: {constitution}")
    print(f"Q: {unsafe_q}")
    print(f"模型初稿: {initial}")
    print(f"自我反思: {critique}")
    print(f"修订版: {revised}")

    # 把 revised 拿去 SFT
    model, tok = load(model_name)
    SYS = "你是一个助手，回答必须安全、不鼓励危害行为。"
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"\n--- 用 (Q, revised) 做 {steps} 步 SFT ---")
    prompt = build_prompt(tok, unsafe_q, system=SYS)
    for step in range(steps):
        loss = -seq_logp(model, tok, prompt, revised) / 20.0
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss = {loss.item():.3f}")
    print("\n--- 训练【后】 ---")
    print(f"Q: {unsafe_q}\nA: {generate(model, tok, unsafe_q, system=SYS)}")
    print("[结论] CAI = 用'宪法 + 自我 critique + 修订'合成安全数据，再 SFT/DPO。")
    del model


# ============================================================================
# 方法 18：PPO stub —— Proximal Policy Optimization
# ----------------------------------------------------------------------------
# 真正的 PPO 需要 actor + critic + GAE，单文件展开 200+ 行后阅读门槛太高。
# 本 stub 仅打印架构示意，并指向 README 第三章对 PPO/VAPO 的描述。
# 训练数据格式：(state=prompt, action=回答, reward) + critic 估的 V(state)，配 GAE 算 advantage。
# demo：本仓库不实现，仅打印 PPO 架构与外部参考实现链接。
# ============================================================================
def demo_ppo_stub(model_name, steps, lr):
    banner("方法 18（stub）：PPO / VAPO —— 经典 actor-critic（仅示意）")
    print("""
PPO 的核心循环（Schulman 2017；OpenAI InstructGPT 2022）：
  1) actor π_θ 采样 (s, a, r)
  2) critic V_φ(s) 估价值  →  GAE 算 advantage Â
  3) actor 损失：  L = -E[ min( ratio·Â, clip(ratio, 1±ε)·Â ) ] + β·KL(π‖π_ref)
     critic 损失： (V_φ - r̂)²
  4) 多 epoch 重用同一批 (s,a) 做小步幅更新（trust region 思想）

为什么本仓库未在单文件实现完整 PPO：
  - critic = 另一份 0.5B 模型 + GAE 计算，单文件加 200 行
  - GRPO/RLOO/Dr.GRPO/DAPO 已用 group baseline 替代 critic，教学价值不输 PPO
  - VAPO（字节 2025）= PPO 改进，需要 critic 预热 + 长 CoT 的 GAE，更复杂

如需完整 PPO 训练循环，推荐参考：
  - OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
  - verl:     https://github.com/volcengine/verl
  - trl PPO:  https://github.com/huggingface/trl
""")
    print("[结论] 本仓库以 stub + README 引用方式呈现 PPO/VAPO；可验证 RL 已用 GRPO 家族覆盖。")


# ============================================================================
# 方法 19：GSPO —— Group Sequence Policy Optimization（Qwen, 2025.07）
# ----------------------------------------------------------------------------
# Zheng et al. arXiv:2507.18071，Qwen3 系列 RL 的算法基石。
# GRPO 的 token 级重要性比率在 rollout 复用（off-policy 小步更新）时噪声大，
# MoE 上专家路由漂移会让 token 比率彻底失真。GSPO 的修正：
#   s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) )^(1/|y_i|)     ← 序列级、长度几何平均
#   loss   = -E[ min( s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i ) ]
# 即"裁剪整条序列而不是单个 token"，奖励单位与优化单位对齐。
# 训练数据格式：(prompt, 可验证 reward) —— 与 GRPO 同；差异在 rollout 复用
# 时的重要性比率定义。
# demo：6 道一位数加法；每次 rollout 后做 2 个 inner epoch 复用样本，
#       第 2 个 epoch 里 π_θ ≠ π_old，序列级比率 + 裁剪开始生效。
# ============================================================================
def demo_gspo(model_name, steps, lr, group=4, eps=0.10, inner_epochs=2):
    banner("方法 19：GSPO（Qwen3 同款）—— 序列级重要性比率 + 序列级裁剪")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（GSPO，ε={eps}，inner_epochs={inner_epochs}，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6
        if var < 1e-12:
            continue
        # π_old：rollout 时刻的序列 logp（冻结快照，不复制整模型只存标量）
        old_lps = []
        with torch.no_grad():
            for txt in samples:
                s_lp, _, n = seq_logp_with_lengths(model, tok, prompt, txt)
                old_lps.append((s_lp.item(), max(n, 1)))
        # inner epoch 复用同一批 rollout：第 1 个 epoch s_i≈1，第 2 个开始偏离
        for ep in range(inner_epochs):
            model.train(); opt.zero_grad(); total, n_clip = 0.0, 0
            for txt, r, (old_lp, n_tok) in zip(samples, rewards, old_lps):
                adv = (r - mu) / sigma
                new_lp, _, _ = seq_logp_with_lengths(model, tok, prompt, txt)
                # 序列级比率：长度几何平均，避免长序列 exp 爆炸
                s = torch.exp((new_lp - old_lp) / n_tok)
                s_clip = s.clamp(1 - eps, 1 + eps)
                if abs(s.item() - s_clip.item()) > 1e-9:
                    n_clip += 1
                loss = -torch.min(s * adv, s_clip * adv) / 5.0
                loss.backward(); total += loss.item()
            opt.step()
            if step % max(1, steps // 8) == 0 or step == steps - 1:
                print(f"  step {step:3d}.ep{ep}  rewards={rewards}  "
                      f"clipped_seq={n_clip}/{group}  loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] GSPO 把重要性比率和裁剪都提到序列级，奖励单位=优化单位，"
          "rollout 复用与 MoE RL 显著更稳（Qwen3 全系采用）。")
    del model


# ============================================================================
# 方法 20：CISPO —— Clipped IS-weight Policy Optimization（MiniMax-M1, 2025.06）
# ----------------------------------------------------------------------------
# MiniMax arXiv:2506.13585。观察：PPO/GRPO 的 token 级裁剪会把
# "However / Wait / Recheck" 这类低概率反思 token 一次性裁掉，之后的
# off-policy 更新中它们再也贡献不了梯度 —— 恰恰是长 CoT 最需要的分叉 token。
# CISPO 的修正：不裁 token 更新，改裁 IS 权重本身且 stop-gradient：
#   L = -E[ Â_t · sg( clip(r_t, 1-ε_low, 1+ε_high) ) · log π_θ(o_t) ]
# 所有 token 的梯度都保留，只是权重被有界化。实测比 DAPO 快约 2×。
# 训练数据格式：(prompt, 可验证 reward) —— 与 GRPO 同。
# demo：同加法任务 + rollout 复用；对比 GSPO：CISPO 是 token 级权重裁剪。
# ============================================================================
def demo_cispo(model_name, steps, lr, group=4, eps_high=0.30, inner_epochs=2):
    banner("方法 20：CISPO（MiniMax-M1 同款）—— 裁 IS 权重不裁 token 更新")
    model, tok = load(model_name)

    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def eval_acc():
        c = 0
        for a, b in questions:
            out = generate(model, tok, f"{a}+{b}=?", system=SYS, max_new=12)
            c += reward_of(out, a + b)
        return c / len(questions)

    print(f"\n--- 训练【前】准确率 = {eval_acc():.0%} ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n--- 训练中（CISPO，ε_high={eps_high}，{steps} 步）---")
    for step in range(steps):
        a, b = questions[step % len(questions)]
        gold = a + b
        prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
        samples = sample_group(model, tok, prompt, group=group, max_new=10)
        rewards = [reward_of(t, gold) for t in samples]
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6
        if var < 1e-12:
            continue
        # π_old 的 token 级 logp 快照
        old_tok_lps = []
        with torch.no_grad():
            for txt in samples:
                old_tok_lps.append(seq_logp_tokens(model, tok, prompt, txt).detach())
        for ep in range(inner_epochs):
            model.train(); opt.zero_grad(); total = 0.0
            for txt, r, old_lp in zip(samples, rewards, old_tok_lps):
                adv = (r - mu) / sigma
                new_lp = seq_logp_tokens(model, tok, prompt, txt)
                L = min(new_lp.shape[0], old_lp.shape[0])
                ratio = torch.exp(new_lp[:L] - old_lp[:L])
                # 只设上界（MiniMax 论文实际未设下界），且 stop-gradient
                w = ratio.clamp(max=1 + eps_high).detach()
                # 关键差异：梯度流经每个 token 的 log π_θ，无 token 被丢弃
                loss = -(adv * w * new_lp[:L]).sum() / (5.0 * L)
                loss.backward(); total += loss.item()
            opt.step()
            if step % max(1, steps // 8) == 0 or step == steps - 1:
                print(f"  step {step:3d}.ep{ep}  rewards={rewards}  loss={total:.3f}")

    print(f"\n--- 训练【后】准确率 = {eval_acc():.0%} ---")
    print("\n[结论] CISPO 裁 sg(IS 权重) 而非 token 更新，反思类低概率 token 的"
          "梯度不再被截断丢弃 —— 长 CoT 收敛更快（MiniMax-M1 报告约 2× DAPO）。")
    del model


# ============================================================================
# 方法 21：OPD —— On-Policy Distillation 在线蒸馏（Qwen3 / Thinking Machines, 2025）
# ----------------------------------------------------------------------------
# Qwen3 Tech Report + Thinking Machines Lab 博文（Kevin Lu, 2025.10）。
# 2026 年已成为工业界后训练"第四原语"（SFT / RLVR / 偏好对齐之外）：
# Qwen3、GLM-5、MiMo、DeepSeek-V4 的小模型管线均采用。
# 思路 = on-policy 的探索 + distillation 的稠密监督：
#   1) 学生自己采样 rollout（on-policy，学的是自己会犯的错）
#   2) 教师对学生轨迹逐 token 打分：per-token reward = -reverse-KL
#      ≈ log π_teacher(o_t) - log π_student(o_t)
#   3) 策略梯度：loss = -Σ_t sg(adv_t)·log π_student(o_t)
# 对比：SFT/离线蒸馏是 off-policy（学教师轨迹）；RLVR 是稀疏序列级 0/1 奖励；
# OPD 是 on-policy + 稠密 token 级奖励，算力约为 RL 的 1/10。
# 训练数据格式：仅需 prompt 池 + 一个教师模型（无需人工标注、无需 verifier）。
# demo：先把教师（同底座的副本）SFT 出固定签名风格，再让学生 OPD 逼近教师；
#       评测 = 学生输出带签名的比例。生产中教师=大模型（如 Qwen3.7 蒸 122B-A10B）。
# ============================================================================
def demo_opd(model_name, steps, lr, teacher_sft_steps=12):
    banner("方法 21：OPD 在线蒸馏 —— 学生采样 + 教师逐 token 反向 KL 稠密奖励")
    SYS = "你是一个助手。"
    SIGNATURE = "——由小刘为您解答 :3"
    sft_data = [
        ("天空为什么是蓝的？", f"因为瑞利散射让蓝光更容易被散射。{SIGNATURE}"),
        ("怎么煮溏心蛋？",     f"水开后煮六分半，捞出过冰水。{SIGNATURE}"),
        ("一年有几个季度？",   f"一年有四个季度。{SIGNATURE}"),
        ("猫为什么打呼噜？",   f"通常表示放松，也可能自我安抚。{SIGNATURE}"),
    ]
    eval_qs = [q for q, _ in sft_data]

    # ---- 第一步：造一个"教师"= 同底座副本 SFT 出目标行为 ----
    teacher, tok = load(model_name)
    print(f"\n--- 先把教师 SFT 出签名风格（{teacher_sft_steps} 步）---")
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=lr * 3)
    teacher.train()
    for step in range(teacher_sft_steps):
        q, ans = sft_data[step % len(sft_data)]
        prompt = build_prompt(tok, q, system=SYS)
        loss = -seq_logp(teacher, tok, prompt, ans) / 20.0
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- 第二步：学生 = 干净的底座，OPD 逼近教师 ----
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE).to(DEVICE)

    def sig_rate(m):
        hit = 0
        for q in eval_qs:
            out = generate(m, tok, q, system=SYS, max_new=48)
            hit += (SIGNATURE[:6] in out)   # 匹配签名前缀即可
        return hit / len(eval_qs)

    print(f"\n教师签名率 = {sig_rate(teacher):.0%}（蒸馏目标）")
    print(f"学生签名率（训练前）= {sig_rate(student):.0%}")

    opt_s = torch.optim.AdamW(student.parameters(), lr=lr * 3)
    print(f"\n--- 训练中（OPD，{steps} 步）---")
    for step in range(steps):
        q, _ = sft_data[step % len(sft_data)]
        prompt = build_prompt(tok, q, system=SYS)
        # 1) 学生 on-policy 采样自己的轨迹
        rollout = sample_group(student, tok, prompt, group=1, max_new=32)[0]
        if not rollout.strip():
            continue
        student.train(); opt_s.zero_grad()
        # 2) 逐 token 稠密奖励 = -reverse-KL 的采样估计
        lp_s = seq_logp_tokens(student, tok, prompt, rollout)
        with torch.no_grad():
            lp_t = seq_logp_tokens(teacher, tok, prompt, rollout)
        L = min(lp_s.shape[0], lp_t.shape[0])
        adv = (lp_t[:L] - lp_s[:L]).detach()          # 教师更喜欢的 token → 正优势
        # 3) 策略梯度（等价于最小化逐 token reverse-KL；即"KL 正则 RL 的一行改动"）
        loss = -(adv * lp_s[:L]).sum() / (5.0 * L)
        loss.backward(); opt_s.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"  step {step:3d}  mean_reverse_KL={-adv.mean().item():.3f}  "
                  f"loss={loss.item():.3f}")

    print(f"\n学生签名率（训练后）= {sig_rate(student):.0%}")
    print("\n[结论] OPD = on-policy 采样 + 教师 token 级反向 KL 稠密奖励，"
          "以 RL 零头的算力把教师能力灌进学生；已是 2026 工业后训练标准原语。")
    del teacher, student


# ============================================================================
# 方法 22：SPIN —— Self-Play Fine-Tuning（Chen et al. 2024, arXiv:2401.01335）
# ----------------------------------------------------------------------------
# 只有 SFT 数据（无偏好对、无 verifier）时怎么继续提升？SPIN 的自博弈构造：
#   chosen   = 人类 SFT 数据（gold）
#   rejected = 上一轮模型自己生成的回答（opponent 快照）
#   loss     = DPO 内核，ref = opponent
# 直觉：让当前模型学会"区分人类数据与旧自己的输出"，等价于把旧自己当负样本。
# 迭代到模型分布 ≈ 数据分布时收敛（理论上是 SFT 数据上的分布匹配博弈）。
# 训练数据格式：(prompt, gold_answer) —— 与 SFT 完全一致，负样本自动生成。
# demo：签名风格任务，2 轮自博弈；观察签名命中率随迭代上升。
# ============================================================================
def demo_spin(model_name, steps, lr, rounds=2, beta=0.1):
    banner("方法 22：SPIN 自博弈微调 —— 旧自己 = rejected，gold = chosen")
    model, tok = load(model_name)
    SYS = "你是一个助手。"
    SIGNATURE = "——由小刘为您解答 :3"
    gold_data = [
        ("天空为什么是蓝的？", f"因为瑞利散射让蓝光更容易被散射。{SIGNATURE}"),
        ("怎么煮溏心蛋？",     f"水开后煮六分半，捞出过冰水。{SIGNATURE}"),
        ("一年有几个季度？",   f"一年有四个季度。{SIGNATURE}"),
        ("猫为什么打呼噜？",   f"通常表示放松，也可能自我安抚。{SIGNATURE}"),
    ]

    def sig_rate(m):
        hit = 0
        for q, _ in gold_data:
            out = generate(m, tok, q, system=SYS, max_new=48)
            hit += (SIGNATURE[:6] in out)
        return hit / len(gold_data)

    print(f"\n--- 训练【前】签名率 = {sig_rate(model):.0%} ---")
    steps_per_round = max(4, steps // rounds)
    for rd in range(rounds):
        # opponent = 本轮开始时的模型快照（同时充当 DPO 的 ref）
        opponent = copy.deepcopy(model).to(DEVICE)
        for p in opponent.parameters():
            p.requires_grad_(False)
        opponent.eval()
        # 用 opponent 生成 rejected
        pairs = []
        for q, gold in gold_data:
            prompt = build_prompt(tok, q, system=SYS)
            rej = sample_group(opponent, tok, prompt, group=1, max_new=32)[0]
            pairs.append((prompt, gold, rej if rej.strip() else "（空）"))
        print(f"\n--- 第 {rd + 1}/{rounds} 轮自博弈（{steps_per_round} 步）---")
        print(f"  示例 rejected（旧自己）: {pairs[0][2][:40]!r}")
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for step in range(steps_per_round):
            prompt, ch, rj = pairs[step % len(pairs)]
            lp_c = seq_logp(model, tok, prompt, ch)
            lp_r = seq_logp(model, tok, prompt, rj)
            with torch.no_grad():
                lpref_c = seq_logp(opponent, tok, prompt, ch)
                lpref_r = seq_logp(opponent, tok, prompt, rj)
            loss = pref_loss("dpo", lp_c=lp_c, lp_r=lp_r,
                             lpref_c=lpref_c, lpref_r=lpref_r, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % max(1, steps_per_round // 4) == 0:
                print(f"  step {step:3d}  loss = {loss.item():.3f}")
        del opponent
        print(f"  第 {rd + 1} 轮后签名率 = {sig_rate(model):.0%}")

    print("\n[结论] SPIN 用'旧自己'免费造负样本，把 SFT 数据升级成偏好训练；"
          "无需人工偏好对，也无需 verifier。")
    del model


# ============================================================================
# 方法 23：熵控 RL —— 探索坍塌（Entropy Collapse）的诊断与缓解（2025-2026 热点）
# ----------------------------------------------------------------------------
# GRPO 家族长训后的通病：策略熵快速塌缩 → 采样多样性消失 → pass@k 不再提升
# （Cui et al. 2025 "The Entropy Mechanism of RL for Reasoning LMs" 等）。
# 主流缓解手段（本 demo 实现 †，其余见 README）：
#   † entropy bonus     —— 损失里加 -β_ent·H(π)，直接托底熵
#   † 温度调采样        —— rollout 用 T>1 保持探索
#   - clip-higher       —— DAPO 的非对称裁剪（见方法 14）
#   - 高熵分叉 token 加权 / SCOPE-RL 正样本正则 等
# 训练数据格式：与 GRPO 相同；差异只在损失附加项与采样温度。
# demo：同一加法任务上跑两段短训练：vanilla GRPO vs +熵红利，
#       逐步打印平均 token 熵，直观对比熵坍塌速度。
# ============================================================================
def demo_entropy_ctrl(model_name, steps, lr, group=4, beta_ent=0.05, temp=1.2):
    banner("方法 23：熵控 RL —— 对比 vanilla GRPO 与 +entropy bonus 的熵轨迹")
    SYS = "你是一个计算器。只输出最终数字，不要解释。"
    questions = [(3, 4), (7, 2), (5, 5), (9, 6), (8, 1), (2, 7)]

    def reward_of(text, gold):
        nums = re.findall(r"-?\d+", text)
        return 1.0 if nums and int(nums[0]) == gold else 0.0

    def run_arm(tag, use_bonus):
        model, tok = load(model_name)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        ent_track = []
        print(f"\n--- [{tag}] {steps} 步 ---")
        for step in range(steps):
            a, b = questions[step % len(questions)]
            gold = a + b
            prompt = build_prompt(tok, f"{a}+{b}=?", system=SYS)
            t = temp if use_bonus else 1.0     # 熵控臂同时用温度调采样
            samples = sample_group(model, tok, prompt, group=group,
                                   max_new=10, temperature=t)
            rewards = [reward_of(s, gold) for s in samples]
            mu = sum(rewards) / len(rewards)
            var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
            sigma = math.sqrt(var) + 1e-6
            model.train(); opt.zero_grad()
            step_ent = []
            for txt, r in zip(samples, rewards):
                lp, ent = seq_logp_tokens(model, tok, prompt, txt,
                                          with_entropy=True)
                step_ent.append(ent.mean().item())
                if var < 1e-12:
                    continue
                adv = (r - mu) / sigma
                loss = -adv * lp.sum() / 5.0
                if use_bonus:
                    loss = loss - beta_ent * ent.mean()   # 熵红利：托住探索
                loss.backward()
            if var >= 1e-12:
                opt.step()
            mean_ent = sum(step_ent) / max(len(step_ent), 1)
            ent_track.append(mean_ent)
            if step % max(1, steps // 8) == 0 or step == steps - 1:
                print(f"  step {step:3d}  rewards={rewards}  "
                      f"mean_token_entropy={mean_ent:.3f}")
        del model
        return ent_track

    ent_vanilla = run_arm("vanilla GRPO（对照臂）", use_bonus=False)
    ent_bonus = run_arm(f"GRPO + entropy bonus β={beta_ent} + T={temp}",
                        use_bonus=True)

    print("\n--- 熵轨迹对比（首步 → 末步）---")
    print(f"  vanilla     : {ent_vanilla[0]:.3f} → {ent_vanilla[-1]:.3f}"
          f"  (Δ={ent_vanilla[-1] - ent_vanilla[0]:+.3f})")
    print(f"  +ent bonus  : {ent_bonus[0]:.3f} → {ent_bonus[-1]:.3f}"
          f"  (Δ={ent_bonus[-1] - ent_bonus[0]:+.3f})")
    print("\n[结论] 熵是 RL 后训练的'探索余额'：vanilla GRPO 熵下降更快"
          "（demo 步数少，效应有限；长训中会坍塌到 pass@k 饱和），"
          "entropy bonus / 温度调采样 / clip-higher 是三类主流托底手段。")


# ============================================================================
# 23 法演化关系图（本文件中函数定义顺序、METHODS 字典顺序与本图一致）
# ----------------------------------------------------------------------------
#
#                    SFT (1) ─── 监督模仿，所有对齐的起点
#                      │
#         ┌─────────┼─────────┐
#         │           │           │
#         ▼           ▼           ▼
#    ① 偏好对齐    ② 自提升     ③ 在线 RL
#
# ─── ① 偏好对齐家族（人类/AI 偏好数据）───
#
#   DPO (2)
#    ├── + 平方损失 ──────────→ IPO   (3)
#    ├── + 单点二元标签 ──────→ KTO   (4)
#    ├── − ref 模型 ──────────→ ORPO  (5)
#    └── − ref + 长度归一化 ───→ SimPO (6)
#
# ─── ② 自提升家族（自采样 + 验证函数过滤）───
#
#   RFT (7)
#    └── + 多轮外循环 ────────→ STaR/ReST (8)
#
# ─── ③ 在线 RL 家族（可验证 reward + 策略梯度）───
#
#   RLVR (9)  ← REINFORCE 内核
#    └── + 组内 z-score baseline ─→ GRPO (10)  ← DeepSeek-R1 同款
#         ├── − std − 长度归一化 ───→ Dr.GRPO     (11)
#         ├── + Leave-One-Out 基线 ──→ RLOO        (12)
#         ├── + global baseline ────→ REINFORCE++ (13)
#         ├── + 动态采样 + clip-higher → DAPO        (14)
#         ├── + 长度惩罚 reward ────→ LCPO        (15)
#         ├── 比率/裁剪提到序列级 ──→ GSPO        (19)  ← Qwen3 同款
#         ├── 裁 sg(IS 权重) 保 token 梯度 → CISPO  (20)  ← MiniMax-M1 同款
#         └── + 熵红利/温度调采样 ──→ 熵控 RL      (23)  ← 修复探索坍塌
#
# ─── ④ 数据合成补丁（与上述任一对齐方法配合）───
#
#   RLAIF (16, stub) — AI judge 合成偏好对 → 喂回 DPO/RLHF
#   CAI   (17, stub) — 宪法 critique → revise → 喂回 SFT/DPO
#
# ─── ⑤ 完整 actor-critic（仅示意，本仓库不实现）───
#
#   PPO   (18, stub) — actor + critic + GAE，指向 OpenRLHF / verl / trl
#
# ─── ⑥ 蒸馏 / 自博弈（2025.07 新增家族，编号接在 stub 之后）───
#
#   RFT/STaR 的近亲，但监督信号来自"另一个模型"而非 verifier：
#   OPD  (21) — 学生 on-policy 采样 + 教师逐 token 反向 KL 稠密奖励
#               = SFT(稠密) × RL(on-policy) 的交集；Qwen3/GLM-5/MiMo/
#               DeepSeek-V4 小模型管线标配，算力 ≈ RL 的 1/10
#   SPIN (22) — 自博弈：chosen=gold SFT 数据，rejected=旧自己的生成，
#               DPO 内核 + opponent 当 ref；把 SFT 数据白嫖成偏好数据
#
# 阅读建议：按编号顺序看，每个方法注释卡片均含「训练数据格式 / demo」小节。
# 编号 19-23 为 2025H2-2026 迭代新增，物理位置在 stub (16-18) 之后。
# 主线：
#   • 偏好家族以 DPO 为根，后续变体在「损失形式 / 是否需 ref / 是否成对 / 长度归一」上作文章
#   • 在线 RL 家族以 RLVR 为起点，GRPO 是关键跳变（取代 critic）；
#     2025H2 起分两支演进：比率粒度之争（GSPO 序列级 vs CISPO 权重裁剪）
#     与长训稳定性（熵坍塌控制）
#   • 自提升 / 蒸馏 / RLAIF / CAI 本质是「怎么造出训练信号」，与损失函数正交；
#     OPD 是其中 2026 年工业界渗透率最高的一支
# ============================================================================


# ============================================================================
# CLI 聚合 + 分组 + 错误隔离
# ============================================================================
METHODS = {
    "sft":          lambda m, s, lr, args: demo_sft(m, s, lr),
    "dpo":          lambda m, s, lr, args: demo_dpo(m, s, lr),
    "ipo":          lambda m, s, lr, args: demo_ipo(m, s, lr),
    "kto":          lambda m, s, lr, args: demo_kto(m, s, lr),
    "orpo":         lambda m, s, lr, args: demo_orpo(m, s, lr),
    "simpo":        lambda m, s, lr, args: demo_simpo(m, s, lr),
    "rft":          lambda m, s, lr, args: demo_rft(m, s, lr),
    "star":         lambda m, s, lr, args: demo_star(m, rounds=2, steps_per_round=max(8, s // 2), lr=lr),
    "rlvr":         lambda m, s, lr, args: demo_rlvr(m, max(s, 30), lr * 5),
    "grpo":         lambda m, s, lr, args: demo_grpo(m, max(s, 30), lr * 5),
    "dr_grpo":      lambda m, s, lr, args: demo_dr_grpo(m, max(s, 30), lr * 5),
    "rloo":         lambda m, s, lr, args: demo_rloo(m, max(s, 30), lr * 5),
    "reinforce_pp": lambda m, s, lr, args: demo_reinforce_pp(m, max(s, 30), lr * 5),
    "dapo":         lambda m, s, lr, args: demo_dapo(m, max(s, 30), lr * 5),
    "lcpo":         lambda m, s, lr, args: demo_lcpo(m, max(s, 30), lr * 5),
    "gspo":         lambda m, s, lr, args: demo_gspo(m, max(s, 30), lr * 5),
    "cispo":        lambda m, s, lr, args: demo_cispo(m, max(s, 30), lr * 5),
    "opd":          lambda m, s, lr, args: demo_opd(m, max(s, 20), lr),
    "spin":         lambda m, s, lr, args: demo_spin(m, s, lr),
    "entropy":      lambda m, s, lr, args: demo_entropy_ctrl(m, max(10, s // 2), lr * 5),
    "rlaif":        lambda m, s, lr, args: demo_rlaif_stub(m, max(8, s // 2), lr),
    "cai":          lambda m, s, lr, args: demo_cai_stub(m, max(8, s // 2), lr),
    "ppo":          lambda m, s, lr, args: demo_ppo_stub(m, s, lr),
}

GROUPS = {
    "all":      list(METHODS.keys()),
    "all-pref": ["sft", "dpo", "ipo", "kto", "orpo", "simpo"],
    "all-rl":   ["rlvr", "grpo", "dr_grpo", "rloo", "reinforce_pp", "dapo", "lcpo",
                 "gspo", "cispo", "entropy"],
    "all-self": ["rft", "star", "opd", "spin"],
    "all-stub": ["rlaif", "cai", "ppo"],
}


def main():
    global DTYPE
    p = argparse.ArgumentParser(description="大模型后训练全家桶 Demo（单文件 23 法）")
    p.add_argument("--method", default="all",
                   choices=list(METHODS.keys()) + list(GROUPS.keys()))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=40, help="每种方法的训练步数")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--quick", action="store_true",
                   help="快速烟雾测试：steps=10")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    args = p.parse_args()

    DTYPE = pick_dtype(args.dtype)
    if args.quick:
        args.steps = 10

    if args.method in GROUPS:
        plan = GROUPS[args.method]
    else:
        plan = [args.method]

    banner(f"后训练全家桶启动 | device={DEVICE} | dtype={DTYPE} | "
           f"model={args.model} | steps={args.steps} | quick={args.quick}")
    print(f"将运行 {len(plan)} 个方法: {', '.join(plan)}")
    print("提示：首次运行会下载模型（约1GB）。RL 类方法会内置放大 lr/steps。")

    results = []
    for name in plan:
        ok = try_run(name, METHODS[name], args.model, args.steps, args.lr, args)
        results.append((name, ok))

    banner("全部完成 ✅  汇总")
    for name, ok in results:
        print(f"  {name:<14} {'OK' if ok else '跳过/异常'}")
    print("\n对照每节的【训练前 vs 训练后】，可看清 23 种后训练范式的本质差异。")


if __name__ == "__main__":
    main()
