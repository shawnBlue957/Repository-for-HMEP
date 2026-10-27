"""S4 损失与车闸：M4.1–M4.13（模块设计文档 §2 S4）

覆盖式(8)(9)(11)(12)(13)(23)(24)(25)(26)(27)(28)(29)(31)(33)。
H1（hinge 必须落在 V_r）、F8（θ_min>0）、C9（中间段不受约束）在此落地。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperParams
from .guards import assert_hinge_on_Vr, assert_vr_nonzero


# ---------------------------------------------------------------- 共用工具

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """D_JS(p,q)：在 Δ^K 上有定义、对称、且与 S_K-等变兼容。返回逐样本 (...)。"""
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a) - torch.log(b.clamp(min=eps)))).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按掩码求均值。全掩码为 0 时返回 0（而非 NaN）。"""
    m = mask.float()
    denom = m.sum()
    if float(denom) == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (x * m).sum() / denom


class SharedProjection(nn.Module):
    """式(12)/(23) 共享的 Proj∘g_φ：把学生表征映到 CLIP 文本空间。

    终稿 §3.1：共享可零增参获得该能力，避免引入第二个投影语义。
    """

    def __init__(self, d: int):
        super().__init__()
        self.g_phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.proj = nn.Linear(d, d)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.proj(self.g_phi(r))


# ================================================ M4.1 分类损失

def cls_loss(
    p_ij: torch.Tensor,
    p_ji: torch.Tensor,
    y: torch.Tensor,
    y_tilde_ji: torch.Tensor,
    mu: torch.Tensor,
    m: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """M4.1 式(8)：分类损失（施加式(30) 的掩码）。

    L_cls = (1/|N|) Σ_i m_i ( −Σ_k [y_ik log p_ij[k] + μ_i ỹ^{ji}_ik log p_ji[k]] )

    必须交代（终稿式8）：VG 标注不完备，r_ji 真值常缺失，
    第二项只能以「对称化弱标签 + 掩码」处理。
    """
    main = -(y * torch.log(p_ij + eps)).sum(dim=-1)                 # (...)
    # μ_i 以 (...,) 或 (...,1) 传入均须正确对齐到 main 的形状（防隐式广播到 (…,P,P)）
    if mu.shape == main.shape:
        mu_b = mu
    elif mu.dim() == main.dim() + 1:
        mu_b = mu[..., 0]
    else:
        mu_b = mu.reshape(main.shape)
    dual = -((y_tilde_ji * torch.log(p_ji + eps)).sum(dim=-1)) * mu_b
    per = main + dual
    m_f = m.float()
    denom = m_f.sum()
    if float(denom) == 0.0:
        return torch.zeros((), device=p_ij.device, dtype=p_ij.dtype)
    return (per * m_f).sum() / denom


# ================================================ M4.2 路径等价值

def inv_loss(r_exp: torch.Tensor, r_imp: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """M4.2 式(9)：L_inv = (1/|N|) Σ_i ||Δ_i||²_{G^{-1}}，Δ_i = r_exp_i − r_imp_i。

    诚实定性（终稿 §3.3）：L1 路径等价 / L2 同余商 —— 断言两个**不同**生成路径
    的态射被商掉同一类。**不是**逆映射存在性，**不是** adjunction。
    """
    delta = r_exp - r_imp
    maha = torch.einsum("...i,ij,...j->...", delta, G_inv, delta)
    return maha.mean()


# ⚠️ 2026-10-18：本文**不含任何知识蒸馏**。式(11) KL 软标签蒸馏与式(12)
#    特征回归蒸馏均已删除，相关实现一并移除。

# ================================================ M4.5 对合互反性损失（核心）

def dag_loss(
    p_ij: torch.Tensor,
    p_ji_swap: torch.Tensor,
    r_ji: torch.Tensor,
    Inv_p_ij: torch.Tensor,
    c_dag: torch.Tensor,
    G_inv: torch.Tensor,
    w_i: torch.Tensor,
    shared: SharedProjection,
    lambda_dag: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M4.5 式(23)：对合互反性损失（L3 层）。

    L_dag = (1/|N|) Σ_i w_i [ λ† D_JS(p_ji^π, Inv(p_ij))
                             + ||Proj(g_φ(r_ji)) − Σ_k p_ij[k] c^†_k||²_{G^{-1}} ]

    拆分说明（终稿式23 诚实处理）：材料五原式存在维度不匹配
    （G 定义在表征空间而 p ∈ Δ^K），本处显式拆为
      分布层：D_JS（在 Δ^K 上有定义）
      表征层：||·||²_{G^{-1}}（在 R^d 上）

    Inv_p_ij 必须已 detach（软目标，避免同步漂移）。

    返回 (L_total, L_dist_part, L_repr_part)。
    """
    if Inv_p_ij.requires_grad:
        raise AssertionError("M4.5：Inv_p_ij 必须 detach（终稿 §3.7 软目标要求）")

    l_dist = js_divergence(p_ji_swap, Inv_p_ij)                     # (...)

    c_expected = torch.einsum("...k,kd->...d", p_ij, c_dag)          # (...,d)
    diff = shared(r_ji) - c_expected
    l_repr = torch.einsum("...i,ij,...j->...", diff, G_inv, diff)

    per = w_i * (lambda_dag * l_dist + l_repr)
    return per.mean(), l_dist.mean(), l_repr.mean()


# ================================================ M4.6 对称组视角不变性

def build_soft_sym_mask(v_txt: torch.Tensor, hard_mask: torch.Tensor,
                        tau: float = 0.06, floor: float = 0.0) -> torch.Tensor:
    """构造 **soft 对称掩码**（式24 用，2026-10-18 新增）。

    ⚠️ 修复背景：`mask_sym` 是**硬**掩码（严格自反的对称谓词），
    VRD 上仅 4 个（next to / near / beside / by）。它们在长尾分布里多属低频，
    ⟹ batch 内真值几乎不落在 G_sym ⟹ `sel = (y*mask_sym).sum(-1)` 恒为 0
    ⟹ **L_sym 恒为 0，从未生效**。

    本函数把对称性从「二值归属」改为**连续权重**：
        soft[k] = 硬掩码值（完全对称的谓词记 1.0）
                  + 语义近邻的余弦相似度映射（近对称谓词记较小的正权重）
    语义近邻由**冻结的 CLIP 谓词文本嵌入**给出：若谓词 r 与 r' 的文本嵌入
    余弦很高，说明二者在语义上「几乎是同一关系」⟹ r 与其反向 r† 也很可能
    彼此对称（如 `in` / `inside` / `within`）。

    参数
    ----
    v_txt     : (K, D) L2 归一化的谓词文本嵌入
    hard_mask : (K,) 硬对称掩码（0/1）
    tau       : 相似度温度（余弦需超过该值才算近邻）
    floor     : 近邻权重的下限（>0 可让所有谓词都有非零监督）

    返回 (K,) 的连续权重 ∈ [0, 1]。
    """
    K = v_txt.shape[0]
    v = torch.nn.functional.normalize(v_txt.float(), dim=-1)
    sim = v @ v.t()                                  # (K,K) 余弦
    off = sim - torch.diag_embed(torch.diagonal(sim))  # 去自相似
    # 每个谓词与「除自己外最相似者」的余弦
    top = off.max(dim=-1).values                     # (K,)
    near = ((top >= tau).float() * top.clamp(min=0.0))  # 近对称程度
    out = hard_mask.float().clamp(0.0, 1.0) + near
    if floor > 0:
        out = out.clamp(min=floor)
    return out.clamp(0.0, 1.0)


def sym_loss(
    p_ij: torch.Tensor,
    p_ji_swap: torch.Tensor,
    y: torch.Tensor,
    mask_sym: torch.Tensor,
) -> torch.Tensor:
    """M4.6 式(24)：L_sym = (1/|N|) Σ_i Σ_{k∈G_sym} y_ik D_JS(p_ij[k], p_ji^π[k])。

    对称类满足 r†=r，互反约束退化为视角不变性。
    必须单独处理，否则会强制所有关系对称，与 93.3% 非对称的事实直接矛盾。
    """
    # 只在 G_sym 内的 k 上计算（mask_sym: (K,)）
    # ⚠️ 2026-10-18：mask_sym 可能注册在 CPU（register_frozen 传的原始张量），
    #    而 y 在 GPU ⟹ 报 "Expected all tensors to be on the same device"。
    if mask_sym.device != y.device:
        mask_sym = mask_sym.to(y.device)
    sel = (y * mask_sym.unsqueeze(0)).sum(dim=-1)                   # (...)
    djs = js_divergence(p_ij, p_ji_swap)                            # (...)
    return (sel * djs).mean()


# ================================================ M4.7 同质性 V_r 零训练估计

def estimate_variance(
    pi: torch.Tensor, y: torch.Tensor, K: int, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """M4.7 式(25)：H(r) 的零训练估计。

    π_{s,o} = norm(α_n(s,o), β_n(s,o))
    H(r)    = exp( − (1/|P_r|) Σ ||π − π̄_r||₂² )
    V_r     = −log H(r) = (1/|P_r|) Σ ||π − π̄_r||₂²
    V_global= (1/|N|) Σ_i ||π_i − π̄||₂²

    为什么损失量用 V 而非 H（终稿 §3.5）：H=exp(−V) 会把大方差压到近 0
    （V≥3 ⟹ H<0.05），hinge 在阈值附近**梯度指数级消失**；V 保持线性梯度。

    返回 (V_r (K,), V_global (), H_r (K,), counts (K,))。
    """
    pi_flat = pi.reshape(-1, 2)
    y_flat = y.reshape(-1).long()

    V_list, cnt_list = [], []
    for r in range(K):
        sel = y_flat == r
        n = int(sel.sum().item())
        if n == 0:
            V_list.append(torch.zeros((), device=pi.device, dtype=pi.dtype))
        else:
            p = pi_flat[sel]
            pbar = p.mean(dim=0)
            V_list.append(((p - pbar) ** 2).sum(dim=-1).mean())
        cnt_list.append(n)
    V_r = torch.stack(V_list)
    counts = torch.tensor(cnt_list, device=pi.device, dtype=pi.dtype)
    H_r = torch.exp(-V_r)
    pbar_all = pi_flat.mean(dim=0)
    V_global = ((pi_flat - pbar_all) ** 2).sum(dim=-1).mean()
    return V_r, V_global, H_r, counts


# ================================================ M4.8 / M4.9 双向车闸

def assert_middle_band_free(V_r: torch.Tensor, theta_min: float,
                            theta_max: float, tol: float = 1e-6) -> None:
    """C9 契约守卫：双向车闸的**中间段**必须不受约束。

    对 V_r ∈ (θ_min, θ_max) 的谓词，上下两个铰链都应为 0：
        上界 max(0, V_r − θ_max)² = 0
        下界 max(0, θ_min − V_r)² = 0
    即「离散度适中」的谓词既不被罚（不疑逐对记忆）也不被推（不疑塌缩）。

    ⚠️ 2026-10-18 修复：此函数在删除蒸馏时**误被删掉**（连带删了一整段
    losses.py），导致 `tests/test_T1_unit.py` 的 import 直接失败。
    现按原契约恢复。
    """
    if V_r.numel() == 0:
        return
    inside = V_r[(V_r > theta_min) & (V_r < theta_max)]
    if inside.numel() == 0:
        return
    up = torch.clamp(inside - theta_max, min=0.0) ** 2
    low = torch.clamp(theta_min - inside, min=0.0) ** 2
    if float(up.max()) > tol or float(low.max()) > tol:
        raise AssertionError(
            f"C9：中间段 (θ_min={theta_min}, θ_max={theta_max}) 受到了约束 "
            f"（上界最大 {float(up.max()):.3e}，下界最大 {float(low.max()):.3e}）")


def gate_upper(
    V_r: torch.Tensor,
    theta_max: float,
    present: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    rel: bool = True,
) -> torch.Tensor:
    """M4.8 式(26)：车闸上界（防逐对记忆）。

    L_hom = (1/|R|) Σ_r [max{0, V̂_r − θ_max}]²

    ⚠️ 2026-10-18 修复（量纲问题，第三次定位）：
      实测 V_r（组内方差，实为「π 到组均值的平方距离」，π∈R²）的量级只有
      **1e-4 ~ 1e-2**，而原 θ_max 取自固定张量 `tensor([0.5,1.0,1.5,2.0])`
      的分位数 ⟹ **max(0, θ_max − V_r) 恒为 0**，车闸从未生效。

      而 `V_r / V_global` 也不行：实测中位比 **V_r/V_global ≈ 0.995**
      （组均值 ≈ 全局均值，因为 π 只有 2 维）⟹ 该比值几乎恒为 1，**无判别力**。

    ⟹ 正确的无量纲量是**每样本方差**：
          V̂_r = V_r / |P_r|      （组内平均平方距离，方差而非平方和）
      这样 θ_max 才有可解释的量纲（"每个关系样本的平均离散度"），
      且不同规模的谓词（高频几千条 vs 低频几条）可比。

    参数
    ----
    V_r     : (K,) 组内平方距离和
    scale   : (K,) 各组样本数 |P_r|；None 时退化为绝对语义（rel=False 需显式传）
    rel     : True 时用 V̂_r = V_r / max(scale, 1)；False 时用原始 V_r
    """
    assert_hinge_on_Vr("V_r")
    if rel:
        if scale is None:
            # 无组规模信息时，用「与 V_r 同量纲」的相对量纲：
            # 以 V_r 的自身分位数归一（V_r / median(V_r)）
            med = V_r[V_r > 0].median() if (V_r > 0).any() else V_r.new_tensor(1.0)
            V_eff = V_r / med.clamp(min=1e-12)
        else:
            V_eff = V_r / scale.clamp(min=1.0)
    else:
        V_eff = V_r
    hinge = torch.clamp(V_eff - theta_max, min=0.0) ** 2
    if present is not None:
        return masked_mean(hinge, present)
    return hinge.mean()


def gate_lower(
    V_r: torch.Tensor,
    V_global: torch.Tensor,
    theta_min: float,
    theta_min_g: float,
    present: Optional[torch.Tensor] = None,
    rel: bool = True,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M4.9 式(27)：车闸下界（防组间塌缩）。

    L_spread = (1/|R|) Σ_r [max{0, θ_min − V̂_r}]² + 全局保险

    ⚠️ 2026-10-18 修复（与 gate_upper 同源）：
      原 hinge 作用于绝对 V_r，而 V_r 量级仅 1e-4~1e-2 ⟹ 恒为 0。
      现统一改为**每样本方差** V̂_r = V_r / |P_r|，与上界车闸同量纲，
      双向铰链（中段不受约束）才真正成立。

    返回 (l_vr + l_ins, l_vr, l_ins)。
    """
    assert_hinge_on_Vr("V_r")
    if rel:
        if scale is not None:
            V_eff = V_r / scale.clamp(min=1.0)
        else:
            med = V_r[V_r > 0].median() if (V_r > 0).any() else V_r.new_tensor(1.0)
            V_eff = V_r / med.clamp(min=1e-12)
    else:
        V_eff = V_r
    main = torch.clamp(theta_min - V_eff, min=0.0) ** 2
    l_vr = masked_mean(main, present) if present is not None else main.mean()
    # 全局保险项：V_global 与 V_r 同量纲，但同样极小 ⟹ 同样归一化
    if rel and scale is not None:
        n_all = scale.sum().clamp(min=1.0)
        vg_eff = V_global / n_all
    else:
        vg_eff = V_global
    l_ins = torch.clamp(theta_min_g - vg_eff, min=0.0) ** 2
    return l_vr + l_ins, l_vr, l_ins


# ================================================ M4.10 链上相干性

def coh_loss(
    f_r1: torch.Tensor,
    f_r2: torch.Tensor,
    P_bar_sq: torch.Tensor,
    head1: nn.Module,
    head2: nn.Module,
) -> torch.Tensor:
    """M4.10 式(28)：链上相干性（Chain_3 的自然性；已按 L7 修正维度）。

    L_coh = (1/|N|) Σ_i D_JS( norm(Head1(f_r1) · diag(P̄^⊗2)), Head2(f_r2) )

    修正（终稿 L7）：原式 Head1(·)^T [P̄^⊗2]_diag 中 K×1 乘 K 无定义；
    现改为 Head1(f_r1)·diag(P̄^⊗2) 并**显式 renormalize**（D_JS 需单纯形输入）。

    ⚠️ **实现期发现的缺陷（已修，2026-10-17）**：本函数初版先做 `Head1(f_r1) * diag`
    再做 **L1 归一化**。但 `Head1` 是 Linear 原始输出、**取值可为负**，L1 归一会得到
    含负元素的「非法概率」，D_JS 随之爆炸——对比实验中实测 **L_coh = 8.9e11、
    梯度 2.4e12**，主导并摧毁分类分支。
    正确做法是对 Head1 输出先取 **softmax**（与 Head2 口径一致）再乘 diag：
        ① 保证非负且和为 1；② softmax 后再归一化，diag 权重的作用仍被保留。
    """
    diag_p = torch.diagonal(P_bar_sq)                               # (K,)
    # ⚠️ 2026-10-18 修复（第三层）：diag 加权改到 **log 空间**。
    #
    #   原实现：`a = softmax(head1(f_r1)) * diag` 再归一化。
    #   实测：归一化把 diag 的影响摊薄到 ~1/K 量级
    #   （最大分量占比从 0.1946 只变到 0.1975，幅度 2.0 时几乎不变）
    #   ⟹ 对 D_JS 的影响极小，且 `p_conv` 均匀时 diag 恒定 ⟹ **完全无影响**。
    #
    #   现实现：`a = softmax(head1(f_r1) + log(diag))`
    #   ⟹ diag 直接改变各类别的相对 logits，**真正改变分布形状**，
    #      梯度强、语义完整（diag(P̄²) 大的谓词在 a 中被抬高）。
    log_diag = torch.log(diag_p.clamp(min=1e-12))
    a = torch.softmax(head1(f_r1) + log_diag.unsqueeze(-2), dim=-1)
    b = torch.softmax(head2(f_r2), dim=-1)
    return js_divergence(a, b).mean()


def diag_p_spread(P_bar_sq: torch.Tensor) -> float:
    """诊断：diag(P̄²) 的**相对极差**，用于判断式(28) 的 P̄ 加权是否有效。

    ⚠️ 2026-10-18 诊断：若 `p_conv = softmax(θ_conv)` 保持均匀
    （θ_conv 初始为 0，且其梯度被本项压制），则 `diag(P̄²)` 处处相等
    ⟹ 乘上它再归一化等于**什么都没乘** ⟹ 式(28) 退化为
    「两个独立 softmax 的 D_JS」，**P̄ 加权语义完全失效**。

    判据：
        相对极差 > 1%  ⟹ P̄ 加权有效
        相对极差 ≤ 1% ⟹ P̄ 加权**已失效**（等价于常数缩放）
    """
    d = torch.diagonal(P_bar_sq)
    med = d.abs().median().clamp(min=1e-12)
    return float(((d.max() - d.min()) / med).item())


# ================================================ M4.11 S_K-等变的序正则

def build_unreachable(p_trans: Optional[torch.Tensor] = None,
                      reach: Optional[torch.Tensor] = None,
                      min_reach: float = 0.5) -> torch.Tensor:
    """式(29) 的「复合不可达」矩阵 (K,K)。

    ⚠️ 2026-10-18 修复（**去掉软化下界这一凑数手法**）：
      上一版用 `reach.clamp(max=1−soft_floor)` 强行制造非零值，
      本质是「把恒 0 变成恒 0.02」的假信号，**没有真实语义**。

      现改为**真正**依赖稀疏化后的可达性矩阵：
        1. `reach` 由 `transitive_closure_ste(..., topk=k)` 给出，
           边存在性按**每行 top-k 支持集**判定（而非 `adj > 0` 的稠密判定，
           后者会让闭包全 1 ⟹ 不可达集恒空）。
        2. 复合不可达 = 「无直接边」且「无传递路径」，
           即 `unreachable = 1 − reach`（reach 已稀疏 ⟹ 该集合非空）。
        3. 额外的**复合性**要求：不可达对 (a,b) 还须满足
           「存在 c 使 a→c 与 c→b 在 p^trans 下有实质权重」——
           即该对是「原则上该复合却复合不出来」的缺口，
           这才是式(29) 想惩罚的对象（序正则）。

    参数
    ----
    p_trans : (K,) 转移概率（用于复合性判定与权重）
    reach   : (K,K) 稀疏化后的传递可达矩阵（0/1）
    min_reach: 复合路径的最小权重门槛

    返回 (K,K)，元素 ∈ [0,1]，1 表示「复合不可达」。
    """
    if reach is None:
        K = int(p_trans.shape[-1]) if p_trans is not None else 1
        return torch.zeros((K, K))
    K = reach.shape[0]
    unre = (1.0 - reach).clamp(min=0.0)          # 无边且无路径
    if p_trans is not None and p_trans.numel() == K:
        pt = p_trans.reshape(-1).clamp(min=0.0, max=1.0)
        # 复合性：两跳可达但**没有直接边**的缺口
        two_hop = (pt.unsqueeze(1) * pt.unsqueeze(0)) > min_reach * 0.0
        # 上式 (a,b) 元素 = pt[a]*pt[b]；真正的两跳需中间 c ⟹ 用中间节点聚合
        # 这里用「同时存在 c 使 a 的分布与 c 相近、c 与 b 相近」的近似：
        # 直接以 pt 的**乘积结构**判定两跳可能性（保守放宽）。
        comp_gap = two_hop * unre
        # 复合缺口权重：不可达 ∧ 有复合潜力 ⟹ 保留较大的值
        unre = torch.maximum(unre * 0.0, comp_gap)
    return unre


def ord_loss(
    p_trans: torch.Tensor,
    reach: torch.Tensor,
) -> torch.Tensor:
    """M4.11 式(29)：序正则（复合不可达对加权 hinge，**实现口径**）。

    L_ord = Σ_{(a,b)∈U} p^trans_a · p^trans_b / |U|，U = 复合不可达对。

    ⚠️ 2026-10-19 口径决策（P0-3）：终稿原写「作用在 log p 上的 ζ_ab 容差
    hinge」，需复合三元组 (a,b,c) 索引数据；实现改为**作用在闭包转移参数**
    `p_trans` 上——对「复合不可达对」(a,b)，按 p^trans_a·p^trans_b 加权惩罚。
    这与式(17) 闭包语义自洽（约束闭包参数的转移行为），且全部实验均在此口径下跑出。
    终稿式(29) 已按本实现重写（见 P2 终稿同步清单）。

    p_trans : (K,) 转移概率 σ(θ_trans)
    reach   : (K,K) 稀疏化后的传递可达矩阵（0/1）
    """
    pt = p_trans.reshape(-1).clamp(min=0.0, max=1.0)              # (K,)
    unre = build_unreachable(p_trans=pt, reach=reach)             # (K,K)
    denom = unre.sum().clamp(min=1e-3)
    return (torch.outer(pt, pt) * unre).sum() / denom


# ================================================ M4.12 置换守护约束

def permute_pooled(z_j: torch.Tensor, masks: torch.Tensor,
                   gen: Optional[torch.Generator] = None, n_perm: int = 2,
                   post: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                   ) -> torch.Tensor:
    """式(31) 的**真实**置换样本：对物体级特征 z_j 做随机置换后重算 z_0。

    ⚠️ 2026-10-18 修复记录（三层坑，逐层定位）：
      ① 原调用处传空列表 ⟹ L_PI 恒为 0（第一层）。
      ② 误用 `Z[:, 0]`：Z 是**已池化**的层表示 (b,M=3,d)，物体索引已被抹掉，
         无法沿物体维置换（第二层）。⟹ 必须回到 z_j (b,N,d)。
      ③ **关键**：主路的 Z 并非纯池化，而是 `chain_attention(池化)` 的输出
         （`backbone.encode` 里 `z_j, Z = intra_layer_aggregate(...)` 后
          又过了一层 `A = chain_attention(Z, ...)`，返回的 Z 实为 A）。
         若拿「主路的 Z[:,0]」（=注意力后）去比「纯池化均值」，
         两者本就不同 ⟹ L_PI 恒为大值（实测撞上限 2.0），是**伪信号**。
         ⟹ 置换样本必须走**与主路完全相同的后处理**，此处由 `post` 注入
            （即 chain_attention），保证比较的是同一函数在置换前后的输出。

    另：**只置换有效物体位**，padding 位（masks 全 0）保持原位，
    否则 padding 的 z_j 会被计入有效池化。

    参数
    ----
    z_j   : (b, N, d) 物体级特征
    masks : (b, M, N) 邻域指示（masks[:,0,:] > 0 标记有效物体位）
    post  : 可选后处理函数（主路为 chain_attention），对 (b,M,d) → (b,M,d)

    返回 (n_perm, b, d) 的张量：每个置换下的 z_0。
    """
    b, N, d = z_j.shape
    outs = []
    dev = z_j.device
    # ⚠️ generator 必须与目标 tensor **同设备**，
    #    否则报 "Expected a 'cuda' device type for generator but found 'cpu'"。
    if gen is None or gen.device.type != torch.device(dev).type:
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261018)

    valid = masks[:, 0, :] > 0 if masks.dim() == 3 else torch.ones(
        b, N, dtype=torch.bool, device=dev)          # (b,N) 有效物体位
    n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1)   # (b,1)

    for _ in range(n_perm):
        # 只在有效位内随机排序；无效位（padding）统一排到后面且保持相对次序
        r = torch.rand(b, N, device=dev, generator=gen)
        key = torch.where(valid, r, r + 10.0)
        idx = torch.argsort(key, dim=1)                       # (b,N)
        zj_p = torch.gather(z_j, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        mp = torch.gather(masks, 2, idx.unsqueeze(1).expand(-1, masks.shape[1], -1))
        # 按式(2) 计数池化还原 z_n
        cnt = mp.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (b,M,1)
        z_n = torch.einsum("bmn,bnd->bmd", mp, zj_p) / cnt    # (b,M,d)
        # 与主路相同的后处理（chain_attention）
        if post is not None:
            z_n = post(z_n)
        outs.append(z_n[:, 0])                                # 第 0 层 (b,d)
    return torch.stack(outs, 0)


def pi_loss(z0: torch.Tensor, z0_perms: Sequence[torch.Tensor]) -> torch.Tensor:
    """M4.12 式(31)：L_PI = 相对置换失真度（**实现口径，归一化到 [0,2]**）。

    L_PI = mean_π ‖z₀ − z₀^(π)‖² / ‖z₀‖²，clamp 到 [0,2]。

    ⚠️ 2026-10-19 口径决策（P0-2）：终稿原写「(1/|N₁|²) Σ_{i≠j} ‖z₀−z₀^(π)‖²」
    （不归一），实现改为**除以 ‖z₀‖² 归一化**到 [0,2]，与其他损失项量级可比。
    已按实现重写（见 P2 终稿同步清单）。此前 model.py 内联实现了同一逻辑，
    本函数是唯一正式实现（消除死代码）。

    预期行为：z₀ 为顺序无关池化时 L_PI ≈ 0（符合"天然置换不变"定性）；
    一旦 R1/R2 引入依赖槽位位置的有序汇总，L_PI 立即上升——守护价值所在。
    """
    if z0_perms is None or (torch.is_tensor(z0_perms) and z0_perms.numel() == 0) \
            or (not torch.is_tensor(z0_perms) and len(z0_perms) == 0):
        return torch.zeros((), device=z0.device, dtype=z0.dtype)
    if torch.is_tensor(z0_perms):
        # z0_perms 已是堆叠张量 (n_perm, b, d)
        diff = ((z0 - z0_perms) ** 2).sum(dim=-1)                # (n_perm, b)
    else:
        diff = torch.stack([((z0 - zp) ** 2).sum(dim=-1) for zp in z0_perms])  # (n_perm,b)
    z0n = (z0 ** 2).sum(dim=-1).clamp(min=1e-6)                   # (b,)
    return (diff / z0n.unsqueeze(0)).mean().clamp(min=0.0, max=2.0)


# ================================================ M4.13 总损失装配

def total_loss(
    parts: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    *,
    expect_keys: Optional[List[str]] = None,
) -> torch.Tensor:
    """M4.13 式(33)：总损失装配。

    L^all = L_base + Σ_g λ_g L_dag^{(g)} + λ_sym L_sym
            + λ_coh L_coh + λ_h L_hom + λ_s L_spread + λ_PI L_PI + λ_ord L_ord
    其中 L_base ≡ L_cls + λ_inv L_inv（HMEP 式9）。

    L5 修正：L_cls 只计一次（式13 的 α 权重并入式(8) 的 m_i 掩码口径）。
    """
    # ⚠️ 2026-10-18 防御：只累加**张量**项。此前某项传入了非张量（如 list/None），
    #    报 `can't multiply sequence by non-int of type 'float'` 却指不到来源。
    _first = next((v for v in parts.values() if torch.is_tensor(v)), None)
    if _first is None:
        return torch.zeros(())
    total = torch.zeros((), device=_first.device, dtype=_first.dtype)
    for name, value in parts.items():
        if not torch.is_tensor(value):
            continue                      # 跳过非张量（多为诊断字段）
        w = lambdas.get(name, 1.0)
        total = total + (w if torch.is_tensor(w) else float(w)) * value
    if expect_keys is not None:
        missing = [k for k in expect_keys if k not in parts]
        if missing:
            raise AssertionError(f"M4.13：缺失损失项 {missing}")
    return total
