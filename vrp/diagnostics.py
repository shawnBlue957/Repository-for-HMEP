"""S5 诊断：M5.1–M5.6（模块设计文档 §2 S5）

覆盖表(2)、式(30)、式(34)。
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .guards import hinge_trigger_rate  # noqa: F401  （M5.6 复用）


# ================================================ M5.1 方向性违反率 DVR / PIV1

def dvr(r_ij: torch.Tensor, r_ji: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M5.1 表(2)(b)：DVR = mean_i ||r_ij − r_ji||₂ / (||r_ij||₂ + ||r_ji||₂ + ε)。

    ⚠️ 必须报告改造前的 DVR 对照值，否则本指标无意义（终稿表(2)）。
    """
    num = (r_ij - r_ji).norm(dim=-1)
    den = r_ij.norm(dim=-1) + r_ji.norm(dim=-1) + eps
    return (num / den).mean()


def piv1(f_u1_so: torch.Tensor, f_u1_os: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M5.1 表(2)(b)：PIV₁。

    便利事实（终稿表(2)）：由 join 的**交换性**，f_{u_1}^{(s,o)} ≡ f_{u_1}^{(o,s)}
    **恒等成立**，故 PIV₁ ≡ 0 是**恒等式而非实验观测**。
    """
    num = (f_u1_so - f_u1_os).norm(dim=-1)
    den = f_u1_so.norm(dim=-1) + f_u1_os.norm(dim=-1) + eps
    return (num / den).mean()


# ================================================ M5.2 互反性违反率 IVR

def ivr(
    p_ji_swap: torch.Tensor, Inv_p_ij: torch.Tensor, w_i: torch.Tensor
) -> torch.Tensor:
    """M5.2 表(2)：IVR = (1/|N|) Σ_i w_i D_JS(p_ji^π, Inv(p_ij))。

    与 M4.5 的分布层项同构（同一量，可互验）。
    """
    from .losses import js_divergence
    return (w_i * js_divergence(p_ji_swap, Inv_p_ij)).mean()


# ================================================ M5.3 富化残差收敛

def res_g(delta: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """M5.3 表(2)：res_g(t) = (1/|N|) Σ_i ||Δ_i(t)||²_{G^{-1}}，逐 epoch 记录。"""
    return torch.einsum("...i,ij,...j->...", delta, G_inv, delta).mean()


# ================================================ M5.4 ε_assoc 与传递违反掩码

def build_epsilon_assoc(
    triples: torch.Tensor, K_plus: int = 51
) -> torch.Tensor:
    """M5.4 式(30)：ε_assoc 的 51³ 查表。

    关键性质：若 σ° 由 VG 标注三元组的**直接共现统计**导出，
    则它是**数据集的组合律违反率**，一个**与训练无关的纯数据属性**。
    检验规模 51³ = 132,651 次查表（注意不是 2501³），一次性离线、可用哈希加速。

    可证伪（终稿）：若 ε_assoc 与 mR@50 无相关，则收益 B 不成立。

    triples: (n,3) 的 (r_ab, r_bc, r_ac) 关系 id（可用 K_plus-1 表示空关系 φ）。
    [实现约定] ε_assoc[rab,rbc,rac] := 1 − P(rac | rab, rbc)（复合律的违反率）。
    返回 (K_plus, K_plus, K_plus)。
    """
    eps_t = torch.zeros((K_plus, K_plus, K_plus), dtype=torch.float32)
    if triples.numel() == 0:
        return eps_t
    rab, rbc, rac = triples[:, 0].long(), triples[:, 1].long(), triples[:, 2].long()
    flat = rab * K_plus * K_plus + rbc * K_plus + rac
    cnt = torch.bincount(flat, minlength=K_plus ** 3).reshape(K_plus, K_plus, K_plus).float()
    tot = cnt.sum(dim=-1, keepdim=True)                             # (K,K,1)
    prob = cnt / tot.clamp(min=1e-12)
    eps_t = 1.0 - prob.max(dim=-1, keepdim=True).values.expand_as(prob)
    # 无观测处置 0（不作为违反）
    eps_t = torch.where(tot > 0, eps_t, torch.zeros_like(eps_t))
    return eps_t


def violation_mask(epsilon_assoc: torch.Tensor, triples: torch.Tensor, epsilon_0: float) -> torch.Tensor:
    """M5.4 式(30) 掩码：m_i = 1[ε_assoc ≤ ε₀] 的可信度加权。

    ε₀ = 第 85 百分位（数据驱动，待标定）。
    """
    rab, rbc, rac = triples[:, 0].long(), triples[:, 1].long(), triples[:, 2].long()
    e = epsilon_assoc[rab, rbc, rac]
    return (e <= epsilon_0).float()


def epsilon0_from_percentile(all_values: torch.Tensor, pct: float = 85.0) -> float:
    """ε₀ = 第 85 百分位（终稿 §3.7 离线预计算）。"""
    if all_values.numel() == 0:
        return 0.0
    return float(torch.quantile(all_values.flatten().float(), pct / 100.0).item())


# ================================================ M5.5 CatCom 口径 Yoneda 两指标

def jacobian_wrt_logp(scalar: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """ψ = ∂(scalar)/∂ log p，返回与 logp 同形。"""
    return torch.autograd.grad(scalar, logp, retain_graph=True, create_graph=False)[0]


def yoneda_metrics(
    r_rep: torch.Tensor, logp: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """M5.5 式(34)：CatCom 口径的 Homogeneity 与 Compositionality。

    H_r         = E_{(s,o),(s',o')} ||ψ_r(s,o) − ψ_r(s',o')||₂
    C_comp      = E ||ψ_{r∘r'} − (ψ_r ⊕ ψ_{r'})||₂
    其中 ψ_r = ∂r/∂log p_r。

    层级必须标清（终稿 §3.6）：CatCom 的 Homogeneity 是**属性态射对不同基元对象的
    变换效应一致性**；Compositionality 是**对象层面**的表征可分解性。本函数把二者
    **迁移到关系态射的雅可比列**上，属**迁移定义，非原定义照搬**。

    ⚠️ 不得与式(25) 的 H(r)（解耦系数组内一致性）混用 —— 是三个不同的量。

    [实现约定] ψ_r(s,o) 取 ∂(Σ_d r_rep[n,d])/∂ log p[n, y_n] 的标量（逐样本）。
    """
    n = r_rep.shape[0]
    psi = torch.zeros(n, device=r_rep.device, dtype=r_rep.dtype)
    for i in range(n):
        g = jacobian_wrt_logp(r_rep[i].sum(), logp)                 # (n,K)
        psi[i] = g[i, y[i].long()]
    H_list = []
    K = logp.shape[-1]
    for r in range(K):
        sel = y.long() == r
        if int(sel.sum().item()) >= 2:
            v = psi[sel]
            H_list.append((v.unsqueeze(0) - v.unsqueeze(1)).abs().mean())
    H_r = torch.stack(H_list).mean() if H_list else torch.zeros((), device=r_rep.device)
    # Compositionality：以 ψ 的逐分量和的偏差近似（[实现约定]）
    C_comp = (psi - psi.mean()).abs().mean()
    return H_r, C_comp


def collect_epoch_diagnostics(
    *,
    pi: torch.Tensor,
    y: torch.Tensor,
    V_r: torch.Tensor,
    V_global: torch.Tensor,
    H_r: torch.Tensor,
    r_ij: torch.Tensor,
    r_ji: torch.Tensor,
    res: torch.Tensor,
    theta_max: float,
    theta_min: float,
) -> Dict[str, float]:
    """一次 epoch 的诊断汇总（终稿 §3.7 记录项）。"""
    up, low = hinge_trigger_rate(V_r, theta_max, theta_min)
    return {
        "DVR": float(dvr(r_ij, r_ji).item()),
        "res_g": float(res.item()),
        "V_global": float(V_global.item()),
        "H_mean": float(H_r.mean().item()),
        "hinge_upper_rate": up,
        "hinge_lower_rate": low,
    }
