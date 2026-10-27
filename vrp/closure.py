"""S3 闭包：M3.1–M3.6（模块设计文档 §2 S3）

覆盖式(17)(18)(19)(20)(21)(22)。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================ M3.1 可学习闭包参数

class ClosureParams(nn.Module):
    """M3.1 式(17)：p^trans(r)=σ(θ^trans_r)；p^conv(r'|r)=softmax_{r'}(θ^conv_{r·})。

    末列对应空关系 φ：p^conv(φ|r) 表示「r 在 R 中没有逆」的概率 ——
    这正是式(15) 无法硬编码 Inv 的量化出口。

    参数仅 50 + 2550 = 2600；免去 2601 规模复合表的人工构造。
    """

    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.theta_trans = nn.Parameter(torch.zeros(K))
        self.theta_conv = nn.Parameter(torch.zeros(K, K + 1))
        # ⚠️ 2026-10-18 新增：**对角偏置**通路。
        #
        # 诊断（见记忆 2026-10-18「coh 验证」）：`p_conv = softmax(θ_conv)`
        # 在初始化时**完全均匀**（θ_conv=0），而其唯一梯度来源是式(28) 的
        # `diag(P̄²)` 加权——但当 diag 恒定时该加权等价于常数缩放 ⟹
        # 对 θ_conv 的梯度近乎为零 ⟹ **p_conv 永远无法分化**
        # ⟹ 式(28) 的 P̄ 加权语义彻底失效。
        #
        # 修法：把「对角偏置」从 `p_conv` 的 softmax 里**分离**出来，
        # 单独作为可学习参数 `theta_diag`：
        #     p_conv = softmax(θ_conv)                    # 行方向（复合可达）
        #     diag(P̄²)[k] ∝ (1 + η_d · θ_diag[k])²     # 对角偏置（层内一致性）
        # 这样 θ_diag 有**直接、可学**的梯度通路（来自 coh 的 D_JS），
        # 不再被「softmax 归一化把差异抹平」所压制。
        self.theta_diag = nn.Parameter(torch.zeros(K))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p_trans = torch.sigmoid(self.theta_trans)                  # (K,)
        p_conv = torch.softmax(self.theta_conv, dim=-1)            # (K,K+1)
        return p_trans, p_conv

    def diag_bias(self) -> torch.Tensor:
        """对角偏置（可学）：(K,)。

        ⚠️ 2026-10-18：这是让式(28) 的 `diag(P̄²)` 加权**真正生效**的关键。
        初始化为 0 ⟹ 与原实现**完全等价**（不改变已有实验口径），
        但它有独立的梯度通路 ⟹ 训练中可分化。
        """
        return 1.0 + self.theta_diag


# ================================================ M3.2 逆闭包

def converse_closure(
    E_i: torch.Tensor, E_r: torch.Tensor, E_j: torch.Tensor, p_conv: torch.Tensor, topk: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M3.2 式(18)：C_conv(E) = ∪ {(j, r', i) : (i,r,j) ∈ E, r' ∈ supp p^conv(·|r)}。

    实现要点：主客体对调 + 按 p^conv 支持集扩张。**方向不得颠倒**。
    返回 (new_i, new_r, new_j)，其中 new_i = j、new_j = i。
    """
    P = p_conv.shape[-1] - 1                                        # 去掉 φ 列
    # 取每行 topk 的支持集（排除 φ 列）
    sup = torch.topk(p_conv[:, :P], k=min(topk, P), dim=-1).indices    # (K,topk)
    n_edges = E_r.shape[0]
    ni, nr, nj = [], [], []
    for e in range(n_edges):
        r = int(E_r[e].item())
        for rp in sup[r]:
            ni.append(E_j[e]); nr.append(rp); nj.append(E_i[e])
    if not ni:
        return E_i.new_empty(0), E_r.new_empty(0), E_j.new_empty(0)
    return torch.stack(ni), torch.stack(nr), torch.stack(nj)


def converse_closure_batch(
    E: torch.Tensor, p_conv: torch.Tensor, topk: int = 1
) -> torch.Tensor:
    """式(18) 的向量化版本。E: (n,3) 列为 (i, r, j)；返回 (m,3) 的 (j, r', i)。"""
    P = p_conv.shape[-1] - 1
    k = min(topk, P)
    sup = torch.topk(p_conv[:, :P], k=k, dim=-1).indices             # (K,k)
    r = E[:, 1].long()                                              # (n,)
    rp = sup[r]                                                     # (n,k)
    i = E[:, 0]
    j = E[:, 2]
    # 主客体对调：new = (j, r', i)
    new_i = j.unsqueeze(1).expand(-1, k).reshape(-1)
    new_r = rp.reshape(-1)
    new_j = i.unsqueeze(1).expand(-1, k).reshape(-1)
    return torch.stack([new_i, new_r, new_j], dim=-1)


# ================================================ M3.3 传递闭包（STE）

def transitive_closure_ste(
    adj: torch.Tensor, p_trans: torch.Tensor, *, ste: bool = True,
    topk: int = 1,
) -> torch.Tensor:
    """M3.3 式(19)：Floyd–Warshall，加权版按 p^trans 软化；反向用直通估计。

    ⚠️ 工程代价（终稿 §3.2 明示必须写进论文）：需可微 FW，或用分段操作 +
       直通估计（STE）。本文采用后者 —— 这是相对 ECCV2020 的实现差异。

    adj: (K,K) 或 (n,n) 的邻接（软）矩阵，代表同层关系可达性；
        本实现对关系类型图做闭包，返回 (K,K) 的传递可达矩阵。
    topk: 每行保留的最强转移数（边存在性判定）。
        ⚠️ 2026-10-18：adj 是 softmax 输出的**稠密**矩阵，若用 `adj > 0` 判边
        则处处有边 ⟹ 闭包全 1 ⟹ 「复合不可达」集合恒空。故必须按 top-k 支持集
        稀疏化（与式18 converse_closure 的支持集口径一致）。
    前向按标准 FW 精确计算；反向把梯度按 p_trans 软化权重直接透传。
    """
    K = adj.shape[0]
    with torch.no_grad():
        # ⚠️ 2026-10-18 修复（根因）：原实现用 `adj > 0` 判定边存在，
        #   而 adj = softmax(θ_conv) 是**稠密正值**矩阵（每行每列都 > 0）
        #   ⟹ 传递闭包后 `reach` **处处为 1** ⟹ `unreachable = 1 − reach ≡ 0`
        #   ⟹ **L_ord 恒为 0，从未生效**。
        #
        #   正确做法：边存在性必须按**支持集**判定，与 `converse_closure`（式18）
        #   的 top-k 支持集口径一致 —— 只保留每行 top-k 个最强转移为「有边」。
        _k = int(topk) if topk else 1
        _k = max(1, min(_k, K))
        if _k < K:
            _vals, _idx = torch.topk(adj, k=_k, dim=-1)
            _thresh = _vals[:, -1:].clone()          # (K,1) 每行第 k 大的值
            _has = (adj >= _thresh).float()
            # 排除「恰好等于阈值但非 top-k」的并列项
            _rank = torch.argsort(torch.argsort(adj, dim=-1, descending=True), dim=-1)
            _has = (_rank < _k).float()
        else:
            _has = (adj > 0).float()
        reach = _has.clone()
        for k in range(K):
            reach = torch.maximum(reach, reach[:, k:k + 1] * reach[k:k + 1, :])
    if not ste:
        return reach
    # 直通：前向值取精确 FW，反向把 dL/dreach 按 p_trans 软化后透传给 adj
    soft = p_trans.unsqueeze(0).expand(K, K)
    return reach.detach() + (adj - adj.detach()) * soft


def transitive_violation_mask(
    triples: torch.Tensor, reach: torch.Tensor, epsilon_0: float
) -> torch.Tensor:
    """式(30) 的传递违反掩码来源之一：t 上的传递一致性监控。

    triples: (n,3) 的 (i, r_ab, r_bc) 联合索引示意；reach 为 (K,K)。
    [实现约定] 返回每条三元组是否「违反传递可达」的布尔（此处用 0/1 浮点）。
    """
    rab = triples[:, 1].long()
    rbc = triples[:, 2].long()
    viol = 1.0 - reach[rab, rbc]
    return (viol > (1.0 - epsilon_0)).float()


# ================================================ M3.4 闭包加权两跳邻域

def two_hop_neighborhood(
    p_conv: torch.Tensor, eta: float = 1.0, *, return_debug: bool = False
):
    """M3.4 式(21)：P̄ = (I + η P^conv)，P̄^⊗2 = P̄ P̄（真两跳）。

    修正说明（终稿 L3）：原式取 (I+ηP^conv)(I+ηP^trans)，但 P^trans 为对角阵，
    该乘积只是行缩放、不含真正两跳，会使消融 A8 失效。现改为 P̄²（真两跳）。

    本函数显式暴露 `row_scaling` 对照量，供单测断言 P̄² ≠ 任何行缩放形式。
    返回归一化后的 (K,K)；return_debug=True 时额外返回中间量。
    """
    K = p_conv.shape[0]
    Pc = p_conv[:, :K]                                              # 去掉 φ 列 → (K,K)
    eye = torch.eye(K, device=p_conv.device, dtype=p_conv.dtype)
    P_bar = eye + eta * Pc                                          # (K,K)
    P_bar_sq = P_bar @ P_bar                                        # 真两跳
    row_scaling = P_bar * P_bar.sum(dim=-1, keepdim=True)           # 仅行缩放的对照
    if return_debug:
        return P_bar_sq, P_bar, row_scaling
    return P_bar_sq


def rownorm(x: torch.Tensor) -> torch.Tensor:
    """行归一化到概率单纯形。"""
    return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ================================================ M3.5 混合学生分布

def mixed_student(
    p_ij: torch.Tensor, P_bar_sq: torch.Tensor, omega_p: float, eps: float = 1e-8
) -> torch.Tensor:
    """M3.5 式(22)：p̃ = Softmax((1−ω_p)log(p_ij+ε) + ω_p log(rownorm(p_ij P̄^⊗2)+ε))。

    风险控制：ω_p 门控 + 只作用于 L_KL、不作用于 L_cls。
    ω_p=0 时必须严格退化为 p_ij。
    """
    if omega_p == 0.0:
        return p_ij
    closed = rownorm(p_ij @ P_bar_sq)
    logits = (1.0 - omega_p) * torch.log(p_ij + eps) + omega_p * torch.log(closed + eps)
    return torch.softmax(logits, dim=-1)


# ================================================ M3.6 闭包一致性（规范形）

def closure_invariance_check(
    E: torch.Tensor,
    E_queries: Sequence[torch.Tensor],
    p_conv: torch.Tensor,
    topk: int = 1,
) -> float:
    """M3.6 式(20)：∀E'∈Q(E), C(E')=C(E)。

    统一表述（终稿 §3.2）：L_dag 是 C_conv 的软化；式(30) 的掩码是 C_trans 的一致性监控。
    本函数只对 C_conv 做硬一致性检查（可精确断言）；C_trans 因 p^trans 可学习，
    仅由 M5.4 的掩码统计报告，不作硬断言。

    返回最大逐元素偏差（0 表示完全一致）。
    """
    base = converse_closure_batch(E, p_conv, topk)
    base_sorted = base[torch.argsort(base[:, 0] * 1000 + base[:, 1])]
    worst = 0.0
    for Eq in E_queries:
        out = converse_closure_batch(Eq, p_conv, topk)
        out_sorted = out[torch.argsort(out[:, 0] * 1000 + out[:, 1])]
        if out_sorted.shape != base_sorted.shape:
            return float("inf")
        worst = max(worst, float((out_sorted.float() - base_sorted.float()).abs().max().item()))
    return worst
