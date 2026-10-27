"""S1 编码与骨架：M1.1–M1.7（模块设计文档 §2 S1）

覆盖式(1)–式(7)。含三处改造 R1/R2/R3。
批维约定（§0.2）：图像级 (b,·)；配对级 (b,P,·)。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperParams


# ---------------------------------------------------------------- M1.1 模态投影

class ModalProjection(nn.Module):
    """M1.1 式(1)：h^m_j = LayerNorm(ReLU(W_m x^m_j + b_m))，m ∈ {v,t,s}。"""

    def __init__(self, d_in: dict, d: int):
        super().__init__()
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(d_in[m], d),
                nn.ReLU(),
                nn.LayerNorm(d, eps=1e-5),
            )
            for m in ("v", "t", "s")
        })

    def forward(self, x: dict) -> dict:
        return {m: self.proj[m](x[m]) for m in ("v", "t", "s")}


# ------------------------------------------------- M1.2 层内聚合与层嵌入

def intra_layer_aggregate(
    h: dict, W_a: nn.Linear, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """M1.2 式(2)。返回 (z_j, Z)。

    z_j = LayerNorm(W_a [h^v;h^t;h^s] + b_a)      -- (b,N,d)
    z_n = (1/c_n) Σ_{j∈N_n} z_j                    -- (b,M,d)，计数池化（非固定除）
    Z   = [z_0;z_1;z_2]                            -- (b,M,d)

    重要副产品（终稿式2）：Z 只有 M=3 行 ⟹ z_0 天然对物体索引置换不变。
    """
    b, N, _ = h["v"].shape
    cat = torch.cat([h["v"], h["t"], h["s"]], dim=-1)          # (b,N,3d)
    z_j = F.layer_norm(W_a(cat), (W_a.out_features,), eps=1e-5)

    # masks: (b, M, N) —— 邻域指示；计数池化
    counts = masks.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (b,M,1)
    z_n = torch.einsum("bmn,bnd->bmd", masks, z_j) / counts     # (b,M,d)
    return z_j, z_n


# ---------------------------------------------------------------- M1.3 层级自注意力

def chain_attention(Z: torch.Tensor, W_q: nn.Linear, W_k: nn.Linear, W_v: nn.Linear) -> torch.Tensor:
    """M1.3 式(3)：Attention(Q,K,V)=Softmax(QK^T/√d_k)V，作用在 M 层级上。

    复杂度 O(N M^2) 而非 O(N^2)（M=3）。
    """
    Q, K, V = W_q(Z), W_k(Z), W_v(Z)                            # (b,M,d)
    d_k = Q.shape[-1]
    att = torch.softmax(Q @ K.transpose(-2, -1) / (d_k ** 0.5), dim=-1)
    return att @ V                                             # (b,M,d)


# ------------------------------------------------- M1.4【改造 R1】有序加权差分

class OrderedDifference(nn.Module):
    """M1.4 式(4)【改造 R1】：方向有序的层级加权差分。

    f_{r_n} = f_{u_n} − α_n ⊙ f_{s_{n−1}} − β_n ⊙ f_{o_{n−1}}

    改造点：系数 MLP 输入由 v_glo 改为「主体/客体有序槽位」：
            MLP_δ([f_s_{n−1}; f_u_n; f_o_{n−1}; v_glo])
    理由：v_glo 是全局图像特征、方向盲，而 α,β 的参数绑定已是方向绑定，
          绑定对象却无方向信息 ⟹ 绑定形同虚设。
    守恒：保留 HMEP 的 α⊙f_s、β⊙f_o 绑定，只改系数 MLP 的输入。
    """

    def __init__(self, d: int, h: int = 4):
        super().__init__()
        # 维度论证（与终稿 §3.8 参数表一致）：
        #   式(4) 的 α,β 是**标量**系数（式(25) 用 π=norm(α,β)∈R² 证实）。
        #   R1 参数增量 Δ=(4d−d)×h=3d·h=3072（d=256,h=4）⟹ MLP_δ: 4d → h → 2。
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(
        self, f_u: torch.Tensor, f_s_prev: torch.Tensor, f_o_prev: torch.Tensor, v_glo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # v_glo 为图像级 (b,d)：显式 unsqueeze 到配对级，禁止隐式广播（§0.2）
        if v_glo.dim() == 2 and f_u.dim() == 3:
            v_glo = v_glo.unsqueeze(1).expand(-1, f_u.shape[1], -1)
        inp = torch.cat([f_s_prev, f_u, f_o_prev, v_glo], dim=-1)     # (b,P,4d)
        ab = self.mlp(inp)                                            # (b,P,2)
        alpha = ab[..., 0:1]                                          # (b,P,1)
        beta = ab[..., 1:2]                                           # (b,P,1)
        f_r = f_u - alpha * f_s_prev - beta * f_o_prev                 # (b,P,d)
        return f_r, alpha, beta


# ------------------------------------------------- M1.5【改造 R2】有序隐式流

class OrderedImplicitFlow(nn.Module):
    """M1.5 式(5)【改造 R2】：方向有序隐式语义流。

    改造点：输入由 f_{u_2} 改为三槽有序拼接 [f_{s_2}; f_{u_2}; f_{o_2}]。
    理由（终稿 §1.4）：由 boxed 式，r_imp 在同一图像内对所有实体对取同一值
          ⟹ L_dag 与 L_inv 塌缩为同构、不可区分；R2 是 L_dag 可达的充要条件。
    残差式扩模：f_{u_2} 仍在首位槽（索引 0），方向信息走新增槽位。
    """

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.LayerNorm(d, eps=1e-5),
        )

    def forward(self, f_s2: torch.Tensor, f_u2: torch.Tensor, f_o2: torch.Tensor) -> torch.Tensor:
        # 槽位顺序与终稿式(5)/伪代码一致：[f_s2; f_u2; f_o2]，索引 0 = f_s2（主体）。
        # ⚠️ 旧注释曾写「索引 0 = f_u2（HMEP 原始语义）」，与实际 concat 顺序矛盾，已订正。
        # HMEP 原始语义（f_u2）保留在索引 1 槽，方向信息由索引 0/2 的主客体槽承载。
        return self.mlp(torch.cat([f_s2, f_u2, f_o2], dim=-1))

    def swap(self, f_s2: torch.Tensor, f_u2: torch.Tensor, f_o2: torch.Tensor) -> torch.Tensor:
        """swap 路（终稿 §3.7）：r_imp^π = MLP_ι([f_o2; f_u2; f_s2])。"""
        return self.mlp(torch.cat([f_o2, f_u2, f_s2], dim=-1))


# ---------------------------------------------------------------- M1.6 门控融合

class GatedFusion(nn.Module):
    """M1.6 式(6)：r = ω ⊙ r_exp + r_imp，ω = σ(W_ω v_glo + b_ω)（不改动）。

    关键语义：仅 r_exp 受门控，r_imp 直接相加 —— 这是 L_inv 不可删的理由
              （r_imp 是并列预测分支，L_inv 是其唯一训练信号）。
    """

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, r_exp: torch.Tensor, r_imp: torch.Tensor, v_glo: torch.Tensor):
        if v_glo.dim() == 2 and r_exp.dim() == 3:
            v_glo = v_glo.unsqueeze(1).expand(-1, r_exp.shape[1], -1)
        omega = torch.sigmoid(self.w(v_glo))                       # (b,P,d)
        return omega * r_exp + r_imp, omega


# ------------------------------------------------- M1.7【改造 R3】双输出态射头

class DualHead(nn.Module):
    """M1.7 式(7)【改造 R3】：输出共轭态射对 (p_ij, p_ji)。

    参数：净增 5dK = 64,000（新头 76,800 − 旧头 12,800）。
    """

    def __init__(self, d: int, K: int):
        super().__init__()
        self.fc = nn.Linear(3 * d, 2 * K)     # 76,800 = 6·d·K
        self.K = K

    def forward(self, f_s: torch.Tensor, f_u: torch.Tensor, f_o: torch.Tensor,
                logits_only: bool = False,
                res_logits: Optional[torch.Tensor] = None):
        out = self.fc(torch.cat([f_s, f_u, f_o], dim=-1))          # (b,P,2K)
        logits = out.view(*out.shape[:-1], 2, self.K)
        if res_logits is not None:
            # 2026-10-21：**MLP 式逐对旁路**残差（借鉴 MLP 基线）。
            # 该旁路只用 pair 端点的原始视觉/文本/几何特征、不经图级上下文，
            # 直接给出 (b,P,K) logits；此处广播到两路（共轭态射对的下标 0/1）。
            # 目的是补上「pair 级可置信度」这一缺失部件，使全局 P×K 排序可比。
            logits = logits + res_logits.unsqueeze(-2)
        if logits_only:
            # 2026-10-21：评测口径修复。全局 P×K 排序要求**跨 pair 可比**的分数；
            # softmax 会按 pair 归一化，破坏跨 pair 可比性（某 pair 得分普遍高时被压低），
            # 而 SOTA 基线输出的是 logits（未归一化）⟹ 直接比 softmax 概率对本文不利。
            # 训练仍走 softmax（损失需要概率），仅推理打分可切到 logits。
            return logits[..., 0, :], logits[..., 1, :]
        p_ij = torch.softmax(logits[..., 0, :], dim=-1)
        p_ji = torch.softmax(logits[..., 1, :], dim=-1)
        return p_ij, p_ji


class PrefixHead(nn.Module):
    """M4.10 式(28) 的 Head_1：Linear(d→K)。"""
    def __init__(self, d: int, K: int):
        super().__init__()
        self.fc = nn.Linear(d, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------- 配对特征容器

@dataclass
class PairFeatures:
    """一次前向中全部配对级中间量。字段名与模块设计文档的记号一一对应。"""
    f_s0: torch.Tensor   # (b,P,d) 主体物体层嵌入
    f_o0: torch.Tensor   # (b,P,d) 客体物体层嵌入
    f_u1: torch.Tensor   # (b,P,d)
    f_u2: torch.Tensor   # (b,P,d)
    f_s1: torch.Tensor
    f_o1: torch.Tensor
    f_s2: torch.Tensor
    f_o2: torch.Tensor
    f_r1: torch.Tensor   # r_exp := f_r1
    f_r2: torch.Tensor
    alpha1: torch.Tensor
    beta1: torch.Tensor
    alpha2: torch.Tensor
    beta2: torch.Tensor

    @property
    def r_exp(self) -> torch.Tensor:
        return self.f_r1

    def sample_pi(self, n: int = 1) -> torch.Tensor:
        """式(25) 的 π = norm(α_n, β_n)，用于零训练方差估计。

        α,β 为标量系数 (b,P,1) ⟹ stack 得 (b,P,2)（与终稿式25 的 R² 一致）。
        """
        pi = torch.cat([self.alpha2, self.beta2], dim=-1)          # (b,P,2)
        return pi / pi.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ---------------------------------------------------------------- 骨架组装

class Backbone(nn.Module):
    """S1 全链：式(1)→(2)→(3)→(4)→(5)→(6)。不含双输出头（由 model 决定用哪组特征）。"""

    def __init__(self, hp: HyperParams, d_in: dict):
        super().__init__()
        self.hp = hp
        d = hp.D
        self.modal = ModalProjection(d_in, d)
        self.W_a = nn.Linear(3 * d, d)
        self.W_q, self.W_k, self.W_v = (nn.Linear(d, d) for _ in range(3))
        self.union1 = nn.Linear(3 * d, d)
        self.union2 = nn.Linear(3 * d, d)
        self.diff1 = OrderedDifference(d, hp.H)
        self.diff2 = OrderedDifference(d, hp.H)
        self.implicit = OrderedImplicitFlow(d)
        self.fusion = GatedFusion(d)

    def encode(self, x: dict, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        h = self.modal(x)
        z_j, Z = intra_layer_aggregate(h, self.W_a, masks)
        A = chain_attention(Z, self.W_q, self.W_k, self.W_v)
        return z_j, A, h

    def pair_features(
        self, z_j: torch.Tensor, Z: torch.Tensor, s_idx: torch.Tensor, o_idx: torch.Tensor
    ) -> PairFeatures:
        b, N, d = z_j.shape
        z0, z1, z2 = Z[:, 0], Z[:, 1], Z[:, 2]                     # (b,d) 各层

        f_s0 = torch.gather(z_j, 1, s_idx.unsqueeze(-1).expand(-1, -1, d))
        f_o0 = torch.gather(z_j, 1, o_idx.unsqueeze(-1).expand(-1, -1, d))

        def img_level(t, P):
            return t.unsqueeze(1).expand(-1, P, -1)

        P = f_s0.shape[1]
        f_u1 = self.union1(torch.cat([f_s0, f_o0, img_level(z1, P)], dim=-1))
        f_r1, a1, b1 = self.diff1(f_u1, f_s0, f_o0, z0)

        f_s1 = f_s0 + img_level(z1, P)
        f_o1 = f_o0 + img_level(z1, P)
        f_u2 = self.union2(torch.cat([f_s1, f_o1, img_level(z2, P)], dim=-1))
        f_r2, a2, b2 = self.diff2(f_u2, f_s1, f_o1, z1)

        f_s2 = f_s1 + img_level(z2, P)
        f_o2 = f_o1 + img_level(z2, P)

        return PairFeatures(
            f_s0=f_s0, f_o0=f_o0, f_u1=f_u1, f_u2=f_u2,
            f_s1=f_s1, f_o1=f_o1, f_s2=f_s2, f_o2=f_o2,
            f_r1=f_r1, f_r2=f_r2,
            alpha1=a1, beta1=b1, alpha2=a2, beta2=b2,
        )

    def forward(
        self, pf: PairFeatures, v_glo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """式(5)(6)：返回 (r_exp, r_imp, r)。"""
        r_exp = pf.r_exp
        r_imp = self.implicit(pf.f_s2, pf.f_u2, pf.f_o2)
        r, omega = self.fusion(r_exp, r_imp, v_glo)
        return r_exp, r_imp, r
