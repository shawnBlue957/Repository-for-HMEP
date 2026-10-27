"""S2 富化度量与对合算子：M2.1–M2.5（模块设计文档 §2 S2）

覆盖式(14)(15)(16)。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================ M2.1 / M2.2 低秩 Cholesky 富化度量

class MetricModule(nn.Module):
    """M2.1 + M2.2 式(14)：G = A A^T + ε I_d（低秩 Cholesky 富化度量）。

    正定性（一行证明）：x^T G x = ||A^T x||₂² + ε||x||₂² ≥ ε||x||₂² > 0，
    故最小特征值至少为 ε —— 由构造自动正定，无需昂贵投影。

    SMW 展开把求逆从 O(d³) 降到 O(r²d + 2dr²)；d=256,r=16 时约 1.98e5 MAC。
    参数量 256×16 = 4096（占 7.01M 的 0.06%）。

    诚实限定（终稿 §3.2）：G 是逐数据集学习的，不具跨数据集的范畴不变性。
    它是富化范畴中「固定度量 ℓ₂ 的可学推广」，而非新的范畴结构本身。
    """

    def __init__(self, d: int, r: int = 16, eps: float = 1e-4):
        super().__init__()
        self.d, self.r, self.eps = d, r, eps
        self.A = nn.Parameter(torch.randn(d, r) * 0.02)

    def metric(self) -> torch.Tensor:
        """式(14) 上半：G = A A^T + ε I_d。"""
        d, eps = self.d, self.eps
        eye = torch.eye(d, device=self.A.device, dtype=self.A.dtype)
        return self.A @ self.A.t() + eps * eye

    def inverse(self) -> torch.Tensor:
        """式(14) 下半：G^{-1} = ε^{-1}I − ε^{-1} A (εI_r + A^T A)^{-1} A^T（SMW）。

        ⚠️ **实测发现的终稿笔误**：终稿写为
              ε^{-1}I_d − ε^{-1} A (εI_r + A^T A)^{-1} ε^{-1} A^T
        其中**多出一个 ε^{-1}**（等价于 ε^{-2}）。数值验证（d=32,r=4,ε=1e-4）：
              终稿写法 与真值最大偏差 4.23×10⁷（G·G^{-1} 偏离 I 达 1.47×10⁷）
              修正写法 与真值最大偏差 6.30×10⁻¹（相对 1.3×10⁻⁵）
        本实现采用修正形式，并在交付报告中标注为终稿待修项。
        """
        A, eps, r = self.A, self.eps, self.r
        d = self.d
        eye_d = torch.eye(d, device=A.device, dtype=A.dtype)
        eye_r = torch.eye(r, device=A.device, dtype=A.dtype)
        inner = torch.linalg.inv(eps * eye_r + A.t() @ A)          # (r,r)
        return (1.0 / eps) * eye_d - (1.0 / eps) * (A @ inner @ A.t())

    def mahalanobis(self, delta: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """||Δ||²_{G^{-1}} = Δ^T G^{-1} Δ，返回逐样本标量 (...)。"""
        return torch.einsum("...i,ij,...j->...", delta, G_inv, delta)


def smw_inverse(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """M2.2 的函数式接口（便于与 torch.inverse 做等价性单测）。

    采用修正形式（终稿多一个 ε^{-1}，见 MetricModule.inverse 的说明）。
    """
    d, r = A.shape
    eye_d = torch.eye(d, device=A.device, dtype=A.dtype)
    eye_r = torch.eye(r, device=A.device, dtype=A.dtype)
    inner = torch.linalg.inv(eps * eye_r + A.t() @ A)
    return (1.0 / eps) * eye_d - (1.0 / eps) * (A @ inner @ A.t())


# ================================================ M2.3 / M2.4 对合矩阵与 Inv 算子

def precompute_M_inv(v_txt: torch.Tensor, v_inv: torch.Tensor) -> torch.Tensor:
    """M2.3 式(15)：[M_inv]_ij = cos(v_txt_i, v_inv_j)。

    离线预计算、冻结、不参与梯度（50×50）。
    """
    a = F.normalize(v_txt, dim=-1)
    b = F.normalize(v_inv, dim=-1)
    return a @ b.t()


@torch.no_grad()
def precompute_M_inv_frozen(v_txt: torch.Tensor, v_inv: torch.Tensor) -> torch.Tensor:
    M = precompute_M_inv(v_txt, v_inv)
    return M.detach().clone()


def Inv(p: torch.Tensor, M_inv: torch.Tensor, eps_inv: float = 0.07) -> torch.Tensor:
    """M2.4 式(15) boxed：Inv(p) = softmax(p^T M_inv / ε_inv)。

    诚实表述（终稿 §3.3）：Inv 是 Rel 上对合算子 † 的软实现，不是 † 本身 ——
    逆谓词常不在 R 内，故 † 不能是 R 上的置换；只能用 CLIP 文本空间
    把预测重投影到其逻辑逆谓词邻域。
    """
    # p: (...,K)  M_inv: (K,K)  →  (...,K)
    return torch.softmax(p @ M_inv / eps_inv, dim=-1)


@torch.no_grad()
def Inv_detached(p: torch.Tensor, M_inv: torch.Tensor, eps_inv: float = 0.07) -> torch.Tensor:
    """终稿 §3.7：detach(p^dagger = Inv(p_ij)) —— 软目标，避免同步漂移。"""
    return Inv(p, M_inv, eps_inv)


def build_c_dagger(M_inv: torch.Tensor, v_inv: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """式(23) 的 c^†_k：由 top1 指示矩阵 J 加权的逆谓词描述嵌入。

    c^†_k = (1 / max(1, Σ_j J_kj)) Σ_j J_kj · v_inv_j
    """
    deg = J.sum(dim=-1, keepdim=True).clamp(min=1.0)               # (K,1)
    return (J @ v_inv) / deg                                      # (K,d)


def top1_indicators(M_inv: torch.Tensor) -> torch.Tensor:
    """J = top1_ind(M_inv)（终稿式23）。返回 (K,K) 的 one-hot 指示。"""
    idx = M_inv.argmax(dim=-1)
    J = torch.zeros_like(M_inv)
    J.scatter_(-1, idx.unsqueeze(-1), 1.0)
    return J


# ================================================ M2.5 三组分类与可信度权重

def converge_affinity(topk_hits: torch.Tensor, n_truth: torch.Tensor) -> torch.Tensor:
    """式(16) 的 ρ̂(r ← r')。

    topk_hits: (K, K) 累计命中数；n_truth: (K,) 被正确召回的真值数。
    """
    return topk_hits / n_truth.clamp(min=1.0).unsqueeze(0)


def build_groups_and_weights(
    rho: torch.Tensor,
    *,
    sym_threshold: float = 0.5,
    ambiguous_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """M2.5 式(16)：由 VG 标注统计得三组划分与可信度权重。

    分组必须由数据统计得出，不是预先假定 —— 这本身是可发表的实证贡献。

    [实现约定] 三等分判据：
      sym      : max(ρ̂(r←r'), ρ̂(r'←r)) ≥ sym_threshold 且 r 的主要逆谓词为自身
      non-sym  : max(ρ̂) ≥ sym_threshold 但主要逆谓词 ≠ 自身
      ambig    : max(ρ̂) < ambiguous_threshold（不可靠）或落在两者之间

    返回 (mask_non_sym, mask_sym, mask_ambig, w)，均为 (K,)。
    """
    K = rho.shape[0]
    aff = torch.maximum(rho, rho.t())
    best = aff.argmax(dim=-1)                                       # r 的主要逆谓词
    self_loop = torch.arange(K, device=rho.device)
    peak = aff.max(dim=-1).values

    is_self = best == self_loop
    mask_sym = is_self & (peak >= sym_threshold)
    mask_non_sym = (~is_self) & (peak >= sym_threshold)
    mask_ambig = ~(mask_sym | mask_non_sym)

    # w_r = max{ρ̂(r←r), ρ̂(r'←r)} / (1 + 1[r ∈ G_ambig])
    w = peak / (1.0 + mask_ambig.float())
    w = w.clamp(0.0, 1.0)
    return mask_non_sym.float(), mask_sym.float(), mask_ambig.float(), w


def instance_weights(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """式(16)：实例级权重 w_i := w_{y_i}（y_i 为金标谓词）。"""
    return w[y]


def build_groups_from_counts(
    pair_counts: torch.Tensor, inverse_hist: torch.Tensor, **kw
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从原始计数构建（pair_counts: (K,) 每组样本数；inverse_hist: (K,K) 逆谓词命中累计）。"""
    rho = converge_affinity(inverse_hist, pair_counts)
    return build_groups_and_weights(rho, **kw)
