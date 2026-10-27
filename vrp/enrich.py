"""S2 enrichment metrics and involution operators: M2.1–M2.5 (Module Design Document §2 S2)

Covering type (14)(15)(16).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================ M2.1 / M2.2 low-rank Cholesky enrichment metric

class MetricModule(nn.Module):
    """M2.1 + M2.2 Equation (14): G = A A^T + ε I_d (low-rank Cholesky enrichment metric).

    Positive certainty (one line proof): x^T G x = ||A^T x||₂² + ε||x||₂² ≥ ε||x||₂² > 0,
    So the minimum eigenvalue is at least ε - automatically positive-determined by construction, without expensive projections.

    SMW expansion reduces the inversion from O(d³) to O(r²d + 2dr²); about 1.98e5 MAC when d=256, r=16.
    Number of parameters 256×16 = 4096 (0.06% of 7.01M).

    Honesty restriction (final draft §3.2): G is learned on a dataset-by-dataset basis and is not category invariant across datasets.
    It is a ”learnable generalization of the fixed metric ℓ₂” in enriched categories, rather than a new category structure itself.
    """

    def __init__(self, d: int, r: int = 16, eps: float = 1e-4):
        super().__init__()
        self.d, self.r, self.eps = d, r, eps
        self.A = nn.Parameter(torch.randn(d, r) * 0.02)

    def metric(self) -> torch.Tensor:
        """The upper half of equation (14): G = A A^T + ε I_d."""
        d, eps = self.d, self.eps
        eye = torch.eye(d, device=self.A.device, dtype=self.A.dtype)
        return self.A @ self.A.t() + eps * eye

    def inverse(self) -> torch.Tensor:
        """The second half of equation (14): G^{-1} = ε^{-1}I − ε^{-1} A (εI_r + A^T A)^{-1} A^T (SMW).

        ⚠️ **Terposal errors found in the final draft**: The final draft is written as
              ε^{-1}I_d − ε^{-1} A (εI_r + A^T A)^{-1} ε^{-1} A^T
        Among them, there is one more ε^{-1}** (equivalent to ε^{-2}). Numerical verification (d=32, r=4, ε=1e-4):
              The final writing method has a maximum deviation of 4.23×10⁷ from the true value (G·G^{-1} deviates I up to 1.47×10⁷)
              Corrected writing: Maximum deviation from true value 6.30×10⁻¹ (relative to 1.3×10⁻⁵)
        This implementation is in revised form and marked as a final revision item in the delivery report.
        """
        A, eps, r = self.A, self.eps, self.r
        d = self.d
        eye_d = torch.eye(d, device=A.device, dtype=A.dtype)
        eye_r = torch.eye(r, device=A.device, dtype=A.dtype)
        inner = torch.linalg.inv(eps * eye_r + A.t() @ A)          # (r,r)
        return (1.0 / eps) * eye_d - (1.0 / eps) * (A @ inner @ A.t())

    def mahalanobis(self, delta: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """||Δ||²_{G^{-1}} = Δ^T G^{-1} Δ, returns a per-sample scalar (...)."""
        return torch.einsum("...i,ij,...j->...", delta, G_inv, delta)


def smw_inverse(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Functional interface of M2.2 (facilitates single equivalence testing with torch.inverse).

    Adopt a revised form (one more ε^{-1} in the final draft, see the description of MetricModule.inverse).
    """
    d, r = A.shape
    eye_d = torch.eye(d, device=A.device, dtype=A.dtype)
    eye_r = torch.eye(r, device=A.device, dtype=A.dtype)
    inner = torch.linalg.inv(eps * eye_r + A.t() @ A)
    return (1.0 / eps) * eye_d - (1.0 / eps) * (A @ inner @ A.t())


# ================================================ M2.3 / M2.4 Involution matrix and Inv operator

def precompute_M_inv(v_txt: torch.Tensor, v_inv: torch.Tensor) -> torch.Tensor:
    """M2.3 Formula (15): [M_inv]_ij = cos(v_txt_i, v_inv_j).

    Offline precomputation, freezing, not participating in gradient (50×50).
    """
    a = F.normalize(v_txt, dim=-1)
    b = F.normalize(v_inv, dim=-1)
    return a @ b.t()


@torch.no_grad()
def precompute_M_inv_frozen(v_txt: torch.Tensor, v_inv: torch.Tensor) -> torch.Tensor:
    M = precompute_M_inv(v_txt, v_inv)
    return M.detach().clone()


def Inv(p: torch.Tensor, M_inv: torch.Tensor, eps_inv: float = 0.07) -> torch.Tensor:
    """M2.4 Formula (15) boxed: Inv(p) = softmax(p^T M_inv / ε_inv).

    Honest statement (final draft §3.3): Inv is a soft implementation of the conjunction operator  on Rel, not  itself -
    Inverse predicates are often not in R, so † cannot be a permutation on R; only CLIP text spaces can be used
    Reproject the prediction to its logical inverse predicate neighborhood.
    """
    # p: (...,K)  M_inv: (K,K)  →  (...,K)
    return torch.softmax(p @ M_inv / eps_inv, dim=-1)


@torch.no_grad()
def Inv_detached(p: torch.Tensor, M_inv: torch.Tensor, eps_inv: float = 0.07) -> torch.Tensor:
    """Final draft §3.7: detach(p^dagger = Inv(p_ij)) - soft target to avoid synchronization drift."""
    return Inv(p, M_inv, eps_inv)


def build_c_dagger(M_inv: torch.Tensor, v_inv: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """c^†_k of Equation (23): The inverse predicate weighted by the top1 indicator matrix J describes the embedding.

    c^†_k = (1 / max(1, Σ_j J_kj)) Σ_j J_kj · v_inv_j
    """
    deg = J.sum(dim=-1, keepdim=True).clamp(min=1.0)               # (K,1)
    return (J @ v_inv) / deg                                      # (K,d)


def top1_indicators(M_inv: torch.Tensor) -> torch.Tensor:
    """J = top1_ind(M_inv) (final formula 23). Returns a one-hot indication of (K,K)."""
    idx = M_inv.argmax(dim=-1)
    J = torch.zeros_like(M_inv)
    J.scatter_(-1, idx.unsqueeze(-1), 1.0)
    return J


# ================================================ M2.5 Three groups of classifications and credibility weights

def converge_affinity(topk_hits: torch.Tensor, n_truth: torch.Tensor) -> torch.Tensor:
    """ρ̂(r ← r’) of Equation (16).

    topk_hits: (K, K) cumulative number of hits; n_truth: (K,) number of truth values ​​correctly recalled.
    """
    return topk_hits / n_truth.clamp(min=1.0).unsqueeze(0)


def build_groups_and_weights(
    rho: torch.Tensor,
    *,
    sym_threshold: float = 0.5,
    ambiguous_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """M2.5 Formula (16): Three group divisions and credibility weights are obtained from VG annotation statistics.

    Groupings must be statistically derived from the data, not presupposed – this in itself is a publishable empirical contribution.

    [Implementation Agreement] Trisection criterion:
      sym : max(ρ̂(r←r’), ρ̂(r’←r)) ≥ sym_threshold and the primary inverse predicate of r is itself
      non-sym : max(ρ̂) ≥ sym_threshold but primary inverse predicate ≠ itself
      ambig : max(ρ̂) < ambiguous_threshold (unreliable) or falls in between

    Return (mask_non_sym, mask_sym, mask_ambig, w), all (K,).
    """
    K = rho.shape[0]
    aff = torch.maximum(rho, rho.t())
    best = aff.argmax(dim=-1)                                       # primary inverse predicate of r
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
    """Formula (16): Instance-level weight w_i := w_{y_i} (y_i is the gold label predicate)."""
    return w[y]


def build_groups_from_counts(
    pair_counts: torch.Tensor, inverse_hist: torch.Tensor, **kw
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructed from raw counts (pair_counts: (K,) number of samples per group; inverse_hist: (K,K) accumulation of inverse predicate hits)."""
    rho = converge_affinity(inverse_hist, pair_counts)
    return build_groups_and_weights(rho, **kw)
