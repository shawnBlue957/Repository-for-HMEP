"""S3 Closures: M3.1–M3.6 (Module Design Document §2 S3)

Covering formula (17)(18)(19)(20)(21)(22).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================ M3.1 Learnable closure parameters

class ClosureParams(nn.Module):
    """M3.1 Formula (17): p^trans(r)=σ(θ^trans_r); p^conv(r’|r)=softmax_{r’}(θ^conv_{r·}).

    The last column corresponds to the empty relation φ: p^conv(φ|r) represents the probability that ”r has no inverse in R”——
    This is exactly the quantized outlet of equation (15) that cannot be hard-coded Inv.

    The parameters are only 50 + 2550 = 2600; eliminating the need for manual construction of a 2601-scale composite table.
    """

    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.theta_trans = nn.Parameter(torch.zeros(K))
        self.theta_conv = nn.Parameter(torch.zeros(K, K + 1))
        # ⚠️ 2026-10-18 New: **Diagonal Offset** pass.
        #
        # Diagnosis (see memory 2026-10-18 ”coh verification”): `p_conv = softmax(θ_conv)`
        # During initialization, it is **completely uniform** (θ_conv=0), and its only gradient source is formula (28)
        # `diag(P̄²)` weighting - but when diag is constant this weighting is equivalent to constant scaling ⟹
        # The gradient for θ_conv is nearly zero ⟹ **p_conv can never differentiate**
        # ⟹ The P̄ weighted semantics of equation (28) is completely invalid.
        #
        # Fix: Separate the ”diagonal bias” from the softmax of `p_conv`,
        # Alone as a learnable parameter `theta_diag`:
        # p_conv = softmax(θ_conv) # Row direction (composite reachable)
        # diag(P̄²)[k] ∝ (1 + η_d · θ_diag[k])² # Diagonal bias (intra-layer consistency)
        # This way θ_diag has a direct, learnable gradient path (D_JS from coh),
        # No longer suppressed by ”softmax normalization smoothes the differences”.
        self.theta_diag = nn.Parameter(torch.zeros(K))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p_trans = torch.sigmoid(self.theta_trans)                  # (K,)
        p_conv = torch.softmax(self.theta_conv, dim=-1)            # (K,K+1)
        return p_trans, p_conv

    def diag_bias(self) -> torch.Tensor:
        """Diagonal offset (learnable): (K,).

        ⚠️ 2026-10-18: This is the key to making the `diag(P̄²)` weighting of equation (28) really take effect**.
        Initialized to 0 ⟹ Completely equivalent to the original implementation (does not change the existing experimental caliber),
        But it has independent gradient pathways ⟹ Differentiable during training.
        """
        return 1.0 + self.theta_diag


# ================================================ M3.2 Inverse closure

def converse_closure(
    E_i: torch.Tensor, E_r: torch.Tensor, E_j: torch.Tensor, p_conv: torch.Tensor, topk: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M3.2 Formula (18): C_conv(E) = ∪ {(j, r’, i) : (i,r,j) ∈ E, r’ ∈ supp p^conv(·|r)}.

    Implementation key points: swap subject and object + press p^conv to support set expansion. **Directions must not be reversed**.
    Returns (new_i, new_r, new_j) where new_i = j, new_j = i.
    """
    P = p_conv.shape[-1] - 1                                        # Remove the φ column
    # Get the support set of topk for each row (exclude φ column)
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
    """The vectorized version of Equation (18). E: (n,3) has columns (i, r, j); returns (j, r’, i) of (m,3)."""
    P = p_conv.shape[-1] - 1
    k = min(topk, P)
    sup = torch.topk(p_conv[:, :P], k=k, dim=-1).indices             # (K,k)
    r = E[:, 1].long()                                              # (n,)
    rp = sup[r]                                                     # (n,k)
    i = E[:, 0]
    j = E[:, 2]
    #Swap subject and object: new = (j, r’, i)
    new_i = j.unsqueeze(1).expand(-1, k).reshape(-1)
    new_r = rp.reshape(-1)
    new_j = i.unsqueeze(1).expand(-1, k).reshape(-1)
    return torch.stack([new_i, new_r, new_j], dim=-1)


# ================================================ M3.3 Transfer Closure (STE)

def transitive_closure_ste(
    adj: torch.Tensor, p_trans: torch.Tensor, *, ste: bool = True,
    topk: int = 1,
) -> torch.Tensor:
    """M3.3 Equation (19): Floyd–Warshall, the weighted version is softened by p^trans; the reverse is estimated by straight-through.

    ⚠️ Engineering cost (final draft §3.2 expressly states that it must be included in the paper): Differentiable FW is required, or segmentation operation +
       Straight-Through Estimation (STE). This article adopts the latter - this is the implementation difference compared to ECCV2020.

    adj: (K,K) or (n,n) adjacency (soft) matrix, representing the reachability of same-level relationships;
        This implementation closes the relationship type graph and returns the transitive reachability matrix of (K,K).
    topk: The number of strongest transitions retained in each row (edge ​​existence determination).
        ⚠️ 2026-10-18: adj is the **dense** matrix output by softmax. If `adj > 0` is used to determine the edge
        Then there are edges everywhere ⟹ The closure is all 1 ⟹ The ”composite unreachable” set is always empty. Therefore, you must press top-k support set
        Sparsification (consistent with the support set diameter of converse_closure in Equation 18).
    The forward direction is accurately calculated according to standard FW; in the reverse direction, the gradient is directly transparently transmitted according to p_trans softened weight.
    """
    K = adj.shape[0]
    with torch.no_grad():
        # ⚠️ 2026-10-18 Fix (root cause): The original implementation used `adj > 0` to determine the existence of an edge.
        # And adj = softmax(θ_conv) is a **dense positive matrix** (each row and column > 0)
        # ⟹ After passing closure `reach` **everywhere is 1** ⟹ `unreachable = 1 − reach ≡ 0`
        # ⟹ **L_ord is always 0 and never takes effect**.
        #
        # Correct approach: The existence of edges must be determined according to the **support set**, which is the same as `converse_closure` (Equation 18)
        # The top-k support set has the same caliber - only the top-k strongest transitions in each row are retained as ”edged”.
        _k = int(topk) if topk else 1
        _k = max(1, min(_k, K))
        if _k < K:
            _vals, _idx = torch.topk(adj, k=_k, dim=-1)
            _thresh = _vals[:, -1:].clone()          # (K,1) The kth largest value in each row
            _has = (adj >= _thresh).float()
            # Exclude tie items that are ”exactly equal to the threshold but not top-k”
            _rank = torch.argsort(torch.argsort(adj, dim=-1, descending=True), dim=-1)
            _has = (_rank < _k).float()
        else:
            _has = (adj > 0).float()
        reach = _has.clone()
        for k in range(K):
            reach = torch.maximum(reach, reach[:, k:k + 1] * reach[k:k + 1, :])
    if not ste:
        return reach
    # Direct pass: The forward value takes the exact FW, and in the reverse direction, dL/dreach is softened by p_trans and then transparently passed to adj
    soft = p_trans.unsqueeze(0).expand(K, K)
    return reach.detach() + (adj - adj.detach()) * soft


def transitive_violation_mask(
    triples: torch.Tensor, reach: torch.Tensor, epsilon_0: float
) -> torch.Tensor:
    """One source of transitive violation mask of Equation (30): transitive consistency monitoring on t.

    triples: (i, r_ab, r_bc) joint index representation of (n,3); reach is (K,K).
    [Implementation Convention] Returns a Boolean indicating whether each triplet ”violates transitive reachability” (0/1 floating point is used here).
    """
    rab = triples[:, 1].long()
    rbc = triples[:, 2].long()
    viol = 1.0 - reach[rab, rbc]
    return (viol > (1.0 - epsilon_0)).float()


# ================================================ M3.4 Closure weighted two-hop neighborhood

def two_hop_neighborhood(
    p_conv: torch.Tensor, eta: float = 1.0, *, return_debug: bool = False
):
    """M3.4 Formula (21): P̄ = (I + η P^conv), P̄^⊗2 = P̄ P̄ (true two hops).

    Correction instructions (final draft L3): The original formula is (I+ηP^conv)(I+ηP^trans), but P^trans is a diagonal matrix,
    This product is only row scaling and does not contain true two hops, which would render ablation A8 ineffective. Now changed to P̄² (true double jump).

    This function explicitly exposes the `row_scaling` control quantity for a single test to assert that P̄² ≠ any row scaling form.
    Returns the normalized (K,K); when return_debug=True, an additional intermediate amount is returned.
    """
    K = p_conv.shape[0]
    Pc = p_conv[:, :K]                                              # Remove φ column → (K,K)
    eye = torch.eye(K, device=p_conv.device, dtype=p_conv.dtype)
    P_bar = eye + eta * Pc                                          # (K,K)
    P_bar_sq = P_bar @ P_bar                                        # True Two Jumps
    row_scaling = P_bar * P_bar.sum(dim=-1, keepdim=True)           # Only scaled comparison
    if return_debug:
        return P_bar_sq, P_bar, row_scaling
    return P_bar_sq


def rownorm(x: torch.Tensor) -> torch.Tensor:
    """Row normalization to probability simplex."""
    return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-12)


# ================================================ M3.5 Mixed Student Distribution

def mixed_student(
    p_ij: torch.Tensor, P_bar_sq: torch.Tensor, omega_p: float, eps: float = 1e-8
) -> torch.Tensor:
    """M3.5 Equation (22): pə = Softmax((1−ω_p)log(p_ij+ε) + ω_p log(rownorm(p_ij P̄^⊗2)+ε)).

    Risk control: ω_p gate + only works on L_KL, not L_cls.
    It must strictly degenerate to p_ij when ω_p=0.
    """
    if omega_p == 0.0:
        return p_ij
    closed = rownorm(p_ij @ P_bar_sq)
    logits = (1.0 - omega_p) * torch.log(p_ij + eps) + omega_p * torch.log(closed + eps)
    return torch.softmax(logits, dim=-1)


# ================================================ M3.6 Closure consistency (canonical form)

def closure_invariance_check(
    E: torch.Tensor,
    E_queries: Sequence[torch.Tensor],
    p_conv: torch.Tensor,
    topk: int = 1,
) -> float:
    """M3.6 Formula (20): ∀E’∈Q(E), C(E’)=C(E).

    Unified expression (final draft §3.2): L_dag is the softening of C_conv; the mask of equation (30) is the consistency monitoring of C_trans.
    This function only performs a hard consistency check on C_conv (can be accurately asserted); C_trans can be learned because p^trans
    Reported only by M5.4’s mask statistics, no hard assertions are made.

    Returns the maximum element-wise deviation (0 means perfect agreement).
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
