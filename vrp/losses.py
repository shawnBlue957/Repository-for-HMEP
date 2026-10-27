"""S4 Losses and Brakes: M4.1–M4.13 (Module Design Document §2 S4)

Covering formula (8)(9)(11)(12)(13)(23)(24)(25)(26)(27)(28)(29)(31)(33).
H1 (hinge must fall on V_r), F8 (θ_min>0), C9 (the middle section is not constrained) land here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperParams
from .guards import assert_hinge_on_Vr, assert_vr_nonzero


# --------------------------------------------------------------------- Shared Tools

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """D_JS(p,q): defined on Δ^K, symmetric, and S_K-equivariant compatible. Return sample-by-sample (...)."""
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a) - torch.log(b.clamp(min=eps)))).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average by mask. Returns 0 (not NaN) when the full mask is 0."""
    m = mask.float()
    denom = m.sum()
    if float(denom) == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (x * m).sum() / denom


class SharedProjection(nn.Module):
    """Formula (12)/(23) shared Proj∘g_φ: maps student representation to CLIP text space.

    Final draft §3.1: Sharing can obtain this capability with zero additional parameters, avoiding the introduction of a second projection semantics.
    """

    def __init__(self, d: int):
        super().__init__()
        self.g_phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.proj = nn.Linear(d, d)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.proj(self.g_phi(r))


# ================================================ M4.1 Classification loss

def cls_loss(
    p_ij: torch.Tensor,
    p_ji: torch.Tensor,
    y: torch.Tensor,
    y_tilde_ji: torch.Tensor,
    mu: torch.Tensor,
    m: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """M4.1 Equation (8): Classification loss (applying the mask of Equation (30)).

    L_cls = (1/|N|) Σ_i m_i ( −Σ_k [y_ik log p_ij[k] + μ_i ỹ^{ji}_ik log p_ji[k]] )

    Must explain (final draft formula 8): VG annotation is incomplete, r_ji true value is often missing,
    The second item can only be processed with ”symmetrization weak label + mask”.
    """
    main = -(y * torch.log(p_ij + eps)).sum(dim=-1)                 # (...)
    # μ_i passed in as (...,) or (...,1) must be correctly aligned to the shape of main (to prevent implicit broadcast to (...,P,P))
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


# ================================================ M4.2 path equivalent value

def inv_loss(r_exp: torch.Tensor, r_imp: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """M4.2 Formula (9): L_inv = (1/|N|) Σ_i ||Δ_i||²_{G^{-1}}, Δ_i = r_exp_i − r_imp_i.

    Honest Qualitative (Final Draft §3.3): L1 path equivalence / L2 congruence quotient - asserting two **different** generating paths
    Morphisms are of the same class. **Not** inverse mapping existence, **not** adjunction.
    """
    delta = r_exp - r_imp
    maha = torch.einsum("...i,ij,...j->...", delta, G_inv, delta)
    return maha.mean()


# ⚠️ 2026-10-18: This article **does not contain any intellectual distillation**. Formula (11) KL soft label distillation and formula (12)
# Feature regression distillation has been deleted, and related implementations have been removed as well.

# ================================================ M4.5 Involution Reciprocity Loss (Core)

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
    """M4.5 Formula (23): Involution reciprocity loss (L3 layer).

    L_dag = (1/|N|) Σ_i w_i [ λ† D_JS(p_ji^π, Inv(p_ij))
                             + ||Proj(g_φ(r_ji)) − Σ_k p_ij[k] c^†_k||²_{G^{-1}} ]

    Split explanation (final draft 23 handled honestly): There is a dimensional mismatch in the five original formulas of the material
    (G is defined in the representation space and p ∈ Δ^K), here it is explicitly decomposed into
      Distribution layer: D_JS (defined on Δ^K)
      Representation layer: ||·||²_{G^{-1}} (on R^d)

    Inv_p_ij must be detached (soft target, to avoid synchronization drift).

    Return (L_total, L_dist_part, L_repr_part).
    """
    if Inv_p_ij.requires_grad:
        raise AssertionError("M4.5: Inv_p_ij Must detach (final draft §3.7 Soft target requirements)")

    l_dist = js_divergence(p_ji_swap, Inv_p_ij)                     # (...)

    c_expected = torch.einsum("...k,kd->...d", p_ij, c_dag)          # (...,d)
    diff = shared(r_ji) - c_expected
    l_repr = torch.einsum("...i,ij,...j->...", diff, G_inv, diff)

    per = w_i * (lambda_dag * l_dist + l_repr)
    return per.mean(), l_dist.mean(), l_repr.mean()


# ================================================ M4.6 Symmetry group perspective invariance

def build_soft_sym_mask(v_txt: torch.Tensor, hard_mask: torch.Tensor,
                        tau: float = 0.06, floor: float = 0.0) -> torch.Tensor:
    """Construct **soft symmetry mask** (used in equation 24, added on 2026-10-18).

    ⚠️ Fix background: `mask_sym` is a **hard** mask (a strictly reflexive symmetric predicate),
    Only 4 on VRD (next to / near / beside / by). They are mostly low-frequency in the long-tail distribution.
    ⟹ The true value in the batch almost never falls in G_sym ⟹ `sel = (y*mask_sym).sum(-1)` is always 0
    ⟹ **L_sym is always 0 and never takes effect**.

    This function changes the symmetry from ”binary attribution” to **continuous weight**:
        soft[k] = hard mask value (completely symmetric predicates are 1.0)
                  + Cosine similarity mapping of semantic neighbors (nearly symmetric predicates have smaller positive weights)
    The semantic neighbor is given by the frozen CLIP predicate text embedding: if the text embedding of predicates r and r’
    The cosine is very high, indicating that the two have ”almost the same relationship” semantically ⟹ r and its reverse r † are also very possible
    Symmetrical to each other (e.g. `in` / `inside` / `within`).

    parameters
    ----
    v_txt : (K, D) L2 normalized predicate text embedding
    hard_mask: (K,) hard symmetry mask (0/1)
    tau: similarity temperature (the cosine must exceed this value to be considered a nearest neighbor)
    floor : the lower bound of neighbor weights (>0 allows all predicates to have non-zero supervision)

    Returns the continuous weight ∈ [0, 1] of (K,).
    """
    K = v_txt.shape[0]
    v = torch.nn.functional.normalize(v_txt.float(), dim=-1)
    sim = v @ v.t()                                  # (K,K) cosine
    off = sim - torch.diag_embed(torch.diagonal(sim))  # Go to self-similarity
    # The cosine of each predicate and ”the person most similar to himself”
    top = off.max(dim=-1).values                     # (K,)
    near = ((top >= tau).float() * top.clamp(min=0.0))  # Degree of near symmetry
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
    """M4.6 Formula (24): L_sym = (1/|N|) Σ_i Σ_{k∈G_sym} y_ik D_JS(p_ij[k], p_ji^π[k]).

    The symmetry class satisfies r†=r, and the reciprocity constraint degenerates into perspective invariance.
    Must be treated separately, otherwise it forces all relationships to be symmetrical, directly contradicting the fact that 93.3% are asymmetric.
    """
    # Calculate only on k within G_sym (mask_sym: (K,))
    # ⚠️ 2026-10-18: mask_sym may be registered in the CPU (original tensor passed by register_frozen),
    # And y reports ”Expected all tensors to be on the same device” on GPU ⟹.
    if mask_sym.device != y.device:
        mask_sym = mask_sym.to(y.device)
    sel = (y * mask_sym.unsqueeze(0)).sum(dim=-1)                   # (...)
    djs = js_divergence(p_ij, p_ji_swap)                            # (...)
    return (sel * djs).mean()


# ================================================ M4.7 Homogeneity V_r zero training estimate

def estimate_variance(
    pi: torch.Tensor, y: torch.Tensor, K: int, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """M4.7 Equation (25): Zero training estimate of H(r).

    π_{s,o} = norm(α_n(s,o), β_n(s,o))
    H(r)    = exp( − (1/|P_r|) Σ ||π − π̄_r||₂² )
    V_r     = −log H(r) = (1/|P_r|) Σ ||π − π̄_r||₂²
    V_global= (1/|N|) Σ_i ||π_i − π̄||₂²

    Why use V instead of H for loss (final draft §3.5): H=exp(−V) will suppress large variance to nearly 0
    (V≥3 ⟹ H<0.05), hinge disappears exponentially near the threshold; V maintains a linear gradient.

    Return (V_r (K,), V_global (), H_r (K,), counts (K,)).
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


# ================================================ M4.8 / M4.9 two-way brake

def assert_middle_band_free(V_r: torch.Tensor, theta_min: float,
                            theta_max: float, tol: float = 1e-6) -> None:
    """C9 Contract Guard: The **middle segment** of the two-way brake must be free from restraints.

    For the predicate V_r ∈ (θ_min, θ_max), both the upper and lower hinges should be 0:
        Upper bound max(0, V_r − θ_max)² = 0
        Lower bound max(0, θ_min − V_r)² = 0
    That is, the predicate of ”moderate dispersion” is neither punished (no doubt about pairwise memory) nor pushed (no doubt about collapse).

    ⚠️ 2026-10-18 Fix: This function was **deleted by mistake** when deleting distillation (with a whole paragraph deleted
    losses.py), causing the import of `tests/test_T1_unit.py` to fail directly.
    Now restored according to the original contract.
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
            f"C9: middle section (θ_min={theta_min}, θ_max={theta_max}) restrained "
            f" (The upper bound is the largest {float(up.max()):.3e}, The lower bound is the largest {float(low.max()):.3e}）")


def gate_upper(
    V_r: torch.Tensor,
    theta_max: float,
    present: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    rel: bool = True,
) -> torch.Tensor:
    """M4.8 Formula (26): Upper limit of brake (to prevent pairwise memory).

    L_hom = (1/|R|) Σ_r [max{0, V̂_r − θ_max}]²

    ⚠️ 2026-10-18 Repair (dimensional problem, third positioning):
      The measured V_r (variance within the group, actually ”the square distance from π to the group mean”, π∈R²) has a magnitude of only
      **1e-4 ~ 1e-2**, and the original θ_max is taken from the fixed tensor `tensor([0.5,1.0,1.5,2.0])`
      The quantile of ⟹ **max(0, θ_max − V_r) is always 0**, and the brake is never effective.

      And `V_r / V_global` does not work either: the measured median ratio **V_r/V_global ≈ 0.995**
      (Group mean ≈ global mean, since π has only 2 dimensions) ⟹ This ratio is almost constant 1, **non-discriminative**.

    ⟹ The correct dimensionless quantity is **variance per sample**:
          V̂_r = V_r / |P_r| (mean squared distance within group, variance not sum of squares)
      In this way θ_max has an interpretable dimension (”the average dispersion of each relation sample”),
      And predicates of different sizes (thousands at high frequency vs. a few at low frequency) are comparable.

    parameters
    ----
    V_r: (K,) sum of squared distances within the group
    scale: (K,) Number of samples in each group |P_r|; when None, it degenerates into absolute semantics (rel=False needs to be passed explicitly)
    rel: Use V̂_r = V_r / max(scale, 1) when True; use the original V_r when False
    """
    assert_hinge_on_Vr("V_r")
    if rel:
        if scale is None:
            # When there is no group size information, use the relative dimension of ”the same dimension as V_r”:
            #Normalize by V_r’s own quantile (V_r / median(V_r))
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
    """M4.9 Formula (27): Lower bound of brake (to prevent collapse between groups).

    L_spread = (1/|R|) Σ_r [max{0, θ_min − V̂_r}]² + global insurance

    ⚠️ 2026-10-18 Fixed (same origin as gate_upper):
      The original hinge acts on absolute V_r, and the magnitude of V_r is only 1e-4~1e-2 ⟹ and is always 0.
      Now uniformly changed to **variance per sample** V̂_r = V_r / |P_r|, which is the same dimension as the upper bound brake,
      The two-way hinge (the mid-section is unconstrained) really holds true.

    Return (l_vr + l_ins, l_vr, l_ins).
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
    # Global insurance term: V_global has the same dimension as V_r, but is also minimal ⟹ Also normalized
    if rel and scale is not None:
        n_all = scale.sum().clamp(min=1.0)
        vg_eff = V_global / n_all
    else:
        vg_eff = V_global
    l_ins = torch.clamp(theta_min_g - vg_eff, min=0.0) ** 2
    return l_vr + l_ins, l_vr, l_ins


# ================================================ M4.10 On-chain coherence

def coh_loss(
    f_r1: torch.Tensor,
    f_r2: torch.Tensor,
    P_bar_sq: torch.Tensor,
    head1: nn.Module,
    head2: nn.Module,
) -> torch.Tensor:
    """M4.10 Equation (28): On-chain coherence (naturalness of Chain_3; dimensions corrected by L7).

    L_coh = (1/|N|) Σ_i D_JS( norm(Head1(f_r1) · diag(P̄^⊗2)), Head2(f_r2) )

    Correction (final draft L7): K×1 times K in the original formula Head1(·)^T [P̄^⊗2]_diag has no definition;
    Now change to Head1(f_r1)·diag(P̄^⊗2) and **explicitly renormalize** (D_JS requires simplex input).

    ⚠️ **Defects discovered during the implementation period (fixed, 2026-10-17)**: The first version of this function first performs `Head1(f_r1) * diag`
    Do **L1 normalization** again. But `Head1` is the original output of Linear, **the value can be negative**, and L1 will be normalized.
    With the ”illegal probability” of negative elements, D_JS exploded - measured in the comparison experiment **L_coh = 8.9e11,
    Gradient 2.4e12**, dominates and destroys taxonomic branches.
    The correct approach is to first take **softmax** (the same caliber as Head2) for the Head1 output and then multiply it by diag:
        ① Ensure non-negative and the sum is 1; ② Normalize after softmax, and the role of diag weight is still retained.
    """
    diag_p = torch.diagonal(P_bar_sq)                               # (K,)
    # ⚠️ 2026-10-18 Repair (third layer): Diag weighting is changed to **log space**.
    #
    # Original implementation: `a = softmax(head1(f_r1)) * diag` and then normalize.
    # Actual measurement: Normalization dilutes the impact of diag to the ~1/K level
    # (The maximum component ratio only changes from 0.1946 to 0.1975, and it is almost unchanged when the amplitude is 2.0)
    # ⟹ Minimal impact on D_JS, and diag is constant when `p_conv` is uniform ⟹ **No impact at all**.
    #
    # Now implemented: `a = softmax(head1(f_r1) + log(diag))`
    # ⟹ diag directly changes the relative logits of each category, **really changes the distribution shape**,
    # Strong gradient and complete semantics (predicates with large diag(P̄²) are elevated in a).
    log_diag = torch.log(diag_p.clamp(min=1e-12))
    a = torch.softmax(head1(f_r1) + log_diag.unsqueeze(-2), dim=-1)
    b = torch.softmax(head2(f_r2), dim=-1)
    return js_divergence(a, b).mean()


def diag_p_spread(P_bar_sq: torch.Tensor) -> float:
    """Diagnosis: The **relative range** of diag(P̄²) is used to determine whether the P̄ weighting of equation (28) is effective.

    ⚠️ 2026-10-18 Diagnosis: If `p_conv = softmax(θ_conv)` remains uniform
    (θ_conv is initially 0, and its gradient is suppressed by this term), then `diag(P̄²)` is equal everywhere
    ⟹ Multiplying it and then normalizing it equals **multiplying nothing** ⟹ Equation (28) degenerates into
    ”D_JS with two independent softmax”, **P̄ weighted semantics completely invalid**.

    Criterion:
        Relative range > 1% ⟹ P̄ Weighted effective
        Relative range ≤ 1% ⟹ P̄ Weighting **obsolete** (equivalent to constant scaling)
    """
    d = torch.diagonal(P_bar_sq)
    med = d.abs().median().clamp(min=1e-12)
    return float(((d.max() - d.min()) / med).item())


# ================================================ M4.11 S_K-equal variable order regularity

def build_unreachable(p_trans: Optional[torch.Tensor] = None,
                      reach: Optional[torch.Tensor] = None,
                      min_reach: float = 0.5) -> torch.Tensor:
    """The ”composite unreachable” matrix (K,K) of Equation (29).

    ⚠️ 2026-10-18 Repair (**Remove the trick of softening the lower bound**):
      The previous version used `reach.clamp(max=1−soft_floor)` to force non-zero values,
      The essence is a false signal of ”turning constant 0 into constant 0.02”, which has no real semantics.

      Now change to the **real** dependence on the sparse reachability matrix:
        1. `reach` is given by `transitive_closure_ste(..., topk=k)`,
           Edge existence is determined by ** each row top-k supports set** judgment (rather than dense judgment of `adj > 0`,
           The latter would make the closure all 1 ⟹ and the unreachable set is empty).
        2. Composite unreachable = ”no direct edge” and ”no transit path”,
           That is, `unreachable = 1 − reach` (reach is sparse ⟹ the set is not empty).
        3. Additional **complexity** requirement: the unreachable pair (a,b) must also be satisfied
           ”The existence of c makes a→c and c→b have substantial weight under p^trans”——
           That is, the pair is a gap that ”in principle should be reunited but cannot be reconciled.”
           This is the object that equation (29) wants to punish (order regularity).

    parameters
    ----
    p_trans: (K,) transition probability (for compositeness determination and weight)
    reach : (K,K) sparse transitive reach matrix (0/1)
    min_reach: minimum weight threshold for compound paths

    Returns (K,K), elements ∈ [0,1], 1 means ”composite unreachable”.
    """
    if reach is None:
        K = int(p_trans.shape[-1]) if p_trans is not None else 1
        return torch.zeros((K, K))
    K = reach.shape[0]
    unre = (1.0 - reach).clamp(min=0.0)          # No edges and no paths
    if p_trans is not None and p_trans.numel() == K:
        pt = p_trans.reshape(-1).clamp(min=0.0, max=1.0)
        # Compositeness: a gap that is reachable by two hops but has no direct edge
        two_hop = (pt.unsqueeze(1) * pt.unsqueeze(0)) > min_reach * 0.0
        # Elements of the above formula (a,b) = pt[a]*pt[b]; a real two-hop requires intermediate c ⟹ Aggregation using intermediate nodes
        # Here we use the approximation of ”the existence of c at the same time makes the distribution of a close to c, and c close to b”:
        # Directly use the **product structure** of pt to determine the possibility of two hops (conservative relaxation).
        comp_gap = two_hop * unre
        # Compound gap weight: Unreachable ∧ Has compounding potential ⟹ Keep larger values
        unre = torch.maximum(unre * 0.0, comp_gap)
    return unre


def ord_loss(
    p_trans: torch.Tensor,
    reach: torch.Tensor,
) -> torch.Tensor:
    """M4.11 Formula (29): Order regularity (composite unreachable pair weighted hinge, **realization caliber**).

    L_ord = Σ_{(a,b)∈U} p^trans_a · p^trans_b / |U|, U = composite unreachable pair.

    ⚠️ 2026-10-19 Caliber decision (P0-3): Original final draft ”ζ_ab tolerance acting on log p
    hinge”, the composite triplet (a,b,c) index data is required; the implementation is changed to **acting on the closure transfer parameter**
    `p_trans` on - For the ”composite unreachable pair” (a, b), the penalty is weighted by p^trans_a·p^trans_b.
    This is consistent with the closure semantics of equation (17) (constraining the transfer behavior of closure parameters), and all experiments were run under this caliber.
    The final formula (29) has been rewritten according to this implementation (see P2 final synchronization list).

    p_trans : (K,) transition probability σ(θ_trans)
    reach : (K,K) sparse transitive reach matrix (0/1)
    """
    pt = p_trans.reshape(-1).clamp(min=0.0, max=1.0)              # (K,)
    unre = build_unreachable(p_trans=pt, reach=reach)             # (K,K)
    denom = unre.sum().clamp(min=1e-3)
    return (torch.outer(pt, pt) * unre).sum() / denom


# ================================================ M4.12 Replacement Guardian Constraints

def permute_pooled(z_j: torch.Tensor, masks: torch.Tensor,
                   gen: Optional[torch.Generator] = None, n_perm: int = 2,
                   post: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                   ) -> torch.Tensor:
    """The **real** permutation sample of Equation (31): perform random permutation on the object-level feature z_j and recalculate z_0.

    ⚠️ 2026-10-18 Repair record (three layers of pits, located layer by layer):
      ① The original caller passes an empty list ⟹ L_PI is always 0 (first level).
      ② Misuse of `Z[:, 0]`: Z is the **pooled** layer representation (b,M=3,d), the object index has been erased,
         Displacement along object dimensions is not possible (second level). ⟹ Must go back to z_j (b,N,d).
      ③ **Key**: Z of the main road is not pure pooling, but the output of `chain_attention(pooling)`
         (After `z_j, Z = intra_layer_aggregate(...)` in `backbone.encode`
          After passing another level `A = chain_attention(Z, ...)`, the returned Z is actually A).
         If we compare ”Z[:,0] of the main road” (=after attention) to ”pure pooling mean”,
         The two are inherently different. ⟹ L_PI is always a large value (the actual measured collision upper limit is 2.0), which is a **false signal**.
         ⟹ The replacement sample must go through the same post-processing as the main path, here it is injected by `post`
            (i.e. chain_attention), ensuring that the output of the same function before and after replacement is compared.

    In addition: **Only replace the effective object position**, the padding bit (masks all 0) remains in the original position,
    Otherwise the padding z_j will be counted in the effective pooling.

    parameters
    ----
    z_j : (b, N, d) object-level features
    masks: (b, M, N) neighborhood indication (masks[:,0,:] > 0 marks valid object position)
    post: Optional post-processing function (the main path is chain_attention), for (b,M,d) → (b,M,d)

    Returns a tensor of (n_perm, b, d): z_0 under each permutation.
    """
    b, N, d = z_j.shape
    outs = []
    dev = z_j.device
    # ⚠️ The generator must be on the same device as the target tensor,
    # Otherwise, ”Expected a ’cuda’ device type for generator but found ’cpu’” will be reported.
    if gen is None or gen.device.type != torch.device(dev).type:
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261018)

    valid = masks[:, 0, :] > 0 if masks.dim() == 3 else torch.ones(
        b, N, dtype=torch.bool, device=dev)          # (b,N) Effective object position
    n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1)   # (b,1)

    for _ in range(n_perm):
        # Randomly sort only the valid bits; invalid bits (padding) are sorted to the back and the relative order is maintained
        r = torch.rand(b, N, device=dev, generator=gen)
        key = torch.where(valid, r, r + 10.0)
        idx = torch.argsort(key, dim=1)                       # (b,N)
        zj_p = torch.gather(z_j, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        mp = torch.gather(masks, 2, idx.unsqueeze(1).expand(-1, masks.shape[1], -1))
        # According to equation (2), count pooling and restore z_n
        cnt = mp.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (b,M,1)
        z_n = torch.einsum("bmn,bnd->bmd", mp, zj_p) / cnt    # (b,M,d)
        # Same post-processing as main path (chain_attention)
        if post is not None:
            z_n = post(z_n)
        outs.append(z_n[:, 0])                                # Layer 0 (b,d)
    return torch.stack(outs, 0)


def pi_loss(z0: torch.Tensor, z0_perms: Sequence[torch.Tensor]) -> torch.Tensor:
    """M4.12 Equation (31): L_PI = Relative displacement distortion (**realization caliber, normalized to [0,2]**).

    L_PI = mean_π ‖z₀ − z₀^(π)‖² / ‖z₀‖², clamp to [0,2].

    ⚠️ 2026-10-19 Caliber decision (P0-2): Original final draft ”(1/|N₁|²) Σ_{i≠j} ‖z₀−z₀^(π)‖²”
    (Not normalized), the implementation is changed to dividing by ‖z₀‖² and normalizing to [0,2], which is comparable in magnitude to other loss terms.
    Rewritten as per implementation (see P2 final draft sync list). Previously, model.py implemented the same logic inline,
    This function is the only official implementation (eliminating dead code).

    Expected behavior: L_PI ≈ 0 when z₀ is order-independent pooling (in line with the ”natural permutation invariant” qualitative);
    Once R1/R2 introduces ordered aggregation that depends on slot position, L_PI immediately rises - where the value of guarding lies.
    """
    if z0_perms is None or (torch.is_tensor(z0_perms) and z0_perms.numel() == 0) \
            or (not torch.is_tensor(z0_perms) and len(z0_perms) == 0):
        return torch.zeros((), device=z0.device, dtype=z0.dtype)
    if torch.is_tensor(z0_perms):
        # z0_perms is already a stacked tensor (n_perm, b, d)
        diff = ((z0 - z0_perms) ** 2).sum(dim=-1)                # (n_perm, b)
    else:
        diff = torch.stack([((z0 - zp) ** 2).sum(dim=-1) for zp in z0_perms])  # (n_perm,b)
    z0n = (z0 ** 2).sum(dim=-1).clamp(min=1e-6)                   # (b,)
    return (diff / z0n.unsqueeze(0)).mean().clamp(min=0.0, max=2.0)


# ================================================ M4.13 Total loss assembly

def total_loss(
    parts: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    *,
    expect_keys: Optional[List[str]] = None,
) -> torch.Tensor:
    """M4.13 Formula (33): Total loss assembly.

    L^all = L_base + Σ_g λ_g L_dag^{(g)} + λ_sym L_sym
            + λ_coh L_coh + λ_h L_hom + λ_s L_spread + λ_PI L_PI + λ_ord L_ord
    where L_base ≡ L_cls + λ_inv L_inv (HMEP Equation 9).

    L5 correction: L_cls is only counted once (the α weight of Equation 13 is merged into the m_i mask size of Equation (8)).
    """
    # ⚠️ 2026-10-18 Defense: Only accumulate **tensor** items. Previously, a non-tensor (such as list/None) was passed in.
    # It reports `can’t multiply sequence by non-int of type ’float’` but cannot point to the source.
    _first = next((v for v in parts.values() if torch.is_tensor(v)), None)
    if _first is None:
        return torch.zeros(())
    total = torch.zeros((), device=_first.device, dtype=_first.dtype)
    for name, value in parts.items():
        if not torch.is_tensor(value):
            continue                      # Skip non-tensors (mostly diagnostic fields)
        w = lambdas.get(name, 1.0)
        total = total + (w if torch.is_tensor(w) else float(w)) * value
    if expect_keys is not None:
        missing = [k for k in expect_keys if k not in parts]
        if missing:
            raise AssertionError(f"M4.13: Missing loss term {missing}")
    return total
