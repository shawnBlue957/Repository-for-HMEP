"""S5 Diagnostics: M5.1–M5.6 (Module Design Document §2 S5)

Covering table (2), formula (30), and formula (34).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .guards import hinge_trigger_rate  # noqa: F401 (M5.6 reuse)


# ================================================ M5.1 Directional Violation Rate DVR/PIV1

def dvr(r_ij: torch.Tensor, r_ji: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M5.1 Table (2)(b): DVR = mean_i ||r_ij − r_ji||₂ / (||r_ij||₂ + ||r_ji||₂ + ε).

    ⚠️ The DVR comparison value before modification must be reported, otherwise this indicator is meaningless (final table (2)).
    """
    num = (r_ij - r_ji).norm(dim=-1)
    den = r_ij.norm(dim=-1) + r_ji.norm(dim=-1) + eps
    return (num / den).mean()


def piv1(f_u1_so: torch.Tensor, f_u1_os: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M5.1 Table (2)(b): PIV₁.

    Convenience fact (final table (2)): By the **commutativity** of join, f_{u_1}^{(s,o)} ≡ f_{u_1}^{(o,s)}
    **The identity holds**, so PIV₁ ≡ 0 is an **identity rather than an experimental observation**.
    """
    num = (f_u1_so - f_u1_os).norm(dim=-1)
    den = f_u1_so.norm(dim=-1) + f_u1_os.norm(dim=-1) + eps
    return (num / den).mean()


# ================================================ M5.2 Reciprocity Violation Rate IVR

def ivr(
    p_ji_swap: torch.Tensor, Inv_p_ij: torch.Tensor, w_i: torch.Tensor
) -> torch.Tensor:
    """M5.2 Table (2): IVR = (1/|N|) Σ_i w_i D_JS(p_ji^π, Inv(p_ij)).

    Isomorphic to the distribution layer term of M4.5 (same quantity, mutually verifiable).
    """
    from .losses import js_divergence
    return (w_i * js_divergence(p_ji_swap, Inv_p_ij)).mean()


# ================================================ M5.3 Enriched residual convergence

def res_g(delta: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
    """M5.3 Table (2): res_g(t) = (1/|N|) Σ_i ||Δ_i(t)||²_{G^{-1}}, recorded epoch by epoch."""
    return torch.einsum("...i,ij,...j->...", delta, G_inv, delta).mean()


# ============================================== M5.4 ε_assoc with pass-through mask violation

def build_epsilon_assoc(
    triples: torch.Tensor, K_plus: int = 51
) -> torch.Tensor:
    """M5.4 Formula (30): 51³ lookup table of ε_assoc.

    Key property: If σ° is derived from **direct co-occurrence statistics** of VG annotation triples,
    Then it is the combinatorial law violation rate of the data set, a pure data attribute that has nothing to do with training.
    Inspection scale 51³ = 132,651 table lookups (note not 2501³), one-time offline, available hash acceleration.

    Falsifiable (final draft): If ε_assoc has no correlation with mR@50, then the benefit B does not hold.

    triples: (r_ab, r_bc, r_ac) relation id of (n,3) (K_plus-1 can be used to represent empty relation φ).
    [Implementation Convention] ε_assoc[rab,rbc,rac] := 1 − P(rac | rab, rbc) (violation rate of compound law).
    Return (K_plus, K_plus, K_plus).
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
    # No observation disposition 0 (violation of inaction)
    eps_t = torch.where(tot > 0, eps_t, torch.zeros_like(eps_t))
    return eps_t


def violation_mask(epsilon_assoc: torch.Tensor, triples: torch.Tensor, epsilon_0: float) -> torch.Tensor:
    """M5.4 Equation (30) mask: credibility weighting of m_i = 1[ε_assoc ≤ ε₀].

    ε₀ = 85th percentile (data driven, to be calibrated).
    """
    rab, rbc, rac = triples[:, 0].long(), triples[:, 1].long(), triples[:, 2].long()
    e = epsilon_assoc[rab, rbc, rac]
    return (e <= epsilon_0).float()


def epsilon0_from_percentile(all_values: torch.Tensor, pct: float = 85.0) -> float:
    """ε₀ = 85th percentile (final draft §3.7 offline precomputation)."""
    if all_values.numel() == 0:
        return 0.0
    return float(torch.quantile(all_values.flatten().float(), pct / 100.0).item())


# ============================================== M5.5 CatCom Caliber Yoneda Two Index

def jacobian_wrt_logp(scalar: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """ψ = ∂(scalar)/∂ log p, returns isomorphic to logp."""
    return torch.autograd.grad(scalar, logp, retain_graph=True, create_graph=False)[0]


def yoneda_metrics(
    r_rep: torch.Tensor, logp: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """M5.5 Formula (34): Homogeneity and Compositionality of CatCom caliber.

    H_r         = E_{(s,o),(s',o')} ||ψ_r(s,o) − ψ_r(s',o')||₂
    C_comp      = E ||ψ_{r∘r'} − (ψ_r ⊕ ψ_{r'})||₂
    where ψ_r = ∂r/∂log p_r.

    Hierarchies must be in standard definition (final draft §3.6): CatCom’s Homogeneity is an attribute morphism for different primitive objects.
    Transformation effect consistency**; Compositionality is the decomposability of representations at the object level. This function combines both
    **Migrating to the Jacobi sequence of relational morphisms** is a **migration definition and is not a copy** of the original definition.

    ⚠️Do not mix with H(r) (decoupling coefficient intra-group consistency) of equation (25) - they are three different quantities.

    [Implementation Convention] ψ_r(s,o) takes the scalar (sample-wise) of ∂(Σ_d r_rep[n,d])/∂ log p[n, y_n].
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
    # Compositionality: Approximated by deviations of component-wise sums of ψ ([Implementation Convention])
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
    """A summary of diagnostics for an epoch (final draft §3.7 record item)."""
    up, low = hinge_trigger_rate(V_r, theta_max, theta_min)
    return {
        "DVR": float(dvr(r_ij, r_ji).item()),
        "res_g": float(res.item()),
        "V_global": float(V_global.item()),
        "H_mean": float(H_r.mean().item()),
        "hinge_upper_rate": up,
        "hinge_lower_rate": low,
    }
