"""S1 Coding and Skeleton: M1.1–M1.7 (Module Design Document §2 S1)

Covering formula (1)–formula (7). Contains three modifications R1/R2/R3.
Batch dimension convention (§0.2): image level (b,·); pairing level (b,P,·).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperParams


# ---------------------------------------------------------------- M1.1 Modal Projection

class ModalProjection(nn.Module):
    """M1.1 Formula (1): h^m_j = LayerNorm(ReLU(W_m x^m_j + b_m)), m ∈ {v,t,s}."""

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


#--------------------------------------------- M1.2 Intra-layer aggregation and layer embedding

def intra_layer_aggregate(
    h: dict, W_a: nn.Linear, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """M1.2 Formula (2). Return (z_j, Z).

    z_j = LayerNorm(W_a [h^v;h^t;h^s] + b_a)      -- (b,N,d)
    z_n = (1/c_n) Σ_{j∈N_n} z_j -- (b,M,d), count pooling (non-fixed division)
    Z   = [z_0;z_1;z_2]                            -- (b,M,d)

    Important by-product (final draft 2): Z only has M=3 lines ⟹ z_0 is naturally invariant to object index replacement.
    """
    b, N, _ = h["v"].shape
    cat = torch.cat([h["v"], h["t"], h["s"]], dim=-1)          # (b,N,3d)
    z_j = F.layer_norm(W_a(cat), (W_a.out_features,), eps=1e-5)

    # masks: (b, M, N) - neighborhood indication; count pooling
    counts = masks.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # (b,M,1)
    z_n = torch.einsum("bmn,bnd->bmd", masks, z_j) / counts     # (b,M,d)
    return z_j, z_n


# ---------------------------------------------------------------- M1.3 Hierarchical self-attention

def chain_attention(Z: torch.Tensor, W_q: nn.Linear, W_k: nn.Linear, W_v: nn.Linear) -> torch.Tensor:
    """M1.3 Formula (3): Attention(Q,K,V)=Softmax(QK^T/√d_k)V, acting on the M level.

    Complexity O(N M^2) instead of O(N^2) (M=3).
    """
    Q, K, V = W_q(Z), W_k(Z), W_v(Z)                            # (b,M,d)
    d_k = Q.shape[-1]
    att = torch.softmax(Q @ K.transpose(-2, -1) / (d_k ** 0.5), dim=-1)
    return att @ V                                             # (b,M,d)


# ---------------------------------------------- M1.4 [Renovation R1] Ordered Weighted Difference

class OrderedDifference(nn.Module):
    """M1.4 Formula (4) [Modification of R1]: Directionally ordered hierarchical weighted difference.

    f_{r_n} = f_{u_n} − α_n ⊙ f_{s_{n−1}} − β_n ⊙ f_{o_{n−1}}

    Modification point: The coefficient MLP input is changed from v_glo to ”subject/object ordered slot”:
            MLP_δ([f_s_{n−1}; f_u_n; f_o_{n−1}; v_glo])
    Reason: v_glo is a global image feature and direction blind, while the parameter binding of α and β is already a direction binding.
          Binding objects but no direction information ⟹ Binding is useless.
    Conservation: Keep the α⊙f_s, β⊙f_o bindings of HMEP, and only change the input of the coefficient MLP.
    """

    def __init__(self, d: int, h: int = 4):
        super().__init__()
        # Dimension argument (consistent with the final draft §3.8 parameter list):
        # α, β in Equation (4) are **scalar** coefficients (Equation (25) is confirmed by π=norm(α,β)∈R²).
        # R1 parameter increment Δ=(4d−d)×h=3d·h=3072 (d=256,h=4)⟹ MLP_δ: 4d → h → 2.
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(
        self, f_u: torch.Tensor, f_s_prev: torch.Tensor, f_o_prev: torch.Tensor, v_glo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # v_glo is image level (b,d): explicitly unsqueeze to pairing level, disable implicit broadcast (§0.2)
        if v_glo.dim() == 2 and f_u.dim() == 3:
            v_glo = v_glo.unsqueeze(1).expand(-1, f_u.shape[1], -1)
        inp = torch.cat([f_s_prev, f_u, f_o_prev, v_glo], dim=-1)     # (b,P,4d)
        ab = self.mlp(inp)                                            # (b,P,2)
        alpha = ab[..., 0:1]                                          # (b,P,1)
        beta = ab[..., 1:2]                                           # (b,P,1)
        f_r = f_u - alpha * f_s_prev - beta * f_o_prev                 # (b,P,d)
        return f_r, alpha, beta


# ----------------------------------------------- M1.5 [Renovation R2] Ordered Implicit Flow

class OrderedImplicitFlow(nn.Module):
    """M1.5 Formula (5) [Modification of R2]: directionally ordered implicit semantic flow.

    Modification point: The input is changed from f_{u_2} to three-slot ordered splicing [f_{s_2}; f_{u_2}; f_{o_2}].
    Reason (final draft §1.4): By boxed formula, r_imp takes the same value for all entity pairs in the same image
          ⟹ L_dag and L_inv collapse to isomorphism and indistinguishable; R2 is the necessary and sufficient condition for L_dag to be reachable.
    Residual mode expansion: f_{u_2} is still in the first slot (index 0), and the direction information goes to the new slot.
    """

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.LayerNorm(d, eps=1e-5),
        )

    def forward(self, f_s2: torch.Tensor, f_u2: torch.Tensor, f_o2: torch.Tensor) -> torch.Tensor:
        # The slot sequence is consistent with the final formula (5)/pseudocode: [f_s2; f_u2; f_o2], index 0 = f_s2 (main body).
        # ⚠️ The old comment once said ”index 0 = f_u2 (HMEP original semantics)”, which is inconsistent with the actual concat order and has been corrected.
        # HMEP original semantics (f_u2) is retained in the index 1 slot, and the direction information is carried by the subject-object slot at index 0/2.
        return self.mlp(torch.cat([f_s2, f_u2, f_o2], dim=-1))

    def swap(self, f_s2: torch.Tensor, f_u2: torch.Tensor, f_o2: torch.Tensor) -> torch.Tensor:
        """swap path (final draft §3.7): r_imp^π = MLP_ι([f_o2; f_u2; f_s2])."""
        return self.mlp(torch.cat([f_o2, f_u2, f_s2], dim=-1))


# ---------------------------------------------------------------- M1.6 Gated Fusion

class GatedFusion(nn.Module):
    """M1.6 Formula (6): r = ω ⊙ r_exp + r_imp, ω = σ(W_ω v_glo + b_ω) (unchanged).

    Key semantics: only r_exp is gated, r_imp is added directly - this is the reason why L_inv cannot be deleted
              (r_imp is the parallel prediction branch and L_inv is its only training signal).
    """

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, r_exp: torch.Tensor, r_imp: torch.Tensor, v_glo: torch.Tensor):
        if v_glo.dim() == 2 and r_exp.dim() == 3:
            v_glo = v_glo.unsqueeze(1).expand(-1, r_exp.shape[1], -1)
        omega = torch.sigmoid(self.w(v_glo))                       # (b,P,d)
        return omega * r_exp + r_imp, omega


#--------------------------------------------- M1.7 [Renovation R3] Dual output morphic head

class DualHead(nn.Module):
    """M1.7 Formula (7) [Modification of R3]: Output the conjugate morphism pair (p_ij, p_ji).

    Parameters: Net increase 5dK = 64,000 (new head 76,800 − old head 12,800).
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
            # 2026-10-21: **MLP-style pairwise bypass** residuals (referring to the MLP baseline).
            # This bypass only uses the original visual/text/geometric features of the pair endpoints without going through the graph-level context.
            # Give (b,P,K) logits directly; broadcast to two paths here (subscript 0/1 for conjugate morphism pairs).
            # The purpose is to fill in the missing component of ”pair-level confidence” to make the global P×K sorting comparable.
            logits = logits + res_logits.unsqueeze(-2)
        if logits_only:
            # 2026-10-21: Review caliber fixed. Global P×K sorting requires scores that are comparable across pairs;
            # Softmax will be normalized by pair, destroying comparability across pairs (when the score of a certain pair is generally high, it will be lowered),
            # The SOTA baseline outputs logits (unnormalized)⟹ which is directly disadvantageous to this article compared to softmax probability.
            # Training still uses softmax (loss requires probability), and only inference scoring can be switched to logits.
            return logits[..., 0, :], logits[..., 1, :]
        p_ij = torch.softmax(logits[..., 0, :], dim=-1)
        p_ji = torch.softmax(logits[..., 1, :], dim=-1)
        return p_ij, p_ji


class PrefixHead(nn.Module):
    """M4.10 Head_1 of equation (28): Linear(d→K)."""
    def __init__(self, d: int, K: int):
        super().__init__()
        self.fc = nn.Linear(d, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# --------------------------------------------------------------------- Matching feature container

@dataclass
class PairFeatures:
    """All paired-level intermediate quantities in one forward pass. The field names correspond to the notations of the module design document one-to-one."""
    f_s0: torch.Tensor   # (b,P,d) Main object layer embedding
    f_o0: torch.Tensor   # (b,P,d) Object layer embedding
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
        """π = norm(α_n, β_n) of Equation (25) is used for zero training variance estimation.

        α,β are scalar coefficients (b,P,1) ⟹ stack to get (b,P,2) (consistent with R² of the final formula 25).
        """
        pi = torch.cat([self.alpha2, self.beta2], dim=-1)          # (b,P,2)
        return pi / pi.norm(dim=-1, keepdim=True).clamp(min=1e-8)


#------------------------------------------------------------- Skeleton assembly

class Backbone(nn.Module):
    """S1 full chain: formula (1)→(2)→(3)→(4)→(5)→(6). Does not include dual output heads (the model determines which set of features to use)."""

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
        z0, z1, z2 = Z[:, 0], Z[:, 1], Z[:, 2]                     # (b,d) each layer

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
        """Formula (5)(6): Return (r_exp, r_imp, r)."""
        r_exp = pf.r_exp
        r_imp = self.implicit(pf.f_s2, pf.f_u2, pf.f_o2)
        r, omega = self.fusion(r_exp, r_imp, v_glo)
        return r_exp, r_imp, r
