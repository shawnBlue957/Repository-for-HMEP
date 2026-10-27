"""S6 training orchestration and inference: M6.2 / M6.5 (Module Design Document §2 S6)

Covers final draft §3.7. Three-way forward (main path/closure path/swap path) + inference stripping.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import chain_attention, Backbone, DualHead, PrefixHead
from .closure import ClosureParams, mixed_student, transitive_closure_ste, two_hop_neighborhood
from .config import HyperParams, all_lambda_dict
from .enrich import (
    Inv_detached,
    MetricModule,
    build_c_dagger,
    top1_indicators,
)
from .losses import (
    permute_pooled,
    SharedProjection,
    cls_loss,
    coh_loss,
    dag_loss,
    gate_lower,
    gate_upper,
    inv_loss,
    ord_loss,
    pi_loss,
    sym_loss,
    total_loss,
)


# 2026-10-21: Evaluation and scoring caliber switch.
# Global P×K sorting requires scores to be comparable across pairs; softmax normalization by pair will destroy comparability.
# And SOTA baselines (GPS-Net/VCTree/SGTR/...) all output unnormalized logits.
#
# 2026-10-21 The default is changed to True (logits):
# Actual measurement of ”same weight, only cutting scoring method” on VG 2000/300 (A arm 10ep)
# top1 0.6161→0.6165 (unchanged), R@50 0.3404→0.3898 (+0.0494, relative +14.5%).
# top1 remains unchanged but R@K increases, proving that this is a **cross-pair comparability** issue rather than a classification ability issue.
# The training loss still uses probability (softmax), and only the reasoning score is switched, so it does not affect any training semantics.
# `--score-softmax` can fall back to the old caliber, which is used to reproduce historical numbers and compare calibers.
SCORE_LOGITS = True

# 2026-10-21: **MLP-style pair-by-pair bypass** switch (drawing ideas from the strongest baseline MLPPairClassifier).
# Motivation: The difference between VG’s top1 in this article and MLP is only 1.65pt, but the difference in R@50 is 17.2pt ⟹ The difference is all
# ”Are the pairs comparable?” The model in this article does not have any pair-level scoring (forward_inference only
# Return the predicate distribution). The MLP baseline outperforms all structural models, indicating that pairwise interaction is more stable under this setting.
# Method: Add a pairwise MLP branch without graph-level context, and output (b,P,K) logits as residuals
# Added to the logits of the two streams; the structural branches and their constraints (inv/dag/coh/...) are completely preserved.
# The parameter cost is about 1.33 M (8.3 M total magnitude), and there is no additional forward direction for reasoning (sharing input with the main path).
USE_PAIR_RES = False
PAIR_RES_HID = 512

# Pair existence supervision (relationness): negative pair sampling + BCE.
# PAIR_REL switch (requires USE_PAIR_RES to be True at the same time)
# PAIR_REL_LAMBDA training loss weight
# PAIR_REL_BETA pair-level additive calibration strength during inference: z[k] += β·log σ(rel_logit)
# (per-pair constant ⟹ top1 is identical, R@K is movable, it is the missing item of R@K)
PAIR_REL = False
PAIR_REL_LAMBDA = 0.5
PAIR_REL_BETA = 1.0
# Negative sample ratio: In addition to GT pairs, each image samples ratio×|GT| evenly from **non-GT ordered pairs**.
#Compared with the method of ”disrupting o_idx within the same batch”, uniform sampling is consistent with the candidate distribution during evaluation
# (predcls candidates = all ordered pairs in the graph), and the ratio is fixed to 1:ratio.
PAIR_REL_NEG_RATIO = 3


class PairResidualHead(nn.Module):
    """MLP-style pair-by-pair bypass: only eats pair endpoint features, **does not inject any graph-level context**.

    Inputs are organized according to the isomorphic caliber of the MLP baseline:
        [v_s, v_o, t_s, t_o, g_s, g_o, g_s−g_o, |g_s−g_o|]
    Output (b,P,K) logits as residuals added to the two-way logits of `DualHead`.
    """

    def __init__(self, d_v: int = 512, d_t: int = 512, n_geo: int = 8,
                 K: int = 100, hid: int = 512, p: float = 0.5):
        super().__init__()
        self.d_v, self.d_t, self.n_geo = d_v, d_t, n_geo
        d_in = d_v * 2 + d_t * 2 + n_geo * 4
        self.net = nn.Sequential(
            nn.Linear(d_in, hid), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hid, hid), nn.ReLU(inplace=True), nn.Dropout(p),
        )
        self.out_k = nn.Linear(hid, K)
        # 2026-10-21: **pair existence** (relationness) header.
        # Motivation (AUC actual measurement): The model `max_k log p` in this article has an AUC of only 0.4935 for ”Is there a relationship between this pair”
        # ⟹ Completely nonjudgmental. The root cause is that training only performs K-type CE on GT pairs, and **never seen negative pairs**,
        # This dimension does not have any supervision. The global P×K ordering of PredCls has to answer it first (R@K backward agent).
        self.out_rel = nn.Linear(hid, 1)

    def forward(self, v_s, v_o, t_s, t_o, g_s, g_o, return_rel: bool = False):
        dg = g_s - g_o
        h = self.net(torch.cat([v_s, v_o, t_s, t_o, g_s, g_o, dg, dg.abs()], dim=-1))
        k = self.out_k(h)
        if not return_rel:
            return k
        return k, self.out_rel(h).squeeze(-1)


class VRPCatModel(nn.Module):
    """A visual relation prediction model for category modeling (three-way forward).

    Component ↔ Module Design Document:
      backbone     M1.1–M1.6
      dual_head    M1.7 (R3)
      metric       M2.1/M2.2
      closure      M3.1
      shared M4.4/M4.5 shared projection
      head1/head2  M4.10
    """

    def __init__(self, hp: HyperParams, d_in: dict):
        super().__init__()
        self.hp = hp
        d, K = hp.D, hp.K
        self.backbone = Backbone(hp, d_in)
        self.dual_head = DualHead(d, K)
        # MLP pair-by-pair bypass (see module-level USE_PAIR_RES for switch). d_in is given by the backbone’s input specification.
        self.pair_res = None
        if USE_PAIR_RES:
            self.pair_res = PairResidualHead(
                d_v=d_in.get("v", d), d_t=d_in.get("t", d),
                n_geo=d_in.get("s", 8), K=K, hid=PAIR_RES_HID)
        self.head1 = PrefixHead(d, K)
        self.head2 = PrefixHead(d, K)
        self.metric = MetricModule(d, hp.R_LOWRANK, hp.EPS_G)
        self.closure = ClosureParams(K)
        self.shared = SharedProjection(d)
        # 2026-10-18: Replacement generator of equation (31) (rebuilt by device in losses, only seeds are saved here)
        self._perm_seed = 20261018
        # 2026-10-18: **soft symmetry mask** (used in equation 24).
        # Hard mask_sym only contains 4 symmetric predicates ⟹ Long-tail predicates have no symmetry supervision ⟹ L_sym is always 0.
        # soft_mask_sym[k] = ”near symmetry” of the predicate (semantic similarity maps to [0,1]).
        self.soft_mask_sym = None
        # 2026-10-18: Full training set predicate presence table (Formula 25/26 used for brakes).
        # Pre-calibrated by `register_fixed` from the training set frequency to avoid the absence of the batch causing the brake to close.
        self.predicate_presence = None

    # --------------------------------------------------------------- M2.3 Freezing volume
    def register_frozen(
        self,
        M_inv: torch.Tensor,
        J: torch.Tensor,
        c_dag: torch.Tensor,
        w: torch.Tensor,
        mask_sym: torch.Tensor,
        mask_non_sym: torch.Tensor,
        mask_ambig: torch.Tensor,
    ) -> None:
        """Register a fixed amount for offline precomputation (final draft §3.7: M_inv/q/group division initial value is a fixed amount)."""
        self.register_buffer("M_inv", M_inv)
        self.register_buffer("J", J)
        self.register_buffer("c_dag", c_dag)
        self.register_buffer("w", w)
        self.register_buffer("mask_sym", mask_sym)
        self.register_buffer("mask_non_sym", mask_non_sym)
        self.register_buffer("mask_ambig", mask_ambig)

    def register_soft_sym(self, soft: torch.Tensor) -> None:
        """Register **soft symmetry mask** (used in equation 24, added on 2026-10-18).

        ⚠️ Fix background: original `mask_sym` was a **hard** mask, marking only strictly reflexive symmetric predicates
        (VRD only 4: next to / near / beside / by). These predicates are in the long-tail distribution
        Most of them belong to low frequency ⟹ The true value in the batch almost does not fall in G_sym ⟹ `sel = (y*mask_sym).sum(-1)`
        Always 0 ⟹ **L_sym is always 0 and never takes effect**.

        This interface accepts a **continuous weight** of (K,) (0=asymmetric, 1=completely symmetric),
        Constructed by ”semantic similarity of predicate text embedding”: the closer the predicate is to its own semantics,
        The more likely it is to be symmetric ⟹ the greater the weight.
        """
        self.soft_mask_sym = soft.float()
        # ⚠️ 2026-10-18: The device must be the same as the model, otherwise the forward times will be
        #    "Expected all tensors to be on the same device"
        if self.soft_mask_sym.device != next(self.parameters()).device:
            self.soft_mask_sym = self.soft_mask_sym.to(next(self.parameters()).device)

    def register_presence(self, presence: torch.Tensor) -> None:
        """Register **full training set** predicate presence table (Formula 25/26 used for brakes, added on 2026-10-18).

        ⚠️ Fix background: original implementation uses `present = (counts in batch > 0)`,
        And the VRD training set 47.1% of the predicate frequency < batch_size (=32),
        The brakes causing these predicates are closed in any batch ⟹ L_hom / L_spread is always 0.
        After switching to training set level presence, the brakes take effect on all predicates that appear at least once.
        """
        self.predicate_presence = presence.float()

    # --------------------------------------------------------------- M6.2 three-way forward
    def forward(
        self,
        x: dict,
        masks: torch.Tensor,
        s_idx: torch.Tensor,
        o_idx: torch.Tensor,
        y_label: Optional[torch.Tensor] = None,
        y_tilde_ji: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        m_i: Optional[torch.Tensor] = None,
        lambda_inv: float = 0.2,
        omega_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """y_label is the integer label (b,P); one-hot is derived internally.

        ⚠️ Interface ambiguity protection: cls_loss requires one-hot, estimate_variance and w table lookup require integer labels,
        Therefore, the two must be passed in/derived separately to avoid the same tensor being reused by both semantics.
        """
        hp = self.hp
        omega_p = hp.OMEGA_P if omega_p is None else omega_p
        K = hp.K
        dtype = x["v"].dtype
        y = None
        if y_label is not None:
            y = torch.nn.functional.one_hot(y_label.long(), K).to(dtype)   # (b,P,K) one-hot

        #---------------- Main Road----------------
        z_j, Z, _ = self.backbone.encode(x, masks)
        v_glo = Z[:, 1]                                             # (b,d) image level
        pf = self.backbone.pair_features(z_j, Z, s_idx, o_idx)
        r_exp, r_imp, r = self.backbone.forward(pf, v_glo)
        # ⚠️ 2026-10-19 (SC Metrics): Recalculated gating ω = σ(W_ω v_glo + b_ω) for diagnostics.
        # backbone.forward discards ω internally; here it is extracted with the same fusion.w, and r is not recalculated.
        # ω∈(0,1) is the dimension-wise retention rate of explicit flow r_exp, and the fusion formula r = ω⊙r_exp + r_imp.
        _vg = v_glo
        if _vg.dim() == 2 and r_exp.dim() == 3:
            _vg = _vg.unsqueeze(1).expand(-1, r_exp.shape[1], -1)
        omega = torch.sigmoid(self.backbone.fusion.w(_vg))

        f_s2, f_u2, f_o2 = pf.f_s2, pf.f_u2, pf.f_o2
        _with_rel = (self.pair_res is not None) and PAIR_REL
        _pout = self._pair_res_logits(x, s_idx, o_idx, return_rel=_with_rel)
        if isinstance(_pout, tuple):
            _res, _rel_pos = _pout
        else:
            _res, _rel_pos = _pout, None
        p_ij, p_ji = self.dual_head(f_s2, f_u2, f_o2, res_logits=_res)

        #---------------- swap path (reuse layer 1, do not recalculate backbone)----------------
        r_ji = self.backbone.implicit.swap(f_s2, f_u2, f_o2)
        p_ji_pi, p_ij_pi = self.dual_head(f_o2, f_u2, f_s2, res_logits=_res)

        # ---------------- Closed Road ----------------
        p_trans, p_conv = self.closure()
        P_bar_sq = two_hop_neighborhood(p_conv, hp.ETA)
        # ⚠️ 2026-10-18: Inject **learnable diagonal bias** (see description of closure.theta_diag).
        #Purpose: Let `diag(P̄²)` in equation (28) be weighted and differentiated, so as to truly constrain the two heads,
        # Instead of degenerating into ”two independent softmax D_JS”.
        #
        # ⚠️ Implementation Note: Offset** is only used for the diag weight** of equation (28),
        # Do not change P_bar_sq itself - because P_bar_sq also needs to be fed to `mixed_student` (Equation 22),
        # There are strict requirements (K,K); if multiplied by (1,K,K) here, it will destroy the downstream broadcast.
        _db = self.closure.diag_bias()                             # (K,)
        # ⚠️ 2026-10-18 Correction: **cannot** use `P_bar_sq * diag_embed(_db)`
        # ——That is **matrix multiplication**, non-diagonal terms will also participate, causing the diagonal to be polluted
        # (The measured diag range is always 0 because the diagonal of P_bar_sq is canceled by the off-diagonal term).
        # Correct approach: Only **change the diagonal element** (the rest remains unchanged):
        #        P̄²_coh[k,k] = P̄²[k,k] · db[k]
        _dp0 = torch.diagonal(P_bar_sq)
        P_bar_sq_coh = P_bar_sq + torch.diag_embed((_db - 1.0) * _dp0)  # (K,K), only for formula (28)
        p_tilde = mixed_student(p_ij, P_bar_sq, omega_p, hp.EPS_ROW)
        # ⚠️ 2026-10-18: topk=1 ⟹ Only the strongest transfer in each row is retained as ”edge”
        # (originally used for dense judgment of adj>0 ⟹ closure all 1 ⟹ composite unreachable set constant empty)
        reach = transitive_closure_ste(p_conv[:, :K], p_trans, topk=1)

        #---------------- Metrics----------------
        G_inv = self.metric.inverse()

        #----------------Loss----------------
        parts: Dict[str, torch.Tensor] = {}
        lambdas = all_lambda_dict(hp, lambda_inv)

        if y is not None:
            y_t = y if y_tilde_ji is None else y_tilde_ji
            mu_t = torch.ones(y.shape[:-1], device=y.device, dtype=dtype) if mu is None else mu
            # ⚠️ Formula (30) passes the violation mask m_i: calculated from the external ε_assoc(P_i) < ε₀ and passed in,
            # ε_assoc is not calculated internally in the model. If the transmission is missed, it will degenerate to all 1 (constraint silent failure).
            # 2026-10-19 (P1): Add explicit assertions and change ”silent failure” to ”error reporting during training”.
            if m_i is None:
                raise AssertionError(
                    "formula(30) Passing violation mask m_i Not passed in. The external must be calculated first ε_assoc(P_i) And "
                    "press ε₀ threshold generation m_i; If there is indeed no need to transmit consistency monitoring, please explicitly transmit the entire 1 Tensor.")
            m_t = m_i
            parts["cls"] = cls_loss(p_ij, p_ji, y, y_t, mu_t, m_t)

        parts["inv"] = inv_loss(r_exp, r_imp, G_inv)

        # ⚠️ 2026-10-18: This article does not contain knowledge distillation (Equation 11 KL and Equation 12 feature regression have been deleted).

        # --- M4.5 L_dag (H2 downstream; p_ji^π must be different from p_ij) ---
        Inv_p_ij = Inv_detached(p_ij, self.M_inv, hp.EPS_INV)
        w_i = self.w[y_label.long()] if y_label is not None else torch.ones(p_ij.shape[:2], device=p_ij.device)
        L_dag, L_dag_dist, L_dag_repr = dag_loss(
            p_ij, p_ji_pi, r_ji, Inv_p_ij, self.c_dag, G_inv, w_i, self.shared, hp.LAMBDA_DAG
        )
        parts["dag_non_sym"] = L_dag

        if y is not None:
            # ⚠️ 2026-10-18: The original sym uses **hard** mask_sym (only 4 symmetric predicates),
            # The predicate is long tail ⟹ The true value in the batch almost never falls on G_sym ⟹ L_sym is always 0.
            # Now changed to **soft symmetry mask**: using ”semantic neighbors of the truth predicate” as the weight,
            # Predicate pairs with similar semantics (such as in/near, above/below) are included with small weights.
            # So that long-tail predicates also have non-zero symmetry supervision.
            _msym = self.soft_mask_sym
            if _msym is None:
                _msym = self.mask_sym
            parts["sym"] = sym_loss(p_ij, p_ji_pi, y, _msym)

        parts["coh"] = coh_loss(pf.f_r1, pf.f_r2, P_bar_sq_coh, self.head1, self.head2)

        # ---- pair existence supervision (explicit negative sampling + BCE) ----
        # Negative samples: Each image is evenly sampled from **non-GT ordered pairs** with ratio×|GT| (same distribution as the evaluation candidates).
        # The positive sample is the GT pair of the current batch. The two items are averaged and then added with equal weights.
        # Avoid the 1:3 ratio that dominates the negative items.
        if _rel_pos is not None:
            _rel_neg = self._sample_negative_rel(x, s_idx, o_idx, masks,
                                                 ratio=PAIR_REL_NEG_RATIO)
            if _rel_neg is not None and _rel_neg.numel() > 0:
                _bce = torch.nn.functional.binary_cross_entropy_with_logits
                parts["pair_rel"] = PAIR_REL_LAMBDA * 0.5 * (
                    _bce(_rel_pos, torch.ones_like(_rel_pos))
                    + _bce(_rel_neg, torch.zeros_like(_rel_neg)))
        # ⚠️ 2026-10-18 Diagnosis: Record the relative range of diag(P̄²).
        # If ≈ 0 ⟹ p_conv is still uniform ⟹ the P̄ weighting of Equation (28) degenerates into constant scaling (semantic failure).
        _dp = torch.diagonal(P_bar_sq_coh)
        _dpmed = _dp.abs().median().clamp(min=1e-12)
        self._diag_p_spread = float(((_dp.max() - _dp.min()) / _dpmed).item())

        # --- M4.11 Order regularity (Formula 29, implementation caliber: composite unreachable pair weighting) ---
        parts["ord"] = ord_loss(p_trans, reach)

        # --- M4.7/M4.8/M4.9 brake (H1: only feed V_r to hinge) ---
        # ⚠️ 2026-10-18 Fix: The original code `present = (counts > 0)` is taken from **current batch**,
        # And the VRD training set has 47.1% predicate frequency < batch_size ⟹ half of the predicates
        # The brakes are closed in any batch ⟹ L_hom / L_spread is always 0.
        # Now changed to: **Presence table pre-calibrated with the frequency of the entire training set** (registered as a fixed amount),
        # And superimpose the count within the batch as a supplement (or the two).
        from .losses import estimate_variance
        pi = pf.sample_pi()
        y_lab = y_label.long() if y_label is not None else torch.zeros(pf.f_s0.shape[:2], dtype=torch.long, device=pf.f_s0.device)
        V_r, V_global, H_r, counts = estimate_variance(pi, y_lab, K)
        present = (counts > 0).float()
        if self.predicate_presence is not None:
            present = torch.maximum(present, self.predicate_presence.to(present.device))
        # ⚠️ 2026-10-18: θ in the **first batch of automatic calibration** (quantile of real V_r),
        # Avoid passing in fixed tensors externally and causing hinge to never trigger.
        if getattr(self, "_theta_auto", False):
            # ⚠️ 2026-10-18: Calibrate θ in the dimension of **variance per sample** V̂_r = V_r / |P_r|,
            # Strictly consistent with the hinge dimension of gate_upper/gate_lower.
            #
            # The calibration method uses **EMA statistics** (mean/standard deviation) instead of the first batch of quantiles:
            # The problem with the first batch of quantiles is that the V̂ distribution of the batch used for calibration and subsequent batches will drift,
            # So θ quickly falls outside all V̂ ⟹ The hinge never triggers (this is the actual measurement).
            # EMA adapts with training and gives **clear semantics**:
            # θ_max = μ + nσ (The dispersion is significantly higher than the average ⟹ Suspicious pair-wise memory)
            # θ_min = μ − nσ (the dispersion is significantly lower than the average ⟹ suspected collapse between groups)
            with torch.no_grad():
                _v = V_r.detach().float()
                _cnt = counts.detach().float().clamp(min=1.0)
                _vhat = _v / _cnt
                _m = _vhat > 0
                if int(_m.sum()) >= 4:
                    _cur = _vhat[_m]
                    # ⚠️ 2026-10-18 The third revision: θ calibration is changed to **quantile EMA**.
                    #
                    # Why not use μ ± nσ: The measured σ of the first batch can be greater than μ
                    # (Example: μ=1.04e-3, σ=1.29e-3) ⟹ θ_min = μ−nσ < 0 is truncated to 0
                    # ⟹ `max(0, θ_min − V̂)` is always 0 ⟹ **L_spread is always 0**.
                    #
                    # Why not use ”first batch quantile”: V̂ distribution of the batch used for calibration and subsequent batches
                    # will drift ⟹ θ quickly falls outside all V̂ ⟹ The hinge never triggers.
                    #
                    # ⟹ Final solution: **Quantile EMA across batches**
                    # θ_max = EMA(Q_hi) (The dispersion is significantly high ⟹ Suspicious pair-wise memory)
                    # θ_min = EMA(Q_lo) (the dispersion is significantly low ⟹ suspected collapse between groups)
                    # The semantics of the quantile caliber are naturally consistent with the ”upper and lower bounds” and are always positive.
                    _qs = torch.quantile(_cur, torch.tensor(
                        [float(getattr(hp, "THETA_Q_LO", 0.10)),
                         float(getattr(hp, "THETA_Q_HI", 0.90))],
                        device=_cur.device))
                    _qlo, _qhi = float(_qs[0]), float(_qs[1])
                    if not hasattr(self, "_theta_ema") or self._theta_ema is None:
                        self._theta_ema = (_qlo, _qhi)
                    else:
                        _ol, _oh = self._theta_ema
                        self._theta_ema = (0.9 * _ol + 0.1 * _qlo,
                                           0.9 * _oh + 0.1 * _qhi)
                    _m2, _m3 = self._theta_ema
                    self.theta_max = max(_m3, 1e-12)
                    self.theta_min = max(min(_m2, self.theta_max * 0.999), 0.0)
        theta_max = float(self.theta_max)
        theta_min = float(self.theta_min)
        # ⚠️ 2026-10-18: Pass in counts (group size |P_r|) for scale ⟹ The brake acts on
        # **Variance per sample** V̂_r = V_r / |P_r| (The absolute magnitude of V_r is only 1e-4~1e-2,
        # The absolute threshold is never triggered; the V_r/V_global median ratio ≈ 0.995 has no discriminative power).
        parts["hom"] = gate_upper(V_r, theta_max, present, scale=counts)
        # ⚠️ 2026-10-18 Fixed: `gate_lower` returns **tuple**(l_vr + l_ins, l_vr, l_ins),
        # The original code is written as `parts[”spread”] = gate_lower(...)` and directly assigns the tuple to the loss term
        # ⟹ Subsequent multiplication reports ”can’t multiply sequence by non-int of type ’float’”.
        _sp = gate_lower(V_r, V_global, theta_min, theta_min, present, scale=counts)
        parts["spread"] = _sp[0] if isinstance(_sp, tuple) else _sp

        # --- M4.12 Replacement Guard (Formula 31) ---
        # ⚠️ 2026-10-18 Fix: The original code passed an empty list ⟹ L_PI is always 0.
        # Now perform random replacement and forward comparison on **object layer pooling Z[:,0]**.
        # Expectation: If Z[:,0] is order-independent pooling, loss is still ≈0 (qualitative);
        # This term goes up as soon as the ordered slot breaks the permutation symmetry.
        # ⚠️ 2026-10-18 Fix (Layer 3): Z of the main path is the output of **chain_attention**
        # (`backbone.encode` returns A instead of pooled Z), so the replacement sample must go
        # **Same post-processing**, otherwise ”after attention” will always be a larger value (false signal) than ”pure pooling”.
        # `post` injects backbone’s chain_attention (same group W_q/W_k/W_v).
        def _pi_post(z_n: torch.Tensor) -> torch.Tensor:
            bb = self.backbone
            return chain_attention(z_n, bb.W_q, bb.W_k, bb.W_v)

        _perms = permute_pooled(z_j, masks, None, n_perm=2, post=_pi_post)
        _z0 = Z[:, 0]                                                # (b,d)
        parts["PI"] = pi_loss(_z0, _perms)

        total = total_loss(parts, lambdas)

        out = dict(parts)
        out["total"] = total
        out["_V_r"] = V_r
        out["_V_global"] = V_global
        out["_H_r"] = H_r
        out["_p_ij"] = p_ij
        out["_p_ji"] = p_ji
        out["_p_ji_pi"] = p_ji_pi
        out["_r_exp"] = r_exp
        out["_r_imp"] = r_imp
        out["_r"] = r
        out["_omega"] = omega
        out["_p_tilde"] = p_tilde
        out["_G_inv"] = G_inv
        return out

    # --------------------------------------------------------------- M6.5 Reasoning Stripping
    def forward_inference(
        self, x: dict, masks: torch.Tensor, s_idx: torch.Tensor, o_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reasoning: Peel off the distillation branches, swap paths, closure paths, and all regular training period rules; leaving only equations (1)–(7)+Head.

        Final draft §3.7: Parameters ≈ 7.01M + R1 3,072 + R2 131,072 + R3 64,000 ≈ 7.21M.
        """
        z_j, Z, _ = self.backbone.encode(x, masks)
        v_glo = Z[:, 1]
        pf = self.backbone.pair_features(z_j, Z, s_idx, o_idx)
        # The fusion of r_exp/r_imp/r is only used to produce Head input during inference, so it follows the same chain.
        self.backbone.forward(pf, v_glo)
        _with_rel = (self.pair_res is not None) and PAIR_REL
        _pout = self._pair_res_logits(x, s_idx, o_idx, return_rel=_with_rel)
        if isinstance(_pout, tuple):
            _res, _rel = _pout
        else:
            _res, _rel = _pout, None
        p_ij, p_ji = self.dual_head(pf.f_s2, pf.f_u2, pf.f_o2,
                                    logits_only=SCORE_LOGITS, res_logits=_res)
        if _rel is not None and SCORE_LOGITS:
            # Pair-level calibration: per-pair constant translation ⟹ top1 unchanged, global R@K movable
            _b = PAIR_REL_BETA * torch.nn.functional.logsigmoid(_rel).unsqueeze(-1)
            return p_ij + _b, p_ji + _b
        return p_ij, p_ji

    def _sample_negative_rel(self, x: dict, s_idx: torch.Tensor, o_idx: torch.Tensor,
                             masks: torch.Tensor, ratio: int = 3):
        """Each graph uniformly samples ratio×|GT| negative pairs from **non-GT ordered pairs**, returning the rel logit of (M,).

        Why not ”shuffle o_idx”: what you get is the rearrangement of (GT subject, GT object),
        The subject side is still limited to the ”relevant subject”, and the negative sample distribution and predcls candidates
        (Any ordered pair in the picture) are inconsistent, and the positive and negative ratio is always 1:1.
        """
        if self.pair_res is None:
            return None
        dev = s_idx.device
        b, P = s_idx.shape
        if P < 1 or masks.dim() != 3:
            return None
        n_obj = masks[:, 2, :].sum(-1).long()                    # (b,) Number of valid objects per picture
        nmax = int(n_obj.max().item())
        if nmax < 2:
            return None
        Kneg = max(1, int(ratio) * P)

        #GT mask (exclude real pairs, avoid treating positive samples as negative samples)
        gt = torch.zeros(b, nmax, nmax, dtype=torch.bool, device=dev)
        _bi = torch.arange(b, device=dev).unsqueeze(1).expand(b, P)
        _ok = ((s_idx >= 0) & (s_idx < nmax) & (o_idx >= 0) & (o_idx < nmax))
        gt[_bi[_ok], s_idx[_ok], o_idx[_ok]] = True

        # After oversampling 4 times, take the first Kneg legal samples according to the picture
        tot = Kneg * 4
        aa = torch.randint(0, nmax, (b, tot), device=dev)
        cc = torch.randint(0, nmax - 1, (b, tot), device=dev)
        cc = cc + (cc >= aa).long()                              # Guarantee aa != cc
        # ⚠️ Must be in the three-index form `gt[bidx, aa, cc]`: just writing `gt[aa, cc]` will treat aa as
        # **batch dimension** index (aa values up to nmax-1 > b-1) ⟹ GPU device-side assert.
        _brow = torch.arange(b, device=dev).unsqueeze(1).expand(b, tot)
        _is_gt = gt[_brow, aa, cc]
        ok = (aa < n_obj[:, None]) & (cc < n_obj[:, None]) & (~_is_gt)
        rank = torch.cumsum(ok.long(), dim=1) - 1
        sel = ok & (rank < Kneg)
        if int(sel.sum()) == 0:
            return None

        _bidx = torch.arange(b, device=dev).unsqueeze(1).expand(b, tot)[sel]
        _a = aa[sel].unsqueeze(1)                                # (M,1)
        _c = cc[sel].unsqueeze(1)
        outs = []
        for key, idx in (("v", _a), ("v", _c), ("t", _a), ("t", _c),
                         ("s", _a), ("s", _c)):
            t = x[key][_bidx]                                    # (M,N,d)
            outs.append(torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])))
        _out = self.pair_res(*[o.squeeze(1) for o in outs], return_rel=True)
        return _out[1] if isinstance(_out, tuple) else None

    def _pair_res_logits(self, x: dict, s_idx: torch.Tensor,
                         o_idx: torch.Tensor, return_rel: bool = False):
        """MLP pairwise bypass: directly output (b, P, K) logits residuals from the **original** endpoint features.

        Deliberately not using `pf.f_s2/f_u2/f_o2` (they have been injected into the context by graph level z1/z2) -
        Only by keeping consistent with the MLP baseline’s ”pure pair-by-pair, no context” caliber can we test
        ”Does context matter?” When the switch is turned off, None is returned, and the logic at the calling location remains unchanged.

        When `return_rel=True`, the pair existence logit of (b,P) is additionally returned.
        """
        if self.pair_res is None:
            return None
        if x["t"].shape[-1] != self.pair_res.d_t or x["v"].shape[-1] != self.pair_res.d_v:
            # Extreme case: the caller has not passed obj_emb, x[”t”] is label id (width 1).
            # At this time, the bypass cannot work according to the isomorphic caliber and is skipped directly (equivalent to closing), which can be seen from the log side.
            return None
        outs = []
        for key, idx in (("v", s_idx), ("v", o_idx), ("t", s_idx), ("t", o_idx),
                         ("s", s_idx), ("s", o_idx)):
            t = x[key]
            outs.append(torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])))
        return self.pair_res(*outs, return_rel=return_rel)

    # --------------------------------------------------------------- Threshold Calibration
    def calibrate_theta(self, V_r_healthy: torch.Tensor,
                        q_hi: float = 0.90, q_lo: float = 0.10,
                        relative: bool = True, V_global: float = 1.0) -> None:
        """Formula (26)/(27): θ takes the Q90/Q10 quantile of **real V_r**, zero labor (final draft §3.7 step ⑥).

        ⚠️ 2026-10-18 Fix: The previous call passed in **fixed tensor** `tensor([0.5,1.0,1.5,2.0])`,
        And the actual V_r (variance within group) magnitude is much less than 0.5 ⟹ Q10 still >real V_r ⟹
        `max(0, θ_min − V_r)` is always 0 ⟹ **L_spread is always 0**.
        Now change to: take the quantile from **real V_r** (or relative dimension V_r/V_global),
        Ensure that θ_min falls within the actual distribution ⟹ hinge must be partially triggered.
        """
        _v = V_r_healthy.float().reshape(-1)
        _v = _v[torch.isfinite(_v)]
        if _v.numel() == 0:
            self.theta_max, self.theta_min = 1.0, 0.0
            return
        if relative:
            _v = _v / max(float(V_global), 1e-8)
        self.theta_max = float(torch.quantile(_v, q_hi).item())
        self.theta_min = float(torch.quantile(_v, q_lo).item())

    def trainable_parameter_names(self) -> list:
        return [n for n, p in self.named_parameters() if p.requires_grad]


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    """Parameter count (acceptance gates G4/G7 of M6.5 depend on this function)."""
    ps = module.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not only_trainable))


def strip_for_inference(model: VRPCatModel) -> nn.Module:
    """M6.5: Return a container containing only the submodules required for inference, for parameter counting and consistency assertions."""
    keep = nn.ModuleDict({
        "backbone": model.backbone,
        "dual_head": model.dual_head,
    })
    return keep
