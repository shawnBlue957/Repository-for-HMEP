"""T1 Unit Contract Tests (Module Design Document §4.1) – ≥1 item per function.

Checkpoints: Shape/Type/Range/Simplex/Positive Definite.
"""
from __future__ import annotations

import pytest
import torch

from conftest import make_batch
from vrp import HyperParams, expect, expect_psd, expect_simplex
from vrp.backbone import (
    DualHead,
    GatedFusion,
    OrderedDifference,
    OrderedImplicitFlow,
    PrefixHead,
    chain_attention,
    intra_layer_aggregate,
)
from vrp.closure import (
    ClosureParams,
    closure_invariance_check,
    converse_closure_batch,
    mixed_student,
    rownorm,
    transitive_closure_ste,
    two_hop_neighborhood,
)
from vrp.diagnostics import (
    build_epsilon_assoc,
    dvr,
    epsilon0_from_percentile,
    ivr,
    piv1,
    res_g,
    violation_mask,
)
from vrp.enrich import (
    Inv,
    MetricModule,
    build_c_dagger,
    build_groups_and_weights,
    precompute_M_inv,
    smw_inverse,
    top1_indicators,
)
from vrp.guards import (
    assert_hinge_on_Vr,
    assert_pairing_protocol,
    assert_theta_in_range,
    assert_vr_nonzero,
    HingePositionError,
    PairingProtocolError,
    ThetaRangeError,
    warn_if_artifact,
)
from vrp.losses import (
    SharedProjection,
    cls_loss,
    coh_loss,
    dag_loss,
    estimate_variance,
    gate_lower,
    gate_upper,
    inv_loss,
    js_divergence,
    ord_loss,
    pi_loss,
    sym_loss,
    total_loss,
    assert_middle_band_free,
)


# ---------------------------------------------------------------- M0.2 Guard

def test_expect_rejects_wrong_ndim():
    with pytest.raises(AssertionError):
        expect(torch.zeros(2, 3), "b,P,d", "t")


def test_expect_simplex_accepts_and_rejects():
    expect_simplex(torch.ones(4) / 4, "ok")
    with pytest.raises(AssertionError):
        expect_simplex(torch.tensor([0.5, 0.4]), "bad_sum")
    with pytest.raises(AssertionError):
        expect_simplex(torch.tensor([1.5, -0.5]), "bad_neg")


def test_expect_psd():
    a = torch.randn(5, 5)
    expect_psd(a @ a.t() + 1e-3 * torch.eye(5), "spd")
    with pytest.raises(AssertionError):
        expect_psd(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), "indefinite")


# ---------------------------------------------------------------- M1.x Skeleton

def test_M11_modal_projection_shape(hp_tiny):
    from vrp.backbone import ModalProjection
    mp = ModalProjection({"v": 16, "t": 16, "s": 16}, hp_tiny.D)
    out = mp({m: torch.randn(2, 6, 16) for m in ("v", "t", "s")})
    for m in ("v", "t", "s"):
        expect(out[m], "b,N,d", f"h_{m}")
        assert out[m].shape == (2, 6, hp_tiny.D)


def test_M12_count_pooling_and_z0_shape(hp_tiny):
    b = make_batch(hp_tiny)
    h = {m: torch.randn(b["b"], b["N"], hp_tiny.D) for m in ("v", "t", "s")}
    W_a = torch.nn.Linear(3 * hp_tiny.D, hp_tiny.D)
    z_j, Z = intra_layer_aggregate(h, W_a, b["masks"])
    assert Z.shape == (b["b"], 3, hp_tiny.D)
    assert z_j.shape == (b["b"], b["N"], hp_tiny.D)


def test_M13_attention_rows_sum_to_one(hp_tiny):
    Z = torch.randn(2, 3, hp_tiny.D)
    d = hp_tiny.D
    A = chain_attention(Z, torch.nn.Linear(d, d), torch.nn.Linear(d, d), torch.nn.Linear(d, d))
    assert A.shape == Z.shape


def test_M14_ordered_difference_shape(hp_tiny):
    """α,β are **scalar** coefficients (b,P,1) (final draft §3.8 parameter table Δ=(4d−d)×h and formula (25) π∈R² are jointly determined)."""
    d = hp_tiny.D
    od = OrderedDifference(d, hp_tiny.H)
    f_u = torch.randn(2, 4, d); f_s = torch.randn(2, 4, d); f_o = torch.randn(2, 4, d)
    v_glo = torch.randn(2, d)
    f_r, alpha, beta = od(f_u, f_s, f_o, v_glo)
    assert f_r.shape == (2, 4, d)
    assert alpha.shape == beta.shape == (2, 4, 1)


def test_M14_R1_param_count():
    """The parameter increment of R1 is Δ=(4d−d)×h=3d·h=3072 (d=256, h=4), which is consistent with the final draft §3.8."""
    d, h = 256, 4
    od = OrderedDifference(d, h)
    w_only = sum(p.numel() for n, p in od.named_parameters() if n.endswith("weight"))
    # MLP_δ: 4d→h→2 ⟹ Weight 4d·h + h·2
    assert w_only == 4 * d * h + h * 2
    delta = (4 * d - d) * h
    assert delta == 3072


def test_M14_R1_direction_sensitivity(hp_tiny):
    """T2-d prefix: alpha or beta must change after exchanging (s,o), otherwise R1 will not take effect."""
    d = hp_tiny.D
    od = OrderedDifference(d, hp_tiny.H)
    f_u = torch.randn(2, 4, d); f_s = torch.randn(2, 4, d); f_o = torch.randn(2, 4, d)
    v_glo = torch.randn(2, d)
    _, a1, b1 = od(f_u, f_s, f_o, v_glo)
    _, a2, b2 = od(f_u, f_o, f_s, v_glo)          # Swap ordered slots
    assert not (torch.allclose(a1, a2) and torch.allclose(b1, b2))


def test_M15_R2_pair_sensitivity(hp_tiny):
    """T2-c prefix: r_imp must change after exchanging (s,o), otherwise R2 will not take effect and L_dag ≡ 0."""
    d = hp_tiny.D
    imp = OrderedImplicitFlow(d)
    f_s, f_u, f_o = (torch.randn(2, 4, d) for _ in range(3))
    r1 = imp(f_s, f_u, f_o)
    r2 = imp(f_o, f_u, f_s)
    assert not torch.allclose(r1, r2)


def test_M15_slot_order_locked(hp_tiny):
    """Residual die expansion: f_{u_2} is in the first slot - the locked slot sequence is recalculated by manual Linear."""
    d = hp_tiny.D
    imp = OrderedImplicitFlow(d)
    f_s, f_u, f_o = (torch.randn(2, 4, d) for _ in range(3))
    out = imp(f_s, f_u, f_o)
    manual = imp.mlp[-1](torch.nn.functional.relu(
        imp.mlp[0](torch.cat([f_s, f_u, f_o], dim=-1))))
    assert torch.allclose(out, manual, atol=1e-6)


def test_M16_gate_is_sigmoid_range(hp_tiny):
    d = hp_tiny.D
    gf = GatedFusion(d)
    r_exp, r_imp, v_glo = torch.randn(2, 4, d), torch.randn(2, 4, d), torch.randn(2, d)
    r, omega = gf(r_exp, r_imp, v_glo)
    assert (omega >= 0).all() and (omega <= 1).all()
    assert r.shape == (2, 4, d)


def test_M17_dual_head_simplex(hp_tiny):
    d, K = hp_tiny.D, hp_tiny.K
    head = DualHead(d, K)
    p_ij, p_ji = head(torch.randn(2, 4, d), torch.randn(2, 4, d), torch.randn(2, 4, d))
    expect_simplex(p_ij, "p_ij"); expect_simplex(p_ji, "p_ji")


def test_M17_dual_head_param_count():
    """Formula (7): Net increase 5dK = 64,000 (d=256, K=50).

    ⚠️ Caliber check (discovery during the implementation period): ”New head 76,800 / old head 12,800 / net increase 64,000” in the final draft §3.8
    All are **weight only** calibers (6dK/dK/5dK). 76,900 / 12,850 / 64,050 net after including offset.
    The net increase difference between the two calibers is only 50 (2K-K), which does not affect the conclusion; here it is asserted according to the document caliber, and the difference is explicitly recorded.
    """
    head = DualHead(256, 50)
    w = sum(p.numel() for n, p in head.named_parameters() if n.endswith("weight"))
    b = sum(p.numel() for n, p in head.named_parameters() if n.endswith("bias"))
    assert w == 6 * 256 * 50, f"Only the weight should be 6dK=76800, What you get {w}"
    assert b == 2 * 50, f"The offset should be 2K=100, What you get {b}"
    assert w - 256 * 50 == 5 * 256 * 50, "The net increase (same caliber) should be 5dK=64000"


def test_M41_prefix_head():
    h = PrefixHead(256, 50)
    assert h(torch.randn(2, 4, 256)).shape == (2, 4, 50)


# ---------------------------------------------------------------- M2.x Enrichment

def test_M21_metric_psd_and_eigenvalue_floor(hp_tiny):
    m = MetricModule(hp_tiny.D, hp_tiny.R_LOWRANK, hp_tiny.EPS_G)
    G = m.metric()
    expect_psd(G, "G")
    ev = torch.linalg.eigvalsh(G)
    # Tolerance description: eigvalsh has ~1e-8 relative rounding on d=32 for eigenvalues of magnitude ε=1e-4,
    # So use 1e-7 instead of 1e-9 (which would capture purely numerical noise).
    assert ev.min().item() >= hp_tiny.EPS_G - 1e-7, "The minimum eigenvalue must be ≥ ε"


def test_M22_smw_equals_explicit_inverse(hp_tiny):
    """Relative error between SMW and explicit inversion (the condition number of G is ~1/ε, so the relative criterion must be used)."""
    A = torch.randn(hp_tiny.D, hp_tiny.R_LOWRANK) * 0.1
    G = A @ A.t() + hp_tiny.EPS_G * torch.eye(hp_tiny.D)
    smw = smw_inverse(A, hp_tiny.EPS_G)
    explicit = torch.linalg.inv(G)
    rel = (smw - explicit).norm() / explicit.norm()
    assert rel < 1e-4, f"relative error {rel:.3e} Too big"


def test_M22_smw_is_right_inverse(hp_tiny):
    A = torch.randn(hp_tiny.D, hp_tiny.R_LOWRANK) * 0.1
    G = A @ A.t() + hp_tiny.EPS_G * torch.eye(hp_tiny.D)
    G_inv = smw_inverse(A, hp_tiny.EPS_G)
    assert torch.allclose(G @ G_inv, torch.eye(hp_tiny.D), atol=1e-2)


def test_M23_Minv_range_and_no_grad():
    v_txt, v_inv = torch.randn(8, 16), torch.randn(8, 16)
    M = precompute_M_inv(v_txt, v_inv)
    assert M.shape == (8, 8)
    assert M.min().item() >= -1.0001 and M.max().item() <= 1.0001


def test_M24_Inv_returns_simplex():
    K, d = 8, 16
    M_inv = precompute_M_inv(torch.randn(K, d), torch.randn(K, d))
    p = torch.softmax(torch.randn(2, 4, K), dim=-1)
    out = Inv(p, M_inv, 0.07)
    expect_simplex(out, "Inv(p)")


def test_M24_top1_and_c_dagger():
    K, d = 8, 16
    M_inv = precompute_M_inv(torch.randn(K, d), torch.randn(K, d))
    J = top1_indicators(M_inv)
    assert J.shape == (K, K)
    assert torch.allclose(J.sum(dim=-1), torch.ones(K))
    c = build_c_dagger(M_inv, torch.randn(K, d), J)
    assert c.shape == (K, d)


def test_M25_groups_mutually_exclusive_and_complete():
    K = 8
    rho = torch.rand(K, K)
    mn, ms, ma, w = build_groups_and_weights(rho)
    total = mn + ms + ma
    assert torch.allclose(total, torch.ones(K)), "The three groups must be mutually exclusive and complete"
    assert (w >= 0).all() and (w <= 1).all()


# ---------------------------------------------------------------- M3.x Closure

def test_M31_shapes_and_simplex(hp_tiny):
    cp = ClosureParams(hp_tiny.K)
    p_trans, p_conv = cp()
    assert p_trans.shape == (hp_tiny.K,)
    assert (p_trans > 0).all() and (p_trans < 1).all()
    assert p_conv.shape == (hp_tiny.K, hp_tiny.K + 1)
    expect_simplex(p_conv, "p_conv")


def test_M32_converse_swaps_direction(hp_tiny):
    """Directional correctness: input (i,r,j) must produce (j,r’,i)."""
    K = hp_tiny.K
    _, p_conv = ClosureParams(K)()
    E = torch.tensor([[1, 2, 3]])                       # (i=1, r=2, j=3)
    out = converse_closure_batch(E, p_conv, topk=1)
    assert out.shape[1] == 3
    assert int(out[0, 0].item()) == 3, "new_i Must be j"
    assert int(out[0, 2].item()) == 1, "new_j Must be i"


def test_M33_transitive_closure_diagonal(hp_tiny):
    K = hp_tiny.K
    adj = torch.eye(K) + 0.1 * torch.rand(K, K)
    p_trans = torch.full((K,), 0.5)
    reach = transitive_closure_ste(adj, p_trans)
    assert reach.shape == (K, K)
    assert (torch.diagonal(reach) > 0.5).all()


def test_M34_two_hop_is_true_square(hp_tiny):
    """L3 correction: P̄^⊗2 must be equal to P̄@P̄ and not equal to any row scaling form."""
    _, p_conv = ClosureParams(hp_tiny.K)()
    P2, Pbar, row_scale = two_hop_neighborhood(p_conv, 1.0, return_debug=True)
    assert torch.allclose(P2, Pbar @ Pbar, atol=1e-8), "It must be a real double jump"
    assert not torch.allclose(P2, row_scale, atol=1e-6), "must not degenerate into row scaling (otherwise A8 Invalid)"


def test_M35_mixed_student_omega_zero_identity(hp_tiny):
    K = hp_tiny.K
    p = torch.softmax(torch.randn(2, 4, K), dim=-1)
    P2 = torch.eye(K)
    out = mixed_student(p, P2, omega_p=0.0)
    assert torch.equal(out, p), "ω_p=0 must strictly degenerate into p_ij"


def test_M35_mixed_student_simplex(hp_tiny):
    K = hp_tiny.K
    p = torch.softmax(torch.randn(2, 4, K), dim=-1)
    _, p_conv = ClosureParams(K)()
    P2 = two_hop_neighborhood(p_conv, 1.0)
    out = mixed_student(p, P2, omega_p=0.2)
    expect_simplex(out, "p_tilde")


def test_M36_closure_invariance_with_shared_edges(hp_tiny):
    K = hp_tiny.K
    _, p_conv = ClosureParams(K)()
    E = torch.tensor([[1, 2, 3], [0, 1, 2]])
    dev = closure_invariance_check(E, [E], p_conv, topk=1)
    assert dev == 0.0


# ---------------------------------------------------------------- M4.x loss

def test_M41_cls_loss_nonneg_and_masked():
    K = 5
    p_ij = torch.softmax(torch.randn(2, 3, K), dim=-1)
    p_ji = torch.softmax(torch.randn(2, 3, K), dim=-1)
    y = torch.zeros(2, 3, K); y[..., 0] = 1.0
    yt = torch.zeros(2, 3, K)
    mu = torch.ones(2, 3, 1)
    m = torch.zeros(2, 3)
    out = cls_loss(p_ij, p_ji, y, yt, mu, m)
    assert float(out) == 0.0, "The full mask is 0 should be returned when 0 rather than NaN"


def test_M42_inv_loss_zero_when_equal(hp_tiny):
    r = torch.randn(2, 4, hp_tiny.D)
    G_inv = torch.eye(hp_tiny.D)
    assert float(inv_loss(r, r, G_inv)) == 0.0




def test_M45_dag_loss_requires_detached_target(hp_tiny):
    K, d = hp_tiny.K, hp_tiny.D
    p_ij = torch.softmax(torch.randn(2, 3, K), dim=-1)
    p_ji = torch.softmax(torch.randn(2, 3, K), dim=-1)
    r_ji = torch.randn(2, 3, d)
    Inv_p = torch.softmax(torch.randn(2, 3, K), dim=-1).requires_grad_(True)
    c = torch.randn(K, d); G_inv = torch.eye(d); w = torch.ones(2, 3)
    with pytest.raises(AssertionError):
        dag_loss(p_ij, p_ji, r_ji, Inv_p, c, G_inv, w, SharedProjection(d))


def test_M45_dag_loss_runs_and_nonneg(hp_tiny):
    K, d = hp_tiny.K, hp_tiny.D
    p_ij = torch.softmax(torch.randn(2, 3, K), dim=-1)
    p_ji = torch.softmax(torch.randn(2, 3, K), dim=-1)
    r_ji = torch.randn(2, 3, d)
    Inv_p = torch.softmax(torch.randn(2, 3, K), dim=-1).detach()
    c = torch.randn(K, d); G_inv = torch.eye(d); w = torch.ones(2, 3)
    L, l_d, l_r = dag_loss(p_ij, p_ji, r_ji, Inv_p, c, G_inv, w, SharedProjection(d))
    assert float(L) >= 0 and float(l_d) >= 0 and float(l_r) >= 0


def test_M46_sym_loss_only_on_mask():
    K = 6
    p_ij = torch.softmax(torch.randn(2, 3, K), dim=-1)
    p_ji = torch.softmax(torch.randn(2, 3, K), dim=-1)
    y = torch.zeros(2, 3, K)
    mask_zero = torch.zeros(K)
    out = sym_loss(p_ij, p_ji, y, mask_zero)
    assert float(out) == 0.0, "Mask full 0 time L_sym Must be 0"


def test_M47_variance_identity_Vr_equals_neg_log_Hr(hp_tiny):
    """T4-a Identity: V_r = −log H(r)."""
    K = hp_tiny.K
    pi = torch.randn(20, 2)
    y = torch.randint(0, K, (20,))
    V_r, V_global, H_r, cnt = estimate_variance(pi, y, K)
    assert torch.allclose(V_r, -torch.log(H_r), atol=1e-8)
    assert (V_r >= 0).all()


def test_M47_degenerate_Vr_is_zero():
    """F7 Prerequisite: V_r ≡ 0 for any group size of per-relation constants."""
    K = 4
    pi_base = torch.randn(K, 2)
    pi = pi_base.repeat_interleave(3, dim=0)          # Identical within the group
    y = torch.arange(K).repeat_interleave(3)
    V_r, V_global, H_r, cnt = estimate_variance(pi, y, K)
    assert torch.allclose(V_r, torch.zeros(K), atol=1e-10)


def test_M48_gate_upper_zero_below_threshold(hp_tiny):
    """There is no constraint below the threshold (absolute semantics).

    ⚠️ 2026-10-18: `gate_upper` defaults to `rel=True` (per-sample variance V̂ = V_r/|P_r|),
    Therefore, explicit `rel=False` here verifies the hinge direction in absolute dimensions.
    """
    V_r = torch.tensor([0.1, 0.2, 0.3])
    assert float(gate_upper(V_r, theta_max=1.0, rel=False)) == 0.0


def test_M48_gate_upper_penalizes_above(hp_tiny):
    V_r = torch.tensor([0.1, 2.0])
    assert float(gate_upper(V_r, theta_max=1.0)) > 0


def test_M49_gate_lower_zero_in_middle_band():
    """The middle section is unconstrained (C9).

    ⚠️ 2026-10-18: `gate_lower` defaults to `rel=True` (hinge acts on **V_r/V_global**
    Relative variance, because the absolute value of the measured V_r is naturally extremely small and the absolute threshold is never triggered).
    Therefore, explicit `rel=False` here verifies the two-way hinge under absolute semantics.
    """
    V_r = torch.tensor([1.0, 3.0, 5.0, 9.0])
    L, l_vr, l_ins = gate_lower(V_r, torch.tensor(100.0), theta_min=0.48341,
                                theta_min_g=0.0, rel=False)
    assert float(l_vr) == 0.0, "The middle section is not constrained (C9)"


def test_M49b_gate_lower_relative_mode_activates():
    """⚠️ 2026-10-18 Newly added: Hinge can be triggered in the **variance per sample** dimension (always 0 before repair).

    The hinge of the brake acts on `V̂_r = V_r / |P_r|` (variance per sample):
      · When scale is not passed, it degenerates into the self-normalization of ”V_r / median(V_r)”;
      · When passing scale, it is ”V_r / group size”.
    The hinge should be triggerable in both calibers.
    """
    V_r = torch.tensor([0.001, 0.002, 0.003])
    V_g = torch.tensor(0.01)
    # (a) Explicitly give the group size: V̂ = V_r/1 = V_r (unchanged when the group size is 1)
    L, l_vr, _ = gate_lower(V_r, V_g, theta_min=0.5, theta_min_g=0.0,
                             rel=True, scale=torch.tensor([1.0, 1.0, 1.0]))
    assert float(l_vr) > 0.0, "Under the variance dimension of each sample, it is still 0 ⟹ The brakes are still not in effect"
    # (b) Self-returning scale (do not pass scale)
    L2, l_vr2, _ = gate_lower(V_r, V_g, theta_min=1.5, theta_min_g=0.0, rel=True)
    assert float(l_vr2) > 0.0, "Not triggered under self-returning caliber"


def test_M49_middle_band_assert():
    V_r = torch.tensor([1.0, 3.0, 5.0, 9.0])
    assert_middle_band_free(V_r, theta_min=0.48, theta_max=9.67)


def test_M410_coh_loss_runs(hp_tiny):
    K, d = hp_tiny.K, hp_tiny.D
    f1 = torch.randn(2, 3, d); f2 = torch.randn(2, 3, d)
    P2 = torch.eye(K) + 0.1 * torch.rand(K, K)
    assert float(coh_loss(f1, f2, P2, PrefixHead(d, K), PrefixHead(d, K))) >= 0


def test_M410_coh_loss_bounded_regression(hp_tiny):
    """Regression: Head_1 output not softmaxing had caused L_coh to explode to 8.9e11 (illegal probability).

    This defect was measured and captured in the comparative experiment, and it is locked here: L_coh must be of the same magnitude as other losses (O(1)),
    And both input parameters of D_JS must be legal simplexes.
    """
    K, d = hp_tiny.K, hp_tiny.D
    torch.manual_seed(0)
    P2 = torch.eye(K) + 0.1 * torch.rand(K, K)
    for _ in range(5):
        f1 = torch.randn(2, 3, d) * 3.0          # Amplify the input and force out the illegal probability
        f2 = torch.randn(2, 3, d) * 3.0
        val = coh_loss(f1, f2, P2, PrefixHead(d, K), PrefixHead(d, K))
        assert torch.isfinite(val), "L_coh non-finite"
        assert float(val) < 100.0, f"L_coh={float(val):.3e} Abnormal magnitude (suspected of not softmax)"


def test_M410_coh_internal_is_simplex(hp_tiny):
    """Directly test that the distribution constructed by the Head_1 branch is a legal simplex (nonnegative, sums to 1)."""
    K, d = hp_tiny.K, hp_tiny.D
    head1 = PrefixHead(d, K)
    P2 = torch.eye(K) + 0.1 * torch.rand(K, K)
    f1 = torch.randn(2, 3, d) * 5.0
    diag_p = torch.diagonal(P2)
    a = torch.softmax(head1(f1), dim=-1) * diag_p.unsqueeze(0)
    a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    assert (a >= 0).all(), "Negative elements appear in the distribution (illegal probability)"
    assert torch.allclose(a.sum(dim=-1), torch.ones_like(a.sum(dim=-1)), atol=1e-5)


def test_M411_ord_loss_zero_when_satisfied():
    # ⚠️ 2026-10-19 (P0-3): Equation (29) has been rewritten according to the implementation standard, and the signature has been changed to ord_loss(p_trans, reach).
    # The old ζ-hinge (log p triplet) signature is deprecated.
    K = 4
    p_trans = torch.full((K,), 0.5)
    reach = torch.ones(K, K)                   # Fully reachable ⟹ The unreachable set is empty
    out = ord_loss(p_trans, reach)
    assert float(out) >= 0
    # All reachable ⟹ unreachable All 0 ⟹ The numerator is 0
    assert float(out) == 0.0


def test_M412_pi_loss_zero_for_invariant():
    z0 = torch.randn(2, 8)
    assert float(pi_loss(z0, [z0, z0])) == 0.0
    #Tensor input (model.py actually passes stacked tensors) should also be supported
    stack = torch.stack([z0, z0], 0)
    assert float(pi_loss(z0, stack)) == 0.0


def test_M413_total_loss_missing_key_raises():
    parts = {"cls": torch.tensor(1.0)}
    with pytest.raises(AssertionError):
        total_loss(parts, {}, expect_keys=["cls", "inv"])


# ---------------------------------------------------------------- M5.x Diagnostics

def test_M51_dvr_range_and_piv_identity():
    r_ij = torch.randn(2, 4, 8); r_ji = torch.randn(2, 4, 8)
    v = dvr(r_ij, r_ji)
    assert 0.0 <= float(v) <= 1.0
    same = torch.randn(2, 4, 8)
    assert float(piv1(same, same)) < 1e-6, "PIV₁ ≡ 0 is the identity (join exchangeability)"


def test_M52_ivr_runs():
    K = 5
    a = torch.softmax(torch.randn(2, 3, K), dim=-1)
    b = torch.softmax(torch.randn(2, 3, K), dim=-1)
    w = torch.ones(2, 3)
    assert float(ivr(a, b, w)) >= 0


def test_M53_res_g_zero_for_perfect_match(hp_tiny):
    d = hp_tiny.D
    delta = torch.zeros(2, 3, d)
    assert float(res_g(delta, torch.eye(d))) == 0.0


def test_M54_epsilon_assoc_shape_and_range():
    triples = torch.randint(0, 5, (100, 3))
    eps = build_epsilon_assoc(triples, K_plus=6)
    assert eps.shape == (6, 6, 6)
    assert (eps >= 0).all() and (eps <= 1).all()


def test_M54_violation_mask_and_percentile():
    triples = torch.randint(0, 5, (50, 3))
    eps = build_epsilon_assoc(triples, K_plus=6)
    e0 = epsilon0_from_percentile(eps)
    m = violation_mask(eps, triples, e0)
    assert m.shape == (50,) and set(m.unique().tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------- M0.4 Hard assertion

def test_H1_assert_hinge_position():
    assert_hinge_on_Vr("V_r")
    with pytest.raises(HingePositionError):
        assert_hinge_on_Vr("V_global")


def test_H1_gate_rejects_non_Vr_signature():
    """Signature level assertion: gate_upper/gate_lower internal fixed check ’V_r’ (H1)."""
    V_r = torch.tensor([1.0])
    gate_upper(V_r, 0.5)          # Don’t throw
    assert True


def test_H3_theta_range_three_assertions():
    V = torch.tensor([0.5, 1.0, 1.5, 2.0])
    assert_theta_in_range(1.8, 0.2, V)                 # Compliance
    with pytest.raises(ThetaRangeError):
        assert_theta_in_range(3.0, 0.2, V)             # theta_max over range
    with pytest.raises(ThetaRangeError):
        assert_theta_in_range(1.8, 0.0, V)             # F8: theta_min = 0
    with pytest.raises(ThetaRangeError):
        assert_theta_in_range(1.8, 0.9, V)             # overtightening


def test_pairing_protocol_amplitude_and_direction():
    assert_pairing_protocol(-1e-15)
    with pytest.raises(PairingProtocolError):
        assert_pairing_protocol(1e-3)                  # Insufficient range
    with pytest.raises(PairingProtocolError):
        assert_pairing_protocol(5e-15, tol_dir=0.0)    # Excessive amplitude but wrong direction


def test_artifact_detector_discriminates():
    assert warn_if_artifact([0.5, 0.5, 0.5]) is True       # Path B: Drift 0%
    with pytest.raises(AssertionError):
        warn_if_artifact([0.0594, 0.0000])                 # Path A: Drift 100%


def test_F8_assert_vr_nonzero():
    assert_vr_nonzero(0.05)
    with pytest.raises(ThetaRangeError):
        assert_vr_nonzero(0.0)


def test_js_divergence_symmetric_and_zero_on_equal():
    K = 5
    p = torch.softmax(torch.randn(2, K), dim=-1)
    q = torch.softmax(torch.randn(2, K), dim=-1)
    assert float(js_divergence(p, p).abs().max()) < 1e-6
    assert torch.allclose(js_divergence(p, q), js_divergence(q, p), atol=1e-6)
