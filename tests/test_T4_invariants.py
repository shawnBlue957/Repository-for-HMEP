"""T4 Invariant and Identity Testing (Module Design Document §4.4) – T4-a … T4-i."""
from __future__ import annotations

import torch

from conftest import build_model, make_batch
from vrp.backbone import intra_layer_aggregate
from vrp.closure import ClosureParams, transitive_closure_ste, two_hop_neighborhood
from vrp.enrich import MetricModule, smw_inverse
from vrp.losses import estimate_variance, gate_lower, gate_upper
from vrp.diagnostics import piv1


def test_T4a_Vr_equals_neg_log_Hr(hp_tiny):
    K = hp_tiny.K
    pi = torch.randn(30, 2)
    y = torch.randint(0, K, (30,))
    V_r, _, H_r, _ = estimate_variance(pi, y, K)
    assert torch.allclose(V_r, -torch.log(H_r), atol=1e-8)


def test_T4b_G_times_Ginv_identity(hp_tiny):
    m = MetricModule(hp_tiny.D, hp_tiny.R_LOWRANK, hp_tiny.EPS_G)
    G, G_inv = m.metric(), m.inverse()
    assert torch.allclose(G @ G_inv, torch.eye(hp_tiny.D), atol=1e-2)


def test_T4c_smw_equals_torch_inverse(hp_tiny):
    A = torch.randn(hp_tiny.D, hp_tiny.R_LOWRANK) * 0.1
    G = A @ A.t() + hp_tiny.EPS_G * torch.eye(hp_tiny.D)
    smw = smw_inverse(A, hp_tiny.EPS_G)
    explicit = torch.linalg.inv(G)
    rel = (smw - explicit).norm() / explicit.norm()
    assert rel < 1e-4, f"relative error {rel:.3e}"


def test_T4d_z0_permutation_invariant(hp_tiny):
    """z_0 is naturally invariant to object index permutation (an important by-product of the final formula 2) - the core invariant of M1.2."""
    torch.manual_seed(0)
    b, N, d = 2, 6, hp_tiny.D
    h = {m: torch.randn(b, N, d) for m in ("v", "t", "s")}
    masks = torch.zeros(b, 3, N)
    masks[:, 0, :] = 1.0
    masks[:, 1, :] = 1.0
    masks[:, 2, :] = 1.0
    W_a = torch.nn.Linear(3 * d, d)
    _, Z = intra_layer_aggregate(h, W_a, masks)

    perm = torch.randperm(N)
    h_p = {m: h[m][:, perm, :] for m in h}
    _, Z_p = intra_layer_aggregate(h_p, W_a, masks)
    assert torch.allclose(Z[:, 0], Z_p[:, 0], atol=1e-6), "z_0 Must be replaced unchanged"


def test_T4e_piv1_identity_zero():
    """PIV₁ ≡ 0 is an identity (join commutativity), not an experimental observation."""
    x = torch.randn(3, 5, 8)
    assert float(piv1(x, x)) < 1e-6


def test_T4f_degenerate_Vr_zero_any_group_size():
    """V_r ≡ 0 is true for **any group size** under per-relation constants (parsing premise of F7)."""
    K = 5
    base = torch.randn(K, 2)
    for gsize in (1, 2, 5, 13):
        pi = base.repeat_interleave(gsize, dim=0)
        y = torch.arange(K).repeat_interleave(gsize)
        V_r, _, _, _ = estimate_variance(pi, y, K)
        assert torch.allclose(V_r, torch.zeros(K), atol=1e-10), f"group size {gsize} time V_r non-zero"


def test_T4g_two_hop_is_true_square_not_row_scaling(hp_tiny):
    _, p_conv = ClosureParams(hp_tiny.K)()
    P2, Pbar, row_scale = two_hop_neighborhood(p_conv, 1.0, return_debug=True)
    assert torch.allclose(P2, Pbar @ Pbar, atol=1e-8)
    assert not torch.allclose(P2, row_scale, atol=1e-6)


def test_T4h_middle_band_total_penalty_zero():
    """The two-way hinges in the middle section (θ_min ≤ V_r ≤ θ_max) are both 0 (C9).

    ⚠️ 2026-10-18: `gate_lower` defaults to `rel=True` (relative variance),
    Therefore, `rel=False` is explicitly stated here to verify the constant zero property of the two-way hinge in the middle under **absolute dimensions**.
    """
    V_r = torch.tensor([1.0, 3.0, 5.0, 9.0])
    tmin, tmax = 0.48341, 9.66821
    up = gate_upper(V_r, tmax)
    lo, _, _ = gate_lower(V_r, torch.tensor(1e6), tmin, 0.0, rel=False)
    assert float(up + lo) == 0.0


def test_T4i_inference_matches_train_main_path(hp_tiny):
    """M6.5: p_ij of the inference path and the main training path must be consistent element by element."""
    model = build_model(hp_tiny)
    model.eval()
    b = make_batch(hp_tiny)
    with torch.no_grad():
        z_j, Z, _ = model.backbone.encode(b["x"], b["masks"])
        v_glo = Z[:, 1]
        pf = model.backbone.pair_features(z_j, Z, b["s_idx"], b["o_idx"])
        model.backbone.forward(pf, v_glo)
        p_ij_train, _ = model.dual_head(pf.f_s2, pf.f_u2, pf.f_o2)
        p_ij_inf, _ = model.forward_inference(b["x"], b["masks"], b["s_idx"], b["o_idx"])
    assert torch.allclose(p_ij_train, p_ij_inf, atol=1e-6)
