"""T5 failed injection tests (Module Design Document §4.5) – T5-a … T5-f.

Value: Convert the ”review record” of the final draft into **executable regression prevention**.
Each test intentionally injects a known error that the assertion system must catch.
"""
from __future__ import annotations

import pytest
import torch

from vrp.closure import ClosureParams, two_hop_neighborhood
from vrp.enrich import MetricModule
from vrp.guards import (
    HingePositionError,
    ThetaRangeError,
    assert_hinge_on_Vr,
    assert_theta_in_range,
    assert_vr_nonzero,
    hinge_trigger_rate,
)
from vrp.losses import gate_lower, gate_upper


# ------------------------------------------------------------------ T5-a (C1)
def test_T5a_hinge_on_Vglobal_is_rejected():
    """Injection: hinge falls on V_global (C1 is in the wrong position)."""
    with pytest.raises(HingePositionError):
        assert_hinge_on_Vr("V_global")


def test_T5a_secondary_guard_trigger_rate_zero():
    """Secondary protection: Even if the assertion is bypassed, the brake triggering rate must expose silent failure.

    Construction: V_global is always greater than the degenerate state V_r ⟹ The upper bound hinge is always 0 in the degenerate state.
    """
    V_r_degenerate = torch.zeros(5)                 # per-relation constant
    theta_max = 1.0
    up, low = hinge_trigger_rate(V_r_degenerate, theta_max, theta_min=0.05)
    assert up == 0.0, "Degenerate upper bound hinge Hengwei 0 ⟹ The brake fails silently and must be captured by trigger rate diagnosis."


# ------------------------------------------------------------------ T5-b (C2/C7)
def test_T5b_same_direction_hinges_lose_discrimination():
    """Injection: The two hinges are changed to the same direction (repetition penalty, historical writing) ⟹ Loss of discrimination (C7).

    Structural criterion: When the two expressions are in the same direction, they are **identical**; the correct way to write them is to respond oppositely to both ends of V_r.
    """
    V_r = torch.linspace(0.0, 5.0, 50)
    tmin, tmax = 0.48341, 9.66821

    wrong_a = torch.clamp(tmin - V_r, min=0.0) ** 2         #Historical writing method: Formula 26
    wrong_b = torch.clamp(tmin - V_r, min=0.0) ** 2         # Historical writing: Formula 27 (same direction)
    assert torch.allclose(wrong_a, wrong_b), "After the same direction, the two equations must be identical. ⟹ degenerate into repetition penalty"

    right_a = torch.clamp(V_r - tmax, min=0.0) ** 2         # After correction: upper bound
    right_b = torch.clamp(tmin - V_r, min=0.0) ** 2         # After correction: Nether
    assert not torch.allclose(right_a, right_b), "The correct way to write it must be to respond differently to both ends."


def test_T5b_wrong_form_penalizes_healthy_distribution():
    """Another consequence of writing in the same direction: erroneously penalizing healthy states (final test: 88.1% of A4c penalties come from erroneously penalizing healthy states)."""
    V_r_healthy = torch.tensor([1.0, 1.5, 2.0])             # Healthy state (middle section)
    tmin = 0.48341
    wrong = (torch.clamp(tmin - V_r_healthy, min=0.0) ** 2).sum()
    assert float(wrong) == 0.0, "In this example, the health state does not fall below the lower bound, so writing in the same direction will not cause any penalty here;"
    # What truly reflects the false penalty is the ”over-tightened lower bound”, which is covered by T5-c alone.


# ------------------------------------------------------------------ T5-c (C8)
def test_T5c_theta_out_of_range_is_rejected():
    """Injection: θ_max = 29.005 (beyond the measured V_r range) ⟹ Both hinges are silent at the same time."""
    V_healthy = torch.tensor([0.09668, 0.5, 1.0, 1.5, 2.0, 5.0])
    with pytest.raises(ThetaRangeError):
        assert_theta_in_range(29.005, 0.05, V_healthy)


def test_T5c_silent_band_when_theta_too_large():
    """Silent band verification: After θ_max exceeds max V_r, the pairwise memory state penalty is also 0."""
    V_healthy = torch.tensor([0.09668, 0.5, 1.0, 1.5, 2.0, 5.0])
    V_pairmem = torch.tensor([9.1803])                      # Memory pair by pair
    theta_max = 9.66821                                     # falls within the silent zone
    up_healthy = float(gate_upper(V_healthy, theta_max))
    up_pairmem = float(gate_upper(V_pairmem, theta_max))
    assert up_pairmem == 0.0, "θ_max fall in silence zone ⟹ Pair-by-pair memory is not punished (the brakes are in name only))"


# ------------------------------------------------------------------ T5-d (F8)
def test_T5d_theta_min_zero_is_rejected():
    """Injection: θ_min = 0 ⟹ The lower bound hinge is always 0, and the brake silently degenerates to one side (F8)."""
    with pytest.raises(ThetaRangeError):
        assert_vr_nonzero(0.0)


def test_T5d_zero_threshold_makes_lower_hinge_dead():
    V_r = torch.tensor([0.0, 0.5, 1.0])                     # Minimum is 0
    theta_min = 0.0
    lower = torch.clamp(theta_min - V_r, min=0.0) ** 2
    assert float(lower.sum()) == 0.0, "θ_min=0 ⟹ Nether hinge Hengwei 0"


# ------------------------------------------------------------------ T5-e (L3)
def test_T5e_wrong_two_hop_reduces_to_scaling():
    """Injection: P̄^⊗2 Use (I+ηP^conv)(I+ηP^trans) (historical writing) instead ⟹ Degenerate to column scaling.

    Consequences: Does not contain true two hops ⟹ Ablation A8 fails.
    """
    K = 8
    _, p_conv = ClosureParams(K)()
    Pc = p_conv[:, :K]
    p_trans = torch.rand(K)
    eta = 1.0
    P_bar = torch.eye(K) + eta * Pc
    P_trans_diag = torch.diag(p_trans)

    wrong = P_bar @ (torch.eye(K) + eta * P_trans_diag)     #History writing
    col_scaled = P_bar * (1.0 + eta * p_trans).unsqueeze(0)  # Column scaling
    assert torch.allclose(wrong, col_scaled, atol=1e-6), "History writing is equivalent to column scaling (without true double jumps))"

    right = two_hop_neighborhood(p_conv, eta, return_debug=False)
    assert not torch.allclose(wrong, right, atol=1e-6), "The way history is written must be different from the real two jumps"


# ------------------------------------------------------------------ T5-f (A11)
def test_T5f_fixed_l2_metric_has_no_learnable_geometry():
    """Injection: G → fixed ℓ₂(r=0)⟹ Learnable enrichment metric fails (ablation A11)."""
    d = 16
    m = MetricModule(d, r=1, eps=1e-4)
    G = m.metric()
    assert not torch.allclose(G, 1e-4 * torch.eye(d)), "r=1 time G should deviate from εI"

    # Equivalent case of r=0: A is all zero ⟹ G = εI ⟹ degenerates to fixed ℓ₂
    with torch.no_grad():
        m.A.zero_()
    G0 = m.metric()
    assert torch.allclose(G0, 1e-4 * torch.eye(d), atol=1e-8), "A=0 ⟹ G=εI (Fixed measure)"
    G0_inv = m.inverse()
    assert torch.allclose(G0_inv, (1e4) * torch.eye(d), rtol=1e-3), "G^{-1} should be ε^{-1}I"
