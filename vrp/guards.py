"""M0.2 Shape/Type Guards and M0.4 Assertions and Diagnostics Basics (Module Design Document §2 S0)

M0.4 carries all the hard premise assertions and is the landing place of H1/H3.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .config import HyperParams


# ============================================================ M0.2 Shape/Type Guard

class ShapeError(AssertionError):
    pass


def expect(tensor: torch.Tensor, shape_spec: str, name: str) -> None:
    """Verify shapes by symbol skeleton.

    shape_spec convention (Module Design Document §0.2):
      ”b” = image-level batch dimension, variable
      ”P” = paired dimension, variable
      ”d”/”K”/”M”/”N” = named dimension, must be equal to the corresponding constant in hp (provided by resolve_dims)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ShapeError(f"{name}: Expectation Tensor, What you get {type(tensor)}")
    spec = shape_spec.split(",")
    if tensor.dim() != len(spec):
        raise ShapeError(f"{name}: Expectation {len(spec)} dimension ({shape_spec}), What you get {tensor.dim()} dimension {tuple(tensor.shape)}")


def expect_named(tensor: torch.Tensor, ref: torch.Tensor, prefix: str, name: str) -> None:
    """Verify that the first prefix dimensions are consistent with ref (used for alignment checks between pair-level tensors)."""
    if tensor.dim() < prefix or ref.dim() < prefix:
        raise ShapeError(f"{name}: Insufficient dimensions to align {prefix} dimension")
    if tuple(tensor.shape[:prefix]) != tuple(ref.shape[:prefix]):
        raise ShapeError(f"{name}: before {prefix} Dimension is inconsistent with reference {tuple(tensor.shape[:prefix])} vs {tuple(ref.shape[:prefix])}")


def expect_simplex(tensor: torch.Tensor, name: str, axis: int = -1, tol: float = 1e-5) -> None:
    """Assert that the probability simplex Δ^K: is nonnegative and sums to 1 along axis."""
    if not torch.isfinite(tensor).all():
        raise ShapeError(f"{name}: Contains non-finite values")
    if (tensor < -tol).any():
        raise ShapeError(f"{name}: There are negative elements {tensor.min().item():.3e} < 0")
    s = tensor.sum(dim=axis)
    if not torch.allclose(s, torch.ones_like(s), atol=tol):
        raise ShapeError(f"{name}: along axis={axis} summation deviation 1 (max Deviation {(s - 1).abs().max().item():.3e}）")


def expect_psd(tensor: torch.Tensor, name: str, tol: float = 1e-6) -> None:
    """Assert symmetric positive definite (for G / G^{-1})."""
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ShapeError(f"{name}: Expect a phalanx, get what you get {tuple(tensor.shape)}")
    if not torch.allclose(tensor, tensor.t(), atol=1e-8):
        raise ShapeError(f"{name}: asymmetric")
    ev = torch.linalg.eigvalsh(tensor.detach())
    if ev.min().item() <= tol:
        raise ShapeError(f"{name}: minimum eigenvalue {ev.min().item():.3e} <= {tol}")


def expect_probability(tensor: torch.Tensor, name: str) -> None:
    if (tensor < 0).any() or (tensor > 1).any() or not torch.isfinite(tensor).all():
        raise ShapeError(f"{name}: is not a legal probability (must fall within [0,1])")


def guard_image_level(tensor: torch.Tensor, name: str, n_batch: int) -> None:
    """Gatekeeper to disallow implicit broadcasting: image-level tensors must be exactly (b, ...) with first dimension == n_batch.

    Module Design Document §0.2: Any operation of (b,P,d) and (b,d) must explicitly unsqueeze(1),
    This function is used to catch the error of ”forgetting to unsqueeze but being covered by broadcast” in single testing.
    """
    if tensor.dim() < 1 or tensor.shape[0] != n_batch:
        raise ShapeError(f"{name}: Desired image level (b={n_batch}, ...), What you get {tuple(tensor.shape)}")


# ============================================================ M0.4 Hard assertion

class HingePositionError(AssertionError):
    """H1: hinge falls on V_global."""


class ThetaRangeError(AssertionError):
    """H3: The threshold exceeds the measured V_r range."""


class PairingProtocolError(AssertionError):
    """Protocol self-test failed (magnitude or direction)."""


class ArtifactError(AssertionError):
    """When ΔB is injected, the accompanying quantity W_health drifts >1%, and this batch of data will be invalidated."""


def assert_hinge_on_Vr(hinge_input_name: str) -> None:
    """H1 (final draft §3.5(a) analytical proof + C1): The principal hinge must fall on the within-group variance V_r.

    V_global contains the between-group term B. The healthy state is always greater than the degraded state. ⟹ The upper bound hinge will inevitably penalize health and let go of degradation.
    """
    if hinge_input_name != "V_r":
        raise HingePositionError(
            f"H1 Violation: Brake hinge Enter as {hinge_input_name!r}, Must be 'V_r'."
            " (final draft §3.5(a): V_global = B + W, Healthy state V_global > Degenerate state, upper bound hinge Must be invalid)"
        )


def assert_theta_in_range(
    theta_max: float,
    theta_min: float,
    V_r_healthy: torch.Tensor,
    *,
    strict_min_ratio: float = 0.5,
) -> None:
    """H3 three assertions in one (final draft §3.5 Protection):

      (1) F8: theta_min > 0 —— Otherwise, the lower bound hinge is always 0, and the brake silently degenerates to one side.
      (2) C8 anti-silence: theta_max < max_r{V_r^healthy} - disable Q99 instead of max (measured silent bandwidth 7.46)
      (3) Prevent excessive tightening: theta_min < strict_min_ratio * min_r{V_r^healthy}
    """
    if V_r_healthy.numel() == 0:
        raise ThetaRangeError("V_r_healthy is empty, the threshold cannot be calibrated")
    v_max = float(V_r_healthy.max().item())
    v_min = float(V_r_healthy.min().item())

    if not (theta_min > 0):
        raise ThetaRangeError(f"(F8) theta_min={theta_min} Must > 0, Otherwise, the brake will silently degenerate into one-sided")
    if not (theta_max < v_max):
        raise ThetaRangeError(
            f"(C8) theta_max={theta_max} Not less than healthy state V_r maximum value {v_max:.5f}。"
            "two hinge will be lost in silence at the same time/All indicators are normal."
        )
    if not (theta_min < strict_min_ratio * v_min):
        raise ThetaRangeError(
            f"(Over tightening) theta_min={theta_min} Not significantly smaller than the healthy state V_r lower end {v_min:.5f}"
            f" (request < {strict_min_ratio}×).The brakes will degrade to『punish everything』."
        )


def assert_pairing_protocol(delta_B: float, *, tol: float = 1e-12, tol_dir: float = 0.0) -> None:
    """Final draft §3.5 Protocol self-test: amplitude + direction biconditional (lack of direction condition will miss systematic sign errors).

    Actual measurement: Both configurations |ΔB| are < 1e-12, and both assertions pass, but one of them has the wrong symbol direction (final draft table).
    """
    if abs(delta_B) >= tol:
        raise PairingProtocolError(f"Amplitude condition failed:|ΔB|={abs(delta_B):.3e} >= {tol:g}")
    if delta_B > tol_dir:
        raise PairingProtocolError(
            f"Orientation condition failed:ΔB={delta_B:+.3e} > 0, Healthy state B is greater than the degenerate state,"
            "It shows that the two states use different c_r (Protocol flaws, even if |ΔB| No matter how small you are, you must refuse)"
        )


def warn_if_artifact(W_health_series: Sequence[float], drift_max: float = 0.01) -> bool:
    """Mechanical criterion for artifacts: If the accompanying quantity W_health drifts across gears >1% when ΔB is injected, the batch of data will be invalidated.

    Measured discrimination: Path A (translation group mean + scaling noise) drifts 100% → invalid;
                Path B (shifted group mean only) drift 0% → valid.
    Returning True means the data is valid.
    """
    ws = [float(w) for w in W_health_series]
    if len(ws) < 2:
        return True
    lo, hi = min(ws), max(ws)
    denom = max(hi, 1e-30)
    drift = abs(hi - lo) / denom
    if drift > drift_max:
        raise ArtifactError(
            f"Artifacts: W_health drift {drift * 100:.2f}% > {drift_max * 100:.0f}%, This batch of data is invalid."
            " (Changes in failure rates shall not be attributed to ΔB)"
        )
    return True


def assert_vr_nonzero(theta_min: float) -> None:
    """F8: theta_min > 0 assertion (final draft table (2) (e)).

    When θ_min = 0, the lower bound hinge is always 0, and the brake silently degenerates to one side.
    The loss curve is completely normal and the code does not report errors - it must be guaranteed by assertions rather than default values.
    """
    if not (theta_min > 0):
        raise ThetaRangeError(f"F8 Violation: theta_min={theta_min} must be strict > 0")


def hinge_trigger_rate(
    V_r: torch.Tensor, theta_max: float, theta_min: float
) -> tuple[float, float]:
    """Brake triggering rate (final draft §3.5 Protection ③ diagnostic value must be reported).

    Any value of 0 means the brakes are disabled on this instance.
    """
    upper = float((V_r > theta_max).float().mean().item())
    lower = float((V_r < theta_min).float().mean().item())
    return upper, lower
