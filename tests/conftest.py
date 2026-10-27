"""Test Fixtures: Synthetic Data Generation and Model Construction.

Corresponding to the batch dimension convention in §0.2 of ”Module Design Document - Before Code Implementation.md”:
  Image level (b,·) | Pairing level (b,P,·)
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrp import HyperParams, VRPCatModel  # noqa: E402
from vrp.enrich import build_c_dagger, precompute_M_inv, top1_indicators, build_groups_and_weights  # noqa: E402


@pytest.fixture
def hp_tiny() -> HyperParams:
    """Small size: only used for runthrough contracts and invariants (CPU fast)."""
    return HyperParams(D=32, K=8, N_OBJ=6, R_LOWRANK=4, L_FUSE=2)


@pytest.fixture
def hp_spec() -> HyperParams:
    """Final draft §3.9 Specification size: used for parameter counting acceptance (G4/G7)."""
    return HyperParams()


def make_batch(hp: HyperParams, b: int = 2, N: int = 6, P: int = 5, d_in: int = 16, seed: int = 0):
    """Generate a batch of synthetic inputs that conform to the conventions of §0.2.

    Return dict: x (modal features), masks (layer neighborhood indications), s_idx, o_idx, y_label, y_onehot
    """
    g = torch.Generator().manual_seed(seed)
    x = {m: torch.randn(b, N, d_in, generator=g) for m in ("v", "t", "s")}

    masks = torch.zeros(b, hp.M_LEVELS, N)
    masks[:, 0, :] = 1.0                                    # Layer 0: all objects
    masks[:, 1, :] = (torch.rand(b, N, generator=g) > 0.35).float()
    masks[:, 2, :] = 1.0                                    # Layer 2: All objects
    # Ensure at least one neighbor in each layer (c_n > 0)
    masks[:, 1, 0] = 1.0

    s_idx = torch.randint(0, N, (b, P), generator=g)
    o_idx = torch.randint(0, N, (b, P), generator=g)
    # Exclude self-loops and ensure that the subject and object are different
    same = s_idx == o_idx
    o_idx = torch.where(same, (o_idx + 1) % N, o_idx)

    y_label = torch.randint(0, hp.K, (b, P), generator=g)
    y_onehot = torch.nn.functional.one_hot(y_label, hp.K).float()
    return {
        "x": x, "masks": masks, "s_idx": s_idx, "o_idx": o_idx,
        "y_label": y_label, "y_onehot": y_onehot,
        "b": b, "N": N, "P": P, "d_in": d_in,
    }


def build_model(hp: HyperParams, d_in: int = 16, seed: int = 0) -> VRPCatModel:
    """Construct the model and register all frozen quantities (M2.3 / M2.5) so that forward can be called directly."""
    torch.manual_seed(seed)
    model = VRPCatModel(hp, {"v": d_in, "t": d_in, "s": d_in})

    K, d = hp.K, hp.D
    g = torch.Generator().manual_seed(seed + 1)
    v_txt = torch.randn(K, d, generator=g)
    v_inv = torch.randn(K, d, generator=g)
    M_inv = precompute_M_inv(v_txt, v_inv)
    J = top1_indicators(M_inv)
    c_dag = build_c_dagger(M_inv, v_inv, J)

    # Group division: all asymmetric by default (contract testing does not rely on real grouping semantics)
    rho = torch.eye(K) * 0.9
    mask_non_sym, mask_sym, mask_ambig, w = build_groups_and_weights(rho)
    if float(mask_non_sym.sum()) == 0:
        # Bottom line: at least one asymmetry. ⚠️ Must be removed from the sym group synchronously, otherwise the mutual exclusion of the three groups will be destroyed
        mask_non_sym[0] = 1.0
        mask_sym[0] = 0.0
        mask_ambig[0] = 0.0
    assert torch.allclose(mask_non_sym + mask_sym + mask_ambig, torch.ones(K)), \
        "The three groups must be mutually exclusive and complete"
    w = w.clamp(0.05, 1.0)

    model.register_frozen(M_inv, J, c_dag, w, mask_sym, mask_non_sym, mask_ambig)
    model.calibrate_theta(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    return model


@pytest.fixture
def batch_tiny(hp_tiny):
    return make_batch(hp_tiny)


@pytest.fixture
def model_tiny(hp_tiny):
    return build_model(hp_tiny)
