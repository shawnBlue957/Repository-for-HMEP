"""范畴建模的视觉关系预测 —— 参考实现（对应《模块设计文档-代码实现前.md》）。

模块 ↔ 设计文档编号：
  config      M0.1
  guards      M0.2 / M0.4
  backbone    M1.1–M1.7
  enrich      M2.1–M2.5
  closure     M3.1–M3.6
  losses      M4.1–M4.13
  diagnostics M5.1–M5.6
  model       M6.2 / M6.5
"""
from .config import HyperParams, get_scheduled, all_lambda_dict, PENDING_SCAN_GRID
from .guards import (
    assert_hinge_on_Vr,
    assert_pairing_protocol,
    assert_theta_in_range,
    assert_vr_nonzero,
    expect,
    expect_psd,
    expect_simplex,
    guard_image_level,
    hinge_trigger_rate,
    warn_if_artifact,
)
from .backbone import Backbone, DualHead, PairFeatures
from .enrich import MetricModule, Inv, Inv_detached, build_c_dagger, precompute_M_inv, top1_indicators
from .closure import ClosureParams, mixed_student, two_hop_neighborhood
from .losses import (
    SharedProjection,
    cls_loss,
    dag_loss,
    estimate_variance,
    gate_lower,
    gate_upper,
    inv_loss,
    sym_loss,
)
from .model import VRPCatModel, count_parameters, strip_for_inference

__all__ = [
    "HyperParams", "get_scheduled", "all_lambda_dict", "PENDING_SCAN_GRID",
    "assert_hinge_on_Vr", "assert_pairing_protocol", "assert_theta_in_range",
    "assert_vr_nonzero", "expect", "expect_psd", "expect_simplex",
    "guard_image_level", "hinge_trigger_rate", "warn_if_artifact",
    "Backbone", "DualHead", "PairFeatures",
    "MetricModule", "Inv", "Inv_detached", "build_c_dagger", "precompute_M_inv",
    "top1_indicators", "ClosureParams", "mixed_student", "two_hop_neighborhood",
    "SharedProjection", "cls_loss", "dag_loss", "estimate_variance",
    "gate_lower", "gate_upper", "inv_loss", "sym_loss",
    "VRPCatModel", "count_parameters", "strip_for_inference",
]
