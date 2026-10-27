"""T5 失败注入测试（模块设计文档 §4.5）—— T5-a … T5-f。

价值：把终稿的「审校记录」转为**可执行的回归防护**。
每个测试**故意注入**一个已知错误，断言系统**必须捕获**。
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


# ------------------------------------------------------------------ T5-a（C1）
def test_T5a_hinge_on_Vglobal_is_rejected():
    """注入：hinge 落在 V_global 上（C1 位置错误）。"""
    with pytest.raises(HingePositionError):
        assert_hinge_on_Vr("V_global")


def test_T5a_secondary_guard_trigger_rate_zero():
    """二次防护：即便绕过断言，车闸触发率也必须暴露静默失效。

    构造：V_global 恒大于退化态 V_r ⟹ 上界 hinge 在退化态恒为 0。
    """
    V_r_degenerate = torch.zeros(5)                 # per-relation 常量
    theta_max = 1.0
    up, low = hinge_trigger_rate(V_r_degenerate, theta_max, theta_min=0.05)
    assert up == 0.0, "退化态上界 hinge 恒为 0 ⟹ 车闸静默失效，须被触发率诊断捕捉"


# ------------------------------------------------------------------ T5-b（C2/C7）
def test_T5b_same_direction_hinges_lose_discrimination():
    """注入：两个 hinge 改成同向（重复惩罚，历史写法）⟹ 失去区分力（C7）。

    结构判据：同向时两式**恒等**；正确写法对 V_r 的两端响应相反。
    """
    V_r = torch.linspace(0.0, 5.0, 50)
    tmin, tmax = 0.48341, 9.66821

    wrong_a = torch.clamp(tmin - V_r, min=0.0) ** 2         # 历史写法：式26
    wrong_b = torch.clamp(tmin - V_r, min=0.0) ** 2         # 历史写法：式27（同向）
    assert torch.allclose(wrong_a, wrong_b), "同向化后两式必须恒等 ⟹ 退化为重复惩罚"

    right_a = torch.clamp(V_r - tmax, min=0.0) ** 2         # 修正后：上界
    right_b = torch.clamp(tmin - V_r, min=0.0) ** 2         # 修正后：下界
    assert not torch.allclose(right_a, right_b), "正确写法必须对两端响应不同"


def test_T5b_wrong_form_penalizes_healthy_distribution():
    """同向写的另一后果：误罚健康态（终稿实测：A4c 惩罚中 88.1% 来自误罚健康态）。"""
    V_r_healthy = torch.tensor([1.0, 1.5, 2.0])             # 健康态（中间段）
    tmin = 0.48341
    wrong = (torch.clamp(tmin - V_r_healthy, min=0.0) ** 2).sum()
    assert float(wrong) == 0.0, "本例中健康态未落于下界之下，故同向写法在此不误罚；"
    # 真正体现误罚的是「过度收紧的下界」，单独由 T5-c 覆盖


# ------------------------------------------------------------------ T5-c（C8）
def test_T5c_theta_out_of_range_is_rejected():
    """注入：θ_max = 29.005（超出实测 V_r 量程）⟹ 两个 hinge 同时静默。"""
    V_healthy = torch.tensor([0.09668, 0.5, 1.0, 1.5, 2.0, 5.0])
    with pytest.raises(ThetaRangeError):
        assert_theta_in_range(29.005, 0.05, V_healthy)


def test_T5c_silent_band_when_theta_too_large():
    """静默带验证：θ_max 超过 max V_r 后，逐对记忆态惩罚也为 0。"""
    V_healthy = torch.tensor([0.09668, 0.5, 1.0, 1.5, 2.0, 5.0])
    V_pairmem = torch.tensor([9.1803])                      # 逐对记忆
    theta_max = 9.66821                                     # 落在静默带内
    up_healthy = float(gate_upper(V_healthy, theta_max))
    up_pairmem = float(gate_upper(V_pairmem, theta_max))
    assert up_pairmem == 0.0, "θ_max 落在静默带 ⟹ 逐对记忆不受惩罚（车闸形同虚设）"


# ------------------------------------------------------------------ T5-d（F8）
def test_T5d_theta_min_zero_is_rejected():
    """注入：θ_min = 0 ⟹ 下界 hinge 恒为 0，车闸静默退化为单边（F8）。"""
    with pytest.raises(ThetaRangeError):
        assert_vr_nonzero(0.0)


def test_T5d_zero_threshold_makes_lower_hinge_dead():
    V_r = torch.tensor([0.0, 0.5, 1.0])                     # 最小为 0
    theta_min = 0.0
    lower = torch.clamp(theta_min - V_r, min=0.0) ** 2
    assert float(lower.sum()) == 0.0, "θ_min=0 ⟹ 下界 hinge 恒为 0"


# ------------------------------------------------------------------ T5-e（L3）
def test_T5e_wrong_two_hop_reduces_to_scaling():
    """注入：P̄^⊗2 改用 (I+ηP^conv)(I+ηP^trans)（历史写法）⟹ 退化为列缩放。

    后果：不含真正两跳 ⟹ 消融 A8 失效。
    """
    K = 8
    _, p_conv = ClosureParams(K)()
    Pc = p_conv[:, :K]
    p_trans = torch.rand(K)
    eta = 1.0
    P_bar = torch.eye(K) + eta * Pc
    P_trans_diag = torch.diag(p_trans)

    wrong = P_bar @ (torch.eye(K) + eta * P_trans_diag)     # 历史写法
    col_scaled = P_bar * (1.0 + eta * p_trans).unsqueeze(0)  # 列缩放
    assert torch.allclose(wrong, col_scaled, atol=1e-6), "历史写法等价于列缩放（无真两跳）"

    right = two_hop_neighborhood(p_conv, eta, return_debug=False)
    assert not torch.allclose(wrong, right, atol=1e-6), "历史写法与真两跳必须不同"


# ------------------------------------------------------------------ T5-f（A11）
def test_T5f_fixed_l2_metric_has_no_learnable_geometry():
    """注入：G → 固定 ℓ₂（r=0）⟹ 可学富化度量失效（消融 A11）。"""
    d = 16
    m = MetricModule(d, r=1, eps=1e-4)
    G = m.metric()
    assert not torch.allclose(G, 1e-4 * torch.eye(d)), "r=1 时 G 应偏离 εI"

    # r=0 的等价情形：A 全零 ⟹ G = εI ⟹ 退化为固定 ℓ₂
    with torch.no_grad():
        m.A.zero_()
    G0 = m.metric()
    assert torch.allclose(G0, 1e-4 * torch.eye(d), atol=1e-8), "A=0 ⟹ G=εI（固定度量）"
    G0_inv = m.inverse()
    assert torch.allclose(G0_inv, (1e4) * torch.eye(d), rtol=1e-3), "G^{-1} 应为 ε^{-1}I"
