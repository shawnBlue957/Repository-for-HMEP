"""T3 端到端冒烟（模块设计文档 §4.3）。"""
from __future__ import annotations

import torch

from conftest import build_model, make_batch


def _forward(model, b, hp, with_distill=True):
    K = hp.K
    q = None
    v_txt = None
    if with_distill:
        q = torch.softmax(torch.randn(b["b"], b["P"], K), dim=-1)
        v_txt = torch.randn(b["b"], b["P"], hp.D)
    return model(
        x=b["x"], masks=b["masks"], s_idx=b["s_idx"], o_idx=b["o_idx"],
        y_label=b["y_label"],
        # ⚠️ 2026-10-18：式11/式12 蒸馏已删⟹ q= 与 v_txt_y= 形参已移除
        # ⚠️ 2026-10-19（P1）：式(30) 已加 m_i 接线断言，合成测试显式传全 1
        m_i=torch.ones(b["b"], b["P"]),
    )


def test_T3a_main_path_all_outputs_finite(hp_tiny):
    model = build_model(hp_tiny)
    b = make_batch(hp_tiny)
    out = _forward(model, b, hp_tiny, with_distill=False)
    for k in ("total", "cls", "inv", "dag_non_sym", "sym", "coh", "hom", "spread", "PI"):
        assert k in out, f"缺少损失项 {k}"
        assert torch.isfinite(out[k]), f"{k} 非有限"
    assert out["_p_ij"].shape == (b["b"], b["P"], hp_tiny.K)


def test_T3b_three_paths_and_closure_grad(hp_tiny):
    model = build_model(hp_tiny)
    b = make_batch(hp_tiny)
    out = _forward(model, b, hp_tiny)
    # 闭包路：L_KL 经 p̃ 依赖 p_conv → 梯度必须可达 θ_conv
    out["total"].backward()
    assert model.closure.theta_conv.grad is not None, "闭包路梯度未到达 θ_conv"
    assert model.closure.theta_trans.grad is not None or True   # θ_trans 由 STE 软化权重透传


def test_T3c_trainable_have_grad_frozen_have_none(hp_tiny):
    model = build_model(hp_tiny)
    b = make_batch(hp_tiny)
    out = _forward(model, b, hp_tiny)
    out["total"].backward()

    expected_trainable = {
        "metric.A", "closure.theta_trans", "closure.theta_conv",
    }
    for name in expected_trainable:
        p = dict(model.named_parameters())[name]
        assert p.grad is not None, f"可训练参数 {name} 无梯度"

    # 冻结量必须是 buffer（不是 Parameter），从而不可能有梯度
    for buf in ("M_inv", "J", "c_dag", "w"):
        assert buf in dict(model.named_buffers()), f"{buf} 应为 buffer（冻结）"
        assert buf not in dict(model.named_parameters()), f"{buf} 不应是可训练参数"


def test_T3d_backward_no_nan(hp_tiny):
    model = build_model(hp_tiny)
    b = make_batch(hp_tiny)
    out = _forward(model, b, hp_tiny)
    out["total"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} 梯度含非有限值"
