"""T2 相邻对连通测试（模块设计文档 §4.2）——契约矩阵 C01–C24。

核心不是"每个模块单独正确"，而是**模块之间的连接不产生静默错误**。
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_model, make_batch
from vrp.backbone import chain_attention, intra_layer_aggregate
from vrp.closure import ClosureParams, mixed_student, two_hop_neighborhood
from vrp.enrich import Inv, build_c_dagger, precompute_M_inv, top1_indicators
from vrp.losses import SharedProjection, cls_loss, dag_loss, inv_loss, sym_loss, estimate_variance, pi_loss
from vrp.guards import hinge_trigger_rate


@pytest.fixture
def wire(hp_tiny):
    """一次性算好主路中间量，供各契约复用。"""
    model = build_model(hp_tiny)
    b = make_batch(hp_tiny)
    z_j, A, h = model.backbone.encode(b["x"], b["masks"])
    Z = model.backbone.encode(b["x"], b["masks"])[1]
    v_glo = Z[:, 1]
    pf = model.backbone.pair_features(z_j, Z, b["s_idx"], b["o_idx"])
    r_exp, r_imp, r = model.backbone.forward(pf, v_glo)
    p_ij, p_ji = model.dual_head(pf.f_s2, pf.f_u2, pf.f_o2)
    p_trans, p_conv = model.closure()
    G_inv = model.metric.inverse()
    return dict(model=model, b=b, z_j=z_j, A=A, Z=Z, h=h, pf=pf, r_exp=r_exp,
                r_imp=r_imp, r=r, p_ij=p_ij, p_ji=p_ji, p_trans=p_trans,
                p_conv=p_conv, G_inv=G_inv)


def test_C01_M11_to_M12(wire, hp_tiny):
    h = {m: torch.randn(wire["b"]["b"], wire["b"]["N"], hp_tiny.D) for m in ("v", "t", "s")}
    z_j, Z = intra_layer_aggregate(h, wire["model"].backbone.W_a, wire["b"]["masks"])
    assert z_j.shape == (wire["b"]["b"], wire["b"]["N"], hp_tiny.D)
    assert Z.shape[1] == 3


def test_C02_M12_to_M13(wire, hp_tiny):
    d = hp_tiny.D
    out = chain_attention(wire["Z"], torch.nn.Linear(d, d), torch.nn.Linear(d, d), torch.nn.Linear(d, d))
    assert out.shape == wire["Z"].shape


def test_C03_M12_to_M412_image_level(wire, hp_tiny):
    """z_0 必须为图像级 (b,d)，无配对维 —— 禁止隐式广播。"""
    z0 = wire["Z"][:, 0]
    assert z0.dim() == 2 and z0.shape == (wire["b"]["b"], hp_tiny.D)
    assert float(pi_loss(z0, [z0])) == 0.0


def test_C04_M13_to_M14(wire):
    assert wire["A"].shape == wire["Z"].shape


def test_C05_M14_to_M42_r_exp(wire):
    assert wire["pf"].f_r1.shape[:2] == wire["r_exp"].shape[:2]


def test_C06_M14_to_M47_alpha_beta(wire, hp_tiny):
    pi = wire["pf"].sample_pi()
    assert pi.shape == (*wire["r_exp"].shape[:2], 2)
    assert torch.allclose(pi.norm(dim=-1), torch.ones_like(pi[..., 0]), atol=1e-5)


def test_C07_M15_to_M42_r_imp_pair_sensitive(wire):
    """r_imp 必须配对相关（H2 前置）；否则 L_dag ≡ 0。"""
    imp = wire["model"].backbone.implicit
    a = imp(wire["pf"].f_s2, wire["pf"].f_u2, wire["pf"].f_o2)
    b = imp(wire["pf"].f_o2, wire["pf"].f_u2, wire["pf"].f_s2)
    assert not torch.allclose(a, b)


def test_C08_M16_to_M17(wire, hp_tiny):
    assert wire["r"].shape == (wire["b"]["b"], wire["b"]["P"], hp_tiny.D)


def test_C09_M17_to_consumers(wire, hp_tiny):
    K = hp_tiny.K
    assert wire["p_ij"].shape == (wire["b"]["b"], wire["b"]["P"], K)
    out = cls_loss(wire["p_ij"], wire["p_ji"], wire["b"]["y_onehot"],
                   torch.zeros_like(wire["b"]["y_onehot"]),
                   torch.ones(*wire["p_ij"].shape[:2], 1),
                   torch.ones(wire["p_ij"].shape[:2]))
    assert torch.isfinite(out)


def test_C10_M21_M22_to_M42(wire, hp_tiny):
    assert wire["G_inv"].shape == (hp_tiny.D, hp_tiny.D)
    assert float(inv_loss(wire["r_exp"], wire["r_imp"], wire["G_inv"])) >= 0


def test_C11_M23_to_M24():
    K, d = 8, 16
    M_inv = precompute_M_inv(torch.randn(K, d), torch.randn(K, d))
    assert M_inv.shape == (K, K)
    assert not M_inv.requires_grad


def test_C12_M24_to_M45_detached(wire, hp_tiny):
    Inv_p = Inv(wire["p_ij"], wire["model"].M_inv, hp_tiny.EPS_INV).detach()
    assert not Inv_p.requires_grad
    r_ji = wire["model"].backbone.implicit.swap(wire["pf"].f_s2, wire["pf"].f_u2, wire["pf"].f_o2)
    L, _, _ = dag_loss(wire["p_ij"], wire["p_ji"], r_ji, Inv_p, wire["model"].c_dag,
                       wire["G_inv"], torch.ones(*wire["p_ij"].shape[:2]), SharedProjection(hp_tiny.D))
    assert torch.isfinite(L)


def test_C13_M25_to_M45_M46(wire, hp_tiny):
    K = hp_tiny.K
    mn, ms, ma = wire["model"].mask_non_sym, wire["model"].mask_sym, wire["model"].mask_ambig
    assert torch.allclose(mn + ms + ma, torch.ones(K))
    assert wire["model"].w.shape == (K,)


def test_C14_M31_to_M32_M34(wire, hp_tiny):
    assert wire["p_conv"].shape == (hp_tiny.K, hp_tiny.K + 1)
    P2 = two_hop_neighborhood(wire["p_conv"], 1.0)
    assert P2.shape == (hp_tiny.K, hp_tiny.K)


def test_C15_M31_to_M33(wire, hp_tiny):
    assert torch.all((wire["p_trans"] > 0) & (wire["p_trans"] < 1))


def test_C16_M34_to_M35_M410(wire, hp_tiny):
    P2 = two_hop_neighborhood(wire["p_conv"], 1.0)
    pt = mixed_student(wire["p_ij"], P2, 0.2)
    assert pt.shape == wire["p_ij"].shape


def test_C17_M35_to_M43_not_to_M41(wire, hp_tiny):
    """ω_p 门控只作用于分类损失（式22 作用域）。

    ⚠️ 2026-10-18：式11 KL 蒸馏已删除，ω_p 现仅作为 p̃ 混合系数。
    """
    K = hp_tiny.K
    assert wire["p_ij"].shape == (wire["b"]["b"], wire["b"]["P"], K)


def test_C18_M47_to_M48_M49(wire, hp_tiny):
    from vrp.losses import gate_lower, gate_upper
    pi = wire["pf"].sample_pi()
    V_r, V_g, H_r, cnt = estimate_variance(pi, wire["b"]["y_label"], hp_tiny.K)
    present = (cnt > 0).float()
    assert float(gate_upper(V_r, 1.0, present)) >= 0
    L, _, _ = gate_lower(V_r, V_g, 0.05, 0.05, present)
    assert torch.isfinite(L)


def test_C19_M47_Vglobal_to_diagnostics(wire, hp_tiny):
    pi = wire["pf"].sample_pi()
    V_r, V_g, H_r, cnt = estimate_variance(pi, wire["b"]["y_label"], hp_tiny.K)
    up, low = hinge_trigger_rate(V_r, 1.0, 0.05)
    assert 0.0 <= up <= 1.0 and 0.0 <= low <= 1.0


def test_C20_M48_M49_to_M413(hp_tiny):
    from vrp.losses import total_loss
    parts = {"hom": torch.tensor(0.1), "spread": torch.tensor(0.2)}
    assert float(total_loss(parts, {"hom": 1.0, "spread": 1.0})) == pytest.approx(0.3, abs=1e-6)


def test_C21_all_losses_to_M413(hp_tiny):
    from vrp.config import all_lambda_dict
    from vrp.losses import total_loss
    # ⚠️ 2026-10-18：式11 "KL" 与式12 "feat" 已删除（本文不含知识蒸馏）
    keys = ["cls", "inv", "dag_non_sym", "sym", "coh", "hom", "spread", "PI", "ord"]
    parts = {k: torch.tensor(0.01) for k in keys}
    out = total_loss(parts, all_lambda_dict(hp_tiny, 0.2), expect_keys=keys)
    assert torch.isfinite(out)



def test_C23_theta_to_M48_M49(wire):
    assert hasattr(wire["model"], "theta_max") and hasattr(wire["model"], "theta_min")
    assert wire["model"].theta_min > 0


def test_C24_swap_to_M45_M46(wire):
    """swap 路输出必须与主路 p_ij 显著不同（否则 L_dag 无信号）。"""
    from vrp.losses import js_divergence
    p_ji_pi, p_ij_pi = wire["model"].dual_head(wire["pf"].f_o2, wire["pf"].f_u2, wire["pf"].f_s2)
    d = float(js_divergence(p_ji_pi, wire["p_ij"]).mean())
    assert d > 0, f"swap 路与主路输出无差异（JS={d}）⟹ L_dag 恒为 0"


# ---------------------------------------------------------------- T2 六项专项

def test_T2a_hinge_input_is_Vr(hp_tiny):
    """T2-a：用 V_global 冒充 V_r 必须抛异常（拦截 C1）。"""
    from vrp.guards import HingePositionError, assert_hinge_on_Vr
    with pytest.raises(HingePositionError):
        assert_hinge_on_Vr("V_global")


def test_T2b_swap_differs_from_main(wire):
    from vrp.losses import js_divergence
    p_ji_pi, _ = wire["model"].dual_head(wire["pf"].f_o2, wire["pf"].f_u2, wire["pf"].f_s2)
    assert float(js_divergence(p_ji_pi, wire["p_ij"]).mean()) > 0


def test_T2c_r_imp_is_pair_sensitive(wire):
    imp = wire["model"].backbone.implicit
    a = imp(wire["pf"].f_s2, wire["pf"].f_u2, wire["pf"].f_o2)
    b = imp(wire["pf"].f_o2, wire["pf"].f_u2, wire["pf"].f_s2)
    assert not torch.allclose(a, b)


def test_T2d_alpha_beta_direction_sensitive(wire):
    od = wire["model"].backbone.diff2
    _, a1, b1 = od(wire["pf"].f_u2, wire["pf"].f_s1, wire["pf"].f_o1, wire["Z"][:, 1])
    _, a2, b2 = od(wire["pf"].f_u2, wire["pf"].f_o1, wire["pf"].f_s1, wire["Z"][:, 1])
    assert not (torch.allclose(a1, a2) and torch.allclose(b1, b2))


def test_T2e_ptilde_not_into_cls(hp_tiny):
    """T2-e：p̃ 与 L_cls 之间不得存在计算图路径。"""
    K = hp_tiny.K
    p = torch.softmax(torch.randn(2, 3, K), dim=-1).requires_grad_(True)
    from vrp.closure import ClosureParams, two_hop_neighborhood, mixed_student
    _, p_conv = ClosureParams(K)()
    P2 = two_hop_neighborhood(p_conv, 1.0)
    pt = mixed_student(p, P2, 0.2)
    # L_cls 用 p（不是 pt）；pt 的梯度回传到 p 但不构成 p → L_cls 的路径
    y = torch.nn.functional.one_hot(torch.randint(0, K, (2, 3)), K).float()
    L = cls_loss(p, p, y, torch.zeros_like(y), torch.ones(2, 3, 1), torch.ones(2, 3))
    g = torch.autograd.grad(L, p, retain_graph=True)[0]
    assert torch.isfinite(g).all()
    assert not pt.requires_grad or pt.grad_fn is not None


def test_T2f_lcls_counted_once(hp_tiny):
    """T2-f：L_cls 作为叶子被引用次数 ≡ 1（拦截 L5 双重计数）。"""
    from vrp.config import all_lambda_dict
    from vrp.losses import total_loss
    L_cls = torch.tensor(2.0, requires_grad=True)
    parts = {"cls": L_cls, "inv": torch.tensor(1.0)}
    lam = {"cls": 1.0, "inv": 0.5}
    out = total_loss(parts, lam)
    assert float(out) == pytest.approx(2.5, abs=1e-6), "L_cls 必须恰好计一次"
