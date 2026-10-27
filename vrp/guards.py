"""M0.2 形状/类型守卫 与 M0.4 断言与诊断基础（模块设计文档 §2 S0）

M0.4 承载全部硬性前提断言，是 H1/H3 的落地处。
"""
from __future__ import annotations

from typing import Sequence

import torch

from .config import HyperParams


# ============================================================ M0.2 形状/类型守卫

class ShapeError(AssertionError):
    pass


def expect(tensor: torch.Tensor, shape_spec: str, name: str) -> None:
    """按符号骨架校验形状。

    shape_spec 约定（模块设计文档 §0.2）：
      "b"    = 图像级批维，可变
      "P"    = 配对维，可变
      "d"/"K"/"M"/"N" = 具名维，必须等于 hp 中对应常量（由 resolve_dims 提供）
    """
    if not isinstance(tensor, torch.Tensor):
        raise ShapeError(f"{name}: 期望 Tensor，实得 {type(tensor)}")
    spec = shape_spec.split(",")
    if tensor.dim() != len(spec):
        raise ShapeError(f"{name}: 期望 {len(spec)} 维（{shape_spec}），实得 {tensor.dim()} 维 {tuple(tensor.shape)}")


def expect_named(tensor: torch.Tensor, ref: torch.Tensor, prefix: str, name: str) -> None:
    """校验前 prefix 个维度与 ref 一致（用于配对级张量间的对齐检查）。"""
    if tensor.dim() < prefix or ref.dim() < prefix:
        raise ShapeError(f"{name}: 维度不足，无法对齐 {prefix} 维")
    if tuple(tensor.shape[:prefix]) != tuple(ref.shape[:prefix]):
        raise ShapeError(f"{name}: 前 {prefix} 维与参考不一致 {tuple(tensor.shape[:prefix])} vs {tuple(ref.shape[:prefix])}")


def expect_simplex(tensor: torch.Tensor, name: str, axis: int = -1, tol: float = 1e-5) -> None:
    """断言概率单纯形 Δ^K：非负且沿 axis 求和为 1。"""
    if not torch.isfinite(tensor).all():
        raise ShapeError(f"{name}: 含非有限值")
    if (tensor < -tol).any():
        raise ShapeError(f"{name}: 存在负元素 {tensor.min().item():.3e} < 0")
    s = tensor.sum(dim=axis)
    if not torch.allclose(s, torch.ones_like(s), atol=tol):
        raise ShapeError(f"{name}: 沿 axis={axis} 求和偏离 1（max 偏差 {(s - 1).abs().max().item():.3e}）")


def expect_psd(tensor: torch.Tensor, name: str, tol: float = 1e-6) -> None:
    """断言对称正定（用于 G / G^{-1}）。"""
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ShapeError(f"{name}: 期望方阵，实得 {tuple(tensor.shape)}")
    if not torch.allclose(tensor, tensor.t(), atol=1e-8):
        raise ShapeError(f"{name}: 非对称")
    ev = torch.linalg.eigvalsh(tensor.detach())
    if ev.min().item() <= tol:
        raise ShapeError(f"{name}: 最小特征值 {ev.min().item():.3e} <= {tol}")


def expect_probability(tensor: torch.Tensor, name: str) -> None:
    if (tensor < 0).any() or (tensor > 1).any() or not torch.isfinite(tensor).all():
        raise ShapeError(f"{name}: 不是合法概率（须落在 [0,1]）")


def guard_image_level(tensor: torch.Tensor, name: str, n_batch: int) -> None:
    """禁止隐式广播的守门人：图像级张量必须恰为 (b, ...) 且首维 == n_batch。

    模块设计文档 §0.2：任何 (b,P,d) 与 (b,d) 的运算必须显式 unsqueeze(1)，
    本函数用于在单测中捕捉"忘记 unsqueeze 却被广播掩盖"的错误。
    """
    if tensor.dim() < 1 or tensor.shape[0] != n_batch:
        raise ShapeError(f"{name}: 期望图像级 (b={n_batch}, ...)，实得 {tuple(tensor.shape)}")


# ============================================================ M0.4 硬性断言

class HingePositionError(AssertionError):
    """H1：hinge 落在了 V_global 上。"""


class ThetaRangeError(AssertionError):
    """H3：阈值超出实测 V_r 量程。"""


class PairingProtocolError(AssertionError):
    """协议自检失败（幅度或方向）。"""


class ArtifactError(AssertionError):
    """注入 ΔB 时伴随量 W_health 漂移 >1%，该批数据作废。"""


def assert_hinge_on_Vr(hinge_input_name: str) -> None:
    """H1（终稿 §3.5(a) 解析证明 + C1）：主 hinge 必须落在组内方差 V_r 上。

    V_global 含组间项 B，健康态恒大于退化态 ⟹ 上界 hinge 必误罚健康、放过退化。
    """
    if hinge_input_name != "V_r":
        raise HingePositionError(
            f"H1 违反：车闸 hinge 输入为 {hinge_input_name!r}，必须为 'V_r'。"
            "（终稿 §3.5(a)：V_global = B + W，健康态 V_global > 退化态，上界 hinge 必失效）"
        )


def assert_theta_in_range(
    theta_max: float,
    theta_min: float,
    V_r_healthy: torch.Tensor,
    *,
    strict_min_ratio: float = 0.5,
) -> None:
    """H3 三断言合一（终稿 §3.5 防护）：

      (1) F8:        theta_min > 0                        —— 否则下界 hinge 恒为 0，车闸静默退化为单边
      (2) C8 防静默:  theta_max < max_r{V_r^healthy}        —— 禁用 Q99 代替 max（实测静默带宽 7.46）
      (3) 防过度收紧: theta_min < strict_min_ratio * min_r{V_r^healthy}
    """
    if V_r_healthy.numel() == 0:
        raise ThetaRangeError("V_r_healthy 为空，无法标定阈值")
    v_max = float(V_r_healthy.max().item())
    v_min = float(V_r_healthy.min().item())

    if not (theta_min > 0):
        raise ThetaRangeError(f"(F8) theta_min={theta_min} 必须 > 0，否则车闸静默退化为单边")
    if not (theta_max < v_max):
        raise ThetaRangeError(
            f"(C8) theta_max={theta_max} 未小于健康态 V_r 最大值 {v_max:.5f}。"
            "两 hinge 将同时静默而损失/指标全部正常。"
        )
    if not (theta_min < strict_min_ratio * v_min):
        raise ThetaRangeError(
            f"(过度收紧) theta_min={theta_min} 未明显小于健康态 V_r 下端 {v_min:.5f}"
            f"（要求 < {strict_min_ratio}×）。车闸将退化为『惩罚一切』。"
        )


def assert_pairing_protocol(delta_B: float, *, tol: float = 1e-12, tol_dir: float = 0.0) -> None:
    """终稿 §3.5 协议自检：幅度 + 方向 双条件（缺方向条件会放过系统性符号错误）。

    实测：两配置 |ΔB| 均 < 1e-12、断言均通过，但其一是符号错误方向（终稿表格）。
    """
    if abs(delta_B) >= tol:
        raise PairingProtocolError(f"幅度条件失败：|ΔB|={abs(delta_B):.3e} >= {tol:g}")
    if delta_B > tol_dir:
        raise PairingProtocolError(
            f"方向条件失败：ΔB={delta_B:+.3e} > 0，健康态 B 大于退化态，"
            "说明两态用了不同的 c_r（协议缺陷，即使 |ΔB| 再小也必须拒绝）"
        )


def warn_if_artifact(W_health_series: Sequence[float], drift_max: float = 0.01) -> bool:
    """伪影机械判据：注入 ΔB 时若伴随量 W_health 跨档漂移 >1%，该批数据作废。

    实测区分度：路径 A（平移组均值+缩放噪声）漂移 100% → 作废；
                路径 B（仅平移组均值）漂移 0% → 有效。
    返回 True 表示数据有效。
    """
    ws = [float(w) for w in W_health_series]
    if len(ws) < 2:
        return True
    lo, hi = min(ws), max(ws)
    denom = max(hi, 1e-30)
    drift = abs(hi - lo) / denom
    if drift > drift_max:
        raise ArtifactError(
            f"伪影：W_health 漂移 {drift * 100:.2f}% > {drift_max * 100:.0f}%，该批数据作废。"
            "（失效率变化不得归因于 ΔB）"
        )
    return True


def assert_vr_nonzero(theta_min: float) -> None:
    """F8：theta_min > 0 断言（终稿表(2)(e)）。

    θ_min = 0 时下界 hinge 恒为 0、车闸静默退化为单边，
    而损失曲线完全正常、代码不报错 —— 必须以断言而非默认值保障。
    """
    if not (theta_min > 0):
        raise ThetaRangeError(f"F8 违反：theta_min={theta_min} 必须严格 > 0")


def hinge_trigger_rate(
    V_r: torch.Tensor, theta_max: float, theta_min: float
) -> tuple[float, float]:
    """车闸触发率（终稿 §3.5 防护③ 必报诊断量）。

    任一为 0 即说明车闸在该实例上无效。
    """
    upper = float((V_r > theta_max).float().mean().item())
    lower = float((V_r < theta_min).float().mean().item())
    return upper, lower
