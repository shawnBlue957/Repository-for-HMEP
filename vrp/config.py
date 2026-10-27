"""M0.1 配置注册表（模块设计文档 §2 S0）

对应式号：无（基础设施）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 终稿 §3.9 超参表中标记为「待验证」的项 —— 必须携带扫描网格，否则拒绝构建
PENDING_SCAN_GRID: Dict[str, List[float]] = {
    "epsilon_inv": [0.05, 0.07, 0.10],
    "omega_p": [0.0, 0.1, 0.2, 0.4],   # 终稿明示必须扫描这四档
    "lambda_dag": [0.5, 1.0],
    "lambda_sym": [0.25, 0.5],
    "lambda_coh": [0.05, 0.1],
    "lambda_h": [0.05, 0.1],
    "lambda_s": [0.05, 0.1],
    "lambda_PI": [0.0, 0.05],
    "lambda_ord": [0.0, 0.05],
    "zeta_ab": [-2.0, -1.0, -0.5],
    "mu_i": [0.5, 1.0],
}


@dataclass
class HyperParams:
    """终稿 §3.9 超参表 + 本文实现约定（标注 [实现约定] 者）。"""

    # ---- 结构（终稿 §3.8/§3.9）----
    N_OBJ: int = 21          # VG 平均物体数 [实现约定：运行时由 batch 决定]
    M_LEVELS: int = 3        # Chain_3
    D: int = 256             # 层嵌入维
    H: int = 4               # 解耦 MLP 隐层宽
    L_FUSE: int = 2          # 融合层数
    K: int = 50              # 谓词类数

    # ---- 富化度量（式14）----
    R_LOWRANK: int = 16
    EPS_G: float = 1e-4

    # ---- 蒸馏（式10/11/12）----
    ALPHA_DISTILL: float = 1.0

    # ---- 对合（式15）----
    EPS_INV: float = 0.07
    # ⚠️ 2026-10-18：车闸 θ 的 EMA 标定倍率（θ = μ ± nσ）
    # ⚠️ 2026-10-18：车闸 θ 的标定口径（分位数 EMA，跨 batch 累积）
    THETA_Q_LO: float = 0.10   # θ_min = EMA(Q10)
    THETA_Q_HI: float = 0.90   # θ_max = EMA(Q90)

    # ---- 闭包（式17/21/22）----
    ETA: float = 1.0
    OMEGA_P: float = 0.2     # 必须扫描 {0,0.1,0.2,0.4}
    EPS_ROW: float = 1e-8

    # ---- 损失权重（式33）----
    LAMBDA_INV_START: float = 0.5
    LAMBDA_INV_END: float = 0.2
    LAMBDA_INV_SCHEDULE_EPOCHS: int = 5    # 前 5 epoch 线性降
    LAMBDA_DAG: float = 1.0                # lambda^dagger 与 lambda_g 的嵌套权重
    LAMBDA_NON_SYM: float = 1.0
    LAMBDA_SYM: float = 0.5
    LAMBDA_AMBIG: float = 0.2
    LAMBDA_COH: float = 0.1
    LAMBDA_H: float = 0.1
    LAMBDA_S: float = 0.1
    LAMBDA_PI: float = 0.05
    LAMBDA_ORD: float = 0.05
    ZETA_AB: float = -1.0
    MU_I: float = 1.0                      # 对称化弱标签可信度 [待验证]

    # ---- 断言容差（M0.4）----
    TOL_DELTA_B: float = 1e-12             # 终稿：充分且简便
    TOL_DIR: float = 0.0                   # 方向条件容差
    ARTIFACT_DRIFT_MAX: float = 0.01       # 伪影判据：W_health 漂移 >1% 即作废

    def scan_grid(self, name: str) -> Optional[List[float]]:
        """返回该超参的待扫描网格（若非待验证项则返回 None）。"""
        return PENDING_SCAN_GRID.get(name)

    def assert_pending_scan(self) -> None:
        """M0.1 必做：所有待验证超参必须有非空扫描网格，防止被静默固化。"""
        missing = [k for k in PENDING_SCAN_GRID if not PENDING_SCAN_GRID[k]]
        if missing:
            raise AssertionError(f"待验证超参缺少扫描网格: {missing}")


def get_scheduled(name: str, epoch: int, hp: HyperParams) -> float:
    """调度型超参。终稿 §3.4：lambda_inv 前 5 epoch 由 0.5 线性降至 0.2。

    动机：L_inv 与 L_dag 在 r_exp 上存在部分相反的梯度分量，
          通过衰减一致性损失权重让位于方向判别（终稿 §3.4）。
    """
    if name == "lambda_inv":
        n = hp.LAMBDA_INV_SCHEDULE_EPOCHS
        if epoch >= n:
            return hp.LAMBDA_INV_END
        if epoch <= 0:
            return hp.LAMBDA_INV_START
        t = epoch / n
        return hp.LAMBDA_INV_START + t * (hp.LAMBDA_INV_END - hp.LAMBDA_INV_START)
    raise KeyError(f"未注册的调度超参: {name}")


def all_lambda_dict(hp: HyperParams, lambda_inv: float) -> Dict[str, float]:
    """式(33) 的权重装配表。

    ⚠️ 2026-10-19（P1 显式降级）：终稿式(33) 写「Σ_g λ_g L_dag^(g)」三组求和，
    但 forward 实际只产出 `dag_non_sym` 一组（对称组走式(24) 的 L_sym，
    歧义组无独立损失项）。故 `dag_sym` / `dag_ambig` 权重**不参与装配**，
    避免权重表与实际损失项不符（此前二者定义了却从未被使用）。
    """
    return {
        "cls": 1.0,
        "inv": lambda_inv,
        "dag_non_sym": hp.LAMBDA_DAG * hp.LAMBDA_NON_SYM,
        "sym": hp.LAMBDA_SYM,
        "coh": hp.LAMBDA_COH,
        "hom": hp.LAMBDA_H,
        "spread": hp.LAMBDA_S,
        "PI": hp.LAMBDA_PI,
        "ord": hp.LAMBDA_ORD,
    }
