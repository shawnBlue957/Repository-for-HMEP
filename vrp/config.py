"""M0.1 Configuration Registry (Module Design Document §2 S0)

Corresponding model number: None (infrastructure)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Final draft §3.9 Items marked ”to be verified” in the hyperparameter table - the scan grid must be carried, otherwise the build will be refused
PENDING_SCAN_GRID: Dict[str, List[float]] = {
    "epsilon_inv": [0.05, 0.07, 0.10],
    "omega_p": [0.0, 0.1, 0.2, 0.4],   # The final draft clearly states that these four files must be scanned
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
    """Final draft §3.9 Hyperparameter table + implementation conventions of this article (those marked [Implementation Conventions])."""

    # ---- Structure (Final Draft §3.8/§3.9)----
    N_OBJ: int = 21          # VG average number of objects [Implementation convention: runtime is determined by batch]
    M_LEVELS: int = 3        # Chain_3
    D: int = 256             # Layer embedding dimension
    H: int = 4               # Decoupling MLP hidden layer width
    L_FUSE: int = 2          # Number of fusion layers
    K: int = 50              # Predicate class number

    # ---- Enrichment measure (Equation 14) ----
    R_LOWRANK: int = 16
    EPS_G: float = 1e-4

    # ---- Distillation (Formula 10/11/12)----
    ALPHA_DISTILL: float = 1.0

    # ---- Involution (Formula 15)----
    EPS_INV: float = 0.07
    # ⚠️ 2026-10-18: EMA calibration factor of brake θ (θ = μ ± nσ)
    # ⚠️ 2026-10-18: Calibration caliber of brake θ (quantile EMA, accumulated across batches)
    THETA_Q_LO: float = 0.10   # θ_min = EMA(Q10)
    THETA_Q_HI: float = 0.90   # θ_max = EMA(Q90)

    # ---- Closure (Formula 17/21/22)----
    ETA: float = 1.0
    OMEGA_P: float = 0.2     # Must scan {0,0.1,0.2,0.4}
    EPS_ROW: float = 1e-8

    # ---- Loss weight (Equation 33) ----
    LAMBDA_INV_START: float = 0.5
    LAMBDA_INV_END: float = 0.2
    LAMBDA_INV_SCHEDULE_EPOCHS: int = 5    # Linear decrease in the first 5 epochs
    LAMBDA_DAG: float = 1.0                # Nested weights of lambda^dagger and lambda_g
    LAMBDA_NON_SYM: float = 1.0
    LAMBDA_SYM: float = 0.5
    LAMBDA_AMBIG: float = 0.2
    LAMBDA_COH: float = 0.1
    LAMBDA_H: float = 0.1
    LAMBDA_S: float = 0.1
    LAMBDA_PI: float = 0.05
    LAMBDA_ORD: float = 0.05
    ZETA_AB: float = -1.0
    MU_I: float = 1.0                      # Symmetric weak label credibility [to be verified]

    # ---- Assertion Tolerance (M0.4) ----
    TOL_DELTA_B: float = 1e-12             # Final draft: sufficient and simple
    TOL_DIR: float = 0.0                   # Directional condition tolerance
    ARTIFACT_DRIFT_MAX: float = 0.01       # Artifact criterion: If W_health drifts >1%, it will be invalid.

    def scan_grid(self, name: str) -> Optional[List[float]]:
        """Returns the grid to be scanned for this hyperparameter (returns None if it is not an item to be verified)."""
        return PENDING_SCAN_GRID.get(name)

    def assert_pending_scan(self) -> None:
        """M0.1 Required: All hyperparameters to be verified must have non-empty scanning grids to prevent them from being silently solidified."""
        missing = [k for k in PENDING_SCAN_GRID if not PENDING_SCAN_GRID[k]]
        if missing:
            raise AssertionError(f"The hyperparameters to be verified are missing the scanning grid.: {missing}")


def get_scheduled(name: str, epoch: int, hp: HyperParams) -> float:
    """Scheduling hyperparameters. Final draft §3.4: The first 5 epochs of lambda_inv are linearly reduced from 0.5 to 0.2.

    Motivation: L_inv and L_dag have partially opposite gradient components on r_exp,
          Loss of weight by attenuating consistency gives way to orientation discrimination (final draft §3.4).
    """
    if name == "lambda_inv":
        n = hp.LAMBDA_INV_SCHEDULE_EPOCHS
        if epoch >= n:
            return hp.LAMBDA_INV_END
        if epoch <= 0:
            return hp.LAMBDA_INV_START
        t = epoch / n
        return hp.LAMBDA_INV_START + t * (hp.LAMBDA_INV_END - hp.LAMBDA_INV_START)
    raise KeyError(f"Unregistered scheduling hyperparameters: {name}")


def all_lambda_dict(hp: HyperParams, lambda_inv: float) -> Dict[str, float]:
    """The weight assembly table of Equation (33).

    ⚠️ 2026-10-19 (P1 explicit downgrade): The final formula (33) is written as the sum of three groups of ”Σ_g λ_g L_dag^(g)”,
    But forward actually only produces a group of `dag_non_sym` (the L_sym of the symmetric group type (24),
    The ambiguous group has no independent loss items). Therefore, `dag_sym` / `dag_ambig` weights** do not participate in assembly**,
    Avoid discrepancies between the weight table and the actual loss items (both were previously defined but never used).
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
