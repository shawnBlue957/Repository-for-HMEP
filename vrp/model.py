"""S6 训练编排与推理：M6.2 / M6.5（模块设计文档 §2 S6）

覆盖终稿 §3.7。三路前向（主路 / 闭包路 / swap 路）+ 推理剥离。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import chain_attention, Backbone, DualHead, PrefixHead
from .closure import ClosureParams, mixed_student, transitive_closure_ste, two_hop_neighborhood
from .config import HyperParams, all_lambda_dict
from .enrich import (
    Inv_detached,
    MetricModule,
    build_c_dagger,
    top1_indicators,
)
from .losses import (
    permute_pooled,
    SharedProjection,
    cls_loss,
    coh_loss,
    dag_loss,
    gate_lower,
    gate_upper,
    inv_loss,
    ord_loss,
    pi_loss,
    sym_loss,
    total_loss,
)


# 2026-10-21：评测打分口径开关。
# 全局 P×K 排序要求分数**跨 pair 可比**；softmax 按 pair 归一化会破坏可比性，
# 而 SOTA 基线（GPS-Net/VCTree/SGTR/...）均输出未归一化 logits。
#
# 2026-10-21 默认改为 True（logits）：
#   实测「同权重、仅切打分方式」在 VG 2000/300（A 臂 10ep）上
#   top1 0.6161→0.6165（不变）、R@50 0.3404→0.3898（+0.0494，相对 +14.5%）。
#   top1 不变而 R@K 上升，证明这是**跨 pair 可比性**问题而非分类能力问题。
#   训练损失仍走概率（softmax），仅推理打分切换，故不影响任何训练语义。
# `--score-softmax` 可回退到旧口径，用于复现历史数字与做口径对照。
SCORE_LOGITS = True

# 2026-10-21：**MLP 式逐对旁路**开关（借鉴最强基线 MLPPairClassifier 的思路）。
# 动机：VG 上本文 top1 与 MLP 仅差 1.65pt，但 R@50 差 17.2pt ⟹ 差距全在
# 「pair 之间是否可比」，而本文模型没有任何 pair 级打分（forward_inference 只
# 返回谓词分布）。MLP 基线胜过全部结构模型，说明该设定下逐对交互更稳。
# 做法：加一条**无图级上下文**的逐对 MLP 支路，输出 (b,P,K) logits 作为残差
# 加到双流的 logits 上；结构分支与其约束（inv/dag/coh/...）完全保留。
# 参数代价约 1.33 M（8.3 M 总量级），推理无额外前向（与主路共享输入）。
USE_PAIR_RES = False
PAIR_RES_HID = 512

# pair 存在性监督（relationness）：负 pair 采样 + BCE。
#   PAIR_REL        开关（需 USE_PAIR_RES 同时为 True）
#   PAIR_REL_LAMBDA 训练损失权重
#   PAIR_REL_BETA   推理时 pair 级加性校准强度：z[k] += β·log σ(rel_logit)
#                   （per-pair 常数 ⟹ top1 恒等、R@K 可动，正是 R@K 缺的那一项）
PAIR_REL = False
PAIR_REL_LAMBDA = 0.5
PAIR_REL_BETA = 1.0
# 负样本比例：每图除 GT pair 外，从**非 GT 有序对**中均匀采样 ratio×|GT| 个。
# 相比「同 batch 内打乱 o_idx」的做法，均匀采样与评测时的候选分布一致
# （predcls 候选 = 图内所有有序对），且比例固定为 1:ratio。
PAIR_REL_NEG_RATIO = 3


class PairResidualHead(nn.Module):
    """MLP 式逐对旁路：只吃 pair 端点特征，**不注入任何图级上下文**。

    输入按 MLP 基线的同构口径组织：
        [v_s, v_o, t_s, t_o, g_s, g_o, g_s−g_o, |g_s−g_o|]
    输出 (b,P,K) logits，作为残差加到 `DualHead` 的两路 logits 上。
    """

    def __init__(self, d_v: int = 512, d_t: int = 512, n_geo: int = 8,
                 K: int = 100, hid: int = 512, p: float = 0.5):
        super().__init__()
        self.d_v, self.d_t, self.n_geo = d_v, d_t, n_geo
        d_in = d_v * 2 + d_t * 2 + n_geo * 4
        self.net = nn.Sequential(
            nn.Linear(d_in, hid), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hid, hid), nn.ReLU(inplace=True), nn.Dropout(p),
        )
        self.out_k = nn.Linear(hid, K)
        # 2026-10-21：**pair 存在性**（relationness）头。
        # 动机（AUC 实测）：本文模型 `max_k log p` 对「这一对有没有关系」的 AUC 仅 0.4935
        # ⟹ 完全无判别力。根因是训练只在 GT pair 上做 K 类 CE，**从未见过负 pair**，
        # 该维度没有任何监督。PredCls 的全局 P×K 排序却必须先回答它（R@K 落后的主体）。
        self.out_rel = nn.Linear(hid, 1)

    def forward(self, v_s, v_o, t_s, t_o, g_s, g_o, return_rel: bool = False):
        dg = g_s - g_o
        h = self.net(torch.cat([v_s, v_o, t_s, t_o, g_s, g_o, dg, dg.abs()], dim=-1))
        k = self.out_k(h)
        if not return_rel:
            return k
        return k, self.out_rel(h).squeeze(-1)


class VRPCatModel(nn.Module):
    """范畴建模的视觉关系预测模型（三路前向）。

    组件 ↔ 模块设计文档：
      backbone     M1.1–M1.6
      dual_head    M1.7 (R3)
      metric       M2.1/M2.2
      closure      M3.1
      shared       M4.4/M4.5 共享投影
      head1/head2  M4.10
    """

    def __init__(self, hp: HyperParams, d_in: dict):
        super().__init__()
        self.hp = hp
        d, K = hp.D, hp.K
        self.backbone = Backbone(hp, d_in)
        self.dual_head = DualHead(d, K)
        # MLP 式逐对旁路（开关见模块级 USE_PAIR_RES）。d_in 由 backbone 的输入规格给出。
        self.pair_res = None
        if USE_PAIR_RES:
            self.pair_res = PairResidualHead(
                d_v=d_in.get("v", d), d_t=d_in.get("t", d),
                n_geo=d_in.get("s", 8), K=K, hid=PAIR_RES_HID)
        self.head1 = PrefixHead(d, K)
        self.head2 = PrefixHead(d, K)
        self.metric = MetricModule(d, hp.R_LOWRANK, hp.EPS_G)
        self.closure = ClosureParams(K)
        self.shared = SharedProjection(d)
        # 2026-10-18：式(31) 的置换生成器（在 losses 内按设备重建，此处仅存种子）
        self._perm_seed = 20261018
        # 2026-10-18：**soft 对称掩码**（式24 用）。
        # 硬 mask_sym 只含 4 个对称谓词 ⟹ 长尾谓词无对称监督 ⟹ L_sym 恒 0。
        # soft_mask_sym[k] = 该谓词的「近对称程度」（语义相似度映射到 [0,1]）。
        self.soft_mask_sym = None
        # 2026-10-18：全训练集谓词 presence 表（式 25/26 车闸用）。
        # 由 `register_fixed` 从训练集频次预标定，避免 batch 内缺席导致车闸关闭。
        self.predicate_presence = None

    # ------------------------------------------------------------ M2.3 冻结量
    def register_frozen(
        self,
        M_inv: torch.Tensor,
        J: torch.Tensor,
        c_dag: torch.Tensor,
        w: torch.Tensor,
        mask_sym: torch.Tensor,
        mask_non_sym: torch.Tensor,
        mask_ambig: torch.Tensor,
    ) -> None:
        """注册离线预计算的固定量（终稿 §3.7：M_inv/q/组划分初始值为固定量）。"""
        self.register_buffer("M_inv", M_inv)
        self.register_buffer("J", J)
        self.register_buffer("c_dag", c_dag)
        self.register_buffer("w", w)
        self.register_buffer("mask_sym", mask_sym)
        self.register_buffer("mask_non_sym", mask_non_sym)
        self.register_buffer("mask_ambig", mask_ambig)

    def register_soft_sym(self, soft: torch.Tensor) -> None:
        """注册 **soft 对称掩码**（式24 用，2026-10-18 新增）。

        ⚠️ 修复背景：原 `mask_sym` 是**硬**掩码，只标记严格自反的对称谓词
        （VRD 仅 4 个：next to / near / beside / by）。这些谓词在长尾分布里
        多属低频 ⟹ batch 内真值几乎不落在 G_sym ⟹ `sel = (y*mask_sym).sum(-1)`
        恒为 0 ⟹ **L_sym 恒为 0，从未生效**。

        本接口接受一个 (K,) 的**连续权重**（0=非对称，1=完全对称），
        由「谓词文本嵌入的语义相似度」构造：与自身语义越近的谓词，
        越可能具有对称性 ⟹ 权重越大。
        """
        self.soft_mask_sym = soft.float()
        # ⚠️ 2026-10-18：需与模型同设备，否则 forward 时报
        #    "Expected all tensors to be on the same device"
        if self.soft_mask_sym.device != next(self.parameters()).device:
            self.soft_mask_sym = self.soft_mask_sym.to(next(self.parameters()).device)

    def register_presence(self, presence: torch.Tensor) -> None:
        """注册**全训练集**谓词 presence 表（式 25/26 车闸用，2026-10-18 新增）。

        ⚠️ 修复背景：原实现用 `present = (batch 内 counts > 0)`，
        而 VRD 训练集 47.1% 的谓词频次 < batch_size(=32)，
        导致这些谓词的车闸在任何 batch 内都被关闭 ⟹ L_hom / L_spread 恒为 0。
        改用训练集级 presence 后，车闸对**所有出现过至少一次**的谓词生效。
        """
        self.predicate_presence = presence.float()

    # ------------------------------------------------------------ M6.2 三路前向
    def forward(
        self,
        x: dict,
        masks: torch.Tensor,
        s_idx: torch.Tensor,
        o_idx: torch.Tensor,
        y_label: Optional[torch.Tensor] = None,
        y_tilde_ji: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        m_i: Optional[torch.Tensor] = None,
        lambda_inv: float = 0.2,
        omega_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """y_label 为整数标签 (b,P)；one-hot 由内部派生。

        ⚠️ 接口歧义防护：cls_loss 需要 one-hot，estimate_variance 与 w 查表需要整数标签，
        故二者必须分开传入/派生，避免同一张量被两种语义复用。
        """
        hp = self.hp
        omega_p = hp.OMEGA_P if omega_p is None else omega_p
        K = hp.K
        dtype = x["v"].dtype
        y = None
        if y_label is not None:
            y = torch.nn.functional.one_hot(y_label.long(), K).to(dtype)   # (b,P,K) one-hot

        # ---------------- 主路 ----------------
        z_j, Z, _ = self.backbone.encode(x, masks)
        v_glo = Z[:, 1]                                             # (b,d) 图像级
        pf = self.backbone.pair_features(z_j, Z, s_idx, o_idx)
        r_exp, r_imp, r = self.backbone.forward(pf, v_glo)
        # ⚠️ 2026-10-19（SC 指标）：重算门控 ω = σ(W_ω v_glo + b_ω) 供诊断。
        #    backbone.forward 内部丢弃了 ω；此处用同一 fusion.w 提取，不重算 r。
        #    ω∈(0,1) 为显式流 r_exp 的逐维保留率，融合式 r = ω⊙r_exp + r_imp。
        _vg = v_glo
        if _vg.dim() == 2 and r_exp.dim() == 3:
            _vg = _vg.unsqueeze(1).expand(-1, r_exp.shape[1], -1)
        omega = torch.sigmoid(self.backbone.fusion.w(_vg))

        f_s2, f_u2, f_o2 = pf.f_s2, pf.f_u2, pf.f_o2
        _with_rel = (self.pair_res is not None) and PAIR_REL
        _pout = self._pair_res_logits(x, s_idx, o_idx, return_rel=_with_rel)
        if isinstance(_pout, tuple):
            _res, _rel_pos = _pout
        else:
            _res, _rel_pos = _pout, None
        p_ij, p_ji = self.dual_head(f_s2, f_u2, f_o2, res_logits=_res)

        # ---------------- swap 路（复用第 1 层，不重算 backbone）----------------
        r_ji = self.backbone.implicit.swap(f_s2, f_u2, f_o2)
        p_ji_pi, p_ij_pi = self.dual_head(f_o2, f_u2, f_s2, res_logits=_res)

        # ---------------- 闭包路 ----------------
        p_trans, p_conv = self.closure()
        P_bar_sq = two_hop_neighborhood(p_conv, hp.ETA)
        # ⚠️ 2026-10-18：注入**可学的对角偏置**（见 closure.theta_diag 的说明）。
        #    目的：让式(28) 的 `diag(P̄²)` 加权有区分度，从而真正约束两个头，
        #    而不是退化为「两个独立 softmax 的 D_JS」。
        #
        #    ⚠️ 实现注意：偏置**只用于式(28) 的 diag 权重**，
        #    不改 P_bar_sq 本身 —— 因为 P_bar_sq 还要喂给 `mixed_student`（式22），
        #    那里要求严格的 (K,K)；若在此处乘成 (1,K,K) 会破坏下游的广播。
        _db = self.closure.diag_bias()                             # (K,)
        # ⚠️ 2026-10-18 修正：**不能**用 `P_bar_sq * diag_embed(_db)`
        #    —— 那是**矩阵乘法**，非对角项也会参与，使对角线被污染
        #    （实测 diag 极差恒为 0，因为 P_bar_sq 的对角被非对角项抵消）。
        #    正确做法：只**改对角元**（其余保持不变）：
        #        P̄²_coh[k,k] = P̄²[k,k] · db[k]
        _dp0 = torch.diagonal(P_bar_sq)
        P_bar_sq_coh = P_bar_sq + torch.diag_embed((_db - 1.0) * _dp0)  # (K,K)，仅供式(28)
        p_tilde = mixed_student(p_ij, P_bar_sq, omega_p, hp.EPS_ROW)
        # ⚠️ 2026-10-18：topk=1 ⟹ 每行只保留最强转移为「有边」
        #    （原用 adj>0 的稠密判定 ⟹ 闭包全 1 ⟹ 复合不可达集恒空）
        reach = transitive_closure_ste(p_conv[:, :K], p_trans, topk=1)

        # ---------------- 度量 ----------------
        G_inv = self.metric.inverse()

        # ---------------- 损失 ----------------
        parts: Dict[str, torch.Tensor] = {}
        lambdas = all_lambda_dict(hp, lambda_inv)

        if y is not None:
            y_t = y if y_tilde_ji is None else y_tilde_ji
            mu_t = torch.ones(y.shape[:-1], device=y.device, dtype=dtype) if mu is None else mu
            # ⚠️ 式(30) 传递违反掩码 m_i：由外部 ε_assoc(P_i) < ε₀ 计算后传入，
            #    模型内部**不**计算 ε_assoc。若漏传则退化为全 1（约束静默失效）。
            #    2026-10-19（P1）：加显式断言，把「静默失效」改为「训练期报错」。
            if m_i is None:
                raise AssertionError(
                    "式(30) 传递违反掩码 m_i 未传入。外部须先计算 ε_assoc(P_i) 并 "
                    "按 ε₀ 阈值生成 m_i；若确无传递一致性监控需求，请显式传全 1 张量。")
            m_t = m_i
            parts["cls"] = cls_loss(p_ij, p_ji, y, y_t, mu_t, m_t)

        parts["inv"] = inv_loss(r_exp, r_imp, G_inv)

        # ⚠️ 2026-10-18：本文不含知识蒸馏（式 11 KL 与式 12 特征回归均已删除）。

        # --- M4.5 L_dag（H2 下游；p_ji^π 必须与 p_ij 不同）---
        Inv_p_ij = Inv_detached(p_ij, self.M_inv, hp.EPS_INV)
        w_i = self.w[y_label.long()] if y_label is not None else torch.ones(p_ij.shape[:2], device=p_ij.device)
        L_dag, L_dag_dist, L_dag_repr = dag_loss(
            p_ij, p_ji_pi, r_ji, Inv_p_ij, self.c_dag, G_inv, w_i, self.shared, hp.LAMBDA_DAG
        )
        parts["dag_non_sym"] = L_dag

        if y is not None:
            # ⚠️ 2026-10-18：原 sym 用**硬** mask_sym（仅 4 个对称谓词），
            #    谓词为长尾 ⟹ batch 内真值几乎不落在 G_sym ⟹ L_sym 恒为 0。
            #    现改为 **soft 对称掩码**：以「真值谓词的语义近邻」作权重，
            #    语义相近的谓词对（如 in/near、above/below）都按小权重计入，
            #    从而让长尾谓词也有非零的对称性监督。
            _msym = self.soft_mask_sym
            if _msym is None:
                _msym = self.mask_sym
            parts["sym"] = sym_loss(p_ij, p_ji_pi, y, _msym)

        parts["coh"] = coh_loss(pf.f_r1, pf.f_r2, P_bar_sq_coh, self.head1, self.head2)

        # ---- pair 存在性监督（显式负采样 + BCE）----
        # 负样本：每图从**非 GT 有序对**中均匀采 ratio×|GT| 个（与评测候选同分布）。
        # 正样本即当前 batch 的 GT pair。两项分别取均值后等权相加，
        # 避免 1:3 的比例把负项主导掉。
        if _rel_pos is not None:
            _rel_neg = self._sample_negative_rel(x, s_idx, o_idx, masks,
                                                 ratio=PAIR_REL_NEG_RATIO)
            if _rel_neg is not None and _rel_neg.numel() > 0:
                _bce = torch.nn.functional.binary_cross_entropy_with_logits
                parts["pair_rel"] = PAIR_REL_LAMBDA * 0.5 * (
                    _bce(_rel_pos, torch.ones_like(_rel_pos))
                    + _bce(_rel_neg, torch.zeros_like(_rel_neg)))
        # ⚠️ 2026-10-18 诊断：记录 diag(P̄²) 的相对极差。
        #    若 ≈ 0 ⟹ p_conv 仍均匀 ⟹ 式(28) 的 P̄ 加权退化为常数缩放（语义失效）。
        _dp = torch.diagonal(P_bar_sq_coh)
        _dpmed = _dp.abs().median().clamp(min=1e-12)
        self._diag_p_spread = float(((_dp.max() - _dp.min()) / _dpmed).item())

        # --- M4.11 序正则（式29，实现口径：复合不可达对加权）---
        parts["ord"] = ord_loss(p_trans, reach)

        # --- M4.7/M4.8/M4.9 车闸（H1：只把 V_r 喂给 hinge）---
        # ⚠️ 2026-10-18 修复：原代码 `present = (counts > 0)` 取自**当前 batch**，
        #    而 VRD 训练集 47.1% 的谓词频次 < batch_size ⟹ 半数谓词的
        #    车闸在任何 batch 内都被关闭 ⟹ L_hom / L_spread 恒为 0。
        #    现改为：**用全训练集频次预标定的 presence 表**（注册为固定量），
        #    并叠加 batch 内计数作为补充（两者取或）。
        from .losses import estimate_variance
        pi = pf.sample_pi()
        y_lab = y_label.long() if y_label is not None else torch.zeros(pf.f_s0.shape[:2], dtype=torch.long, device=pf.f_s0.device)
        V_r, V_global, H_r, counts = estimate_variance(pi, y_lab, K)
        present = (counts > 0).float()
        if self.predicate_presence is not None:
            present = torch.maximum(present, self.predicate_presence.to(present.device))
        # ⚠️ 2026-10-18：θ 在**首批自动标定**（真实 V_r 的分位数），
        #    避免外部传入固定张量导致 hinge 永不触发。
        if getattr(self, "_theta_auto", False):
            # ⚠️ 2026-10-18：在**每样本方差** V̂_r = V_r / |P_r| 的量纲上标定 θ，
            #    与 gate_upper/gate_lower 的 hinge 量纲严格一致。
            #
            #    标定方式用 **EMA 统计量**（均值/标准差）而非首批分位数：
            #    首批分位数的问题是——标定用的那一批与后续批的 V̂ 分布会漂移，
            #    于是 θ 很快落到所有 V̂ 之外 ⟹ 铰链恒不触发（实测如此）。
            #    EMA 随训练自适应，且给出**明确语义**：
            #        θ_max = μ + nσ  （离散度显著高于平均 ⟹ 疑逐对记忆）
            #        θ_min = μ − nσ  （离散度显著低于平均 ⟹ 疑组间塌缩）
            with torch.no_grad():
                _v = V_r.detach().float()
                _cnt = counts.detach().float().clamp(min=1.0)
                _vhat = _v / _cnt
                _m = _vhat > 0
                if int(_m.sum()) >= 4:
                    _cur = _vhat[_m]
                    # ⚠️ 2026-10-18 第三次修正：θ 标定改用**分位数的 EMA**。
                    #
                    #   为何不用 μ ± nσ：实测首个 batch 的 σ 可大于 μ
                    #   （例：μ=1.04e-3、σ=1.29e-3）⟹ θ_min = μ−nσ < 0 被截到 0
                    #   ⟹ `max(0, θ_min − V̂)` 恒为 0 ⟹ **L_spread 永远为 0**。
                    #
                    #   为何不用「首批分位数」：标定用的那一批与后续批的 V̂ 分布
                    #   会漂移 ⟹ θ 很快落到所有 V̂ 之外 ⟹ 铰链恒不触发。
                    #
                    #   ⟹ 最终方案：**跨 batch 的分位数 EMA**
                    #       θ_max = EMA(Q_hi)   （离散度显著偏高 ⟹ 疑逐对记忆）
                    #       θ_min = EMA(Q_lo)   （离散度显著偏低 ⟹ 疑组间塌缩）
                    #   分位数口径与「上下界」的语义天然一致，且恒为正。
                    _qs = torch.quantile(_cur, torch.tensor(
                        [float(getattr(hp, "THETA_Q_LO", 0.10)),
                         float(getattr(hp, "THETA_Q_HI", 0.90))],
                        device=_cur.device))
                    _qlo, _qhi = float(_qs[0]), float(_qs[1])
                    if not hasattr(self, "_theta_ema") or self._theta_ema is None:
                        self._theta_ema = (_qlo, _qhi)
                    else:
                        _ol, _oh = self._theta_ema
                        self._theta_ema = (0.9 * _ol + 0.1 * _qlo,
                                           0.9 * _oh + 0.1 * _qhi)
                    _m2, _m3 = self._theta_ema
                    self.theta_max = max(_m3, 1e-12)
                    self.theta_min = max(min(_m2, self.theta_max * 0.999), 0.0)
        theta_max = float(self.theta_max)
        theta_min = float(self.theta_min)
        # ⚠️ 2026-10-18：传入 counts（组规模 |P_r|）作 scale ⟹ 车闸作用于
        #    **每样本方差** V̂_r = V_r / |P_r|（V_r 绝对量级仅 1e-4~1e-2，
        #    绝对阈值恒不触发；V_r/V_global 中位比≈0.995 亦无判别力）。
        parts["hom"] = gate_upper(V_r, theta_max, present, scale=counts)
        # ⚠️ 2026-10-18 修复：`gate_lower` 返回 **tuple**(l_vr + l_ins, l_vr, l_ins)，
        #    原代码写成 `parts["spread"] = gate_lower(...)` 直接把 tuple 赋给了损失项
        # ⟹ 后续乘法报 "can't multiply sequence by non-int of type 'float'"。
        _sp = gate_lower(V_r, V_global, theta_min, theta_min, present, scale=counts)
        parts["spread"] = _sp[0] if isinstance(_sp, tuple) else _sp

        # --- M4.12 置换守护（式31）---
        # ⚠️ 2026-10-18 修复：原代码传空列表 ⟹ L_PI 恒为 0。
        #    现对**物体层池化 Z[:,0]** 真正做随机置换并前向比较。
        #    预期：若 Z[:,0] 是顺序无关池化，loss 仍≈0（符合定性）；
        #          一旦有序槽位破坏了置换对称性，该项立即上升。
        # ⚠️ 2026-10-18 修复（第 3 层）：主路的 Z 是 **chain_attention 的输出**
        #    （`backbone.encode` 返回 A 而非池化 Z），故置换样本必须走
        #    **同一后处理**，否则拿「注意力后」比「纯池化」⟹ 恒为大值（伪信号）。
        #    `post` 注入 backbone 的 chain_attention（同一组 W_q/W_k/W_v）。
        def _pi_post(z_n: torch.Tensor) -> torch.Tensor:
            bb = self.backbone
            return chain_attention(z_n, bb.W_q, bb.W_k, bb.W_v)

        _perms = permute_pooled(z_j, masks, None, n_perm=2, post=_pi_post)
        _z0 = Z[:, 0]                                                # (b,d)
        parts["PI"] = pi_loss(_z0, _perms)

        total = total_loss(parts, lambdas)

        out = dict(parts)
        out["total"] = total
        out["_V_r"] = V_r
        out["_V_global"] = V_global
        out["_H_r"] = H_r
        out["_p_ij"] = p_ij
        out["_p_ji"] = p_ji
        out["_p_ji_pi"] = p_ji_pi
        out["_r_exp"] = r_exp
        out["_r_imp"] = r_imp
        out["_r"] = r
        out["_omega"] = omega
        out["_p_tilde"] = p_tilde
        out["_G_inv"] = G_inv
        return out

    # ------------------------------------------------------------ M6.5 推理剥离
    def forward_inference(
        self, x: dict, masks: torch.Tensor, s_idx: torch.Tensor, o_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理：剥离蒸馏分支、swap 路、闭包路、全部训练期正则；仅留式(1)–(7)+Head。

        终稿 §3.7：参数 ≈ 7.01M + R1 3,072 + R2 131,072 + R3 64,000 ≈ 7.21M。
        """
        z_j, Z, _ = self.backbone.encode(x, masks)
        v_glo = Z[:, 1]
        pf = self.backbone.pair_features(z_j, Z, s_idx, o_idx)
        # r_exp/r_imp/r 的融合在推理时仅用于产出 Head 输入，故走同一条链
        self.backbone.forward(pf, v_glo)
        _with_rel = (self.pair_res is not None) and PAIR_REL
        _pout = self._pair_res_logits(x, s_idx, o_idx, return_rel=_with_rel)
        if isinstance(_pout, tuple):
            _res, _rel = _pout
        else:
            _res, _rel = _pout, None
        p_ij, p_ji = self.dual_head(pf.f_s2, pf.f_u2, pf.f_o2,
                                    logits_only=SCORE_LOGITS, res_logits=_res)
        if _rel is not None and SCORE_LOGITS:
            # pair 级校准：per-pair 常数平移 ⟹ top1 不变、全局 R@K 可动
            _b = PAIR_REL_BETA * torch.nn.functional.logsigmoid(_rel).unsqueeze(-1)
            return p_ij + _b, p_ji + _b
        return p_ij, p_ji

    def _sample_negative_rel(self, x: dict, s_idx: torch.Tensor, o_idx: torch.Tensor,
                             masks: torch.Tensor, ratio: int = 3):
        """每图从**非 GT 有序对**中均匀采样 ratio×|GT| 个负对，返回 (M,) 的 rel logit。

        为什么不用「打乱 o_idx」：那样得到的是 (GT 主语, GT 宾语) 的重排，
        主语侧仍被限定在「有关系的主语」上，负样本分布与 predcls 的候选
        （图内任意有序对）不一致，且正负比恒为 1:1。
        """
        if self.pair_res is None:
            return None
        dev = s_idx.device
        b, P = s_idx.shape
        if P < 1 or masks.dim() != 3:
            return None
        n_obj = masks[:, 2, :].sum(-1).long()                    # (b,) 每图有效物体数
        nmax = int(n_obj.max().item())
        if nmax < 2:
            return None
        Kneg = max(1, int(ratio) * P)

        # GT 掩码（排除真实对，避免把正样本当负样本）
        gt = torch.zeros(b, nmax, nmax, dtype=torch.bool, device=dev)
        _bi = torch.arange(b, device=dev).unsqueeze(1).expand(b, P)
        _ok = ((s_idx >= 0) & (s_idx < nmax) & (o_idx >= 0) & (o_idx < nmax))
        gt[_bi[_ok], s_idx[_ok], o_idx[_ok]] = True

        # 过采样 4 倍后按图取前 Kneg 个合法样本
        tot = Kneg * 4
        aa = torch.randint(0, nmax, (b, tot), device=dev)
        cc = torch.randint(0, nmax - 1, (b, tot), device=dev)
        cc = cc + (cc >= aa).long()                              # 保证 aa != cc
        # ⚠️ 必须是三索引形式 `gt[bidx, aa, cc]`：只写 `gt[aa, cc]` 会把 aa 当成
        #    **batch 维**索引（aa 值可达 nmax-1 > b-1）⟹ GPU device-side assert。
        _brow = torch.arange(b, device=dev).unsqueeze(1).expand(b, tot)
        _is_gt = gt[_brow, aa, cc]
        ok = (aa < n_obj[:, None]) & (cc < n_obj[:, None]) & (~_is_gt)
        rank = torch.cumsum(ok.long(), dim=1) - 1
        sel = ok & (rank < Kneg)
        if int(sel.sum()) == 0:
            return None

        _bidx = torch.arange(b, device=dev).unsqueeze(1).expand(b, tot)[sel]
        _a = aa[sel].unsqueeze(1)                                # (M,1)
        _c = cc[sel].unsqueeze(1)
        outs = []
        for key, idx in (("v", _a), ("v", _c), ("t", _a), ("t", _c),
                         ("s", _a), ("s", _c)):
            t = x[key][_bidx]                                    # (M,N,d)
            outs.append(torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])))
        _out = self.pair_res(*[o.squeeze(1) for o in outs], return_rel=True)
        return _out[1] if isinstance(_out, tuple) else None

    def _pair_res_logits(self, x: dict, s_idx: torch.Tensor,
                         o_idx: torch.Tensor, return_rel: bool = False):
        """MLP 式逐对旁路：从**原始**端点特征直接产出 (b,P,K) logits 残差。

        刻意不用 `pf.f_s2/f_u2/f_o2`（它们已被图级 z1/z2 注入上下文）——
        与 MLP 基线「纯逐对、不碰上下文」的口径保持一致，才能检验
        「上下文到底有没有用」。关闭开关时返回 None，调用处逻辑不变。

        `return_rel=True` 时额外返回 (b,P) 的 pair 存在性 logit。
        """
        if self.pair_res is None:
            return None
        if x["t"].shape[-1] != self.pair_res.d_t or x["v"].shape[-1] != self.pair_res.d_v:
            # 极端情形：调用方未过 obj_emb，x["t"] 是 label id（宽度 1）。
            # 此时旁路无法按同构口径工作，直接跳过（等价于关闭），由日志侧可见。
            return None
        outs = []
        for key, idx in (("v", s_idx), ("v", o_idx), ("t", s_idx), ("t", o_idx),
                         ("s", s_idx), ("s", o_idx)):
            t = x[key]
            outs.append(torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])))
        return self.pair_res(*outs, return_rel=return_rel)

    # ------------------------------------------------------------ 阈值标定
    def calibrate_theta(self, V_r_healthy: torch.Tensor,
                        q_hi: float = 0.90, q_lo: float = 0.10,
                        relative: bool = True, V_global: float = 1.0) -> None:
        """式(26)/(27)：θ 取**真实 V_r** 的 Q90/Q10 分位数，零人工（终稿 §3.7 步骤⑥）。

        ⚠️ 2026-10-18 修复：此前调用处传入的是**固定张量** `tensor([0.5,1.0,1.5,2.0])`，
        而实际 V_r（组内方差）量级远小于 0.5 ⟹ Q10 仍 >真实 V_r ⟹
        `max(0, θ_min − V_r)` 恒为 0 ⟹ **L_spread 恒为 0**。
        现改为：从**真实 V_r**（或相对量纲 V_r/V_global）取分位数，
        保证 θ_min 落在实际分布内 ⟹ hinge 必然被部分触发。
        """
        _v = V_r_healthy.float().reshape(-1)
        _v = _v[torch.isfinite(_v)]
        if _v.numel() == 0:
            self.theta_max, self.theta_min = 1.0, 0.0
            return
        if relative:
            _v = _v / max(float(V_global), 1e-8)
        self.theta_max = float(torch.quantile(_v, q_hi).item())
        self.theta_min = float(torch.quantile(_v, q_lo).item())

    def trainable_parameter_names(self) -> list:
        return [n for n, p in self.named_parameters() if p.requires_grad]


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    """参数计数（M6.5 的验收门 G4/G7 依赖此函数）。"""
    ps = module.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not only_trainable))


def strip_for_inference(model: VRPCatModel) -> nn.Module:
    """M6.5：返回仅含推理所需子模块的容器，用于参数计数与一致性断言。"""
    keep = nn.ModuleDict({
        "backbone": model.backbone,
        "dual_head": model.dual_head,
    })
    return keep
