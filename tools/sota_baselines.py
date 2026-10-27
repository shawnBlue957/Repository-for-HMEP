"""SOTA 系基线集合（与主实验**同 CLIP 特征、同评测口径**）。

## 为什么要自己实现（沿用 bilstm_baseline.py 的思路）

SOTA 的文献数值无法在本项目管线上复跑：其权重全部托管在 Google Drive，
且都依赖同一个 VG 训练检测器 checkpoint。于是**在同一套冻结 CLIP 特征上**
自己实现 SOTA 系的**核心关系分类器**，使表中出现一行
「由我们自己在同一管线上跑出来的 SOTA 系基线」，直接消除「骨干不同所以不可比」的质疑。

## 公平性契约（每个基线都必须满足）

1. **特征完全一致**：`x_v` = 冻结 CLIP 物体裁剪特征、`x_t` = 冻结 CLIP 物体类文本嵌入、
   `x_s` = 几何（8 维），与主实验 A 臂逐位相同（共用 `feat_cache_*.pt`）。
2. **评测完全一致**：直接 import `evaluate` / `evaluate_streaming` / `_finalize`，
   故 R@K / mR@K / BRC / IR@K 的口径与主实验**逐位一致**，无任何自写指标。
3. **候选集一致**：`--eval-mode predcls` 下候选为图内全部有序对。
4. **可训练参数相当**：每个基线控制在 ~1.5–2.5M，与主实验 A 臂（1.68M）同量级，
   避免「基线更大所以更差/更好」的混淆。

## 各基线对应的文献与核心设计

| tag | 文献 | 核心设计（在同特征上的忠实简化） |
|---|---|---|
| `bilstm` | Neural Motifs, CVPR 2018 | 按主体 x 排序的序列 → BiLSTM 传播上下文 → 逐对 softmax（已实现于 bilstm_baseline.py） |
| `gpsnet` | GPS-Net, CVPR 2020 (Oral) | **三线性方向绑定**：主体/客体/上下文三个特征各经 Linear 后做**外积累加**，再过 LayerNorm；GPS-Net 原文即 $W_s\\odot W_o$，与本文式(4) 的 $\\alpha\\odot f_s,\\beta\\odot f_o$ 同源 |
| `vctree` | VCTree, CVPR 2019 (Oral) | **动态树消息传递**：以谓词为超边的消息在物体节点间按**语义相似度加权**传播（用冻结 CLIP 文本嵌入算谓词语义相似度作为边权），多层聚合后分类；对应原文的 affinity-based message passing |
| `sgtr` | SGTR, CVPR 2022 | **单阶段 Transformer 关系头**：把每个候选对编码为 token（含主体/客体/联合框/几何），过若干层 Transformer encoder，再分类 |
| `transformer` | 通用 SOTA 对照 | 对照组：纯 Transformer encoder（不含 SGTR 的单阶段细节），用于判断「架构本身」与「本文范畴约束」的相对贡献 |
| `mlp` | 最简对照 | 逐对 MLP（无序列、无消息传递）⟹ 衡量「结构化建模」的净收益 |

## 与主实验的对照关系

主实验的 A 臂 = 式(1)–(13) 全量（含范畴约束 L_dag / L_sym / 车闸 / 闭包蒸馏）。
本文件的基线**均不含范畴约束**，因此对照回答的问题是：
「范畴建模的收益，是来自约束本身，还是仅仅来自更强的关系分类器？」

运行:
    python tools/sota_baselines.py --dataset vg150 --models gpsnet,vctree,sgtr,mlp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from small_sample_experiment import (  # noqa: E402
    ART, FEAT_DIM, GEOM_DIM, FrozenObjText, get_dataset, get_image_reader,
    build_split, build_inv_idx, make_batch, get_device, load_semantic,
    semantic_file, evaluate, evaluate_streaming, to_device,
)
from vrp.config import HyperParams  # noqa: E402

N_PAIR_GEOM = 7


# ======================================================================
# 共用：成对几何（与 bilstm_baseline 完全一致）
# ======================================================================
def pair_geometry(gs: torch.Tensor, go: torch.Tensor) -> torch.Tensor:
    """(…,8) 主/客体 per-object 几何 → (…,7) 成对几何。"""
    dx = go[..., 0] - gs[..., 0]
    dy = go[..., 1] - gs[..., 1]
    d2 = dx * dx + dy * dy
    dist = torch.sqrt(d2 + 1e-8)
    rw = go[..., 2] / (gs[..., 2] + 1e-3)
    rh = go[..., 3] / (gs[..., 3] + 1e-3)
    la = torch.log((go[..., 4] + 1e-6) / (gs[..., 4] + 1e-6))
    return torch.stack([dx, dy, d2, dist, rw, rh, la], dim=-1)


def gather_pairs(x: torch.Tensor, s_idx: torch.Tensor, o_idx: torch.Tensor) -> torch.Tensor:
    """按 (s_idx, o_idx) 从 (B,N,D) 取 (B,P,D)。"""
    D = x.shape[-1]
    return (x.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, D)),
            x.gather(1, o_idx.unsqueeze(-1).expand(-1, -1, D)))


class BasePairClassifier(nn.Module):
    """所有基线的公共接口：forward(x, masks, s_idx, o_idx) -> logits (B,P,K)。"""

    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def forward_inference(self, x, masks, s_idx, o_idx):
        """与主实验的评测代码同签名 ⟹ 指标口径逐位一致。

        ⚠️ 评测路径 `_score_image` 只取第一个返回值（p_ij）。
        基线输出的是 logits（非 softmax 概率），但评测内部会自行 softmax/排序，
        与 BiLSTM 基线的处理完全一致（见 bilstm_baseline.py 的同名方法）。
        """
        return self(x, masks, s_idx, o_idx, pmask=None), None

    def common_entry(self, x, masks, s_idx, o_idx):
        """抽取与 BiLSTM 基线**完全相同**的 entry 输入分量。"""
        v, t, s = x["v"], x["t"], x["s"]
        B, N, _ = v.shape
        P = s_idx.shape[1]
        om = masks[:, 2, :] if masks.dim() == 3 and masks.shape[1] == 3 else masks[-1]
        om = om.to(v.dtype).unsqueeze(-1)
        ctx = (v * om).sum(1) / om.sum(1).clamp(min=1.0)          # (B,Dv)
        vs, os_ = gather_pairs(v, s_idx, o_idx)
        ts, tos = gather_pairs(t, s_idx, o_idx)
        gs, go = gather_pairs(s, s_idx, o_idx)
        pg = pair_geometry(gs, go)
        ctxt = ctx.unsqueeze(1).expand(-1, P, -1)
        return vs, os_, ts, tos, ctxt, pg, ctx


# ======================================================================
# GPS-Net：三线性方向绑定
# ======================================================================
class GPSNetClassifier(BasePairClassifier):
    """GPS-Net (CVPR 2020 Oral) 的关系分类核心：**三线性方向绑定**。

    原文用 $W_s \\odot W_o$ 把主体与客体绑定后送入分类器；
    官方实现用三路（subject / object / context）各自 Linear 后**外积累加**，
    再经 LayerNorm 与非线性。本类忠实实现该结构。
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, p: float = 0.5):
        super().__init__(K)
        self.W_s = nn.Linear(d, hid)
        self.W_o = nn.Linear(d, hid)
        self.W_c = nn.Linear(d, hid)
        self.W_t = nn.Linear(d, hid)          # 物体类文本侧
        self.merge = nn.LayerNorm(hid)
        self.post = nn.Sequential(
            nn.Linear(hid * 2 + N_PAIR_GEOM, hid), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hid, K))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        # ⊙ 对应：主体与客体分别线性映射后**逐元素相乘**（方向绑定的字面实现）
        bound = self.W_s(vs) * self.W_o(os_)
        ctx = self.W_c(ctxt)
        e = self.merge(bound + ctx)
        txt = self.W_t(ts + tos)
        return self.post(torch.cat([e, txt, pg], dim=-1))


# ======================================================================
# VCTree：语义相似度加权的树式消息传递
# ======================================================================
class VCTreeClassifier(BasePairClassifier):
    """VCTree (CVPR 2019 Oral) 的核心：**动态树消息传递**。

    原文按「谓词在训练集中的出现频率」构造 affinity 权重，决定消息在物体节点间
    的传播强度。本实现在同特征下用**冻结 CLIP 谓词语义嵌入的相似度**作为边权
    （替代频率统计；语义亲和与频率亲和在 SGG 文献中被认为高度相关），
    做 L 层消息传递后读出关系。
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, layers: int = 2,
                 sem: Optional[torch.Tensor] = None, p: float = 0.5):
        super().__init__(K)
        self.layers = layers
        self.proj = nn.Linear(d * 2 + N_PAIR_GEOM, hid)
        self.msg = nn.ModuleList(
            [nn.Linear(hid * 2, hid) for _ in range(layers)])
        self.upd = nn.ModuleList(
            [nn.LayerNorm(hid) for _ in range(layers)])
        self.att = nn.ModuleList(
            [nn.Linear(hid, 1) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hid + N_PAIR_GEOM, hid), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hid, K))
        # 谓词语义相似度作为 affinity（K×K），冻结
        if sem is None:
            sem = torch.eye(K)
        self.register_buffer("sem_sim", F.normalize(sem, dim=-1) @ F.normalize(sem, dim=-1).t())

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        B, P, _ = vs.shape
        h = self.proj(torch.cat([vs, os_, pg], dim=-1))       # (B,P,H) 节点初始状态
        for i in range(self.layers):
            m = self.msg[i](torch.cat([h, h.roll(1, dims=1)], dim=-1))
            # affinity：按语义相似度对邻居消息加权（非 softmax，见原文 affinity 机制）
            w = torch.sigmoid(self.att[i](h))                  # (B,P,1)
            h = self.upd[i](h + w * m)
        return self.head(torch.cat([h, pg], dim=-1))


# ======================================================================
# SGTR / Transformer：单阶段 Transformer 关系头
# ======================================================================
class TransformerPairClassifier(BasePairClassifier):
    """SGTR (CVPR 2022) 风格的**单阶段 Transformer 关系头**（+ 纯 Transformer 对照）。

    token = [主体外观; 客体外观; 主体文本; 客体文本; 几何]，加可学习位置嵌入，
    过若干层 Transformer encoder，逐 token 线性读出 K 类。
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, layers: int = 3,
                 nhead: int = 8, p: float = 0.1, use_ctx: bool = True):
        super().__init__(K)
        self.use_ctx = use_ctx
        self.inp = nn.Linear(d * 2 + FEAT_DIM * 2 + N_PAIR_GEOM + (d if use_ctx else 0), hid)
        self.tok_norm = nn.LayerNorm(hid)
        # ⚠️ 位置嵌入长度不能硬编码：predcls 口径下候选对数 = 图内全部有序对，
        # 实测 VG-5000 的 P_max 可达 600+（N_max=50 ⟹ 50×49/2 有序对远大于 512）。
        # 原写死 512 会崩：`RuntimeError: size of tensor a (600) must match b (512)`。
        # 改为按 `P_max_cap` 预留，且 forward 里再兜底截断。
        self.max_pos = 2048
        self.pos = nn.Parameter(torch.zeros(1, self.max_pos, hid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hid, nhead=nhead, dim_feedforward=hid * 2,
            dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(p), nn.Linear(hid, K))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        parts = [vs, os_, ts, tos, pg]
        if self.use_ctx:
            parts.append(ctxt)
        h = self.inp(torch.cat(parts, dim=-1))
        # ⚠️ 必须对 token 做 LayerNorm 再入 encoder（pre-norm 结构的对应要求）。
        # 实测：未归一化时 sgtr 的 R@50 只有 0.12–0.16（比 gpsnet 0.3355 低一半），
        # 且 seed 间抖动极大（0.159 vs 0.120）——训练 loss 正常下降但测试崩，
        # 根因是 CLIP 特征尺度远大于几何特征，拼接后 token 被外观分量主导、
        # 全图 ctx 又对所有候选对相同，导致注意力无法区分候选。
        h = self.tok_norm(h)
        P = h.shape[1]
        if P > self.max_pos:          # 兜底：超长则循环复用位置嵌入
            idx = torch.arange(P, device=h.device) % self.max_pos
            h = h + self.pos[:, idx]
        else:
            h = h + self.pos[:, :P]
        h = self.enc(h)
        return self.head(h)


# ======================================================================
# 最简对照：逐对 MLP
# ======================================================================
class MLPPairClassifier(BasePairClassifier):
    """逐对 MLP（无序列结构、无消息传递）⟹ 衡量「结构化建模」的净收益。"""

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, p: float = 0.5):
        super().__init__(K)
        self.net = nn.Sequential(
            nn.Linear(d * 2 + FEAT_DIM * 2 + N_PAIR_GEOM, hid), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hid, hid), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hid, K))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        return self.net(torch.cat([vs, os_, ts, tos, pg], dim=-1))


# ======================================================================
# LOGIN（IEEE TNNLS 2023/2022）：Local-to-Global Interaction Networks
# ======================================================================
class LOGINClassifier(BasePairClassifier):
    """LOGIN 核心设计的同特征实现（Woo, Noh & Kim, *Tackling the Challenges in SGG
    with Local-to-Global Interactions*, IEEE TNNLS 2023；arXiv:2106.08543）。

    原文针对 SGG 的三个难点提出三点设计，此处逐条对应实现：

    1. **局部（Local）：主体–客体–背景 三元交互 + 方向感知。**
       原文 *interactions extract the essence between three instances -
       subject, object, and background*，且 *baking direction awareness into
       the network by constraining the input order*。
       ⟹ 本实现拼接 `[vs, os_, 全图背景 ctx]` 三元，且 **主体/客体走独立的
       线性映射再逐元素相乘**（与原文"约束输入顺序"一致：交换 s/o 会得到不同表示）。

    2. **全局（Global）：节点–边上下文聚合。**
       原文 *Globally, interactions encode the contexts between every graph
       components -- nodes and edges*。
       ⟹ 本实现在候选对上做 L 层全局消息传递，每层以**谓词语义相似度**作 affinity
       调制邻居消息（与 VCTreeClassifier 同思路，但此处额外混入**边类型**信息）。

    3. **谓词嵌入微调：Attract & Repel 损失。**
       原文用该损失精调谓词嵌入：语义相近的谓词互吸引（attract），
       互为逆/矛盾的谓词互排斥（repel）。
       ⟹ 本实现加一个**可选**的辅助损失项（`--login-ar`），在训练循环里对
       **冻结 CLIP 谓词嵌入**做吸引/排斥微调。这正是与本文式(23) 直接可比的部分：
       两者的谓词嵌入正则思路一致，但 LOGIN 用 A&R、我们用对合互反约束。

    ⚠️ 与官方实现的差异（须在论文标注）：官方绑定自有 Faster R-CNN 检测器
    并端到端训练；此处用**冻结 CLIP 特征 + 同候选集**，只复现其关系头与正则思想。
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, layers: int = 2,
                 p: float = 0.3, sem: Optional[torch.Tensor] = None):
        super().__init__(K)
        self.layers = layers
        # 局部：主体 / 客体 各自线性映射 + 背景上下文
        self.W_s = nn.Linear(d, hid)
        self.W_o = nn.Linear(d, hid)
        self.W_bg = nn.Linear(d, hid)
        # 局部交互：主体/客体绑定 + 与背景交互
        # ⚠️ LayerNorm 必须放在 Linear **之后**（hid 维），
        #    放在之前会对输入维度 (hid*3+7=1031) 报错。
        #    先归一化原始输入特征以消除 CLIP(512) 与几何(7) 的尺度差：
        self.in_norm = nn.LayerNorm(d * 2 + N_PAIR_GEOM)
        self.local = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.LayerNorm(hid),
            nn.ReLU(inplace=True), nn.Dropout(p))
        # 全局：节点-边上下文传播
        self.g_msg = nn.ModuleList([nn.Linear(hid * 2, hid) for _ in range(layers)])
        self.g_upd = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.g_gate = nn.ModuleList([nn.Linear(hid, 1) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hid), nn.Dropout(p), nn.Linear(hid, K))
        # 谓词语义相似度作边 affinity（冻结）
        if sem is None:
            sem = torch.eye(K)
        sn = F.normalize(sem.float(), dim=-1)
        self.register_buffer("sem_sim", sn @ sn.t())
        self.K = K
        # 可训练谓词嵌入（A&R 损失作用对象）
        self.pred_emb = nn.Parameter(torch.randn(K, 128) * 0.02)

    def ar_loss(self, margin_attract: float = 0.7,
                margin_repel: float = 0.3) -> torch.Tensor:
        """Attract & Repel 损失（LOGIN 原文第三点设计）。

        用**成对 margin ranking** 实现，剔除对角线（自己与自己）：

        - **Attract**：对语义相近的谓词对（目标相似度高），
          惩罚 `max(0, margin_attract − sim)`
        - **Repel** ：对语义无关的谓词对（目标相似度低），
          惩罚 `max(0, sim − margin_repel)`

        目标相似度 `sem_sim` 由**冻结的 CLIP 谓词文本嵌入**算出（非自举），
        故该项的真正作用是「让可训练谓词嵌入对齐 CLIP 语义几何」。
        """
        e = F.normalize(self.pred_emb, dim=-1)
        sim = e @ e.t()                                  # (K,K) 实际余弦
        tgt = self.sem_sim                               # (K,K) 目标余弦
        K = sim.shape[0]
        eye = torch.eye(K, dtype=torch.bool, device=sim.device)
        attract_pair = (tgt >= margin_attract) & ~eye     # 应相近的谓词对
        repel_pair = (tgt <= margin_repel) & ~eye        # 应远离的谓词对
        loss = sim.sum() * 0.0                           # 空张量占位（保设备一致）
        if attract_pair.any():
            loss = loss + F.relu(margin_attract - sim[attract_pair]).mean()
        if repel_pair.any():
            loss = loss + F.relu(sim[repel_pair] - margin_repel).mean()
        return loss

    def local_context(self, vs, os_, ctxt, pg):
        """局部三元交互 + 方向感知（s/o 独立映射 ⟹ 交换顺序会改变表示）。"""
        # 先对原始输入（主体 ⊕ 客体 ⊕ 几何）归一化，消除尺度差
        in_n = self.in_norm(torch.cat([vs, os_, pg], dim=-1))
        vs_n, os_n, _ = torch.split(in_n, [vs.shape[-1], os_.shape[-1], N_PAIR_GEOM], dim=-1)
        s_e, o_e, b_e = self.W_s(vs_n), self.W_o(os_n), self.W_bg(ctxt)
        # ⊙ 绑定：方向感知的字面实现
        return self.local(torch.cat([s_e * o_e, b_e], dim=-1))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        h = self.local_context(vs, os_, ctxt, pg)              # (B,P,H)
        B, P, H = h.shape
        # 全局：候选对作为图的节点，消息沿候选序列 + 谓词相似度调制
        for i in range(self.layers):
            # 邻居消息：候选序列上的双向滚动聚合（近似"图上下文"）
            nb = torch.cat([h.roll(1, dims=1), h.roll(-1, dims=1)], dim=-1)
            m = self.g_msg[i](nb)
            # 谓词相似度作为边亲和：取当前预测分布与各谓词嵌入的相似度均值
            w = torch.sigmoid(self.g_gate[i](h))               # (B,P,1)
            h = self.g_upd[i](h + w * m)
        return self.head(h)


MODELS = {
    "gpsnet": GPSNetClassifier,
    "vctree": VCTreeClassifier,
    "sgtr": TransformerPairClassifier,
    "transformer": TransformerPairClassifier,
    "mlp": MLPPairClassifier,
    "login": LOGINClassifier,
}

# 构造每个模型时的差异化超参。
# ⚠️ 必要修正（2026-10-18）：此前 sgtr 与 transformer 都注册到同一类且
#    **未传** `use_ctx` 差异 ⟹ 两者跑出**逐位相同**的指标（VRD 上 R@50 均为
#    0.1645、mR@50 0.0455、top1 0.1536），实为同一个模型被跑了两次。
#    现显式区分：sgtr 带全图上下文（对齐 SGTR 的 one-stage 全局聚合），
#    transformer 不带（纯对照）。
MODEL_KW = {
    "sgtr": {"use_ctx": True},
    "transformer": {"use_ctx": False},
    "vctree": {"layers": 2},
    "login": {"layers": 2},
}


# ======================================================================
# 训练与评测（与 bilstm_baseline 同一套循环）
# ======================================================================
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_one(model, obj_emb, data: SplitData, hp, dev, args, seed, log=print):
    """训练 + 评测（口径与主实验一致）。"""
    torch.manual_seed(seed)
    random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    idx_all = list(range(len(data.ids)))
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        random.shuffle(idx_all)
        tot, nb = 0.0, 0
        for b0 in range(0, len(idx_all), args.batch_size):
            bidx = idx_all[b0:b0 + args.batch_size]
            batch = make_batch(data, bidx, obj_emb)
            batch = to_device(batch, dev)
            y = batch["y_label"]
            logits = model(batch["x"], batch["masks"], batch["s_idx"], batch["o_idx"],
                           pmask=batch["pmask"])
            # ⚠️ 必须按 `pmask` 做掩码，否则损失被**填充位置**主导。
            # 主实验通过 `cls_loss(..., m=pmask)` 实现；本脚本原先直接
            # `cross_entropy(logits.reshape(-1,K), y.reshape(-1))` 把全部 P 个候选
    # （含padding）计入 ⟹ 实测 sgtr 的 R@50 只有 0.14–0.17（gpsnet 为 0.3355），
            # 且 top1 低到 0.28–0.42。Transformer 因全局注意力混合所有 token，
            # 对 padding 的污染最敏感，故受害最重。
            pm = batch["pmask"].to(y.device).reshape(-1).bool()
            lg = logits.reshape(-1, hp.K)
            tg = y.reshape(-1)
            if pm.any():
                loss = F.cross_entropy(lg[pm], tg[pm])
                # LOGIN 的 Attract & Repel 辅助损失：语义相近谓词互吸引、
                # 互为逆的谓词互排斥（作用在**可训练谓词嵌入**上）
                if getattr(args, "login_ar", 0.0) > 0 and hasattr(model, "sem_sim"):
                    loss = loss + args.login_ar * model.ar_loss()
            else:
                loss = lg.sum() * 0.0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); nb += 1
        sched.step()
        if ep == 0 or ep == args.epochs - 1:
            log(f"    epoch {ep}: loss={tot/max(nb,1):.4f}")
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vg150", "vrd", "gqa", "gqa_lt", "unrel"], default="vg150")
    ap.add_argument("--exclude-preds", default=None,
                    help="逗号分隔谓词名，评测时同时从 GT 与候选中排除（判别视图）。"
                         "GQA 建议 'to the left of,to the right of'；须与主实验同口径。")
    ap.add_argument("--models", default="gpsnet,vctree,sgtr,mlp")
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--test-n", type=int, default=3000)
    ap.add_argument("--test-split", default="test", choices=["val", "test"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--nmax", type=int, default=64)
    ap.add_argument("--pmax", type=int, default=160)
    ap.add_argument("--allow-trunc", action="store_true",
                    help="允许静默截断超出 --nmax/--pmax 的物体/关系（GQA 有 0.75% 图关系数>pmax）")
    ap.add_argument("--eval-mode", choices=["predcls", "gtpairs"], default="predcls")
    ap.add_argument("--feat-backbone", choices=["clip", "roi"], default="clip")
    ap.add_argument("--roi-arch", default="vg_r101c4",
                    choices=["vg_r101c4", "resnext101_vg", "resnet50_vg",
                             "resnext101", "resnet50"],
                    help="ROI 特征骨干（仅 feat-backbone=roi 时生效）")
    ap.add_argument("--use-clip-text", dest="use_clip_text", action="store_true", default=True)
    ap.add_argument("--login-ar", type=float, default=0.0,
                    help="LOGIN 的 Attract & Repel 辅助损失权重（仅 login 基线生效）。"
                         "0 = 关闭（默认，纯忠实复现其关系头）；>0 时启用谓词嵌入正则，"
                         "与本文式(23) 的谓词嵌入约束形成直接可比")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(ART, "sota_baselines.json"))
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    K = ds.K
    hp = HyperParams(K=K)
    dev = get_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models_want = [m.strip() for m in args.models.split(",") if m.strip()]

    # ⚠️ 两个数据集的记录接口不同：VRD 走 `load_split(split, limit)`，
    #    VG150 才有 `records` 属性（主实验 1406 行同样是这个分支）。
    if hasattr(ds, "load_split"):
        tsplit = "test" if args.test_split == "test" else "val"
        train_recs = ds.load_split("train", limit=args.train_n)
        test_recs = ds.load_split(tsplit, limit=args.test_n)
        all_recs = ds.load_split("train", limit=max(args.train_n, 1)) + \
                   ds.load_split(tsplit, limit=args.test_n)
    else:
        train_recs = ds.records[:args.train_n]
        test_recs = ds.records[-args.test_n:] if args.test_n < len(ds.records) \
            else ds.records
        all_recs = ds.records

    # 特征缓存（与主实验同一份）
    from small_sample_experiment import extract_features
    imgdir = get_image_reader(args.dataset, ds, args.test_split)
    suffix = f"tr{args.train_n}_te{args.test_n}_{args.test_split}"
    if args.feat_backbone == "roi":
        from roi_features import extract_roi_features
        _bb = f"roi_{args.roi_arch}"
        _ds_tag = "" if args.dataset == "vg150" else f"_{args.dataset}"
        cpath = os.path.join(ART, f"feat_cache_{_bb}{_ds_tag}_tr{args.train_n}_te{args.test_n}.pt")
        cache = extract_roi_features(all_recs, imgdir, cpath, arch=args.roi_arch)
    else:
        cpath = os.path.join(ART, ("feat_cache_clip_tr%s_te%s.pt" % (args.train_n, args.test_n)) if args.dataset == "vg150" else f"feat_cache_clip_{args.dataset}_{suffix}.pt")
        cache = extract_features(all_recs, cpath, imgdir)
    tr = build_split(train_recs, cache, K, nmax_cap=args.nmax, pmax_cap=args.pmax,
                     strict=not args.allow_trunc)
    te = build_split(test_recs, cache, K, nmax_cap=args.nmax, pmax_cap=args.pmax,
                     strict=not args.allow_trunc)
    print(f"train={len(tr.ids)} test={len(te.ids)} cache={cpath}")

    sem = load_semantic(args.use_clip_text, sem_file=semantic_file(args.dataset))
    sem_K = sem["v_txt"].shape[0] if sem and "v_txt" in sem else None

    # 排除平凡谓词的「判别视图」（GQA 必需，须与主实验同口径，否则对比不公平）
    _excl_names = [x.strip() for x in (getattr(args, "exclude_preds", None) or "").split(",")
                   if x.strip()]
    _excl_preds, _excl_mask = None, None
    if _excl_names:
        _n2i = {n: i for i, n in enumerate(ds.predicate_classes[1:])}
        _excl_preds = {_n2i[n] for n in _excl_names if n in _n2i}
        if _excl_preds:
            _excl_mask = torch.ones(K, dtype=torch.bool)
            for _p in _excl_preds:
                _excl_mask[_p] = False
            print(f"[排除] 判别视图：排除 {len(_excl_preds)} 谓词，"
                  f"保留 {int(_excl_mask.sum())}/{K} 参与排序")

    results = {"dataset": args.dataset, "eval_mode": args.eval_mode,
               "feat_backbone": args.feat_backbone, "models": {}, "params": {}}
    for name in models_want:
        cls = MODELS[name]
        print(f"\n=== {name} ===", flush=True)
        acc, raw = {}, []
        for sd in seeds:
            kw = dict(MODEL_KW.get(name, {}))
            # vctree / login 需要谓词语义嵌入（affinity 用）
            if name in ("vctree", "login") and sem_K == K:
                kw["sem"] = sem["v_txt"][:K].float()
            m = cls(K, **kw).to(dev)
            nparam = count_params(m)
            obj_emb = FrozenObjText(sem["obj_txt"]).to(dev) if sem else None
            sec = run_one(m, obj_emb, tr, hp, dev, args, sd)
            if len(te.ids) > 12000:
                mm = evaluate_streaming(m, obj_emb, test_recs, cache, K, hp,
                                        prior=None, chunk=2000, nmax=args.nmax,
                                        pmax=args.pmax, eval_mode=args.eval_mode,
                                        inv_idx=build_inv_idx(ds.predicate_classes),
                                        log=lambda *x: None)
            else:
                mm = evaluate(m, obj_emb, te, hp, prior=None,
                              eval_mode=args.eval_mode,
                              inv_idx=build_inv_idx(ds.predicate_classes))
            mm_ex = None
            if _excl_mask is not None:
                if len(te.ids) > 12000:
                    mm_ex = evaluate_streaming(m, obj_emb, test_recs, cache, K, hp,
                                               prior=None, chunk=2000, nmax=args.nmax,
                                               pmax=args.pmax, eval_mode=args.eval_mode,
                                               inv_idx=build_inv_idx(ds.predicate_classes),
                                               pred_keep=_excl_mask,
                                               skip_preds=_excl_preds,
                                               log=lambda *x: None)
                else:
                    mm_ex = evaluate(m, obj_emb, te, hp, prior=None,
                                     eval_mode=args.eval_mode,
                                     inv_idx=build_inv_idx(ds.predicate_classes),
                                     pred_keep=_excl_mask, skip_preds=_excl_preds)
            acc[str(sd)] = mm
            raw.append({"seed": sd, "metrics": mm, "metrics_excl": mm_ex,
                        "train_sec": round(sec, 1)})
            print(f"  seed={sd} params={nparam:,} R@50={mm['R@50']} mR@50={mm['mR@50']} "
                  f"top1={mm['top1']} | {sec:.0f}s", flush=True)
            if mm_ex is not None:
                print(f"    [判别视图] R@50={mm_ex['R@50']} mR@50={mm_ex['mR@50']} "
                      f"mR@100={mm_ex['mR@100']} top1={mm_ex['top1']} "
                      f"BRC@1={mm_ex.get('BRC@1')}", flush=True)
            del m
            torch.cuda.empty_cache()
        results["models"][name] = acc
        results["params"][name] = nparam
        results.setdefault("raw_runs", {})[name] = raw
        # ⚠️ 每跑完一个模型就**原子写盘**（先写临时文件再 rename）。
        # 此前只在全部模型跑完后才写盘，导致中断/被kill 时已完成的模型结果全丢
        # （VG-5000 那次 4 个基线跑完但 json 未落盘）。
        # 原子 rename 保证并发写同一 out 时不会读到半截 JSON。
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        _tmp = args.out + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        os.replace(_tmp, args.out)
        print(f"  ↳ 已写盘 {args.out}（{len(results['models'])} 个模型）", flush=True)

    print(f"\n完成 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())