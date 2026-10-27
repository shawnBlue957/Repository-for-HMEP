"""SOTA is a baseline set (same CLIP features and evaluation caliber as the main experiment**).

## Why should you implement it yourself (following the idea of bilstm_baseline.py)

The literature values of SOTA cannot be rerun in this project pipeline: all their weights are hosted on Google Drive.
And they all rely on the same VG training detector checkpoint. So **on the same set of frozen CLIP features**
Implement the **core relationship classifier** of the SOTA system by yourself, so that a row appears in the table
”The SOTA baseline run by ourselves on the same pipeline” directly eliminates the doubt that ”the backbone is different and therefore incomparable”.

## Fairness contract (every baseline must be met)

1. **Features are exactly the same**: `x_v` = frozen CLIP object cropping feature, `x_t` = frozen CLIP object class text embedding,
   `x_s` = geometry (8 dimensions), bit-by-bit identical to the main experiment arm A (shares `feat_cache_*.pt`).
2. **The evaluation is completely consistent**: directly import `evaluate` / `evaluate_streaming` / `_finalize`,
   Therefore, the caliber of R@K / mR@K / BRC / IR@K is consistent with the main experiment** bit by bit, without any self-written indicators.
3. **Candidate set is consistent**: `--eval-mode predcls` The next candidate is all ordered pairs in the graph.
4. **Trainable parameters are comparable**: Each baseline is controlled at ~1.5–2.5M, which is the same magnitude as the A arm of the main experiment (1.68M).
   Avoid the confusion of ”the baseline is bigger so it’s worse/better”.

## Documents and core designs corresponding to each baseline

| tag | documentation | core design (faithful simplification of the same features) |
|---|---|---|
| `bilstm` | Neural Motifs, CVPR 2018 | Sequence sorted by subject x → BiLSTM propagation context → pairwise softmax (implemented in bilstm_baseline.py) |
| `gpsnet` | GPS-Net, CVPR 2020 (Oral) | **Trilinear direction binding**: The three characteristics of subject/object/context are each passed through Linear and then **external accumulation**, and then through LayerNorm; the original text of GPS-Net is $W_s\\odot W_o$, which has the same origin as $\\alpha\\odot f_s,\\beta\\odot f_o$ in this formula (4) |
| `vctree` | VCTree, CVPR 2019 (Oral) | **Dynamic tree message passing**: Messages with predicates as hyperedges are propagated between object nodes according to **semantic similarity weighting** (using frozen CLIP text embedding to calculate predicate semantic similarity as edge weight), classified after multi-layer aggregation; corresponding to the affinity-based message passing of the original text |
| `sgtr` | SGTR, CVPR 2022 | **Single-stage Transformer relation header**: Encode each candidate pair into a token (including subject/object/union box/geometry), pass through several layers of Transformer encoder, and then classify |
| `transformer` | General SOTA comparison | Control group: pure Transformer encoder (without single-stage details of SGTR), used to judge the relative contribution of ”the architecture itself” and ”the scope constraints of this article” |
| `mlp` | Simplest comparison | Pairwise MLP (no sequence, no message passing) ⟹ Measuring the net benefit of ”structured modeling” |

## Comparison with the main experiment

Arm A of the main experiment = the full amount of formulas (1)–(13) (including category constraints L_dag / L_sym / brake / closure distillation).
None of the baselines in this document contain category constraints, so the questions answered are:
“Do the benefits of category modeling come from the constraints themselves, or just from stronger relational classifiers?”

Run:
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
# Common: pairwise geometry (exactly the same as bilstm_baseline)
# ======================================================================
def pair_geometry(gs: torch.Tensor, go: torch.Tensor) -> torch.Tensor:
    """(…,8) subject/object per-object geometry → (…,7) pairwise geometry."""
    dx = go[..., 0] - gs[..., 0]
    dy = go[..., 1] - gs[..., 1]
    d2 = dx * dx + dy * dy
    dist = torch.sqrt(d2 + 1e-8)
    rw = go[..., 2] / (gs[..., 2] + 1e-3)
    rh = go[..., 3] / (gs[..., 3] + 1e-3)
    la = torch.log((go[..., 4] + 1e-6) / (gs[..., 4] + 1e-6))
    return torch.stack([dx, dy, d2, dist, rw, rh, la], dim=-1)


def gather_pairs(x: torch.Tensor, s_idx: torch.Tensor, o_idx: torch.Tensor) -> torch.Tensor:
    """Get (B,P,D) from (B,N,D) by (s_idx, o_idx)."""
    D = x.shape[-1]
    return (x.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, D)),
            x.gather(1, o_idx.unsqueeze(-1).expand(-1, -1, D)))


class BasePairClassifier(nn.Module):
    """Common interface for all baselines: forward(x, masks, s_idx, o_idx) -> logits (B,P,K)."""

    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def forward_inference(self, x, masks, s_idx, o_idx):
        """The same signature as the evaluation code of the main experiment ⟹ The index caliber is consistent bit by bit.

        ⚠️ The evaluation path `_score_image` only takes the first return value (p_ij).
        The baseline output is logits (non-softmax probability), but the evaluation will softmax/sort itself internally.
        Exactly the same processing as the BiLSTM baseline (see the method of the same name in bilstm_baseline.py).
        """
        return self(x, masks, s_idx, o_idx, pmask=None), None

    def common_entry(self, x, masks, s_idx, o_idx):
        """Extract entry input components that are **exactly** the same as the BiLSTM baseline."""
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
# GPS-Net: trilinear direction binding
# ======================================================================
class GPSNetClassifier(BasePairClassifier):
    """The relationship classification core of GPS-Net (CVPR 2020 Oral): **Trilinear Orientation Binding**.

    The original text uses $W_s \\odot W_o$ to bind the subject and object and send them to the classifier;
    The official implementation uses three channels (subject/object/context) to accumulate and add after each Linear.
    Then go through LayerNorm and nonlinearity. This class faithfully implements this structure.
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, p: float = 0.5):
        super().__init__(K)
        self.W_s = nn.Linear(d, hid)
        self.W_o = nn.Linear(d, hid)
        self.W_c = nn.Linear(d, hid)
        self.W_t = nn.Linear(d, hid)          # Object text side
        self.merge = nn.LayerNorm(hid)
        self.post = nn.Sequential(
            nn.Linear(hid * 2 + N_PAIR_GEOM, hid), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hid, K))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        # ⊙ Correspondence: subject and object are linearly mapped respectively and then **element-wise multiplication** (literal implementation of directional binding)
        bound = self.W_s(vs) * self.W_o(os_)
        ctx = self.W_c(ctxt)
        e = self.merge(bound + ctx)
        txt = self.W_t(ts + tos)
        return self.post(torch.cat([e, txt, pg], dim=-1))


# ======================================================================
# VCTree: Semantic similarity weighted tree messaging
# ======================================================================
class VCTreeClassifier(BasePairClassifier):
    """The core of VCTree (CVPR 2019 Oral): **Dynamic Tree Messaging**.

    The original article constructs affinity weights based on ”the frequency of occurrence of the predicate in the training set” to determine the message distribution between object nodes.
    the intensity of transmission. This implementation uses the similarity of frozen CLIP predicate semantic embeddings as edge weights under the same characteristics.
    (an alternative to frequency statistics; semantic affinity and frequency affinity are considered highly correlated in the SGG literature),
    Read out the relationship after doing L-layer messaging.
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
        # Predicate semantic similarity as affinity (K×K), frozen
        if sem is None:
            sem = torch.eye(K)
        self.register_buffer("sem_sim", F.normalize(sem, dim=-1) @ F.normalize(sem, dim=-1).t())

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        B, P, _ = vs.shape
        h = self.proj(torch.cat([vs, os_, pg], dim=-1))       # (B,P,H) node initial state
        for i in range(self.layers):
            m = self.msg[i](torch.cat([h, h.roll(1, dims=1)], dim=-1))
            # Affinity: weight neighbor messages according to semantic similarity (not softmax, see the original affinity mechanism)
            w = torch.sigmoid(self.att[i](h))                  # (B,P,1)
            h = self.upd[i](h + w * m)
        return self.head(torch.cat([h, pg], dim=-1))


# ======================================================================
# SGTR / Transformer: Single-stage Transformer relationship header
# ======================================================================
class TransformerPairClassifier(BasePairClassifier):
    """SGTR (CVPR 2022) style **single-stage Transformer relation header** (+ pure Transformer comparison).

    token = [subject appearance; object appearance; subject text; object text; geometry], plus learnable position embedding,
    After several layers of Transformer encoder, K categories are linearly read out token by token.
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, layers: int = 3,
                 nhead: int = 8, p: float = 0.1, use_ctx: bool = True):
        super().__init__(K)
        self.use_ctx = use_ctx
        self.inp = nn.Linear(d * 2 + FEAT_DIM * 2 + N_PAIR_GEOM + (d if use_ctx else 0), hid)
        self.tok_norm = nn.LayerNorm(hid)
        # ⚠️ The position embedding length cannot be hard-coded: the number of candidate pairs under the predcls caliber = all ordered pairs in the picture,
        # The measured P_max of VG-5000 can reach 600+ (N_max=50 ⟹ 50×49/2 ordered pair is much larger than 512).
        # The original code 512 will crash: `RuntimeError: size of tensor a (600) must match b (512)`.
        # Instead, reserve it according to `P_max_cap`, and then truncate it in forward.
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
        # ⚠️ A LayerNorm reentry encoder must be applied to the token (corresponding requirement of the pre-norm structure).
        # Actual measurement: The R@50 of sgtr without normalization is only 0.12–0.16 (half lower than gpsnet 0.3355),
        # And the jitter between seeds is extremely large (0.159 vs 0.120) - the training loss decreases normally but the test crashes,
        # The root cause is that the scale of CLIP features is much larger than that of geometric features. After splicing, the token is dominated by the appearance component.
        # The ctx of the whole image is the same for all candidate pairs, causing attention to be unable to distinguish candidates.
        h = self.tok_norm(h)
        P = h.shape[1]
        if P > self.max_pos:          # Bottom line: If the length is too long, the location will be reused and embedded.
            idx = torch.arange(P, device=h.device) % self.max_pos
            h = h + self.pos[:, idx]
        else:
            h = h + self.pos[:, :P]
        h = self.enc(h)
        return self.head(h)


# ======================================================================
# The simplest comparison: pairwise MLP
# ======================================================================
class MLPPairClassifier(BasePairClassifier):
    """Pairwise MLP (no sequence structure, no message passing) ⟹ Measure the net benefit of “structured modeling”."""

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
# LOGIN (IEEE TNNLS 2023/2022): Local-to-Global Interaction Networks
# ======================================================================
class LOGINClassifier(BasePairClassifier):
    """Implementation of the same features of LOGIN core design (Woo, Noh & Kim, *Tackling the Challenges in SGG
    with Local-to-Global Interactions*, IEEE TNNLS 2023; arXiv:2106.08543).

    The original article proposed three designs for the three difficulties of SGG, which are implemented one by one here:

    1. **Local: subject-object-background three-dimensional interaction + direction perception. **
       Original text *interactions extract the essence between three instances -
       subject, object, and background*, and *baking direction awareness into
       the network by constraining the input order*.
       ⟹ This implementation splices `[vs, os_, full image background ctx]` ternary, and **subject/object are independent
       The linear mapping is then multiplied element-wise** (consistent with the original ”Constrain input order”: exchanging s/o will result in different representations).

    2. **Global: node-edge context aggregation. **
       Original text *Globally, interactions encode the contexts between every graph
       components -- nodes and edges*.
       ⟹ This implementation performs L-layer global message passing on candidate pairs, and each layer uses **predicate semantic similarity** as affinity
       Modulate neighbor messages (same idea as VCTreeClassifier, but additional edge type information is mixed here).

    3. **Predicate embedding fine-tuning: Attract & Repel loss. **
       The original article uses this loss to fine-tune predicate embeddings: predicates with similar semantics attract each other,
       Predicates that are inverse/contradictory to each other are mutually exclusive (repel).
       ⟹ This implementation adds an **optional** auxiliary loss term (`--login-ar`) to
       **Freeze CLIP predicate embedding** Do attraction/repulsion fine-tuning. This is the part directly comparable to equation (23) in this article:
       The predicate embedding regular ideas of the two are the same, but LOGIN uses A&R, and we use involution and reciprocity constraints.

    ⚠️ Differences from the official implementation (must be noted in the paper): The official binding is its own Faster R-CNN detector
    And end-to-end training; here we use **frozen CLIP features + the same candidate set** to only reproduce the relationship header and regular ideas.
    """

    def __init__(self, K: int, d: int = FEAT_DIM, hid: int = 512, layers: int = 2,
                 p: float = 0.3, sem: Optional[torch.Tensor] = None):
        super().__init__(K)
        self.layers = layers
        # Local: subject/object respective linear mapping + background context
        self.W_s = nn.Linear(d, hid)
        self.W_o = nn.Linear(d, hid)
        self.W_bg = nn.Linear(d, hid)
        # Local interaction: subject/object binding + interaction with background
        # ⚠️ LayerNorm must be placed **after Linear** (hid dimension),
        # If placed before, an error will be reported for the input dimension (hid*3+7=1031).
        # First normalize the original input features to eliminate the scale difference between CLIP(512) and geometry(7):
        self.in_norm = nn.LayerNorm(d * 2 + N_PAIR_GEOM)
        self.local = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.LayerNorm(hid),
            nn.ReLU(inplace=True), nn.Dropout(p))
        # Global: node-edge context propagation
        self.g_msg = nn.ModuleList([nn.Linear(hid * 2, hid) for _ in range(layers)])
        self.g_upd = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.g_gate = nn.ModuleList([nn.Linear(hid, 1) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hid), nn.Dropout(p), nn.Linear(hid, K))
        # Predicate semantic similarity as edge affinity (freeze)
        if sem is None:
            sem = torch.eye(K)
        sn = F.normalize(sem.float(), dim=-1)
        self.register_buffer("sem_sim", sn @ sn.t())
        self.K = K
        # Trainable predicate embedding (A&R loss object)
        self.pred_emb = nn.Parameter(torch.randn(K, 128) * 0.02)

    def ar_loss(self, margin_attract: float = 0.7,
                margin_repel: float = 0.3) -> torch.Tensor:
        """Attract & Repel loss (the third point of the original LOGIN design).

        Implemented using **pairwise margin ranking**, eliminating diagonals (self and self):

        - **Attract**: For predicate pairs with similar semantics (high target similarity),
          Penalty `max(0, margin_attract − sim)`
        - **Repel**: For semantically irrelevant predicate pairs (low target similarity),
          Penalty `max(0, sim − margin_repel)`

        Target similarity `sem_sim` is computed from **frozen CLIP predicate text embedding** (not bootstrapped),
        Therefore, the real role of this item is to ”allow trainable predicate embeddings to align with CLIP semantic geometry.”
        """
        e = F.normalize(self.pred_emb, dim=-1)
        sim = e @ e.t()                                  # (K,K) actual cosine
        tgt = self.sem_sim                               # (K,K) target cosine
        K = sim.shape[0]
        eye = torch.eye(K, dtype=torch.bool, device=sim.device)
        attract_pair = (tgt >= margin_attract) & ~eye     # Pairs of predicates that should be similar
        repel_pair = (tgt <= margin_repel) & ~eye        # Predicate pairs to stay away from
        loss = sim.sum() * 0.0                           # Empty tensor occupancy (to ensure device consistency)
        if attract_pair.any():
            loss = loss + F.relu(margin_attract - sim[attract_pair]).mean()
        if repel_pair.any():
            loss = loss + F.relu(sim[repel_pair] - margin_repel).mean()
        return loss

    def local_context(self, vs, os_, ctxt, pg):
        """Local ternary interaction + orientation awareness (s/o independent mapping ⟹ swapping order changes representation)."""
        # First normalize the original input (subject ⊕ object ⊕ geometry) to eliminate scale differences.
        in_n = self.in_norm(torch.cat([vs, os_, pg], dim=-1))
        vs_n, os_n, _ = torch.split(in_n, [vs.shape[-1], os_.shape[-1], N_PAIR_GEOM], dim=-1)
        s_e, o_e, b_e = self.W_s(vs_n), self.W_o(os_n), self.W_bg(ctxt)
        # ⊙ Binding: Literal implementation of direction awareness
        return self.local(torch.cat([s_e * o_e, b_e], dim=-1))

    def forward(self, x, masks, s_idx, o_idx, pmask=None):
        vs, os_, ts, tos, ctxt, pg, _ = self.common_entry(x, masks, s_idx, o_idx)
        h = self.local_context(vs, os_, ctxt, pg)              # (B,P,H)
        B, P, H = h.shape
        # Global: Candidate pairs serve as nodes of the graph, and messages are modulated along the candidate sequence + predicate similarity
        for i in range(self.layers):
            # Neighbor messages: two-way rolling aggregation on candidate sequences (approximate ”graph context”)
            nb = torch.cat([h.roll(1, dims=1), h.roll(-1, dims=1)], dim=-1)
            m = self.g_msg[i](nb)
            # Predicate similarity as edge affinity: take the mean similarity between the current prediction distribution and each predicate embedding
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

# Differentiated hyperparameters when constructing each model.
# ⚠️ Necessary correction (2026-10-18): Previously, sgtr and transformer were both registered to the same class and
# **Not passed** `use_ctx` Difference ⟹ Both run out **bit by bit same** indicators (R@50 on VRD are both
# 0.1645, mR@50 0.0455, top1 0.1536), actually the same model was run twice.
# Explicit distinction now: sgtr with full graph context (aligned to SGTR’s one-stage global aggregation),
# transformer without (pure control).
MODEL_KW = {
    "sgtr": {"use_ctx": True},
    "transformer": {"use_ctx": False},
    "vctree": {"layers": 2},
    "login": {"layers": 2},
}


# ======================================================================
# Training and evaluation (same set of loops as bilstm_baseline)
# ======================================================================
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_one(model, obj_emb, data: SplitData, hp, dev, args, seed, log=print):
    """Training + evaluation (the caliber is consistent with the main experiment)."""
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
            # ⚠️ You must press `pmask` to do the mask, otherwise the loss will be dominated by **fill position**.
            # The main experiment is implemented through `cls_loss(..., m=pmask)`; this script was originally directly
            # `cross_entropy(logits.reshape(-1,K), y.reshape(-1))` Put all P candidates
    # (including padding) included ⟹ The measured R@50 of sgtr is only 0.14–0.17 (gpsnet is 0.3355),
            # and top1 is as low as 0.28–0.42. Transformer mixes all tokens due to global attention,
            # It is most sensitive to padding pollution, so it suffers the most.
            pm = batch["pmask"].to(y.device).reshape(-1).bool()
            lg = logits.reshape(-1, hp.K)
            tg = y.reshape(-1)
            if pm.any():
                loss = F.cross_entropy(lg[pm], tg[pm])
                # LOGIN’s Attract & Repel auxiliary loss: semantically similar predicates attract each other,
                # Predicates that are inverses of each other are mutually exclusive (acts on **trainable predicate embedding**)
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
                    help="Commas separate predicate names, and evaluate from GT with candidates excluded (discriminative view)."
                         "GQA Suggestions 'to the left of,to the right of'; It must be of the same caliber as the main experiment.")
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
                    help="Allow silent truncation beyond --nmax/--pmax object/relationship (GQA Yes 0.75% Graph relationship coefficient>pmax)")
    ap.add_argument("--eval-mode", choices=["predcls", "gtpairs"], default="predcls")
    ap.add_argument("--feat-backbone", choices=["clip", "roi"], default="clip")
    ap.add_argument("--roi-arch", default="vg_r101c4",
                    choices=["vg_r101c4", "resnext101_vg", "resnet50_vg",
                             "resnext101", "resnet50"],
                    help="ROI Feature backbone (only feat-backbone=roi effective when)")
    ap.add_argument("--use-clip-text", dest="use_clip_text", action="store_true", default=True)
    ap.add_argument("--login-ar", type=float, default=0.0,
                    help="LOGIN of Attract & Repel Auxiliary loss weights (only login Baseline takes effect)."
                         "0 = Off (default, pure faithful reproduction of its relationship header); >0 Enable predicate embedding regularity when"
                         "With this style(23) The predicate embedding constraints form a directly comparable")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(ART, "sota_baselines.json"))
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    K = ds.K
    hp = HyperParams(K=K)
    dev = get_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models_want = [m.strip() for m in args.models.split(",") if m.strip()]

    # ⚠️ The recording interfaces of the two data sets are different: VRD uses `load_split(split, limit)`,
    # Only VG150 has the `records` attribute (line 1406 of the main experiment is also this branch).
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

    # Feature cache (same as the main experiment)
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

    # Exclude the ”discriminant view” of trivial predicates (required for GQA, must be of the same caliber as the main experiment, otherwise the comparison will be unfair)
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
            print(f"[exclude] Discriminant view: exclude {len(_excl_preds)} predicate,"
                  f"Reserve {int(_excl_mask.sum())}/{K} Participate in sorting")

    results = {"dataset": args.dataset, "eval_mode": args.eval_mode,
               "feat_backbone": args.feat_backbone, "models": {}, "params": {}}
    for name in models_want:
        cls = MODELS[name]
        print(f"\n=== {name} ===", flush=True)
        acc, raw = {}, []
        for sd in seeds:
            kw = dict(MODEL_KW.get(name, {}))
            # vctree/login requires predicate semantic embedding (for affinity)
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
                print(f"    [discriminant view] R@50={mm_ex['R@50']} mR@50={mm_ex['mR@50']} "
                      f"mR@100={mm_ex['mR@100']} top1={mm_ex['top1']} "
                      f"BRC@1={mm_ex.get('BRC@1')}", flush=True)
            del m
            torch.cuda.empty_cache()
        results["models"][name] = acc
        results["params"][name] = nparam
        results.setdefault("raw_runs", {})[name] = raw
        # ⚠️ Every time a model is run, **atomically write to disk** (write a temporary file first and then rename).
        # Previously, only the disk was written after all models were run, causing all completed model results to be lost when interrupted/killed.
        # (VG-5000 ran 4 baselines but json was not placed).
        # Atomic rename ensures that half of the JSON will not be read when writing the same out concurrently.
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        _tmp = args.out + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        os.replace(_tmp, args.out)
        print(f"  ↳ Disk written {args.out}（{len(results['models'])} models)", flush=True)

    print(f"\n Complete → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())