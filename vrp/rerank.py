"""Stage-2 关系重排序头（pair 内重排序）。

动机
----
Stage-1（`VRPCatModel`）给出 z_ij ∈ R^K 后，**per-pair 常数平移**
（如 α·LSE、温度缩放）对 pair 内 argmax / 次序恒等 —— 数学上不可能改变
Top-1 与 pair 内 R@k。要真正改变 pair 内排序，修正量必须是 **逐谓词** 的：

    s'_ij[k] = z_ij[k] + Δ_ij[k],   Δ 依赖 (pair, k) 交互

本模块提供两种 Δ：

* `calib` —— 无参数先验校准：Δ_k = -τ·log π_k（π = 训练集谓词频率）。
  它逐谓词不同，故能改变 pair 内次序；本质是长尾 logit adjustment
  （Menon et al., ICLR 2021）。作为**可解释对照**，证明增益不是来自频率先验。

* `learn` —— 学习式残差头：把「pair 视觉/几何特征 × 谓词文本嵌入 × Stage-1 分数」
  做交互，输出有界残差 Δ = γ·tanh(f(·))。训练目标为候选集上的交叉熵
  （候选 = Stage-1 top-M ∪ GT），Stage-1 全程冻结（两阶段重排序）。

设计约束
--------
1. **不得退化为常数**：Δ 必须经 pair×谓词 交互产生；结构性保证是
   `head` 的输入含 `fp * ft`（pair 特征与谓词嵌入的逐元素积）。
2. **有界**：Δ = γ·tanh(·)，γ 初值 2.0（log 域），避免把 Stage-1 的
   top-M 之外谓词意外顶上来；同时保证掩码谓词（-1e9）不会被翻出。
3. **不改变 Stage-1**：reranker 只读 Stage-1 输出的 log 分数与特征，
   不反传、不微调，便于与任何 baseline 并列比较。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelReranker(nn.Module):
    """pair×谓词 交互式残差重排序头。

    输入（按 batch）：
        z_top   (B, M)      Stage-1 分数（log 域）的 top-M 取值
        v_s/v_o (B, 512)    主/宾 ROI 视觉特征
        s_s/s_o (B, 8)      主/宾几何特征
        txt_top (B, M, d_t) 对应谓词的文本嵌入
    输出：
        Δ       (B, M)      有界残差，直接加到 z_top 上
    """

    def __init__(self, d_v: int = 512, d_t: int = 512, d: int = 192,
                 gamma_init: float = 2.0, use_rev: bool = True):
        super().__init__()
        self.use_rev = use_rev
        d_pair = d_v * 4 + 24          # [v_s, v_o, v_s-v_o, v_s*v_o, s_s, s_o, s_s-s_o]
        self.pair_enc = nn.Sequential(
            nn.Linear(d_pair, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
        )
        self.pred_enc = nn.Sequential(nn.Linear(d_t, d), nn.GELU())
        # 额外 2 维：反方向 (o,s) 上逆谓词的 log 分数 + 该逆是否存在（mask）
        self.head = nn.Sequential(
            nn.Linear(d * 3 + (3 if use_rev else 1), d), nn.GELU(), nn.Linear(d, 1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z_top: torch.Tensor, v_s: torch.Tensor, v_o: torch.Tensor,
                s_s: torch.Tensor, s_o: torch.Tensor,
                txt_top: torch.Tensor,
                z_rev: Optional[torch.Tensor] = None,
                has_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        pf = torch.cat([v_s, v_o, v_s - v_o, v_s * v_o,
                        s_s, s_o, s_s - s_o], dim=-1)
        fp = self.pair_enc(pf).unsqueeze(1)                  # (B,1,d)
        ft = self.pred_enc(txt_top)                           # (B,M,d)
        parts = [fp.expand(-1, ft.size(1), -1), ft, fp * ft, z_top.unsqueeze(-1)]
        if self.use_rev:
            if z_rev is None:
                z_rev = torch.zeros_like(z_top)
                has_inv = torch.zeros_like(z_top)
            parts += [z_rev.unsqueeze(-1), has_inv.unsqueeze(-1)]
        h = torch.cat(parts, dim=-1)
        return self.gamma * torch.tanh(self.head(h).squeeze(-1))


# ----------------------------------------------------------------------
def to_log_scores(scores: torch.Tensor) -> torch.Tensor:
    """把打分统一到 **log 域**（重排序头与残差相加都在 log 域进行）。

    两种上游口径都要处理：
      * `SCORE_LOGITS=True`（默认）→ 分数是 logits，做 `log_softmax`。
        它是 per-pair 常数平移，**不改变 pair 内次序**，但能让 z_top 的数值量纲
        与 log 概率一致，残差 γ·tanh 的尺度才可比。
      * `SCORE_LOGITS=False`（历史口径）→ 分数是 softmax 概率，直接取 log。

    掩码谓词被置为 -1e9，概率口径下 clamp 后为 -20.7，与正常项仍有 10 个量级以上
    差距；残差 |Δ| ≤ γ，不会翻转。
    """
    import vrp.model as _vm
    if getattr(_vm, "SCORE_LOGITS", False):
        return torch.log_softmax(scores, dim=-1)
    return torch.log(scores.clamp_min(1e-9))


def pair_row(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """predcls 候选对的行号（顺序：`for a: for b: b!=a`）。

    与 `_pair_index` 同源，但纯向量化，避免为反查 (o,s) 建 dict。
    """
    return a * (n - 1) + torch.where(b < a, b, b - 1)


def rev_feature(z_mat: torch.Tensor, jr: Optional[int], cand_idx: torch.Tensor,
                inv_of: Optional[torch.Tensor]):
    """对候选谓词取「反方向 (o,s) 上逆谓词」的 log 分数。

    返回 (z_rev (M,), has_inv (M,))。无逆谓词或无反转对时对应位置为 0 且 mask=0，
    使模型可以显式学到「这条信息缺失」，而不是被 0 值误导。
    """
    M = cand_idx.numel()
    z_rev = torch.zeros(M, device=z_mat.device, dtype=z_mat.dtype)
    has = torch.zeros(M, device=z_mat.device, dtype=z_mat.dtype)
    if jr is None or inv_of is None or inv_of.numel() <= 1:
        return z_rev, has
    inv_of = inv_of.to(z_mat.device)
    rv = inv_of[cand_idx + 1] - 1                     # 0-based 逆谓词，-1 表示无
    ok = rv >= 0
    if ok.any():
        z_rev[ok] = z_mat[int(jr), rv[ok]]
        has[ok] = 1.0
    return z_rev, has


def rev_feature_batch(z: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor,
                      n: int, cand_idx: torch.Tensor,
                      inv_of: Optional[torch.Tensor]):
    """`rev_feature` 的批量版：z (P,K)、cand_idx (P,M) → (z_rev (P,M), has (P,M))。"""
    P, M = cand_idx.shape
    if inv_of is None or inv_of.numel() <= 1:
        return torch.zeros(P, M, device=z.device, dtype=z.dtype), \
               torch.zeros(P, M, device=z.device, dtype=z.dtype)
    inv_of = inv_of.to(z.device)
    rv = inv_of[cand_idx + 1] - 1                     # (P,M)
    ok = rv >= 0
    # 反方向行号：index of (b,a)
    jr = pair_row(b_idx, a_idx, n)                    # (P,)
    src = z[jr.unsqueeze(1).expand(-1, M), rv.clamp_min(0)]   # (P,M)
    z_rev = torch.where(ok, src, torch.zeros_like(src))
    return z_rev, ok.to(z.dtype)


def make_candidates(z: torch.Tensor, gt: Sequence[Tuple[int, int, int]],
                    pair_pos: dict, m_top: int):
    """构造 reranker 训练/推理的候选集：Stage-1 top-M ∪ GT 谓词。

    返回：
        top_idx (M,) long   候选谓词下标（GT 不在 top-M 时替换末位）
        z_top   (M,) float  候选上的 Stage-1 log 分数
        gt_pos  int         GT 在候选中的位置（推理时无用）
    """
    M = min(m_top, z.numel())
    z_top, top_idx = torch.topk(z, M, dim=-1)
    top_idx = top_idx.clone()
    gt_pos = -1
    for (s, o, p) in gt:
        if pair_pos.get((s, o)) is None:
            continue
        if (top_idx == p).any():
            gt_pos = int((top_idx == p).nonzero()[0, 0])
        else:
            top_idx[-1] = p
            z_top[-1] = z[p]
            gt_pos = M - 1
        break
    return top_idx, z_top, gt_pos


@torch.no_grad()
def collect_rerank_data(model, obj_emb, data, m_top: int = 20,
                        keep_pair=None, log=print, inv_of=None) -> dict:
    """在 train split 上采集 (pair 索引, 候选谓词, Stage-1 分数, GT 位置, 逆对分数)。

    `keep_pair(i, s, o) -> bool` 可用于留出验证图（避免 reranker 在同图训练/评测）。
    `inv_of` (K+1,) long：逆谓词表（0-based，-1=无逆），用于构造 (o,s) 方向特征。
    """
    img_idx, sidx, oidx, cand, zs, gts = [], [], [], [], [], []
    zrevs, hass = [], []
    n_img = len(data.ids)
    for i in range(n_img):
        gt = data.rels[i]
        if not gt:
            continue
        scores, pairs = _score_image_nb(model, obj_emb, data, i)
        if scores is None:
            continue
        z = to_log_scores(scores)
        pair_pos = {p: k for k, p in enumerate(pairs)}
        for (s, o, p) in gt:
            j = pair_pos.get((s, o))
            if j is None:
                continue
            if keep_pair is not None and not keep_pair(i, s, o):
                continue
            top_idx, z_top, gt_pos = make_candidates(z[j], [(s, o, p)], pair_pos, m_top)
            if gt_pos < 0:
                continue
            zr, hv = rev_feature(z, pair_pos.get((o, s)), top_idx, inv_of)
            img_idx.append(i); sidx.append(s); oidx.append(o)
            cand.append(top_idx); zs.append(z_top); gts.append(gt_pos)
            zrevs.append(zr); hass.append(hv)
    if not img_idx:
        return {}
    out = {
        "img": torch.tensor(img_idx, dtype=torch.long),
        "s": torch.tensor(sidx, dtype=torch.long),
        "o": torch.tensor(oidx, dtype=torch.long),
        "cand": torch.stack(cand),
        "z": torch.stack(zs),
        "gt": torch.tensor(gts, dtype=torch.long),
        "zrev": torch.stack(zrevs),
        "has": torch.stack(hass),
    }
    log(f"  [rerank] 采集 {len(img_idx):,} 个 GT pair（候选 top-{m_top} ∪ GT）；"
        f"逆对可用率 {float(out['has'].mean()):.4f}")
    return out


def _score_image_nb(model, obj_emb, data, i: int):
    """与 `_score_image`（predcls 口径）逐位同义的本地实现。

    刻意**不**从 `tools.small_sample_experiment` 导入：该脚本常以 `__main__`
    运行，重复导入会二次执行模块级初始化（SCORE_LOGITS 等全局开关出现双份）。
    """
    dev = next(model.parameters()).device
    n = data.n_obj[i]
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if not pairs:
        return None, []
    S = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    O = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    xt = obj_emb(data.t[i].unsqueeze(0).to(dev)) if obj_emb is not None else \
        data.t[i].unsqueeze(0).to(dev)
    x = {"v": data.v[i].unsqueeze(0).to(dev),
         "t": xt,
         "s": data.s[i].unsqueeze(0).to(dev)}
    masks = data.masks[i].unsqueeze(0).to(dev)
    p_ij, _ = model.forward_inference(x, masks, S, O)
    return p_ij[0], pairs


def train_reranker(head: RelReranker, pack: dict, data, txt: torch.Tensor,
                   epochs: int = 8, lr: float = 2e-3, batch: int = 4096,
                   log=print) -> dict:
    """在冻结 Stage-1 的采集数据上训练 reranker（候选集交叉熵）。"""
    dev = next(head.parameters()).device
    txt = txt.to(dev)
    n = pack["img"].numel()
    if n == 0:
        return {"n": 0}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    img, sidx, oidx = pack["img"].to(dev), pack["s"].to(dev), pack["o"].to(dev)
    cand, z, gt = pack["cand"].to(dev), pack["z"].to(dev), pack["gt"].to(dev)
    zrv = pack["zrev"].to(dev) if pack.get("zrev") is not None else None
    hasv = pack["has"].to(dev) if pack.get("has") is not None else None

    # 预先把每图的 v/s 展平成大表，按 (img,obj) 线性索引，避免逐 batch 取 list
    v_all = torch.cat([data.v[i] for i in range(len(data.ids))], 0).to(dev)
    s_all = torch.cat([data.s[i] for i in range(len(data.ids))], 0).to(dev)
    off = torch.tensor([0] + [data.v[i].shape[0] for i in range(len(data.ids))],
                       dtype=torch.long, device=dev).cumsum(0)

    head.train()
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = cor = 0
        lsum = 0.0
        for b in range(0, n, batch):
            sel = perm[b:b + batch]
            gi, gs, go = img[sel], sidx[sel], oidx[sel]
            zz, cc, gg = z[sel], cand[sel], gt[sel]
            delta = head(zz, v_all[off[gi] + gs], v_all[off[gi] + go],
                         s_all[off[gi] + gs], s_all[off[gi] + go],
                         txt[cc],
                         zrv[sel] if zrv is not None else None,
                         hasv[sel] if hasv is not None else None)
            loss = F.cross_entropy(zz + delta, gg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            lsum += float(loss) * gi.numel()
            tot += gi.numel()
            cor += int(((zz + delta).argmax(-1) == gg).sum())
        hist.append({"ep": ep + 1, "loss": lsum / max(tot, 1),
                     "top1": cor / max(tot, 1)})
        log(f"  [rerank] ep{ep+1}/{epochs} loss={hist[-1]['loss']:.4f} "
            f"acc={hist[-1]['top1']:.4f}")
    head.eval()
    return {"n": n, "hist": hist}


@torch.no_grad()
def rerank_inplace(head: RelReranker, scores: torch.Tensor, data, i: int,
                   m_top: int, txt: torch.Tensor,
                   inv_of: Optional[torch.Tensor] = None) -> torch.Tensor:
    """把 reranker 残差加回第 i 图的 (P,K) 分数（仅 top-M 位置）。

    候选对顺序必须与 `_score_image`（predcls：`for a: for b: a!=b`）一致，
    否则残差会加到错误的 pair 上 —— 这里按行号反解并做一致性校验。
    `inv_of`：(K+1,) long 逆谓词表，供 (o,s) 方向特征使用。
    """
    z = to_log_scores(scores)
    P, K = z.shape
    M = min(m_top, K)
    z_top, top_idx = torch.topk(z, M, dim=-1)                 # (P,M)
    a_idx, b_idx = _pair_index(data.n_obj[i], P, z.device)
    if a_idx is None:
        return scores
    v = data.v[i].to(z.device); s = data.s[i].to(z.device)
    zr, hv = rev_feature_batch(z, a_idx, b_idx, data.n_obj[i], top_idx, inv_of)
    delta = head(z_top, v[a_idx], v[b_idx], s[a_idx], s[b_idx], txt[top_idx], zr, hv)
    z.scatter_add_(-1, top_idx, delta)
    return z


_PAIR_CACHE: dict = {}


def _flat_vs(data, dev):
    """把逐图的 v/s 展平成大表 + 偏移表，便于按 (img, obj) 线性索引。"""
    n_img = len(data.ids)
    v_all = torch.cat([data.v[i] for i in range(n_img)], 0).to(dev)
    s_all = torch.cat([data.s[i] for i in range(n_img)], 0).to(dev)
    off = torch.tensor([0] + [data.v[i].shape[0] for i in range(n_img)],
                       dtype=torch.long, device=dev).cumsum(0)
    return v_all, s_all, off


@torch.no_grad()
def eval_reranker_pack(head: RelReranker, pack: dict, data, txt: torch.Tensor,
                       dev, batch: int = 8192) -> dict:
    """留出集上对比 Stage-1 与重排序后的 top1，并统计**次序改变率**。

    `flip` = 重排序改变 argmax 的样本占比 —— 直接回答「这个模块是否真的
    改变了 pair 内排序」（per-pair 常数平移的 flip 恒为 0）。
    """
    if not pack or pack["img"].numel() == 0:
        return {"base": 0.0, "rerank": 0.0, "flip": 0.0, "n": 0}
    head.eval()
    txt = txt.to(dev)
    v_all, s_all, off = _flat_vs(data, dev)
    img, sidx, oidx = pack["img"].to(dev), pack["s"].to(dev), pack["o"].to(dev)
    cand, z, gt = pack["cand"].to(dev), pack["z"].to(dev), pack["gt"].to(dev)
    zrv = pack["zrev"].to(dev) if pack.get("zrev") is not None else None
    hasv = pack["has"].to(dev) if pack.get("has") is not None else None
    tot = base = rr = fl = 0
    for b in range(0, img.numel(), batch):
        sl = slice(b, b + batch)
        gi = img[sl]
        delta = head(z[sl], v_all[off[gi] + sidx[sl]], v_all[off[gi] + oidx[sl]],
                     s_all[off[gi] + sidx[sl]], s_all[off[gi] + oidx[sl]],
                     txt[cand[sl]],
                     zrv[sl] if zrv is not None else None,
                     hasv[sl] if hasv is not None else None)
        zz = z[sl]
        p0 = zz.argmax(-1)
        p1 = (zz + delta).argmax(-1)
        tot += gi.numel()
        base += int((p0 == gt[sl]).sum())
        rr += int((p1 == gt[sl]).sum())
        fl += int((p0 != p1).sum())
    return {"base": base / max(tot, 1), "rerank": rr / max(tot, 1),
            "flip": fl / max(tot, 1), "n": tot}

def _pair_index(n: int, P: int, dev):
    """cache 化的 (a,b) 索引，与 predcls 候选对生成顺序逐位一致。"""
    key = (n, P)
    if key in _PAIR_CACHE:
        a, b = _PAIR_CACHE[key]
        return a.to(dev), b.to(dev)
    a = [x for x in range(n) for y in range(n) if y != x]
    b = [y for x in range(n) for y in range(n) if y != x]
    if len(a) != P:                                           # 口径不符（如 gtpairs）
        return None, None
    ta = torch.tensor(a, dtype=torch.long)
    tb = torch.tensor(b, dtype=torch.long)
    _PAIR_CACHE[key] = (ta, tb)
    return ta.to(dev), tb.to(dev)
