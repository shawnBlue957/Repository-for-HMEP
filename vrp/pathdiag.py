"""Triangular path (implicit flow) consistency diagnosis: does it have pair discriminability?

Test goal (deciding whether to do the ”use implicit flow to verify explicit flow” path):
    For each candidate object pair (A,C), use two first-order relations to compose the predicate distribution of A→C
        u_AC = Σ_B  s_BC · (p_AB @ P_conv)
    Then calculate its consistency with the explicit prediction p_AC
        c_AC = 1 − JS(p_AC ‖ rownorm(u_AC))
    If c_AC can separate ”true relational pairs” from ”conveniently sampled non-relational pairs” (AUC is significant > 0.5),
    Then it makes sense to treat c as a pair-level calibration term (added to logits).

Two benchmarks are given at the same time:
    z_conf = max_k log p_AC existing pair confidence
    AUC(z+c) The fused AUC after the average of the two ranks

Usage (triggered by --diag-path-auc in small_sample_experiment.py):
    path_auc_diag(model, obj_emb, te, hp, log=log)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .closure import rownorm


#----------------------------------------------------------------Tools
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC of Mann-Whitney U (average ranks tied)."""
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Parallel correction: average rank for the same score
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise JS divergence (natural logarithm, values ​​[0, ln2])."""
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a) - np.log(b))).sum(-1)   # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _mod_dev(m) -> torch.device:
    """The device where the module is located. `FrozenObjText` freezes all parameters, `parameters()` may be empty ⟹ Degenerate to buffer."""
    for t in list(m.parameters()) + list(m.buffers()):
        return t.device
    return torch.device("cpu")


def _scores(model, obj_emb, data, i: int, dev):
    n = int(data.n_obj[i])
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if not pairs:
        return None, []
    S = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    O = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    if obj_emb is not None:
        _te = _mod_dev(obj_emb)
        xt = obj_emb(data.t[i].unsqueeze(0).to(_te))
    else:
        xt = data.t[i].unsqueeze(0).to(dev)
    x = {"v": data.v[i].unsqueeze(0).to(dev),
         "t": xt.to(dev),
         "s": data.s[i].unsqueeze(0).to(dev)}
    p_ij, _ = model.forward_inference(x, data.masks[i].unsqueeze(0).to(dev), S, O)
    return p_ij[0], pairs


# --------------------------------------------------------------------- Main Diagnosis
@torch.no_grad()
def path_auc_diag(model, obj_emb, data, hp, log=print,
                  max_img: int = 0, eta: float = 1.0) -> dict:
    """Count the pair discriminative power (AUC) of c_AC on a given split."""
    model.eval()
    dev = next(model.parameters()).device
    K = hp.K
    p_conv = model.closure()[1][:, :K]              # (K,K+1) → Remove φ and get (K,K)
    Pc = rownorm(p_conv)                            # p_conv has been softmaxed. After column truncation, the row sum is <1 and needs to be normalized again.
    Pc_np = Pc.detach().cpu().numpy()

    Z, C, Y = [], [], []
    n_img = len(data.ids) if not max_img else min(max_img, len(data.ids))
    n_path = 0
    for i in range(n_img):
        gt = data.rels[i]
        if not gt:
            continue
        n = int(data.n_obj[i])
        if n < 3:
            continue
        sc, pairs = _scores(model, obj_emb, data, i, dev)
        if sc is None:
            continue
        # logits → probability (normalized within pair, only used for path aggregation and consistency)
        p = torch.softmax(sc.float(), dim=-1)                    # (P,K)
        # ⚠️ The pair confidence must be the max of the **logits domain**: log-softmax is a per-pair constant translation,
        # If softmax is taken first and then max, pair-level calibration such as `z += β·log σ(rel)` will be completely offset.
        # AUC cannot detect any changes (because of this, pair_rel was misjudged to be invalid).
        s_conf = sc.float().max(-1).values                       # (P,) logits domain pair confidence

        # Densification: P3[a,b,:]=p, S2[b,c]=s_conf
        P3 = np.zeros((n, n, K), dtype=np.float32)
        S2 = np.zeros((n, n), dtype=np.float32)
        pos = {}
        for j, (a, b) in enumerate(pairs):
            P3[a, b] = p[j].cpu().numpy()
            S2[a, b] = float(s_conf[j])
            pos[(a, b)] = j

        Q3 = P3 @ Pc_np                                          # (n,n,K) predicate compound
        U = np.einsum("abl,bc->acl", Q3, S2)                     # (n,n,K) path aggregation
        npair = len(pairs)
        a_arr = np.array([a for a, _ in pairs])
        b_arr = np.array([b for _, b in pairs])
        u_rows = np.zeros((npair, K), dtype=np.float32)
        for j in range(npair):
            row = U[a_arr[j], b_arr[j]]
            tot = row.sum()
            u_rows[j] = row / tot if tot > 1e-8 else np.full(K, 1.0 / K, dtype=np.float32)
        n_path += npair

        p_rows = P3[a_arr, b_arr]                                # (P,K) explicit prediction
        js = _js(p_rows, u_rows)                                 # (P,)
        c_conf = 1.0 - js / np.log(2.0)                          # Map to [0,1]

        is_gt = np.zeros(npair, dtype=np.int64)
        for (a, b, _pred) in gt:
            if (a, b) in pos:
                is_gt[pos[(a, b)]] = 1

        Z.append(sc.max(-1).values.cpu().numpy().astype(np.float32))
        C.append(c_conf.astype(np.float32))
        Y.append(is_gt)

    if not Z:
        log("  [path-auc] No valid data")
        return {}
    z = np.concatenate(Z); c = np.concatenate(C); y = np.concatenate(Y)
    a_z, a_c = _auc(y, z), _auc(y, c)
    # rank average fusion
    def _rank(v):
        o = np.argsort(np.argsort(v))
        return o / max(len(v) - 1, 1)
    a_f = _auc(y, 0.5 * _rank(z) + 0.5 * _rank(c))
    res = {"n_pair": int(len(y)), "n_pos": int(y.sum()),
           "auc_z": a_z, "auc_c": a_c, "auc_fusion": a_f,
           "c_pos_mean": float(c[y == 1].mean()), "c_neg_mean": float(c[y == 0].mean()),
           "z_pos_mean": float(z[y == 1].mean()), "z_neg_mean": float(z[y == 0].mean())}
    log("  [path-auc] candidate pair %d (Right %d / Negative %d)" % (res["n_pair"], res["n_pos"], res["n_pair"] - res["n_pos"]))
    log("  [path-auc] AUC(z_conf)=%.4f | AUC(Consistency c)=%.4f | AUC(z⊕c rank fusion)=%.4f"
        % (a_z, a_c, a_f))
    log("  [path-auc] Consistency mean: positive samples %.4f / negative sample %.4f (Poor %+.4f)"
        % (res["c_pos_mean"], res["c_neg_mean"], res["c_pos_mean"] - res["c_neg_mean"]))
    return res
