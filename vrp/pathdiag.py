"""三角路径（隐式流）一致性诊断：它到底有没有 pair 判别力？

检验目标（决定要不要做「用隐式流验证显式流」这条路）：
    对每个候选物体对 (A,C)，用两条一阶关系复合出 A→C 的谓词分布
        u_AC = Σ_B  s_BC · (p_AB @ P_conv)
    再算它与显式预测 p_AC 的一致性
        c_AC = 1 − JS(p_AC ‖ rownorm(u_AC))
    若 c_AC 能把「真关系对」与「顺手采样的非关系对」分开（AUC 显著 > 0.5），
    则把 c 作为 pair 级校准项（加到 logits 上）才有意义。

同时给出两个基准：
    z_conf = max_k log p_AC      现有 pair 置信度
    AUC(z+c)                     两者 rank 平均后的融合 AUC

用法（在 small_sample_experiment.py 里由 --diag-path-auc 触发）：
    path_auc_diag(model, obj_emb, te, hp, log=log)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .closure import rownorm


# ---------------------------------------------------------------- 工具
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U 的 AUC（并列取平均秩）。"""
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # 并列修正：同分值取平均秩
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐行 JS 散度（自然对数，取值 [0, ln2]）。"""
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a) - np.log(b))).sum(-1)   # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _mod_dev(m) -> torch.device:
    """模块所在设备。`FrozenObjText` 全参数冻结，`parameters()` 可能为空 ⟹ 退化到 buffer。"""
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


# ---------------------------------------------------------------- 主诊断
@torch.no_grad()
def path_auc_diag(model, obj_emb, data, hp, log=print,
                  max_img: int = 0, eta: float = 1.0) -> dict:
    """在给定 split 上统计 c_AC 的 pair 判别力（AUC）。"""
    model.eval()
    dev = next(model.parameters()).device
    K = hp.K
    p_conv = model.closure()[1][:, :K]              # (K,K+1) → 去掉 φ 列得 (K,K)
    Pc = rownorm(p_conv)                            # p_conv 已 softmax，截列后行和<1，需再归一
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
        # logits → 概率（pair 内归一，仅用于路径聚合与一致性）
        p = torch.softmax(sc.float(), dim=-1)                    # (P,K)
        # ⚠️ pair 置信度必须取 **logits 域**的 max：log-softmax 是 per-pair 常数平移，
        #    若先 softmax 再取 max，`z += β·log σ(rel)` 这类 pair 级校准会被完全抵消，
        #    AUC 就测不出任何变化（曾因此误判 pair_rel 无效）。
        s_conf = sc.float().max(-1).values                       # (P,) logits 域 pair 置信度

        # 稠密化：P3[a,b,:]=p, S2[b,c]=s_conf
        P3 = np.zeros((n, n, K), dtype=np.float32)
        S2 = np.zeros((n, n), dtype=np.float32)
        pos = {}
        for j, (a, b) in enumerate(pairs):
            P3[a, b] = p[j].cpu().numpy()
            S2[a, b] = float(s_conf[j])
            pos[(a, b)] = j

        Q3 = P3 @ Pc_np                                          # (n,n,K) 谓词复合
        U = np.einsum("abl,bc->acl", Q3, S2)                     # (n,n,K) 路径聚合
        npair = len(pairs)
        a_arr = np.array([a for a, _ in pairs])
        b_arr = np.array([b for _, b in pairs])
        u_rows = np.zeros((npair, K), dtype=np.float32)
        for j in range(npair):
            row = U[a_arr[j], b_arr[j]]
            tot = row.sum()
            u_rows[j] = row / tot if tot > 1e-8 else np.full(K, 1.0 / K, dtype=np.float32)
        n_path += npair

        p_rows = P3[a_arr, b_arr]                                # (P,K) 显式预测
        js = _js(p_rows, u_rows)                                 # (P,)
        c_conf = 1.0 - js / np.log(2.0)                          # 映射到 [0,1]

        is_gt = np.zeros(npair, dtype=np.int64)
        for (a, b, _pred) in gt:
            if (a, b) in pos:
                is_gt[pos[(a, b)]] = 1

        Z.append(sc.max(-1).values.cpu().numpy().astype(np.float32))
        C.append(c_conf.astype(np.float32))
        Y.append(is_gt)

    if not Z:
        log("  [path-auc] 无有效数据")
        return {}
    z = np.concatenate(Z); c = np.concatenate(C); y = np.concatenate(Y)
    a_z, a_c = _auc(y, z), _auc(y, c)
    # rank 平均融合
    def _rank(v):
        o = np.argsort(np.argsort(v))
        return o / max(len(v) - 1, 1)
    a_f = _auc(y, 0.5 * _rank(z) + 0.5 * _rank(c))
    res = {"n_pair": int(len(y)), "n_pos": int(y.sum()),
           "auc_z": a_z, "auc_c": a_c, "auc_fusion": a_f,
           "c_pos_mean": float(c[y == 1].mean()), "c_neg_mean": float(c[y == 0].mean()),
           "z_pos_mean": float(z[y == 1].mean()), "z_neg_mean": float(z[y == 0].mean())}
    log("  [path-auc] 候选对 %d（正 %d / 负 %d）" % (res["n_pair"], res["n_pos"], res["n_pair"] - res["n_pos"]))
    log("  [path-auc] AUC(z_conf)=%.4f | AUC(一致性 c)=%.4f | AUC(z⊕c 秩融合)=%.4f"
        % (a_z, a_c, a_f))
    log("  [path-auc] 一致性均值：正样本 %.4f / 负样本 %.4f（差 %+.4f）"
        % (res["c_pos_mean"], res["c_neg_mean"], res["c_pos_mean"] - res["c_neg_mean"]))
    return res
