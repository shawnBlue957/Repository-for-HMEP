#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""几何规则基线：只用框中心比较预测空间谓词，量化「平凡解天花板」。

## 为什么必须报这个基线

实测（GQA train，3,785,797 条）：
    to the left of  / to the right of  与「框中心 x 比较」一致率 = **100.00%**
    above 87.75% / below 89.95%
且左右两谓词占 **89.34%**。

⟹ 一个**完全不看图像**、只用框中心比较的规则，其 top-1 上限就是 89.34%。
在 GQA 上报 R@K / top1 而不报此基线，会让人误以为模型学到了视觉关系，
实际上可能只是在拟合几何。本脚本把这个「平凡天花板」显式算出来。

## 口径

- 得分只给 4 个空间谓词（left/right/above/below），其余谓词恒为极低分 ⟹ 永不获胜。
  于是该基线的 top-1 = 「几何规则判对的 GT 占比」。
- 同时报告 `--exclude-preds` 后的判别视图（排除左右后，几何规则应显著下降，
  这才是模型真正要赢的战场）。

用法：
    python tools/eval_geom_baseline.py --dataset gqa \
        --exclude-preds "to the left of,to the right of"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import small_sample_experiment as S
from vrp.config import HyperParams

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts")

# 各数据集里这些语义的不同写法（取第一个命中的）
LEFT = ["to the left of", "on the left of", "left of", "to the left"]
RIGHT = ["to the right of", "on the right of", "right of", "to the right"]
ABOVE = ["above", "on the top of", "over", "on top of"]
BELOW = ["below", "under", "beneath"]


def pick(names, vocab):
    for n in names:
        if n in vocab:
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gqa", choices=["gqa", "vrd", "vg150"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--test-n", type=int, default=3000)
    ap.add_argument("--test-split", default="test", choices=["val", "test"],
                    help="GQA 用 val（其 train/val 划分），VG 老脚本默认 val 由 dataset 推断")
    ap.add_argument("--nmax", type=int, default=64)
    ap.add_argument("--pmax", type=int, default=288)
    ap.add_argument("--exclude-preds", default=None)
    ap.add_argument("--out", default="/tmp/geom_baseline.json")
    args = ap.parse_args()

    log = lambda *a: print(*a, flush=True)
    log("=" * 78)
    log(f"几何规则基线 | dataset={args.dataset}")
    log("=" * 78)

    ds = S.get_dataset(args.dataset)
    K = ds.K
    vocab = set(ds.predicate_classes[1:])
    name2i = {n: i for i, n in enumerate(ds.predicate_classes[1:])}

    l_name, r_name = pick(LEFT, vocab), pick(RIGHT, vocab)
    a_name, b_name = pick(ABOVE, vocab), pick(BELOW, vocab)
    if not (l_name and r_name):
        log(f"!! 词表中找不到左右谓词（LEFT={LEFT}, RIGHT={RIGHT}）")
        return 1
    iL, iR = name2i[l_name], name2i[r_name]
    iA = name2i[a_name] if a_name else None
    iB = name2i[b_name] if b_name else None
    log(f"  左={l_name}({iL}) 右={r_name}({iR}) "
        f"上={a_name}({iA}) 下={b_name}({iB})")

    # 特征缓存
    cache_path = args.cache
    if cache_path is None:
        cands = [
            os.path.join(ART, f"feat_cache_roi_resnet50_{args.dataset}.pt"),
            os.path.join(ART, f"feat_cache_clip_{args.dataset}_tr{args.test_n}_te{args.test_n}_test.pt"),
        ]
        for c in cands:
            if os.path.exists(c):
                cache_path = c
                break
    if not cache_path or not os.path.exists(cache_path):
        log(f"!! 找不到特征缓存（试过 {cands}）；请用 --cache 指定")
        return 1
    cache = torch.load(cache_path, weights_only=False)
    log(f"[cache] {os.path.basename(cache_path)}：{len(cache):,} 图")

    split = args.test_split
    if split == "test" and args.dataset == "vg150":
        split = "val"
    recs = [r for r in ds.load_split(split, limit=args.test_n) if r.image_id in cache]
    data = S.build_split(recs, cache, K, args.nmax, args.pmax, strict=False)
    log(f"[split] {len(data.ids):,} 图 | N≤{data.N_max} P≤{data.P_max}")

    hi, lo = 10.0, -50.0

    def geom_scores(d, i, cand_pairs):
        """(n_pairs, K)：left/right 由中心 x 决定，above/below 由中心 y 决定。"""
        ss = d.s[i].cpu().numpy()            # (N, 8): ncx, ncy, nw, nh, area, aspect, ...
        sc = np.full((len(cand_pairs), K), lo, dtype=np.float32)
        for j, (a, b) in enumerate(cand_pairs):
            dx = float(ss[b, 0] - ss[a, 0])   # >0 ⟹ a 在 b 左侧
            dy = float(ss[b, 1] - ss[a, 1])   # >0 ⟹ a 在 b 上方
            sc[j, iL] = hi * dx
            sc[j, iR] = -hi * dx
            if iA is not None:
                sc[j, iA] = hi * dy
            if iB is not None:
                sc[j, iB] = -hi * dy
        return sc

    hp = HyperParams()
    hp.K = K

    # 原始图片尺寸已归一化进 s，无需 W/H

    inv_idx = None
    sem_p = os.path.join(ART, S.semantic_file(args.dataset))
    if os.path.exists(sem_p):
        inv_idx = np.asarray(torch.load(sem_p, weights_only=False)["inv_idx"],
                             dtype=np.int64)

    res = {}
    m_full = S.evaluate(None, None, data, hp, prior=geom_scores,
                        eval_mode="predcls", inv_idx=inv_idx)
    res["full"] = m_full
    log("\n[全量口径] " + json.dumps(m_full, ensure_ascii=False))

    if args.exclude_preds:
        excl = [s_.strip() for s_ in args.exclude_preds.split(",") if s_.strip()]
        exset = {name2i[n] for n in excl if n in name2i}
        keep = torch.ones(K, dtype=torch.bool)
        for p in exset:
            keep[p] = False
        m_ex = S.evaluate(None, None, data, hp, prior=geom_scores,
                          eval_mode="predcls", inv_idx=inv_idx,
                          pred_keep=keep, skip_preds=exset)
        res["excl"] = m_ex
        log(f"[判别视图 排除{sorted(excl)}] " + json.dumps(m_ex, ensure_ascii=False))

    log("\n" + "=" * 78)
    log("判读：该基线的 top1 即『平凡几何解』的天花板。")
    log("  若模型 top1 ≈ 此值 ⟹ 模型等价于几何规则，未学到视觉关系；")
    log("  若模型在『判别视图』上显著高于此值 ⟹ 增益来自视觉，而非几何。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "cache": os.path.basename(cache_path),
                   "n_img": len(data.ids), "results": res}, f, ensure_ascii=False, indent=2)
    log(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
