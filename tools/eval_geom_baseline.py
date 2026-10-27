#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometric rule baseline: only use box centers to compare prediction space predicates to quantify the ”trivial solution ceiling”.

## Why must we report this baseline?

Actual measurement (GQA train, 3,785,797 items):
    to the left of / to the right of and ”box center x comparison” consistency rate = **100.00%**
    above 87.75% / below 89.95%
And the left and right predicates account for **89.34%**.

⟹ A rule that does not look at the image at all and only uses the center of the box to compare, its top-1 upper limit is 89.34%.
Reporting R@K / top1 to GQA without reporting this baseline will make people mistakenly think that the model has learned visual relationships.
It might actually just be fitting geometry. This script explicitly calculates this ”trivial ceiling”.

## Caliber

- Scores are only given to 4 spatial predicates (left/right/above/below), and the remaining predicates always have extremely low scores ⟹ Never win.
  So the top-1 of this baseline = ”the proportion of GT that is correct in geometric rules”.
- Also report the discriminant view after `--exclude-preds` (after excluding left and right, geometric rules should drop significantly,
  This is the battlefield where the model really wants to win).

Usage:
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

# Different ways of writing these semantics in each data set (take the first hit)
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
                    help="GQA use val (its train/val divide), VG Old script default val by dataset inference")
    ap.add_argument("--nmax", type=int, default=64)
    ap.add_argument("--pmax", type=int, default=288)
    ap.add_argument("--exclude-preds", default=None)
    ap.add_argument("--out", default="/tmp/geom_baseline.json")
    args = ap.parse_args()

    log = lambda *a: print(*a, flush=True)
    log("=" * 78)
    log(f"geometric rule baseline | dataset={args.dataset}")
    log("=" * 78)

    ds = S.get_dataset(args.dataset)
    K = ds.K
    vocab = set(ds.predicate_classes[1:])
    name2i = {n: i for i, n in enumerate(ds.predicate_classes[1:])}

    l_name, r_name = pick(LEFT, vocab), pick(RIGHT, vocab)
    a_name, b_name = pick(ABOVE, vocab), pick(BELOW, vocab)
    if not (l_name and r_name):
        log(f"!! Left and right predicates not found in vocabulary (LEFT={LEFT}, RIGHT={RIGHT}）")
        return 1
    iL, iR = name2i[l_name], name2i[r_name]
    iA = name2i[a_name] if a_name else None
    iB = name2i[b_name] if b_name else None
    log(f"  left={l_name}({iL}) right={r_name}({iR}) "
        f"on={a_name}({iA}) down={b_name}({iB})")

    # Feature cache
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
        log(f"!! Feature cache not found (tried {cands}); Please use --cache Specify")
        return 1
    cache = torch.load(cache_path, weights_only=False)
    log(f"[cache] {os.path.basename(cache_path)}：{len(cache):,} Figure")

    split = args.test_split
    if split == "test" and args.dataset == "vg150":
        split = "val"
    recs = [r for r in ds.load_split(split, limit=args.test_n) if r.image_id in cache]
    data = S.build_split(recs, cache, K, args.nmax, args.pmax, strict=False)
    log(f"[split] {len(data.ids):,} Figure | N≤{data.N_max} P≤{data.P_max}")

    hi, lo = 10.0, -50.0

    def geom_scores(d, i, cand_pairs):
        """(n_pairs, K): left/right is determined by center x, above/below is determined by center y."""
        ss = d.s[i].cpu().numpy()            # (N, 8): ncx, ncy, nw, nh, area, aspect, ...
        sc = np.full((len(cand_pairs), K), lo, dtype=np.float32)
        for j, (a, b) in enumerate(cand_pairs):
            dx = float(ss[b, 0] - ss[a, 0])   # >0 ⟹ a is to the left of b
            dy = float(ss[b, 1] - ss[a, 1])   # >0 ⟹ a is above b
            sc[j, iL] = hi * dx
            sc[j, iR] = -hi * dx
            if iA is not None:
                sc[j, iA] = hi * dy
            if iB is not None:
                sc[j, iB] = -hi * dy
        return sc

    hp = HyperParams()
    hp.K = K

    # Original image size has been normalized into s, no W/H required

    inv_idx = None
    sem_p = os.path.join(ART, S.semantic_file(args.dataset))
    if os.path.exists(sem_p):
        inv_idx = np.asarray(torch.load(sem_p, weights_only=False)["inv_idx"],
                             dtype=np.int64)

    res = {}
    m_full = S.evaluate(None, None, data, hp, prior=geom_scores,
                        eval_mode="predcls", inv_idx=inv_idx)
    res["full"] = m_full
    log("\n[Full caliber] " + json.dumps(m_full, ensure_ascii=False))

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
        log(f"[Discriminant view exclude{sorted(excl)}] " + json.dumps(m_ex, ensure_ascii=False))

    log("\n" + "=" * 78)
    log("Interpretation: The baseline top1 That is『trivial geometric solution』ceiling.")
    log("  If model top1 ≈ this value ⟹ The model is equivalent to geometric rules and no visual relationships are learned;")
    log("  If the model is『discriminant view』is significantly higher than this value ⟹ The gain is visual, not geometric.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "cache": os.path.basename(cache_path),
                   "n_img": len(data.ids), "results": res}, f, ensure_ascii=False, indent=2)
    log(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
