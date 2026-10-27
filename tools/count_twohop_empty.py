"""Statistics of the true proportion of ”when the second hop is established, the third hop is empty”, that is, the empirical value of column ∅ of the composite table.

Purpose (corresponding to R1 of formalization-empty hom and composite table absorption law):
  Currently, each row of p_conv is normalized to K non-empty predicates, which is equivalent to assuming that P(∅|two hops)=0.
  This script directly counts the data: when A→B and B→C are both in GT, what is the probability that (A,C) is not GT.

Usage: python3 tools/count_twohop_empty.py
"""
import json
import sys
from collections import defaultdict


def _norm(rec):
    """Unified into (img_id, [(s,o)]) list."""
    out = []
    if isinstance(rec, dict):
        rec = rec.get("data", [rec])
    for im in rec:
        rels = []
        if isinstance(im, dict):
            rs = im.get("relationships") or im.get("relations") or []
            for r in rs:
                if isinstance(r, dict):
                    s = r.get("subject_id", r.get("subject"))
                    o = r.get("object_id", r.get("object"))
                else:
                    s, o = r[0], r[1]
                if s is not None and o is not None:
                    rels.append((s, o))
        out.append(rels)
    return out


def count(imgs, name):
    n2 = n_empty = 0
    n_same = 0
    for rels in imgs:
        if not rels:
            continue
        gt = set(rels)
        out_e = defaultdict(list)
        in_e = defaultdict(list)
        for a, b in rels:
            out_e[a].append(b)
            in_e[b].append(a)
        for b, srcs in in_e.items():
            dsts = out_e.get(b)
            if not dsts:
                continue
            for a in srcs:
                for c in dsts:
                    if a == c:
                        n_same += 1
                        continue
                    n2 += 1
                    if (a, c) not in gt:
                        n_empty += 1
    rate = (n_empty / n2) if n2 else 0.0
    print(f"[{name}] Second hop path {n2:,} strips, three of which jump to ∅ of {n_empty:,} Article "
          f"⟹ P(∅|second jump) = {rate:.4f} (Skip from loop {n_same:,}）")
    print(f"[{name}] On the contrary P(Not empty|second jump) = {1-rate:.4f}，"
          f"The current implementation forces 1.0000 ⟹ Deviation multiple ≈ {1/max(1-rate,1e-9):.1f}×")
    return rate


if __name__ == "__main__":
    print("=" * 78)
    # VG-150
    try:
        d = json.load(open("<PROJECT_ROOT>/data/vg150/annotations_train.json"))
        imgs = _norm(d)
        print(f"VG Number of training images {len(imgs):,}")
        count(imgs, "VG-150")
    except Exception as e:                                    # noqa: BLE001
        print("VG skip:", type(e).__name__, e)

    print("-" * 78)
    # GQA
    try:
        d = json.load(open("<PROJECT_ROOT>/data/gqa/artifacts/gqa_train_triplets.json"))
        imgs = _norm(d)
        print(f"GQA Number of training images {len(imgs):,}")
        count(imgs, "GQA")
    except Exception as e:                                    # noqa: BLE001
        print("GQA skip:", type(e).__name__, e)
    print("=" * 78)
