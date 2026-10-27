"""统计「二跳成立时三跳为空」的真实比例，即复合表 ∅ 列的经验值。

目的（对应 形式化-空hom与复合表吸收律 的 R1）：
  当前 p_conv 每行归一到 K 个非空谓词，等价于假设 P(∅|二跳)=0。
  本脚本直接数数据：A→B 与 B→C 都在 GT 时，(A,C) 有多大概率不是 GT。

用法：python3 tools/count_twohop_empty.py
"""
import json
import sys
from collections import defaultdict


def _norm(rec):
    """统一成 (img_id, [(s,o)]) 列表。"""
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
    print(f"[{name}] 二跳路径 {n2:,} 条，其中三跳为 ∅ 的 {n_empty:,} 条 "
          f"⟹ P(∅|二跳) = {rate:.4f}（自环跳过 {n_same:,}）")
    print(f"[{name}] 反之 P(非空|二跳) = {1-rate:.4f}，"
          f"当前实现强制为 1.0000 ⟹ 偏差倍数 ≈ {1/max(1-rate,1e-9):.1f}×")
    return rate


if __name__ == "__main__":
    print("=" * 78)
    # VG-150
    try:
        d = json.load(open("<PROJECT_ROOT>/data/vg150/annotations_train.json"))
        imgs = _norm(d)
        print(f"VG 训练图数 {len(imgs):,}")
        count(imgs, "VG-150")
    except Exception as e:                                    # noqa: BLE001
        print("VG 跳过:", type(e).__name__, e)

    print("-" * 78)
    # GQA
    try:
        d = json.load(open("<PROJECT_ROOT>/data/gqa/artifacts/gqa_train_triplets.json"))
        imgs = _norm(d)
        print(f"GQA 训练图数 {len(imgs):,}")
        count(imgs, "GQA")
    except Exception as e:                                    # noqa: BLE001
        print("GQA 跳过:", type(e).__name__, e)
    print("=" * 78)
