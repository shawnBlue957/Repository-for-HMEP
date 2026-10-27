"""GQA 场景图 → 本文统一三元组格式 转换器，并生成数据驱动逆谓词表。

## 背景

方案 B：贡献点移除 CLIP 蒸馏，M_inv 改数据驱动逆谓词表。GQA（Hudson &
Manning, CVPR 2019）由 Visual Genome 清洗而来，方向/空间关系规整
（`on`/`near`/`left of`/`holding`），长尾分布利于凸显结构约束价值，是
VETO（ICCV 2023）/ DRM（CVPR 2024）等近三年方法共用的现代评测场。

## 数据格式

GQA 官方 sceneGraphs（https://nlp.stanford.edu/data/gqa/sceneGraphs.zip），
解压得 train_sceneGraphs.json / val_sceneGraphs.json，结构：

    {
      "<imageId>": {
        "width": 640, "height": 480,
        "objects": {
          "<objectId>": {
            "name": "chair", "x": 220, "y": 310, "w": 50, "h": 80,
            "attributes": ["brown", ...],
            "relations": {
              "<relId>": { "name": "on", "object": "<objectId>" }
            }
          }
        }
      }
    }

关键：关系以 `relations` 列表挂在**源对象**下，每条含 `name`（谓词）与
`object`（目标对象 id）。三元组 = (源对象, 谓词, 目标对象)。bbox 直接可用
（x/y/w/h → [x1,y1,x2,y2]）。

## 产出

1. `<out>/gqa_<split>_triplets.json`  —— 统一三元组（含 boxes/谓词名）
2. `<out>/gqa_predicates.json`         —— 谓词名表（按频率降序）
3. `<out>/gqa_inv_pairs.json`          —— 数据驱动逆谓词表

## 运行

    python tools/prepare_gqa.py \
        --scene-graphs /path/to/train_sceneGraphs.json \
        --out artifacts/gqa --min-n 50 --top-preds 100

## 谓词筛选策略

GQA 关系种类极多（数百），直接全量会导致：① 长尾谓词样本不足；② 逆谓词表
噪声大。故默认按频率筛选 Top-N 谓词（`--top-preds`），并把低频谓词标记为
「无定义逆」，走式(17) φ 出口——这与本文「对合仅约束有证据的谓词」一致。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_scene_graphs(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}。请先下载：\n"
            f"  wget https://nlp.stanford.edu/data/gqa/sceneGraphs.zip\n"
            f"  unzip sceneGraphs.zip   # 得到 train/val_sceneGraphs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_triplets(scene_graphs: dict) -> Tuple[Dict[str, int], List[dict]]:
    """从 GQA 场景图提取 (subject, predicate, object) 三元组。

    返回 (pred_counter, triplets)。triplets 每条：
      {image_id, subj_id, obj_id, predicate, subj_name, obj_name,
       s_box:[x1,y1,x2,y2], o_box:[x1,y1,x2,y2]}
    """
    pred_counter: Dict[str, int] = collections.Counter()
    triplets: List[dict] = []

    for image_id, sg in scene_graphs.items():
        objects = sg.get("objects", {})
        # 先建 objectId → {name, box}
        oinfo = {}
        for oid, ob in objects.items():
            x, y, w, h = ob.get("x", 0), ob.get("y", 0), ob.get("w", 0), ob.get("h", 0)
            oinfo[oid] = {
                "name": ob.get("name"),
                "box": [x, y, x + w, y + h],
            }
        # 遍历源对象的 relations（GQA 中 relations 是 list，每条 {name, object}）
        for subj_id, ob in objects.items():
            subj = oinfo.get(subj_id)
            if not subj:
                continue
            rels = ob.get("relations", [])
            if isinstance(rels, dict):
                rels = list(rels.values())
            for rel in rels:
                pred_name = rel.get("name")
                obj_id = rel.get("object")
                if not pred_name or obj_id not in oinfo:
                    continue
                obj = oinfo[obj_id]
                pred_counter[pred_name] += 1
                triplets.append({
                    "image_id": str(image_id),
                    "subj_id": str(subj_id),
                    "obj_id": str(obj_id),
                    "predicate": pred_name,
                    "subj_name": subj["name"],
                    "obj_name": obj["name"],
                    "s_box": subj["box"],
                    "o_box": obj["box"],
                })
    return pred_counter, triplets


def build_inv_pairs(predicates: List[str], triplets: List[dict],
                    min_rate: float = 0.05, min_n: int = 50,
                    top_m: int = 3) -> dict:
    """数据驱动对合构造（复用 build_inv_pairs_from_data 的 Top-m 互反逻辑）。

    反向证据：同图内，若 (s,o) 上有谓词 r，且 (o,s) 上有谓词 r'，则 r' 是 r 的
    反向候选。GQA 图内物体对有唯一 objectId，分组精确（优于 UnRel 的 box 近似）。
    """
    K = len(predicates)
    p2i = {p: i for i, p in enumerate(predicates)}
    tot = [0] * K
    rev = [collections.Counter() for _ in range(K)]

    # 按 (image_id, subj_id, obj_id) 分组
    pair_map = collections.defaultdict(list)
    for t in triplets:
        pi = p2i.get(t["predicate"], -1)
        if pi < 0:
            continue
        pair_map[(t["image_id"], t["subj_id"], t["obj_id"])].append(pi)

    for t in triplets:
        ri = p2i.get(t["predicate"], -1)
        if ri < 0:
            continue
        tot[ri] += 1
        for rj in pair_map.get((t["image_id"], t["obj_id"], t["subj_id"]), ()):
            rev[ri][rj] += 1

    sigma = list(range(K))
    sym, noinv, pairs = set(), set(), []
    for i in range(K):
        n = tot[i]
        if n < min_n:
            noinv.add(i); continue
        c = rev[i]
        if not c:
            noinv.add(i); continue
        cands = [j for j, _ in c.most_common(top_m)]
        if cands and cands[0] == i and c[i] / n >= min_rate:
            sym.add(i); continue
        chosen = None
        for j in cands:
            if j == i:
                continue
            if c[j] / n < min_rate:
                break
            cj = rev[j]
            tops_j = [x for x, _ in cj.most_common(top_m)]
            if i in tops_j and cj[i] > 0:
                chosen = j
                break
        if chosen is None:
            noinv.add(i); continue
        pairs.append((i, chosen))

    used, cyc = set(), []
    for i, j in pairs:
        if i in used or j in used or i == j:
            continue
        used.add(i); used.add(j); cyc.append((i, j))
    cyc_members = {i for i, _ in cyc} | {j for _, j in cyc}
    noinv -= cyc_members
    sym -= cyc_members

    return {
        "predicates": predicates,
        "cycles": [[predicates[i], predicates[j]] for i, j in cyc],
        "symmetric": [predicates[i] for i in sorted(sym)],
        "no_inverse": [predicates[i] for i in sorted(noinv)],
        "min_rate": min_rate, "min_n": min_n,
        "coverage": len(cyc_members) / K if K else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-graphs", required=True,
                    help="train_sceneGraphs.json 或 val_sceneGraphs.json 路径")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-preds", type=int, default=100,
                    help="只保留 Top-N 高频谓词，其余标记为无逆")
    ap.add_argument("--min-rate", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "artifacts", "gqa")
    os.makedirs(out_dir, exist_ok=True)

    sg = load_scene_graphs(args.scene_graphs)
    pred_counter, triplets = extract_triplets(sg)

    # 频率降序取 Top-N
    top = [p for p, _ in pred_counter.most_common(args.top_preds)]
    top_set = set(top)
    # 只保留 Top-N 谓词的三元组（其余谓词走 φ 出口，不进对合）
    kept = [t for t in triplets if t["predicate"] in top_set]

    inv = build_inv_pairs(top, kept, min_rate=args.min_rate, min_n=args.min_n)

    split = "train" if "train" in os.path.basename(args.scene_graphs) else \
            ("val" if "val" in os.path.basename(args.scene_graphs) else "unknown")

    # 三元组（含 boxes，供 ROI 特征抽取 + 训练装配）
    ann_path = os.path.join(out_dir, f"gqa_{split}_triplets.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"predicates": top, "triplets": kept}, f, ensure_ascii=False)

    # 逆谓词表：顶层字段与 build_inv_pairs_from_data.py 输出**完全对齐**，
    # 使主实验 load_data_driven_groups() 能直接读 cycles 字段消费
    # （--inv-source data 口径）。附加元信息放 *_meta 键，不干扰读取。
    inv_out = dict(inv)                       # {predicates, cycles, symmetric,
                                              #  no_inverse, min_rate, min_n, coverage}
    inv_out["dataset"] = "gqa"
    inv_out["split"] = split
    inv_out["meta"] = {
        "n_images": len(sg),
        "n_relations_total": len(triplets),
        "n_predicates_total": len(pred_counter),
        "top_preds": args.top_preds,
        "kept_triplets": len(kept),
    }
    inv_path = os.path.join(out_dir, f"gqa_{split}_inv_pairs.json")
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv_out, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"GQA[{split}] 转换完成：图像 {len(sg):,}，关系 {len(triplets):,} 条，"
          f"谓词 {len(pred_counter)} 种")
    print(f"  保留 Top-{args.top_preds} 谓词，覆盖 {len(kept):,} 条关系")
    print(f"  2-循环（有定义逆）: {len(inv['cycles'])} 对，覆盖 "
          f"{len(inv['cycles'])*2}/{len(top)} 谓词（{inv['coverage']*100:.1f}%）")
    for a, b in inv["cycles"][:20]:
        print(f"    {a} ↔ {b}")
    if len(inv["cycles"]) > 20:
        print(f"    ... 共 {len(inv['cycles'])} 对")
    print(f"  对称类: {len(inv['symmetric'])} -> {inv['symmetric'][:12]}")
    print(f"  无定义逆: {len(inv['no_inverse'])}（占 {len(inv['no_inverse'])/len(top)*100:.1f}%）")
    print(f"\n[saved] {ann_path}")
    print(f"[saved] {inv_path}")
    print("=" * 72)
    print("⚠️ 下一步：")
    print("  1. 下载图像：wget https://nlp.stanford.edu/data/gqa/images.zip (~20GB)")
    print("  2. 用本文 ROI 流水线抽取特征（注意磁盘，见成本评估）")
    print("  3. 跑四臂 + 3-5 个 SOTA 基线")
    return 0


if __name__ == "__main__":
    sys.exit(main())
