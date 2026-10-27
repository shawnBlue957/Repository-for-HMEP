"""GQA scene graph → This article unifies the triple format converter and generates a data-driven inverse predicate table.

## Background

Option B: Contribution points are removed for CLIP distillation, and M_inv is changed to a data-driven inverse predicate table. GQA (Hudson &
Manning, CVPR 2019) Cleaned from Visual Genome, the direction/space relationship is regular
(`on`/`near`/`left of`/`holding`), long-tail distribution is conducive to highlighting the value of structural constraints, which is
A modern evaluation site shared by methods in the past three years such as VETO (ICCV 2023)/DRM (CVPR 2024).

## Data format

GQA official sceneGraphs (https://nlp.stanford.edu/data/gqa/sceneGraphs.zip),
Unzip train_sceneGraphs.json / val_sceneGraphs.json, structure:

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

Key: The relationship is hung under the **source object** as a `relations` list, each containing `name` (predicate) and
`object` (target object id). Triplet = (source object, predicate, target object). bbox is directly available
 (x/y/w/h → [x1,y1,x2,y2]).

## Output

1. `<out>/gqa_<split>_triplets.json` - unified triplet (including boxes/predicate name)
2. `<out>/gqa_predicates.json` - Predicate name list (in descending order of frequency)
3. `<out>/gqa_inv_pairs.json` - data-driven inverse predicate table

## Run

    python tools/prepare_gqa.py \
        --scene-graphs /path/to/train_sceneGraphs.json \
        --out artifacts/gqa --min-n 50 --top-preds 100

## Predicate filtering strategy

There are so many types of GQA relationships (hundreds) that directly counting them will result in: ① Insufficient long-tail predicate samples; ② Inverse predicate table
Loud noise. Therefore, Top-N predicates are filtered by frequency (`--top-preds`) by default, and low-frequency predicates are marked as
”Undefined inverse”, the exit of formula (17) φ - This is consistent with the ”involution only constrains predicates with evidence” in this article.
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
            f"not found {path}.Please download first:\n"
            f"  wget https://nlp.stanford.edu/data/gqa/sceneGraphs.zip\n"
            f"  unzip sceneGraphs.zip   # get train/val_sceneGraphs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_triplets(scene_graphs: dict) -> Tuple[Dict[str, int], List[dict]]:
    """Extract (subject, predicate, object) triples from the GQA scene graph.

    Return (pred_counter, triplets). triplets each:
      {image_id, subj_id, obj_id, predicate, subj_name, obj_name,
       s_box:[x1,y1,x2,y2], o_box:[x1,y1,x2,y2]}
    """
    pred_counter: Dict[str, int] = collections.Counter()
    triplets: List[dict] = []

    for image_id, sg in scene_graphs.items():
        objects = sg.get("objects", {})
        # First create objectId → {name, box}
        oinfo = {}
        for oid, ob in objects.items():
            x, y, w, h = ob.get("x", 0), ob.get("y", 0), ob.get("w", 0), ob.get("h", 0)
            oinfo[oid] = {
                "name": ob.get("name"),
                "box": [x, y, x + w, y + h],
            }
        # Traverse the relations of the source object (relations in GQA is a list, each {name, object})
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
    """Data-driven involution construction (reusing the Top-m reciprocal logic of build_inv_pairs_from_data).

    Reverse evidence: In the same graph, if there is a predicate r on (s,o) and there is a predicate r’ on (o,s), then r’ is r’s
    Reverse candidate. Object pairs in GQA images have unique objectIds and are grouped accurately (better than UnRel’s box approximation).
    """
    K = len(predicates)
    p2i = {p: i for i, p in enumerate(predicates)}
    tot = [0] * K
    rev = [collections.Counter() for _ in range(K)]

    # Group by (image_id, subj_id, obj_id)
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
                    help="train_sceneGraphs.json or val_sceneGraphs.json path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-preds", type=int, default=100,
                    help="only keep Top-N High-frequency predicates, the rest are marked as inverse")
    ap.add_argument("--min-rate", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "artifacts", "gqa")
    os.makedirs(out_dir, exist_ok=True)

    sg = load_scene_graphs(args.scene_graphs)
    pred_counter, triplets = extract_triplets(sg)

    # Take Top-N in descending order of frequency
    top = [p for p, _ in pred_counter.most_common(args.top_preds)]
    top_set = set(top)
    # Only keep the triplet of Top-N predicates (the rest of the predicates take the φ exit and do not enter the involution)
    kept = [t for t in triplets if t["predicate"] in top_set]

    inv = build_inv_pairs(top, kept, min_rate=args.min_rate, min_n=args.min_n)

    split = "train" if "train" in os.path.basename(args.scene_graphs) else \
            ("val" if "val" in os.path.basename(args.scene_graphs) else "unknown")

    # Triplets (including boxes for ROI feature extraction + training assembly)
    ann_path = os.path.join(out_dir, f"gqa_{split}_triplets.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"predicates": top, "triplets": kept}, f, ensure_ascii=False)

    # Inverse predicate table: top-level fields are **completely aligned** with build_inv_pairs_from_data.py output,
    # Enable the main experiment load_data_driven_groups() to directly read the cycles field consumption
    # (--inv-source data caliber). Additional meta information is placed in the *_meta key and does not interfere with reading.
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
    print(f"GQA[{split}] Conversion completed: image {len(sg):,}, relationship {len(triplets):,} Article,"
          f"predicate {len(pred_counter)} species")
    print(f"  Reserve Top-{args.top_preds} predicate, override {len(kept):,} relationship")
    print(f"  2-Loop (with defined inverse): {len(inv['cycles'])} Yes, cover "
          f"{len(inv['cycles'])*2}/{len(top)} predicate ({inv['coverage']*100:.1f}%）")
    for a, b in inv["cycles"][:20]:
        print(f"    {a} ↔ {b}")
    if len(inv["cycles"]) > 20:
        print(f"    ... total {len(inv['cycles'])} Yes")
    print(f"  Symmetry class: {len(inv['symmetric'])} -> {inv['symmetric'][:12]}")
    print(f"  undefined inverse: {len(inv['no_inverse'])} (account for {len(inv['no_inverse'])/len(top)*100:.1f}%）")
    print(f"\n[saved] {ann_path}")
    print(f"[saved] {inv_path}")
    print("=" * 72)
    print("⚠️ Next step:")
    print("  1. Download image: wget https://nlp.stanford.edu/data/gqa/images.zip (~20GB)")
    print("  2. Use this article ROI Pipeline feature extraction (note disk, see cost assessment)")
    print("  3. running four arms + 3-5 a SOTA baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
