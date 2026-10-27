"""Data-driven derivation of inverse predicate involutions from ground-truth annotations (replacing manual semantic pairing).

## Why it is necessary to change to data-driven (important discovery)

In the last round, I manually matched 30 groups of inverse predicates semantically (`tools/inv_groups_full.py`),
Then use **VRD train real annotation** group-by-group inspection (`/tmp/_ck.py`), the result is:

    5 hits/25 no hits

That is to say, only 5 of the 30 manually matched groups can stand up in the real annotation, and the hit rate of the remaining 25 groups is ≈ 0%.
Actual measurement group by group (r → candidate inverse → hit rate → Top3 true inverse):

| r | Manual candidate inversion | Hit rate | Top3 true inversion |
|---|---|---|---|
| `above` | under | 10.6% | below(355), under(256), beneath(160) |
| `behind` | in the front of | **19.1%** | in the front of(428) ✓ |
| `on` | under | 6.7% | wear(580), has(427), under(342) |
| `beside` | across | 0.3% | beside(179), next to(98), near(42) |
| `hold` | carry | 0.0% | on(20), next to(19), above(18) |
| `wear` | attach to | 0.0% | on(580), near(16), cover(14) |
| `in` | outside of | 0.0% | has(47), on(25), under(14) |

**Conclusion**: Most of the inverse annotations of VRD are not the logical inverse of the predicate at all (such as the inverse of `wear`
Top1 is `on` rather than ”worn”; Top1 of `hold` is `on`). This is consistent with what was measured previously in this project.
”The reverse label is inconsistent with the inverse predicate table 42.2% of the time, and the predicate itself has no defined inverse 57.5%” is completely consistent.
This illustrates a systematic gap between manual semantic tables and dataset annotation habits.

**⟹ Correct approach: Let the data decide who is the logical inverse. ** This module automatically constructs an involution according to the strength of the marked evidence.
And list the exits for `φ` of formula (17) without evidence (”r has no defined inverse”),
Rather than degenerating into `Inv(r)=r` (the latter is equivalent to requiring symmetry, and the semantics are opposite to ”having an inverse”).

## Construction rules (all recalculable, no manual semantic judgment)

For each predicate r:
1. Count the co-occurrence distribution `rev[r]` of the reverse pair (o, r’, s) when (s, r, o) appears in train;
2. Take `best = argmax_r’ rev[r][r’]`, `rate = rev[r][best] / n(r)`;
3. **Reciprocal consistency test**: only when `best(best) == r` **and** `rev[best][r] > 0`
   Accept the matching - otherwise the ”inverse” is a one-way pseudo-correlation and cannot be combined;
4. Strength threshold `MIN_RATE` (default 0.05): Hit rate lower than this is regarded as ”undefined inverse” ⟹ Take the φ exit;
5. Frequency threshold `MIN_N` (default 100): Too few samples are not enough to determine ⟹ Take the φ exit;
6. Reflexive predicates (`best == r` and have a high reflexive proportion)⟹ are marked as **symmetry class** and are processed by Equation (24) `L_sym`,
   **No involution** (involution requires `r ≠ r’`).

**Output**: `{2-cyclic pair list, symmetry class list, undefined inverse list}`, the latter is the semantic set of φ.

Run:
    python tools/build_inv_pairs_from_data.py --dataset vrd
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_RATE = 0.05      # Reverse hit rate threshold
MIN_N = 100          # Minimum number of occurrences of predicate


def load_vg(split: int = 0):
    """Locate and read VG-SGG.h5.

    ⚠️ The path also needs to be cross-platform: the data is in `data/vg/` at the project root, and the server is in `~/vrp/data/vg/`.
    """
    import h5py
    env = os.environ.get("VG_H5")
    cands = [env] if env else []
    cands += [
        os.path.join(ROOT, "data", "vg", "VG-SGG.h5"),
        os.path.join(os.path.dirname(ROOT), "data", "vg", "VG-SGG.h5"),
        os.path.join(ROOT, "..", "data", "vg", "VG-SGG.h5"),
        os.path.expanduser("~/vrp/data/vg/VG-SGG.h5"),
    ]
    path = next((c for c in cands if c and os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(f"not found VG-SGG.h5.Tried:{cands}\n"
                                f"Please set environment variables VG_H5 point to the file")
    d = json.load(open(os.path.join(os.path.dirname(path), "VG-SGG-dicts.json"),
                       encoding="utf-8"))
    # The key order of dicts.json is the label/predicate index; remove the background placeholder
    labels = d["label_to_idx"]
    pmap = d["predicate_to_idx"]
    # *_to_idx of VG-SGG-dicts.json is a {name: index} dictionary, and **index starts from 1**
    # (Number 0 is reserved for background and does not exist in dict).
    # Sort by index value and restore to 0-based list: index i → i-1 of the list,
    # This way predicates[0] == ’above’ is consistent with the 0-based predicate table of semantic_path.pt.
    by_idx = sorted(pmap.items(), key=lambda x: x[1])
    assert by_idx[0][1] == 1, f"Expected predicate index from 1 Start, get it right {by_idx[0][1]}"
    preds = [name for name, _ in by_idx]
    print(f"data file: {path}")
    return path, labels, preds


def load_vg_json(root: str = None, split: str = "train"):
    """⚠️ 2026-10-18 New: Read VG from **JSON annotations** (vg150/annotations_*.json).

    Background: `VG-SGG.h5` on the server no longer exists (the data is in json format instead),
    The original `load_vg()` will inevitably cause FileNotFoundError when taking the h5py branch.
    The json version directly gives `relations[*].predicate_name`, which is more direct than the integer index of h5.

    Return (path, labels_dict, predicate_names).
    """
    cands = [root] if root else []
    cands += [os.path.expanduser("~/vrp/data/vg150"),
              os.path.join(ROOT, "data", "vg150"),
              os.path.join(os.path.dirname(ROOT), "data", "vg150")]
    base = next((c for c in cands if c and os.path.exists(
        os.path.join(c, f"annotations_{split}.json"))), None)
    if base is None:
        raise FileNotFoundError(
            f"not found vg150/annotations_{split}.json.Tried:{cands}")
    path = os.path.join(base, f"annotations_{split}.json")
    data = json.load(open(path, encoding="utf-8"))
    # Predicate name list: in the order of first appearance, maintaining the same 0-based order as semantic_path.pt
    preds, seen = [], set()
    labels = {}
    for rec in data:
        for ob in rec.get("objects", []):
            ln = ob.get("label_name")
            if ln is not None and ln not in labels:
                labels[ln] = len(labels) + 1     # 1-based, 0 is left for background
        for rel in rec.get("relations", []):
            pn = rel.get("predicate_name")
            if pn is not None and pn not in seen:
                seen.add(pn)
                preds.append(pn)
    print(f"data file: {path}  Number of predicates={len(preds)}  object class={len(labels)}")
    return path, labels, preds


def collect_reverse_vg_json(path: str, labels: dict, preds: list):
    """⚠️ 2026-10-18: Collecting reverse co-occurrences from **json annotations** (aligned to h5 version semantics).

    Definition of reciprocal evidence (consistent with h5 version):
        For the relationship (s, o, r), if **the same pair of objects** (o, s) is also marked with the predicate r’,
        Then r’ is the **reverse candidate** of r. rev[r][r’] = count (appears in reverse).
    This is the most direct annotation evidence of ”logical inverse” (the reverse of `on` is `under`, rather than arbitrary co-occurrence).

    Return (tot, rev):
        tot[r] = total number of occurrences of predicate r
        rev[r] = Counter{r’: times}
    """
    import collections
    p2i = {n: i for i, n in enumerate(preds)}
    data = json.load(open(path, encoding="utf-8"))
    K = len(preds)
    tot = [0] * K
    rev = [collections.Counter() for _ in range(K)]
    for rec in data:
        objs = rec.get("objects", [])
        n_obj = len(objs)
        rels = rec.get("relations", [])
        # (subj,obj) → [predicate subscript,...]
        pair = collections.defaultdict(list)
        for rel in rels:
            pn = rel.get("predicate_name")
            if pn is None or pn not in p2i:
                continue
            si, oi = rel.get("subject"), rel.get("object")
            if si is None or oi is None:
                continue
            if not (0 <= si < n_obj and 0 <= oi < n_obj):
                continue
            pair[(si, oi)].append(p2i[pn])
        for rel in rels:
            pn = rel.get("predicate_name")
            if pn is None or pn not in p2i:
                continue
            si, oi = rel.get("subject"), rel.get("object")
            if si is None or oi is None:
                continue
            if not (0 <= si < n_obj and 0 <= oi < n_obj):
                continue
            ri = p2i[pn]
            tot[ri] += 1
            for rj in pair.get((oi, si), ()):      # Predicate on reverse pair
                rev[ri][rj] += 1
    return tot, rev


def collect_reverse_vg(path: str, labels: dict, preds: dict, split: int):
    """Traverse VG-SGG.h5 to count reverse co-occurrences. split: 0=train, 1=val, 2=test"""
    import h5py
    import numpy as np
    f = h5py.File(path, "r")
    sp = f["split"][:]
    lb = f["labels"][:].ravel()
    rl = f["predicates"][:].ravel()
    rr = f["relationships"][:]
    i2b = f["img_to_first_box"][:]
    i2lb = f["img_to_last_box"][:]
    i2r = f["img_to_first_rel"][:]
    i2lr = f["img_to_last_rel"][:]

    K = len(preds)
    rev = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for i in range(len(sp)):
        if int(sp[i]) != split:
            continue
        objs = i2b[i] + np.arange(i2lb[i] - i2b[i])
        rels = i2r[i] + np.arange(i2lr[i] - i2r[i])
        lab = {int(o): int(lb[o]) for o in objs}
        # Reverse index: (subj,obj) -> [(pred, ...)]
        pair = collections.defaultdict(list)
        for r in rels:
            s, o = int(rr[r][0]), int(rr[r][1])
            if s in lab and o in lab:
                pair[(s, o)].append(int(rl[r]))   # Keep h5 original index
        for r in rels:
            s, o = int(rr[r][0]), int(rr[r][1])
            if s not in lab or o not in lab:
                continue
            ri = int(rl[r]) - 1          # h5 predicate index starts from 1
            tot[ri] += 1
            for pi in pair.get((o, s), ()):
                rev[ri][pi - 1] += 1     # Same as above
    f.close()
    return tot, rev


def load_vrd(split: str = "train"):
    """Position VRD callouts.

    ⚠️ Path benchmark: This file is under `code/tools/`, but the data is in the **upper level** `data/vrd/`;
    On the server, it is in `~/vrp/data/vrd/`. Therefore, click on the ”candidate list” to detect one by one.
    And supports `VRD_JSON_DIR` environment variable override.
    """
    env = os.environ.get("VRD_JSON_DIR")
    cands = [env] if env else []
    cands += [
        os.path.join(ROOT, "data", "vrd", "json_dataset"),
        os.path.join(os.path.dirname(ROOT), "data", "vrd", "json_dataset"),
        os.path.join(ROOT, "..", "data", "vrd", "json_dataset"),
        os.path.expanduser("~/vrp/data/vrd/json_dataset"),
    ]
    base = None
    for c in cands:
        if c and os.path.exists(os.path.join(c, "predicates.json")):
            base = os.path.abspath(c); break
    if base is None:
        raise FileNotFoundError(
            f"not found VRD Label. Tried:{cands}\n"
            f"Please set environment variables VRD_JSON_DIR Point to include predicates.json directory")
    ann = json.load(open(os.path.join(base, f"annotations_{split}.json"), encoding="utf-8"))
    preds = json.load(open(os.path.join(base, "predicates.json"), encoding="utf-8"))
    print(f"data directory: {base}")
    return ann, preds


def key_of(o):
    return (tuple(o["bbox"]), o["category"])


def collect_reverse(ann) -> Tuple[collections.Counter, collections.defaultdict]:
    rev = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for _img, lst in ann.items():
        idx = {}
        for r in lst:
            idx.setdefault(key_of(r["subject"]), []).append((r["predicate"], key_of(r["object"])))
        for r in lst:
            ri = r["predicate"]
            tot[ri] += 1
            sk, ok_ = key_of(r["subject"]), key_of(r["object"])
            for (pi, oj) in idx.get(ok_, []):
                if oj == sk:
                    rev[ri][pi] += 1
    return tot, rev


def build_involution(preds: List[str], tot, rev, *,
                     min_rate: float = MIN_RATE, min_n: int = MIN_N,
                     top_m: int = 3):
    """Strict involution constructed from labeled evidence (2-cycle ∪ fixed point).

    ⚠️ Correction of reciprocal test (actual measurement driver):
       The reverse Top1 of `above` of VG is `under`(279), but the reverse Top1 of `under` is `on`(895),
       Top2 is `above`(279). If ”each other’s Top1” is required, `above↔under` will be used
       **Correct spatial inverse relationship** was misjudged as one-way pseudo-correlation and discarded (only 1 pair of has↔on is left in the measured VG).

       Therefore, it is changed to: **Candidate r’ must appear in the Top-m reverse of r, and r appears in the Top-m reverse of r’**
       (Top-m means acceptance), and then choose the best one based on hit rate. In this way, above↔under can be retained,
       The spurious correlation of actions such as `on→has` is eliminated because there is no `on` in the Top-m of `has`.
    """
    K = len(preds)
    sigma = list(range(K))          #Default fixed point
    sym, noinv, pairs = set(), set(), []

    for i, p in enumerate(preds):
        n = tot[i]
        if n < min_n:
            noinv.add(i); continue
        c = rev[i]
        if not c:
            noinv.add(i); continue

        # Candidates: Top-m in descending order of hit rate
        cands = [j for j, _ in c.most_common(top_m)]
        # Symmetry class: Top1 is itself and the reflexive proportion reaches the threshold
        if cands and cands[0] == i and c[i] / n >= min_rate:
            sym.add(i); continue

        chosen = None
        for j in cands:
            if j == i:
                continue
            rate = c[j] / n
            if rate < min_rate:
                break
            cj = rev[j]
            tops_j = [x for x, _ in cj.most_common(top_m)]
            if i in tops_j and cj[i] > 0:      # mutually Top-m
                chosen = j
                break
        if chosen is None:
            noinv.add(i); continue
        pairs.append((i, chosen))

    # Only **mutually disjoint** 2-cycles are retained; the conflictors degenerate into fixed points (otherwise σ² ≠ id)
    used, cyc = set(), []
    for i, j in pairs:
        if i in used or j in used or i == j:
            continue
        used.add(i); used.add(j); cyc.append((i, j))
    for i, j in cyc:
        sigma[i], sigma[j] = j, i

    # ⚠️ 2026-10-18 Fix (A8 contradiction): **Remove overlap of noinv/sym with 2-ring**.
    #
    # Cause: When the **one-way** hit rate of predicate i is insufficient, it will fall into noinv (take the φ exit);
    # But the **reverse** check of predicate j may still select i as a partner ⟹ i also appears in
    # In the two sets ”with reciprocal evidence (enter 2-ring)” and ”without reciprocal evidence (take φ exit)”.
    # Measured VG: `above` / `behind` / `on` three predicates appear in two places at the same time
    # (The pairings are under↔above, in front of↔behind, and has↔on).
    #
    # Semantically the two are **mutually exclusive**: Now that the reciprocity (2-ring) has been confirmed, we should no longer claim ”undefined inverse”.
    # So 2-ring takes precedence and removes its members from noinv/sym.
    cyc_members = {i for i, _ in cyc} | {j for _, j in cyc}
    noinv -= cyc_members
    sym -= cyc_members
    return sigma, cyc, sym, noinv


def sigma_to_matrix(sigma, K) -> List[List[float]]:
    J = [[0.0] * K for _ in range(K)]
    for i in range(K):
        J[i][sigma[i]] = 1.0
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vrd", "vg150"], default="vrd")
    ap.add_argument("--split", default="train",
                    help="vrd: train/test；vg150: 0=train 1=val 2=test")
    ap.add_argument("--min-rate", type=float, default=MIN_RATE)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "vg150":
        # ⚠️ 2026-10-18: VG-SGG.h5 no longer exists on the server (the data uses json format instead),
        # Therefore, the json annotation path is preferred; h5 will be returned only when json is unavailable.
        try:
            path, labels, preds = load_vg_json(split=args.split)
            tot, rev = collect_reverse_vg_json(path, labels, preds)
        except FileNotFoundError:
            sp = {"train": 0, "val": 1, "test": 2}.get(args.split, 0)
            path, labels, preds = load_vg()
            tot, rev = collect_reverse_vg(path, labels, preds, sp)
    else:
        ann, preds = load_vrd(args.split)
        tot, rev = collect_reverse(ann)

    sigma, cyc, sym, noinv = build_involution(preds, tot, rev,
                                              min_rate=args.min_rate, min_n=args.min_n)
    K = len(preds)

    # Verify alignment
    import numpy as np
    J = np.asarray(sigma_to_matrix(sigma, K))
    adj = float(np.abs(J @ J - np.eye(K)).max())

    print(f"\n Data-driven involution construction ({args.dataset} {args.split}）")
    print("=" * 72)
    print(f"Number of predicates K={K}  min_rate={args.min_rate}  min_n={args.min_n}")
    print(f"2-Loops (disjoint)): {len(cyc)} Yes, cover {2*len(cyc)} predicate"
          f"（{2*len(cyc)/K*100:.1f}%）")
    for i, j in cyc:
        ri = rev[i][j] / tot[i]
        rj = rev[j][i] / tot[j]
        print(f"    {preds[i]:<16s} ↔ {preds[j]:<16s} "
              f"hit rate {ri*100:5.1f}% / {rj*100:5.1f}%")
    print(f"Symmetry type (walking type 24 L_sym, No involution): {len(sym)} a → "
          f"{[preds[i] for i in sorted(sym)][:12]}")
    print(f"Undefined inverse (moving form 17 of φ export): {len(noinv)} (accounting for {len(noinv)/K*100:.1f}%）")
    print(f"J²−I Deviation = {adj:.2e}  {'✅ strict involution' if adj < 1e-9 else '❌'}")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out) if args.out else
                os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = args.out or os.path.join(ROOT, "artifacts", f"inv_pairs_{args.dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset, "split": args.split,
            "predicates": preds,
            "cycles": [[preds[i], preds[j]] for i, j in cyc],
            "symmetric": [preds[i] for i in sorted(sym)],
            "no_inverse": [preds[i] for i in sorted(noinv)],
            "min_rate": args.min_rate, "min_n": args.min_n, "split": args.split,
            "adj_err": adj,
        }, f, ensure_ascii=False, indent=2)
    print(f"has been written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())