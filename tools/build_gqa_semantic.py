"""Constructing GQA’s CLIP semantic tensor + inverse predicate mapping.

Isomorphic to `build_vrd_semantic.py`, **100 predicates / 200 object classes** for GQA.

Inverse predicate table source
------------
GQA’s inverse relationship does not require artificial semantic grouping like VRD: `prepare_gqa.py` has been obtained from the data
Count `cycles` (reciprocal predicate pairs, 8 pairs). This script directly reads the product
`cycles` field of `artifacts/gqa_train_inv_pairs.json`, **no manual judgment**,
Therefore, the coverage of MIR@K is fully reviewable.

Product: `artifacts/gqa_semantic.pt`
    obj_txt (201,512) / v_txt / v_inv / M_inv / J / c_dag / q_feat
    v_*_d / M_inv_d / c_dag_d / inv_idx (K+1,) / predicates etc.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from clip_text_cpu import CLIPTextCPU
from _paths import clip_weights, clip_bpe
from vrp.gqa_data import GQA

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts")
OUT_DIM_D = 256

PRED_TMPL = [
    "a photo of one object that is {pred} another object",
    "{pred}",
    "a photo where one object is {pred} another object",
]


def load_inv_groups(ds) -> list:
    """Read cycles (reciprocal pairs) from the product of prepare_gqa and return pairs of the form [[a],[b]].

    Only pairs with both ends in the predicate table are retained; if one end is missing, it cannot be evaluated and is directly discarded and reported.
    """
    p = os.path.join(ds.artifacts, "gqa_train_inv_pairs.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing inverse predicate statistics:{p} (Run first prepare_gqa.py)")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    valid = set(ds.predicates)
    groups, dropped = [], []
    for cyc in d.get("cycles", []):
        if len(cyc) != 2:
            continue
        a, b = cyc[0], cyc[1]
        if a in valid and b in valid and a != b:
            groups.append(([a], [b]))
        else:
            dropped.append((a, b))
    print(f"[inv] read cycles {len(d.get('cycles', []))} Yes → valid {len(groups)} Yes")
    if dropped:
        print(f"[inv] discard {len(dropped)} Right (not in the predicate table): {dropped[:4]}")
    return groups


def build_inv_idx(pred_names_1b, inv_groups):
    """Predicate name list (including background, index is 1-based predicate number) → inverse predicate index array.

    ⚠️ **Array position and array value are both 1-based predicate numbers** (no inverse is -1),
       The value is **0-based subscript**; read-side uniform `rp = int(inv_idx[p+1]) - 1`.
    """
    name_to_1b = {n: i for i, n in enumerate(pred_names_1b)}
    inv = [-1] * len(pred_names_1b)
    for grp_a, grp_b in inv_groups:
        for A, B in ((grp_a, grp_b), (grp_b, grp_a)):
            if not A or not B or A[0] not in name_to_1b or B[0] not in name_to_1b:
                continue
            inv[name_to_1b[A[0]]] = name_to_1b[B[0]] - 1
    return inv


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    ds = GQA()
    K = ds.K
    preds = [ds.predicate_classes[i + 1] for i in range(K)]
    preds_1b = list(ds.predicate_classes)
    objs = ds.object_classes
    print("=" * 78)
    print(f"GQA Semantic tensor construction | predicate {K} | object class {ds.num_objects}")
    print("=" * 78)

    inv_groups = load_inv_groups(ds)
    enc = CLIPTextCPU(clip_weights(), clip_bpe())

    # ---------- v_txt / v_inv ----------
    def rel_sent(p):
        return f"a photo of one object that is {p} another object"

    v_txt = F.normalize(enc.encode_text([rel_sent(p) for p in preds]), dim=-1)
    inv_name = {}
    for grp_a, grp_b in inv_groups:
        for A, B in ((grp_a, grp_b), (grp_b, grp_a)):
            for p in A:
                if p in preds:
                    inv_name[p] = B[0]
    v_inv = F.normalize(
        enc.encode_text([rel_sent(inv_name.get(p, p)) for p in preds]), dim=-1)
    print(f"v_txt {tuple(v_txt.shape)} | v_inv {tuple(v_inv.shape)}")

    M_inv = v_txt @ v_inv.t()
    diag = M_inv.diag()
    off = M_inv - torch.diag(torch.diag(M_inv))
    print(f"formula(15) M_inv {tuple(M_inv.shape)} range "
          f"[{float(M_inv.min()):.4f}, {float(M_inv.max()):.4f}]")
    print(f"  Diagonal(predicate↔Its reverse) {float(diag.mean()):.4f} | "
          f"off diagonal(random pairing) {float(off.mean()):.4f} | "
          f"Poor {float(diag.mean()-off.mean()):+.4f}")

    J = torch.zeros_like(M_inv)
    J.scatter_(-1, M_inv.argmax(dim=-1).unsqueeze(-1), 1.0)
    deg = J.sum(dim=-1, keepdim=True).clamp(min=1.0)
    c_dag = (J @ v_inv) / deg

    # ---------- obj_txt ----------
    prompts = []
    for o in objs:
        if o == "__background__":
            prompts.append("a photo of background")
        else:
            art = "an" if o[0].lower() in "aeiou" else "a"
            prompts.append(f"{art} photo of {o}, a type of object")
    obj_txt = enc.encode_text(prompts)
    print(f"obj_txt {tuple(obj_txt.shape)} norm mean "
          f"{float(obj_txt.norm(dim=-1).mean()):.4f}")

    # ---------- q_feat ----------
    qp = []
    for p in preds:
        qp.extend(t.format(pred=p) for t in PRED_TMPL)
    q_feat = enc.encode_text(qp).view(K, len(PRED_TMPL), -1).mean(dim=1)
    q_feat = F.normalize(q_feat, dim=-1)
    print(f"q_feat {tuple(q_feat.shape)}（{len(PRED_TMPL)} template mean)")

    # ---------- Dimensionality reduction to model d ----------
    from vrp.config import HyperParams
    d = HyperParams().D
    g = torch.Generator().manual_seed(20261017)
    Q, _ = torch.linalg.qr(torch.randn(512, d, generator=g))
    v_txt_d = F.normalize(v_txt @ Q, dim=-1)
    v_inv_d = F.normalize(v_inv @ Q, dim=-1)
    M_inv_d = v_txt_d @ v_inv_d.t()
    c_dag_d = (J @ v_inv_d) / deg
    agree = int((M_inv.argmax(-1) == M_inv_d.argmax(-1)).sum())
    print(f"Dimensionality reduction 512→{d}: M_inv argmax consistent {agree}/{K} ({100.0*agree/K:.0f}%)")

    # ---------- Inverse predicate table + self-test ----------
    inv_idx = build_inv_idx(ds.predicate_classes, inv_groups)
    n_self = n_oob = n_asym = n_chain = 0
    for i in range(1, K + 1):
        j = inv_idx[i]
        if j < 0:
            continue
        if j == i - 1:
            n_self += 1
        if not (0 <= j < K):
            n_oob += 1
    for grp_a, grp_b in inv_groups:
        if not grp_a or not grp_b:
            continue
        ia = ds.predicate_classes.index(grp_a[0])
        ib = ds.predicate_classes.index(grp_b[0])
        if inv_idx[ia] != ib - 1 or inv_idx[ib] != ia - 1:
            n_asym += 1
    # σ² = id
    for k in range(1, K + 1):
        q = inv_idx[k]
        if q < 0:
            continue
        if inv_idx[q + 1] + 1 != k:
            n_chain += 1
    print(f"\n[self-test] self-referential {n_self} | Value range out of bounds {n_oob} | Head asymmetry {n_asym} | "
          f"Chain is not closed {n_chain}")
    if n_self or n_oob or n_asym or n_chain:
        raise AssertionError(
            f"Inverse predicate table is not self-consistent: self-referential {n_self}, Cross the line {n_oob}、"
            f"asymmetric {n_asym}, Chain is not closed {n_chain}")

    covered = [i for i in range(1, K + 1) if inv_idx[i] >= 0]
    print(f"\n Inverse predicate table:{len(covered)}/{K} predicate has a decidable inverse")
    for i in covered:
        print(f"   {ds.predicate_classes[i]:<20} → "
              f"{ds.predicate_classes[inv_idx[i] + 1]}")

    # Coverage: Use val statistics (train is too large and distributed with the same distribution as training)
    nrel = tot = 0
    for r in ds.load_split("val"):
        for rel in r.relations:
            nrel += 1
            if inv_idx[rel.predicate] >= 0:
                tot += 1
    print(f"\nval relationship {nrel:,} strips, among which {tot:,} The bar falls on the evaluable predicate "
          f"= {100.0*tot/max(nrel,1):.2f}%")
    print("Remarks: GQA of 'to the left of'/'to the right of' account for 89.3%, "
          "Main indicator application mR@K, It is also recommended to add a comparison of ”excluding left and right”.")
    print("=" * 78)

    out = os.path.join(ART, "gqa_semantic.pt")
    torch.save({
        "obj_txt": obj_txt, "v_txt": v_txt, "v_inv": v_inv,
        "M_inv": M_inv, "J": J, "c_dag": c_dag, "q_feat": q_feat,
        "v_txt_d": v_txt_d, "v_inv_d": v_inv_d, "M_inv_d": M_inv_d,
        "c_dag_d": c_dag_d, "proj_Q": Q, "model_dim": d,
        "inv_idx": inv_idx, "inv_groups": inv_groups,
        "predicates": preds, "objects": objs,
        "predicates_1b": preds_1b,
        "coverage_val": {"n_rel": nrel, "n_covered": tot,
                         "pct": round(100.0 * tot / max(nrel, 1), 2)},
    }, out)
    print(f"[Place the order] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
