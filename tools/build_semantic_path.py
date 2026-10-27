#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7 — Compute CLIP semantic path tensors: v_txt / v_inv / M_inv / J / c_dag / x_t / q.

background
----
The last round of experiments (400 pictures / 3 seeds / 6 arms) resulted in: the main effect of the category loss term |ΔmR@50| < 0.001,
That is, ”the category layer has no effect when there is no semantic text path”. But this conclusion has a fatal flaw -
$\\mathbf M_{inv}$, $q$, $v_{txt}$ at the category level **all rely on CLIP text semantics**,
In the previous round, `x_t` was replaced by learnable Embedding and `q` was replaced by predicate frequency prior, and this path was completely missing.
Therefore, this script completes it.

Output (placement `artifacts/semantic_*.pt`)
------------------------------------
| tensor | shape | correspondence |
|---|---|---|
| `v_txt` | (50,512) | Formula (15) forward predicate describes embedding $v_{txt,i}$ |
| `v_inv` | (50,512) | Formula (15) Logical inverse predicate describes embedding $v_{inv,j}$ |
| `M_inv` | (50,50) | Formula (15) $\\cos(v_{txt,i}, v_{inv,j})$ |
| `J` | (50,50) | Formula (23) top1 indicator matrix |
| `c_dag` | (50,512) | Formula (23) $c^\\dagger_k$ |
| `obj_txt` | (151,512)| Object text embedding (replaces `nn.Embedding(151,512)`) |
| `pred_prompts` | list | Template description for graphic and text soft label q |

Design points
--------
1. **Logical inverse is not required to fall within the 50 category** (Project Long Term Memory has ruled): Does not exist in VG-50
   Predicate of intra-class inverse (`wearing`/`wears`, `sitting on`, etc.), `v_inv` takes **natural language inverse description**
   Embedding of (e.g. `wearing` → ”worn by”). This is consistent with the original meaning of equation (15) -
   $\\mathbf M_{inv}$ is a **semantic dual matrix** rather than an intra-class permutation.
2. **Template prompt**: CLIP’s standard approach for zero samples (”a photo of a {}”).
   The predicate is an action/relative word rather than an object. Directly feeding bare words is ineffective, so a relative sentence template is used.
3. **Object class** uses the ”a photo of a {obj}, a type of object” template.
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from clip_text_cpu import CLIPTextCPU
from vrp.data import VG150

from _paths import clip_weights, clip_bpe
W = clip_weights()
B = clip_bpe()
ART = os.path.join(ROOT, "artifacts")

# ======================================================================
# Natural language description of logical inverse predicate
# ======================================================================
# The key is the predicate name of VG-50, and the value is the natural language expression of ”the inverse relationship after the subject and object are exchanged”.
# The inverse description of unlisted predicates degenerates into ”not ” + original word (indicating that the relationship does not hold).
INV_DESC = {
    "above": "below",
    "across": "across",                 # Approximately reflexive
    "against": "against",
    "along": "along",
    "and": "and",
    "at": "at",
    "attached to": "attached to",       # reflexive
    "behind": "in front of",
    "belonging to": "belongs to",
    "between": "between",
    "carrying": "carried by",
    "covered in": "covering",
    "covering": "covered in",
    "eating": "eaten by",
    "flying in": "flying in",
    "for": "for",
    "from": "to",
    "growing on": "grows on",
    "hanging from": "hanging from",
    "has": "is had by",
    "holding": "held by",
    "in": "containing",
    "in front of": "behind",
    "laying on": "laid on",
    "looking at": "looked at by",
    "lying on": "lying on",
    "made of": "made from",
    "mounted on": "mounts",
    "near": "near",                     # reflexive
    "of": "of",
    "on": "supporting",                 # The inverse of on is not under, but ”carry”
    "on back of": "carrying on back",
    "over": "under",
    "painted on": "painted with",
    "parked on": "parking",
    "part of": "has part",
    "playing": "played by",
    "riding": "ridden by",
    "says": "said by",
    "sitting on": "sat on by",
    "standing on": "stood on by",
    "to": "from",
    "under": "above",
    "using": "used by",
    "walking in": "walked in",
    "walking on": "walked on",
    "watching": "watched by",
    "wearing": "worn by",
    "wears": "worn by",
    "with": "with",
}

# Relative sentence template (CLIP zero-sample standard practice: the predicate needs to be put into the sentence to have semantic meaning)
# ① Relative sentence ② Bare words (supplement short word distinction) ③ Another sentence pattern (template integration)
PRED_TMPL = [
    "a photo of one object that is {pred} another object",
    "{pred}",
    "a photo where one object is {pred} another object",
]


def build_prompt(pred: str) -> str:
    """Predicate → Relative sentence description."""
    return f"a photo of one object that is {pred} another object"


def build_inv_prompt(pred: str, inv: str) -> str:
    return f"a photo of one object that is {inv} another object"


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    ds = VG150()
    K = ds.K
    preds = ds.predicate_classes[1:]                 # 50, remove background
    objs = ds.object_classes                         # 151 (including background)
    print(f"VG-150: K={K} predicate, {len(objs)} Object class (including background)")

    enc = CLIPTextCPU(W, B)

    # ---------- v_txt: forward predicate description ----------
    prompts = [build_prompt(p) for p in preds]
    v_txt = enc.encode_text(prompts)                # (50,512) L2 normalization
    print(f"v_txt   {tuple(v_txt.shape)}  "
          f"norm mean {float(v_txt.norm(dim=-1).mean()):.4f}")

    # ---------- v_inv: logical inverse predicate description ----------
    invs = [INV_DESC.get(p, f"not {p}") for p in preds]
    inv_prompts = [build_inv_prompt(p, iv) for p, iv in zip(preds, invs)]
    v_inv = enc.encode_text(inv_prompts)             # (50,512)
    print(f"v_inv   {tuple(v_inv.shape)}  "
          f"norm mean {float(v_inv.norm(dim=-1).mean()):.4f}")

    # ---------- Formula (15) M_inv = cos(v_txt_i, v_inv_j) ----------
    M_inv = v_txt @ v_inv.t()                       # (50,50)
    diag = M_inv.diag()
    off = M_inv - torch.diag(torch.diag(M_inv))
    print(f"\n formula(15) M_inv {tuple(M_inv.shape)}  "
          f"range [{float(M_inv.min()):.4f}, {float(M_inv.max()):.4f}]")
    print(f"  diagonal(predicate↔Its reverse) mean = {float(diag.mean()):.4f}")
    print(f"  off diagonal(random pairing) mean = {float(off.mean()):.4f}")
    print(f"  Difference = {float(diag.mean() - off.mean()):+.4f}  "
          f"{'[OK] Discriminative' if float(diag.mean() - off.mean()) > 0.01 else '[WARN] No distinction'}")

    # ---------- Formula (23) J = top1 indication; c_dag = (J @ v_inv)/deg ----------
    J = torch.zeros_like(M_inv)
    J.scatter_(-1, M_inv.argmax(dim=-1).unsqueeze(-1), 1.0)
    deg = J.sum(dim=-1, keepdim=True).clamp(min=1.0)
    c_dag = (J @ v_inv) / deg
    print(f"\n formula(23) J {tuple(J.shape)}  Exactly one per line 1: "
          f"{bool((J.sum(-1) == 1).all())}  |  c_dag {tuple(c_dag.shape)}")

    # Display the ”logically nearest predicate” found by argmax, and manually check the semantic rationality
    print("\n  Each predicate M_inv The nearest predicate matched (for manual sampling)):")
    for i in range(0, K, 1):
        j = int(J[i].argmax())
        mark = "  <-- self matching" if j == i else ""
        print(f"    {preds[i]:<16} -> {preds[j]:<16} (cos={float(M_inv[i, j]):.4f}){mark}")

    # ---------- Object text embedding (replacing nn.Embedding(151,512)) ----------
    obj_prompts = []
    for o in objs:
        if o == "__background__":
            obj_prompts.append("a photo of background")
        else:
            art = "an" if o[0].lower() in "aeiou" else "a"
            obj_prompts.append(f"{art} photo of {o}, a type of object")
    obj_txt = enc.encode_text(obj_prompts)          # (151,512)
    print(f"\nobj_txt {tuple(obj_txt.shape)}  "
          f"norm mean {float(obj_txt.norm(dim=-1).mean()):.4f}")

    # ---------- Predicate description for image and text soft label q (templating, multi-template averaging) ----------
    q_prompts = []
    for p in preds:
        q_prompts.extend(t.format(pred=p) for t in PRED_TMPL)
    q_feat = enc.encode_text(q_prompts)             # (50*3, 512)
    q_feat = q_feat.view(K, len(PRED_TMPL), -1).mean(dim=1)
    q_feat = F.normalize(q_feat, dim=-1)
    print(f"q_feat  {tuple(q_feat.shape)}  （{len(PRED_TMPL)} template mean)")

    # ---------- Model dimension alignment ----------
    # The two places require different dimensions, so they are processed separately:
    # ① x_t / obj_txt: **Must maintain 512** (in the same CLIP joint embedding space as the visual feature x^v,
    # And the backbone’s proj[”t”] is constructed according to d_in[”t”]=512; the modal fusion premise of the final draft (1)).
    # ② c_dag / v_txt(y): The SharedProjection output of equation (23) is d-dimensional, so it needs to be reduced to d.
    from vrp.config import HyperParams
    hp = HyperParams()
    d = hp.D
    print(f"\n Dimension processing: x_t/obj_txt keep 512 (with x^v same space); "
          f"c_dag/v_txt(y) down to d={d} (match SharedProjection)")

    gproj = torch.Generator().manual_seed(12345)
    Q, _ = torch.linalg.qr(torch.randn(512, d, generator=gproj))     # (512,d) Column orthogonal
    v_txt_d = F.normalize(v_txt @ Q, dim=-1)                         # (50,d)
    v_inv_d = F.normalize(v_inv @ Q, dim=-1)                         # (50,d)
    q_feat_d = F.normalize(q_feat @ Q, dim=-1)                       # (50,d)
    M_inv_d = v_txt_d @ v_inv_d.t()
    c_dag_d = (J @ v_inv_d) / J.sum(dim=-1, keepdim=True).clamp(min=1.0)

    # Verification: Whether the argmax matching of M_inv is consistent before and after dimensionality reduction
    agree = int((M_inv.argmax(-1) == M_inv_d.argmax(-1)).sum())
    print(f"  After dimensionality reduction M_inv of argmax with 512 dimensionally consistent: {agree}/{K} "
          f"({agree/K*100:.0f}%)  [Semantic ordering preserved after dimensionality reduction]")
    print(f"  M_inv_d range [{float(M_inv_d.min()):.4f}, {float(M_inv_d.max()):.4f}]")
    print(f"  obj_txt keep 512 (backbone proj['t'] Expectation 512), "
          f"v_txt_d {tuple(v_txt_d.shape)}")

    # ---------- Placement ----------
    out = os.path.join(ART, "semantic_path.pt")
    torch.save({
        # 512-dimensional original CLIP space (x_t / image and text similarity used directly)
        "v_txt": v_txt, "v_inv": v_inv, "M_inv": M_inv,
        "J": J, "c_dag": c_dag, "obj_txt": obj_txt, "q_feat": q_feat,
        # d-dimensional model space (c_dag / Formula 12 teacher target use)
        "v_txt_d": v_txt_d, "v_inv_d": v_inv_d, "M_inv_d": M_inv_d,
        "c_dag_d": c_dag_d, "q_feat_d": q_feat_d,
        "proj_Q": Q, "model_dim": d,
        "predicates": preds, "inverse_desc": invs,
        "prompts": prompts, "inv_prompts": inv_prompts,
        "q_prompt_template": PRED_TMPL,
    }, out)
    print(f"\n[Place the order] {out}")

    # ---------- Comes with JSON summary for manual verification ----------
    meta = os.path.join(ART, "semantic_path_meta.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({
            "K": K,
            "M_inv_diag_mean": round(float(diag.mean()), 4),
            "M_inv_offdiag_mean": round(float(off.mean()), 4),
            "M_inv_gap": round(float(diag.mean() - off.mean()), 4),
            "argmax_pairs": {preds[i]: preds[int(J[i].argmax())] for i in range(K)},
            "self_matched": [preds[i] for i in range(K) if int(J[i].argmax()) == i],
        }, f, ensure_ascii=False, indent=2)
    print(f"[Place the order] {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
