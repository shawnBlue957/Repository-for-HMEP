"""构建 GQA 的 CLIP 语义张量 + 逆谓词映射。

与 `build_vrd_semantic.py` 同构，面向 GQA 的 **100 谓词 / 200 物体类**。

逆谓词表来源
------------
GQA 的逆关系不像 VRD 那样需要人工语义分组：`prepare_gqa.py` 已从数据
统计出 `cycles`（互逆谓词对，8 对）。本脚本直接读取该产物
`artifacts/gqa_train_inv_pairs.json` 的 `cycles` 字段，**不做人工判定**，
故 MIR@K 的覆盖范围完全可复核。

产物：`artifacts/gqa_semantic.pt`
    obj_txt (201,512) / v_txt / v_inv / M_inv / J / c_dag / q_feat
    v_*_d / M_inv_d / c_dag_d / inv_idx (K+1,) / predicates 等
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
    """从 prepare_gqa 的产物读 cycles（互逆对），返回 [[a],[b]] 形式的组对。

    只保留两端都在谓词表内的对；缺一端则无法评测，直接丢弃并报告。
    """
    p = os.path.join(ds.artifacts, "gqa_train_inv_pairs.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少逆谓词统计：{p}（先运行 prepare_gqa.py）")
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
    print(f"[inv] 读到 cycles {len(d.get('cycles', []))} 对 → 有效 {len(groups)} 对")
    if dropped:
        print(f"[inv] 丢弃 {len(dropped)} 对（不在谓词表内）：{dropped[:4]}")
    return groups


def build_inv_idx(pred_names_1b, inv_groups):
    """谓词名列表（含 background，索引即 1-based 谓词号）→ 逆谓词索引数组。

    ⚠️ **数组位置与数组值都是 1-based 谓词号**（无逆为 -1），
       值为 **0-based 下标**；读取端统一 `rp = int(inv_idx[p+1]) - 1`。
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
    print(f"GQA 语义张量构建 | 谓词 {K} | 物体类 {ds.num_objects}")
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
    print(f"式(15) M_inv {tuple(M_inv.shape)} 值域 "
          f"[{float(M_inv.min()):.4f}, {float(M_inv.max()):.4f}]")
    print(f"  对角(谓词↔其逆) {float(diag.mean()):.4f} | "
          f"非对角(随机配对) {float(off.mean()):.4f} | "
          f"差 {float(diag.mean()-off.mean()):+.4f}")

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
    print(f"obj_txt {tuple(obj_txt.shape)} 范数均值 "
          f"{float(obj_txt.norm(dim=-1).mean()):.4f}")

    # ---------- q_feat ----------
    qp = []
    for p in preds:
        qp.extend(t.format(pred=p) for t in PRED_TMPL)
    q_feat = enc.encode_text(qp).view(K, len(PRED_TMPL), -1).mean(dim=1)
    q_feat = F.normalize(q_feat, dim=-1)
    print(f"q_feat {tuple(q_feat.shape)}（{len(PRED_TMPL)} 模板均值）")

    # ---------- 降维到模型 d ----------
    from vrp.config import HyperParams
    d = HyperParams().D
    g = torch.Generator().manual_seed(20261017)
    Q, _ = torch.linalg.qr(torch.randn(512, d, generator=g))
    v_txt_d = F.normalize(v_txt @ Q, dim=-1)
    v_inv_d = F.normalize(v_inv @ Q, dim=-1)
    M_inv_d = v_txt_d @ v_inv_d.t()
    c_dag_d = (J @ v_inv_d) / deg
    agree = int((M_inv.argmax(-1) == M_inv_d.argmax(-1)).sum())
    print(f"降维 512→{d}：M_inv argmax 一致 {agree}/{K} ({100.0*agree/K:.0f}%)")

    # ---------- 逆谓词表 + 自检 ----------
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
    print(f"\n[自检] 自指 {n_self} | 值域越界 {n_oob} | 组首非对称 {n_asym} | "
          f"链不闭合 {n_chain}")
    if n_self or n_oob or n_asym or n_chain:
        raise AssertionError(
            f"逆谓词表不自洽：自指 {n_self}、越界 {n_oob}、"
            f"非对称 {n_asym}、链不闭合 {n_chain}")

    covered = [i for i in range(1, K + 1) if inv_idx[i] >= 0]
    print(f"\n逆谓词表：{len(covered)}/{K} 个谓词有可判定的逆")
    for i in covered:
        print(f"   {ds.predicate_classes[i]:<20} → "
              f"{ds.predicate_classes[inv_idx[i] + 1]}")

    # 覆盖率：用 val 统计（train 太大且与训练同分布）
    nrel = tot = 0
    for r in ds.load_split("val"):
        for rel in r.relations:
            nrel += 1
            if inv_idx[rel.predicate] >= 0:
                tot += 1
    print(f"\nval 关系 {nrel:,} 条，其中 {tot:,} 条落在可评测谓词上 "
          f"= {100.0*tot/max(nrel,1):.2f}%")
    print("备注：GQA 的 'to the left of'/'to the right of' 占 89.3%，"
          "主指标应用 mR@K，并建议补「排除左右」的对照。")
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
    print(f"[落盘] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
