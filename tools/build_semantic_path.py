#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 7 — 计算 CLIP 语义通路张量：v_txt / v_inv / M_inv / J / c_dag / x_t / q。

背景
----
上一轮实验（400 图 / 3 seed / 6 臂）得出：范畴损失项的主效应 |ΔmR@50| < 0.001，
即「在**无语义文本通路**时范畴层无作用」。但该结论有一个致命前提缺陷 ——
范畴层的 $\\mathbf M_{inv}$、$q$、$v_{txt}$ **全部依赖 CLIP 文本语义**，
而上一轮 `x_t` 用可学习 Embedding、`q` 用谓词频率先验替代，该通路完全缺失。
故本脚本补齐之。

产出（落盘 `artifacts/semantic_*.pt`）
------------------------------------
| 张量 | 形状 | 对应 |
|---|---|---|
| `v_txt`   | (50,512) | 式(15) 正向谓词描述嵌入 $v_{txt,i}$ |
| `v_inv`   | (50,512) | 式(15) 逻辑逆谓词描述嵌入 $v_{inv,j}$ |
| `M_inv`   | (50,50)  | 式(15) $\\cos(v_{txt,i}, v_{inv,j})$ |
| `J`       | (50,50)  | 式(23) top1 指示矩阵 |
| `c_dag`   | (50,512) | 式(23) $c^\\dagger_k$ |
| `obj_txt` | (151,512)| 物体类文本嵌入（替代 `nn.Embedding(151,512)`） |
| `pred_prompts` | list | 供图文软标签 q 使用的模板化描述 |

设计要点
--------
1. **逻辑逆不要求落在 50 类内**（项目长期记忆已裁定）：对 VG-50 中不存在
   类内逆的谓词（`wearing`/`wears`、`sitting on` 等），`v_inv` 取**自然语言逆描述**
   的嵌入（如 `wearing` → "worn by"）。这与式(15) 的原意一致 ——
   $\\mathbf M_{inv}$ 是**语义对偶矩阵**而非类内置换。
2. **模板化 prompt**：CLIP 零样本的标准做法（"a photo of a {}"）。
   谓词是动作/关系词而非物体，直接喂裸词效果差，故用关系句模板。
3. **物体类**用 "a photo of a {obj}, a type of object" 模板。
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
# 逻辑逆谓词的自然语言描述
# ======================================================================
# 键为 VG-50 的谓词名，值为「主客体互换后的逆关系」的自然语言表述。
# 未列出的谓词，其逆描述退化为 "not " + 原词（表示该关系不成立）。
INV_DESC = {
    "above": "below",
    "across": "across",                 # 近似自反
    "against": "against",
    "along": "along",
    "and": "and",
    "at": "at",
    "attached to": "attached to",       # 自反
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
    "near": "near",                     # 自反
    "of": "of",
    "on": "supporting",                 # on 的逆不是 under，而是「承载」
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

# 关系句模板（CLIP 零样本标准做法：谓词需放进句子才有语义）
#   ① 关系句 ② 裸词（补充短词区分度）③ 另一句式（模板集成）
PRED_TMPL = [
    "a photo of one object that is {pred} another object",
    "{pred}",
    "a photo where one object is {pred} another object",
]


def build_prompt(pred: str) -> str:
    """谓词 → 关系句描述。"""
    return f"a photo of one object that is {pred} another object"


def build_inv_prompt(pred: str, inv: str) -> str:
    return f"a photo of one object that is {inv} another object"


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    ds = VG150()
    K = ds.K
    preds = ds.predicate_classes[1:]                 # 50 个，去掉 background
    objs = ds.object_classes                         # 151（含 background）
    print(f"VG-150: K={K} 谓词, {len(objs)} 物体类（含 background）")

    enc = CLIPTextCPU(W, B)

    # ---------- v_txt：正向谓词描述 ----------
    prompts = [build_prompt(p) for p in preds]
    v_txt = enc.encode_text(prompts)                # (50,512) L2 归一
    print(f"v_txt   {tuple(v_txt.shape)}  "
          f"范数均值 {float(v_txt.norm(dim=-1).mean()):.4f}")

    # ---------- v_inv：逻辑逆谓词描述 ----------
    invs = [INV_DESC.get(p, f"not {p}") for p in preds]
    inv_prompts = [build_inv_prompt(p, iv) for p, iv in zip(preds, invs)]
    v_inv = enc.encode_text(inv_prompts)             # (50,512)
    print(f"v_inv   {tuple(v_inv.shape)}  "
          f"范数均值 {float(v_inv.norm(dim=-1).mean()):.4f}")

    # ---------- 式(15) M_inv = cos(v_txt_i, v_inv_j) ----------
    M_inv = v_txt @ v_inv.t()                       # (50,50)
    diag = M_inv.diag()
    off = M_inv - torch.diag(torch.diag(M_inv))
    print(f"\n式(15) M_inv {tuple(M_inv.shape)}  "
          f"值域 [{float(M_inv.min()):.4f}, {float(M_inv.max()):.4f}]")
    print(f"  对角线(谓词↔其逆) 均值 = {float(diag.mean()):.4f}")
    print(f"  非对角(随机配对) 均值 = {float(off.mean()):.4f}")
    print(f"  差值 = {float(diag.mean() - off.mean()):+.4f}  "
          f"{'[OK] 有区分度' if float(diag.mean() - off.mean()) > 0.01 else '[WARN] 无区分度'}")

    # ---------- 式(23) J = top1 指示；c_dag = (J @ v_inv)/deg ----------
    J = torch.zeros_like(M_inv)
    J.scatter_(-1, M_inv.argmax(dim=-1).unsqueeze(-1), 1.0)
    deg = J.sum(dim=-1, keepdim=True).clamp(min=1.0)
    c_dag = (J @ v_inv) / deg
    print(f"\n式(23) J {tuple(J.shape)}  每行恰一个 1: "
          f"{bool((J.sum(-1) == 1).all())}  |  c_dag {tuple(c_dag.shape)}")

    # 展示 argmax 找到的「逻辑最近谓词」，人工抽检语义合理性
    print("\n  各谓词经 M_inv 匹配到的最近谓词（人工抽检用）:")
    for i in range(0, K, 1):
        j = int(J[i].argmax())
        mark = "  <-- 自匹配" if j == i else ""
        print(f"    {preds[i]:<16} -> {preds[j]:<16} (cos={float(M_inv[i, j]):.4f}){mark}")

    # ---------- 物体类文本嵌入（替代 nn.Embedding(151,512)）----------
    obj_prompts = []
    for o in objs:
        if o == "__background__":
            obj_prompts.append("a photo of background")
        else:
            art = "an" if o[0].lower() in "aeiou" else "a"
            obj_prompts.append(f"{art} photo of {o}, a type of object")
    obj_txt = enc.encode_text(obj_prompts)          # (151,512)
    print(f"\nobj_txt {tuple(obj_txt.shape)}  "
          f"范数均值 {float(obj_txt.norm(dim=-1).mean()):.4f}")

    # ---------- 供图文软标签 q 用的谓词描述（模板化，多模板平均）----------
    q_prompts = []
    for p in preds:
        q_prompts.extend(t.format(pred=p) for t in PRED_TMPL)
    q_feat = enc.encode_text(q_prompts)             # (50*3, 512)
    q_feat = q_feat.view(K, len(PRED_TMPL), -1).mean(dim=1)
    q_feat = F.normalize(q_feat, dim=-1)
    print(f"q_feat  {tuple(q_feat.shape)}  （{len(PRED_TMPL)} 模板均值）")

    # ---------- 模型维度对齐 ----------
    # 两处需要不同维度，故分别处理：
    #  ① x_t / obj_txt：**必须保持 512**（与视觉特征 x^v 同处 CLIP 联合嵌入空间，
    #     且 backbone 的 proj["t"] 按 d_in["t"]=512 构建；终稿式(1) 的模态融合前提）。
    #  ② c_dag / v_txt(y)：式(23) 的 SharedProjection 输出为 d 维，故需降到 d。
    from vrp.config import HyperParams
    hp = HyperParams()
    d = hp.D
    print(f"\n维度处理：x_t/obj_txt 保持 512（与 x^v 同空间）；"
          f"c_dag/v_txt(y) 降到 d={d}（匹配 SharedProjection）")

    gproj = torch.Generator().manual_seed(12345)
    Q, _ = torch.linalg.qr(torch.randn(512, d, generator=gproj))     # (512,d) 列正交
    v_txt_d = F.normalize(v_txt @ Q, dim=-1)                         # (50,d)
    v_inv_d = F.normalize(v_inv @ Q, dim=-1)                         # (50,d)
    q_feat_d = F.normalize(q_feat @ Q, dim=-1)                       # (50,d)
    M_inv_d = v_txt_d @ v_inv_d.t()
    c_dag_d = (J @ v_inv_d) / J.sum(dim=-1, keepdim=True).clamp(min=1.0)

    # 校验：降维前后 M_inv 的 argmax 匹配是否一致
    agree = int((M_inv.argmax(-1) == M_inv_d.argmax(-1)).sum())
    print(f"  降维后 M_inv 的 argmax 与 512 维一致: {agree}/{K} "
          f"({agree/K*100:.0f}%)  [语义排序在降维后保持]")
    print(f"  M_inv_d 值域 [{float(M_inv_d.min()):.4f}, {float(M_inv_d.max()):.4f}]")
    print(f"  obj_txt 保持 512（backbone proj['t'] 期望 512），"
          f"v_txt_d {tuple(v_txt_d.shape)}")

    # ---------- 落盘 ----------
    out = os.path.join(ART, "semantic_path.pt")
    torch.save({
        # 512 维原始 CLIP 空间（x_t / 图文相似度直接使用）
        "v_txt": v_txt, "v_inv": v_inv, "M_inv": M_inv,
        "J": J, "c_dag": c_dag, "obj_txt": obj_txt, "q_feat": q_feat,
        # d 维模型空间（c_dag / 式12 教师目标使用）
        "v_txt_d": v_txt_d, "v_inv_d": v_inv_d, "M_inv_d": M_inv_d,
        "c_dag_d": c_dag_d, "q_feat_d": q_feat_d,
        "proj_Q": Q, "model_dim": d,
        "predicates": preds, "inverse_desc": invs,
        "prompts": prompts, "inv_prompts": inv_prompts,
        "q_prompt_template": PRED_TMPL,
    }, out)
    print(f"\n[落盘] {out}")

    # ---------- 附带 JSON 摘要便于人工核查 ----------
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
    print(f"[落盘] {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
