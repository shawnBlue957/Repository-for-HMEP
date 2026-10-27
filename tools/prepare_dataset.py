#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实 VG-150 标注产出 M6.1 的离线产物，并跑通「真实数据 → 模型前向」链路。

产出（写入 code/artifacts/）：
  group_partition.pt   —— 式(16) 三组划分 + 可信度权重 w（数据驱动，非人工假定）
  epsilon_assoc.pt     —— 式(30) 的 51³ 组合律违反率表（纯数据属性，与训练无关）
  converse_stats.json  —— 逆关系共现统计（供论文报告）

用法：
  python tools/prepare_dataset.py [--split val] [--limit 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrp.data import VG150, converse_statistics, epsilon_assoc_from_images, split_converse_stats, to_model_batch

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print(f"加载 VG-150 [{args.split}]，limit={args.limit}")
    print("=" * 78)
    ds = VG150()
    print(f"  物体类 {ds.num_objects} | 谓词类 {ds.K}")
    images = ds.load_split(args.split, limit=args.limit)
    n_rel = sum(len(im.relations) for im in images)
    n_obj = sum(len(im.labels) for im in images)
    print(f"  图像 {len(images)} | 物体 {n_obj} | 关系 {n_rel}")
    print(f"  平均每图物体 {n_obj / max(len(images), 1):.1f} | 关系 {n_rel / max(len(images), 1):.1f}")

    # ---- 式(16) 三组划分（数据驱动）----
    print()
    print("=" * 78)
    print("式(16) 三组划分：由逆关系共现统计得出")
    print("=" * 78)
    counts, pair_counts = converse_statistics(images, K=ds.K)
    mask_non_sym, mask_sym, mask_ambig, w, meta = split_converse_stats(counts, pair_counts)
    n_ns, n_s, n_a = (float(mask_non_sym.sum()), float(mask_sym.sum()), float(mask_ambig.sum()))
    print(f"  非对称组 {int(n_ns):>2} | 对称组 {int(n_s):>2} | 混淆组 {int(n_a):>2}  (共 {ds.K})")
    print(f"  最小支持度过滤后参与判定 {meta['n_supported']}/{ds.K} 个谓词"
          f"（阈值 min_support={meta['min_support']}）")
    print(f"  自适应对称阈值 {meta['sym_threshold']:.4f} | 亲和度 max {meta['aff_max']:.4f} / mean {meta['aff_mean']:.5f}")
    print(f"  ⚠️ 双向标注率 {meta['reverse_rate'] * 100:.2f}%"
          f"（{meta['n_reverse_pairs']}/{meta['n_pairs_total']}）⟹ 逆共现统计在 VG 上极稀疏")
    print(f"  权重 w ∈ [{w.min():.3f}, {w.max():.3f}]")

    sym_names = [ds.predicate_classes[i + 1] for i in range(ds.K) if mask_sym[i] > 0]
    ns_names = [ds.predicate_classes[i + 1] for i in range(ds.K) if mask_non_sym[i] > 0][:10]
    print(f"  对称组谓词: {sym_names}")
    print(f"  非对称组谓词示例: {ns_names}")
    torch.save({
        "mask_non_sym": mask_non_sym, "mask_sym": mask_sym, "mask_ambig": mask_ambig,
        "w": w, "predicate_classes": ds.predicate_classes, "meta": meta,
    }, os.path.join(ART, "group_partition.pt"))

    # ---- 式(30) ε_assoc ----
    print()
    print("=" * 78)
    print("式(30) ε_assoc：组合律违反率（纯数据属性）")
    print("=" * 78)
    eps = epsilon_assoc_from_images(images, K_plus=ds.K + 1, max_len=args.limit)
    nz = eps[eps > 0]
    print(f"  表形状 {tuple(eps.shape)}（51³ = {51 ** 3:,}）")
    print(f"  非零条目 {nz.numel():,} | 非零处均值 {nz.mean():.4f} | 最大 {eps.max():.4f}")
    torch.save({"epsilon_assoc": eps, "predicate_classes": ds.predicate_classes},
               os.path.join(ART, "epsilon_assoc.pt"))

    # ---- 真实结构批 → 模型前向（连通性验证）----
    print()
    print("=" * 78)
    print("真实数据 → 模型前向（端到端连通性）")
    print("=" * 78)
    from vrp import HyperParams
    from _support import build_model_for_data

    hp = HyperParams()
    batch = to_model_batch(images[:8], K=ds.K, max_pairs=32, feature_dim=512)
    print(f"  批: x[v]={tuple(batch['x']['v'].shape)}  s_idx={tuple(batch['s_idx'].shape)}  "
          f"pair_mask 均值={float(batch['pair_mask'].mean()):.3f}")
    model = build_model_for_data(hp, ds, batch, ART)
    out = model(x=batch["x"], masks=batch["masks"], s_idx=batch["s_idx"],
                o_idx=batch["o_idx"], y_label=batch["y_label"])
    print(f"  total={float(out['total']):.4f}  cls={float(out['cls']):.4f}  "
          f"inv={float(out['inv']):.4f}  dag={float(out['dag_non_sym']):.6f}")
    print(f"  V_r 非零组 {int((out['_V_r'] > 0).sum())}/{ds.K}  V_global={float(out['_V_global']):.6f}")
    out["total"].backward()
    grad_ok = all(p.grad is not None for n, p in model.named_parameters()
                  if n in ("metric.A", "closure.theta_conv"))
    print(f"  反传: metric.A / closure.theta_conv 梯度存在 = {grad_ok}")

    stats = {
        "split": args.split, "limit": args.limit,
        "images": len(images), "objects": n_obj, "relations": n_rel,
        "groups": {"non_sym": int(n_ns), "sym": int(n_s), "ambig": int(n_a)},
        "eps_assoc_nonzero": int(nz.numel()),
        "eps_assoc_mean_nonzero": float(nz.mean()) if nz.numel() else 0.0,
        "forward_total": float(out["total"]),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(ART, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print()
    print(f"产物已写入 {ART}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
