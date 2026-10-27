#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The M6.1 offline product is generated from the real VG-150 annotation and runs through the ”real data → model forward” link.

Output (written to code/artifacts/):
  group_partition.pt —— Formula (16) three group partition + credibility weight w (data-driven, non-artificial assumption)
  epsilon_assoc.pt - 51³ combination law violation rate table of equation (30) (pure data attribute, irrelevant to training)
  converse_stats.json - Inverse relationship co-occurrence statistics (for paper report)

Usage:
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
    print(f"Load VG-150 [{args.split}]，limit={args.limit}")
    print("=" * 78)
    ds = VG150()
    print(f"  object class {ds.num_objects} | predicate class {ds.K}")
    images = ds.load_split(args.split, limit=args.limit)
    n_rel = sum(len(im.relations) for im in images)
    n_obj = sum(len(im.labels) for im in images)
    print(f"  image {len(images)} | object {n_obj} | relationship {n_rel}")
    print(f"  average object per image {n_obj / max(len(images), 1):.1f} | relationship {n_rel / max(len(images), 1):.1f}")

    # ---- Formula (16) Three group division (data driven) ----
    print()
    print("=" * 78)
    print("formula(16) Three groups of divisions: derived from inverse relationship co-occurrence statistics")
    print("=" * 78)
    counts, pair_counts = converse_statistics(images, K=ds.K)
    mask_non_sym, mask_sym, mask_ambig, w, meta = split_converse_stats(counts, pair_counts)
    n_ns, n_s, n_a = (float(mask_non_sym.sum()), float(mask_sym.sum()), float(mask_ambig.sum()))
    print(f"  asymmetric group {int(n_ns):>2} | Symmetry group {int(n_s):>2} | Confused group {int(n_a):>2}  (total {ds.K})")
    print(f"  Participate in judgment after minimum support filtering {meta['n_supported']}/{ds.K} predicate"
          f" (threshold min_support={meta['min_support']}）")
    print(f"  adaptive symmetry threshold {meta['sym_threshold']:.4f} | Affinity max {meta['aff_max']:.4f} / mean {meta['aff_mean']:.5f}")
    print(f"  ⚠️ Bidirectional labeling rate {meta['reverse_rate'] * 100:.2f}%"
          f"（{meta['n_reverse_pairs']}/{meta['n_pairs_total']})⟹ Inverse co-occurrence statistics in VG sparse upper pole")
    print(f"  weight w ∈ [{w.min():.3f}, {w.max():.3f}]")

    sym_names = [ds.predicate_classes[i + 1] for i in range(ds.K) if mask_sym[i] > 0]
    ns_names = [ds.predicate_classes[i + 1] for i in range(ds.K) if mask_non_sym[i] > 0][:10]
    print(f"  Symmetric group predicate: {sym_names}")
    print(f"  Asymmetric group predicate example: {ns_names}")
    torch.save({
        "mask_non_sym": mask_non_sym, "mask_sym": mask_sym, "mask_ambig": mask_ambig,
        "w": w, "predicate_classes": ds.predicate_classes, "meta": meta,
    }, os.path.join(ART, "group_partition.pt"))

    # ---- Formula (30) ε_assoc ----
    print()
    print("=" * 78)
    print("formula(30) ε_assoc: Combinatorial law violation rate (pure data attribute)")
    print("=" * 78)
    eps = epsilon_assoc_from_images(images, K_plus=ds.K + 1, max_len=args.limit)
    nz = eps[eps > 0]
    print(f"  table shape {tuple(eps.shape)}（51³ = {51 ** 3:,}）")
    print(f"  non-zero entries {nz.numel():,} | non-zero mean {nz.mean():.4f} | maximum {eps.max():.4f}")
    torch.save({"epsilon_assoc": eps, "predicate_classes": ds.predicate_classes},
               os.path.join(ART, "epsilon_assoc.pt"))

    # ---- Real structure batch → Model forward (connectivity verification) ----
    print()
    print("=" * 78)
    print("real data → Model forward (end-to-end connectivity)")
    print("=" * 78)
    from vrp import HyperParams
    from _support import build_model_for_data

    hp = HyperParams()
    batch = to_model_batch(images[:8], K=ds.K, max_pairs=32, feature_dim=512)
    print(f"  batch: x[v]={tuple(batch['x']['v'].shape)}  s_idx={tuple(batch['s_idx'].shape)}  "
          f"pair_mask mean={float(batch['pair_mask'].mean()):.3f}")
    model = build_model_for_data(hp, ds, batch, ART)
    out = model(x=batch["x"], masks=batch["masks"], s_idx=batch["s_idx"],
                o_idx=batch["o_idx"], y_label=batch["y_label"])
    print(f"  total={float(out['total']):.4f}  cls={float(out['cls']):.4f}  "
          f"inv={float(out['inv']):.4f}  dag={float(out['dag_non_sym']):.6f}")
    print(f"  V_r non-zero group {int((out['_V_r'] > 0).sum())}/{ds.K}  V_global={float(out['_V_global']):.6f}")
    out["total"].backward()
    grad_ok = all(p.grad is not None for n, p in model.named_parameters()
                  if n in ("metric.A", "closure.theta_conv"))
    print(f"  Backpass: metric.A / closure.theta_conv Gradient exists = {grad_ok}")

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
    print(f"Product has been written {ART}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
