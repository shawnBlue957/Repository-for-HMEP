"""Real data loading: VG-150/VRD (corresponding to module design document M6.1 offline precalculation).

Data set location (it already exists on this machine, do not download it again):
    <PROJECT_ROOT>

VG-150 actual measured scale (check_report.json all PASS):
    Images 108,073 | Object classes 150 | Predicate classes 50 | Relations 622,705
    Divide train 57,723 / val 5,000 / test 26,446
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

DEFAULT_ROOT = (
    r"/path/to/vg_data"
)


def dataset_root() -> str:
    """Data root directory: Prioritize the environment variable VG_DATA_ROOT, otherwise use the existing path of the local machine."""
    return os.environ.get("VG_DATA_ROOT", DEFAULT_ROOT)


@dataclass
class RelationRecord:
    subject: int
    object: int
    predicate: int          # 1..50 (0 is __background__)


@dataclass
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int
    labels: List[int]                    # 1..150
    boxes: List[List[float]]             # xyxy
    relations: List[RelationRecord]


class VG150:
    """VG-150 Read-only loader for standard SGG partitions."""

    def __init__(self, root: Optional[str] = None):
        self.root = os.path.join(root or dataset_root(), "vg150")
        dicts_path = os.path.join(self.root, "VG-SGG-dicts.json")
        with open(dicts_path, encoding="utf-8") as f:
            d = json.load(f)
        self.object_classes: List[str] = d["object_classes"]        # 151 (including background)
        self.predicate_classes: List[str] = d["predicate_classes"]  # 51 (including background)
        self.K = len(self.predicate_classes) - 1                    # 50
        self.num_objects = len(self.object_classes) - 1             # 150

    def load_split(self, split: str = "val", limit: Optional[int] = None) -> List[ImageRecord]:
        path = os.path.join(self.root, f"annotations_{split}.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        out: List[ImageRecord] = []
        for item in raw:
            rels = [RelationRecord(int(r["subject"]), int(r["object"]), int(r["predicate"]))
                    for r in item.get("relations", [])]
            objs = item.get("objects", [])
            out.append(ImageRecord(
                image_id=str(item["image_id"]),
                file_name=str(item.get("file_name", "")),
                width=int(item.get("width", 0)),
                height=int(item.get("height", 0)),
                labels=[int(o["label"]) for o in objs],
                boxes=[list(map(float, o["bbox"])) for o in objs],
                relations=rels,
            ))
        return out

    def stats(self) -> Dict[str, int]:
        with open(os.path.join(self.root, "stats.json"), encoding="utf-8") as f:
            return json.load(f)


# --------------------------------------------------------------- Structure → Batch Tensor

def to_model_batch(
    images: Sequence[ImageRecord],
    *,
    K: int = 50,
    max_pairs: int = 32,
    feature_dim: int = 512,
    visual_features: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Convert label records into structured batches required by the model (pair-level s_idx/o_idx/y_label).

    ⚠️ This function **only provides graph structure** (subject and object index + predicate label + layer neighborhood mask).
    The visual/text modal feature x needs to be provided by the caller (CLIP encoding), otherwise it will be occupied by a fixed seed.
    Only used for **contract and consistency verification**, not for training.
    """
    import torch.nn.functional as F

    b = len(images)
    N = max(1, min(80, max((len(im.labels) for im in images), default=1)))
    P = max_pairs
    g = torch.Generator().manual_seed(seed)

    s_idx = torch.zeros(b, P, dtype=torch.long)
    o_idx = torch.zeros(b, P, dtype=torch.long)
    y_label = torch.zeros(b, P, dtype=torch.long)
    pair_mask = torch.zeros(b, P)
    masks = torch.zeros(b, 3, N)
    Ns = torch.zeros(b, dtype=torch.long)

    for i, im in enumerate(images):
        n = min(len(im.labels), N)
        Ns[i] = n
        masks[i, 0, :n] = 1.0
        masks[i, 1, :n] = 1.0
        masks[i, 2, :n] = 1.0
        rels = [r for r in im.relations if r.subject < n and r.object < n][:P]
        for j, r in enumerate(rels):
            s_idx[i, j] = r.subject
            o_idx[i, j] = r.object
            y_label[i, j] = max(0, min(K - 1, r.predicate - 1))
            pair_mask[i, j] = 1.0

    if visual_features is not None:
        x = {m: visual_features[m] for m in ("v", "t", "s")}
    else:
        x = {m: torch.randn(b, N, feature_dim, generator=g) for m in ("v", "t", "s")}

    return {"x": x, "masks": masks, "s_idx": s_idx, "o_idx": o_idx,
            "y_label": y_label, "pair_mask": pair_mask, "N": Ns}


# --------------------------------------------------------------- M2.5 / Formula (16)

def converse_statistics(
    images: Sequence[ImageRecord], K: int = 50, K_plus: int = 51
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ”inverse relationship” co-occurrences are counted from the annotation and used for the three group divisions of Equation (16).

    The predicate index is 1..K (0 is __background__), and is remapped to 0..K−1** when returned,
    Therefore, the output shapes are (K, K) and (K,), which are consistent with the K of the model.

    ⟹ This is a **data-driven** grouping basis (final draft §3.3: Grouping must be derived from VG annotation statistics,
       is not presupposed – this is a publishable empirical contribution in itself).
    """
    counts = torch.zeros(K, K, dtype=torch.float64)
    pair_counts = torch.zeros(K, dtype=torch.float64)
    n_rev = 0
    for im in images:
        direct = {}
        for r in im.relations:
            direct[(r.subject, r.object)] = r.predicate
        for (s, o), p in direct.items():
            if not (1 <= p <= K):
                continue
            pi = p - 1
            pair_counts[pi] += 1
            q = direct.get((o, s))
            if q is not None and 1 <= q <= K:
                counts[pi, q - 1] += 1
                n_rev += 1
    converse_statistics.n_reverse_pairs = n_rev          # for reporting
    return counts.float(), pair_counts.float()


def epsilon_assoc_from_images(
    images: Sequence[ImageRecord], K_plus: int = 51, max_len: Optional[int] = None
) -> torch.Tensor:
    """Construct a 51³ lookup table of ε_assoc from the labeled triples (Eq. 30).

    For each picture, enumerate the relationship pair (a→b, b→c). If (a→c) also exists, record a triplet.
    (r_ab, r_bc, r_ac). Returns the violation rate table for (K_plus,K_plus,K_plus).

    ⚠️ The final draft clearly states: ε_assoc has nothing to do with training, it is the combinatorial law violation rate of the data set (pure data attribute);
    If ε_assoc is not related to mR@50, then the return B is not true (falsifiable).
    """
    from vrp.diagnostics import build_epsilon_assoc

    triples: List[List[int]] = []
    for idx, im in enumerate(images):
        if max_len is not None and idx >= max_len:
            break
        direct = {}
        for r in im.relations:
            if r.predicate < K_plus:
                direct[(r.subject, r.object)] = r.predicate
        # Explicit two hops: (a,b) and (b,c) exist at the same time ⟹ Remember triplet (r_ab, r_bc, r_ac)
        for (a, b), rab in direct.items():
            for (b2, c), rbc in direct.items():
                if b2 != b or a == c:
                    continue
                rac = direct.get((a, c))
                if rac is not None:
                    triples.append([rab, rbc, rac])
    if not triples:
        return torch.zeros(K_plus, K_plus, K_plus)
    return build_epsilon_assoc(torch.tensor(triples, dtype=torch.long), K_plus)


def split_converse_stats(
    counts: torch.Tensor,
    pair_counts: torch.Tensor,
    *,
    min_support: int = 30,
    sym_ratio: float = 0.20,
    ambig_ratio: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Equation (16): ρ̂ and three group divisions + credibility weight are obtained from the inverse relationship co-occurrence statistics.

    ⚠️ **Actual measurement found (this project was verified on VG-150 val)**: The label** of VG-150 rarely gives the same
    Inverse relationships of ordered pairs** - only 492 out of 7,494 relationships in 1,500 images have inverse pairs
    (6.6%). Therefore, if ρ̂ is defined according to ”two-way co-occurrence of the same pair”, the statistics will be extremely sparse.
    (Max affinity 0.286, mean 0.002), low support predicates will be dominated by small sample noise.

    ⟹ The data in this letter undergoes two engineering processes (both are **implementation conventions** and must be explained in the paper):
       ① **Minimum support filter**: Predicates with pair_counts < min_support do not participate in the determination;
       ② **Adaptive threshold**: The threshold is based on the ”affinity quantile with supported predicates” instead of a fixed 0.5.

    ⟹ A more fundamental alternative (consistent with the original version of the final equation 16): ρ̂ should have been predicted by the **model**
       TopK hit rate statistics (CVPRW2022 practice) instead of annotating co-occurrences. Annotations co-occur only in the above table
       It is only available after sparsity has been explicitly handled - this point is recommended to be written in the ”Grouping Method” section of the paper.

    Return (mask_non_sym, mask_sym, mask_ambig, w, meta).
    """
    K = counts.shape[0]
    rho = counts / pair_counts.clamp(min=1.0).unsqueeze(0)
    aff = torch.maximum(rho, rho.t())
    support = pair_counts >= min_support

    # Adaptive threshold: based on the (diagonal/non-diagonal) affinity quantile with supporting predicates
    diag = torch.diagonal(aff).clone()
    off = aff.clone()
    off.fill_diagonal_(0.0)
    sup_vals = diag[support]
    sym_thr = float(sup_vals.quantile(sym_ratio)) if sup_vals.numel() else 0.0
    sym_thr = max(sym_thr, float(sup_vals.max()) if sup_vals.numel() else 0.0) * sym_ratio
    amb_thr = sym_thr * (ambig_ratio / sym_ratio)

    best = aff.argmax(dim=-1)
    self_loop = torch.arange(K)
    peak = aff.max(dim=-1).values

    is_self = (best == self_loop)
    mask_sym = is_self & support & (peak >= sym_thr)
    mask_non_sym = (~is_self) & support & (peak >= sym_thr)
    mask_ambig = ~(mask_sym | mask_non_sym)

    w = peak / (1.0 + mask_ambig.float())
    w = w.clamp(0.0, 1.0)

    meta = {
        "min_support": min_support,
        "n_supported": int(support.sum()),
        "sym_threshold": sym_thr,
        "ambig_threshold": amb_thr,
        "aff_max": float(aff.max()),
        "aff_mean": float(aff.mean()),
        "n_reverse_pairs": int(counts.sum()),
        "n_pairs_total": int(pair_counts.sum()),
        "reverse_rate": float(counts.sum() / pair_counts.sum().clamp(min=1.0)),
    }
    return mask_non_sym, mask_sym, mask_ambig, w, meta
