"""UnRel dataset loader (interface aligned VRD/VG150/GQA).

## Why do we need a separate loader?

UnRel (Peyre et al., ICCV 2017) is a **pure evaluation set**: `annotations.mat` is 1071×1
Evaluation annotation, **no train/test split**, average 1.4 relationships/graph ⟹ not enough for training.

Therefore, this article performs **zero sample migration across data sets** according to the official standard usage:
    Trained on VRD (70 predicates) ⟹ Evaluating unusual relationships on UnRel.

## Key: The index space must be mapped to the source domain (VRD)

Zero sample requires ”the same model weight can directly eat the target domain data”, therefore:
- **Predicate**: All 18 predicate names of UnRel ⊆ 70 predicate names of VRD,
  Therefore, the predicate is mapped to the 1-based predicate number of **VRD** (`K = 70`, consistent with the output dimension of the source domain model).
- **Object class**: also mapped to **VRD’s object vocabulary**; classes in UnRel that are not in the VRD vocabulary → 0 (background).
  This is the **honesty price** of zero samples (unknown objects in the target domain cannot be represented by source domain embeddings) and must be stated in the paper.

When `pred_space=”own”` / `obj_space=”own”` is used, it degenerates into UnRel’s own vocabulary (only for diagnosis or self-training and self-testing).

## Product format

`~/vrp/data/unrel/artifacts/unrel_annotations.json`
    {"predicates": [...18], "triplets": [{image_id, subject_cat, object_cat,
     predicate, s_box, o_box}, ...]}
Image: `~/vrp/data/unrel/images/{image_id}.jpg`
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _root() -> str:
    v = os.environ.get("UNREL_ROOT")
    if v:
        return v
    base = os.environ.get("VG_DATA_ROOT")
    if base:
        return os.path.join(base, "unrel")
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(os.path.dirname(here))
    return os.path.join(proj, "data", "unrel")


@dataclass
class URelation:
    subject: int          #Local id of the object in the picture (0-based)
    object: int
    predicate: int        # 1-based predicate number (default in VRD space)


@dataclass
class URecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)     # 0=background
    relations: List[URelation] = field(default_factory=list)


class UnRel:
    """UnRel read-only loader (default mapped to VRD index space, for zero-sample evaluation)."""

    def __init__(self, root: Optional[str] = None, top_objects: int = 200,
                 pred_space: str = "vrd", obj_space: str = "vrd"):
        self.root = root or _root()
        self.artifacts = os.path.join(self.root, "artifacts")
        self.top_objects = top_objects
        self.pred_space = pred_space
        self.obj_space = obj_space
        self._cache: Dict[str, List[URecord]] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        ap = os.path.join(self.artifacts, "unrel_annotations.json")
        if not os.path.exists(ap):
            raise FileNotFoundError(
                f"missing UnRel mark {ap}\n  First run: python tools/prepare_unrel.py "
                f"--root <unrel Unzip the root> --out artifacts/unrel")
        with open(ap, encoding="utf-8") as f:
            d = json.load(f)
        self.predicates_raw: List[str] = list(d["predicates"])
        self.triplets: List[dict] = d["triplets"]

        if self.pred_space == "vrd":
            from .vrd_data import VRD
            vrd = VRD()
            self.predicate_classes = list(vrd.predicate_classes)   # with background
            self.K = vrd.K                                          # 70
            self._p2i = {p: i + 1 for i, p in enumerate(self.predicate_classes[1:])}
            self.object_classes = list(vrd.object_classes)
            self.num_objects = vrd.num_objects
            self._o2i = {n: i + 1 for i, n in enumerate(self.object_classes[1:])}
            self.source = "vrd"
        else:
            self.predicate_classes = ["__background__"] + self.predicates_raw
            self.K = len(self.predicates_raw)
            self._p2i = {p: i + 1 for i, p in enumerate(self.predicates_raw)}
            if self.obj_space == "vrd":
                from .vrd_data import VRD
                vrd = VRD()
                self.object_classes = list(vrd.object_classes)
                self.num_objects = vrd.num_objects
                self._o2i = {n: i + 1 for i, n in enumerate(self.object_classes[1:])}
            else:
                cnt = Counter()
                for t in self.triplets:
                    cnt[t["subject_cat"]] += 1
                    cnt[t["object_cat"]] += 1
                top = [n for n, _ in cnt.most_common(self.top_objects)]
                self.object_classes = ["__background__"] + top
                self.num_objects = len(top)
                self._o2i = {n: i + 1 for i, n in enumerate(top)}
            self.source = "unrel"

        # Predicate coverage self-test: 18/18 should hit in VRD space
        miss = [p for p in self.predicates_raw if p not in self._p2i]
        if miss:
            raise ValueError(f"predicate cannot be mapped to {self.pred_space} Glossary:{miss}")

    # ------------------------------------------------------------------
    def image_dir(self, split: str = "test") -> Optional[str]:
        d = os.path.join(self.root, "images")
        return d if os.path.isdir(d) else None

    def pred_indices(self) -> List[int]:
        """UnRel predicate’s 1-based predicate number in target space (in ascending order)."""
        return sorted(self._p2i[p] for p in self.predicates_raw)

    def load_split(self, split: str = "test", limit: Optional[int] = None) -> List[URecord]:
        if split.startswith("train"):
            raise ValueError(
                "UnRel No training partition (annotations.mat for 1071 Figure evaluation annotation, approx. 1.4 relationship/Figure).\n"
                "  Please use cross-dataset zero-sample protocol: VRD training → UnRel Review\n"
                "  （tools/zeroshot_vrd_to_unrel.py）。")
        key = f"{split}:{limit}"
        if key in self._cache:
            return self._cache[key]

        by_img: Dict[str, list] = defaultdict(list)
        for t in self.triplets:
            by_img[str(t["image_id"])].append(t)

        recs: List[URecord] = []
        n_unmapped_obj = 0
        for image_id, ts in by_img.items():
            rec = URecord(image_id=image_id)
            seen: Dict[tuple, int] = {}
            nodes = []
            for t in ts:
                for node in ((t["subject_cat"], tuple(t["s_box"])),
                             (t["object_cat"], tuple(t["o_box"]))):
                    if node not in seen:
                        seen[node] = len(nodes)
                        nodes.append(node)
            for name, box in nodes:
                rec.boxes.append([float(v) for v in box])
                idx = self._o2i.get(name, 0)
                if idx == 0:
                    n_unmapped_obj += 1
                rec.labels.append(idx)
            for t in ts:
                si = seen[(t["subject_cat"], tuple(t["s_box"]))]
                oi = seen[(t["object_cat"], tuple(t["o_box"]))]
                rec.relations.append(URelation(si, oi, self._p2i[t["predicate"]]))
            recs.append(rec)

        self._cache[key] = recs
        self.n_unmapped_obj = n_unmapped_obj
        return recs[:limit] if limit else recs
