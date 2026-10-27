"""GQA scene graph → VRecord dataset loader (interface aligned to VRD/VG150).

Data source: product of `prepare_gqa.py` (~/vrp/data/gqa/artifacts/gqa_{split}_triplets.json),
Format: {”predicates”: [100 predicates], ”triplets”: [{image_id, subj_id, obj_id,
predicate, subj_name, obj_name, s_box:[x1,y1,x2,y2], o_box:[x1,y1,x2,y2]}, ...]}.

Image: ~/vrp/data/gqa/images/{image_id}.jpg (uploaded after being downloaded by this machine).

Consistent with the interface of VRD/VG150 (root/object_classes/predicate_classes/K/
num_objects / load_split / image_dir), the upper training code can be reused as it is.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _root() -> str:
    v = os.environ.get("GQA_ROOT")
    if v:
        return v
    base = os.environ.get("VG_DATA_ROOT")
    if base:
        return os.path.join(base, "gqa")
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(os.path.dirname(here))
    return os.path.join(proj, "data", "gqa")


@dataclass
class GRelation:
    subject: int          #Local id of the object in the picture (0-based)
    object: int
    predicate: int        # Predicate id (1..K)


@dataclass
class GRecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)     # [x1,y1,x2,y2]
    labels: List[int] = field(default_factory=list)            # 1..num_objects
    relations: List[GRelation] = field(default_factory=list)


class GQA:
    """GQA read-only loader."""

    def __init__(self, root: Optional[str] = None, top_objects: Optional[int] = None):
        self.root = root or _root()
        self.artifacts = os.path.join(self.root, "artifacts")
        # top_objects=None → Read environment variable GQA_TOP_OBJECTS (default 200, GQA-LT requires 0=all 1702 categories)
        self.top_objects = int(os.environ.get("GQA_TOP_OBJECTS", "200")) \
            if top_objects is None else int(top_objects)
        self._load_vocab()

    def _load_vocab(self):
        # The predicate + object vocabulary is uniformly determined by the train triplet (val reuses the same vocabulary to avoid train/val inconsistency)
        train_path = os.path.join(self.artifacts, "gqa_train_triplets.json")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"not found GQA Triplet {train_path}.Please run first prepare_gqa.py.")
        with open(train_path, encoding="utf-8") as f:
            d = json.load(f)
        self.predicates = list(d["predicates"])                # 100
        self.K = len(self.predicates)
        self.predicate_classes: List[str] = ["__background__"] + self.predicates
        self._p2i = {p: i + 1 for i, p in enumerate(self.predicates)}   # 1-based

        # Object vocabulary: take top-N according to subj/obj name frequency
        from collections import Counter
        cnt = Counter()
        for t in d["triplets"]:
            cnt[t["subj_name"]] += 1
            cnt[t["obj_name"]] += 1
        if self.top_objects and self.top_objects > 0:
            top = [name for name, _ in cnt.most_common(self.top_objects)]
        else:                                    # 0 / negative = keep all object classes (GQA-LT)
            top = [name for name, _ in cnt.most_common()]
        self.object_classes: List[str] = ["__background__"] + top
        self.num_objects = len(top)
        self._o2i = {n: i + 1 for i, n in enumerate(top)}       # 1-based
        self._cache: Dict[str, List[GRecord]] = {}

    def image_dir(self, split: str) -> Optional[str]:
        d = os.path.join(self.root, "images")
        return d if os.path.isdir(d) else None

    def load_split(self, split: str, limit: Optional[int] = None) -> List[GRecord]:
        key = f"{split}:{limit}"
        if key in self._cache:
            return self._cache[key][:limit] if limit else self._cache[key]
        # Support train/val/test three-partition (GQA-LT has independent test); the old two-partition data is still available
        if split.startswith("train"):
            fname = "gqa_train_triplets.json"
        elif split.startswith("test"):
            fname = "gqa_test_triplets.json"
        else:
            fname = "gqa_val_triplets.json"
        path = os.path.join(self.artifacts, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing GQA Triplet:{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        triplets = raw["triplets"]

        #Group by image_id
        from collections import defaultdict
        by_img: Dict[str, list] = defaultdict(list)
        for t in triplets:
            by_img[t["image_id"]].append(t)

        recs: List[GRecord] = []
        for image_id, ts in by_img.items():
            rec = GRecord(image_id=image_id)
            seen: Dict[tuple, int] = {}          # (name, box) -> local id
            nodes = []                            # (name, box)
            for t in ts:
                for node in ((t["subj_name"], tuple(t["s_box"])),
                             (t["obj_name"], tuple(t["o_box"]))):
                    if node not in seen:
                        seen[node] = len(nodes)
                        nodes.append(node)
            for name, box in nodes:
                rec.boxes.append(list(box))
                rec.labels.append(self._o2i.get(name, 0))       # 0 = background (not in top-N)
            for t in ts:
                si = seen[(t["subj_name"], tuple(t["s_box"]))]
                oi = seen[(t["obj_name"], tuple(t["o_box"]))]
                pi = self._p2i.get(t["predicate"], 0)
                if pi > 0:
                    rec.relations.append(GRelation(si, oi, pi))
            recs.append(rec)

        self._cache[key] = recs
        return recs[:limit] if limit else recs
