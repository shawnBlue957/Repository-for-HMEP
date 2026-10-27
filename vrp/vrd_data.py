"""VRD (Visual Relationship Detection, Lu et al. ECCV 2016) data loader.

Why VRD is needed
--------------
IR@K / MIR@K requires that ”the logical inverse of the truth predicate is also in the predicate table of the data set”.
Among the 50 predicates of VG-150, there are only 4 sets of semantically valid inverse pairs, covering **6.05%** relations;
The 70 predicates of VRD include 6 sets of high-frequency balanced inverse pairs, covering **32.81%** (5.4 times),
And VRD is a data set designed for inverse relationship analysis. For details, see `Evaluation Indicator Design.md` §2.1.

The interface is exactly the same as `vgp.data.VG150` (root/object_classes/predicate_classes/
K / num_objects / load_split), so the data assembly of `small_sample_experiment`,
The training and evaluation code can be reused as is.

⚠️ Three different agreements with VG-150 (this module is responsible for digesting, and the upper layer does not need to sense)
-----------------------------------------------------------
1. **The order of the boxes is `[YMIN, YMAX, XMIN, XMAX]`**, while for VG-150 it is `[x1,y1,x2,y2]`.
   This module is uniformly converted to the VG-150 sequence; if copied, all ROI features will be silently misaligned.
2. The predicate id is **0-based** (0..69), and the object category is also **0-based** (0..99);
   VG-150 is 1-based (1..50 / 1..150). This module is uniformly +1, leaving 0 as background.
3. In the annotation, the **predicate is directly boxed**: `{’subject’: {’category’:…, ’bbox’:…}, ’predicate’:…,
   ’object’: {...}}`, no additional detection result file is required.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def _root() -> str:
    """VRD root directory. Priority: $VRD_ROOT > $VG_DATA_ROOT/vrd > project data/vrd.

    ⚠️ The project data directory is `Topic 5.../data`, and this file is in `Topic 5.../code/vrp/`,
    Therefore, it is necessary to go up two levels (vp→code→project root) to find `data/vrd`.
    """
    v = os.environ.get("VRD_ROOT")
    if v:
        return v
    base = os.environ.get("VG_DATA_ROOT")
    if base:
        return os.path.join(base, "vrd")
    here = os.path.dirname(os.path.abspath(__file__))          # .../code/vrp
    proj = os.path.dirname(os.path.dirname(here))             # .../Topic 5
    return os.path.join(proj, "data", "vrd")


@dataclass
class VRelation:
    subject: int          # Object id (0-based, +1 → 1..100)
    object: int
    predicate: int        # Predicate id (1..K)


@dataclass
class VRecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)     # [x1,y1,x2,y2]
    labels: List[int] = field(default_factory=list)            # 1..100
    relations: List[VRelation] = field(default_factory=list)


def _find_dir(base: str, *names: str, _maxdepth: int = 4) -> Optional[str]:
    """Find the first existing candidate directory name recursively (limited depth) under base.

    ⚠️ Must be recursive: NeSy4VRD’s image path is
    `extracted/vrd_images/<split>/<split>_images/` (**three levels** nesting),
    The official original package of VRD is `sg_dataset/<split>_images/` (two layers).
    Earlier versions only searched one layer, causing None to be silently returned under the new image.
    """
    for n in names:
        p = os.path.join(base, n)
        if os.path.isdir(p):
            return p
    if _maxdepth <= 0:
        return None
    try:
        subs = sorted(os.listdir(base))
    except OSError:
        return None
    # Drill down into ordinary directories first (determinism is preferred to avoid hitting zip decompression residual directories)
    for s in subs:
        sp = os.path.join(base, s)
        if os.path.isdir(sp):
            got = _find_dir(sp, *names, _maxdepth=_maxdepth - 1)
            if got:
                return got
    return None


class VRD:
    """VRD read-only loader (interface aligned VG150)."""

    def __init__(self, root: Optional[str] = None, ann_dir: Optional[str] = None):
        self.root = root or _root()
        self.ann_dir = ann_dir or self._resolve_ann_dir()
        with open(os.path.join(self.ann_dir, "objects.json"), encoding="utf-8") as f:
            objects = json.load(f)                       # list[str], index = category
        with open(os.path.join(self.ann_dir, "predicates.json"), encoding="utf-8") as f:
            predicates = json.load(f)                    # list[str], index = predicate
        # Consistent with VG150: position 0 is reserved for background
        self.object_classes: List[str] = ["__background__"] + list(objects)
        self.predicate_classes: List[str] = ["__background__"] + list(predicates)
        self.K = len(predicates)                        # 70
        self.num_objects = len(objects)                 # 100
        self._cache: Dict[str, List[VRecord]] = {}

    def _resolve_ann_dir(self) -> str:
        for cand in (os.path.join(self.root, "json_dataset"),
                     os.path.join(self.root, "sg_dataset"),
                     self.root):
            if os.path.exists(os.path.join(cand, "objects.json")):
                return cand
        raise FileNotFoundError(
            f"not found VRD mark (objects.json), Search root directory {self.root}。"
            f" Please unzip first json_dataset.zip")

    # --------------------------------------------------------------- Image Directory
    def image_dir(self, split: str) -> Optional[str]:
        """The image directory corresponding to train/test (the directory name/nesting depth of different images is inconsistent, recursive matching).

        Candidates cover three practical layouts:
          1. VRD original package: `sg_dataset/<split>_images/`
          2. NeSy4VRD: `extracted/vrd_images/<split>/<split>_images/` (three layers)
          3. Directly `<split>_images/`
        """
        is_tr = split.startswith("train")
        suffix = "train_images" if is_tr else "test_images"
        names = (f"sg_{suffix}", suffix, f"sg_dataset/{suffix}")
        got = _find_dir(self.root, *names)
        if got:
            return got
        # Back to the bottom: Recursively find any directory ending with suffix that does contain images
        for root, dirs, files in os.walk(self.root):
            if os.path.basename(root) == suffix and any(
                    f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root
        return None

    # --------------------------------------------------------------- Read annotations
    def load_split(self, split: str, limit: Optional[int] = None) -> List[VRecord]:
        key = f"{split}:{limit}"
        if key in self._cache:
            return self._cache[key][:limit] if limit else self._cache[key]
        fname = ("annotations_train.json" if split.startswith("train")
                 else "annotations_test.json")
        path = os.path.join(self.ann_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing VRD Annotation:{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)                           # {img_name: [rel, ...]}

        recs: List[VRecord] = []
        for img_name, rels in raw.items():
            rec: VRecord = VRecord(image_id=img_name)
            # If the frame of the same object appears repeatedly in different relationships, press (category, box) to remove the duplicates.
            seen: Dict[tuple, int] = {}
            for r in rels:
                s, o = r["subject"], r["object"]
                for node in (s, o):
                    key2 = (node["category"], tuple(node["bbox"]))
                    if key2 in seen:
                        continue
                    ymin, ymax, xmin, xmax = [float(v) for v in node["bbox"]]
                    # ⚠️ VRD is [YMIN, YMAX, XMIN, XMAX] ⟹ Go to [x1,y1,x2,y2] of VG150
                    rec.boxes.append([xmin, ymin, xmax, ymax])
                    rec.labels.append(int(node["category"]) + 1)   # 0 → leave 0 as background
                    seen[key2] = len(rec.boxes) - 1
                rec.relations.append(VRelation(
                    subject=seen[(s["category"], tuple(s["bbox"]))] + 1,
                    object=seen[(o["category"], tuple(o["bbox"]))] + 1,
                    predicate=int(r["predicate"]) + 1,           # 0-based → 1-based
                ))
            recs.append(rec)
        self._cache[key] = recs
        return recs[:limit] if limit else recs

    def describe(self) -> str:
        tr = self.load_split("train")
        te = self.load_split("test")
        return (f"VRD: train {len(tr):,} Figure / test {len(te):,} Figure | "
                f"object class {self.num_objects} | predicate class {self.K} | "
                f"relationship {sum(len(r.relations) for r in tr):,} (train) / "
                f"{sum(len(r.relations) for r in te):,} (test)")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = VRD()
    print(d.describe())
    print("image directory train:", d.image_dir("train"))
    print("image directory test :", d.image_dir("test"))
    r = d.load_split("test")[0]
    print("Sample diagram:", r.image_id)
    print("  number of objects:", len(r.boxes), "| label:", r.labels[:8])
    print("  first two boxes [x1,y1,x2,y2]:", [[round(v) for v in b] for b in r.boxes[:2]])
    print("  relationship (1-based):", [(x.subject, x.predicate, x.object) for x in r.relations[:4]])
    bad = [b for b in r.boxes if b[2] <= b[0] or b[3] <= b[1]]
    print("  Illegal box(x2<=x1 or y2<=y1)Quantity:", len(bad))
