"""GQA 场景图 → VRecord 数据集加载器（接口对齐 VRD / VG150）。

数据源：`prepare_gqa.py` 的产物（~/vrp/data/gqa/artifacts/gqa_{split}_triplets.json），
格式：{"predicates": [100 个谓词], "triplets": [{image_id, subj_id, obj_id,
predicate, subj_name, obj_name, s_box:[x1,y1,x2,y2], o_box:[x1,y1,x2,y2]}, ...]}。

图片：~/vrp/data/gqa/images/{image_id}.jpg（由本机下载后上传）。

与 VRD/VG150 的接口一致（root / object_classes / predicate_classes / K /
num_objects / load_split / image_dir），上层训练代码可原样复用。
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
    subject: int          # 图内物体局部 id（0-based）
    object: int
    predicate: int        # 谓词 id（1..K）


@dataclass
class GRecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)     # [x1,y1,x2,y2]
    labels: List[int] = field(default_factory=list)            # 1..num_objects
    relations: List[GRelation] = field(default_factory=list)


class GQA:
    """GQA 只读加载器。"""

    def __init__(self, root: Optional[str] = None, top_objects: Optional[int] = None):
        self.root = root or _root()
        self.artifacts = os.path.join(self.root, "artifacts")
        # top_objects=None → 读环境变量 GQA_TOP_OBJECTS（默认 200，GQA-LT 需 0=全部 1702 类）
        self.top_objects = int(os.environ.get("GQA_TOP_OBJECTS", "200")) \
            if top_objects is None else int(top_objects)
        self._load_vocab()

    def _load_vocab(self):
        # 谓词 + 物体词表由 train 三元组统一确定（val 复用同一词表，避免 train/val 不一致）
        train_path = os.path.join(self.artifacts, "gqa_train_triplets.json")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"未找到 GQA 三元组 {train_path}。请先运行 prepare_gqa.py。")
        with open(train_path, encoding="utf-8") as f:
            d = json.load(f)
        self.predicates = list(d["predicates"])                # 100
        self.K = len(self.predicates)
        self.predicate_classes: List[str] = ["__background__"] + self.predicates
        self._p2i = {p: i + 1 for i, p in enumerate(self.predicates)}   # 1-based

        # 物体词表：按 subj/obj name 频率取 top-N
        from collections import Counter
        cnt = Counter()
        for t in d["triplets"]:
            cnt[t["subj_name"]] += 1
            cnt[t["obj_name"]] += 1
        if self.top_objects and self.top_objects > 0:
            top = [name for name, _ in cnt.most_common(self.top_objects)]
        else:                                    # 0 / 负数 = 保留全部物体类（GQA-LT）
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
        # 支持 train/val/test 三划分（GQA-LT 有独立 test）；旧的两划分数据仍可用
        if split.startswith("train"):
            fname = "gqa_train_triplets.json"
        elif split.startswith("test"):
            fname = "gqa_test_triplets.json"
        else:
            fname = "gqa_val_triplets.json"
        path = os.path.join(self.artifacts, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 GQA 三元组：{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        triplets = raw["triplets"]

        # 按 image_id 分组
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
                rec.labels.append(self._o2i.get(name, 0))       # 0 = background（未在 top-N）
            for t in ts:
                si = seen[(t["subj_name"], tuple(t["s_box"]))]
                oi = seen[(t["obj_name"], tuple(t["o_box"]))]
                pi = self._p2i.get(t["predicate"], 0)
                if pi > 0:
                    rec.relations.append(GRelation(si, oi, pi))
            recs.append(rec)

        self._cache[key] = recs
        return recs[:limit] if limit else recs
