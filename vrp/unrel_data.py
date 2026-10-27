"""UnRel 数据集加载器（接口对齐 VRD / VG150 / GQA）。

## 为什么需要单独的加载器

UnRel（Peyre et al., ICCV 2017）是**纯评测集**：`annotations.mat` 为 1071×1 的
评测标注，**没有 train/test 划分**，平均 1.4 条关系/图 ⟹ 不足以训练。

故本文按官方标准用法做**跨数据集零样本迁移**：
    在 VRD（70 谓词）上训练 ⟹ 在 UnRel 上评测不寻常关系。

## 关键：索引空间必须映射到源域（VRD）

零样本要求「同一个模型权重能直接吃目标域数据」，因此：
- **谓词**：UnRel 的 18 个谓词名全部 ⊆ VRD 的 70 谓词，
  故把谓词映射为 **VRD 的 1-based 谓词号**（`K = 70`，与源域模型输出维一致）。
- **物体类**：同样映射到 **VRD 的物体词表**；UnRel 中不在 VRD 词表内的类 → 0（background）。
  这是零样本的**诚实代价**（目标域未知物体无法用源域嵌入表示），必须在论文中说明。

`pred_space="own"` / `obj_space="own"` 时退化为 UnRel 自有词表（仅作诊断或自训自测用）。

## 产物格式

`~/vrp/data/unrel/artifacts/unrel_annotations.json`
    {"predicates": [...18], "triplets": [{image_id, subject_cat, object_cat,
     predicate, s_box, o_box}, ...]}
图片：`~/vrp/data/unrel/images/{image_id}.jpg`
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
    subject: int          # 图内物体局部 id（0-based）
    object: int
    predicate: int        # 1-based 谓词号（默认在 VRD 空间）


@dataclass
class URecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)     # 0=background
    relations: List[URelation] = field(default_factory=list)


class UnRel:
    """UnRel 只读加载器（默认映射到 VRD 索引空间，供零样本评测）。"""

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
                f"缺少 UnRel 标注 {ap}\n  先运行：python tools/prepare_unrel.py "
                f"--root <unrel 解压根> --out artifacts/unrel")
        with open(ap, encoding="utf-8") as f:
            d = json.load(f)
        self.predicates_raw: List[str] = list(d["predicates"])
        self.triplets: List[dict] = d["triplets"]

        if self.pred_space == "vrd":
            from .vrd_data import VRD
            vrd = VRD()
            self.predicate_classes = list(vrd.predicate_classes)   # 含 background
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

        # 谓词覆盖自检：VRD 空间下应 18/18 命中
        miss = [p for p in self.predicates_raw if p not in self._p2i]
        if miss:
            raise ValueError(f"谓词无法映射到 {self.pred_space} 词表：{miss}")

    # ------------------------------------------------------------------
    def image_dir(self, split: str = "test") -> Optional[str]:
        d = os.path.join(self.root, "images")
        return d if os.path.isdir(d) else None

    def pred_indices(self) -> List[int]:
        """UnRel 谓词在目标空间中的 1-based 谓词号（升序）。"""
        return sorted(self._p2i[p] for p in self.predicates_raw)

    def load_split(self, split: str = "test", limit: Optional[int] = None) -> List[URecord]:
        if split.startswith("train"):
            raise ValueError(
                "UnRel 无训练划分（annotations.mat 为 1071 图评测标注，约 1.4 条关系/图）。\n"
                "  请使用跨数据集零样本协议：VRD 训练 → UnRel 评测\n"
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
