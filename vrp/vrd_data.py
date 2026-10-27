"""VRD（Visual Relationship Detection，Lu et al. ECCV 2016）数据加载器。

为什么需要 VRD
--------------
IR@K / MIR@K 要求「真值谓词的逻辑逆也在该数据集谓词表内」。
VG-150 的 50 个谓词里语义成立的逆对只有 4 组、覆盖 **6.05%** 的关系；
VRD 的 70 个谓词含 6 组高频平衡逆对，覆盖 **32.81%**（5.4 倍），
且 VRD 本就是**为逆关系分析而设计**的数据集。详见 `评测指标设计.md` §2.1。

接口与 `vgp.data.VG150` **完全一致**（root / object_classes / predicate_classes /
K / num_objects / load_split），因此 `small_sample_experiment` 的数据装配、
训练与评测代码可原样复用。

⚠️ 三处与 VG-150 不同的约定（本模块负责消化，上层无需感知）
-----------------------------------------------------------
1. **框的顺序是 `[YMIN, YMAX, XMIN, XMAX]`**，而 VG-150 是 `[x1,y1,x2,y2]`。
   本模块统一转成 VG-150 顺序；若照搬会让所有 ROI 特征静默错位。
2. 谓词 id 是 **0-based**（0..69），物体 category 也是 **0-based**（0..99）；
   VG-150 则是 1-based（1..50 / 1..150）。本模块统一 +1，留 0 作 background。
3. 标注里**谓词直接带框**：`{'subject': {'category':…, 'bbox':…}, 'predicate':…,
   'object': {...}}`，无需额外的检测结果文件。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def _root() -> str:
    """VRD 根目录。优先级：$VRD_ROOT > $VG_DATA_ROOT/vrd > 项目 data/vrd。

    ⚠️ 项目 data 目录是 `专题五.../data`，而本文件在 `专题五.../code/vrp/`，
    故要**上溯两级**（vp→code→项目根）才是 `data/vrd`。
    """
    v = os.environ.get("VRD_ROOT")
    if v:
        return v
    base = os.environ.get("VG_DATA_ROOT")
    if base:
        return os.path.join(base, "vrd")
    here = os.path.dirname(os.path.abspath(__file__))          # .../code/vrp
    proj = os.path.dirname(os.path.dirname(here))             # .../专题五
    return os.path.join(proj, "data", "vrd")


@dataclass
class VRelation:
    subject: int          # 物体 id（0-based，已 +1 → 1..100）
    object: int
    predicate: int        # 谓词 id（1..K）


@dataclass
class VRecord:
    image_id: str
    boxes: List[List[float]] = field(default_factory=list)     # [x1,y1,x2,y2]
    labels: List[int] = field(default_factory=list)            # 1..100
    relations: List[VRelation] = field(default_factory=list)


def _find_dir(base: str, *names: str, _maxdepth: int = 4) -> Optional[str]:
    """在 base 下递归（限深）找第一个存在的候选目录名。

    ⚠️ 必须递归：NeSy4VRD 的图像路径是
    `extracted/vrd_images/<split>/<split>_images/`（**三层**嵌套），
    而 VRD 官方原包是 `sg_dataset/<split>_images/`（两层）。
    早期版本只搜一层，导致在新镜像下静默返回 None。
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
    # 先下钻普通目录（确定性优先，避免命中 zip 解压残留目录）
    for s in subs:
        sp = os.path.join(base, s)
        if os.path.isdir(sp):
            got = _find_dir(sp, *names, _maxdepth=_maxdepth - 1)
            if got:
                return got
    return None


class VRD:
    """VRD 只读加载器（接口对齐 VG150）。"""

    def __init__(self, root: Optional[str] = None, ann_dir: Optional[str] = None):
        self.root = root or _root()
        self.ann_dir = ann_dir or self._resolve_ann_dir()
        with open(os.path.join(self.ann_dir, "objects.json"), encoding="utf-8") as f:
            objects = json.load(f)                       # list[str]，索引 = category
        with open(os.path.join(self.ann_dir, "predicates.json"), encoding="utf-8") as f:
            predicates = json.load(f)                    # list[str]，索引 = predicate
        # 与 VG150 一致：0 号位留作 background
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
            f"未找到 VRD 标注（objects.json），搜索根目录 {self.root}。"
            f" 请先解压 json_dataset.zip")

    # ------------------------------------------------------------ 图像目录
    def image_dir(self, split: str) -> Optional[str]:
        """train/test 对应的图像目录（不同镜像的目录名/嵌套深度不一致，递归匹配）。

        候选覆盖三种实际布局：
          1. VRD 原包：`sg_dataset/<split>_images/`
          2. NeSy4VRD：`extracted/vrd_images/<split>/<split>_images/`（三层）
          3. 直接就是 `<split>_images/`
        """
        is_tr = split.startswith("train")
        suffix = "train_images" if is_tr else "test_images"
        names = (f"sg_{suffix}", suffix, f"sg_dataset/{suffix}")
        got = _find_dir(self.root, *names)
        if got:
            return got
        # 兜底：递归找任何以 suffix 结尾、且确实含图像的目录
        for root, dirs, files in os.walk(self.root):
            if os.path.basename(root) == suffix and any(
                    f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root
        return None

    # ------------------------------------------------------------ 读标注
    def load_split(self, split: str, limit: Optional[int] = None) -> List[VRecord]:
        key = f"{split}:{limit}"
        if key in self._cache:
            return self._cache[key][:limit] if limit else self._cache[key]
        fname = ("annotations_train.json" if split.startswith("train")
                 else "annotations_test.json")
        path = os.path.join(self.ann_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 VRD 标注：{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)                           # {img_name: [rel, ...]}

        recs: List[VRecord] = []
        for img_name, rels in raw.items():
            rec: VRecord = VRecord(image_id=img_name)
            # 同一物体的框在不同关系里重复出现，按 (category, box) 去重
            seen: Dict[tuple, int] = {}
            for r in rels:
                s, o = r["subject"], r["object"]
                for node in (s, o):
                    key2 = (node["category"], tuple(node["bbox"]))
                    if key2 in seen:
                        continue
                    ymin, ymax, xmin, xmax = [float(v) for v in node["bbox"]]
                    # ⚠️ VRD 是 [YMIN, YMAX, XMIN, XMAX] ⟹ 转到 VG150 的 [x1,y1,x2,y2]
                    rec.boxes.append([xmin, ymin, xmax, ymax])
                    rec.labels.append(int(node["category"]) + 1)   # 0 → 留 0 作背景
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
        return (f"VRD: train {len(tr):,} 图 / test {len(te):,} 图 | "
                f"物体类 {self.num_objects} | 谓词类 {self.K} | "
                f"关系 {sum(len(r.relations) for r in tr):,} (train) / "
                f"{sum(len(r.relations) for r in te):,} (test)")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = VRD()
    print(d.describe())
    print("图像目录 train:", d.image_dir("train"))
    print("图像目录 test :", d.image_dir("test"))
    r = d.load_split("test")[0]
    print("样例图:", r.image_id)
    print("  物体数:", len(r.boxes), "| 标签:", r.labels[:8])
    print("  前两个框 [x1,y1,x2,y2]:", [[round(v) for v in b] for b in r.boxes[:2]])
    print("  关系 (1-based):", [(x.subject, x.predicate, x.object) for x in r.relations[:4]])
    bad = [b for b in r.boxes if b[2] <= b[0] or b[3] <= b[1]]
    print("  非法框(x2<=x1 或 y2<=y1)数量:", len(bad))
