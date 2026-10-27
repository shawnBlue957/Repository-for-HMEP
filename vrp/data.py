"""真实数据加载：VG-150 / VRD（对应模块设计文档 M6.1 离线预计算）。

数据集位置（本机既有，勿重复下载）：
    <PROJECT_ROOT>

VG-150 实测规模（check_report.json 全部 PASS）：
    图像 108,073 | 物体类 150 | 谓词类 50 | 关系 622,705
    划分 train 57,723 / val 5,000 / test 26,446
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
    """数据根目录：优先环境变量 VG_DATA_ROOT，否则用本机既有路径。"""
    return os.environ.get("VG_DATA_ROOT", DEFAULT_ROOT)


@dataclass
class RelationRecord:
    subject: int
    object: int
    predicate: int          # 1..50（0 为 __background__）


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
    """VG-150 标准 SGG 划分的只读加载器。"""

    def __init__(self, root: Optional[str] = None):
        self.root = os.path.join(root or dataset_root(), "vg150")
        dicts_path = os.path.join(self.root, "VG-SGG-dicts.json")
        with open(dicts_path, encoding="utf-8") as f:
            d = json.load(f)
        self.object_classes: List[str] = d["object_classes"]        # 151（含 background）
        self.predicate_classes: List[str] = d["predicate_classes"]  # 51（含 background）
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


# --------------------------------------------------------------- 结构 → 批张量

def to_model_batch(
    images: Sequence[ImageRecord],
    *,
    K: int = 50,
    max_pairs: int = 32,
    feature_dim: int = 512,
    visual_features: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """把标注记录转成模型所需的结构化批（配对级 s_idx/o_idx/y_label）。

    ⚠️ 本函数**只提供图结构**（主客体索引 + 谓词标签 + 层邻域掩码）。
    视觉/文本模态特征 x 需由调用方提供（CLIP 编码），否则以固定种子占位，
    仅用于**契约与一致性验证**，不可用于训练。
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


# --------------------------------------------------------------- M2.5 / 式(16)

def converse_statistics(
    images: Sequence[ImageRecord], K: int = 50, K_plus: int = 51
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从标注统计「逆关系」共现，供式(16) 的三组划分使用。

    谓词索引为 1..K（0 为 __background__），返回时**重映射到 0..K−1**，
    故输出形状为 (K, K) 与 (K,)，与模型的 K 一致。

    ⟹ 这是**数据驱动的**分组依据（终稿 §3.3：分组必须由 VG 标注统计得出，
       不是预先假定 —— 这本身是可发表的实证贡献）。
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
    converse_statistics.n_reverse_pairs = n_rev          # 供报告
    return counts.float(), pair_counts.float()


def epsilon_assoc_from_images(
    images: Sequence[ImageRecord], K_plus: int = 51, max_len: Optional[int] = None
) -> torch.Tensor:
    """从标注三元组构建 ε_assoc 的 51³ 查表（式30）。

    对每张图，枚举关系对 (a→b, b→c)，若 (a→c) 也存在则记一条三元组
    (r_ab, r_bc, r_ac)。返回 (K_plus,K_plus,K_plus) 的违反率表。

    ⚠️ 终稿明示：ε_assoc 与训练无关，是**数据集的组合律违反率**（纯数据属性）；
    若 ε_assoc 与 mR@50 无相关，则收益 B 不成立（可证伪）。
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
        # 显式两跳：(a,b) 与 (b,c) 同时存在 ⟹ 记三元组 (r_ab, r_bc, r_ac)
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
    """式(16)：由逆关系共现统计得 ρ̂ 与三组划分 + 可信度权重。

    ⚠️ **实测发现（本项目在 VG-150 val 上验证）**：VG-150 的标注**极少给出同一
    有序对的反向关系** —— 1,500 张图中 7,494 条关系里仅有 492 条存在逆向配对
    （6.6%）。因此若按「同一对的双向共现」定义 ρ̂，统计量极度稀疏
    （最大亲和度 0.286、均值 0.002），低支持度谓词会被小样本噪声主导。

    ⟹ 本函数据此做两项工程处理（均为**实现约定**，须在论文中说明）：
       ① **最小支持度过滤**：pair_counts < min_support 的谓词不参与判定；
       ② **自适应阈值**：阈值以「有支持谓词的亲和度分位」为基准，而非固定的 0.5。

    ⟹ 更根本的替代方案（与终稿式16 的原式一致）：ρ̂ 本应由**模型预测**的
       TopK 命中率统计（CVPRW2022 做法），而非标注共现。标注共现仅在上表
       稀疏性被显式处理后才可用 —— 该点建议写入论文的「分组方法」小节。

    返回 (mask_non_sym, mask_sym, mask_ambig, w, meta)。
    """
    K = counts.shape[0]
    rho = counts / pair_counts.clamp(min=1.0).unsqueeze(0)
    aff = torch.maximum(rho, rho.t())
    support = pair_counts >= min_support

    # 自适应阈值：以有支持谓词的（对角 / 非对角）亲和度分位为基准
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
