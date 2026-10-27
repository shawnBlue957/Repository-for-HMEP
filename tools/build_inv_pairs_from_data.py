"""从真实标注**数据驱动**导出逆谓词对合（替代手工语义配对）。

## 为什么必须改成数据驱动（重要发现）

上一轮我手工按语义配对了 30 组逆谓词（`tools/inv_groups_full.py`），
随后用 **VRD train 真实标注**逐组检验（`/tmp/_ck.py`），结果：

    有命中 5 / 无命中 25

即**手工配的 30 组里只有 5 组在真实标注中站得住**，其余 25 组命中率 ≈ 0%。
逐组实测（r → 候选逆 → 命中率 → Top3 真实反向）：

| r | 手工候选逆 | 命中率 | Top3 真实反向 |
|---|---|---|---|
| `above` | under | 10.6% | below(355), under(256), beneath(160) |
| `behind` | in the front of | **19.1%** | in the front of(428) ✓ |
| `on` | under | 6.7% | wear(580), has(427), under(342) |
| `beside` | across | 0.3% | beside(179), next to(98), near(42) |
| `hold` | carry | 0.0% | on(20), next to(19), above(18) |
| `wear` | attach to | 0.0% | on(580), near(16), cover(14) |
| `in` | outside of | 0.0% | has(47), on(25), under(14) |

**结论**：VRD 的反向标注**绝大多数根本不是该谓词的逻辑逆**（如 `wear` 的反向
Top1 是 `on` 而非「被穿戴」；`hold` 的 Top1 是 `on`）。这与本项目此前实测的
「反向标签与逆谓词表不一致 42.2%、该谓词本身无定义逆 57.5%」完全一致，
说明**手工语义表与数据集标注习惯之间存在系统性鸿沟**。

**⟹ 正确做法：让数据决定谁是逻辑逆。** 本模块按标注证据强度自动构造对合，
并对无证据者走式(17) 的 `φ` 列出口（"r 无定义逆"），
而不是退化为 `Inv(r)=r`（后者等价于要求对称，语义与「有逆」相反）。

## 构造规则（全部可复算，无人工语义判断）

对每个谓词 r：
1. 统计 train 中 (s, r, o) 出现时反向对 (o, r', s) 的共现分布 `rev[r]`；
2. 取 `best = argmax_r' rev[r][r']`，`rate = rev[r][best] / n(r)`；
3. **互反一致性检验**：仅当 `best(best) == r` **且** `rev[best][r] > 0` 时
   接受配对——否则该「逆」是单向伪相关，不能进对合；
4. 强度门槛 `MIN_RATE`（默认 0.05）：命中率低于此视为「无定义逆」⟹ 走 φ 出口；
5. 频率门槛 `MIN_N`（默认 100）：样本太少不足以判定 ⟹ 走 φ 出口；
6. 自反谓词（`best == r` 且自反占比高）⟹ 标记为**对称类**，交由式(24) `L_sym` 处理，
   **不进对合**（对合要求 `r ≠ r'`）。

**产出**：`{2-循环对列表, 对称类列表, 无定义逆列表}`，后者即 φ 的语义集合。

运行:
    python tools/build_inv_pairs_from_data.py --dataset vrd
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_RATE = 0.05      # 反向命中率门槛
MIN_N = 100          # 谓词最少出现次数


def load_vg(split: int = 0):
    """定位并读取 VG-SGG.h5。

    ⚠️ 路径同样需跨平台：数据在项目根的 `data/vg/`，服务器在 `~/vrp/data/vg/`。
    """
    import h5py
    env = os.environ.get("VG_H5")
    cands = [env] if env else []
    cands += [
        os.path.join(ROOT, "data", "vg", "VG-SGG.h5"),
        os.path.join(os.path.dirname(ROOT), "data", "vg", "VG-SGG.h5"),
        os.path.join(ROOT, "..", "data", "vg", "VG-SGG.h5"),
        os.path.expanduser("~/vrp/data/vg/VG-SGG.h5"),
    ]
    path = next((c for c in cands if c and os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(f"未找到 VG-SGG.h5。已尝试：{cands}\n"
                                f"请设环境变量 VG_H5 指向该文件")
    d = json.load(open(os.path.join(os.path.dirname(path), "VG-SGG-dicts.json"),
                       encoding="utf-8"))
    # dicts.json 的键顺序即 label/predicate 索引；去掉 background 占位
    labels = d["label_to_idx"]
    pmap = d["predicate_to_idx"]
    # VG-SGG-dicts.json 的 *_to_idx 是 {名称: 索引} 字典，且 **索引从 1 开始**
    # （0 号保留给 background，dict 里不存在）。
    # 按索引值排序还原为 0-based 列表：索引 i → 列表第 i-1 位，
    # 这样 predicates[0] == 'above'，与 semantic_path.pt 的 0-based 谓词表一致。
    by_idx = sorted(pmap.items(), key=lambda x: x[1])
    assert by_idx[0][1] == 1, f"预期谓词索引从 1 开始，实得 {by_idx[0][1]}"
    preds = [name for name, _ in by_idx]
    print(f"数据文件: {path}")
    return path, labels, preds


def load_vg_json(root: str = None, split: str = "train"):
    """⚠️ 2026-10-18 新增：从 **JSON 标注**（vg150/annotations_*.json）读取 VG。

    背景：服务器上的 `VG-SGG.h5` 已不存在（数据改用 json 格式），
    原 `load_vg()` 走 h5py 分支必然 FileNotFoundError。
    json 版直接给出 `relations[*].predicate_name`，比 h5 的整数索引更直接。

    返回 (path, labels_dict, predicate_names)。
    """
    cands = [root] if root else []
    cands += [os.path.expanduser("~/vrp/data/vg150"),
              os.path.join(ROOT, "data", "vg150"),
              os.path.join(os.path.dirname(ROOT), "data", "vg150")]
    base = next((c for c in cands if c and os.path.exists(
        os.path.join(c, f"annotations_{split}.json"))), None)
    if base is None:
        raise FileNotFoundError(
            f"未找到 vg150/annotations_{split}.json。试过：{cands}")
    path = os.path.join(base, f"annotations_{split}.json")
    data = json.load(open(path, encoding="utf-8"))
    # 谓词名表：按首次出现顺序，保持与 semantic_path.pt 一致的 0-based 序
    preds, seen = [], set()
    labels = {}
    for rec in data:
        for ob in rec.get("objects", []):
            ln = ob.get("label_name")
            if ln is not None and ln not in labels:
                labels[ln] = len(labels) + 1     # 1-based，0 留给 background
        for rel in rec.get("relations", []):
            pn = rel.get("predicate_name")
            if pn is not None and pn not in seen:
                seen.add(pn)
                preds.append(pn)
    print(f"数据文件: {path}  谓词数={len(preds)}  物体类={len(labels)}")
    return path, labels, preds


def collect_reverse_vg_json(path: str, labels: dict, preds: list):
    """⚠️ 2026-10-18：从 **json 标注**收集反向共现（对齐 h5 版语义）。

    互反证据的定义（与 h5 版一致）：
        对关系 (s, o, r)，若**同一对物体** (o, s) 上也标注了谓词 r'，
        则 r' 是 r 的**反向候选**。rev[r][r'] = count(反向出现)。
    这是「逻辑逆」最直接的标注证据（`on` 的反向是 `under`，而非任意共现）。

    返回 (tot, rev)：
        tot[r]  = 谓词 r 的总出现次数
        rev[r] = Counter{r': 次数}
    """
    import collections
    p2i = {n: i for i, n in enumerate(preds)}
    data = json.load(open(path, encoding="utf-8"))
    K = len(preds)
    tot = [0] * K
    rev = [collections.Counter() for _ in range(K)]
    for rec in data:
        objs = rec.get("objects", [])
        n_obj = len(objs)
        rels = rec.get("relations", [])
        # (subj,obj) → [谓词下标,...]
        pair = collections.defaultdict(list)
        for rel in rels:
            pn = rel.get("predicate_name")
            if pn is None or pn not in p2i:
                continue
            si, oi = rel.get("subject"), rel.get("object")
            if si is None or oi is None:
                continue
            if not (0 <= si < n_obj and 0 <= oi < n_obj):
                continue
            pair[(si, oi)].append(p2i[pn])
        for rel in rels:
            pn = rel.get("predicate_name")
            if pn is None or pn not in p2i:
                continue
            si, oi = rel.get("subject"), rel.get("object")
            if si is None or oi is None:
                continue
            if not (0 <= si < n_obj and 0 <= oi < n_obj):
                continue
            ri = p2i[pn]
            tot[ri] += 1
            for rj in pair.get((oi, si), ()):      # 反向对上的谓词
                rev[ri][rj] += 1
    return tot, rev


def collect_reverse_vg(path: str, labels: dict, preds: dict, split: int):
    """遍历 VG-SGG.h5 统计反向共现。split: 0=train, 1=val, 2=test"""
    import h5py
    import numpy as np
    f = h5py.File(path, "r")
    sp = f["split"][:]
    lb = f["labels"][:].ravel()
    rl = f["predicates"][:].ravel()
    rr = f["relationships"][:]
    i2b = f["img_to_first_box"][:]
    i2lb = f["img_to_last_box"][:]
    i2r = f["img_to_first_rel"][:]
    i2lr = f["img_to_last_rel"][:]

    K = len(preds)
    rev = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for i in range(len(sp)):
        if int(sp[i]) != split:
            continue
        objs = i2b[i] + np.arange(i2lb[i] - i2b[i])
        rels = i2r[i] + np.arange(i2lr[i] - i2r[i])
        lab = {int(o): int(lb[o]) for o in objs}
        # 反向索引：(subj,obj) -> [(pred, ...)]
        pair = collections.defaultdict(list)
        for r in rels:
            s, o = int(rr[r][0]), int(rr[r][1])
            if s in lab and o in lab:
                pair[(s, o)].append(int(rl[r]))   # 保留 h5 原索引
        for r in rels:
            s, o = int(rr[r][0]), int(rr[r][1])
            if s not in lab or o not in lab:
                continue
            ri = int(rl[r]) - 1          # h5 谓词索引从 1 开始
            tot[ri] += 1
            for pi in pair.get((o, s), ()):
                rev[ri][pi - 1] += 1     # 同上
    f.close()
    return tot, rev


def load_vrd(split: str = "train"):
    """定位 VRD 标注。

    ⚠️ 路径基准：本文件在 `code/tools/` 下，但数据在**上一级**的 `data/vrd/`；
    服务器上则在 `~/vrp/data/vrd/`。故按「候选列表」逐个探测，
    并支持 `VRD_JSON_DIR` 环境变量覆盖。
    """
    env = os.environ.get("VRD_JSON_DIR")
    cands = [env] if env else []
    cands += [
        os.path.join(ROOT, "data", "vrd", "json_dataset"),
        os.path.join(os.path.dirname(ROOT), "data", "vrd", "json_dataset"),
        os.path.join(ROOT, "..", "data", "vrd", "json_dataset"),
        os.path.expanduser("~/vrp/data/vrd/json_dataset"),
    ]
    base = None
    for c in cands:
        if c and os.path.exists(os.path.join(c, "predicates.json")):
            base = os.path.abspath(c); break
    if base is None:
        raise FileNotFoundError(
            f"未找到 VRD 标注。已尝试：{cands}\n"
            f"请设环境变量 VRD_JSON_DIR 指向含 predicates.json 的目录")
    ann = json.load(open(os.path.join(base, f"annotations_{split}.json"), encoding="utf-8"))
    preds = json.load(open(os.path.join(base, "predicates.json"), encoding="utf-8"))
    print(f"数据目录: {base}")
    return ann, preds


def key_of(o):
    return (tuple(o["bbox"]), o["category"])


def collect_reverse(ann) -> Tuple[collections.Counter, collections.defaultdict]:
    rev = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for _img, lst in ann.items():
        idx = {}
        for r in lst:
            idx.setdefault(key_of(r["subject"]), []).append((r["predicate"], key_of(r["object"])))
        for r in lst:
            ri = r["predicate"]
            tot[ri] += 1
            sk, ok_ = key_of(r["subject"]), key_of(r["object"])
            for (pi, oj) in idx.get(ok_, []):
                if oj == sk:
                    rev[ri][pi] += 1
    return tot, rev


def build_involution(preds: List[str], tot, rev, *,
                     min_rate: float = MIN_RATE, min_n: int = MIN_N,
                     top_m: int = 3):
    """按标注证据构造 (2-循环 ∪ 不动点) 的严格对合。

    ⚠️ 互反检验的修正（实测驱动）：
       VG 的 `above` 反向 Top1 是 `under`(279)，但 `under` 的反向 Top1 是 `on`(895)、
       Top2 才是 `above`(279)。若要求「互为 Top1」，会把 `above↔under` 这类
       **正确的空间逆关系**误判为单向伪相关而丢弃（实测 VG 只剩 1 对 has↔on）。

       故改为：**候选 r' 须出现在 r 的 Top-m 反向中，且 r 出现在 r' 的 Top-m 反向中**
       （互为 Top-m 即接受），再按命中率择优。这样 above↔under 可保留，
       而 `on→has` 这类动作伪相关因 `has` 的 Top-m 里没有 `on` 而被排除。
    """
    K = len(preds)
    sigma = list(range(K))          # 默认不动点
    sym, noinv, pairs = set(), set(), []

    for i, p in enumerate(preds):
        n = tot[i]
        if n < min_n:
            noinv.add(i); continue
        c = rev[i]
        if not c:
            noinv.add(i); continue

        # 候选：按命中率降序的 Top-m
        cands = [j for j, _ in c.most_common(top_m)]
        # 对称类：Top1 是自身且自反占比达门槛
        if cands and cands[0] == i and c[i] / n >= min_rate:
            sym.add(i); continue

        chosen = None
        for j in cands:
            if j == i:
                continue
            rate = c[j] / n
            if rate < min_rate:
                break
            cj = rev[j]
            tops_j = [x for x, _ in cj.most_common(top_m)]
            if i in tops_j and cj[i] > 0:      # 互为 Top-m
                chosen = j
                break
        if chosen is None:
            noinv.add(i); continue
        pairs.append((i, chosen))

    # 只保留**互不相交**的 2-循环；冲突者退化为不动点（否则 σ² ≠ id）
    used, cyc = set(), []
    for i, j in pairs:
        if i in used or j in used or i == j:
            continue
        used.add(i); used.add(j); cyc.append((i, j))
    for i, j in cyc:
        sigma[i], sigma[j] = j, i

    # ⚠️ 2026-10-18 修复（A8 矛盾）：**去掉 noinv / sym 与 2-环的重叠**。
    #
    # 成因：谓词 i 的**单向**命中率不足时会落入 noinv（走 φ 出口）；
    #       但谓词 j 的**反向**检查可能仍把 i 选为伙伴 ⟹ i 同时出现在
    #       「有互反证据（入 2-环）」与「无互反证据（走 φ 出口）」两个集合里。
    #       实测 VG：`above` / `behind` / `on` 三个谓词同时出现在两处
    #       （配对分别为 under↔above、in front of↔behind、has↔on）。
    #
    # 语义上二者**互斥**：既然已确证互反（2-环），就不该再声称「无定义逆」。
    # 故 2-环优先，从 noinv / sym 中剔除其成员。
    cyc_members = {i for i, _ in cyc} | {j for _, j in cyc}
    noinv -= cyc_members
    sym -= cyc_members
    return sigma, cyc, sym, noinv


def sigma_to_matrix(sigma, K) -> List[List[float]]:
    J = [[0.0] * K for _ in range(K)]
    for i in range(K):
        J[i][sigma[i]] = 1.0
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vrd", "vg150"], default="vrd")
    ap.add_argument("--split", default="train",
                    help="vrd: train/test；vg150: 0=train 1=val 2=test")
    ap.add_argument("--min-rate", type=float, default=MIN_RATE)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "vg150":
        # ⚠️ 2026-10-18：服务器上 VG-SGG.h5 已不存在（数据改用 json 格式），
        #    故优先走 json 标注路径；仅当 json 不可用时才回退 h5。
        try:
            path, labels, preds = load_vg_json(split=args.split)
            tot, rev = collect_reverse_vg_json(path, labels, preds)
        except FileNotFoundError:
            sp = {"train": 0, "val": 1, "test": 2}.get(args.split, 0)
            path, labels, preds = load_vg()
            tot, rev = collect_reverse_vg(path, labels, preds, sp)
    else:
        ann, preds = load_vrd(args.split)
        tot, rev = collect_reverse(ann)

    sigma, cyc, sym, noinv = build_involution(preds, tot, rev,
                                              min_rate=args.min_rate, min_n=args.min_n)
    K = len(preds)

    # 验证对合
    import numpy as np
    J = np.asarray(sigma_to_matrix(sigma, K))
    adj = float(np.abs(J @ J - np.eye(K)).max())

    print(f"\n数据驱动对合构造（{args.dataset} {args.split}）")
    print("=" * 72)
    print(f"谓词数 K={K}  min_rate={args.min_rate}  min_n={args.min_n}")
    print(f"2-循环（互不相交）: {len(cyc)} 对，覆盖 {2*len(cyc)} 个谓词"
          f"（{2*len(cyc)/K*100:.1f}%）")
    for i, j in cyc:
        ri = rev[i][j] / tot[i]
        rj = rev[j][i] / tot[j]
        print(f"    {preds[i]:<16s} ↔ {preds[j]:<16s} "
              f"命中率 {ri*100:5.1f}% / {rj*100:5.1f}%")
    print(f"对称类（走式24 L_sym，不进对合）: {len(sym)} 个 → "
          f"{[preds[i] for i in sorted(sym)][:12]}")
    print(f"无定义逆（走式17 的 φ 出口）: {len(noinv)} 个（占 {len(noinv)/K*100:.1f}%）")
    print(f"J²−I 偏差 = {adj:.2e}  {'✅ 严格对合' if adj < 1e-9 else '❌'}")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out) if args.out else
                os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = args.out or os.path.join(ROOT, "artifacts", f"inv_pairs_{args.dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset, "split": args.split,
            "predicates": preds,
            "cycles": [[preds[i], preds[j]] for i, j in cyc],
            "symmetric": [preds[i] for i in sorted(sym)],
            "no_inverse": [preds[i] for i in sorted(noinv)],
            "min_rate": args.min_rate, "min_n": args.min_n, "split": args.split,
            "adj_err": adj,
        }, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())