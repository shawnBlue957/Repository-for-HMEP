#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 6 — VG-150 小样本对比实验（范畴建模视觉关系预测 vs 基线）。

纪律：只报实测数字。所有数字由本脚本运行产生；默认输出到
      code/artifacts/small_sample_log.json，并打印汇总表。

设计要点（详见项目根目录《小样本对比实验.md》）：
  * 样本：train = annotations_train.json 前 N 张；test = annotations_val.json 前 M 张
  * 视觉特征 x_v：CLIP ViT-B/32（CPU 重建，见 tools/clip_cpu.py）对物体裁剪区域编码
  * 类别特征 x_t：nn.Embedding(151,512) 可学习（本机无 CLIP 文本词表 → 替代方案）
  * 几何特征 x_s：8 维（归一化 cx/cy/w/h/area/aspect + 相对图像中心的带符号偏移）
  * 蒸馏 q：平滑后的谓词频率先验（本机无 CLIP 文本编码器 → 替代方案）
  * 评价：PredCls 协议下的 R@20 / R@50 / mR@50（按谓词宏平均）+ top1 + 平均关系数

用法示例：
  python tools/small_sample_experiment.py --quick
  python tools/small_sample_experiment.py --train-n 400 --test-n 150 --epochs 20 \
      --seeds 0,1,2 --arms A,B,C,D
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

DEFAULT_DATA_ROOT = (
    r"/path/to/vg_data"
)
os.environ.setdefault("VG_DATA_ROOT", DEFAULT_DATA_ROOT)

from vrp import HyperParams, VRPCatModel  # noqa: E402
from vrp.config import all_lambda_dict, get_scheduled  # noqa: E402
from vrp.enrich import build_c_dagger, precompute_M_inv, top1_indicators  # noqa: E402
from vrp.backbone import OrderedDifference, OrderedImplicitFlow  # noqa: E402
from vrp.data import VG150  # noqa: E402

def _resolve_clip_weights() -> str:
    """定位 CLIP 权重，优先级：`CLIP_WEIGHTS` 环境变量 > `$VG_DATA_ROOT/clip/` > 本机默认路径。

    ⚠️ 原先硬编码为 Windows 路径，在服务器（腾讯云）上直接报
    `ValueError: The provided filename /path/to/clip/ViT-B-32.pt does not exist`。
    跨机器运行时必须走环境变量。
    """
    p = os.environ.get("CLIP_WEIGHTS")
    if p and os.path.exists(p):
        return p
    cand = os.path.join(os.environ.get("VG_DATA_ROOT", ""), "clip", "ViT-B-32.pt")
    if os.path.exists(cand):
        return cand
    return r"/path/to/clip/ViT-B-32.pt"


CLIP_WEIGHTS = _resolve_clip_weights()
ART = os.path.join(ROOT, "artifacts")
FEAT_DIM = 512
GEOM_DIM = 8


def apply_loss_fixes(normalize_maha: bool = True, fix_coh: bool = True) -> dict:
    """运行时替换 vrp.model 所用的损失函数（**不修改 vrp/ 源文件**）。

    两项均为「尺度校准」，对**所有臂统一施加**，故不破坏臂间公平比较：

    (1) fix_coh：vrp/losses.py 的 coh_loss 对 Head_1 输出未做 softmax 就做 L1
        归一化；Head_1 输出含负值时归一化除以近零和 → 非法"概率"（含负/爆炸），
        实测 L_coh=8.9e11，主导并摧毁分类分支。此处补 softmax（恢复单纯形）。

    (2) normalize_maha：式(9) 的 ||Δ||²_{G^{-1}} 随维度 d 线性增长（d=256 时
        L_inv≈4.7e2 vs L_cls≈5.5，梯度 9.8e3 vs 2.2e1）→ 分类分支被淹没。此处
        对 Mahalanobis 项按维度取均值（除以 d），使各损失项量级可比。
        式(23) 的表征项同法处理，分布项(D_JS)不变。
    """
    import vrp.model as vrp_model
    from vrp.losses import js_divergence

    info = {}
    if fix_coh:
        def coh_loss_fixed(f_r1, f_r2, P_bar_sq, head1, head2):
            # ⚠️ 2026-10-18：与 vrp/losses.py 的 coh_loss 保持一致
            #    （diag 加权在 **log 空间**，见该函数的说明）。
            #    注：此补丁现已是**冗余**（losses.py 早已修好 softmax），
            #    保留仅为不改变既有校准流程。
            diag_p = torch.diagonal(P_bar_sq)
            log_diag = torch.log(diag_p.clamp(min=1e-12))
            a = torch.softmax(head1(f_r1) + log_diag.unsqueeze(-2), dim=-1)
            b = torch.softmax(head2(f_r2), dim=-1)
            return js_divergence(a, b).mean()
        vrp_model.coh_loss = coh_loss_fixed
        info["coh_fix"] = True

    if normalize_maha:
        def inv_loss_fixed(r_exp, r_imp, G_inv):
            delta = r_exp - r_imp
            maha = torch.einsum("...i,ij,...j->...", delta, G_inv, delta)
            return maha.mean() / delta.shape[-1]
        vrp_model.inv_loss = inv_loss_fixed

        def dag_loss_fixed(p_ij, p_ji_swap, r_ji, Inv_p_ij, c_dag, G_inv, w_i, shared,
                           lambda_dag=1.0):
            l_dist = js_divergence(p_ji_swap, Inv_p_ij)
            c_expected = torch.einsum("...k,kd->...d", p_ij, c_dag)
            diff = shared(r_ji) - c_expected
            l_repr = torch.einsum("...i,ij,...j->...", diff, G_inv, diff) / diff.shape[-1]
            per = w_i * (lambda_dag * l_dist + l_repr)
            return per.mean(), l_dist.mean(), l_repr.mean()
        vrp_model.dag_loss = dag_loss_fixed
        info["maha_norm"] = True
    return info


# ======================================================================
# 消融模块（方向盲 R1 / R2、单输出头 R3）
# ======================================================================

class BlindOrderedDifference(nn.Module):
    """式(4) 的方向盲版本【消融 R1】：系数 MLP 仅吃 v_glo，不含主/客体有序槽位。"""

    def __init__(self, d: int, h: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, f_u, f_s_prev, f_o_prev, v_glo):
        if v_glo.dim() == 2 and f_u.dim() == 3:
            v_glo = v_glo.unsqueeze(1).expand(-1, f_u.shape[1], -1)
        ab = self.mlp(v_glo)
        alpha, beta = ab[..., 0:1], ab[..., 1:2]
        return f_u - alpha * f_s_prev - beta * f_o_prev, alpha, beta


class BlindImplicitFlow(nn.Module):
    """式(5) 的方向盲版本【消融 R2】：隐式流仅吃 f_{u_2}，不含有序三槽。"""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.LayerNorm(d, eps=1e-5))

    def forward(self, f_s2, f_u2, f_o2):
        return self.mlp(f_u2)

    def swap(self, f_s2, f_u2, f_o2):
        return self.mlp(f_u2)   # 方向盲 ⟹ swap 不变


class SingleHead(nn.Module):
    """式(7) 去掉 R3 双输出后的单头：仅输出一份谓词分布。"""

    def __init__(self, d: int, K: int):
        super().__init__()
        self.fc = nn.Linear(3 * d, K)
        self.K = K

    def forward(self, f_s, f_u, f_o):
        p = torch.softmax(self.fc(torch.cat([f_s, f_u, f_o], dim=-1)), dim=-1)
        return p, p


# ======================================================================
# 特征提取（CLIP 物体裁剪 + 几何）
# ======================================================================

class ImageZip:
    """按名读取 VG 图像（只解压所需成员，勿全量解压）。

    ⚠️ **容忍 zip 缺失**（2026-10-18 修复）：`get_image_reader` 在**进入特征分支之前**
    就构造本对象，而 `__init__` 会立即 `zipfile.ZipFile(p)` 打开文件⟹ 即使 ROI/CLIP
    特征缓存已命中（根本不需要读原图），也会因源 zip 不存在而直接崩溃：
        FileNotFoundError: .../data/raw/vg_images/images.zip
    （实测导致全量任务的 roi_vg_ours / roi_vg_abl 全部失败。）

    故改为：缺失的 zip **跳过并告警**，仅当**确实需要读像素**时才会静默返回 None。
    缓存命中路径不受影响。
    """

    def __init__(self, paths: Sequence[str], strict: bool = False):
        self.zips = []
        self.missing: List[str] = []
        for p in paths:
            if not p or not os.path.exists(p):
                self.missing.append(p)
                continue
            try:
                self.zips.append(zipfile.ZipFile(p))
            except Exception as e:                      # 损坏/非 zip
                self.missing.append(f"{p} ({e})")
        if self.missing:
            msg = (f"ImageZip: {len(self.missing)} 个源 zip 不可用，已跳过："
                   f"{[os.path.basename(m) for m in self.missing]}")
            if strict:
                raise FileNotFoundError(msg)
            # 非 strict：仅告警。特征缓存命中时无需读像素；未命中时 read() 返回 None，
            # 上游会统计丢图数并照常跑完（不会静默产出错误结果）。
            print(f"  ⚠️ {msg}", flush=True)
        self.index: Dict[str, Tuple[int, str]] = {}
        for zi, z in enumerate(self.zips):
            for name in z.namelist():
                if name.lower().endswith((".jpg", ".jpeg")):
                    base = os.path.basename(name)
                    stem = os.path.splitext(base)[0]
                    self.index.setdefault(stem, (zi, name))

    def read(self, image_id) -> Optional[bytes]:
        hit = self.index.get(str(image_id))
        if hit is None:
            return None
        zi, name = hit
        try:
            return self.zips[zi].read(name)
        except Exception:
            return None


def geom_features(box, w, h) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    W, H = max(float(w), 1.0), max(float(h), 1.0)
    ncx, ncy = cx / W, cy / H
    nw, nh = bw / W, bh / H
    area = nw * nh
    aspect = float(np.clip(bw / bh, 0.1, 10.0)) / 10.0
    # 相对图像中心的带符号偏移（方向极性的可学入口；主客体相对位移由 R1 的有序差分内隐实现）
    return np.array([ncx, ncy, nw, nh, area, aspect, ncx - 0.5, ncy - 0.5], dtype=np.float32)


def extract_features(records: List[dict], cache_path: str, imgzip: ImageZip,
                     log=print) -> Dict[str, dict]:
    if os.path.exists(cache_path):
        t = time.time()
        cache = torch.load(cache_path, weights_only=False)
        log(f"  载入特征缓存 {os.path.basename(cache_path)}（{len(cache)} 图，{time.time()-t:.1f}s）")
        return cache

    from clip_cpu import CLIPVisionCPU
    from PIL import Image

    # CLIPVisionCPU 自动选设备（CUDA 可用即 GPU），GPU 下 ViT 前向比 CPU 快约 46×。
    enc = CLIPVisionCPU(CLIP_WEIGHTS)
    log(f"  CLIP 视觉塔设备 = {enc.device} / {enc.dtype}")

    # ⚠️ 内存设计（全量 84,169 图时必须）：最终缓存本身约 2.1 GB
    # （84,169 图 × 均值 12.3 物体 × 512 × 4 B）。
    # 早期版本先累积 `feats{iid:{j:vec}}` 再转 `cache{iid:{v:fv}}`，
    # **两份同时驻留 ⟹ 峰值约 4.2 GB**，在本机（可用内存仅 4.5 GB）会大量换页。
    # 现改为**按块直接填充 `cache`**，峰值 ≈ cache + 1 块的裁剪 ≈ 2.4 GB。
    BLOCK_IMGS = 512
    cache: Dict[str, dict] = {}
    miss_img, miss_crop = 0, 0
    t0 = time.time()
    n_enc = 0
    last_log = t0

    for b0 in range(0, len(records), BLOCK_IMGS):
        block = records[b0:b0 + BLOCK_IMGS]
        crops, keys, entries = [], [], []       # entries: (im, fv, W, H)
        for im in block:
            raw = imgzip.read(im.image_id)
            if raw is None:
                miss_img += 1
                continue
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                miss_img += 1
                continue
            W, H = pil.size
            fv = torch.zeros(len(im.labels), FEAT_DIM)
            ei = len(entries)
            for j, box in enumerate(im.boxes):
                x1, y1, x2, y2 = box
                x1, y1 = max(0, min(int(x1), W - 1)), max(0, min(int(y1), H - 1))
                x2, y2 = max(x1 + 1, min(int(x2), W)), max(y1 + 1, min(int(y2), H))
                try:
                    crops.append(pil.crop((x1, y1, x2, y2)))
                except Exception:
                    miss_crop += 1
                    continue
                keys.append((ei, j))
            entries.append((im, fv, W, H))
            pil.close()

        if crops:
            f = enc.encode_pils(crops, batch_size=256)
            for (ei, j), vec in zip(keys, f):
                entries[ei][1][j] = vec
            n_enc += len(crops)

        # 裁剪编码完成后才写入常驻 cache，随即释放本块临时对象
        for im, fv, W, H in entries:
            cache[im.image_id] = {"v": fv, "labels": list(im.labels),
                                  "boxes": [list(b) for b in im.boxes], "W": W, "H": H}
        del crops, keys, entries

        if time.time() - last_log > 20:
            done = min(b0 + BLOCK_IMGS, len(records))
            log(f"    已处理 {done:,}/{len(records):,} 图 | 已编码 {n_enc:,} 裁剪 | "
                f"{n_enc / max(time.time()-t0, 1e-6):.0f} 个/s | "
                f"已用 {(time.time()-t0)/60:.1f} 分钟")

    log(f"  编码完成：{n_enc:,} 裁剪，用时 {(time.time()-t0)/60:.1f} 分钟"
        f"（丢图 {miss_img}，丢框 {miss_crop}）")
    torch.save(cache, cache_path)
    log(f"  特征提取完成：{len(cache)} 图，用时 {time.time()-t0:.1f}s → {os.path.basename(cache_path)}")
    return cache


def extract_image_features(records: List, imgzip, cache_path: str, log=print):
    """图像级 CLIP 特征（整图 resized 224），供 q 的 CLIP 模式使用。

    与 `extract_features` 的物体裁剪特征分开缓存：裁剪缓存可复用，
    但 q = softmax(cos(v_img, v_txt)/τ) 需要**整图**语义，两者的输入不同。
    """
    import time as _t
    if os.path.exists(cache_path):
        t = _t.time()
        cache = torch.load(cache_path, weights_only=False)
        log(f"  载入图像级特征缓存 {os.path.basename(cache_path)}"
            f"（{len(cache)} 图，{_t.time()-t:.1f}s）")
        return cache

    from clip_cpu import CLIPVisionCPU
    from PIL import Image

    enc = CLIPVisionCPU(CLIP_WEIGHTS)
    log(f"  图像级 CLIP 设备 = {enc.device} / {enc.dtype}")

    # 流式：每 BLK 张图解码后立即编码并释放，避免一次性持有上万张 PIL
    BLK = 512
    cache: Dict[str, torch.Tensor] = {}
    buf, buf_keys = [], []
    t0 = _t.time()

    def _flush_img():
        nonlocal buf, buf_keys
        if not buf:
            return
        f = enc.encode_pils(buf, batch_size=256)
        for k, v in zip(buf_keys, f):
            cache[k] = v
        buf, buf_keys = [], []

    for k, im in enumerate(records):
        raw = imgzip.read(im.image_id)
        if raw is not None:
            try:
                buf.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                buf_keys.append(im.image_id)
            except Exception:
                pass
        if len(buf) >= BLK:
            _flush_img()
            log(f"    整图已编码 {len(cache):,}/{len(records):,} | "
                f"{(len(cache)/max(_t.time()-t0,1e-6)):.0f} 张/s | "
                f"已用 {(_t.time()-t0)/60:.1f} 分钟")
    _flush_img()
    torch.save(cache, cache_path)
    log(f"  图像级特征完成：{len(cache)} 张，用时 {(_t.time()-t0)/60:.1f} 分钟"
        f" → {os.path.basename(cache_path)}")
    return cache


# ======================================================================
# 数据装配
# ======================================================================

@dataclass
class SplitData:
    ids: List[str]
    N_max: int
    P_max: int
    # 每图张量
    v: List[torch.Tensor]      # (N_max,512)
    t: List[torch.Tensor]      # (N_max,) long
    s: List[torch.Tensor]      # (N_max,8)
    masks: List[torch.Tensor]  # (3,N_max)
    sidx: List[torch.Tensor]   # (P_max,) long
    oidx: List[torch.Tensor]
    ylab: List[torch.Tensor]
    ytil: List[torch.Tensor]   # (P_max,K) one-hot
    mu: List[torch.Tensor]
    pmask: List[torch.Tensor]
    npairs: List[int]
    rels: List[List[Tuple[int, int, int]]]   # 原始 (s,o,pred0)
    n_obj: List[int]


def build_split(records: List[dict], cache: Dict[str, dict], K: int,
                nmax_cap: int = 80, pmax_cap: int = 64,
                strict: bool = True, inv_of: Optional[np.ndarray] = None,
                rev_mode: str = "legacy", mu_present: float = 1.0,
                mu_absent: float = 0.0, prior_mode: str = "none",
                _rev_reliability: Optional[dict] = None,
                _cooc: Optional[np.ndarray] = None,
                _cooc_total: Optional[np.ndarray] = None,
                cooc_temp: float = 1.0) -> SplitData:
    """装配张量批。

    ⚠️ `nmax_cap` / `pmax_cap` 会**静默截断**超出上限的关系。
    实测 VG150 的分布：val 5,000 张 P_max=150、test 26,446 张 P_max=77，
    故默认 64 在 val 上会丢掉 **2.08% 的关系**（6 张图），在大样本实验中不可接受。
    `strict=True` 时对任何截断直接抛错，避免把「丢了数据」误当成「模型不行」。

    ---------- 式(8) 反向弱标签：`rev_mode` 的三种口径 ----------
    `legacy`（默认，= 论文原式(8) 口径）：
        有反向标注 ⟹ y_tilde_ji = 该反向标注, μ=1；否则 y_tilde_ji = p 自身, μ=0.2。
        **实测该口径有害**：VRD 上单头+BASE 的 D 臂 vs 去反向监督的 I 臂，
        mR@50 −0.0043（t=−15.04）、BRC@1 −0.0119（t=−4.38）。
        根因（VRD train 实测，有反向标注的 9,588 条中）：
          反向标签与逆谓词表一致     **0.3%**
          反向标签与逆谓词表不一致   **42.2%**（假互反）
          该谓词本身无定义逆         **57.5%**
        即模型被要求拟合 42% 的**错误**反向目标。
    `sym_check`：只有「反向标注 == p 的逻辑逆」才 μ=1，其余 μ=0（丢弃假互反）。
    `stats`：按每个谓词的**实测一致率**给 μ（需配合 `_rev_reliability`）。
    """
    recs = [r for r in records if r.image_id in cache]
    max_n = max(len(r.labels) for r in recs)
    max_p = max((len(r.relations) for r in recs), default=1)
    if strict and max_n > nmax_cap:
        raise ValueError(f"N 上限 {nmax_cap} < 实际最大物体数 {max_n}；"
                         f"请调高 nmax_cap（否则会静默丢弃物体）")
    if strict and max_p > pmax_cap:
        raise ValueError(f"P 上限 {pmax_cap} < 实际最大关系数 {max_p}；"
                         f"请调高 pmax_cap（否则会静默丢弃关系）")
    N_max = max(1, min(nmax_cap, max_n))
    P_max = max(1, min(pmax_cap, max_p))

    d = SplitData(ids=[], N_max=N_max, P_max=P_max, v=[], t=[], s=[], masks=[],
                  sidx=[], oidx=[], ylab=[], ytil=[], mu=[], pmask=[], npairs=[], rels=[], n_obj=[])
    n_align_drop = 0
    for r in recs:
        c = cache[r.image_id]
        # ⚠️ 缓存（ROI 特征）与标注（三元组）的**物体顺序必须一致**，否则关系索引会指向错误物体。
        #    两者由不同脚本生成时可能差 1~几 个物体（如 GQA 与 GQA-LT 的对象过滤口径不同），
        #    此处做前缀一致性校验：不一致整图丢弃，而不是静默错位。
        nc = int(c["v"].shape[0])
        n = min(len(r.labels), N_max, nc)
        if len(r.labels) != nc:
            mis = 0
            for j in range(min(n, len(r.boxes), len(c["boxes"]))):
                if (abs(float(r.boxes[j][0]) - float(c["boxes"][j][0])) > 1.0
                        or abs(float(r.boxes[j][1]) - float(c["boxes"][j][1])) > 1.0):
                    mis += 1
                    break
            if mis:
                n_align_drop += 1
                continue
        vv = torch.zeros(N_max, FEAT_DIM)
        vv[:n] = c["v"][:n]
        tt = torch.zeros(N_max, dtype=torch.long)
        tt[:n] = torch.tensor(c["labels"][:n], dtype=torch.long)
        ss = torch.zeros(N_max, GEOM_DIM)
        for j in range(n):
            ss[j] = torch.from_numpy(geom_features(c["boxes"][j], c["W"], c["H"]))
        mk = torch.zeros(3, N_max)
        mk[:, :n] = 1.0

        direct = {}
        rels_kept = []
        for rel in r.relations:
            if rel.subject < n and rel.object < n:
                direct[(rel.subject, rel.object)] = rel.predicate
        for rel in r.relations:
            if rel.subject < n and rel.object < n and len(rels_kept) < P_max:
                rels_kept.append((rel.subject, rel.object, rel.predicate - 1))

        sidx = torch.zeros(P_max, dtype=torch.long)
        oidx = torch.zeros(P_max, dtype=torch.long)
        ylab = torch.zeros(P_max, dtype=torch.long)
        ytil = torch.zeros(P_max, K)
        mu = torch.zeros(P_max)
        pmask = torch.zeros(P_max)
        for j, (s_, o_, p_) in enumerate(rels_kept):
            sidx[j], oidx[j], ylab[j] = s_, o_, p_
            rev = direct.get((o_, s_))
            if rev_mode == "legacy":
                # 论文原式(8) 口径：保持逐位复现（已有全部结果都基于此）
                if rev is not None:
                    ytil[j, rev - 1] = 1.0
                    mu[j] = mu_present
                else:
                    ytil[j, p_] = 1.0
                    mu[j] = 0.2 if mu_absent == 0.0 else mu_absent
                pmask[j] = 1.0
                continue
            # ---------- 修正口径：可信度加权的反向弱标签 ----------
            # 实测旧口径有害（VRD：D vs I 的 mR@50 −0.0043, t=−15.04），
            # 根因是「有反向标注」并不蕴含「反向标签是 p 的逻辑逆」：
            # VRD train 有反向标注的 9,588 条里，一致仅 0.3%、不一致 42.2%、
            # 该谓词无定义逆 57.5%。旧口径把 42% 的**错误**目标当可信监督。
            ytil[j, p_] = 1.0
            inv_p = inv_of[p_] if (inv_of is not None and p_ < len(inv_of)) else -1
            if rev_mode == "cooc":
                # ---------- 共现统计替代：P(K | p, 反向对) ----------
                # 动机（VRD train 实测，9,588 条有反向标注的样本上给出正确反向目标）：
                #   式(8) 原样 3.5% / 理论逆 argmax 3.5% / CLIP 文本相似度 12.2%
                #   **共现统计 top-1 50.1%**（高 14 倍）
                # 优势：只需训练集**整体**统计，不依赖单条标注可信度，
                #       因此可覆盖「无反向标注」的 68.4% 样本（式(8) 在那里只能用
                #       有害的「自反弱标签」）。
                if _cooc is None or _cooc_total is None:
                    ytil[j, p_] = 0.0
                    mu[j] = 0.0
                else:
                    row = _cooc[p_].astype(np.float64)
                    s = float(_cooc_total[p_])
                    if s <= 0:
                        ytil[j, p_] = 0.0
                        mu[j] = 0.0
                    else:
                        prob = row / row.sum()
                        # μ = 该谓词「有反向共现」的比例（样本支持度），
                        # 再按温度软化分布；无任何共现则 μ=0
                        supp = float(row.sum() / max(_cooc_total[p_], 1.0))
                        ytil[j] = torch.from_numpy(
                            (prob ** (1.0 / max(cooc_temp, 1e-6))).astype(np.float32))
                        mu[j] = mu_present * supp
            elif rev is None:
                # 无反向标注：旧实现用「自反」弱标签（μ=0.2），实测同样有害——
                # 它要求 p_ji 在 p 自身上有质量，而反向对真正的谓词常是别的类。
                ytil[j, p_] = 0.0
                mu[j] = mu_absent
            elif inv_p >= 0 and (rev - 1) == inv_p:
                ytil[j, inv_p] = 1.0          # 一致 ⟹ 可信监督
                mu[j] = mu_present
            else:
                # 假互反 ⟹ 丢弃该监督（sym_check）或按实测一致率给 μ（stats）
                ytil[j, inv_p if inv_p >= 0 else p_] = 1.0
                mu[j] = (_rev_reliability or {}).get(p_, 0.0) \
                    if prior_mode == "stats" else 0.0
            pmask[j] = 1.0

        d.ids.append(r.image_id); d.v.append(vv); d.t.append(tt); d.s.append(ss)
        d.masks.append(mk); d.sidx.append(sidx); d.oidx.append(oidx); d.ylab.append(ylab)
        d.ytil.append(ytil); d.mu.append(mu); d.pmask.append(pmask)
        d.npairs.append(len(rels_kept)); d.rels.append(rels_kept); d.n_obj.append(n)
    if n_align_drop:
        print(f"  ⚠ 缓存/标注物体顺序不一致，丢弃 {n_align_drop} 图（避免关系索引错位）")
    return d


def _module_device(m: nn.Module) -> Optional[torch.device]:
    """取模块所在设备：先看参数，再看 buffer（FrozenObjText 只有 buffer）。"""
    for p in m.parameters():
        return p.device
    for b in m.buffers():
        return b.device
    return None


def make_batch(data: SplitData, idx: List[int], obj_emb: Optional[nn.Module] = None) -> dict:
    xv = torch.stack([data.v[i] for i in idx])
    xt = torch.stack([data.t[i] for i in idx])
    if obj_emb is not None:
        # ⚠️ `data.t` 常驻 CPU，而 obj_emb 可能已上 GPU。
        # `nn.Embedding` 走 `F.embedding`，**要求索引与权重同设备**（否则报
        # "index is on cpu, different from other tensors on cuda:0"）；
        # 而 `FrozenObjText` 用的高级索引恰好允许 CPU 索引 ⟹ 只有 Embedding 路径会炸。
        dev_e = _module_device(obj_emb)
        if dev_e is not None and dev_e != xt.device:
            xt = xt.to(dev_e)
        xt = obj_emb(xt)
    xs = torch.stack([data.s[i] for i in idx])
    return {
        "x": {"v": xv, "t": xt, "s": xs},
        "masks": torch.stack([data.masks[i] for i in idx]),
        "s_idx": torch.stack([data.sidx[i] for i in idx]),
        "o_idx": torch.stack([data.oidx[i] for i in idx]),
        "y_label": torch.stack([data.ylab[i] for i in idx]),
        "y_tilde": torch.stack([data.ytil[i] for i in idx]),
        "mu": torch.stack([data.mu[i] for i in idx]),
        "pmask": torch.stack([data.pmask[i] for i in idx]),
    }


# ======================================================================
# 模型装配与臂定义
# ======================================================================

@dataclass
class Arm:
    name: str
    direction_aware: bool
    dual_head: bool
    loss: str          # "FULL" / "BASE"
    prior: bool = False
    no_rev_sup: bool = False   # 置零 μ_i，去掉式(8) 的反向弱标签监督项
    no_coh: bool = False       # 去掉式(28) 的链上相干性 L_coh（2026-10-18 新增）

# 路线 B 的 α（见 _acc_chunk）：s_ij[k] = z_ij[k] + α·LSE(z_ij)。0 = 纯 logits。
SCORE_LSE_ALPHA = 0.0

# 排名口径（2026-10-21 新增，用于诊断 α·LSE 的收益来源）：
#   global —— 历史口径：把「候选对 × K 谓词」展平成一条全局排名取 top-k。
#             此时 pair 间的置信度尺度**参与**排名，故 per-pair 常数平移（α·LSE）
#             会改变 R@K；top1（pair 内 argmax）不受影响。
#   pair   —— 标准 PredCls 口径：每个 (s,o) 对**内部**对 K 个谓词排序取 top-k。
#             此时 α·LSE 对指标**恒等**（同对常数平移不改变内部次序）。
RANK_SCOPE = "global"

# Stage-2 重排序模块（2026-10-21）。三者同时只能有一个非 None。
#   RERANKER      —— 学习式残差头（vrp.rerank.RelReranker）
#   RERANK_CALIB  —— 先验校准偏移 (K,)，Δ_k = -τ·log π_k
#   RERANK_TXT    —— 谓词文本嵌入 (K,d)，学习式头的条件输入
RERANKER = None
RERANK_CALIB = None
RERANK_TXT = None
RERANK_M_TOP = 20
# 逆谓词表 (K+1,) long：供重排序头取「(o,s) 方向逆谓词」的分数（A 方案：信息注入）
RERANK_INV = None

ARMS = {
    "A": Arm("A 本文方法(完整)", True, True, "FULL"),
    "B": Arm("B HMEP主体(消融A1)", False, True, "BASE"),
    "C": Arm("C 频率先验(下界)", True, True, "BASE", prior=True),
    "D": Arm("D 方向盲双流(去R1/R2/R3)", False, False, "BASE"),
    "E": Arm("E 完整架构/仅基础损失", True, True, "BASE"),
    "F": Arm("F 方向盲/完整损失", False, True, "FULL"),
    "G": Arm("G 有序槽位/单头/基础损失", True, False, "BASE"),
    "H": Arm("H 方向盲单头/完整损失", False, False, "FULL"),
    # ---- 分离「双头=方向建模」与「双头=避免矛盾监督」的决定性对照 ----
    # D 与 I 均为单头 + BASE 损失，唯一差别：I 去掉式(8) 的反向弱标签项。
    #   若 I 回到 A/B/E/F 水平 ⟹ 双头收益来自「避免同一分布被矛盾监督」；
    #   若 I 仍≈D ⟹ 双头确有方向建模贡献。
    "I": Arm("I 单头/去反向监督(对照)", False, False, "BASE", no_rev_sup=True),
    # ---- 隔离式(28) 的贡献：FULL 损失但去掉 L_coh ----
    #   与 A 臂唯一差别是式(28) ⟹ A vs J 即式(28) 的净效应。
    "J": Arm("J 完整/去式28相干性", True, True, "FULL", no_coh=True),
}

LOSS_KEYS = {
    # ⚠️ 2026-10-18：本文**不含知识蒸馏**——"KL"(式11) 与 "feat"(式12) 均已删除。
    "BASE": ["cls", "inv"],
    "FULL": ["cls", "inv", "dag_non_sym", "sym", "coh",
             "hom", "spread", "PI", "ord"],
    # ⚠️ 2026-10-18：FULL 但去掉式(28)（用于 J 臂隔离 L_coh 的净效应）
    "FULL_NCOH": ["cls", "inv", "dag_non_sym", "sym",
                  "hom", "spread", "PI", "ord"],
}


_SEM = None
_SEM_NAMES = {}          # sem_file -> 谓词名表（含 background，1-based 位置）


_DEVICE = None


def get_device(explicit: Optional[str] = None) -> torch.device:
    """解析运行设备。代码内部全部用 `device=张量.device` 跟随，故迁移只需移动入口张量。

    ⚠️ CLIP 特征提取（物体裁剪 / 整图）走的是 `clip_cpu.py`，**始终在 CPU**：
    该模块是按 torch CPU 版重建的，且 CLIP 权重为 fp16，在 3060 上收益有限
    而搬运成本高。故本开关只加速**训练与评估**，特征提取仍为 CPU（一次性、可缓存）。
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if explicit:
        dev = torch.device(explicit)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    _DEVICE = dev
    return dev


def load_semantic(use_clip_text: bool, sem_file: str = "semantic_path.pt"):
    """加载 CLIP 语义通路张量（式15/23）。

    `--use-clip-text` 关闭时返回 None，此时 v_txt/v_inv 退化为随机初始化
    （即上一轮实验的口径，M_inv 无语义、范畴层必然测不出作用）。

    ⚠️ 缓存按**文件名**键控：VG 用 `semantic_path.pt`、VRD 用 `vrd_semantic.pt`，
    两者形状不同（(50,512) vs (70,512)、(151,512) vs (101,512)）。
    若沿用单一全局变量，跑完 VRD 再回 VG 会**静默拿到另一数据集的张量**。
    """
    global _SEM
    if not use_clip_text:
        return None
    if _SEM is None or _SEM.get("__file__") != sem_file:
        path = os.path.join(ART, sem_file)
        if not os.path.exists(path):
            hint = ("python tools/build_vrd_semantic.py"
                    if "vrd" in sem_file else "python tools/build_semantic_path.py")
            raise SystemExit(f"缺少 artifacts/{sem_file} —— 请先运行：\n  {hint}")
        d = torch.load(path, weights_only=False)
        d["__file__"] = sem_file
        _SEM = d
        # 记录**0-based 谓词名表**（长度 K，不含 background），
        # 供对合投影（--use-adjoint）把语义组名字映射到谓词下标。
        # ⚠️ `predicates_1b` 在 VG 版是「含 background 的 1-based 表」，
        #    在 VRD 版是「剥掉 background 的 0-based 表」—— 口径不同，
        #    故按「首项是否像 background」判断，避免再错一次。
        _p = None
        if "predicates_1b" in d:
            _p = list(d["predicates_1b"])
            if _p and str(_p[0]).lower().startswith("__"):
                _p = _p[1:]
        elif "predicates" in d:
            _p = list(d["predicates"])
        _SEM_NAMES[sem_file] = _p if _p else []
    return _SEM


def build_arm_model(hp: HyperParams, arm: Arm, seed: int,
                    use_clip_text: bool = False,
                    sem_file: str = "semantic_path.pt",
                    use_adjoint: bool = False,
                    inv_groups=None, train_recs=None, batch_size: int = 32,
                    sym_mask_mode: str = "hard",
                    ds_key: str = "vg150",
                    log=print) -> VRPCatModel:
    torch.manual_seed(seed)
    model = VRPCatModel(hp, {"v": FEAT_DIM, "t": FEAT_DIM, "s": GEOM_DIM})
    K, d = hp.K, hp.D
    # ⚠️ 必须传 sem_file：VRD 用 vrd_semantic.pt（M_inv 70×70），
    #    VG 用 semantic_path.pt（M_inv 50×50）。漏传会加载错数据集的张量，
    #    在 forward 时报 `mat1 and mat2 shapes cannot be multiplied`。
    sem = load_semantic(use_clip_text, sem_file=sem_file)
    if sem is not None:
        # 式(15)：真实 CLIP 语义对偶矩阵（非随机），已对齐到模型维度 d
        M_inv = sem["M_inv_d"].to(dtype=torch.float32)
        J = sem["J"].to(dtype=torch.float32)
        c_dag = sem["c_dag_d"].to(dtype=torch.float32)
        assert M_inv.shape[0] == K, (
            f"语义张量谓词数 {M_inv.shape[0]} != 数据集 K={K}"
            f"（sem_file={sem_file}）")
        if use_adjoint:
            # ---- 对合投影修正（tools/adjoint_projection.py）----
            # 实测：CLIP 余弦型 M_inv 的 top1 指示 J **不满足对合**
            #   （VG 复原率 2.1%、29/50 自指；VRD 1.3%、57/70 自指），
            #   故式(23) 的「互反约束」约束的是非对合算子。
            # 修正：改用由**语义组**构造的严格对合 J_adj（J² = I、σ² = id）。
            from adjoint_projection import (build_adjoint_projection_from_sigma,
                                           build_sigma_from_groups)
            # _SEM_NAMES 已是**0-based 谓词名表**（长度 K，不含 background），
            #    故直接用，**不要**再切 [1:]（那会错位一位）。
            _names0 = list(_SEM_NAMES.get(sem_file, []))
            assert len(_names0) == K or not _names0, (
                f"谓词名表长度 {len(_names0)} != K={K}")
            # ⚠️ 语义组里的名字可能不在谓词表内（不同数据集的谓词名不同），
            #    build_sigma_from_groups 会跳过不存在的名字 —— 若结果 0 个 2-循环，
            #    说明组名与谓词表不匹配，必须核对。
            if inv_groups:
                _missing = sorted({n for A, B in inv_groups
                                   for n in list(A) + list(B)
                                   if n not in _names0})
                if _missing:
                    print(f"[adjoint] ⚠ 语义组中 {len(_missing)} 个名字不在谓词表："
                          f"{_missing[:6]}")
            sigma = build_sigma_from_groups(inv_groups or [], K, names=_names0)
            J_adj = build_adjoint_projection_from_sigma(sigma)
            J = torch.as_tensor(J_adj, dtype=torch.float32)
            M_inv = J.clone()          # Inv_adj 以 J_adj 本身作 M
            # ⚠️ c^†_k 需与 SharedProjection 的输出维度 d 相同，
            #    故必须用**降维后**的 v_inv_d（d 维），不是 512 维的 v_inv
            _v_inv_d = sem.get("v_inv_d", sem.get("v_inv"))
            if _v_inv_d is not None and _v_inv_d.shape[-1] == d:
                from adjoint_projection import c_dagger_from_adj
                c_dag = c_dagger_from_adj(J_adj, _v_inv_d.float())
    else:
        g = torch.Generator().manual_seed(0)
        v_txt = torch.randn(K, d, generator=g)
        v_inv = torch.randn(K, d, generator=g)
        M_inv = precompute_M_inv(v_txt, v_inv)
        J = top1_indicators(M_inv)
        c_dag = build_c_dagger(M_inv, v_inv, J)
    # ⚠️ 2026-10-19 重大修正（v2）：掩码口径必须用**数据驱动分组**，
    #    与主实验 `load_data_driven_groups()` 同源（inv_pairs_{dataset}.json），
    #    而非 `vrd_semantic.pt` 的对合算子 J。
    #    历史踩坑：① group_partition.pt 是 50 维(VG) ⟹ 旧降级分支把 mask_sym 置零；
    #              ② v1 用 J[k,k]>0.5 判「自反」得到 57（错把「σ 不动点」当「自反」）。
    #    正确口径：symmetric 字段 = 真正自反（VRD 4 个：next to/near/beside/by）。
    _gp_cands = []
    # ⚠️ 2026-10-19 修正：分组键必须用 **args.dataset**，不能从 sem_file 猜。
    #    旧写法 `"vrd" if "vrd" in sem_file else "vg150"` 会把 GQA/UnRel 也判成 vg150，
    #    导致 GQA 载入 inv_pairs_vg150.json（100 维凑巧通过维度检查）而静默用错分组 ——
    #    实测得到「自反 3」而 GQA 真实 self-reflexive 为 0。
    _ds_key = str(ds_key)
    _gp_file = os.path.join(ART, f"group_partition_{_ds_key}.pt")
    if os.path.exists(_gp_file):
        _gp_cands.append(_gp_file)
    _gp_cands.append(os.path.join(ART, "group_partition.pt"))

    ms = mns = ma = w = None
    for _p in _gp_cands:
        if not os.path.exists(_p):
            continue
        _g = torch.load(_p, weights_only=False)
        _w = _g["w"].reshape(-1)
        if _w.numel() != K:
            log(f"  [分组] 跳过 {os.path.basename(_p)}（{_w.numel()} 维 ≠ K={K}）")
            continue
        w, mns = _w, _g["mask_non_sym"].reshape(-1)
        ms, ma = _g["mask_sym"].reshape(-1), _g["mask_ambig"].reshape(-1)
        log(f"  [分组] 载入 {os.path.basename(_p)}（{K} 维）："
            f"自反 {int((ms>0).sum())} / 2-循环 {int((mns>0).sum())} / 歧义 {int((ma>0).sum())}")
        break

    if ms is None:
        # 无匹配文件：从 inv_pairs_{dataset}.json 的权威分组现场导出
        # （与 build_vrd_group_partition.py 完全同口径）
        log(f"  [分组] ⚠ 无 K={K} 的分组文件，改由 inv_pairs_{_ds_key}.json 现场导出")
        _invp = os.path.join(ART, f"inv_pairs_{_ds_key}.json")
        if os.path.exists(_invp):
            import json as _json
            _d = _json.load(open(_invp, encoding="utf-8"))
            _preds = _d["predicates"]; _n2i = {n: i for i, n in enumerate(_preds)}
            _sym = {_n2i[n] for n in _d.get("symmetric", []) if n in _n2i}
            _cyc = set()
            for _a, _b in _d.get("cycles", []):
                if _a in _n2i: _cyc.add(_n2i[_a])
                if _b in _n2i: _cyc.add(_n2i[_b])
            ms = torch.zeros(K); mns = torch.zeros(K)
            for _k in _sym: ms[_k] = 1.0
            for _k in _cyc: mns[_k] = 1.0
            ma = torch.zeros(K)
            for _k in range(K):
                if _k not in _sym and _k not in _cyc: ma[_k] = 1.0
            w = torch.full((K,), 0.5); w[ms > 0] = 1.0; w[mns > 0] = 0.0
            log(f"  [分组] 现场导出：自反 {int((ms>0).sum())} / 2-循环 {int((mns>0).sum())} "
                f"/ 歧义 {int((ma>0).sum())}")
        else:
            raise FileNotFoundError(
                f"无 K={K} 的分组文件，且缺少 inv_pairs_{_ds_key}.json。"
                f"请先运行 tools/build_inv_pairs_from_data.py 与 "
                f"tools/build_vrd_group_partition.py。")

    # ---- 掩码维度一致性断言（2026-10-19）----
    # 背景：曾因 group_partition.pt 是 VG 口径(50 维) 而 VRD 的 K=70，
    #       触发降级分支把 mask_sym 静默置零 —— 硬掩码从未生效且不报错。
    #       这两条断言把「静默失败」改为「显式报错」。
    for _nm, _v in (("mask_sym", ms), ("mask_non_sym", mns), ("mask_ambig", ma)):
        assert _v.numel() == K, f"{_nm} 维度 {_v.numel()} != K={K}：分组文件口径不匹配"
    if int((ms > 0).sum()) == 0:
        # GQA / UnRel 的 self-reflexive 谓词实测为 0（GQA symmetric=0，见
        # 《分析报告-GQA数据特征与本文适配性》），这是**数据集性质**而非配置错误：
        # 式(24) 恒为 0，L_sym 在论文中标注 N/A，不作为消融项解读。
        if _ds_key in ("gqa", "gqa_lt", "unrel"):
            log(f"  [sym_mask] ⚠ {_ds_key} 无自反谓词（symmetric=0，数据驱动分组）："
                f"式(24) L_sym 恒为 0 ⟹ 论文标注 N/A；其余约束不受影响")
        else:
            raise AssertionError(
                f"mask_sym 全零：式(24) 对称性约束将静默失效（K={K}）。"
                f"请先运行 tools/build_vrd_group_partition.py 生成 "
                f"{K} 维的 group_partition_vrd.pt")

    model.register_frozen(M_inv, J, c_dag, w, ms, mns, ma)

    # 2026-10-18：构造 **soft 对称掩码**（修复式24 的 L_sym 恒为 0）
    #    硬 mask_sym 仅含 4 个对称谓词 ⟹ 长尾谓词无对称监督 ⟹ batch 内真值
    #    几乎不命中 ⟹ L_sym 恒 0。现用谓词 CLIP 文本嵌入的语义近邻构造连续权重。
    #
    # ⚠️ 2026-10-19：λ 扫描 + λ=0 对照证明 **soft 掩码过宽**——
    #    实测「近对称(>0) 70 个 / 全部 K 个，权重均值 0.9745」，
    #    即几乎所有谓词都被当作「近对称」而施加对称约束。
    #    但 VRD 关系语义本身强烈非对称（left-of vs right-of、on vs under），
    #    强行对称化直接抹平方向信息 ⟹ λ×100 时 sym 独占 16.9% 梯度，
    #    指标单调下降（mR@50 t=−7.90）。
    #    故新增 `--sym-mask-mode`：
    #       hard —— **只约束真正自反的谓词**（mask_sym 内的严格对称谓词），
    #              宁可 L_sym 监督稀疏，也不误伤非对称谓词；
    #       soft —— 旧行为（语义近邻连续权重，全覆盖），保留供消融对比；
    #       tiered —— 折中：硬掩码 ∪ 高阈值(τ=0.45)近邻，近邻权重减半。
    _sym_mode = sym_mask_mode or "hard"
    try:
        _vt = sem.get("v_txt_d", sem.get("v_txt"))
        if _sym_mode == "hard" or _vt is None or _vt.shape[0] != K:
            if _sym_mode == "hard":
                log(f"  [sym_mask] **硬掩码模式**：仅约束 {int((ms > 0).sum())}/{K} "
                    f"个严格自反谓词，不施加上软权重")
            else:
                log("  [sym_mask] ⚠ 谓词文本嵌入不可用，沿用硬 mask_sym")
        elif _sym_mode == "soft":
            from vrp.losses import build_soft_sym_mask as _bsm
            _soft = _bsm(_vt.float(), ms, tau=0.06, floor=0.02)
            model.register_soft_sym(_soft)
            log(f"  [sym_mask] soft 模式(消融对照)：完全对称 {int((ms>0).sum())} 个，"
                f"近对称(>0) {int((_soft>0).sum())} 个，权重均值 {float(_soft.mean()):.4f}")
        else:  # tiered
            from vrp.losses import build_soft_sym_mask as _bsm
            _soft = _bsm(_vt.float(), ms, tau=0.45, floor=0.0)
            # 近邻权重减半，惩罚误伤
            _soft = torch.where(_soft > 0, _soft * 0.5, _soft)
            model.register_soft_sym(_soft)
            log(f"  [sym_mask] tiered 模式：τ=0.45，受约束谓词 {int((_soft>0).sum())}/{K} 个，"
                f"权重均值 {float(_soft.mean()):.4f}")
    except Exception as e:
        log(f"  [sym_mask] ⚠ 构造失败（{type(e).__name__}: {e}），沿用硬 mask_sym")

    # 2026-10-18：用**全训练集**谓词频次预标定 presence（修复式25/26 车闸恒为 0）
    _cnt = np.zeros(K, dtype=np.float32)
    for _r in (train_recs or []):
        for _rel in _r.relations:
            if 1 <= _rel.predicate <= K:
                _cnt[_rel.predicate - 1] += 1.0
    _presence = torch.from_numpy((_cnt > 0).astype(np.float32))
    model.register_presence(_presence)
    log(f"  [presence] 训练集出现过 {int((_cnt > 0).sum())}/{K} 个谓词；"
        f"频次<{batch_size} 的低频谓词 {int((_cnt < batch_size).sum())} 个（车闸仍生效）")
    # 2026-10-18：θ 改为在 **forward 内首批自动标定**（见 model.py 的 _theta_calibrated）。
    # 原先传固定张量 tensor([0.5,1.0,1.5,2.0])，而真实 V_r 量级远小于 0.5
    # ⟹ Q10 分位数仍高于所有 V_r ⟹ max(0, θ_min − V_r) 恒 0 ⟹ L_spread 永不生效。
    model.calibrate_theta(torch.tensor([0.5, 1.0, 1.5, 2.0]), relative=False)
    model._theta_auto = True

    if not arm.direction_aware:
        model.backbone.diff1 = BlindOrderedDifference(d, hp.H)
        model.backbone.diff2 = BlindOrderedDifference(d, hp.H)
        model.backbone.implicit = BlindImplicitFlow(d)
    if not arm.dual_head:
        model.dual_head = SingleHead(d, K)
    return model


def reassemble_total(out: dict, arm: Arm, lambda_inv: float, hp: HyperParams) -> torch.Tensor:
    lambdas = all_lambda_dict(hp, lambda_inv)
    total = None
    keys = list(LOSS_KEYS[arm.loss])
    # 2026-10-21：**可选附加项**。它们不属任何 arm 的固定组合，是否产出由
    # vrp.model 的模块级开关决定（如 PAIR_REL 控制 pair_rel）。
    # ⚠️ 白名单机制曾导致 pair_rel 损失被静默丢弃（模型产出、total 不计 ⟹
    #   梯度为 0，AUC 停在 0.50 且毫无报错）。凡新增可选损失必须在此登记。
    for _opt in ("pair_rel",):
        if _opt in out and _opt not in keys:
            keys.append(_opt)
    for k in keys:
        if k not in out:
            continue
        # ⚠️ 2026-10-18 防御：跳过非张量项（此前某项为非张量导致
        #    `can't multiply sequence by non-int of type 'float'` 且指不到来源）
        _v = out[k]
        if not torch.is_tensor(_v):
            continue          # 防御：跳过非张量项（诊断字段等）
        _w = lambdas.get(k, 1.0)
        if not torch.is_tensor(_w) and not isinstance(_w, (int, float)):
            continue
        term = _w * _v
        total = term if total is None else total + term
    if total is None:
        total = torch.zeros(())
    return total


# ======================================================================
# 训练 / 评估
# ======================================================================

class FrozenObjText(nn.Module):
    """x_t 的 CLIP 物体类文本嵌入（替代 nn.Embedding(151,512)）。

    语义通路的类别侧入口：终稿 §3.7 的 `v_txt/v_inv` 对应谓词侧，
    本类提供**物体侧**的冻结文本特征。padding_idx=0（background）置零。
    表已由 build_semantic_path 降到模型维度 d（冻结正交投影）。
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table.float())
        with torch.no_grad():
            self.table[0].zero_()
        self.dim = table.shape[1]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx.long()]

    def parameters(self, recurse: bool = True):
        return iter(())          # 全冻结，无可训练参数


def to_device(batch: dict, dev: torch.device) -> dict:
    """把一个 batch（含嵌套的 x 字典）搬到目标设备。"""
    out = dict(batch)
    out["x"] = {k: v.to(dev, non_blocking=True) for k, v in batch["x"].items()}
    for k, v in batch.items():
        if k != "x" and torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
    return out



# ======================================================================
# 分块流式装配（2026-10-19）
# 背景：GQA 全量 train 72,581 图 × nmax=64 的 padding 使 build_split 峰值显存/RAM
#       达 10 GB 以上（x: 72581*64*512*4B ≈ 9.5 GB），30 GB 机器 + 2.4 GB 特征缓存
#       直接触发 swap 抖动，sshd 被拖死。
# 方案：把 train_recs 切成固定块，**逐块装配 → 训练 → 释放**，峰值内存 ≈ 1 块；
#       块结果落盘复用（默认缓存目录），避免每个 epoch 重算。
# ⚠️ 语义边界：块内打乱 + 每 epoch 打乱块顺序 ⟹ 全局随机性略弱于全量 shuffle，
#    但每个 epoch 仍然看过全部样本一次；论文需在实验设置里注明「分块装配」。
# ======================================================================

def _save_chunk(sd, path, fp16=False):
    """分块落盘。fp16=True 时把最大的两块 v(N,512) 与 ytil(P,K) 存成半精度，
    磁盘/读盘减半（GQA 全量 72,581 图 f32 约 18 GB，f16 约 9 GB）。
    载入时无条件还原 f32，训练数值与 f32 缓存**同分布**（仅存储精度差异）。"""
    if fp16:
        sd.v = [t.half() for t in sd.v]
        sd.ytil = [t.half() for t in sd.ytil]
    torch.save({"fp16": bool(fp16), "sd": sd}, path)


def _load_chunk(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "sd" in obj:
        sd = obj["sd"]
        if obj.get("fp16"):
            sd.v = [t.float() for t in sd.v]
            sd.ytil = [t.float() for t in sd.ytil]
        return sd
    return obj

def build_train_chunks(train_recs, cache, K, nmax, pmax, chunk, cache_dir,
                       split_kw, log=print, rebuild=False, fp16=False):
    """把训练记录切成固定块，逐块 build_split 并落盘。

    返回 (paths, first_sd, agg)：
      paths    块文件路径（按原顺序）
      first_sd 第 0 块（已在内存，供标定/兼容下游代码）
      agg      聚合统计 {n_img, n_kept, n_raw}
    """
    os.makedirs(cache_dir, exist_ok=True)
    n = len(train_recs)
    parts = [train_recs[i:i + chunk] for i in range(0, n, chunk)]
    paths, first_sd = [], None
    n_kept = n_raw = 0
    t0 = time.time()
    for ci, part in enumerate(parts):
        path = os.path.join(cache_dir, f"chunk_{ci:04d}_n{len(part)}.pt")
        if os.path.exists(path) and not rebuild:
            sd = _load_chunk(path)
        else:
            sd = build_split(part, cache, K, nmax, pmax, **split_kw)
            _save_chunk(sd, path, fp16=fp16)
        paths.append(path)
        if first_sd is None:
            first_sd = sd
        n_kept += int(sum(len(x) for x in sd.rels))
        n_raw += sum(len(r.relations) for r in part if r.image_id in cache)
        log(f"  [chunk {ci+1}/{len(parts)}] {len(part)} 图 → {os.path.basename(path)}"
            f"（累计 {n_kept:,}/{n_raw:,} 条关系，{time.time()-t0:.0f}s）")
        if ci > 0:
            del sd
    log(f"  [分块装配] {len(parts)} 块，每块 ≤{chunk} 图；"
        f"总关系 {n_kept:,}/{n_raw:,}（保留率 {100*n_kept/max(n_raw,1):.4f}%）"
        f"；耗时 {time.time()-t0:.0f}s；缓存目录 {cache_dir}")
    return paths, first_sd, {"n_img": n, "n_kept": n_kept, "n_raw": n_raw}


def train_arm(arm: Arm, hp: HyperParams, data: SplitData, args, seed: int,
             ds=None, train_recs=None, log=print, chunk_paths=None):
    args._train_recs_cache = train_recs
    torch.manual_seed(seed)
    random.seed(seed)
    dev = get_device(getattr(args, "device", None))
    use_clip_text = getattr(args, "use_clip_text", False)
    sem_file = semantic_file(getattr(args, "dataset", "vg150"))
    sem = load_semantic(use_clip_text, sem_file=sem_file)
    _use_adj = getattr(args, "use_adjoint", False)
    _groups = None
    # 逆谓词来源（修复 P0）：手工语义组 vs **标注数据驱动**
    #   manual  = 原 VG_INV_GROUPS / VRD_INV_GROUPS（VG 仅 4 组 ⟹ 84% 退化为不动点）
    #   data    = tools/build_inv_pairs_from_data.py 的产物，只保留标注证据支持的
    #             互反 2-循环；无证据者走式(17) 的 φ 出口，而非退化为 Inv(r)=r
    _src = getattr(args, "inv_source", "data")
    if _use_adj and _src == "data":
        _groups = load_data_driven_groups(args.dataset)
        if _groups is None:
            print("[inv] ⚠ 未找到数据驱动逆谓词表，回退到 manual 语义组")
            _groups = (VRD_INV_GROUPS if args.dataset == "vrd" else VG_INV_GROUPS)
    elif _use_adj:
        if args.dataset == "vrd":
            from build_vrd_semantic import VRD_INV_GROUPS as _groups
        else:
            _groups = VG_INV_GROUPS
    model = build_arm_model(hp, arm, seed, use_clip_text=use_clip_text,
                            sem_file=sem_file, use_adjoint=_use_adj,
                            inv_groups=_groups,
                            train_recs=(getattr(args, "_train_recs_cache", None)),
                            batch_size=getattr(args, "batch_size", 32),
                            sym_mask_mode=getattr(args, "sym_mask_mode", "hard"),
                            ds_key=getattr(args, "dataset", "vg150"),
                            log=log)
    if sem is not None:
        # obj_txt 保持 512 维：x_t 与 x^v 同处 CLIP 联合嵌入空间，
        # 且 backbone 的 proj["t"] 按 d_in["t"]=FEAT_DIM 构建。
        obj_emb = FrozenObjText(sem["obj_txt"])
        params = list(model.parameters())          # obj_emb 全冻结
    else:
        n_obj = len(getattr(ds, "object_classes", [None] * FEAT_DIM))
        obj_emb = nn.Embedding(n_obj, FEAT_DIM)
        params = list(model.parameters()) + list(obj_emb.parameters())
    model.to(dev)
    obj_emb.to(dev)
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    n = len(data.ids)
    bs = args.batch_size
    # 分块模式下，标定与训练都从磁盘按块取数据（峰值内存 = 1 块）
    _sd0 = data
    if chunk_paths:
        _sd0 = _load_chunk(chunk_paths[0])
        n = len(_sd0.ids)
    # theta 标定：用首个 batch 的健康态 V_r
    if arm.loss == "FULL":
        with torch.no_grad():
            b0 = to_device(make_batch(_sd0, list(range(min(bs, n))), obj_emb), dev)
            out0 = model(x=b0["x"], masks=b0["masks"], s_idx=b0["s_idx"], o_idx=b0["o_idx"],
                         y_label=b0["y_label"], m_i=b0["pmask"], lambda_inv=0.5, omega_p=hp.OMEGA_P)
            model.calibrate_theta(out0["_V_r"].detach())

    model.train()
    losses = []
    parts_seen = {}
    t0 = time.time()
    def _iter_batches(ep):
        """按 epoch 产出 (SplitData, 索引批)。分块模式逐块加载、块间顺序随机。"""
        if not chunk_paths:
            perm = list(range(n))
            random.Random(seed * 1000 + ep).shuffle(perm)
            for i in range(0, n, bs):
                yield data, perm[i:i + bs]
            return
        order = list(range(len(chunk_paths)))
        random.Random(seed * 7919 + ep).shuffle(order)
        for ci in order:
            sd = _load_chunk(chunk_paths[ci])
            m = len(sd.ids)
            perm = list(range(m))
            random.Random(seed * 1000 + ep * 97 + ci).shuffle(perm)
            for i in range(0, m, bs):
                yield sd, perm[i:i + bs]
            del sd

    for ep in range(args.epochs):
        lam_inv = get_scheduled("lambda_inv", ep, hp)
        ep_loss, nb = 0.0, 0
        for _sd, idx in _iter_batches(ep):
            b = to_device(make_batch(_sd, idx, obj_emb), dev)
            if arm.no_rev_sup:
                # 去掉式(8) 的反向弱标签项（μ_i ≡ 0），用于分离双头的作用机制
                b["mu"] = torch.zeros_like(b["mu"])
            # ⚠️ 2026-10-18：式(12) 特征蒸馏已删，故不再取 v_txt_y
            kw = {}
            out = model(x=b["x"], masks=b["masks"], s_idx=b["s_idx"], o_idx=b["o_idx"],
                        y_label=b["y_label"], y_tilde_ji=b["y_tilde"], mu=b["mu"], m_i=b["pmask"],
                        lambda_inv=lam_inv, omega_p=hp.OMEGA_P, **kw)
            total = reassemble_total(out, arm, lam_inv, hp)
            if not torch.isfinite(total):
                continue
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            ep_loss += float(total); nb += 1
            if ep == 0:
                for k in LOSS_KEYS[arm.loss]:
                    if k in out:
                        parts_seen[k] = float(out[k])
        sched.step()
        losses.append(ep_loss / max(nb, 1))
        # 每个 epoch 打一行心跳（供服务器监控判断"是否在推进"，不参与任何计算）
        log(f"  [ep {ep+1}/{args.epochs}] loss={losses[-1]:.4f}"
            f"  batches={nb}  用时 {time.time()-t0:.0f}s")
    train_time = time.time() - t0
    return model, obj_emb, {"train_sec": round(train_time, 1), "loss_last": losses[-1] if losses else None,
                            "loss_first": losses[0] if losses else None,
                            "loss_curve": [round(x, 4) for x in losses],
                            "loss_parts_ep0": {k: round(v, 4) for k, v in parts_seen.items()}}


KS = (20, 50, 100)          # 标准 PredCls 的 K 取值（SOTA 表统一报 R@20/50/100 与 mR@20/50/100）

# 聚合与打印用的指标键（顺序即报告顺序）
_METRIC_KEYS = tuple(
    [f"R@{k}" for k in KS] + [f"mR@{k}" for k in KS] + ["top1", "BRC@1", "IR@20", "IR@50", "MIR@50"])

# ======================================================================
# 方向性专用指标（B 组）的逆谓词表
# ======================================================================
def load_data_driven_groups(dataset: str):
    """载入**标注数据驱动**的逆谓词组（修复 P0）。

    产物由 `tools/build_inv_pairs_from_data.py` 生成：只保留标注中有**互反证据**的
    2-循环（命中率 ≥ min_rate 且互反检验通过），其余谓词不配对 ⟹ 走式(17) 的 φ 出口。

    相比手工语义组的优势（实测）：
      · VRD：手工 30 组中仅 5 组在标注里站得住（有命中 5 / 无命中 25）；
             数据驱动只认 4 对（on↔wear、above↔below、behind↔in the front of、
             on the left of↔on the right of），命中率 11%–45%，且 J²=I 偏差 0.00e+00。
      · 关键：无法确认有逆的谓词**不再退化为 Inv(r)=r**。退化为不动点会让式(23)
        变成「要求 p_ji ≈ p_ij」（强制对称），与式(16)「93.3% 非对称」矛盾；
        走 φ 出口才是式(15) 的正确语义（"该谓词在 R 中没有逆"）。

    返回 [(A, B), ...]（组对组形式），失败返回 None。
    """
    import json as _json
    p = os.path.join(ART, f"inv_pairs_{dataset}.json")
    if not os.path.exists(p):
        return None
    try:
        d = _json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    cycles = d.get("cycles") or []
    groups = [([a], [b]) for a, b in cycles]
    n_inv = 2 * len(cycles)
    K = len(d.get("predicates") or []) or (n_inv + len(d.get("no_inverse") or []))
    print(f"[inv] 数据驱动逆谓词（{dataset}）：{len(cycles)} 对 2-循环，"
          f"覆盖 {n_inv}/{K} 谓词（{n_inv/K*100:.1f}%）；"
          f"对称类 {len(d.get('symmetric') or [])} 个走式(24)，"
          f"无定义逆 {len(d.get('no_inverse') or [])} 个走 φ 出口")
    return groups


# ⚠️ 口径说明（必须写进论文）：
#   VG-50 中**大量谓词的逻辑逆不在 50 类内**（如 `ridden by`/`worn by`/`used by`
#   均非 VG-50 类），故 Inv 不能是 50 类的置换。此处只收录
#   **语义明确、且逆谓词也在 VG-50 内**的保守集合，用于 IR@K 与 BRC。
#   这是**下界式**评测：未收录的谓词不计入分母，故指标只反映可判定的子集。
#
# ⚠️ 用**组对组**映射而非逐对：旧版逐对定义里 `over → under` 与 `under → above`
#    冲突（`under` 被两条规则指向），导致 `to → from` 查反向时得到 `above` 的
#    错误结果（实测 1 处非对称）。组对组保证「组首 ↔ 组首」双向一致。
#
# ⚠️ 本组**仅 4 条**，覆盖率 16%（8/50 谓词有定义逆）——这正是 P0 问题所在：
#    其余 42 个谓词退化为不动点，式(23) 对它们退化为「强制对称」。
#    实验默认改用 `--inv-source data`（标注数据驱动）。
VG_INV_GROUPS = [
    (["above", "over"], ["under"]),          # 「在上方」族 ↔ 「在下方」
    (["behind"], ["in front of"]),
    (["covered in"], ["covering"]),
    (["from"], ["to"]),
]

# 兼容旧引用：逐对展开（组内成员都指向对侧组首），仅用于查阅/打印
INV_PAIR = {}
for _A, _B in VG_INV_GROUPS:
    for _X, _Y in ((_A, _B), (_B, _A)):
        for _p in _X:
            INV_PAIR[_p] = _Y[0]


class ImageDir:
    """按名读取**目录**中的图像（VRD 用；接口与 `ImageZip.read` 一致）。

    这样特征抽取代码 `extract_features` 无需区分数据源——它只调 `read(name)`
    拿原始字节。
    """

    def __init__(self, dirs: Sequence[str]):
        self.dirs = [d for d in dirs if d and os.path.isdir(d)]
        if not self.dirs:
            raise FileNotFoundError(f"图像目录不存在：{dirs}")
        self.index: Dict[str, str] = {}
        for d in self.dirs:
            for nm in os.listdir(d):
                if nm.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    self.index.setdefault(nm, os.path.join(d, nm))
                    base = os.path.splitext(nm)[0]
                    self.index.setdefault(base, os.path.join(d, nm))

    def read(self, name: str) -> Optional[bytes]:
        p = self.index.get(name) or self.index.get(os.path.basename(str(name)))
        if p is None or not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.index)


def get_dataset(name: str):
    """按名取数据集对象（接口一致：K / num_objects / object_classes /
    predicate_classes / load_split）。"""
    if name == "vrd":
        from vrp.vrd_data import VRD
        return VRD()
    if name == "gqa":
        from vrp.gqa_data import GQA
        return GQA()
    if name == "gqa_lt":
        # GQA-LT（复现划分）：根目录由 GQA_LT_ROOT 指定，默认 <VG_DATA_ROOT>/gqa_lt
        from vrp.gqa_data import GQA
        _lt = os.environ.get("GQA_LT_ROOT") or os.path.join(
            os.environ.get("VG_DATA_ROOT", ""), "gqa_lt")
        return GQA(root=_lt)
    if name == "unrel":
        from vrp.unrel_data import UnRel
        return UnRel()
    return VG150()


def get_image_reader(name: str, ds, test_split: str = "test"):
    """构造图像读取器：VG 走 zip，VRD/GQA/UnRel 走目录。"""
    if name in ("vrd", "gqa", "gqa_lt", "unrel"):
        dirs = [ds.image_dir("train"), ds.image_dir(test_split)]
        return ImageDir([d for d in dirs if d])
    root = os.environ["VG_DATA_ROOT"]
    return ImageZip([os.path.join(root, "raw", "vg_images", f)
                     for f in ("images.zip", "images2.zip")])


def semantic_file(name: str) -> str:
    if name == "vrd":
        return "vrd_semantic.pt"
    if name == "gqa":
        return "gqa_semantic.pt"
    if name == "gqa_lt":
        return "gqa_lt_semantic.pt"
    if name == "unrel":
        return "unrel_semantic.pt"
    return "semantic_path.pt"


def build_inv_idx(pred_names) -> np.ndarray:
    """谓词名列表（索引 0 = background）→ 逆谓词索引数组；无逆者为 -1。

    口径：**位置** = 1-based 谓词号（`pred_names` 的下标），
    **值** = 逻辑逆的 **0-based** 下标；读取端 `rp = int(inv_idx[p+1]) - 1`。
    ⚠️ 早期版本把**值**也存成 `pred_names.index(q)`（1-based 位置），
    读取端再减 1 ⟹ **整体偏移一位**，实测得到
    `above → using`、`behind → laying on`、`covering → covering`（自指）
    等完全错误的映射。两处口径必须明确区分。

    数据源用组对组的 `VG_INV_GROUPS`：组内成员都指向对侧**组首**，
    故「多对一」；对称性只要求**组首 ↔ 组首**双向一致。
    """
    inv = np.full(len(pred_names), -1, dtype=int)
    for grp_a, grp_b in VG_INV_GROUPS:
        for A, B in ((grp_a, grp_b), (grp_b, grp_a)):
            if not A or not B or A[0] not in pred_names or B[0] not in pred_names:
                continue
            # ⚠️ 存 **0-based 下标**（读取端会再减 1 还原成 1-based 谓词号）
            tgt = pred_names.index(B[0]) - 1
            for p in A:
                if p in pred_names:
                    inv[pred_names.index(p)] = tgt
    # ---------- 自检：不得自指；组首之间必须双向对称 ----------
    heads = set()
    for ga, gb in VG_INV_GROUPS:
        if ga and gb and ga[0] in pred_names and gb[0] in pred_names:
            heads.add((ga[0], gb[0]))
    for i in range(1, len(pred_names)):
        # ⚠️ `inv` 的**值**是 0-based 下标，而循环变量 i 是 1-based 位置，
        #    故自指判定必须比较 inv[i] == i - 1。
        if inv[i] == i - 1:
            raise AssertionError(f"逆谓词表自指：{pred_names[i]!r}")
        if inv[i] >= 0 and not (0 <= inv[i] < len(pred_names)):
            raise AssertionError(f"逆谓词值域越界：{pred_names[i]!r} -> {inv[i]}")
    for a, b in heads:
        ia, ib = pred_names.index(a), pred_names.index(b)
        # 值是 0-based 下标，故期望值也要 -1
        if inv[ia] != ib - 1 or inv[ib] != ia - 1:
            raise AssertionError(
                f"逆谓词组首非对称：{a!r}->{pred_names[inv[ia]+1]!r} "
                f"(期望 {b!r})，{b!r}->{pred_names[inv[ib]+1]!r} (期望 {a!r})")
    # ---- 链式对合自检：σ² = id ----
    # 组对组映射是「多对一」：组内成员都指向对侧**组首**，故对**组首**要求
    # σ²=id；对组内非首成员，正确性质是「σ² 回到本组组首」（两步收敛）。
    # 例：over 属「在上方」族 [above, over]，σ(over)=under，σ(under)=above
    #   —— 回到本组组首 above，是正确的多对一链，**不是**错误。
    n_chain = 0
    group_of = {}
    for gi, (ga, gb) in enumerate(VG_INV_GROUPS):
        for p in (ga or []) + (gb or []):
            if p in pred_names:
                group_of[pred_names.index(p)] = (gi, p in (ga or []))
    for k in range(1, len(pred_names)):
        q = inv[k]
        if q < 0:
            continue
        back = inv[q + 1]                 # σ² 的 0-based 值
        back1 = back + 1                  # 转成 1-based 位置便于比较
        if back1 == k:
            continue
        # 非组首成员：允许 σ² 回到本组组首
        gk = group_of.get(k)
        if gk is not None:
            gi, is_head_member = gk
            ga, gb = VG_INV_GROUPS[gi]
            grp_head = (ga if is_head_member else gb)[0]
            if grp_head in pred_names and back1 == pred_names.index(grp_head):
                continue
        n_chain += 1
    if n_chain:
        raise AssertionError(
            f"逆谓词链不闭合：{n_chain} 个谓词的 k→q→q' 既未回到 k、也未回到本组组首，"
            f"违反对合公理 σ²=id。检查存值口径（应存 0-based 下标）"
            f"与 VG_INV_GROUPS 是否存在抢占。")
    return inv


def _new_acc(K: int) -> dict:
    """评测累加器。所有量都是可加的，故可分块累积（见 evaluate_streaming）。"""
    return {"hit": {k: 0.0 for k in KS},
            "per_hit": {k: np.zeros(K) for k in KS},
            "per_cnt": np.zeros(K),
            "tot": 0.0, "top1": 0.0,
            "n_rel": 0, "n_img": 0,
            # --- B 组：方向性专用 ---
            # BRC：双向关系分类。以「无序对」为一组，要求**两个方向都预测正确**
            "brc_correct": 0.0, "brc_total": 0.0,
            # IR@K：互反召回。对 (s,o) 的真值 r，检查其**逻辑逆 r'** 在反转对 (o,s) 上的秩
            "ir_hit": {k: 0.0 for k in KS},
            "ir_per_hit": {k: np.zeros(K) for k in KS},
            "ir_per_cnt": np.zeros(K),
            "ir_tot": 0.0, "n_inv_pred": 0}


@torch.no_grad()
def _score_image(model, obj_emb, data: SplitData, i: int, dev,
                 eval_mode: str = "predcls"):
    """对第 i 张图产出候选打分与候选对清单。

    **两种候选集口径（关键，决定能否与 SOTA 对比）**：
    - `predcls`（标准 PredCls，与 Neural Motifs / RePEAT / Motifs 系一致）：
      候选 = 图内**所有有序物体对** × K 谓词。12.3 个物体 ⟹ 约 139 对、6950 候选。
    - `gtpairs`（本实现早期口径，**偏易**）：
      候选 = **仅标注中出现的实体对** × K 谓词。平均仅 6.6 对、330 候选。
      ⟹ 在只有 6.6 个候选对里选谓词，任务被大幅简化，**数值会显著虚高**。
    """
    if eval_mode == "gtpairs":
        pairs = [(s, o) for (s, o, _) in data.rels[i]]
    else:
        n = data.n_obj[i]
        pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if not pairs:
        return None, []
    S = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    O = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev).unsqueeze(0)
    xt = obj_emb(data.t[i].unsqueeze(0).to(dev)) if obj_emb is not None else \
        data.t[i].unsqueeze(0).to(dev)
    x = {"v": data.v[i].unsqueeze(0).to(dev),
         "t": xt,
         "s": data.s[i].unsqueeze(0).to(dev)}
    masks = data.masks[i].unsqueeze(0).to(dev)
    p_ij, _ = model.forward_inference(x, masks, S, O)
    return p_ij[0], pairs


@torch.no_grad()
def _acc_chunk(model, obj_emb, data: SplitData, hp: HyperParams,
               prior, acc: dict,
               eval_mode: str = "predcls",
               inv_idx: Optional[np.ndarray] = None,
               pred_keep: Optional[torch.Tensor] = None,
               skip_preds: Optional[set] = None) -> None:
    """把一个 SplitData 分块的评测结果累加进 acc。

    `pred_keep`：可选的 (K,) bool 掩码，只在该谓词子集内排序。
        用于**跨数据集零样本评测**（如 VRD→UnRel），或**排除平凡谓词**后的判别视图
        （如 GQA 排除 left/right，见 `--exclude-preds`）。
        `None` = 全词表排序（严格口径）。
    `skip_preds`：可选的 0-based 谓词集合，**GT 中出现这些谓词的关系不计入分母**。
        与 `pred_keep` 配合即得「排除 X 子集」的完整口径（既不计该谓词的 GT，
        也不允许它进入排序）。
    `prior`：普通 (K,) 数组 = 频率先验；**可调用对象**时按
        `prior(data, i, cand_pairs) -> (n_pairs, K)` 现场产出分数，
        用于无模型的几何基线等。
    """
    if model is not None:
        model.eval()
    dev = next(model.parameters()).device if model is not None else torch.device("cpu")
    K = hp.K
    acc["n_img"] += len(data.ids)

    for i in range(len(data.ids)):
        gt = data.rels[i]                       # (s,o,pred0)
        if not gt:
            continue
        acc["n_rel"] += len(gt)

        if callable(prior):
            # 无模型基线：由调用方按 (data, i, cand_pairs) 现场产出分数
            cand_pairs = ([(s2, o2) for (s2, o2, _) in gt] if eval_mode == "gtpairs"
                          else [(a, b) for a in range(data.n_obj[i])
                                for b in range(data.n_obj[i]) if a != b])
            if not cand_pairs:
                continue
            sc = np.asarray(prior(data, i, cand_pairs), dtype=np.float32)
            scores = torch.tensor(sc, dtype=torch.float32, device=dev)
        elif prior is not None:
            cand_pairs = ([(s2, o2) for (s2, o2, _) in gt] if eval_mode == "gtpairs"
                          else [(a, b) for a in range(data.n_obj[i])
                                for b in range(data.n_obj[i]) if a != b])
            scores = torch.tensor(np.tile(prior, (len(cand_pairs), 1)),
                                  dtype=torch.float32, device=dev)
        else:
            scores, cand_pairs = _score_image(model, obj_emb, data, i, dev,
                                              eval_mode=eval_mode)
            if scores is None:
                continue

        if pred_keep is not None:
            # 子集外谓词压到 -inf ⟹ 不参与排名（目标域词表限定 / 排除平凡谓词）
            scores = scores.masked_fill(~pred_keep.unsqueeze(0).to(scores.device), -1e9)

        # 2026-10-21 路线 B：pair 级存在性校准
        #   s_ij[k] = z_ij[k] + α · LSE(z_ij)
        # α=0 等价于纯 logits（已去掉 per-pair 归一化）；
        # α=1 等价于 log-softmax（完全恢复 pair 置信度）；
        # 0<α<1 为折中：既保留跨 pair 可比性，又利用「这一对整体有多确定」。
        # LSE 沿谓词维（dim=-1）取，得到每对一个标量，再广播回 (P,K)。
        if SCORE_LSE_ALPHA > 0:
            _lse = torch.logsumexp(scores, dim=-1, keepdim=True)   # (P,1) free energy
            scores = scores + SCORE_LSE_ALPHA * _lse

        # 2026-10-21 Stage-2：**pair 内重排序**（真正改变 pair 内次序）
        #   learn —— 学习式残差头（vrp.rerank.RelReranker），Δ 逐谓词不同
        #   calib —— 先验校准 Δ_k = -τ·log π_k（长尾 logit adjustment，对照用）
        if RERANKER is not None:
            from vrp.rerank import rerank_inplace as _rerank_inplace
            scores = _rerank_inplace(RERANKER, scores, data, i, RERANK_M_TOP,
                                     RERANK_TXT.to(scores.device), RERANK_INV)
        elif RERANK_CALIB is not None:
            scores = torch.log(scores.clamp_min(1e-9)) + RERANK_CALIB.to(scores.device)
        pair_pos = {p: k for k, p in enumerate(cand_pairs)}
        if RANK_SCOPE == "pair":
            _ord = torch.argsort(-scores, dim=-1, stable=True)      # (P,K)
            _rk = torch.empty_like(_ord)
            _rk.scatter_(1, _ord,
                         torch.arange(K, device=scores.device).expand_as(_ord))
            rank_of = _rk.reshape(-1)
        else:
            flat = scores.reshape(-1)               # 候选 = 候选对 × K 谓词
            order = torch.argsort(-flat, stable=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(len(order), device=order.device)
        for (s, o, p) in gt:
            if skip_preds and p in skip_preds:
                continue                        # 排除谓词的 GT 不计入分母
            j = pair_pos.get((s, o))
            if j is None:
                continue                        # 该真值对不在候选集内（仅 gtpairs 口径可能）
            r = int(rank_of[j * K + p]) + 1
            acc["tot"] += 1
            for k in KS:
                if r <= k:
                    acc["hit"][k] += 1
                    acc["per_hit"][k][p] += 1
            acc["per_cnt"][p] += 1
            acc["top1"] += float(int(torch.argmax(scores[j])) == p)

        # ---------- B 组：方向性指标（需反转对也在候选集内 ⟹ 仅 predcls 口径）----------
        if eval_mode != "predcls" or inv_idx is None:
            continue
        gt_map = {(s, o): p for (s, o, p) in gt}    # 同对取首个谓词，与 build_split 一致
        if skip_preds:
            gt_map = {k: v for k, v in gt_map.items() if v not in skip_preds}

        # BRC：以无序对 {a,b} 为组，要求两个方向都预测正确
        seen = set()
        for (s, o), p in gt_map.items():
            key = (min(s, o), max(s, o))
            if key in seen:
                continue
            seen.add(key)
            q = gt_map.get((o, s))
            if q is None:
                continue                            # 非双向对，不计入 BRC 分母
            js, jo = pair_pos.get((s, o)), pair_pos.get((o, s))
            if js is None or jo is None:
                continue
            acc["brc_total"] += 1
            if (int(torch.argmax(scores[js])) == p) and (int(torch.argmax(scores[jo])) == q):
                acc["brc_correct"] += 1

        # IR@K：真值 r 的逻辑逆 r' 在反转对 (o,s) 上的秩
        for (s, o), p in gt_map.items():
            rp = int(inv_idx[p + 1]) - 1            # inv_idx 按 1-based 谓词号建表，转 0-based
            if rp < 0:
                continue                            # 该谓词的逆不在 VG-50 内 ⟹ 不可判定，不计入分母
            jo = pair_pos.get((o, s))
            if jo is None:
                continue
            rr = int(rank_of[jo * K + rp]) + 1
            acc["ir_tot"] += 1
            for k in KS:
                if rr <= k:
                    acc["ir_hit"][k] += 1
                    acc["ir_per_hit"][k][rp] += 1
            acc["ir_per_cnt"][rp] += 1


def _finalize(acc: dict) -> dict:
    hit, tot = acc["hit"], acc["tot"]
    present = acc["per_cnt"] > 0
    out = {
        "top1": round(acc["top1"] / max(tot, 1), 4),
        "n_gt_triplets": int(tot),
        "avg_rel_per_img": round(acc["n_rel"] / max(acc["n_img"], 1), 3),
        "n_pred_present": int(present.sum()),
    }
    for k in KS:
        out[f"R@{k}"] = round(hit[k] / max(tot, 1), 4)
        if present.any():
            v = acc["per_hit"][k][present] / acc["per_cnt"][present]
            out[f"mR@{k}"] = round(float(v.mean()), 4)
        else:
            out[f"mR@{k}"] = 0.0

    # ---------- B 组：方向性指标 ----------
    out["BRC@1"] = round(acc["brc_correct"] / max(acc["brc_total"], 1), 4)
    out["n_brc_pairs"] = int(acc["brc_total"])
    for k in KS:
        out[f"IR@{k}"] = round(acc["ir_hit"][k] / max(acc["ir_tot"], 1), 4)
    irp = acc["ir_per_cnt"] > 0
    out["MIR@50"] = (round(float((acc["ir_per_hit"][50][irp] / acc["ir_per_cnt"][irp]).mean()), 4)
                     if irp.any() else 0.0)
    out["n_ir_triples"] = int(acc["ir_tot"])
    out["n_ir_preds"] = int(irp.sum())
    return out


@torch.no_grad()
def evaluate(model, obj_emb, data: SplitData, hp: HyperParams,
             prior=None, eval_mode: str = "predcls",
             inv_idx: Optional[np.ndarray] = None,
             pred_keep: Optional[torch.Tensor] = None,
             skip_preds: Optional[set] = None) -> dict:
    acc = _new_acc(hp.K)
    _acc_chunk(model, obj_emb, data, hp, prior, acc, eval_mode=eval_mode,
               inv_idx=inv_idx, pred_keep=pred_keep, skip_preds=skip_preds)
    return _finalize(acc)


@torch.no_grad()
def evaluate_streaming(model, obj_emb, records: List, cache: Dict[str, dict], K: int,
                       hp: HyperParams, prior: Optional[np.ndarray] = None,
                       chunk: int = 2000, nmax: int = 80, pmax: int = 160,
                       eval_mode: str = "predcls",
                       inv_idx: Optional[np.ndarray] = None, log=print,
                       pred_keep: Optional[torch.Tensor] = None,
                       skip_preds: Optional[set] = None) -> dict:
    """分块流式评测：逐块装配 → 评测 → 释放，内存上界由 `chunk` 决定。

    ⚠️ 必要性：`SplitData` 把每张图都展开成定长张量常驻内存
    （N_max×512 + P_max×K），26,446 张 test 集约占 4 GB，
    而本机可用内存仅 4.6 GB。一次性装配会触发大量换页。
    累加器设计（`_new_acc`）保证分块结果与整体评测**逐位等价**。
    """
    acc = _new_acc(K)
    recs = [r for r in records if r.image_id in cache]
    n_chunk = max(1, math.ceil(len(recs) / chunk))
    for ci in range(n_chunk):
        part = recs[ci * chunk:(ci + 1) * chunk]
        sd = build_split(part, cache, K, nmax, pmax, strict=False)  # 评测侧恒 legacy
        _acc_chunk(model, obj_emb, sd, hp, prior, acc, eval_mode=eval_mode,
                   inv_idx=inv_idx, pred_keep=pred_keep, skip_preds=skip_preds)
        del sd
        if ci % 4 == 0 or ci == n_chunk - 1:
            log(f"    评测分块 {ci+1}/{n_chunk}（已覆盖 {acc['n_img']:,}/{len(recs):,} 图）")
    return _finalize(acc)


# ======================================================================
# 主流程
# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--test-n", type=int, default=150)
    ap.add_argument("--test-split", default="val", choices=["val", "test"],
                    help="评测划分。⚠️ SOTA 论文的 PredCls 指标报在 test 上，"
                         "要与 SOTA 对比须用 --test-split test（26,446 张）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--quick", action="store_true", help="100/50/3 的链路自检")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--calibrate-loss", action="store_true",
                    help="启用损失尺度校准：修复 coh_loss 缺失 softmax + EPS_G=1.0（量级校准）")
    ap.add_argument("--eps-g", type=float, default=None, help="覆盖 hp.EPS_G")
    ap.add_argument("--lam-cat-scale", type=float, default=1.0,
                    help="六项范畴约束(sym/coh/hom/spread/PI/ord)的 λ 统一放大倍率；"
                         "1.0=原始配置(实测合计仅占梯度0.83%%)，10/50/100 用于扫描约束强度")
    ap.add_argument("--sym-mask-mode", default="hard",
                    choices=["hard", "soft", "tiered"],
                    help="式(24)对称掩码口径：hard=只约束真正自反的谓词(默认，"
                         "λ扫描证明软掩码过宽会抹平方向信息)；soft=旧行为(全谓词"
                         "近邻连续权重，仅供消融对照)；tiered=硬掩码∪高阈值近邻")
    # ⚠️ 2026-10-18 修复：**default=True**。
    #    原为 `action="store_true"`（默认 False），于是**不显式传该开关时**
    #    `args.use_clip_text=False` ⟹ `load_semantic` 返回 None ⟹
    #    M_inv / J / c_dag 全部退化为**随机初始化**，
    #    即整条 CLIP 语义通路（式15/式23 的真实语义）**静默失效**。
    #    （式12 的 L_feat 已随蒸馏一并删除，故帮助文本同步更新。）
    ap.add_argument("--use-clip-text", dest="use_clip_text",
                    action="store_true", default=True,
                    help="启用 CLIP 语义通路：M_inv/c_dag 用真实 CLIP 文本语义"
                         "（需先运行 tools/build_semantic_path.py），x_t 用物体类文本嵌入。"
                         "默认开启；用 --no-clip-text 关闭（退化为随机语义，仅供消融）")
    ap.add_argument("--no-clip-text", dest="use_clip_text",
                    action="store_false",
                    help="关闭 CLIP 语义通路（消融用：M_inv/J/c_dag 退化为随机）")
    ap.add_argument("--use-adjoint", action="store_true",
                    help="用**严格对合**的 J_adj 替换式(15) 的 CLIP 型 M_inv。"
                         "实测 CLIP 余弦型 M_inv 的 top1 指示不满足对合"
                         "（VG 复原率 2.1%%、29 自指；VRD 1.3%%、57 自指），"
                         "修正后复原率 100%%，使式(23) 的「互反约束」名副其实")
    ap.add_argument("--inv-source", choices=["data", "manual"], default="data",
                    help="逆谓词来源：data=标注数据驱动（tools/build_inv_pairs_from_data.py "
                         "的产物，只保留标注中有互反证据的 2-循环，其余走式(17) 的 φ 出口）；"
                         "manual=手工语义组（VG 仅 4 组，覆盖率 16%%，其余谓词退化为不动点、"
                         "式(23) 退化为强制对称）。默认 data。")
    ap.add_argument("--rev-mode",
                    choices=["legacy", "sym_check", "stats", "cooc"],
                    default="legacy",
                    help="式(8) 反向弱标签口径。legacy=论文原式(8)（默认，逐位复现"
                         "已有结果）；sym_check=仅当反向标注确实等于该谓词的逻辑逆时"
                         "才计入可信监督；stats=按每谓词实测一致率给 μ（需先跑 "
                         "tools/estimate_rev_reliability.py）；"
                         "cooc=用反向共现统计 P(K|p,反向对) 作软目标"
                         "（需先跑 tools/build_rev_cooc.py）")
    ap.add_argument("--cooc-temp", type=float, default=1.0,
                    help="rev_mode=cooc 时共现分布的温度（1.0 = 原始频率）")
    ap.add_argument("--mu-present", type=float, default=1.0,
                    help="rev_mode!=legacy 时，可信反向标签的 μ")
    ap.add_argument("--mu-absent", type=float, default=0.0,
                    help="rev_mode!=legacy 时，无反向标注时的 μ"
                         "（0=丢弃；>0 则沿用『自反弱标签』并取该值）")
    ap.add_argument("--eval-mode", choices=["predcls", "gtpairs"], default="predcls",
                    help="候选集口径。predcls=标准 PredCls（候选=图内所有物体对，"
                         "与 SOTA 可比）；gtpairs=仅标注对（偏易，数值虚高，仅供内部对照）")
    ap.add_argument("--dataset", choices=["vg150", "vrd", "gqa", "gqa_lt", "unrel"], default="vg150",
                    help="vg150=Visual Genome 50 谓词（与 SOTA 同协议）；"
                         "vrd=Visual Relationship Detection 70 谓词"
                         "（方向性覆盖 33.5%% vs VG 的 6.05%%）；"
                         "gqa=GQA 场景图 100 谓词；unrel=UnRel 76 关系（组合性）")
    ap.add_argument("--feat-backbone", choices=["clip", "roi"], default="clip",
                    help="视觉骨干。clip=CLIP 物体裁剪（快，但丢失物体间上下文）；"
                         "roi=Faster R-CNN(ResNet50-FPN)+ROI-Align+双FC头（对标 SOTA 特征通路）")
    ap.add_argument("--roi-arch", default="vg_r101c4",
                    choices=["vg_r101c4", "resnext101_vg", "resnet50_vg",
                             "resnext101", "resnet50"],
                    help="ROI 骨干："
                         "vg_r101c4=Anderson 等在 VG 上训练的检测器（ResNet-101-C4，官方权重）；"
                         "resnext101_vg / resnet50_vg=我们自己在 VG 上微调的检测器"
                         "（tools/finetune_detector_vg.py，ResNeXt-101-32x8d-FPN 与 SOTA 同族）；"
                         "resnext101 / resnet50=同架构但未做 VG 微调，作为"
                         "「唯一变量=预训练数据」的配对对照")
    ap.add_argument("--roi-min-size", type=int, default=800,
                    help="ROI 骨干：短边缩放到该尺寸（SOTA 用 800）")
    ap.add_argument("--roi-max-size", type=int, default=1333,
                    help="ROI 骨干：长边上限（SOTA 用 1333；调小可省显存/提速）")
    ap.add_argument("--roi-batch", type=int, default=4,
                    help="ROI 编码批大小（800×1333 下显存开销大；OOM 时自动对半拆分）")
    ap.add_argument("--roi-amp", dest="roi_amp", action="store_true", default=True,
                    help="ROI 抽取启用 fp16 自动混合精度（默认开；T4 实测 1.27→5.16 图/s）")
    ap.add_argument("--no-roi-amp", dest="roi_amp", action="store_false",
                    help="关闭 ROI 的 AMP（fp32 抽取，仅用于复现/排障）")
    ap.add_argument("--device", default=None,
                    help="运行设备：cuda / cpu / cuda:0。不填则自动（CUDA 可用即用）")
    ap.add_argument("--nmax", type=int, default=80, help="每图最大物体数（张量 N 维上限）")
    ap.add_argument("--pmax", type=int, default=64, help="每图最大关系对数（张量 P 维上限）")
    ap.add_argument("--progress-dir", default=None,
                    help="逐 (arm,seed) 结果落盘目录，用于断点续跑（默认 artifacts/progress_{dataset}）")
    ap.add_argument("--progress-rebuild", action="store_true",
                    help="忽略已有 (arm,seed) 结果，强制重训")
    ap.add_argument("--score-logits", action="store_true",
                    help="【已为默认，保留以兼容旧脚本】推理打分用 logits 而非 softmax 概率")
    ap.add_argument("--score-softmax", action="store_true",
                    help="回退到 per-pair softmax 打分（历史口径）。默认已是 logits，"
                         "该开关仅用于复现旧数字或做口径对照")
    ap.add_argument("--pair-res", action="store_true",
                    help="启用 **MLP 式逐对旁路**（借鉴最强基线 MLPPairClassifier）："
                         "只用 pair 端点原始视觉/文本/几何特征、不含图级上下文，"
                         "输出 (b,P,K) logits 作为残差加到双流上。结构分支与其全部"
                         "范畴约束保留。用于补上模型缺失的「pair 级可置信度」")
    ap.add_argument("--pair-res-hid", type=int, default=512,
                    help="逐对旁路的隐层宽度（默认 512，参数约 1.33M @K=100）")
    ap.add_argument("--pair-rel", action="store_true",
                    help="启用 **pair 存在性监督**（同 batch 内负 pair 采样 + BCE）。需配合 "
                         "--pair-res。推理时 z += β·log σ(rel)，为 per-pair 常数平移 ⟹ "
                         "top1 不变、全局 R@K 可动。针对 AUC 实测「pair 置信度无判别力(0.49)」")
    ap.add_argument("--pair-rel-beta", type=float, default=1.0,
                    help="推理期 pair 校准强度 β")
    ap.add_argument("--pair-rel-lambda", type=float, default=0.5,
                    help="训练期 pair 存在性损失权重 λ")
    ap.add_argument("--pair-rel-neg-ratio", type=int, default=3,
                    help="负样本比例：每图从非 GT 有序对中均匀采样 ratio×|GT| 个（默认 3）")
    ap.add_argument("--pair-rel-beta-scan", default="",
                    help="β 敏感性扫描，逗号分隔如 '0,0.5,1.0,2.0'。**同一训练权重**上"
                         "多次评测，用于判断 R@K 是否被 rel 项主导（β=0 退化为无校准）")
    ap.add_argument("--score-lse-alpha", type=float, default=0.0,
                    help="路线B：s = z + α·LSE(z)。0=纯 logits（默认）；1=log-softmax；"
                         "0<α<1 为折中。需配合 --score-logits 使用")
    ap.add_argument("--rank-scope", choices=["global", "pair"], default="global",
                    help="R@K 的排名口径。global=历史口径（候选对×谓词全局排序，"
                         "pair 置信度参与排名）；pair=标准 PredCls（每个 (s,o) 对内部排序，"
                         "α·LSE 等 per-pair 常数平移对其恒等）。默认 global 保持历史可比。")
    ap.add_argument("--rerank-mode", choices=["none", "learn", "calib"], default="none",
                    help="Stage-2 **pair 内重排序**（唯一能改变 pair 内次序的模块）。"
                         "learn=学习式残差头（冻结 Stage-1，候选集 CE）；"
                         "calib=先验校准 Δ_k=-τ·log π_k（长尾 logit adjustment，对照）。"
                         "注意：α·LSE 之类 per-pair 常数平移对本模块之外的排序恒等。")
    ap.add_argument("--rerank-topm", type=int, default=20,
                    help="重排序候选集大小 M：只对 Stage-1 的 top-M 谓词施加残差")
    ap.add_argument("--rerank-epochs", type=int, default=8)
    ap.add_argument("--rerank-lr", type=float, default=2e-3)
    ap.add_argument("--rerank-dim", type=int, default=192)
    ap.add_argument("--rerank-tau", type=float, default=1.0,
                    help="calib 模式的 τ：Δ_k = -τ·log π_k（τ=0 退化为不校准）")
    ap.add_argument("--rerank-baseline-first", action="store_true",
                    help="重排序前先用**同一 Stage-1 权重**评一遍基线（严格同权重对照，"
                         "避免跨进程 cuDNN 非确定性污染归因）")
    ap.add_argument("--diag-path-auc", action="store_true",
                    help="训练后诊断「三角路径（隐式流）一致性」的 pair 判别力（AUC）。"
                         "用于判断是否值得做 pair 级校准，不改变任何评测结果")
    ap.add_argument("--diag-path-max-img", type=int, default=0,
                    help="诊断最多用多少张测试图（0=全部）")
    ap.add_argument("--rerank-rev", type=int, default=1, choices=[0, 1],
                    help="重排序头是否输入「(o,s) 方向逆谓词分数 + 有无逆 mask」（A 方案）。"
                         "1=启用（默认，用于恢复 IR@k）；0=关闭（消融对照）")
    ap.add_argument("--allow-trunc", action="store_true",
                    help="允许静默截断超出 --nmax/--pmax 的物体/关系（默认禁止）")
    ap.add_argument("--train-chunk", type=int, default=0,
                    help="训练集分块流式装配：每块图片数（0=关闭，一次性全量装配）。"
                         "GQA 全量必须开启（72,581 图全量 padding 需 ~10GB RAM）")
    ap.add_argument("--chunk-dir", default=None,
                    help="分块缓存目录（默认 artifacts/chunks_{dataset}_tr{train_n}_c{chunk}）")
    ap.add_argument("--chunk-rebuild", action="store_true",
                    help="忽略已存在的分块缓存，强制重建（改了 nmax/pmax/rev_mode 时必须加）")
    ap.add_argument("--chunk-fp16", action="store_true",
                    help="分块缓存用半精度存储（磁盘减半；载入时还原 f32，训练数值不变）")
    ap.add_argument("--out", default=os.path.join(ART, "small_sample_log.json"))
    ap.add_argument("--save-ckpt", default=None,
                    help="保存每个 arm×seed 的模型权重到该目录（供跨数据集零样本评测，"
                         "如 VRD→UnRel）。默认不保存。")
    ap.add_argument("--exclude-preds", default=None,
                    help="逗号分隔的谓词名，评测时**同时**从 GT 与排序候选中排除，得到"
                         "「判别视图」。GQA 建议传 "
                         "'to the left of,to the right of'（其标签与框几何 100%% 一致，"
                         "不排除则 R@K 被平凡解支配）。结果另存 metrics_excl，不改全量口径。")
    args = ap.parse_args()

    # 2026-10-21：推理打分**默认改为未归一化 logits**（与 SOTA 基线同口径）。
    # 全局 P×K 排序要求跨 pair 可比，softmax 按 pair 归一化会破坏可比性
    # （实测同权重下 R@50 被低估 14.5%，而 top1 不变）。
    # `--score-softmax` 回退到历史口径，仅用于复现旧数字 / 做口径对照。
    import vrp.model as _vmodel
    if getattr(args, "score_softmax", False):
        _vmodel.SCORE_LOGITS = False
        print("[score] 打分口径 = softmax（历史口径，仅用于复现/对照）", flush=True)
    else:
        _vmodel.SCORE_LOGITS = True
        print("[score] 打分口径 = logits（与 SOTA 基线一致，默认）", flush=True)

    if getattr(args, "pair_res", False):
        _vmodel.USE_PAIR_RES = True
        _vmodel.PAIR_RES_HID = int(args.pair_res_hid)
        print("[pair-res] 启用 MLP 式逐对旁路（无图级上下文）hid=%d"
              % args.pair_res_hid, flush=True)
        if getattr(args, "pair_rel", False):
            _vmodel.PAIR_REL = True
            _vmodel.PAIR_REL_BETA = float(args.pair_rel_beta)
            _vmodel.PAIR_REL_LAMBDA = float(args.pair_rel_lambda)
            _vmodel.PAIR_REL_NEG_RATIO = int(args.pair_rel_neg_ratio)
            print("[pair-rel] 启用 pair 存在性监督：λ=%.2f β=%.2f 负采样 1:%d"
                  % (args.pair_rel_lambda, args.pair_rel_beta,
                     args.pair_rel_neg_ratio), flush=True)

    if getattr(args, "score_lse_alpha", 0.0):
        # 路线 B：在 logits 上加回 α·LSE(z)，α 由验证集选择（防泄漏）
        globals()["SCORE_LSE_ALPHA"] = float(args.score_lse_alpha)
        print("[score] LSE alpha = %.3f" % args.score_lse_alpha, flush=True)

    if getattr(args, "rank_scope", "global") != "global":
        globals()["RANK_SCOPE"] = args.rank_scope
        print("[rank] 排名口径 = %s" % args.rank_scope, flush=True)

    if getattr(args, "rerank_mode", "none") != "none":
        globals()["RERANK_M_TOP"] = int(args.rerank_topm)
        print("[rerank] Stage-2 重排序模式 = %s（top-%d）"
              % (args.rerank_mode, args.rerank_topm), flush=True)

    if args.quick:
        args.train_n, args.test_n, args.epochs = 100, 50, 3
        args.seeds = "0"

    if False:  # args.q_mode != "prior" and not args.use_clip_text
        args.use_clip_text = True          # q 的 CLIP 模式依赖语义通路

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    os.makedirs(ART, exist_ok=True)

    log = print
    t_start = time.time()
    log("=" * 78)
    log(f"VG-150 小样本对比实验 | train_n={args.train_n} test_n={args.test_n} "
        f"test_split={args.test_split} epochs={args.epochs} seeds={seeds} arms={arm_names}")
    log("=" * 78)
    log(f"  语义通路: {'CLIP 文本塔（式15/23 真实语义）' if args.use_clip_text else '随机 v_txt/v_inv（无语义，上一轮口径）'}")
    # log(f"  q 来源  : {args.q_mode}")  # 已删 KL
    log(f"  式(15) M_inv : {'**严格对合 J_adj**（--use-adjoint）' if args.use_adjoint else 'CLIP 余弦 M_inv（原式）'}")
    _dev = get_device(args.device)
    log(f"  运行设备: {_dev}" + (f"  ({torch.cuda.get_device_name(_dev)})"
                                if _dev.type == "cuda" else "  ⚠️ 无 CUDA 可用"))

    ds = get_dataset(args.dataset)
    K = ds.K
    hp = HyperParams()
    hp.K = K
    # ---- 范畴约束 λ 统一放大（2026-10-19）----
    # 动机：实测六项合计仅占梯度 0.83%（cls 独占 85.2%），约束形不成可测增益。
    # 本开关把 sym/coh/hom/spread/PI/ord 的 λ 乘以同一倍率，其余项不变，
    # 用于扫描「约束需多大强度才能在指标上显形」。
    if getattr(args, "lam_cat_scale", 1.0) != 1.0:
        s = float(args.lam_cat_scale)
        for _k in ("LAMBDA_SYM", "LAMBDA_COH", "LAMBDA_H",
                   "LAMBDA_S", "LAMBDA_PI", "LAMBDA_ORD"):
            setattr(hp, _k, getattr(hp, _k) * s)
        log(f"[范畴约束 λ 放大] scale=×{s:g} → "
            f"sym={hp.LAMBDA_SYM:g} coh={hp.LAMBDA_COH:g} hom={hp.LAMBDA_H:g} "
            f"spread={hp.LAMBDA_S:g} PI={hp.LAMBDA_PI:g} ord={hp.LAMBDA_ORD:g}")
    if args.calibrate_loss:
        info = apply_loss_fixes(normalize_maha=True, fix_coh=True)
        hp.EPS_G = 1.0
        log(f"[损失尺度校准] {info} 且 EPS_G=1.0")
    if args.eps_g is not None:
        hp.EPS_G = args.eps_g
    log(f"  hp.EPS_G={hp.EPS_G}")
    train_recs = ds.load_split("train", limit=args.train_n)
    test_recs = ds.load_split(args.test_split, limit=args.test_n)

    # 谓词频率先验（训练集）
    cnt = np.zeros(K)
    for r in train_recs:
        for rel in r.relations:
            if 1 <= rel.predicate <= K:
                cnt[rel.predicate - 1] += 1
    prior = (cnt + 1.0) / (cnt.sum() + K)
    log(f"训练集关系数 {int(cnt.sum())} | 最频繁谓词 #{int(np.argmax(cnt))}="
        f"{ds.predicate_classes[int(np.argmax(cnt)) + 1]} freq={float(prior.max()):.4f}")

    imgzip = get_image_reader(args.dataset, ds, args.test_split)
    # 视觉骨干可选择：clip=CLIP 物体裁剪（快，但丢上下文）；roi=Faster R-CNN ROI（对标 SOTA）
    backbone = getattr(args, "feat_backbone", "clip")
    # ⚠️ 缓存名必须区分 arch：resnet50 与 resnext101 若共用文件名会互相覆盖，
    # 导致第二次运行静默复用错误的骨干特征（且不会报错）。
    _bb_tag = backbone if backbone == "clip" else f"{backbone}_{args.roi_arch}"
    _ds_tag = "" if getattr(args, "dataset", "vg150") == "vg150" \
        else f"_{args.dataset}"
    cache_path = os.path.join(
        ART, f"feat_cache_{_bb_tag}{_ds_tag}_tr{args.train_n}_te{args.test_n}.pt")
    if args.no_cache and os.path.exists(cache_path):
        os.remove(cache_path)
    all_recs = train_recs + test_recs
    if backbone == "roi":
        from roi_features import extract_roi_features
        log(f"  视觉骨干 = {args.roi_arch} + FPN + ROI-Align + 双FC头【对标 SOTA 特征通路】"
            f" 分辨率 {args.roi_min_size}/{args.roi_max_size}")
        cache = extract_roi_features(all_recs, imgzip, cache_path, log=log,
                                     batch_imgs=args.roi_batch,
                                     min_size=args.roi_min_size,
                                     max_size=args.roi_max_size,
                                     arch=args.roi_arch,
                                     amp=bool(getattr(args, "roi_amp", True)))
    else:
        log("  视觉骨干 = CLIP ViT-B/32 物体裁剪【快，但裁剪丢失上下文】")
        cache = extract_features(all_recs, cache_path, imgzip, log=log)

    # ---------- q：蒸馏教师软标签 ----------
    # ⚠️ 2026-10-18 删除 KL 蒸馏（式 11）：教师软标签 q 整体移除。
    # # prior  = 谓词频率先验（上一轮口径，纯统计、无语义）
    # # clip   = CLIP 图文相似度 softmax(cos(v_img, v_txt)/τ)  ← 语义通路
    # # both   = 二者几何平均
    # q_vec = prior.copy()
    # q_desc = "谓词频率先验(平滑)"
    # if args.q_mode in ("clip", "both"):
    # sem = load_semantic(True, sem_file=semantic_file(args.dataset))
    # img_cache_path = os.path.join(ART, f"imgfeat{_ds_tag}_tr{args.train_n}_te{args.test_n}.pt")
    # img_feats = extract_image_features(all_recs, imgzip, img_cache_path, log=log)
    # q_feat = sem["q_feat"]                                  # (50,512) L2 归一
    # tau = float(getattr(hp, "TAU", 4.0))
    # rows = []
    # for r in train_recs:
    # f = img_feats.get(r.image_id)
    # if f is None:
    # continue
    # logits = (torch.nn.functional.normalize(f, dim=-1) @ q_feat.t()) * tau
    # rows.append(torch.softmax(logits, dim=-1).numpy())
    # if rows:
    # q_clip = np.mean(np.stack(rows, 0), axis=0)
    # q_clip = q_clip / q_clip.sum()
    # q_vec = q_clip if args.q_mode == "clip" else np.sqrt(np.clip(prior * q_clip, 0, None))
    # q_vec = q_vec / q_vec.sum()
    # q_desc = ("CLIP 图文相似度 softmax(cos/τ)"
    # if args.q_mode == "clip" else "prior 与 CLIP 的几何平均")
    # log(f"  q[CLIP] top3 谓词: "
    # + ", ".join(f"{ds.predicate_classes[i+1]}={q_vec[i]:.4f}"
    #             f" for i in np.argsort(-q_vec)[:3])
    # log(f"  q[CLIP] 与频率先验的余弦 = "
    #     f"{float(q_vec @ prior / (np.linalg.norm(q_vec)*np.linalg.norm(prior))):.4f}")
    # args.q_prior = q_vec.tolist()
    # log(f"  q 来源 = {q_desc}")

    # ---------- 逆谓词索引（B 组方向性指标 + 式(8) 修正口径都要用） ----------
    # inv_idx[k] = 谓词 k 的逻辑逆下标，无逆为 -1。
    # ⚠️ 必须**在 build_split 之前**算好：rev_mode != legacy 时 build_split 需要它。
    if args.dataset == "vrd":
        _sv = torch.load(os.path.join(ART, "vrd_semantic.pt"), weights_only=False)
        # vrd_semantic.pt 里的 inv_idx 是 **Python list**（不是 tensor），
        # 直接 .tolist() 会 AttributeError。统一走 np.asarray。
        inv_idx = np.asarray(_sv["inv_idx"], dtype=np.int64)
    else:
        inv_idx = build_inv_idx(ds.predicate_classes)
    n_inv = int((inv_idx > 0).sum())
    log(f"  逆谓词表：{n_inv} 个谓词有可判定的逆（共 {K} 类），"
        f"其余谓词的逻辑逆不在本数据集谓词表内 ⟹ IR@K 为下界式评测")

    # 反向共现统计（rev_mode=cooc 时才需要）
    _cooc = _cooc_total = None
    if args.rev_mode == "cooc":
        _cp = os.path.join(ART, f"{args.dataset}_rev_cooc.json")
        if not os.path.exists(_cp):
            log(f"!! rev_mode=cooc 需要 {_cp}，请先运行 tools/build_rev_cooc.py")
            return 1
        _cj = json.load(open(_cp, encoding="utf-8"))
        _cooc = np.asarray(_cj["cnt"], dtype=np.float64)
        _cooc_total = np.asarray(_cj["total"], dtype=np.float64)
        log(f"式(8) rev_mode=cooc：加载 {int((_cooc.sum(1) > 0).sum())} 个谓词的"
            f"反向共现分布（共 {_cooc_total.sum():,} 条关系），温度 {args.cooc_temp}")

    # 逐谓词可靠度（rev_mode=stats 时才需要）
    _rel = {}
    if args.rev_mode == "stats":
        _rp = os.path.join(ART, f"{args.dataset}_rev_reliability.json")
        if not os.path.exists(_rp):
            log(f"!! rev_mode=stats 需要 {_rp}，请先运行 tools/estimate_rev_reliability.py")
            return 1
        _rel = {int(k): float(v)
                for k, v in json.load(open(_rp, encoding="utf-8"))["rel"].items()}
        log(f"式(8) rev_mode=stats：加载 {len(_rel)} 个谓词的实测一致率")

    _split_kw = dict(strict=not args.allow_trunc,
                     rev_mode=args.rev_mode,
                     mu_present=args.mu_present,
                     mu_absent=args.mu_absent,
                     prior_mode=("stats" if args.rev_mode == "stats" else "none"),
                     _rev_reliability=_rel,
                     _cooc=_cooc, _cooc_total=_cooc_total,
                     cooc_temp=args.cooc_temp)
    if args.rev_mode != "legacy":
        _split_kw["inv_of"] = inv_idx
        log(f"式(8) 口径 = {args.rev_mode}"
            f"（mu_present={args.mu_present}, mu_absent={args.mu_absent}）")
    _chunk_paths, _chunk_agg = None, None
    if getattr(args, "train_chunk", 0) > 0 and len(train_recs) > args.train_chunk:
        _cdir = args.chunk_dir or os.path.join(
            ART, f"chunks_{args.dataset}_tr{args.train_n}_c{args.train_chunk}")
        log(f"分块流式装配：每块 ≤{args.train_chunk} 图 → {_cdir}")
        _chunk_paths, tr, _chunk_agg = build_train_chunks(
            train_recs, cache, K, args.nmax, args.pmax, args.train_chunk, _cdir,
            _split_kw, log=log, rebuild=getattr(args, "chunk_rebuild", False),
            fp16=getattr(args, "chunk_fp16", False))
        log(f"装配完成：train {_chunk_agg['n_img']:,} 图 / {len(_chunk_paths)} 块"
            f"(N≤{tr.N_max},P≤{tr.P_max})")
    else:
        tr = build_split(train_recs, cache, K, args.nmax, args.pmax, **_split_kw)
        log(f"装配完成：train {len(tr.ids)} 图(N≤{tr.N_max},P≤{tr.P_max})")

    # ---------- 排除平凡谓词的「判别视图」（GQA 用：排除 to the left of / to the right of）----------
    # 依据：GQA 上左右关系的标签与框中心几何 100% 一致 ⟹ 全量 R@K 被平凡解支配。
    # `--exclude-preds` 同时做两件事：① 这些谓词的 GT 不计入分母；② 不允许它们参与排序。
    _excl_names = [s.strip() for s in (args.exclude_preds or "").split(",") if s.strip()]
    _excl_preds, _excl_mask = None, None
    if _excl_names:
        _name2i = {n: i for i, n in enumerate(ds.predicate_classes[1:])}
        _excl_preds = {_name2i[n] for n in _excl_names if n in _name2i}
        _missing = [n for n in _excl_names if n not in _name2i]
        if _missing:
            log(f"  [排除] ⚠ 谓词名不在 {args.dataset} 词表内，已忽略：{_missing}")
        if _excl_preds:
            _excl_mask = torch.ones(K, dtype=torch.bool)
            for _p in _excl_preds:
                _excl_mask[_p] = False
            log(f"  [排除] 判别视图：排除 {len(_excl_preds)} 个谓词 "
                f"{sorted(_excl_names)}；保留 {int(_excl_mask.sum())}/{K} 参与排序")

    # ---------- 测试集评测：大集合走分块流式（内存受控）----------
    STREAM_THRESHOLD = 3000
    if len(test_recs) > STREAM_THRESHOLD:
        te = None
        log(f"测试集 {len(test_recs):,} 图（split={args.test_split}）"
            f" → 启用**分块流式评测**（SplitData 常驻会占约 4 GB，本机可用内存不足）")

        def eval_test(model, obj_emb, prior=None):
            return evaluate_streaming(model, obj_emb, test_recs, cache, K, hp, prior,
                                      chunk=2000, nmax=args.nmax, pmax=args.pmax,
                                      eval_mode=args.eval_mode, inv_idx=inv_idx, log=log)

        def eval_test_excl(model, obj_emb):
            return evaluate_streaming(model, obj_emb, test_recs, cache, K, hp, None,
                                      chunk=2000, nmax=args.nmax, pmax=args.pmax,
                                      eval_mode=args.eval_mode, inv_idx=inv_idx, log=log,
                                      pred_keep=_excl_mask, skip_preds=_excl_preds)
    else:
        te = build_split(test_recs, cache, K, args.nmax, args.pmax, **_split_kw)
        log(f"装配完成：test {len(te.ids)} 图(N≤{te.N_max},P≤{te.P_max})")

        def eval_test(model, obj_emb, prior=None):
            return evaluate(model, obj_emb, te, hp, prior, eval_mode=args.eval_mode,
                            inv_idx=inv_idx)

        def eval_test_excl(model, obj_emb):
            return evaluate(model, obj_emb, te, hp, None, eval_mode=args.eval_mode,
                            inv_idx=inv_idx,
                            pred_keep=_excl_mask, skip_preds=_excl_preds)

    # 关系总数核对：确认没有静默丢数据
    # ⚠️ 流式评测模式下 `te is None`（测试集分块流式装配，无法一次性统计），
    #    此时只核对训练集。
    if _chunk_agg is not None:
        n_kept, n_raw = _chunk_agg["n_kept"], _chunk_agg["n_raw"]
        log(f"训练集关系数（分块聚合）：装配后 {n_kept:,} / 标注 {n_raw:,}"
            f"（保留率 {100.0 * n_kept / max(n_raw, 1):.4f}%）")
        _check = []
    else:
        _check = [("训练集", tr, train_recs)]
    if te is not None:
        _check.append(("测试集", te, test_recs))
    for tag, sd_, recs_ in _check:
        n_kept = int(sum(len(x) for x in sd_.rels))
        n_raw = sum(len(r.relations) for r in recs_ if r.image_id in cache)
        log(f"{tag}关系数：装配后 {n_kept} / 标注 {n_raw}"
            f"（保留率 {100.0 * n_kept / max(n_raw, 1):.4f}%）")
        # 保留率不足 100% 时必须区分两种原因，否则会把「数据性质」误判为「截断 bug」
        if n_kept < n_raw:
            n_oob = 0
            for r in recs_:
                if r.image_id not in cache:
                    continue
                nobj = len(r.labels)
                n_oob += sum(1 for rel in r.relations
                             if not (rel.subject < nobj and rel.object < nobj))
            n_cap = n_raw - n_kept - n_oob
            log(f"    └ 丢弃 {n_raw - n_kept} 条 = 引用了图中无框物体 {n_oob}"
                f" + 张量上限截断 {max(n_cap, 0)}")
            if n_oob > 0:
                log(f"    ⚠ 前者源于数据集本身：这些关系的主体/客体在 objects 表中"
                    f"只有类别、没有框，GT 框协议下**无法评测**（非截断 bug）")
    # 注：流式评测模式下 te 为 None（测试集分块流式装配），
    # 此时无法统计 test 的装配保留率；train 侧始终可统计。

    if len(tr.ids) == 0:
        log("!! 训练数据不足，终止"); return 1
    if te is not None and len(te.ids) == 0:
        log("!! 测试数据不足，终止"); return 1

    results = {"config": {
        "train_n": args.train_n, "test_n": args.test_n, "test_split": args.test_split,
        "epochs": args.epochs,
        "score_mode": ("softmax（历史口径）" if getattr(args, "score_softmax", False)
                       else "logits（与 SOTA 一致，默认）"),
        "rank_scope": getattr(args, "rank_scope", "global"),
        "batch_size": args.batch_size, "lr": args.lr, "seeds": seeds, "arms": arm_names,
        "clip": "ViT-B/32 (CPU 重建, 物体裁剪)",
        "semantic_path": ("CLIP 文本塔（式15/23 真实语义）" if args.use_clip_text
                          else "随机 v_txt/v_inv（无语义）"),
        "x_t": ("CLIP 物体类文本嵌入（冻结）" if args.use_clip_text
                else "nn.Embedding(151,512) 可学习"),
        "L_feat": "启用（式12）" if args.use_clip_text else "未启用",
        # "q": q_desc, "q_mode": args.q_mode,   # 已删 KL 蒸馏
        "device": str(_dev),
        "eval_mode": args.eval_mode,
        "feat_backbone": backbone,
        "roi_arch": (args.roi_arch if backbone == "roi" else None),
        "roi_resolution": (f"{args.roi_min_size}/{args.roi_max_size}"
                           if backbone == "roi" else None),
        "dataset": args.dataset, "n_pred": K,
        "n_obj": len(ds.object_classes),
        "geom_dim": GEOM_DIM, "pred_prior_top": int(np.argmax(cnt)),
    }, "arms": {}, "raw_runs": []}

    # 逐 (arm, seed) 结果落盘：进程被杀/机器重启后，重跑同一条命令即可跳过已完成部分
    _prog_dir = getattr(args, "progress_dir", None) or os.path.join(ART, f"progress_{args.dataset}")
    os.makedirs(_prog_dir, exist_ok=True)
    log(f"[断点续跑] 逐 seed 结果目录：{_prog_dir}")

    for aname in arm_names:
        arm = ARMS[aname]
        # ⚠️ 2026-10-18：J 臂（no_coh）改用 FULL_NCOH 损失集（隔离式28 的净效应）
        if getattr(arm, "no_coh", False):
            arm = replace(arm, loss="FULL_NCOH")
        log("-" * 78)
        log(f"臂 {arm.name}  (direction_aware={arm.direction_aware}, dual_head={arm.dual_head}, loss={arm.loss})")
        if arm.prior:
            m = eval_test(None, None, prior=prior)
            log(f"  [频率先验] " + json.dumps(m, ensure_ascii=False))
            rec = {"label": arm.name, "prior": True, "metrics": m}
            if _excl_mask is not None:
                m2 = eval_test_excl(None, None)
                rec["metrics_excl"] = m2
                log(f"  [频率先验·判别视图] mR@50={m2.get('mR@50')} R@50={m2.get('R@50')}")
            results["arms"][aname] = rec
            results["raw_runs"].append({"arm": aname, "seed": -1, "metrics": m, "train": {}})
            continue
        per_seed = []
        per_seed_excl = []
        for sd in seeds:
            _pf = os.path.join(_prog_dir, f"{aname}_s{sd}.json")
            if os.path.exists(_pf) and not getattr(args, "progress_rebuild", False):
                _rec = json.load(open(_pf, encoding="utf-8"))
                m = _rec["metrics"]
                m2 = _rec.get("metrics_excl")
                per_seed.append(m)
                if m2:
                    per_seed_excl.append(m2)
                    log(f"  seed={sd} [判别视图] （续跑载入）" + json.dumps(m2, ensure_ascii=False))
                log(f"  seed={sd} [续跑] arm {aname} seed={sd} 已有结果，跳过训练 | 训练 {_rec.get('train', {}).get('train_sec', '?')}s")
                results["raw_runs"].append({"arm": aname, "seed": sd, "metrics": m,
                                            "metrics_excl": m2,
                                            "train": _rec.get("train", {}),
                                            "n_params": _rec.get("n_params")})
                continue
            model, obj_emb, tinfo = train_arm(arm, hp, tr, args, sd, ds=ds,
                                              train_recs=train_recs, log=log,
                                              chunk_paths=_chunk_paths)
            # 保存权重：供跨数据集零样本评测（VRD→UnRel）复用同一训练产物
            if getattr(args, "save_ckpt", None):
                os.makedirs(args.save_ckpt, exist_ok=True)
                _cp = os.path.join(args.save_ckpt, f"{args.dataset}_{aname}_seed{sd}.pth")
                torch.save({
                    "model": model.state_dict(),
                    "obj_emb": obj_emb.state_dict(),
                    "arm": aname, "seed": sd, "dataset": args.dataset, "K": hp.K,
                }, _cp)
                log(f"  [ckpt] {_cp}")
            # ---------- 诊断：三角路径（隐式流）一致性的 pair 判别力 ----------
            if getattr(args, "diag_path_auc", False) and te is not None:
                try:
                    from vrp.pathdiag import path_auc_diag
                    _pa = path_auc_diag(model, obj_emb, te, hp, log=log,
                                        max_img=getattr(args, "diag_path_max_img", 0))
                    globals().setdefault("_PATH_AUC", {})[f"{aname}_{sd}"] = _pa
                except Exception as _e:                       # noqa: BLE001
                    log(f"  [path-auc] ⚠ 诊断失败（{type(_e).__name__}: {_e}）")

            # ---------- Stage-2：构造/训练 **pair 内重排序** 模块 ----------
            _rmode = getattr(args, "rerank_mode", "none")
            globals()["RERANKER"] = None
            globals()["RERANK_CALIB"] = None
            # 严格同权重对照：在重排序前先用**同一 Stage-1 权重**评一遍基线。
            # 必要性：跨进程重跑存在 cuDNN 非确定性（实测 |Δtop1| 可达 0.005），
            # 与重排序增益同量级，跨进程比较无法归因。
            if _rmode != "none" and getattr(args, "rerank_baseline_first", False):
                _mb = eval_test(model, obj_emb, prior=None)
                _mb2 = eval_test_excl(model, obj_emb) if _excl_mask is not None else None
                globals().setdefault("_RR_BASE", {})[f"{aname}_{sd}"] = {
                    "metrics": _mb, "metrics_excl": _mb2}
                log("  [rerank] 同权重基线：top1=%.4f mR@20=%.4f R@20=%.4f"
                    % (_mb.get("top1", 0), _mb.get("mR@20", 0), _mb.get("R@20", 0))
                    + (" | 判别视图 top1=%.4f mR@20=%.4f"
                       % (_mb2.get("top1", 0), _mb2.get("mR@20", 0)) if _mb2 else ""))
            if _rmode != "none":
                _sem_r = load_semantic(True, sem_file=semantic_file(args.dataset))
                _txt = (_sem_r or {}).get("v_txt")
                if _txt is None:
                    raise RuntimeError("rerank 需要谓词文本嵌入（v_txt），但语义文件缺失")
                if _rmode == "calib":
                    _c = np.zeros(hp.K, dtype=np.float64)
                    for _r in train_recs:
                        for _rel in _r.relations:
                            if 1 <= _rel.predicate <= hp.K:
                                _c[_rel.predicate - 1] += 1
                    _pi = (_c + 1.0) / (_c.sum() + hp.K)
                    _off = torch.tensor(-args.rerank_tau * np.log(_pi), dtype=torch.float32)
                    globals()["RERANK_CALIB"] = _off
                    log(f"  [rerank] 先验校准 τ={args.rerank_tau}：偏移 "
                        f"[{float(_off.min()):.2f}, {float(_off.max()):.2f}]（Δ_k=-τ·log π_k）")
                else:
                    from vrp.rerank import (RelReranker, collect_rerank_data,
                                            train_reranker)
                    _dev = get_device(getattr(args, "device", None))
                    _inv_t = torch.as_tensor(inv_idx, dtype=torch.long)
                    globals()["RERANK_INV"] = _inv_t
                    _head = RelReranker(d_v=FEAT_DIM, d_t=_txt.shape[-1],
                                        d=args.rerank_dim, gamma_init=2.0,
                                        use_rev=bool(getattr(args, "rerank_rev", 1))
                                        ).to(_dev)
                    _pack = collect_rerank_data(model, obj_emb, tr, args.rerank_topm,
                                                log=log,
                                                inv_of=(_inv_t if getattr(args, "rerank_rev", 1) else None))
                    _rng = np.random.RandomState(sd)
                    _idx = _rng.permutation(len(_pack["img"]))
                    _n_val = max(1, len(_idx) // 10)
                    _val = {k: v[_idx[:_n_val]] for k, v in _pack.items()}
                    _trn = {k: v[_idx[_n_val:]] for k, v in _pack.items()}
                    log(f"  [rerank] 训练 {len(_trn['img']):,} / 留出 {len(_val['img']):,}")
                    train_reranker(_head, _trn, tr, _txt, epochs=args.rerank_epochs,
                                   lr=args.rerank_lr, log=log)
                    from vrp.rerank import eval_reranker_pack as _erp
                    _vm = _erp(_head, _val, tr, _txt, _dev)
                    log(f"  [rerank] 留出集：Stage-1 top1={_vm['base']:.4f} → "
                        f"重排序 top1={_vm['rerank']:.4f}（Δ={_vm['rerank']-_vm['base']:+.4f}，"
                        f"次序改变率 {_vm['flip']:.2%}）")
                    globals()["RERANKER"] = _head
                    globals()["RERANK_TXT"] = _txt
                    log(f"  [rerank] 头参数量 {sum(p.numel() for p in _head.parameters()):,}")
            m = eval_test(model, obj_emb, prior=None)
            # ---------- β 敏感性扫描（同一权重，仅改推理期校准强度）----------
            _scan_s = str(getattr(args, "pair_rel_beta_scan", "") or "").strip()
            if _scan_s and getattr(_vmodel, "PAIR_REL", False):
                _betas = [float(x) for x in _scan_s.split(",") if x.strip()]
                _scan = {}
                _b0 = float(_vmodel.PAIR_REL_BETA)
                for _b in _betas:
                    _vmodel.PAIR_REL_BETA = _b
                    _ms = eval_test(model, obj_emb, prior=None)
                    _ms2 = eval_test_excl(model, obj_emb) if _excl_mask is not None else None
                    _scan[str(_b)] = {"metrics": _ms, "metrics_excl": _ms2}
                    log("  [beta-scan] β=%.2f top1=%.4f R@20=%.4f R@50=%.4f R@100=%.4f"
                        % (_b, _ms.get("top1", 0), _ms.get("R@20", 0),
                           _ms.get("R@50", 0), _ms.get("R@100", 0))
                        + (" | 判别 R@50=%.4f" % _ms2.get("R@50", 0) if _ms2 else ""))
                _vmodel.PAIR_REL_BETA = _b0
                globals().setdefault("_BETA_SCAN", {})[f"{aname}_{sd}"] = _scan
            per_seed.append(m)
            if _excl_mask is not None:
                m2 = eval_test_excl(model, obj_emb)
                per_seed_excl.append(m2)
            n_param = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                sum(p.numel() for p in obj_emb.parameters() if p.requires_grad)
            log(f"  seed={sd} {json.dumps(m, ensure_ascii=False)} | 训练 {tinfo['train_sec']}s | 可训练参数 {n_param:,}")
            if _excl_mask is not None:
                log(f"  seed={sd} [判别视图] " + json.dumps(m2, ensure_ascii=False))
            _rec_out = {"arm": aname, "seed": sd, "metrics": m,
                        "metrics_excl": (m2 if _excl_mask is not None else None),
                        "train": tinfo, "n_params": n_param,
                        "label": arm.name, "dataset": args.dataset,
                        "epochs": args.epochs, "K": hp.K}
            json.dump(_rec_out, open(_pf, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            results["raw_runs"].append({"arm": aname, "seed": sd, "metrics": m,
                                        "metrics_excl": (m2 if _excl_mask is not None else None),
                                        "train": tinfo, "n_params": n_param})
        agg = {}
        for key in _METRIC_KEYS:
            vals = np.array([s[key] for s in per_seed])
            agg[key] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std(ddof=0)), 4),
                        "vals": [round(float(v), 4) for v in vals]}
        agg["avg_rel_per_img"] = round(float(np.mean([s["avg_rel_per_img"] for s in per_seed])), 3)
        rec = {"label": arm.name, "prior": False, "metrics": agg, "loss_curve_mean": None}
        if per_seed_excl:
            agg2 = {}
            for key in _METRIC_KEYS:
                vals = np.array([s[key] for s in per_seed_excl])
                agg2[key] = {"mean": round(float(vals.mean()), 4),
                             "std": round(float(vals.std(ddof=0)), 4),
                             "vals": [round(float(v), 4) for v in vals]}
            agg2["avg_rel_per_img"] = round(
                float(np.mean([s["avg_rel_per_img"] for s in per_seed_excl])), 3)
            rec["metrics_excl"] = agg2
        results["arms"][aname] = rec
        log(f"  → {arm.name}: " + " ".join(
            f"{k}={agg[k]['mean']}±{agg[k]['std']}" for k in _METRIC_KEYS))
        if per_seed_excl:
            log(f"  → {arm.name} [判别视图]: " + " ".join(
                f"{k}={agg2[k]['mean']}±{agg2[k]['std']}"
                for k in ("R@50", "mR@50", "mR@100", "top1", "BRC@1") if k in agg2))

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    if globals().get("_RR_BASE"):
        results["rerank_baseline"] = globals()["_RR_BASE"]
        results["rerank_mode"] = getattr(args, "rerank_mode", "none")
        results["rerank_topm"] = getattr(args, "rerank_topm", None)
    if globals().get("_PATH_AUC"):
        results["path_auc"] = globals()["_PATH_AUC"]
    if globals().get("_BETA_SCAN"):
        results["beta_scan"] = globals()["_BETA_SCAN"]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("=" * 78)
    log(f"完成，总耗时 {results['elapsed_sec']}s → {args.out}")
    # 打印汇总表
    hdr = "".join(f"{k:>15}" for k in _METRIC_KEYS)
    log(f"{'臂':<30}{hdr}")
    for aname in arm_names:
        r = results["arms"][aname]
        mm = r["metrics"]
        if r.get("prior"):
            row = "".join(f"{mm[k]:>15.4f}" for k in _METRIC_KEYS)
        else:
            row = "".join(f"{mm[k]['mean']:>10.4f}±{mm[k]['std']:<4.4f}" for k in _METRIC_KEYS)
        log(f"{r['label']:<30}{row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
