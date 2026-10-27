#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨机器路径解析（本机 Windows ↔ 腾讯云服务器）。

背景：多个脚本原先**硬编码 Windows 路径**（`<USER_HOME>`、
`<PROJECT_ROOT>
`ValueError: The provided filename ... does not exist`。

统一规则（按优先级）：
  1. 显式环境变量：`CLIP_WEIGHTS` / `CLIP_BPE` / `VG_DATA_ROOT`
  2. 由 `VG_DATA_ROOT` 推导（`$VG_DATA_ROOT/clip/...`）
  3. 仓库内相对路径（`<项目>/code/..` 下的 `data/clip/...`）
  4. 本机默认硬编码路径（兜底）
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_PROJ = os.path.dirname(_CODE)

_WIN_CLIP_W = r"/path/to/clip/ViT-B-32.pt"
_WIN_DATA = (r"/path/to/vg_data")


def data_root() -> str:
    """数据根：优先环境变量，其次仓库内 `../data`，最后本机默认。"""
    v = os.environ.get("VG_DATA_ROOT")
    if v:
        return v
    local = os.path.join(_PROJ, "data")
    if os.path.isdir(os.path.join(local, "vg150")):
        return local
    return _WIN_DATA


def clip_weights() -> str:
    v = os.environ.get("CLIP_WEIGHTS")
    if v and os.path.exists(v):
        return v
    for cand in (os.path.join(data_root(), "clip", "ViT-B-32.pt"),
                 os.path.join(_PROJ, "data", "clip", "ViT-B-32.pt"),
                 _WIN_CLIP_W):
        if os.path.exists(cand):
            return cand
    return _WIN_CLIP_W


def clip_bpe() -> str:
    v = os.environ.get("CLIP_BPE")
    if v and os.path.exists(v):
        return v
    for cand in (os.path.join(data_root(), "clip", "bpe_simple_vocab_16e6.txt.gz"),
                 os.path.join(_PROJ, "data", "clip", "bpe_simple_vocab_16e6.txt.gz")):
        if os.path.exists(cand):
            return cand
    return os.path.join(_WIN_DATA, "clip", "bpe_simple_vocab_16e6.txt.gz")


def _torch_loadable(path: str) -> bool:
    """验证该权重能否被 `torch.load(weights_only=False)` 读出。

    ⚠️ 必要性（实测踩坑）：`faster_rcnn_vg_r101c4.pkl` 虽名为 .pkl，实际是从 Caffe
    转换而来的权重，**`torch.load` 会报 `RuntimeError: Invalid magic number; corrupt file?`**
    （torch 误走 legacy 反序列化分支）。而 `artifacts/det_vg_resnext101.pth`
    才是已转成 torch zip 序列化、可正常加载的版本（实测能读出 OrderedDict，
    键为 `backbone.body.0.weight` 等）。
    """
    if not path or not os.path.exists(path):
        return False
    try:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        return isinstance(sd, dict) and any(
            hasattr(v, "shape") for v in list(sd.values())[:8])
    except Exception:
        return False


def vg_detector_ckpt() -> str:
    """在 Visual Genome 上训练的检测器权重（Anderson et al. bottom-up-attention）。

    优先环境变量 `VG_DETECTOR_CKPT`；否则按「**可被 torch.load 成功**」筛选：
      1. `artifacts/det_vg_resnext101.pth`（已转 torch 序列化，**首选**）
      2. `detectors/faster_rcnn_vg_r101c4.pkl`（Caffe 原件，torch.load 会失败）
      3. 其它候选
    """
    v = os.environ.get("VG_DETECTOR_CKPT")
    if v and _torch_loadable(v):
        return v

    # ⚠️ 候选路径须覆盖**代码目录下的 artifacts/**（det_vg_resnext101.pth 实际在此（_CODE/artifacts），
    #   而非 data/artifacts/）。
    roots = [os.path.join(_CODE, "artifacts"), data_root(), os.path.join(_PROJ, "data")]
    rels = ["det_vg_resnext101.pth",
            "../artifacts/det_vg_resnext101.pth",
            "detectors/faster_rcnn_vg_r101c4.pkl",
            "detectors/det_vg_resnext101.pth"]
    for base in roots:
        for rel in rels:
            cand = os.path.join(base, rel)
            if _torch_loadable(cand):
                return cand
    # 都不行时仍返回首选路径，让上层给出明确的 FileNotFoundError 提示
    return os.path.join(data_root(), rels[0])


def finetuned_detector(arch: str = "resnet50") -> str:
    """我们自己在 VG 上微调出来的检测器权重。

    命名约定：`artifacts/det_vg_<arch>.pth`，arch ∈ {resnet50, resnext101}。
    """
    a = arch.replace("_vg", "")
    v = os.environ.get("VG_DET_FT_CKPT")
    if v:
        return v
    for base in (os.path.dirname(os.path.abspath(__file__)),):
        cand = os.path.join(os.path.dirname(base), "artifacts",
                            f"det_vg_{a}.pth")
        if os.path.exists(cand):
            return cand
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "artifacts", f"det_vg_{a}.pth")


def describe() -> str:
    return (f"data_root={data_root()}\n"
            f"clip_weights={clip_weights()} (exists={os.path.exists(clip_weights())})\n"
            f"clip_bpe={clip_bpe()} (exists={os.path.exists(clip_bpe())})\n"
            f"vg_detector={vg_detector_ckpt()} (exists={os.path.exists(vg_detector_ckpt())})\n"
            f"finetuned={finetuned_detector()} "
            f"(exists={os.path.exists(finetuned_detector())})")


if __name__ == "__main__":
    print(describe())
