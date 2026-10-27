#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-machine path resolution (native Windows ↔ Tencent Cloud Server).

Background: Several scripts originally had hard-coded Windows paths (`<USER_HOME>`,
`<PROJECT_ROOT>
`ValueError: The provided filename ... does not exist`.

Unified rules (by priority):
  1. Explicit environment variables: `CLIP_WEIGHTS` / `CLIP_BPE` / `VG_DATA_ROOT`
  2. Deduced from `VG_DATA_ROOT` (`$VG_DATA_ROOT/clip/...`)
  3. Relative path in the repository (`data/clip/...` under `<project>/code/..`)
  4. The default hard-coded path of this machine (the bottom line)
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_PROJ = os.path.dirname(_CODE)

_WIN_CLIP_W = r"/path/to/clip/ViT-B-32.pt"
_WIN_DATA = (r"/path/to/vg_data")


def data_root() -> str:
    """Data root: environment variables first, then `../data` in the repository, and finally the local default."""
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
    """Verify that the weights can be read by `torch.load(weights_only=False)`.

    ⚠️ Necessity (actual test pitfall): `faster_rcnn_vg_r101c4.pkl` Although named .pkl, it is actually from Caffe
    Converted weights, **`torch.load` will report `RuntimeError: Invalid magic number; corrupt file?`**
    (torch mistakenly took the legacy deserialization branch). And `artifacts/det_vg_resnext101.pth`
    is the version that has been converted to torch zip serialization and can be loaded normally (the actual test can read OrderedDict,
    The keys are `backbone.body.0.weight` etc.).
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
    """Detector weights trained on Visual Genome (Anderson et al. bottom-up-attention).

    Prioritize the environment variable `VG_DETECTOR_CKPT`; otherwise filter by ”**can be torch.loaded successfully**”:
      1. `artifacts/det_vg_resnext101.pth` (converted to torch serialization, **preferred**)
      2. `detectors/faster_rcnn_vg_r101c4.pkl` (Caffe original, torch.load will fail)
      3. Other candidates
    """
    v = os.environ.get("VG_DETECTOR_CKPT")
    if v and _torch_loadable(v):
        return v

    # ⚠️ The candidate path must cover artifacts/** in the code directory (det_vg_resnext101.pth is actually here (_CODE/artifacts),
    # instead of data/artifacts/).
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
    # If neither of them works, still return the preferred path and let the upper layer give a clear FileNotFoundError prompt.
    return os.path.join(data_root(), rels[0])


def finetuned_detector(arch: str = "resnet50") -> str:
    """Detector weights fine-tuned by ourselves on VG.

    Naming convention: `artifacts/det_vg_<arch>.pth`, arch ∈ {resnet50, resnext101}.
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
