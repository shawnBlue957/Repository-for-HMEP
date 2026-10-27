#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faster R-CNN style ROI feature extraction (benchmarked against SOTA backbone).

motivation
----
The original implementation uses **CLIP to encode the cropping of the object frame** as $x^v$. Its structural defects are
**Cropping loses the context and relative position between two objects**, while `on`/`in`/`above` such spatial relationships
It just depends on the relative layout of the two. SOTA (Neural Motifs/RePEAT/PE-Net/CAGE-SGG)
Unified use of **Faster R-CNN’s ROI features**:
    Backbone(ResNeXt-101-FPN) → MultiScaleRoIAlign → box_head(2×FC→1024)

This module supports two implementations:
1. `resnext101` (**default, recommended**): Use the torchvision component** to assemble ResNeXt-101-FPN** by yourself,
   That is, layer2/3/4 → FPN(256) → ROI-Align(7×7) → TwoMLLPead(1024) of `resnext101_32x8d`.
   This is a characteristic pathway that is isomorphic to SOTA (consistent with the backbone family).
2. `resnet50`: Directly reuse torchvision’s `fasterrcnn_resnet50_fpn` internal ROI header (lighter and faster),
   For comparison purposes.

⚠️Remaining differences from SOTA (**must be included in the paper**)
------------------------------------------
1. **Pre-training data**: This module uses **ImageNet** (ResNeXt-101) or **COCO** (ResNet-50-FPN),
   SOTA uses detectors trained on VG. ⟹ This is a recurrence of the ”characteristic pathway structure”,
   It is not a reproduction of the ”quality of detector adaptation on VG”.
2. 512-dimensional projection by **frozen orthogonal matrix** (consistent with the ”frozen features” caliber of the CLIP path,
   Do not introduce trainable parameters and keep the parameter amount comparison fair).
3. Uses **GT box** (PredCls protocol), so there is no detection error - consistent with SOTA’s PredCls.

The product format is fully compatible with the CLIP cache of `small_sample_experiment.extract_features`:
    `{image_id: {"v": (n,512) float32, "labels": [...], "boxes": [...], "W": W, "H": H}}`
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

OUT_DIM = 512
BOX_HEAD_DIM = 1024          # Consistent with SOTA’s box_head output dimension
FPN_CH = 256
ROI_SIZE = 7


def _frozen_proj(in_dim: int, out_dim: int, device) -> torch.Tensor:
    """Freeze orthographic projection (column orthogonal, inner product structure preserved)."""
    g = torch.Generator().manual_seed(20261017)
    Q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim, generator=g))
    return Q.to(device)


def build_detector(arch: str, num_classes: Optional[int] = None,
                   min_size: int = 800, max_size: int = 1333):
    """Construct torchvision Faster R-CNN: **The only entrance shared by ROI extraction and VG fine-tuning**.

    It must be a single construction entry: otherwise the state_dict saved in `finetune_detector_vg.py`
    It does not match the key name/structure expected during feature extraction, the weight cannot be transferred back, and an error may not be reported.

    arch
      resnet50: torchvision native ResNet-50-FPN (COCO pre-trained)
      resnext101: ResNeXt-101-32x8d(ImageNet) + FPN, **Same family as SOTA**
    num_classes
      If given, replace the classification header with the number of categories (VG-150 uses 151 = 150 categories + background).
      ⚠️ torchvision **not allowed** to pass `num_classes` with `weights`
      (It will be verified according to COCO’s 91 category and throw a ValueError), so build it by default first and then change the header.
    """
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)

    if arch not in ("resnet50", "resnext101"):
        raise ValueError(f"unknown arch: {arch}")
    det = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        min_size=min_size, max_size=max_size)

    if arch == "resnext101":
        from torchvision.models import ResNeXt101_32X8D_Weights, resnext101_32x8d
        from torchvision.models.detection.backbone_utils import BackboneWithFPN
        from torchvision.models.detection.faster_rcnn import TwoMLPHead
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        from torchvision.ops import MultiScaleRoIAlign
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

        m = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        # children order: conv1(0) bn1(1) relu(2) maxpool(3)
        #              layer1(4) layer2(5) layer3(6) layer4(7) avgpool(8) fc(9)
        # Truncate avgpool/fc ⟹ The named submodule of Sequential is ’0’..’7’
        body = nn.Sequential(*list(m.children())[:-2])
        # ⚠️ Must be aligned with torchvision’s default ResNet-50-FPN **structure**: take layer1..layer4
        # Four feature layers + LastLevelMaxPool ⟹ Output ’0’, ’1’, ’2’, ’3’, ’pool’, a total of 5 levels,
        # It is consistent with the 5 sets of anchor points of RPN and the featmap_names=[’0’,’1’,’2’,’3’] of ROI.
        # If only 3 feature layers (which have been stepped on) are mapped, the mismatch between the number of anchor points and the number of feature maps will directly
        # Throw ”Anchors should be Tuple[Tuple[int]]...”.
        # ResNeXt-101 channel number: layer1=256, layer2=512, layer3=1024, layer4=2048
        det.backbone = BackboneWithFPN(
            body, return_layers={"4": "0", "5": "1", "6": "2", "7": "3"},
            in_channels_list=[256, 512, 1024, 2048], out_channels=FPN_CH,
            extra_blocks=LastLevelMaxPool())
        det.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=ROI_SIZE, sampling_ratio=2)
        det.roi_heads.box_head = TwoMLPHead(FPN_CH * ROI_SIZE * ROI_SIZE, BOX_HEAD_DIM)
        det.transform = GeneralizedRCNNTransform(
            min_size, max_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if num_classes is not None:
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_f = det.roi_heads.box_predictor.cls_score.in_features
        det.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    return det


class ROIHead(nn.Module):
    """ROI-Align + Dual FC Head + Freeze Projection."""

    def __init__(self, arch: str, out_dim: int = OUT_DIM, device=None):
        super().__init__()
        self.arch = arch
        # `_vg` suffix = load our own fine-tuned weights on VG
        #(tools/finetune_detector_vg.py), the rest is exactly the same as the base arch,
        # So `resnet50` vs `resnet50_vg`, `resnext101` vs `resnext101_vg`
        #Constitute the paired ablation of ”unique variable = pre-training data”.
        det = build_detector(arch.replace("_vg", ""))
        self.backbone = det.backbone
        self.transform = det.transform
        self.roi_pool = det.roi_heads.box_roi_pool
        self.box_head = det.roi_heads.box_head
        if arch.endswith("_vg"):
            self._load_vg_finetuned(arch)

        self.out_dim = out_dim
        self.proj = _frozen_proj(BOX_HEAD_DIM, out_dim, device)

    def _load_vg_finetuned(self, arch: str) -> None:
        """Inject backbone/ROI header weights ”fine-tuned on VG”.

        The weights come from the **full detector** state_dict saved in `tools/finetune_detector_vg.py`,
        Here we only take the three parts needed for the ROI feature channel:
            backbone.*            → self.backbone.*
            roi_heads.box_head.*  → self.box_head.*
        (`box_roi_pool` has no training parameters; the classification header is not used under PredCls.)
        """
        from _paths import finetuned_detector
        ck = os.environ.get("VG_DET_FT_CKPT") or finetuned_detector(arch)
        if not os.path.exists(ck):
            raise FileNotFoundError(
                f"missing VG Fine-tune weights:{ck}\n"
                f"  First run: python tools/finetune_detector_vg.py --arch {arch.replace('_vg','')}"
                f" --epochs 1")
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        tgt: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            if k.startswith("backbone."):
                tgt[k] = v
            elif k.startswith("roi_heads.box_head."):
                tgt["box_head." + k[len("roi_heads.box_head."):]] = v
        if not tgt:
            raise RuntimeError(f"VG Can’t find it in weights backbone/box_head: {list(sd)[:5]}")
        missing, unexpected = self.load_state_dict(tgt, strict=False)
        crit = [m for m in missing
                if m.startswith(("backbone.", "box_head.")) and "proj" not in m]
        if crit:
            raise RuntimeError(f"VG weight vs. {arch} Structure does not match, missing {len(crit)} Item:{crit[:5]}")
        print(f"[roi] arch={arch} Loaded VG Fine-tune weights {os.path.basename(ck)}"
              f"（{len(tgt)} item)")

    @torch.no_grad()
    def forward(self, tensors: List[torch.Tensor],
                boxes_list: Sequence[torch.Tensor], device,
                amp: bool = True) -> List[torch.Tensor]:
        images, _ = self.transform(tensors, None)
        # ⚠️ torchvision’s transform only does resize/pad/normalize, **does not move the device**;
        # If the model is not moved to the GPU, it will report
        # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        images = images.to(device)
        # ⚠️⚠️ Key: `transform(images, targets=None)` **Only scale the image, not the frame**
        # (`_resize_boxes` is only called if targets is non-None). The feature map of ROI-Align
        # From the **scaled** image, so the frame must be scaled by the same ratio. Otherwise the image is enlarged ~2×
        # (short side 800), the frame of the original image coordinates only covers a small area in the upper left corner of the feature map.
        # The overall pooling position is misaligned - but **no error** will be reported, it will only cause the feature to fail silently.
        bl = []
        for b, t, size in zip(boxes_list, tensors, images.image_sizes):
            bb = b.to(device).float()
            if bb.numel() == 0:
                bl.append(bb)
                continue
            oh, ow = int(t.shape[1]), int(t.shape[2])
            nh, nw = int(size[0]), int(size[1])
            bb = bb.clone()
            bb[:, 0::2] *= (nw / float(ow))
            bb[:, 1::2] *= (nh / float(oh))
            bl.append(bb)
        # AMP: Actual measurement on T4: 1.27 → 5.16 pictures/s (4.06×), vector-wise cosine is the lowest 0.999894.
        # The two backbones are turned on uniformly to ensure that the characteristic calibers of R50 / RX101 are consistent.
        # ⚠️ Do not open cudnn.benchmark at the same time: batch-by-batch size changes will trigger the algorithm to re-search, which will be 5× slower.
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=bool(amp and device.type == "cuda")):
            feats = self.backbone(images.tensors)
            pooled = self.roi_pool(feats, bl, images.image_sizes)
            h = self.box_head(pooled)                    # (ΣN, 1024)
            h = h @ self.proj                            # (ΣN, out_dim)
        h = h.float()
        out, k = [], 0
        for b in boxes_list:
            n = b.shape[0]
            out.append(h[k:k + n])
            k += n
        return out


class ROIEncoder:
    """External interface: batch (PIL, boxes) → each picture (N, out_dim).

    Configurable resolution with arch; split by caller (`_enc_safe`) on OOM.
    """

    def __init__(self, arch: str = "resnext101", out_dim: int = OUT_DIM,
                 device: Optional[str] = None, min_size: int = 800,
                 max_size: int = 1333, verbose: bool = False, amp: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = amp
        self.head = ROIHead(arch, out_dim, self.device)
        self.head.to(self.device).eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.head.proj = self.head.proj.to(self.device)
        # The resolution must be coverable for both arches: `det.transform` of the resnet50 branch is constructed
        # Fixed by 800/1333, if not rebuilt, `--roi-min-size` will be silently disabled.
        if (min_size, max_size) != (800, 1333) or arch == "resnext101":
            self.head.transform = type(self.head.transform)(
                min_size, max_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if verbose:
            n = sum(p.numel() for p in self.head.parameters())
            print(f"[roi] arch={arch} Loading completed,{n:,} Parameter (freeze); device {self.device}；"
                  f"resolution {min_size}/{max_size}; projection {BOX_HEAD_DIM}→{out_dim}；"
                  f"AMP={'on' if self.amp else 'off'}")

    @torch.no_grad()
    def encode_batch(self, pils: Sequence, boxes_list: Sequence[torch.Tensor]
                     ) -> List[torch.Tensor]:
        from torchvision.transforms.functional import to_tensor
        tensors = [to_tensor(p.convert("RGB")) for p in pils]
        return self.head(tensors, boxes_list, self.device, amp=self.amp)


def extract_roi_features(records: List, imgzip, cache_path: str,
                         log=print, batch_imgs: int = 2,
                         min_size: int = 800, max_size: int = 1333,
                         arch: str = "resnext101", amp: bool = True) -> Dict[str, dict]:
    """Write the ROI features to a cache file in the same format as the CLIP cache (read images in blocks → encode → download).

    ⚠️ **OOM security**: The backbone is padded in batches at 800×1333, and the video memory grows with the square of the maximum side length.
    Therefore, `batch_imgs` defaults to 2, and `_enc_safe` automatically splits in half until a single image occurs when OOM occurs.
    """
    if os.path.exists(cache_path):
        t = time.time()
        cache = torch.load(cache_path, weights_only=False)
        log(f"  Load ROI feature cache {os.path.basename(cache_path)}"
            f"（{len(cache)} Figure,{time.time()-t:.1f}s）")
        return cache

    from PIL import Image
    import io as _io

    if arch == "vg_r101c4":
        # Detector trained on Visual Genome (Anderson et al.), same as SOTA pre-trained.
        from vg_detector import VGDetectorEncoder
        from _paths import vg_detector_ckpt
        _ck = vg_detector_ckpt()
        if not os.path.exists(_ck):
            raise FileNotFoundError(
                f"missing VG Detector weights:{_ck}\n"
                f"  Download: curl -L -o <target> "
                f"https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl")
        enc = VGDetectorEncoder(_ck, out_dim=OUT_DIM, device=None,
                                min_size=min_size, max_size=max_size,
                                verbose=True, amp=amp)
    else:
        enc = ROIEncoder(arch=arch, device=None, min_size=min_size,
                         max_size=max_size, verbose=True, amp=amp)

    def _enc_safe(pils, boxes_list):
        try:
            return enc.encode_batch(pils, boxes_list)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(pils) == 1:
                raise
            torch.cuda.empty_cache()
            m = len(pils) // 2
            return _enc_safe(pils[:m], boxes_list[:m]) + _enc_safe(pils[m:], boxes_list[m:])

    cache: Dict[str, dict] = {}
    miss_img = 0
    t0 = time.time()
    n_box = 0
    last_log = t0

    for b0 in range(0, len(records), batch_imgs):
        block = records[b0:b0 + batch_imgs]
        pils, boxes_list, metas = [], [], []
        for im in block:
            raw = imgzip.read(im.image_id)
            if raw is None:
                miss_img += 1
                continue
            try:
                pil = Image.open(_io.BytesIO(raw)).convert("RGB")
            except Exception:
                miss_img += 1
                continue
            W, H = pil.size
            bx = []
            for (x1, y1, x2, y2) in im.boxes:
                x1c, y1c = max(0, min(float(x1), W - 1)), max(0, min(float(y1), H - 1))
                x2c, y2c = max(x1c + 1, min(float(x2), W)), max(y1c + 1, min(float(y2), H))
                bx.append([x1c, y1c, x2c, y2c])
            pils.append(pil)
            boxes_list.append(torch.tensor(bx, dtype=torch.float32)
                              if bx else torch.zeros(0, 4))
            metas.append((im, W, H))
        if pils:
            feats = _enc_safe(pils, boxes_list)
            for (im, W, H), fv in zip(metas, feats):
                cache[im.image_id] = {"v": fv.cpu().float(), "labels": list(im.labels),
                                      "boxes": [list(b) for b in im.boxes],
                                      "W": W, "H": H}
                n_box += fv.shape[0]
            for p in pils:
                p.close()
        if time.time() - last_log > 20:
            done = min(b0 + batch_imgs, len(records))
            log(f"    Processed {done:,}/{len(records):,} Figure | {n_box:,} box | "
                f"{done / max(time.time()-t0, 1e-6):.1f} Figure/s | "
                f"Used {(time.time()-t0)/60:.1f} minutes")
            last_log = time.time()

    log(f"  ROI Coding completed:{n_box:,} frame, time {(time.time()-t0)/60:.1f} minutes (dropped picture {miss_img}）")
    torch.save(cache, cache_path)
    log(f"  ROI Feature extraction is completed:{len(cache)} Figure → {os.path.basename(cache_path)}")
    return cache


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from PIL import Image
    arch = sys.argv[1] if len(sys.argv) > 1 else "resnext101"
    enc = ROIEncoder(arch=arch, verbose=True)
    p = Image.new("RGB", (400, 300), (128, 128, 128))
    bx = torch.tensor([[10., 10., 200., 180.], [200., 100., 380., 280.]])
    f = enc.encode_batch([p], [bx])[0]
    print("output", tuple(f.shape), "Expectation (2, 512); norm",
          [round(float(v), 3) for v in f.norm(dim=-1)])
