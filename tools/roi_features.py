#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faster R-CNN 风格的 ROI 特征提取（对标 SOTA 骨干）。

动机
----
原实现用 **CLIP 对物体框的裁剪编码** 作为 $x^v$。其结构性缺陷是
**裁剪丢失两物体间的上下文与相对位置**，而 `on`/`in`/`above` 这类空间关系
恰恰依赖二者的相对布局。SOTA（Neural Motifs / RePEAT / PE-Net / CAGE-SGG）
统一使用 **Faster R-CNN 的 ROI 特征**：
    骨干(ResNeXt-101-FPN) → MultiScaleRoIAlign → box_head(2×FC→1024)

本模块支持两种实现：
1. `resnext101`（**默认，推荐**）：用 torchvision 构件**自行拼装 ResNeXt-101-FPN**，
   即 `resnext101_32x8d` 的 layer2/3/4 → FPN(256) → ROI-Align(7×7) → TwoMLPHead(1024)。
   这是**与 SOTA 同构**的特征通路（骨干族一致）。
2. `resnet50`：直接复用 torchvision 的 `fasterrcnn_resnet50_fpn` 内部 ROI 头（较轻、较快），
   作对照用。

⚠️ 与 SOTA 的剩余差异（**必须写进论文**）
------------------------------------------
1. **预训练数据**：本模块用 **ImageNet**（ResNeXt-101）或 **COCO**（ResNet-50-FPN），
   而 SOTA 用的是 **在 VG 上训练的检测器**。⟹ 这是「特征通路结构」的复现，
   不是「检测器在 VG 上的适配质量」的复现。
2. 512 维由**冻结正交矩阵**投影（与 CLIP 路径的"冻结特征"口径一致，
   不引入可训练参数，保持参数量对比公平）。
3. 使用 **GT 框**（PredCls 协议），故不含检测误差——与 SOTA 的 PredCls 一致。

产物格式与 `small_sample_experiment.extract_features` 的 CLIP 缓存**完全兼容**：
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
BOX_HEAD_DIM = 1024          # 与 SOTA 的 box_head 输出维一致
FPN_CH = 256
ROI_SIZE = 7


def _frozen_proj(in_dim: int, out_dim: int, device) -> torch.Tensor:
    """冻结正交投影（列正交，保持内积结构）。"""
    g = torch.Generator().manual_seed(20261017)
    Q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim, generator=g))
    return Q.to(device)


def build_detector(arch: str, num_classes: Optional[int] = None,
                   min_size: int = 800, max_size: int = 1333):
    """构造 torchvision Faster R-CNN：**ROI 抽取与 VG 微调共用的唯一入口**。

    必须是单一构造入口：否则 `finetune_detector_vg.py` 存下的 state_dict
    与特征抽取时期望的键名/结构会对不上，权重灌不回来，且不一定报错。

    arch
      resnet50   : torchvision 原生 ResNet-50-FPN（COCO 预训练）
      resnext101 : ResNeXt-101-32x8d(ImageNet) + FPN，**与 SOTA 同族**
    num_classes
      给定则把分类头换成该类别数（VG-150 用 151 = 150 类 + 背景）。
      ⚠️ torchvision **不允许**在带 `weights` 时传 `num_classes`
      （会按 COCO 的 91 类校验并抛 ValueError），故先默认构建再换头。
    """
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)

    if arch not in ("resnet50", "resnext101"):
        raise ValueError(f"未知 arch: {arch}")
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
        # children 顺序：conv1(0) bn1(1) relu(2) maxpool(3)
        #              layer1(4) layer2(5) layer3(6) layer4(7) avgpool(8) fc(9)
        # 截掉 avgpool/fc ⟹ Sequential 的命名子模块为 '0'..'7'
        body = nn.Sequential(*list(m.children())[:-2])
        # ⚠️ 必须与 torchvision 默认 ResNet-50-FPN **结构对齐**：取 layer1..layer4
        # 四个特征层 + LastLevelMaxPool ⟹ 输出 '0','1','2','3','pool' 共 5 级，
        # 才与 RPN 的 5 组锚点、以及 ROI 的 featmap_names=['0','1','2','3'] 一致。
        # 若只映射 3 个特征层（曾踩过），锚点数与特征图数不匹配会直接
        # 抛 "Anchors should be Tuple[Tuple[int]]..."。
        #   ResNeXt-101 通道数：layer1=256, layer2=512, layer3=1024, layer4=2048
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
    """ROI-Align + 双 FC 头 + 冻结投影。"""

    def __init__(self, arch: str, out_dim: int = OUT_DIM, device=None):
        super().__init__()
        self.arch = arch
        # `_vg` 后缀 = 加载我们自己在 VG 上微调过的权重
        #（tools/finetune_detector_vg.py），其余部分与基座 arch 完全一致，
        # 于是 `resnet50` vs `resnet50_vg`、`resnext101` vs `resnext101_vg`
        # 构成「唯一变量 = 预训练数据」的配对消融。
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
        """灌入「在 VG 上微调过」的 backbone / ROI 头权重。

        权重来自 `tools/finetune_detector_vg.py` 保存的**完整检测器** state_dict，
        这里只取 ROI 特征通路需要的三部分：
            backbone.*            → self.backbone.*
            roi_heads.box_head.*  → self.box_head.*
        （`box_roi_pool` 无可训练参数；分类头在 PredCls 下不使用。）
        """
        from _paths import finetuned_detector
        ck = os.environ.get("VG_DET_FT_CKPT") or finetuned_detector(arch)
        if not os.path.exists(ck):
            raise FileNotFoundError(
                f"缺少 VG 微调权重：{ck}\n"
                f"  先运行：python tools/finetune_detector_vg.py --arch {arch.replace('_vg','')}"
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
            raise RuntimeError(f"VG 权重里找不到 backbone/box_head：{list(sd)[:5]}")
        missing, unexpected = self.load_state_dict(tgt, strict=False)
        crit = [m for m in missing
                if m.startswith(("backbone.", "box_head.")) and "proj" not in m]
        if crit:
            raise RuntimeError(f"VG 权重与 {arch} 结构不匹配，缺 {len(crit)} 项：{crit[:5]}")
        print(f"[roi] arch={arch} 已载入 VG 微调权重 {os.path.basename(ck)}"
              f"（{len(tgt)} 项）")

    @torch.no_grad()
    def forward(self, tensors: List[torch.Tensor],
                boxes_list: Sequence[torch.Tensor], device,
                amp: bool = True) -> List[torch.Tensor]:
        images, _ = self.transform(tensors, None)
        # ⚠️ torchvision 的 transform 只做 resize/pad/normalize，**不搬设备**；
        # 模型在 GPU 时若不搬会报
        # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        images = images.to(device)
        # ⚠️⚠️ 关键：`transform(images, targets=None)` **只缩放图像，不缩放框**
        # （`_resize_boxes` 仅在 targets 非 None 时被调用）。而 ROI-Align 的特征图
        # 来自**缩放后**的图像，所以必须把框按同一比例缩放。否则图像被放大 ~2×
        # （短边 800）时，原图坐标的框只覆盖特征图左上角的一小块，
        # 池化位置整体错位——但**不会报任何错**，只会让特征静默失效。
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
        # AMP：T4 上实测 1.27 → 5.16 图/s（4.06×），逐向量余弦最低 0.999894。
        # 两个骨干统一开启，保证 R50 / RX101 的特征口径一致。
        # ⚠️ 不要同时开 cudnn.benchmark：批内尺寸逐批变化会触发算法重搜索，反而慢 5×。
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
    """对外接口：一批 (PIL, boxes) → 每图 (N, out_dim)。

    可配置分辨率与 arch；OOM 时由调用方（`_enc_safe`）拆分。
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
        # 分辨率对两个 arch 都要可覆盖：resnet50 分支的 `det.transform` 是构造时
        # 按 800/1333 固定的，若不重建则 `--roi-min-size` 静默失效。
        if (min_size, max_size) != (800, 1333) or arch == "resnext101":
            self.head.transform = type(self.head.transform)(
                min_size, max_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if verbose:
            n = sum(p.numel() for p in self.head.parameters())
            print(f"[roi] arch={arch} 加载完成，{n:,} 参数（冻结）；设备 {self.device}；"
                  f"分辨率 {min_size}/{max_size}；投影 {BOX_HEAD_DIM}→{out_dim}；"
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
    """把 ROI 特征写入与 CLIP 缓存**同格式**的缓存文件（按块读图→编码→落盘）。

    ⚠️ **OOM 安全**：骨干在 800×1333 下按批 padding，显存随最大边长平方增长。
    故 `batch_imgs` 默认 2，且 `_enc_safe` 在 OOM 时自动对半拆分直到单图。
    """
    if os.path.exists(cache_path):
        t = time.time()
        cache = torch.load(cache_path, weights_only=False)
        log(f"  载入 ROI 特征缓存 {os.path.basename(cache_path)}"
            f"（{len(cache)} 图，{time.time()-t:.1f}s）")
        return cache

    from PIL import Image
    import io as _io

    if arch == "vg_r101c4":
        # 在 Visual Genome 上训练的检测器（Anderson et al.），与 SOTA 同源预训练。
        from vg_detector import VGDetectorEncoder
        from _paths import vg_detector_ckpt
        _ck = vg_detector_ckpt()
        if not os.path.exists(_ck):
            raise FileNotFoundError(
                f"缺少 VG 检测器权重：{_ck}\n"
                f"  下载：curl -L -o <目标> "
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
            log(f"    已处理 {done:,}/{len(records):,} 图 | {n_box:,} 框 | "
                f"{done / max(time.time()-t0, 1e-6):.1f} 图/s | "
                f"已用 {(time.time()-t0)/60:.1f} 分钟")
            last_log = time.time()

    log(f"  ROI 编码完成：{n_box:,} 框，用时 {(time.time()-t0)/60:.1f} 分钟（丢图 {miss_img}）")
    torch.save(cache, cache_path)
    log(f"  ROI 特征提取完成：{len(cache)} 图 → {os.path.basename(cache_path)}")
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
    print("输出", tuple(f.shape), "期望 (2, 512)；范数",
          [round(float(v), 3) for v in f.norm(dim=-1)])
