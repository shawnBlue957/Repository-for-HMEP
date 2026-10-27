#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 版 CLIP ViT-B/32 视觉塔（从既有 OpenAI JIT 存档离线重建）。

背景（必须写进实验报告）：
  本机 `data/clip/ViT-B-32.pt` 是用 **CUDA 追踪**导出的 TorchScript 存档，
  其计算图内硬编码了 `device=torch.device("cuda:0")`，在本机（torch 2.14.1+cpu，
  无 CUDA）上 `torch.jit.load(...).visual(x)` 直接抛错，**无法在 CPU 运行**。

  但该存档的 `state_dict()` 可在 CPU 上正常提取（305 个键，视觉塔权重大量为
  fp16）。故此处用标准 ViT-B/32 结构**离线重建**视觉塔并注入原权重，
  从而在 CPU 上得到与官方 CLIP 一致的图像编码器。

仅重建**视觉塔**（encode_image）。开放 CLIP 的文本塔还需 BPE 词表
（bpe_simple_vocab_16e6.txt.gz，本机缺失），故本脚本不提供文本编码；
x_t 与蒸馏目标 q 改用可学习嵌入 / 谓词频率先验（实验报告中标注为替代方案）。

用法：
  from clip_cpu import CLIPVisionCPU
  enc = CLIPVisionCPU(weights_path)
  feat = enc.encode_pils([pil1, pil2, ...])   # (N, 512) L2 归一化前的原始输出
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP 官方图像归一化常数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_RES = 224


class QuickGELU(nn.Module):
    """OpenAI CLIP 使用的 QuickGELU（非 nn.GELU）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # nn.MultiheadAttention 的状态字键名与 CLIP 一致：attn.in_proj_weight 等
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        # 键名须与 CLIP 一致：mlp.c_fc / mlp.c_proj
        from collections import OrderedDict
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, N, E)，batch_first=False
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    """容器：使状态字键名呈 transformer.resblocks.{i}.*"""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768,
                 layers=12, heads=12, output_dim=512):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                                   # (B,768,7,7)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)                              # (B,49,768)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                      # (B,50,768)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                              # (L,N,E)
        for blk in self.transformer.resblocks:
            x = blk(x)
        x = x.permute(1, 0, 2)                              # (N,L,E)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIPVisionCPU:
    """从 OpenAI JIT 存档提取权重、重建的 CLIP ViT-B/32 视觉塔。

    ⚠️ 类名保留 `CPU` 是为兼容既有调用（`tools/small_sample_experiment.py` 等），
    但**现已支持 GPU**：`CLIPVisionCPU(weights_path, device="cuda")` 即可。
    迁移原因：本机 RTX 3060 可用，而 CLIP 图像编码是全流程最重的一步
    （57,723+26,446 图的裁剪编码需数小时），GPU 上可快一个量级。
    """

    def __init__(self, weights_path: str, verbose: bool = False,
                 device: Optional[str] = None):
        warnings.filterwarnings("ignore")
        jit = torch.jit.load(weights_path, map_location="cpu")
        jit.eval()
        sd = jit.state_dict()

        self.visual = VisionTransformer()
        sub = {k[len("visual."):]: v.float().clone()
               for k, v in sd.items() if k.startswith("visual.")}
        missing, unexpected = self.visual.load_state_dict(sub, strict=False)
        if verbose:
            print(f"[clip_cpu] 注入视觉塔权重 {len(sub)} 项；"
                  f"missing={missing} unexpected={unexpected}")
        if missing:
            raise RuntimeError(f"CLIP 视觉塔权重缺失: {missing}")

        # 设备选择：显式 > CUDA 可用 > CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual.to(self.device)
        # fp16 在 GPU 上显著更快且显存减半；CPU 上 fp16 支持差，保持 fp32
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.visual.to(self.dtype)
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor(CLIP_MEAN, dtype=self.dtype,
                                  device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, dtype=self.dtype,
                                 device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,224,224) 已归一化 → (N,512)。"""
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return self.visual(x).float()

    @torch.no_grad()
    def encode_pils(self, pils: Sequence, batch_size: int = 64) -> torch.Tensor:
        """pil list（RGB）→ (N,512) **CPU float32**，内部 resize 224 / 归一化。

        ⚠️ PIL 解码与 resize 仍在 CPU（不可避免），但 ViT 前向在 GPU。
        返回值统一回 CPU，以便与既有缓存格式（CPU 张量）兼容。
        """
        import numpy as np

        outs: List[torch.Tensor] = []
        for i in range(0, len(pils), batch_size):
            chunk = pils[i:i + batch_size]
            arrs = []
            for im in chunk:
                if im.size != (CLIP_RES, CLIP_RES):
                    im = im.resize((CLIP_RES, CLIP_RES))
                a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                arrs.append(a.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(arrs, 0))
            x = (x - self._mean.cpu()) / self._std.cpu()
            outs.append(self.encode_tensor(x))
        out = torch.cat(outs, 0) if outs else torch.zeros(0, 512)
        return out.cpu()


if __name__ == "__main__":
    import sys
    enc = CLIPVisionCPU(sys.argv[1] if len(sys.argv) > 1
                        else r"/path/to/clip/ViT-B-32.pt", verbose=True)
    from PIL import Image
    im = Image.new("RGB", (64, 64), (200, 30, 30))
    f = enc.encode_pils([im, im])
    print("out", tuple(f.shape), "cos(same)=", float(F.cosine_similarity(f[0:1], f[1:2])))
