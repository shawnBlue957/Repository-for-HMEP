#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU version of CLIP ViT-B/32 vision tower (offline rebuild from existing OpenAI JIT archive).

Background (must be included in the lab report):
  The native `data/clip/ViT-B-32.pt` is a TorchScript archive exported using **CUDA Trace**.
  `device=torch.device(”cuda:0”)` is hard-coded in its calculation graph. On this machine (torch 2.14.1+cpu,
  Without CUDA), `torch.jit.load(...).visual(x)` directly throws an error and **cannot run on the CPU**.

  But the archive’s `state_dict()` fetches fine on CPU (305 keys, visual tower weight massive
  fp16). Therefore, the standard ViT-B/32 structure is used to **offline reconstruct** the visual tower and inject the original weights.
  This results in an image encoder consistent with the official CLIP on the CPU.

Rebuild only the visual tower (encode_image). Opening CLIP’s text tower also requires a BPE vocabulary
(bpe_simple_vocab_16e6.txt.gz, missing in this machine), so this script does not provide text encoding;
x_t with distillation target q using a learnable embedding/predicate frequency prior instead (marked as an alternative in the lab report).

Usage:
  from clip_cpu import CLIPVisionCPU
  enc = CLIPVisionCPU(weights_path)
  feat = enc.encode_pils([pil1, pil2, ...]) # (N, 512) L2 original output before normalization
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP official image normalization constant
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_RES = 224


class QuickGELU(nn.Module):
    """QuickGELU (not nn.GELU) used by OpenAI CLIP."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # The status word key name of nn.MultiheadAttention is consistent with CLIP: attn.in_proj_weight, etc.
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        # The key name must be consistent with CLIP: mlp.c_fc / mlp.c_proj
        from collections import OrderedDict
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, N, E), batch_first=False
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    """Container: Make the status key name transformer.resblocks.{i}.*"""

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
    """Extracting weights from OpenAI JIT archive, reconstructed CLIP ViT-B/32 vision tower.

    ⚠️ The class name `CPU` is reserved for compatibility with existing calls (`tools/small_sample_experiment.py`, etc.),
    But **now supports GPU**: `CLIPVisionCPU(weights_path, device=”cuda”)` will do.
    Reason for migration: Native RTX 3060 is available, and CLIP image encoding is the most important step in the process
    (Crop encoding of 57,723+26,446 images takes hours) and is an order of magnitude faster on the GPU.
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
            print(f"[clip_cpu] Inject visual tower weights {len(sub)} item;"
                  f"missing={missing} unexpected={unexpected}")
        if missing:
            raise RuntimeError(f"CLIP Visual tower weight missing: {missing}")

        # Device selection: Explicit > CUDA available > CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual.to(self.device)
        # fp16 is significantly faster on GPU and has half the memory; fp16 support is poor on CPU, keep fp32
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
        """x: (N,3,224,224) normalized → (N,512)."""
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return self.visual(x).float()

    @torch.no_grad()
    def encode_pils(self, pils: Sequence, batch_size: int = 64) -> torch.Tensor:
        """pil list (RGB) → (N,512) **CPU float32**, internal resize 224 / normalized.

        ⚠️ PIL decoding and resize are still on the CPU (inevitable), but ViT forward is on the GPU.
        Return values ​​are unified back to the CPU for compatibility with existing cache formats (CPU tensors).
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
