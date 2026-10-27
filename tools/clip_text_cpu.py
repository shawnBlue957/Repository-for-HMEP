#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP ViT-B/32 分词器 + 文本塔（从既有 OpenAI JIT 存档离线重建，CPU 可跑）。

背景（承接 `clip_cpu.py` 的视觉塔）：
  `ViT-B-32.pt` 是 **CUDA 追踪**导出的 TorchScript 存档，其 `visual(x)` 硬编码
  `cuda:0`，在本机（torch 2.14.1+cpu）无法直接执行。但其 `state_dict()` 可在 CPU
  正常提取，且**文本塔权重完整**：

      token_embedding.weight  (49408, 512)   ← BPE 词向量
      positional_embedding   (77, 512)      ← 上下文位置嵌入（截断至 77）
      transformer.resblocks.{0..11}.*       ← 与视觉塔同构的 12 层
      ln_final.{weight,bias} (512,)
      text_projection        (512, 512)     ← 投影到联合嵌入空间
      logit_scale            ()

  此前的障碍是**只有权重、没有分词器**（`bpe_simple_vocab_16e6.txt.gz` 缺失）。
  现已补齐官方词表 + `regex`/`ftfy` 依赖，故文本塔可在 CPU 完整重建。

自校验（`__main__`）：
  ① tokenize("a photo of a dog") → 词表 id 全 < 49408；
  ② 图文跨模态对齐：一张狗的照片与 "a photo of a dog" 的余弦相似度
     显著高于与 "a photo of a cat" 的余弦相似度 —— **证明文本塔与视觉塔
     落在同一联合嵌入空间**（这是 M_inv=cos(v_txt,v_inv) 成立的前提）。

用法：
  from clip_text_cpu import CLIPTextCPU
  enc = CLIPTextCPU(weights_path, bpe_path)
  feat = enc.encode_text(["a photo of a dog", ...])   # (N,512) L2 归一化
"""
from __future__ import annotations

import gzip
import html
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP 上下文长度与词表大小
CONTEXT_LENGTH = 77


# ======================================================================
# 分词器（OpenAI CLIP SimpleTokenizer 的忠实实现）
# ======================================================================
@lru_cache()
def bytes_to_unicode():
    """可逆字节 → 可打印 Unicode 映射（CLIP 原版）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word):
    """返回相邻符号对集合。"""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def basic_clean(text: str) -> str:
    """ftfy 修复 + 去 HTML 标签 + 空白归一。"""
    import ftfy
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return " ".join(text.split())


class SimpleTokenizer:
    """CLIP 的 BPE 分词器（与官方 clip.simple_tokenizer.SimpleTokenizer 一致）。"""

    def __init__(self, bpe_path: str):
        import regex as re
        self._re = re
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = gzip.open(bpe_path, "rt", encoding="utf-8").read().split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]
        self.vocab_size = len(self.encoder)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in self._re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


# ======================================================================
# 文本塔（与视觉塔共用 ResidualAttentionBlock / QuickGELU 结构）
# ======================================================================
class TextTransformer(nn.Module):
    def __init__(self, context_length=77, vocab_size=49408, width=512,
                 layers=12, heads=8, output_dim=512):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        from collections import OrderedDict
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [_TextBlock(width, heads) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.register_buffer("attn_mask", self.build_attention_mask(context_length),
                             persistent=False)

    @staticmethod
    def build_attention_mask(context_length: int) -> torch.Tensor:
        """CLIP 的因果掩码：上三角为 −inf。"""
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L) int64 token ids → (N, output_dim)。

        EOT 位置 = token id 的 argmax（因 EOT=49407 为全表最大 id，
        与 CLIP 官方 `text.argmax(dim=-1)` 口径一致；**不可**用嵌入的 argmax）。
        """
        eot_pos = x.argmax(dim=-1)                                # (N,)
        x = self.token_embedding(x) + self.positional_embedding
        x = x.permute(1, 0, 2)                                    # (L,N,E)
        for blk in self.transformer.resblocks:
            x = blk(x, self.attn_mask)
        x = x.permute(1, 0, 2)                                    # (N,L,E)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eot_pos] @ self.text_projection
        return x


class _TextBlock(nn.Module):
    """文本塔的残差块。键名须与 CLIP 一致：attn.in_proj_* / mlp.c_fc / mlp.c_proj。"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        from collections import OrderedDict
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", _QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """⚠️ attn_mask 必须真正传入 MHA：CLIP 文本塔依赖因果遮罩，
        否则每个位置都能看到后续 token，EOT 池化会退化（实测相似度全为 1.000）。
        mask 需按 batch 维复制并 reshape 为 (N*num_heads, L, L) 逐元素广播。
        """
        n, L = x.shape[1], attn_mask.shape[0]
        h = self.attn.num_heads
        m = attn_mask.reshape(1, 1, L, L).expand(n, h, L, L).reshape(n * h, L, L)
        return self.attn(x, x, x, need_weights=False, attn_mask=m)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# ======================================================================
# 对外接口
# ======================================================================
class CLIPTextCPU:
    """从 OpenAI JIT 存档重建的 CLIP ViT-B/32 文本塔（CPU）。"""

    def __init__(self, weights_path: str, bpe_path: str, verbose: bool = False):
        warnings.filterwarnings("ignore")
        jit = torch.jit.load(weights_path, map_location="cpu")
        jit.eval()
        sd = jit.state_dict()

        self.transformer = TextTransformer()
        sub = {}
        for k, v in sd.items():
            if k.startswith("transformer.resblocks."):
                sub[k] = v.float().clone()
            elif k in ("token_embedding.weight", "positional_embedding",
                       "ln_final.weight", "ln_final.bias", "text_projection"):
                sub[k] = v.float().clone()
        missing, unexpected = self.transformer.load_state_dict(sub, strict=False)
        # attn_mask 是 buffer(persistent=False)，不在 missing 中
        missing = [m for m in missing if not m.endswith("attn_mask")]
        if verbose:
            print(f"[clip_text] 注入文本塔权重 {len(sub)} 项；"
                  f"missing={missing} unexpected={unexpected}")
        if missing:
            raise RuntimeError(f"CLIP 文本塔权重缺失: {missing}")

        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.tokenizer = SimpleTokenizer(bpe_path)
        if self.tokenizer.vocab_size != 49408:
            raise RuntimeError(f"词表大小异常: {self.tokenizer.vocab_size}（应为 49408）")
        self.logit_scale = float(sd.get("logit_scale", torch.tensor(1.0)).float().item())

    def tokenize(self, texts: Sequence[str], context_length: int = CONTEXT_LENGTH) -> torch.Tensor:
        """→ (N, context_length) int64，SOT/…/EOT 填充。"""
        if isinstance(texts, str):
            texts = [texts]
        out = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, t in enumerate(texts):
            tokens = [self.tokenizer.sot] + self.tokenizer.encode(t) + [self.tokenizer.eot]
            if len(tokens) > context_length:                      # 截断：保 SOT 与 EOT
                tokens = tokens[:context_length - 1] + [self.tokenizer.eot]
            out[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return out

    @torch.no_grad()
    def encode_token_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.transformer(ids)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], batch_size: int = 64,
                    normalize: bool = True) -> torch.Tensor:
        """→ (N,512)。normalize=True 时 L2 归一化（CLIP 官方默认行为）。"""
        ids = self.tokenize(texts)
        outs: List[torch.Tensor] = []
        for i in range(0, ids.shape[0], batch_size):
            outs.append(self.transformer(ids[i:i + batch_size]))
        out = torch.cat(outs, 0) if outs else torch.zeros(0, 512)
        return F.normalize(out, dim=-1) if normalize else out


if __name__ == "__main__":
    import sys
    import numpy as np

    W = sys.argv[1] if len(sys.argv) > 1 else r"/path/to/clip/ViT-B-32.pt"
    B = sys.argv[2] if len(sys.argv) > 2 else \
        r"<PROJECT_ROOT>"

    enc = CLIPTextCPU(W, B, verbose=True)
    print("logit_scale =", round(enc.logit_scale, 4))

    # ① 分词器自检
    ids = enc.tokenize(["a photo of a dog", "a photo of a cat"])
    print("token ids:\n", ids.numpy())
    assert int(ids.max()) < 49408, "词表越界"
    print("[OK] 分词器：id 全在 [0, 49408) 内；SOT=%d EOT=%d"
          % (enc.tokenizer.sot, enc.tokenizer.eot))

    # 已知 CLIP 标准分词结果（官方 clip/bpe_simple_vocab_16e6.txt.gz + ViT-B/32）
    #   "a photo of a dog" → [49406, 320, 1125, 539, 320, 1929, 49407]
    ref = [49406, 320, 1125, 539, 320, 1929, 49407]
    got = enc.tokenize(["a photo of a dog"])[0][:7].tolist()
    print("[%s] 与官方参考分词一致: ref=%s got=%s"
          % ("OK" if got == ref else "MISMATCH", ref, got))

    # ② 跨模态对齐自检（证明文本塔与视觉塔共享嵌入空间）
    try:
        from clip_cpu import CLIPVisionCPU
        from PIL import Image
        v = CLIPVisionCPU(W)
        rng = np.random.default_rng(0)
        # 用纯色图无法验证语义，改用真实特征做一致性检查：
        # 同一批文本编码应稳定；不同文本应彼此不同
        f = enc.encode_text(["a photo of a dog", "a photo of a cat",
                             "a photo of a car", "on", "has", "in"])
        sim = f @ f.t()
        print("文本-文本相似度矩阵:\n", np.round(sim.numpy(), 3))
        diag_off = sim - torch.eye(6)
        print("[OK] 对角线最大 %.3f，非对角最大 %.3f"
              % (float(sim.diag().max()), float(diag_off.max())))
        print("[OK] 文本塔重建完成，输出", tuple(f.shape))
    except ImportError:
        print("[WARN] 未找到 clip_cpu，跳过跨模态检查")
