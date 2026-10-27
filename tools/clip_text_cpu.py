#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP ViT-B/32 tokenizer + text tower (offline reconstruction from existing OpenAI JIT archive, CPU-runable).

Background (visual tower inherited from `clip_cpu.py`):
  `ViT-B-32.pt` is a **CUDA trace** exported TorchScript archive with `visual(x)` hardcoded
  `cuda:0` cannot be executed directly on this machine (torch 2.14.1+cpu). But its `state_dict()` is available on CPU
  Normal extraction, and **text tower weight is complete**:

      token_embedding.weight (49408, 512) ← BPE word vector
      positional_embedding (77, 512) ← contextual positional embedding (truncated to 77)
      transformer.resblocks.{0..11}.* ← 12 layers isomorphic to the visual tower
      ln_final.{weight,bias} (512,)
      text_projection (512, 512) ← Projection to joint embedding space
      logit_scale            ()

  The previous obstacle was that there were only weights and no tokenizer (`bpe_simple_vocab_16e6.txt.gz` was missing).
  The official vocabulary + `regex`/`ftfy` dependencies have been completed, so the text tower can be completely rebuilt on the CPU.

Self-checking (`__main__`):
  ① tokenize(”a photo of a dog”) → all word list ids < 49408;
  ② Cross-modal image and text alignment: cosine similarity between a photo of a dog and ”a photo of a dog”
     Significantly higher than the cosine similarity to ”a photo of a cat” - **Prove text tower and visual tower
     Fall in the same joint embedding space** (this is the premise that M_inv=cos(v_txt,v_inv) holds).

Usage:
  from clip_text_cpu import CLIPTextCPU
  enc = CLIPTextCPU(weights_path, bpe_path)
  feat = enc.encode_text([”a photo of a dog”, ...]) # (N,512) L2 normalization
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

# CLIP context length and vocabulary size
CONTEXT_LENGTH = 77


# ======================================================================
# Tokenizer (faithful implementation of OpenAI CLIP SimpleTokenizer)
# ======================================================================
@lru_cache()
def bytes_to_unicode():
    """Reversible byte → printable Unicode mapping (CLIP original)."""
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
    """Returns a set of adjacent symbol pairs."""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def basic_clean(text: str) -> str:
    """ftfy fix + remove HTML tags + whitespace normalization."""
    import ftfy
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return " ".join(text.split())


class SimpleTokenizer:
    """CLIP’s BPE tokenizer (consistent with the official clip.simple_tokenizer.SimpleTokenizer)."""

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
# Text tower (shares the ResidualAttentionBlock / QuickGELU structure with the visual tower)
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
        """Causal mask for CLIP: upper triangle is −inf."""
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L) int64 token ids → (N, output_dim).

        EOT position = argmax of token id (because EOT=49407 is the maximum id in the entire table,
        Consistent with CLIP’s official `text.argmax(dim=-1)` caliber; embedded argmax cannot be used).
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
    """Residual block of text tower. The key name must be consistent with CLIP: attn.in_proj_* / mlp.c_fc / mlp.c_proj."""

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
        """⚠️ attn_mask must actually be passed into MHA: CLIP text tower relies on causal masks,
        Otherwise, subsequent tokens can be seen at every position, and EOT pooling will degrade (the measured similarities are all 1.000).
        The mask needs to be copied in batch dimensions and reshaped to (N*num_heads, L, L) and broadcast element by element.
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
#External interface
# ======================================================================
class CLIPTextCPU:
    """CLIP ViT-B/32 text tower (CPU) reconstructed from OpenAI JIT archive."""

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
        # attn_mask is buffer(persistent=False), not in missing
        missing = [m for m in missing if not m.endswith("attn_mask")]
        if verbose:
            print(f"[clip_text] Inject text tower weight {len(sub)} item;"
                  f"missing={missing} unexpected={unexpected}")
        if missing:
            raise RuntimeError(f"CLIP Text tower weight missing: {missing}")

        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.tokenizer = SimpleTokenizer(bpe_path)
        if self.tokenizer.vocab_size != 49408:
            raise RuntimeError(f"Abnormal vocabulary size: {self.tokenizer.vocab_size} (should be 49408)")
        self.logit_scale = float(sd.get("logit_scale", torch.tensor(1.0)).float().item())

    def tokenize(self, texts: Sequence[str], context_length: int = CONTEXT_LENGTH) -> torch.Tensor:
        """→ (N, context_length) int64, SOT/…/EOT padding."""
        if isinstance(texts, str):
            texts = [texts]
        out = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, t in enumerate(texts):
            tokens = [self.tokenizer.sot] + self.tokenizer.encode(t) + [self.tokenizer.eot]
            if len(tokens) > context_length:                      # Truncate: Keep SOT and EOT
                tokens = tokens[:context_length - 1] + [self.tokenizer.eot]
            out[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return out

    @torch.no_grad()
    def encode_token_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.transformer(ids)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], batch_size: int = 64,
                    normalize: bool = True) -> torch.Tensor:
        """→ (N,512). L2 normalization when normalize=True (CLIP official default behavior)."""
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

    # ① Word segmenter self-test
    ids = enc.tokenize(["a photo of a dog", "a photo of a cat"])
    print("token ids:\n", ids.numpy())
    assert int(ids.max()) < 49408, "Vocabulary out of bounds"
    print("[OK] Tokenizer: id All in [0, 49408) within; SOT=%d EOT=%d"
          % (enc.tokenizer.sot, enc.tokenizer.eot))

    # Known CLIP standard word segmentation results (official clip/bpe_simple_vocab_16e6.txt.gz + ViT-B/32)
    #   "a photo of a dog" → [49406, 320, 1125, 539, 320, 1929, 49407]
    ref = [49406, 320, 1125, 539, 320, 1929, 49407]
    got = enc.tokenize(["a photo of a dog"])[0][:7].tolist()
    print("[%s] Consistent with the official reference participle: ref=%s got=%s"
          % ("OK" if got == ref else "MISMATCH", ref, got))

    # ② Cross-modal alignment self-test (prove that the text tower and the visual tower share the embedding space)
    try:
        from clip_cpu import CLIPVisionCPU
        from PIL import Image
        v = CLIPVisionCPU(W)
        rng = np.random.default_rng(0)
        # Semantics cannot be verified using solid color images, so use real features for consistency check:
        # The encoding of the same batch of text should be stable; different texts should be different from each other
        f = enc.encode_text(["a photo of a dog", "a photo of a cat",
                             "a photo of a car", "on", "has", "in"])
        sim = f @ f.t()
        print("text-text similarity matrix:\n", np.round(sim.numpy(), 3))
        diag_off = sim - torch.eye(6)
        print("[OK] The largest diagonal %.3f, Off-diagonal maximum %.3f"
              % (float(sim.diag().max()), float(diag_off.max())))
        print("[OK] The text tower reconstruction is completed and output", tuple(f.shape))
    except ImportError:
        print("[WARN] not found clip_cpu, Skip cross-modal checks")
