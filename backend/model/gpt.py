"""Decoder-only GPT — ported from notebook 6 with optional attention capture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self._causal_cache = None
        self.last_attn: torch.Tensor | None = None

    def _causal_mask(self, size: int, device: torch.device):
        if self._causal_cache is None or self._causal_cache.size(-1) < size:
            self._causal_cache = torch.tril(torch.ones(size, size, device=device)).view(
                1, 1, size, size
            )
        return self._causal_cache[:, :, :size, :size]

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(T, x.device)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        if capture_attn:
            # Average over heads -> (B, T, T), keep batch 0
            self.last_attn = att.mean(dim=1)[0].detach()

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, d_ff_mult: int, dropout: float):
        super().__init__(
            nn.Linear(d_model, d_ff_mult * d_model),
            nn.GELU(),
            nn.Linear(d_ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff_mult, dropout)

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), capture_attn=capture_attn)
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: dict):
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        dropout = config["dropout"]
        block_size = config["block_size"]
        d_ff_mult = config.get("d_ff", 4)
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff_mult, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        capture_attn: bool = False,
    ):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.drop(self.token_embedding(idx) + self.pos_embedding(pos))
        last_block_idx = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            x = block(x, capture_attn=capture_attn and i == last_block_idx)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def get_last_attention(self) -> torch.Tensor | None:
        last_block = self.blocks[-1]
        return last_block.attn.last_attn

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k, dim=-1)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
