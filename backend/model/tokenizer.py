"""GPT-2 BPE tokenizer loaded from data/artifacts/bpe_tokenizer/."""

from __future__ import annotations

from pathlib import Path

from transformers import GPT2Tokenizer


class Tokenizer:
    def __init__(self, tokenizer_dir: Path):
        self._tok = GPT2Tokenizer.from_pretrained(str(tokenizer_dir))
        self.vocab_size = self._tok.vocab_size
        self.tokenizer_type = "gpt2_bpe"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, indices) -> str:
        return self._tok.decode(list(indices))

    def decode_token(self, index: int) -> str:
        return self._tok.decode([int(index)])
