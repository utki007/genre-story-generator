"""Character-level tokenizer loaded from char_vocab.json."""

from __future__ import annotations

import json
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab_path: Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.vocab_size = payload["vocab_size"]
        self.stoi: dict[str, int] = payload["stoi"]
        self.itos: dict[int, str] = {int(k): v for k, v in payload["itos"].items()}

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi.get("<UNK>", 0)
        return [self.stoi.get(ch, unknown) for ch in text]

    def decode(self, indices) -> str:
        return "".join(self.itos[int(i)] for i in indices)

    def decode_token(self, index: int) -> str:
        return self.itos[int(index)]
