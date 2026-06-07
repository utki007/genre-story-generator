"""Load GPT checkpoint and tokenizer from data/artifacts."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from .gpt import GPTLanguageModel
from .tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "bpe_tokenizer"
CHECKPOINT_PATH = ARTIFACTS_DIR / "model" / "gpt_best.pt"
CHARACTERS_PATH = ARTIFACTS_DIR / "selected_characters.json"


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tokenizer_ready(tokenizer_dir: Path) -> bool:
    if not tokenizer_dir.is_dir():
        return False
    if (tokenizer_dir / "tokenizer.json").exists():
        return True
    return (tokenizer_dir / "vocab.json").exists() and (tokenizer_dir / "merges.txt").exists()


def load_model_and_tokenizer(
    checkpoint_path: Path | None = None,
    tokenizer_dir: Path | None = None,
    device: str | None = None,
):
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    tokenizer_dir = tokenizer_dir or TOKENIZER_DIR
    device = device or _resolve_device()

    if not _tokenizer_ready(tokenizer_dir):
        raise FileNotFoundError(
            f"Missing {tokenizer_dir}. Run notebook 2 first."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run notebooks 3–5 first."
        )

    tokenizer = Tokenizer(tokenizer_dir)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = copy.deepcopy(payload["config"])
    config.setdefault("batch_size", 64)
    config.setdefault("use_amp", device == "cuda")

    model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, config=config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    return model, tokenizer, config, device
