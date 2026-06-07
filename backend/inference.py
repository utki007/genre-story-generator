"""Streaming and batch inference helpers."""

from __future__ import annotations

import math
import re
from typing import Generator, Iterable

import torch
import torch.nn.functional as F

from model.gpt import GPTLanguageModel
from model.tokenizer import Tokenizer

# Shakespeare script speaker cues: "\n\nMERCUTIO:" or "\n\nSecond Murderer:"
_SPEAKER_BREAK = re.compile(r"\n\n+([^\n:]{1,48}):\s*")


def _looks_like_speaker(name: str) -> bool:
    name = name.strip()
    if not name or len(name) > 48:
        return False
    words = [w for w in re.split(r"\s+", name) if w]
    if not words:
        return False
    # ALL CAPS or Title Case (typical TinyShakespeare speaker lines)
    if name.isupper():
        return True
    return all(w[0].isupper() for w in words if w[0].isalpha())


def format_prompt(prompt: str, character: str | None) -> str:
    if not character:
        return prompt.strip()
    prefix = f"{character.upper()}:"
    stripped = prompt.strip()
    if not stripped:
        return f"{prefix}\n"
    # Only skip re-prefixing when the final line is already this character's cue.
    last_line = stripped.split("\n")[-1].strip()
    if last_line.upper().startswith(prefix):
        return stripped
    return f"{prefix} {stripped}"


def strip_leading_speaker_cue(text: str, character: str | None = None) -> str:
    """Remove leaked speaker prefixes from model output (e.g. 'IET:' for JULIET)."""
    text = text.lstrip("\n\r \t")
    if not text:
        return text

    char_upper = (character or "").upper()
    if char_upper:
        prefix = f"{char_upper}:"
        if text.upper().startswith(prefix):
            return text[len(prefix) :].lstrip("\n\r \t")
        # Partial suffix leaks when the prompt already ends with "JULIET:" etc.
        for i in range(1, len(char_upper)):
            suffix = char_upper[i:] + ":"
            if text.upper().startswith(suffix):
                return text[len(suffix) :].lstrip("\n\r \t")

    match = re.match(r"^([^\n:]{1,48}):\s*", text)
    if match and _looks_like_speaker(match.group(1)):
        return text[match.end() :].lstrip("\n\r \t")

    return text


def _decode_generated(tokenizer: Tokenizer, idx: torch.Tensor, start_len: int) -> str:
    """Decode only newly generated tokens; avoid BPE boundary artifacts."""
    full = tokenizer.decode(idx[0].tolist())
    prefix = tokenizer.decode(idx[0, :start_len].tolist())
    if full.startswith(prefix):
        return full[len(prefix) :]
    return tokenizer.decode(idx[0, start_len:].tolist())


MIN_PROMPT_TOKENS = 32


def _fit_context_window(
    encoded: list[int], block_size: int, max_new_tokens: int
) -> tuple[list[int], int]:
    """Fit prompt and generation budget inside the model context window."""
    if not encoded:
        raise ValueError("Prompt encoding is empty")

    max_new = min(max_new_tokens, block_size - 1)
    min_prompt = min(MIN_PROMPT_TOKENS, block_size - 1)
    if max_new > block_size - min_prompt:
        max_new = max(1, block_size - min_prompt)

    max_prompt = max(1, block_size - max_new)
    if len(encoded) > max_prompt:
        encoded = encoded[-max_prompt:]
    return encoded, max_new


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k, dim=-1)
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


def trim_at_speaker_boundary(text: str, character: str) -> str:
    """Cut off when the model hands the floor to another speaker (or repeats a cue)."""
    match = _SPEAKER_BREAK.search(text)
    if match:
        speaker = match.group(1).strip()
        if _looks_like_speaker(speaker):
            return text[: match.start()].rstrip()

    # Stop before a partial cue at end-of-stream, e.g. "\n\nMERCUTIO"
    tail = re.search(r"\n\n+([^\n:]{1,48})$", text)
    if tail:
        speaker = tail.group(1).strip()
        if _looks_like_speaker(speaker):
            return text[: tail.start()].rstrip()

    return text


def should_stop_generation(generated: str, character: str, min_chars: int = 30) -> bool:
    if trim_at_speaker_boundary(generated, character) != generated:
        return True
    # End of a completed speech turn (blank line after dialogue).
    if len(generated) >= min_chars and generated.endswith("\n\n"):
        return True
    return False


def downsample_attention(attn: torch.Tensor, size: int = 8) -> list[list[float]]:
    """Average-pool attention matrix to size x size, normalized 0–1."""
    if attn is None or attn.numel() == 0:
        return [[0.0] * size for _ in range(size)]
    matrix = attn.float().cpu()
    t = matrix.size(0)
    if t == 0:
        return [[0.0] * size for _ in range(size)]
    if t < size:
        pad = torch.zeros(size - t, t)
        matrix = torch.cat([matrix, pad], dim=0)
        pad_col = torch.zeros(size, size - t)
        matrix = torch.cat([matrix, pad_col], dim=1)
    pooled = torch.zeros(size, size)
    block_h = max(1, matrix.size(0) // size)
    block_w = max(1, matrix.size(1) // size)
    for i in range(size):
        for j in range(size):
            h0, h1 = i * block_h, min((i + 1) * block_h, matrix.size(0))
            w0, w1 = j * block_w, min((j + 1) * block_w, matrix.size(1))
            pooled[i, j] = matrix[h0:h1, w0:w1].mean()
    mx = pooled.max().item()
    if mx > 0:
        pooled = pooled / mx
    return pooled.tolist()


@torch.no_grad()
def get_top_tokens(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    top_k: int = 10,
    character: str | None = None,
) -> list[dict]:
    text = format_prompt(prompt, character)
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    idx_cond = idx[:, -model.block_size :]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)[0]
    k = min(top_k, probs.size(0))
    values, indices = torch.topk(probs, k)
    return [
        {"token": tokenizer.decode_token(int(indices[i])), "prob": float(values[i])}
        for i in range(k)
    ]


@torch.no_grad()
def _generate_loop(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    idx: torch.Tensor,
    device: str,
    start_len: int,
    character: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    capture_attn: bool = False,
) -> tuple[str, list[float], torch.Tensor | None]:
    logprobs: list[float] = []
    last_attn: torch.Tensor | None = None
    generated = ""

    model.eval()
    for _ in range(max_new_tokens):
        if idx.size(1) == 0:
            break
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond, capture_attn=capture_attn)
        if capture_attn:
            attn = model.get_last_attention()
            if attn is not None:
                last_attn = attn.clone()

        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = _apply_top_k(logits, top_k)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        next_id = torch.multinomial(probs, num_samples=1)
        logprobs.append(log_probs[0, next_id.item()].item())
        idx = torch.cat([idx, next_id], dim=1)

        generated = _decode_generated(tokenizer, idx, start_len)
        generated = trim_at_speaker_boundary(generated, character)
        generated = strip_leading_speaker_cue(generated, character)

        if should_stop_generation(generated, character):
            break

    return generated, logprobs, last_attn


@torch.no_grad()
def stream_generate(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    top_k: int = 40,
    character: str | None = None,
) -> Generator[dict, None, None]:
    text = format_prompt(prompt, character or "")
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    start_len = idx.size(1)
    logprobs: list[float] = []
    last_attn: torch.Tensor | None = None
    prev_len = 0

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond, capture_attn=True)
        attn = model.get_last_attention()
        if attn is not None:
            last_attn = attn.clone()

        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = _apply_top_k(logits, top_k)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        next_id = torch.multinomial(probs, num_samples=1)
        token_logprob = log_probs[0, next_id.item()].item()
        logprobs.append(token_logprob)
        idx = torch.cat([idx, next_id], dim=1)

        raw = _decode_generated(tokenizer, idx, start_len)
        trimmed = strip_leading_speaker_cue(
            trim_at_speaker_boundary(raw, character or ""),
            character or "",
        )
        new_text = trimmed[prev_len:]
        prev_len = len(trimmed)

        if new_text:
            running_ppl = math.exp(-sum(logprobs) / len(logprobs))
            yield {"token": new_text, "logprob": token_logprob, "perplexity": running_ppl}

        if should_stop_generation(trimmed, character or ""):
            break

    full_ppl = math.exp(-sum(logprobs) / len(logprobs)) if logprobs else 1.0
    yield {
        "done": True,
        "full_perplexity": full_ppl,
        "attn_weights": downsample_attention(last_attn),
    }


@torch.no_grad()
def arena_generate(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    characters: Iterable[str],
    device: str,
    max_new_tokens: int = 150,
    temperature: float = 0.5,
    top_k: int = 40,
) -> list[dict]:
    results = []
    for character in characters:
        char_prompt = format_prompt(prompt, character)
        idx = torch.tensor([tokenizer.encode(char_prompt)], dtype=torch.long, device=device)
        start_len = idx.size(1)
        generated, logprobs, _ = _generate_loop(
            model,
            tokenizer,
            idx,
            device,
            start_len,
            character,
            max_new_tokens,
            temperature,
            top_k,
        )
        ppl = math.exp(-sum(logprobs) / len(logprobs)) if logprobs else 1.0
        results.append(
            {
                "character": character.upper(),
                "text": generated,
                "perplexity": round(ppl, 2),
            }
        )
    return results


@torch.no_grad()
def roundtable_generate(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    topic: str,
    characters: list[str],
    device: str,
    turns: int = 2,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
) -> Generator[dict, None, None]:
    """Generate a multi-character roundtable discussion about a topic.

    Yields events: {character, token} for streaming, then {character, done, text}
    when each character finishes their turn.
    """
    context = topic.strip()

    for turn in range(turns):
        for character in characters:
            char = character.upper()
            prompt_text = f"{context}\n\n{char}:\n"
            encoded = tokenizer.encode(prompt_text)
            encoded, turn_max_tokens = _fit_context_window(
                encoded, model.block_size, max_new_tokens
            )
            idx = torch.tensor([encoded], dtype=torch.long, device=device)
            start_len = idx.size(1)

            generated, logprobs, _ = _generate_loop(
                model, tokenizer, idx, device, start_len,
                char, turn_max_tokens, temperature, top_k,
            )

            generated = generated.strip()
            if not generated:
                generated = "..."

            yield {"character": char, "text": generated, "done": True, "turn": turn}

            context += f"\n\n{char}: {generated}"
