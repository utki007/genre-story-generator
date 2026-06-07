"""FastAPI inference server for Shakespeare Studio."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference import arena_generate, get_top_tokens, roundtable_generate, stream_generate
from model.loader import ARTIFACTS_DIR, CHARACTERS_PATH, load_model_and_tokenizer
from schemas import (
    ArenaRequest,
    ArenaResponse,
    ArenaResult,
    GenerateRequest,
    RoundtableRequest,
    TokenProb,
    TopTokensRequest,
    TopTokensResponse,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, tokenizer, config, device = load_model_and_tokenizer()
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["config"] = config
        state["device"] = device
        state["loaded"] = True
        state["error"] = None
    except FileNotFoundError as exc:
        state["loaded"] = False
        state["error"] = str(exc)
    yield


app = FastAPI(title="Shakespeare Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if ARTIFACTS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(ARTIFACTS_DIR)), name="static")


def _require_model():
    if not state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail=state.get("error", "Model not loaded. Run notebooks 2–5 first."),
        )
    return state["model"], state["tokenizer"], state["device"]


@app.get("/health")
def health():
    tokenizer = state.get("tokenizer")
    return {
        "status": "ok" if state.get("loaded") else "degraded",
        "model_loaded": bool(state.get("loaded")),
        "tokenizer": getattr(tokenizer, "tokenizer_type", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "error": state.get("error"),
        "device": state.get("device"),
    }


@app.get("/characters")
def characters():
    if not CHARACTERS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing {CHARACTERS_PATH}. Run notebook 1 first.",
        )
    with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/generate")
async def generate(req: GenerateRequest):
    model, tokenizer, device = _require_model()

    async def event_stream():
        for event in stream_generate(
            model,
            tokenizer,
            req.prompt,
            device,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            character=req.character,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/top_tokens", response_model=TopTokensResponse)
def top_tokens(req: TopTokensRequest):
    model, tokenizer, device = _require_model()
    tokens = get_top_tokens(
        model,
        tokenizer,
        req.prompt,
        device,
        top_k=req.top_k,
        character=req.character,
    )
    return TopTokensResponse(tokens=[TokenProb(**t) for t in tokens])


@app.post("/arena", response_model=ArenaResponse)
def arena(req: ArenaRequest):
    model, tokenizer, device = _require_model()
    results = arena_generate(
        model,
        tokenizer,
        req.prompt,
        req.characters,
        device,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return ArenaResponse(results=[ArenaResult(**r) for r in results])


@app.post("/roundtable")
async def roundtable(req: RoundtableRequest):
    model, tokenizer, device = _require_model()

    async def event_stream():
        for event in roundtable_generate(
            model,
            tokenizer,
            req.prompt,
            req.characters,
            device,
            turns=req.turns,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'finished': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
