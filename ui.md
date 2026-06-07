# Shakespeare Studio — React UI

A research-grade, character-conditioned Shakespeare dialogue interface built on top of the GPT pipeline from this repo. Three-panel layout: character selector, streaming conversation, live model diagnostics.

---

## What it looks like

```
┌─────────────────────────────────────────────────────────────────────┐
│  🪶 Shakespeare Studio    [Chat] [Scene Generator] [Arena] [Explorer]│
├──────────────┬──────────────────────────────┬───────────────────────┤
│  Characters  │                              │   Diagnostics         │
│              │   HAMLET: —that is the       │                       │
│  ✓ Hamlet    │   question: Whether 'tis     │   Perplexity   3.2    │
│  ✓ Ophelia   │   nobler in the mind…        │   ████░░░░░░          │
│              │                              │                       │
│  ✓ Romeo     │   > What would Romeo say     │   Temperature  0.8 ── │
│  ○ Juliet    │     about death?             │   Top-k        40  ── │
│              │                              │                       │
│  ○ Macbeth   │   ROMEO: Death, that hath    │   Top tokens          │
│              │   suck'd the honey of thy    │   "that"  ████  72%   │
│              │   breath…                    │   "this"  █░░   14%   │
│              │                              │   "the"   ░░░    8%   │
│              │ [Hamlet ▾] [Enter prompt…] ▶ │                       │
│              │                              │   Attention heatmap   │
│              │                              │   ░▓█░▒▓ ▓░█▒░▓      │
└──────────────┴──────────────────────────────┴───────────────────────┘
```

Token color coding in generated text: **warm (amber)** = high surprise, **cool (teal)** = low surprise, based on per-token log-probabilities. Streaming tokens are **GPT-2 BPE subwords** (e.g. `"that"`, `" is"`) — not individual characters.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| UI framework | React 18 | Hooks + context |
| Styling | Tailwind CSS v3 | Responsive utilities |
| Charts | Recharts | Top-k token bars |
| Streaming | SSE (`EventSource`) | Token-by-token typewriter output |
| State | Zustand | Lightweight; no Redux overhead |
| HTTP | Axios | REST calls to FastAPI |
| Build | Vite | Fast HMR |
| Backend | FastAPI | Inference wrapper around `gpt_best.pt` + `bpe_tokenizer/` |

---

## Project structure

```
shakespeare-studio/
├── public/
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── TopBar.jsx           # Tab navigation + model badge
│   │   │   ├── LeftPanel.jsx        # Character selector grouped by play
│   │   │   └── RightPanel.jsx       # Diagnostics panel
│   │   ├── chat/
│   │   │   ├── ChatArea.jsx         # Message list with token heatmap
│   │   │   ├── MessageBubble.jsx    # Single message: user or character
│   │   │   ├── TokenSpan.jsx        # Per-token colored span from logprobs
│   │   │   └── InputBar.jsx         # Character pill + text input + send
│   │   ├── diagnostics/
│   │   │   ├── PerplexityMeter.jsx  # Live ppl + bar
│   │   │   ├── SamplingSliders.jsx  # Temperature + top-k sliders
│   │   │   ├── TopKTokens.jsx       # Recharts bar for next-token dist
│   │   │   └── AttentionGrid.jsx    # Last-layer attention heatmap
│   │   └── tabs/
│   │       ├── SceneGenerator.jsx   # Multi-character scene builder
│   │       ├── Arena.jsx            # Side-by-side character comparison
│   │       └── ModelExplorer.jsx    # Per-character perplexity, param stats
│   ├── hooks/
│   │   ├── useSSEStream.js          # EventSource hook with token callbacks
│   │   └── useInference.js          # REST calls for non-streaming endpoints
│   ├── store/
│   │   └── studioStore.js           # Zustand: selected chars, messages, params
│   ├── api/
│   │   └── client.js                # Axios base + endpoint helpers
│   ├── constants/
│   │   └── characters.js            # Character list with play groupings
│   ├── App.jsx
│   └── main.jsx
├── .env.example
├── vite.config.js
├── tailwind.config.js
└── package.json
```

---

## FastAPI backend contract

The React app expects these endpoints. Wire them to your existing inference code from notebooks 3–5.

### `POST /generate` — streaming (SSE)

```http
POST /generate
Content-Type: application/json

{
  "prompt": "HAMLET: To be or not to be",
  "character": "HAMLET",
  "temperature": 0.8,
  "top_k": 40,
  "max_new_tokens": 200
}
```

**Response:** `text/event-stream`

```
data: {"token": "—", "logprob": -0.12, "perplexity": 1.13}
data: {"token": "that", "logprob": -0.85, "perplexity": 2.34}
data: {"token": " is", "logprob": -0.21, "perplexity": 1.23}
...
data: {"done": true, "full_perplexity": 3.2, "attn_weights": [[...]]}
```

Return `attn_weights` only on the final `done` event — it's the averaged last-layer attention over the full sequence (shape: `[seq_len, seq_len]`, downsample to 8×8 before sending). Each streamed `token` is a decoded BPE subword string from GPT-2 tokenization.

### `GET /health` — model and tokenizer status

```json
{
  "status": "ok",
  "model_loaded": true,
  "tokenizer": "gpt2_bpe",
  "vocab_size": 50257,
  "device": "cpu",
  "error": null
}
```

The frontend calls this on mount to show a status badge and disable generation when artifacts are missing.

### `POST /top_tokens` — next-token distribution

```http
POST /top_tokens
Content-Type: application/json

{
  "prompt": "HAMLET: To be or not to be—that",
  "character": "HAMLET",
  "top_k": 10
}
```

```json
{
  "tokens": [
    {"token": "that", "prob": 0.72},
    {"token": "this", "prob": 0.14},
    {"token": "the",  "prob": 0.08},
    {"token": "a",    "prob": 0.04}
  ]
}
```

### `POST /arena` — parallel generation (for Arena tab)

```http
POST /arena
Content-Type: application/json

{
  "prompt": "What is love?",
  "characters": ["HAMLET", "ROMEO"],
  "temperature": 0.8,
  "top_k": 40,
  "max_new_tokens": 150
}
```

```json
{
  "results": [
    {"character": "HAMLET", "text": "...", "perplexity": 3.1},
    {"character": "ROMEO",  "text": "...", "perplexity": 4.2}
  ]
}
```

---

## FastAPI setup (minimal)

Wrap your existing notebook inference code:

```python
# api/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load once at startup
model = torch.load("data/artifacts/model/gpt_best.pt", map_location="cpu")
model.eval()

@app.post("/generate")
async def generate(req: GenerateRequest):
    async def token_stream():
        for token, logprob in model.stream(req.prompt, req.temperature, req.top_k):
            yield f"data: {json.dumps({'token': token, 'logprob': logprob})}\n\n"
        attn = model.last_attn_weights()  # hook into your forward pass
        yield f"data: {json.dumps({'done': True, 'attn_weights': attn})}\n\n"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

---

## Key component details

### `useSSEStream.js`

```js
export function useSSEStream() {
  const appendToken = useStudioStore(s => s.appendToken);

  const stream = (prompt, character, params) => {
    const es = new EventSource(
      `/generate?` + new URLSearchParams({ prompt, character, ...params })
    );
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.done) {
        es.close();
        useStudioStore.getState().finalizeMessage(data);
      } else {
        appendToken(data.token, data.logprob);
      }
    };
  };
  return { stream };
}
```

### `TokenSpan.jsx` — per-token color from logprob

```jsx
// logprob range: 0 (certain) → -∞ (surprised)
// Map to: low surprise = teal, high surprise = amber
function logprobToColor(lp) {
  const surprise = Math.min(Math.abs(lp) / 5, 1); // normalize 0–1
  if (surprise > 0.6) return 'bg-amber-100 text-amber-800';
  if (surprise > 0.3) return 'bg-teal-50  text-teal-800';
  return '';
}

export function TokenSpan({ token, logprob }) {
  return <span className={`rounded px-0.5 ${logprobToColor(logprob)}`}>{token}</span>;
}
```

### `AttentionGrid.jsx`

```jsx
// Expects attn: number[][] (8×8 after downsampling server-side)
export function AttentionGrid({ attn }) {
  return (
    <div className="grid grid-cols-8 gap-0.5">
      {attn.flat().map((v, i) => (
        <div
          key={i}
          className="h-3 rounded-sm"
          style={{ background: `rgba(127,119,221,${v.toFixed(2)})` }}
          title={`${Math.round(v * 100)}%`}
        />
      ))}
    </div>
  );
}
```

---

## Responsive breakpoints

The three-panel layout collapses gracefully on smaller screens:

| Breakpoint | Layout |
|---|---|
| `≥ 1280px` (xl) | Full 3-panel: left 200px + center flex + right 220px |
| `≥ 768px` (md) | 2-panel: character selector hidden behind toggle, diagnostics in drawer |
| `< 768px` (sm) | Single column: tabs → chat → diagnostics stacked vertically |

```jsx
// App.jsx skeleton
<div className="grid xl:grid-cols-[200px_1fr_220px] md:grid-cols-[1fr_200px] grid-cols-1 h-screen">
  <LeftPanel className="hidden md:block xl:block" />
  <CenterPanel />
  <RightPanel className="hidden xl:block" />
</div>
```

On `md`, add a `<CharacterDrawer>` (slide-in sheet) triggered by a button in the top bar. On `sm`, render diagnostics below the chat area as a collapsible accordion.

---

## Getting started

```bash
# 1. Run the full ML pipeline first (notebooks 1–5; notebook 2 writes bpe_tokenizer/)
python src/ingest-data.py

# 2. Start the FastAPI backend
cd api
pip install fastapi uvicorn torch
uvicorn main:app --reload --port 8000

# 3. Start the React dev server
cd shakespeare-studio
npm install
npm run dev          # http://localhost:5173
```

```bash
# .env.example
VITE_API_BASE_URL=http://localhost:8000
```

---

## Characters available

Pulled from `data/artifacts/selected_characters.json` (output of notebook 1):

| Play | Characters |
|---|---|
| Hamlet | HAMLET, OPHELIA, POLONIUS, HORATIO |
| Romeo & Juliet | ROMEO, JULIET, MERCUTIO, NURSE |
| Macbeth | MACBETH, LADY MACBETH |
| King Lear | LEAR, CORDELIA |
| Othello | OTHELLO, IAGO, DESDEMONA |

Characters with fewer than ~500 lines are excluded from conditioning (insufficient training signal — see `speaker_stats.csv`).

---

## Tab reference

| Tab | What it does |
|---|---|
| Chat | Single-character streaming dialogue with live diagnostics |
| Scene generator | Multi-turn improv: user assigns roles, model plays other characters |
| Character arena | Same prompt → parallel generation from 2 selected characters side-by-side |
| Model explorer | Per-character perplexity from NB6 `evaluation_report.json`, parameter efficiency plot, overfitting diagnostics |

---

## Connecting to evaluation artifacts

The Model Explorer tab reads directly from notebook 6 outputs:

```js
// src/api/client.js
export const getEvalReport = () =>
  axios.get('/static/evaluation_report.json');  // serve from data/artifacts/model/
```

Expose `data/artifacts/` as a static directory in FastAPI:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="data/artifacts"), name="static")
```

---
