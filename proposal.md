# MLOps-Driven Shakespearean Character Chatbot
## GPT From Scratch

Build a scalable, experiment-driven **generative AI system** that learns to speak as individual Shakespearean characters, culminating in an interactive React-based chat interface where multiple characters converse in real time. The project centers on a decoder-only Transformer built entirely from scratch, with systematic experimentation and reproducibility under modern MLOps practices.

Unlike a one-off tutorial, this project emphasizes **systematic experimentation**, **reproducibility**, and **interactive deployment** following modern MLOps practices, with all implementation tracked via Weights & Biases and served through a locally-hosted React + FastAPI application.

---

## 1. Overview

The system supports controlled experimentation, automated hyperparameter tuning, and structured evaluation for a generative model trained on character-attributed Shakespearean dialogue. The final deliverable is a web UI where users select characters and observe AI-generated multi-turn dialogue — each character's voice shaped by its training distribution.

---

## 2. Dataset

- **Source**: `karpathy/tiny_shakespeare` (Hugging Face / Karpathy repo)
- **Size**: ~1M characters; ~40,000 lines across 40+ named characters.
- **Preprocessing Pipeline**:
  - **Character Extraction**: Parse dialogue by speaker (e.g., `ROMEO:`, `HAMLET:`, `KING HENRY VI:`) into per-character corpora.
  - **Tokenization**: Character-level tokenizer (vocab ~66).
  - **Chunking**: Fixed `block_size` context windows (e.g., 128 or 256 tokens) with character-identity prefix prepended to each chunk.
  - **Conditioning Format**: Prompts structured as `[CHARACTER]: {dialogue}` to enable character-conditioned generation at inference.
  - **DataLoaders**: PyTorch `DataLoader` for randomized batched tensor generation.

---

## 3. Problem Formulation

- **Input**: A character-conditioned prefix, e.g., `"ROMEO: "` or `"HAMLET: To be"`.
- **Output**: Autoregressively generated continuation in that character's voice.

Modeled as causal language modeling with character conditioning:

$$P(x_t \mid x_{<t},\, c) \quad \text{where } c \in \{\text{ROMEO, HAMLET, \ldots}\}$$

The character identity $c$ is injected as a text prefix — not a learned control token.

---

## 4. Model Architecture

### 4.1 GPT From Scratch (PyTorch)

A **decoder-only Transformer** built entirely from scratch:

- **Embeddings**: Token embeddings + learned positional embeddings.
- **Transformer Blocks**: Causal (masked) multi-head self-attention → position-wise FFN (4× hidden dim expansion) → pre-norm LayerNorm + residual connections.
- **Self-Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **LM Head**: Linear projection → vocabulary logits (weight-tied with token embedding).
- **Parameter range**: ~1M–10M depending on sweep config.

---

## 5. Training Strategy

Causal language modeling via cross-entropy loss:

$$L = -\sum_t \log P(x_t \mid x_{<t})$$

$$\text{Perplexity} = e^{\,L}$$

### Enhancements

- **Optimizer**: AdamW with weight decay.
- **LR Scheduling**: Cosine decay with linear warmup — compared against constant LR as a baseline.
- **Mixed Precision**: `torch.cuda.amp` for GPU throughput.
- **Gradient Clipping**: Stabilizes from-scratch attention training.
- **Character-Conditioned Batching**: Oversample low-frequency characters to balance voice quality across the cast.

---

## 6. MLOps Pipeline (Weights & Biases)

### 6.1 Experiment Tracking

**Logged Metrics**:
- Train / Validation Loss
- Validation Perplexity (global + per-character breakdown)
- Tokens/sec throughput
- Model parameter count and size (MB)

**Artifacts**:
- Per-epoch qualitative samples: fixed prompts per character logged as W&B Tables — tracks progression from random characters → English words → structured Shakespearean script format.

### 6.2 Hyperparameter Optimization

**W&B Sweeps (Bayesian search)** on the from-scratch model:

| Parameter | Range |
|---|---|
| Learning rate | `1e-4` – `1e-2` |
| `n_layers` | 4 – 8 |
| `d_model` | 128 – 512 |
| `n_heads` | 4 – 8 |
| Dropout | 0.1 – 0.3 |
| `block_size` | 128 / 256 |

**Objective**: Minimize validation loss; secondary metric is per-character perplexity variance (proxy for voice consistency). Bayesian search is used over grid search — more sample-efficient given the 6-dimensional space.

### 6.3 Model Versioning

W&B Artifacts track:
- Tokenizer mapping dictionaries.
- Versioned checkpoints: `scratch-baseline-v1`, `scratch-sweep-best-v2`, `scratch-final-v1`.
- Character–corpus index (for contamination-free reproducibility).

---

## 7. Notebook Structure

Seven notebooks, each with a single responsibility:

### Notebook 1 — Data Exploration (`1_data_exploration.ipynb`)

**Objectives**: Understand dataset characteristics before any modeling.

**Analysis**:
- Dataset statistics: total characters, vocabulary size, line count, average line length.
- Character (speaker) frequency distribution — identifies dominant vs. rare characters.
- Special character and punctuation inventory.
- Sequence length distribution across speakers.

**Visualizations**: Character histogram, word frequency plot, sequence length distribution, speaker dialogue share.

**Deliverables**: Vocabulary report, dataset summary, list of characters selected for conditioning.

---

### Notebook 2 — Preprocessing (`2_preprocessing.ipynb`)

**Objectives**: Convert raw text into model-ready tensors.

**Tasks**:
- **Char-level tokenizer**: Build `stoi`, `itos`, `encode()`, `decode()`.
- **Character extraction**: Regex-based speaker parsing → per-character text splits.
- **Train/Val split**: 90/10 by character chunk (not random line split — avoids context bleed).
- **Context windows**: Experiment with `block_size` ∈ {64, 128, 256}.
- **DataLoader pipeline**: Batched `(input, target)` tensor pairs.

**Deliverables**: Tokenized dataset, DataLoader pipeline, encoding/decoding verification.

---

### Notebook 3 — Transformer From Scratch (`3_baseline_gpt_scratch.ipynb`)

**Objectives**: Implement every component manually; establish baseline metrics.

**Components**:
- `Head` — single causal self-attention head.
- `MultiHeadAttention` — parallel heads with output projection.
- `FeedForward` — Linear → GELU → Linear (4× expansion).
- `TransformerBlock` — attention + FFN with pre-norm and residual.
- `GPTLanguageModel` — embedding + N blocks + LM head.

**Training**:
- Baseline config: `n_layers=4`, `d_model=128`, `n_heads=4`, `block_size=128`.
- AdamW optimizer; cross-entropy loss; W&B logging from epoch 1.
- Parameter count analysis across small / medium / large configs.

**Deliverables**: Complete Transformer implementation, baseline train/val curves, W&B run link.

---

### Notebook 4 — Hyperparameter Sweep (`4_hyperparameter_sweep.ipynb`)

**Objectives**: Identify optimal architecture via automated search.

**Setup**:
- W&B Sweep with Bayesian search over the parameter space in §6.2.
- LR schedule comparison: constant vs. cosine decay vs. warmup+cosine.
- Each run logs loss, perplexity, throughput, and parameter count.

**Analysis**:
- Hyperparameter importance plot (W&B built-in).
- Pareto frontier: val perplexity vs. parameter count.
- Identification of capacity ceiling — point at which train/val gap grows.

**Deliverables**: Hyperparameter results table, best configuration, W&B sweep dashboard link.

---

### Notebook 5 — Final Model Training (`5_model_training.ipynb`)

**Objectives**: Train the best scratch configuration on the full budget; compare against baseline and tuning results.

**Tasks**:
- Load winning hyperparameters from notebook 4.
- Warm-start from the tuning checkpoint when available.
- Full training run with AdamW + cosine LR; log to same W&B project.
- Side-by-side metric table: baseline vs. tuning best vs. final scratch.

**Deliverables**: Final checkpoint artifact, comparative metrics table, W&B run link.

---

### Notebook 6 — Model Evaluation (`6_model_evaluation.ipynb`)

**Objectives**: Rigorous quantitative and qualitative analysis of the trained model.

**Quantitative Metrics**:
- Validation loss and perplexity per character.
- Parameter efficiency: perplexity per million parameters across small/medium/large scratch configs.
- Train/val loss gap over training steps — overfitting diagnostic.

**Qualitative / Error Analysis**:
- Fixed prompt battery: `"ROMEO: "`, `"HAMLET: To be"`, `"KING HENRY VI: "`, `"JULIET: "`.
- Sample generations logged as W&B Table.
- Failure mode analysis:
  - Repetition loops (common at low temperature).
  - Script format breakdown (missing `CHARACTER:` structure).
  - Cross-character voice confusion (does ROMEO sound like HAMLET?).

**Deliverables**: Evaluation report, parameter efficiency table, failure mode examples.

---

### Notebook 7 — Inference, Sampling & Demo (`7_inference_and_demo.ipynb`)

**Objectives**: Production-ready generation with sampling controls; launch the chat UI backend.

**Sampling Methods**:

| Method | Config |
|---|---|
| Greedy | `argmax` |
| Temperature | `T` ∈ {0.7, 1.0, 1.2} |
| Top-K | `k` ∈ {10, 20, 50} |
| Top-P (nucleus) | `p` ∈ {0.9, 0.95} |

**Generation Gallery**: Same prompt across all sampling strategies — side-by-side comparison logged to W&B Tables.

**FastAPI Server Launch**:
- `POST /generate` — accepts `{character, prompt, max_tokens, temperature, top_p}`.
- `GET /characters` — returns character list with per-character perplexity metadata.

**Deliverables**: Sampling comparison gallery, running FastAPI server, React app connection verified.

---

## 8. Interactive Chat UI (React)

A locally-hosted React application — the primary demo deliverable.

**Features**:
- **Character Selection**: Panel to pick 2–4 active characters (e.g., ROMEO vs JULIET, HAMLET vs HORATIO).
- **Turn-Based Generation**: User provides an opening line; each character responds in sequence, with prior turn prepended as context.
- **Generation Controls**: Sliders for temperature and top-p.
- **Chat Transcript**: Color-coded conversation log with character name headers in Shakespearean styling.

**Architecture**:
- React frontend → FastAPI backend (local) → PyTorch inference.
- Stateless generation: full conversation history prepended per request.

---

## 9. Experimental Plan

| Experiment | Change | Expected Outcome |
|---|---|---|
| **Baseline** | Scratch: `n_layers=4`, `d_model=128`, `n_heads=4` | Learns word boundaries and script format; high val loss; generic voice. |
| **Sweep 1** | Tune LR & dropout | Faster convergence; reduced overfitting on small per-character corpora. |
| **Sweep 2** | Scale `d_model` and `n_layers` | Identifies capacity ceiling before memorization; best perplexity–diversity tradeoff. |
| **Final Run** | Best scratch config + cosine decay | Coherent character-attributed dialogue; deployed in React chat UI. |

---

## 10. Conclusion

This project delivers hands-on experience across the full generative AI stack: Transformer architecture design, training from scratch, MLOps instrumentation with W&B, and interactive deployment. The React chat interface transforms the trained model into a demonstrable, portfolio-ready Gen AI application with a natural multi-character conversational interface.
