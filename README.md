# Genre Story Generator

**Decoder-only GPT** (PyTorch, from scratch) trained on [TinyShakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) (~1M characters). The pipeline covers exploration → preprocessing → baseline → local hyperparameter search → final training → evaluation (with explainability) → interactive inference → deployment prototypes. All work lives in eight Jupyter notebooks under `notebooks/`, with metrics and checkpoints saved under `data/artifacts/`.

## Installation & run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data** — from repo root:

```bash
python src/ingest-data.py
```

**Notebooks** (run in order; launch Jupyter from repo root or `notebooks/`):

| # | Notebook |
|---|----------|
| 1 | Data Exploration |
| 2 | Preprocessing |
| 3 | Baseline Model Architecture |
| 4 | Hyperparameter Tuning |
| 5 | Model Training |
| 6 | Evaluation & Model Comparison (exact test ppl, baseline compare, XAI) |
| 7 | Inference & Interactive Generation |
| 8 | Deployment & Model Export |

Artifacts land in `data/artifacts/` (`char_vocab.json`, token tensors, `tuning_runs.json`, `best_hparams.json`, `gpt_best.pt`, `experiment.json`, `evaluation_report.json`). Figures go to `docs/`. Checkpoints are gitignored; re-run notebooks 2 → 5 to reproduce locally.

## Model results

Best config from notebook 4 Hyperband tuning (cached in `best_hparams.json`): **7 layers, d_model 128, 8 heads**, character vocab ~65, `block_size` 128. Final cosine-decay training in notebook 5 uses those hyperparameters.

| Split | Loss | Perplexity |
|-------|------|------------|
| Train | 1.126 | 3.08 |
| Val | 1.412 | 4.10 |
| Test | 1.580 | 4.86 |

Qualitatively, generations follow Shakespearean script structure (`CHARACTER:` headers, dialogue blocks). See notebooks 6–7 for evaluation, attention/saliency views, and sampling demos.

![Final model metrics on train, validation, and test splits](docs/model_results.png)

## Extra criteria pursued

| Criterion | What we did |
|-----------|-------------|
| **Reproducible workflow** | Fixed seeds, JSON experiment logs, saved checkpoints and tuning leaderboards |
| **Hyperparameter optimization** | In-notebook Hyperband tuning over LR, depth, width, heads, dropout, weight decay, warmup (notebook 4) |
| **Deployment / serving** | Export bundle + character-level **streaming** inference; FastAPI `/stream` prototype (notebook 8) |
| **Rigorous evaluation** | Train/val/**test** metrics, generalization gap, baseline vs tuning-best comparison (notebook 6) |
| **Explainable AI** | Attention heatmap, input saliency, top-token predictions on a prompt (notebook 6) |
| **Interactive inference** | Temperature / top-k / top-p controls with `ipywidgets` (notebook 7) |

*Not claimed:* full Streamlit/Gradio app (noted as future work in notebook 8), BPE tokenizer, multi-genre corpora.

## Difficulties & fixes

- **Small-corpus overfitting** — Large models memorized quickly; tuning + val/test monitoring and early stopping kept a modest val–train gap.
- **Notebook path resolution** — Running from `notebooks/` vs repo root broke `data/` lookups; added `PROJECT_ROOT` detection in every notebook.
- **Baseline checkpoint overwrite** — Notebook 5 can replace baseline weights; evaluation documents saving `gpt_baseline.pt` / `baseline_experiment.json` from notebook 3 first.
- **128-token context in deployment** — Long multi-character transcripts lose coherence; mitigated with `stop_on_double_newline` and shorter turns.

## Repository layout

```
notebooks/     # 1–8 pipeline
src/           # dataset ingest script
data/          # CSVs + artifacts (gitignored)
proposal.md    # project spec
docs/          # README figures
```
