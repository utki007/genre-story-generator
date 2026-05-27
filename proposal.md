# MLOps-Driven Genre-Conditioned Story Generation

Build a scalable, experiment-driven system for **conditional short-story generation** using transformer models. Given a **genre label** and a **user prompt**, the model generates coherent, genre-consistent stories.

Unlike a “just train a model” implementation, this project emphasizes **systematic experimentation**, **hyperparameter optimization**, **reproducibility**, **evaluation**, and **deployment** following modern MLOps practices (tracked via Weights & Biases and modular pipelines).

## 1. Overview

The system supports controlled experimentation, automated tuning, and structured evaluation, with full tracking and reproducibility.

## 2. Dataset

- **Source**: Hugging Face dataset: [FareedKhan/1k_stories_100_genre](https://huggingface.co/datasets/FareedKhan/1k_stories_100_genre)
- **Size**: ~1,000 stories across **100 distinct genres**
- **Fields per sample**:
  - genre label
  - full story text

### Preprocessing

- Convert into `(genre + prompt → story)` pairs
- Tokenize using a transformer tokenizer
- Truncate/pad sequences to a fixed length

## 3. Problem formulation

- **Inputs**: genre label `g`, input prompt `x`
- **Output**: generated story `y` conditioned on `(g, x)`

Modeled as: `P(y | x, g)`

where the transformer learns autoregressive conditional generation.

## 4. Model architecture

A vanilla **encoder-decoder Transformer (PyTorch)**.

### Encoder

- Token embeddings
- Positional encodings
- Multi-head self-attention layers
- Feed-forward networks

### Decoder

- Masked self-attention (autoregressive)
- Cross-attention over encoder outputs
- Linear projection to vocabulary logits

### Genre conditioning

One of:
- Prepend a special `[GENRE]` token to the input sequence
- Add a learned genre embedding to token embeddings

## 5. Training strategy

Teacher forcing with cross-entropy loss: `L = - Σ log P(y_t | y_<t, x, g)`

### Enhancements

- Label smoothing
- Dropout regularization
- Gradient clipping
- Mixed precision training (optional)

## 6. MLOps pipeline

### 6.1 Experiment tracking (Weights & Biases)

All experiments are tracked in W&B.

- **Logged metrics**:
  - training loss, validation loss
  - perplexity
  - learning rate schedules
- **Artifacts**:
  - generated sample stories
  - model checkpoints

### 6.2 Hyperparameter optimization

Use **W&B Sweeps** for automated tuning.

- **Hyperparameters**:
  - learning rate
  - number of layers
  - hidden dimension size
  - number of attention heads
  - dropout rate
  - batch size
  - context length
- **Objective**:
  - minimize validation perplexity
  - maximize generation quality (via metrics + qualitative samples)

### 6.3 Model versioning

Using W&B Artifacts for:
- versioned model checkpoints
- best-model tracking
- reproducible experiment storage

Example versions:
- `baseline-v1`
- `sweep-best-v2`
- `large-model-v3`

### 6.4 Reproducible training pipeline

A modular training pipeline:
- `config.yaml` → hyperparameters
- `train.py` → training loop
- `model.py` → transformer architecture
- `data.py` → dataset preprocessing

This supports reproducibility, consistent comparisons, and scalable tuning.

## 7. Evaluation framework

### 7.1 Automatic metrics

- Perplexity
- BLEU
- ROUGE-L
- Distinct-n (diversity)
- Repetition rate

### 7.2 Semantic metrics

- BERTScore (embedding-based similarity)
- Cosine similarity between generated and reference stories

### 7.3 Human evaluation (optional)

User study evaluating:
- coherence
- creativity
- genre relevance

## 8. Inference & deployment

### 8.1 API service (FastAPI)

A REST API for inference:
- **Input**: genre + prompt
- **Output**: generated story

Example endpoint:

`POST /generate`

### 8.2 Interactive UI

A simple UI (Streamlit or Gradio) to:
- select genre
- enter prompt
- generate stories in real time

## 9. Experimental plan

| Experiment | Change | Expected outcome |
|---|---|---|
| Baseline | vanilla transformer | reference performance |
| Sweep 1 | learning rate tuning | faster convergence |
| Sweep 2 | deeper model | improved coherence |
| Sweep 3 | dropout tuning | reduced overfitting |
| Sweep 4 | decoding strategies | higher diversity |

## 10. Expected contributions

- End-to-end transformer-based conditional generation system
- Automated hyperparameter optimization using W&B Sweeps
- Full experiment tracking and reproducibility framework
- Deployment-ready inference system (API + UI)
- Analysis of trade-offs in generation quality vs model complexity

## 11. Future work

- Scale to larger corpora (BookCorpus, WikiText)
- RLHF-style fine-tuning for story quality improvement
- Advanced decoding strategies (top-k, nucleus sampling)
- Multi-turn interactive storytelling system

## 12. Conclusion

This project presents a complete MLOps-driven pipeline for transformer-based story generation, emphasizing reproducibility, scalability, and systematic optimization rather than only model training.