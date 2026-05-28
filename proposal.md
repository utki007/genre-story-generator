# MLOps-Driven Classical Text Generation (GPT From Scratch)

Build a scalable, experiment-driven system for **autoregressive text generation** using a custom-built, decoder-only Transformer. Trained on the TinyShakespeare dataset, the model learns the spelling, grammar, and structural format of Shakespearean plays to generate coherent, character-accurate continuations.

Unlike a standard notebook tutorial, this project emphasizes **systematic experimentation**, **hyperparameter optimization**, **reproducibility**, and **deployment** following modern MLOps practices (tracked via Weights & Biases and modular pipelines).

## 1. Overview

The system supports controlled experimentation, automated hyperparameter tuning, and structured evaluation for a custom PyTorch Language Model, built entirely from scratch and optimized for rapid training cycles.

## 2. Dataset

* **Source**: Hugging Face / Karpathy repo: `karpathy/tiny_shakespeare`
* **Size**: ~1MB of text (approx. 40,000 lines / 1 million characters).
* **Preprocessing Pipeline**:
* **Tokenization**: Implementation of a Character-level tokenizer (creating a vocabulary of ~65 unique characters) or a minimalist Byte Pair Encoding (BPE).
* **Chunking**: Splitting the text into contiguous blocks of fixed context length (e.g., `block_size = 64` or `256`).
* **Data Loaders**: PyTorch `DataLoader` setup for randomized, batched tensor generation.



## 3. Problem Formulation

* **Input**: A sequence of text tokens $x_{<t}$.
* **Output**: The predicted next token $x_t$.

Modeled as standard causal language modeling: $P(x_t | x_{<t})$

Where the Transformer learns to maximize the likelihood of the training corpus via autoregressive next-token prediction.

## 4. Model Architecture

A **Decoder-only Transformer (GPT-style)** built from scratch in PyTorch.

### Components

* **Embeddings**: Token embeddings + Learned Positional embeddings.
* **Transformer Blocks**:
* Causal (Masked) Multi-Head Self-Attention (ensuring tokens only attend to previous tokens).
* Position-wise Feed-Forward Networks (4x hidden dimension expansion).
* Layer Normalization (pre-norm formulation) and residual connections.


* **LM Head**: Final linear projection to the vocabulary logits.

## 5. Training Strategy

Causal language modeling using standard Cross-Entropy Loss:


$$L = - \sum \log P(x_t | x_{<t})$$

### Enhancements

* **Mixed Precision Training**: For maximizing GPU utilization.
* **Gradient Clipping**: To stabilize the training of the from-scratch attention layers.
* **Learning Rate Scheduling**: Cosine decay with a linear warmup phase.

## 6. MLOps Pipeline (Weights & Biases)

Because TinyShakespeare trains rapidly, the project heavily leans into MLOps automation to find the optimal architecture size.

### 6.1 Experiment Tracking

All training runs are logged to a central W&B dashboard.

* **Logged Metrics**:
* Train/Validation Loss
* Validation Perplexity
* Tokens processed per second (throughput)


* **Artifacts**:
* Qualitative text samples generated at the end of each epoch to visualize the model learning structure (e.g., character names, line breaks, words).



### 6.2 Hyperparameter Optimization

Using **W&B Sweeps** (Bayesian search) for automated tuning.

* **Hyperparameters to Sweep**:
* Learning rate: (e.g., `1e-4` to `1e-2`)
* Number of Transformer layers: `n_layers` (e.g., 4 to 8)
* Hidden dimension size: `d_model` (e.g., 128 to 512)
* Number of attention heads: `n_heads` (e.g., 4 to 8)
* Dropout rate: (e.g., 0.1 to 0.3)


* **Objective**:
* Minimize validation loss without overfitting the small dataset.



### 6.3 Model Versioning

Using W&B Artifacts to track the lineage of:

* The tokenizer mapping dictionary.
* Versioned model checkpoints (e.g., `baseline-char-v1`, `sweep-best-v2`).

### 6.4 Reproducible Codebase

A modular, production-ready Python project structure rather than a single Jupyter Notebook:

* `config.yaml` → Sweep and run parameters.
* `model.py` → The from-scratch PyTorch Transformer class.
* `data.py` → Dataset streaming, encoding, and chunking.
* `train.py` → The main training loop with W&B hooks.

## 7. Evaluation Framework

* **Quantitative**: Validation Loss/Perplexity. (Tracking the gap between Train and Val loss is critical here, as TinyShakespeare is small enough that a large model will quickly memorize it).
* **Qualitative Generation**: Periodic inference prompts logged as W&B Tables to visually track how quickly the model stops outputting random characters and starts structuring text as proper Shakespearean script (Character Name -> Colon -> Dialogue).

## 8. Inference & Deployment

### 8.1 API Service (FastAPI)

A REST API for inference, loading the best W&B model artifact:

* **Input**: A text prompt (e.g., "ROMEO:") and max generation length.
* **Output**: The autoregressively generated continuation.

### 8.2 Interactive Script Generator UI

A **Streamlit** dashboard to interact with the deployed model, allowing users to adjust text generation parameters:

* **Temperature** (controlling randomness)
* **Top-K / Top-P sampling** (preventing degenerate text loops)

## 9. Experimental Plan

| Experiment | Change | Expected Outcome |
| --- | --- | --- |
| **Baseline** | `n_layers=4`, `d_model=128`, `n_heads=4` | Learns word boundaries and line breaks; high validation loss. |
| **Sweep 1** | Tune Learning Rate & Dropout | Faster convergence; regularization prevents early overfitting. |
| **Sweep 2** | Scale `d_model` and `n_layers` | Identifies the maximum model size the 1MB dataset can support before catastrophic overfitting. |
| **Final Run** | Train best config with Cosine Decay | Generation of highly coherent, correctly spelled pseudo-Shakespeare. |

## 10. Conclusion

This project delivers a deep technical understanding of generative AI by building a Transformer from the ground up. By utilizing a compact dataset, the project shifts the focus from raw compute time to mastering production-grade MLOps skills through rigorous W&B tracking, sweeping, and modular deployment.