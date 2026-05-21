# Genre-Conditioned Story Generation using a Transformer

## Text Source

The project uses the dataset:

https://huggingface.co/datasets/FareedKhan/1k_stories_100_genre

containing approximately 1000 stories across 100 genres with labeled genre tags and full story text.

---

## Model Architecture(s)

A vanilla encoder-decoder Transformer implemented in PyTorch for conditional text generation.

### Core Components

- Token embeddings + positional encoding
- Multi-head self-attention
- Encoder-decoder cross-attention
- Feed-forward Transformer blocks
- Autoregressive decoding
- Input conditioning using genre + context prompt

The model takes a genre and optional prompt as input and generates a genre-consistent story continuation.

---

## Extra Criteria

### Chatbot GUI

An interactive interface where users can:

- Select a genre
- Enter a custom prompt
- Generate stories using the trained Transformer model

### MLOps + Experiment Tracking

The project will also include MLOps-focused components such as:

- Hyperparameter tuning (learning rate, embedding size, number of heads, layers, dropout, sequence length)
- Automated checkpointing and best-model saving
- Training and validation metric tracking
- Experiment logging and comparison using Weights & Biases or TensorBoard
- Reproducible training pipeline with configurable experiments