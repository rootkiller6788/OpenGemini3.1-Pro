# Gemini 3.1 Pro Architecture

## Overview

Gemini 3.1 Pro is a native multimodal sparse Mixture-of-Experts (MoE) model.

## Key Components

### 1. Multi-Modal Input Processor
Handles four modalities: text, image, audio, video.

### 2. Unified Multi-modal Embedding
Maps all modalities into a single token space with type and position embeddings.

### 3. Hybrid Attention (CSA + HCA)
- **CSA (Compressed Sparse Attention)**: Compresses KV by 4×, uses Lightning Indexer for top-k selection.
- **HCA (Heavily Compressed Attention)**: Compresses KV by 128× for global context capture.

### 4. Manifold Constrained Residual (mHC)
Applies Sinkhorn-Knopp iteration for doubly stochastic matrix projection on residuals.

### 5. Sparse MoE
- 1 shared expert
- 256 routed experts
- Top-8 per token activation

### 6. Thinking Level Controller
Three-tier control: Low, Medium, High.
Adds learned level embeddings to the representation.

### 7. Native Tool Strategy Layer
Planning → Execution → Self-verification pipeline with 16 tool types.
