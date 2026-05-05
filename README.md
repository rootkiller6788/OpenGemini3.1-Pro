# Gemini 3.1 Pro Architecture Inference

Layer-by-layer architectural inference of Gemini 3.1 Pro — a native multimodal sparse MoE model.
For learning and understanding the model structure.

---

## Project Structure

```
opengemini/
├── src/
│   ├── __init__.py
│   ├── model.py                  # Full main model (assembles all submodules)
│   └── layers/
│       ├── __init__.py           # Unified exports
│       ├── common.py             # LayerNorm base component
│       ├── embedding.py          # Multimodal input processing + unified embedding
│       ├── indexer.py            # Lightning Indexer (scoring + filtering)
│       ├── csa_attention.py      # Compressed Sparse Attention (CSA)
│       ├── hca_attention.py      # Heavily Compressed Attention (HCA)
│       ├── mhc_residual.py       # Manifold Constrained Residual (mHC)
│       ├── moe.py                # Sparse MoE (1 shared + 256 routed)
│       ├── thinking_ctrl.py      # Thinking depth three-tier control
│       └── tool_strategy.py      # Native autonomous tool strategy layer
├── demo/
│   └── infer_demo.py             # Inference demo entry point
├── docs/
│   ├── ARCHITECTURE.md           # Architecture deep-dive
│   └── ascii_arch.txt            # ASCII architecture diagram
├── assets/                       # Architecture diagram (TBD)
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How to Read This Project

### Recommended Reading Order (Bottom-Up)

The project is organized as "base components → submodules → main model".

**Step 1: Base Components** — `src/layers/common.py`
- `LayerNorm`: the lowest-level infrastructure for the entire model.

**Step 2: Embedding Layer** — `src/layers/embedding.py`
- `MultiModalUnifiedProcessor`: Encodes four input modalities (text/image/audio/video) into a unified dimension (8192) and concatenates them into a single sequence.
- `UnifiedMultimodalEmbedding`: Stacks token embedding + modality type embedding + position embedding in the unified dimension space.

**Step 3: Attention Mechanisms** — `src/layers/indexer.py` → `src/layers/csa_attention.py` → `src/layers/hca_attention.py`
- `LightningIndexer`: Scores compressed KV and keeps only the top-k most important tokens — core dependency of CSA.
- `CSA` (Compressed Sparse Attention): Conv1d compresses KV by 4×, Lightning Indexer selects top-k, attention is computed only on these sparse tokens.
- `HCA` (Heavily Compressed Attention): Conv1d compresses KV by 128×, full attention over compressed results to capture global structure (no filtering).
- Key insight: CSA captures local critical information, HCA captures global long-range dependencies — used alternately.

**Step 4: Residual and MoE** — `src/layers/mhc_residual.py` → `src/layers/moe.py`
- `ManifoldConstrainedResidual`: Projects residuals onto a doubly stochastic matrix manifold via Sinkhorn-Knopp iteration, applied after every attention block.
- `SparseMoEGemini`: 1 shared expert (all tokens) + 256 routed experts (top-8 per token).

**Step 5: High-Level Control** — `src/layers/thinking_ctrl.py` → `src/layers/tool_strategy.py`
- `ThinkingLevelController`: Low/Medium/High three-tier control via learnable level embeddings.
- `NativeToolStrategyLayer`: Plan → verify two-stage pipeline, outputs tool invocation plans + verification probabilities, then fuses results back.

**Step 6: Assemble Everything** — `src/model.py`
- `HybridAttentionBlock`: Thin wrapper (Pre-Norm residual) for CSA/HCA — even layers use CSA, odd layers use HCA.
- `Gemini31ProFull`: Wires all modules together into the complete pipeline.

---

## Data Flow

```
Input (text_ids, img_feat, audio_feat, video_feat,
       modality_type_ids, pos_ids)
  │
  ▼
MultiModalUnifiedProcessor   —— 4 modalities → unified sequence
  │
  ▼
UnifiedMultimodalEmbedding   —— token + type + pos embeddings
  │
  ▼
┌─ HybridAttentionBlock × 40 ─┐
│   ├─ CSA (even layers)        │
│   └─ HCA (odd layers)         │
│   └─ mHC residual (per layer) │
└──────────────────────────────┘
  │
  ▼
SparseMoEGemini              —— shared + top-8 routed experts
  │
  ▼
ThinkingLevelController      —— Low/Medium/High level injection
  │
  ▼
NativeToolStrategyLayer      —— tool planning + self-verify + result fusion
  │
  ▼
LayerNorm → LM Head          —— final norm → logits output
```

---

## Setup and Run

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

```bash
# Run inference demo (verify model forward pass)
python demo/infer_demo.py
```

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 8192 | Hidden dimension |
| `heads` | 64 | Attention heads |
| `head_dim` | 128 | Per-head dimension (8192/64) |
| `num_layers` | 40 | Hybrid attention blocks (20 CSA + 20 HCA) |
| `vocab_size` | 50000 | Vocabulary size |
| `num_experts` | 256 | Number of routed experts |
| `shared_experts` | 1 | Number of shared experts |
| `top_k` | 8 | Tokens activated per expert |
| `CSA compression` | 4× | Every 4 tokens → 1 summary |
| `HCA compression` | 128× | Every 128 tokens → 1 summary |
| `topk (Indexer)` | 512 | Top-k compressed tokens in CSA |
| `n_streams (mHC)` | 4 | Parallel streams for manifold constraint |
| `max_pos` | 1M | Maximum position embedding length |
| `num_tools` | 16 | Number of supported tools |

---

## Three-Model Architecture Comparison

| Aspect | GPT-5.5 | Claude Mythos | Gemini 3.1 Pro |
|--------|---------|--------------|----------------|
| **Depth approach** | 48-layer stack | RDT loop (weight reuse ×12) | 40-layer CSA+HCA alternating |
| **Attention** | MLA (latent KV compression) | Local window (128) | CSA(4×)+HCA(128×) |
| **MoE role** | Specialized only | All tokens | All tokens |
| **MoE scale** | 256 experts / top-7 | 64 experts / top-2 | 256 experts / top-8 |
| **Modal fusion** | Early fusion (embedding stage) | Type embeddings | Post-embedding concat |
| **Safety** | RLHF alignment layer | Constitutional AI (pre+post) | None |
| **Tool strategy** | Autonomous scheduler (LSTM) | Passive (user-gated) | Autonomous (plan→verify) |
| **Dimension** | 8192 | 4096 | 8192 |
| **Total layers** | 48 | 1 (looped 12 times) | 40 |
