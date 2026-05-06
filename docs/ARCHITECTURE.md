# Architecture Discussion

## Important Disclaimer

Google has **not** publicly disclosed Gemini's internal model architecture. Layer count, hidden dimension,
attention type, MoE structure, activation functions — these are all unknown.

What IS known: Gemini's observable system behaviors, published capabilities, API-exposed parameters,
and product-level features. This document infers **behavioral system components** from those public signals.
Implementation details (CSA/GCA, 40 layers, 256 experts, mHC residual) are **speculative toy designs**
for research exploration.

---

## Evidence Hierarchy

| Tag | Source |
|-----|--------|
| `[Observed]` | Directly observable via API behavior, black-box testing, Google AI Studio usage |
| `[Reported]` | Google blog posts, papers, system cards, official docs |
| `[Inferred]` | Reasonably deduced system component from observed/reported behavior |
| `[Speculative]` | Pure architectural hypothesis / toy implementation |

---

## Inferred System Components

### 1. Multimodal Frontend
`[Observed] + [Reported]`

**What we know:** Gemini API accepts text, image, audio, and video in a single call.
Google has described a vision encoder + audio encoder + text tokenizer pipeline.

**Inferred system component:** A frontend that maps each modality into a unified hidden
representation and concatenates them into a single token sequence.

**Toy implementation:** `MultimodalFrontend` — embeds text tokens, projects image/audio/video
features to the same hidden dimension, then concatenates.

### 2. Unified Token Encoder
`[Inferred]`

**What we know:** All modalities are merged into a single context window with positional
and type information.

**Inferred:** Modality type embeddings + position/temporal encoding over the unified sequence.
Any multimodal model needs this.

**Toy implementation:** `UnifiedTokenEncoder` — adds learnable type + position embeddings.

### 3. Long-Context Transformer Core
`[Speculative]`

**What we know:** Gemini supports 1M+ token context windows. How it achieves this architecturally
is unknown.

**Speculative hypotheses (all marked as toy implementations):**
- `CompressedSparseAttention` — KV compression ×4 + top-k indexer selection. Pure hypothesis.
- `GlobalCompressedAttention` — KV compression ×128 + full attention over compressed results. Pure hypothesis.
- Alternating CSA/GCA layers (even/odd). Arbitrary design choice.
- `SparseMoELayer` — 1 shared expert + 256 routed with top-8. MoE is industry-common but Gemini's
  specific MoE architecture is unconfirmed.

**Note:** The specific compression rates (4×, 128×), layer count (40), expert count (256),
top-k settings — all are arbitrary speculative numbers with no evidence.

### 4. Manifold Constrained Residual
`[Speculative]`

Sinkhorn-Knopp iteration projecting residuals onto a doubly stochastic matrix manifold.
An interesting mathematical construction, but there is no evidence Gemini uses this.

**Toy implementation:** `ManifoldConstrainedResidual` with 4 streams and 5 Sinkhorn iterations.

### 5. Reasoning Budget Controller
`[Observed] + [Inferred]`

**What we know:** Gemini API exposes `thinking_level` (low/medium/high) and related
thinking budget parameters. This is an `[Observed]` fact.

**Inferred:** There exists a reasoning budget controller that gates dynamic compute allocation
based on the requested level.

**Toy implementation:** `ReasoningBudgetController` — adds a learnable level embedding
corresponding to the requested budget tier. The underlying mechanism is unknown.

### 6. Tool-use Runtime
`[Observed] + [Inferred]`

**What we know:** Gemini supports function calling with user-defined JSON schemas.
Gemini can autonomously orchestrate tools in a plan→execute→verify cycle. `[Observed]`

**Inferred system components:**
1. A planner that generates tool call plans from the current context
2. A function call renderer that emits structured tool call tokens/JSON
3. An external executor (outside the model — API-level concern)
4. An observation reinjection mechanism that feeds tool results back into context

**Toy implementation:** `ToolRuntime` — plan network + verify network + result projector.
This only models the planning/verification aspect, not the full agentic loop.

### 7. Safety / Policy Runtime
`[Inferred] + [Reported]`

Google reports RLHF post-training for alignment. Safety filters are observable in API behavior.
This is a system-layer concern, not a fixed neural network layer inside the transformer.

**Not implemented as a separate module** in this toy project. A real implementation would be:
- Policy classifier / refusal controller
- Output safety filter
- Operating at the API/system layer, not inside the model backbone

---
## Architecture Split: Why Two Classes

The project is split into two separate classes for correctness:

### ToyGeminiBackbone
```
MultimodalFrontend → UnifiedTokenEncoder → [Attention + Residual] × N → SparseMoE → Norm → LM Head
```
Pure model forward pass. Text/image/audio/video → hidden → logits.

### GeminiStyleAgentRuntime
```
ReasoningBudgetController → ToolRuntime (plan + verify + integrate)
```
System-level behaviors: thinking control, tool orchestration, result integration.

These are separated because:
1. Tool use is a system-layer concern (API → router → executor → observation), not embedded inside transformer layers
2. Reasoning budget is a controller wrapping the model, not a neural network block
3. Safety/policy is a system runtime, not a fixed transformer layer
