# Gemini 3.1 Pro Architecture Inference

Gemini 3.1 Pro 完整主模型 — 原生多模态稀疏 MoE 架构的逐层解剖级实现，用于学习和理解模型结构。

---

## 项目结构

```
opengemini/
├── src/
│   ├── __init__.py
│   ├── model.py                  # 完整主模型（串联所有子模块）
│   └── layers/
│       ├── __init__.py           # 子模块统一导出
│       ├── common.py             # LayerNorm 基础组件
│       ├── embedding.py          # 多模态输入处理 + 统一嵌入
│       ├── indexer.py            # Lightning Indexer（信息打分筛选）
│       ├── csa_attention.py      # 压缩稀疏注意力 CSA
│       ├── hca_attention.py      # 重度压缩注意力 HCA
│       ├── mhc_residual.py       # 流形约束残差 mHC
│       ├── moe.py                # 稀疏 MoE（1 共享 + 256 路由）
│       ├── thinking_ctrl.py      # 思考深度三档控制
│       └── tool_strategy.py      # 原生工具策略层
├── demo/
│   └── infer_demo.py             # 推理演示入口
├── docs/
│   ├── ARCHITECTURE.md           # 架构详解
│   └── ascii_arch.txt            # ASCII 架构图
├── assets/                       # 架构图（待补充）
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 如何阅读这个项目

### 推荐阅读顺序（自底向上）

这个项目按「基础组件 → 子模块 → 主模型」组织，建议按以下顺序阅读：

**第一步：基础组件** — `src/layers/common.py`
- 只有 `LayerNorm`，是整个模型最底层的基础设施。

**第二步：嵌入层** — `src/layers/embedding.py`
- `MultiModalUnifiedProcessor`：将文本 / 图像 / 音频 / 视频四路输入分别编码到统一维度（8192），拼接成一个序列。
- `UnifiedMultimodalEmbedding`：在统一维度空间上叠加 token 嵌入、模态类型嵌入、位置嵌入，形成最终的输入表示。
- 阅读关键：理解 `forward` 的输入输出 shape 变化，这是模型的第一站。

**第三步：注意力机制** — `src/layers/indexer.py` → `src/layers/csa_attention.py` → `src/layers/hca_attention.py`
- `LightningIndexer`：给压缩后的 KV 打分，只保留 top-k 个最重要的 token。这是 CSA 的核心依赖。
- `CSA`（Compressed Sparse Attention）：
  1. 用 Conv1d 把 KV 按 `compression_rate=4` 压缩（每 4 个 token → 1 个摘要）
  2. 用 LightningIndexer 从压缩结果中选出 topk 个
  3. 仅在这 topk 个上做注意力
- `HCA`（Heavily Compressed Attention）：
  1. 类似 CSA，但 `compression_rate=128`（极度压缩）
  2. 不做筛选，直接在全量压缩结果上做注意力，捕捉全局结构

CSA 和 HCA 的对比是阅读重点：前者抓局部关键信息，后者抓全局长程依赖，两者交替使用。

**第四步：残差与 MoE** — `src/layers/mhc_residual.py` → `src/layers/moe.py`
- `ManifoldConstrainedResidual`：用 Sinkhorn-Knopp 迭代将残差投影到双随机矩阵流形上再做融合。每经过一个注意力块都会走一次 mHC。
- `SparseMoEGemini`：
  - 1 个共享专家（所有 token 都过）
  - 256 个路由专家（每个 token 只激活 top-8 个）
  - 阅读关键：理解 `gate` 如何决定路由，以及 `for k in range(top_k)` 的循环如何组合专家输出。

**第五步：高层控制** — `src/layers/thinking_ctrl.py` → `src/layers/tool_strategy.py`
- `ThinkingLevelController`：Low/Medium/High 三档，通过可学习的 level embedding 注入到表示中，控制推理深度。
- `NativeToolStrategyLayer`：规划 → 校验两阶段，输出 16 种工具的调用计划和校验概率，最后将结果融合回主表示。

**第六步：串联全部** — `src/model.py`
- `HybridAttentionBlock`：CSA/HCA 的薄封装（Pre-Norm 残差），偶数层用 CSA，奇数层用 HCA。
- `Gemini31ProFull`：按设计将以上所有模块串联成完整流水线。

---

## 数据流全景

```
输入（text_ids, img_feat, audio_feat, video_feat,
      modality_type_ids, pos_ids）
  │
  ▼
MultiModalUnifiedProcessor   —— 四模态 → 统一序列
  │
  ▼
UnifiedMultimodalEmbedding   —— 叠加 token + type + pos embedding
  │
  ▼
┌─ HybridAttentionBlock × 40 ─┐
│   ├─ CSA (偶数层)             │
│   └─ HCA (奇数层)             │
│   └─ mHC 残差（每层）          │
└──────────────────────────────┘
  │
  ▼
SparseMoEGemini              —— 共享 + top-8 路由专家
  │
  ▼
ThinkingLevelController      —— Low/Medium/High 档位注入
  │
  ▼
NativeToolStrategyLayer      —— 工具规划 + 自校验 + 结果融合
  │
  ▼
LayerNorm → LM Head          —— 最终归一化 → logits 输出
```

---

## 安装与运行

```bash
# 创建并激活 venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

```bash
# 运行推理演示（检查模型能否正常 forward）
python demo/infer_demo.py
```

---

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 8192 | 模型隐层维度 |
| `heads` | 64 | 注意力头数 |
| `head_dim` | 128 | 每头维度 (8192/64) |
| `num_layers` | 40 | 混合注意力块层数（20 CSA + 20 HCA） |
| `vocab_size` | 50000 | 词表大小 |
| `num_experts` | 256 | 路由专家数 |
| `shared_experts` | 1 | 共享专家数 |
| `top_k` | 8 | 每 token 激活专家数 |
| `CSA compression` | 4× | 每 4 token 压缩为 1 个摘要 |
| `HCA compression` | 128× | 每 128 token 压缩为 1 个摘要 |
| `topk (Indexer)` | 512 | CSA 保留的 top-k 压缩 token 数 |
| `n_streams (mHC)` | 4 | 流形约束的并行流数 |
| `max_pos` | 1M | 最大位置编码长度 |
| `num_tools` | 16 | 工具策略支持的工具数 |

---

## 三模型架构全景对比

| 维度 | GPT-5.5 | Claude Mythos | Gemini 3.1 Pro |
|------|---------|--------------|----------------|
| **深度方式** | 48 层堆叠 | RDT 循环（权重复用 12 轮） | 40 层 CSA+HCA 交替 |
| **注意力** | MLA（潜空间 KV 压缩） | 局部窗口（128） | CSA(4×)+HCA(128×) |
| **MoE 定位** | 仅专项补强 | 全部 token | 全部 token |
| **MoE 规模** | 256 专家 / top-7 | 64 专家 / top-2 | 256 专家 / top-8 |
| **模态融合** | 早期融合（embedding 阶段） | Type embeddings | 后置拼接 |
| **安全机制** | RLHF 对齐层 | 宪法AI（前+后） | 无 |
| **工具策略** | 自主调度（LSTM 多步） | 被动（用户指令门控） | 自主（规划→校验） |
| **维度** | 8192 | 4096 | 8192 |
| **总层数** | 48 | 1 (循环12轮) | 40 |