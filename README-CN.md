# Gemini 风格多模态 Agent 架构假设

**从公开可观察信号推断 Gemini 的系统级行为架构。**

这是对 Gemini **可观察行为架构**的推理，**不是**对 Gemini 内部神经网络权重或源码的逆向工程。
Google 没有公开 Gemini 3.1 Pro 的内部模型架构（层数、注意力类型、MoE 结构、隐藏维度等均未公开）。
公开可知的是 Gemini 的**能力、产品机制、工具使用行为、推理预算控制、系统级交互模式** —— 这些均可通过 Gemini API、Google AI Studio、官方文档和博客观察到。

本项目从这些可观察行为中推理出**行为架构**：要产生所观察到的行为，系统层面必须存在哪些组件。
内部的层细节（具体层数、MoE 结构、注意力类型、压缩率等）均为**推测性质的玩具实现**，仅用于学习参考，并已明确标注。

---

## 证据层级

本项目中的每个组件均标注其证据级别：

| 标签 | 含义 | 示例来源 |
|------|------|---------|
| `[Observed]` 可直接观察 | 通过 API 行为、黑盒测试、Google AI Studio 使用可直接观察 | Function calling 返回结构化 JSON；thinking_level 参数在 API 中暴露 |
| `[Reported]` 官方报告 | Google 官方博客、论文、系统卡或可靠媒体报道 | Gemini 技术报告；Google AI 博客关于 Gemini 能力的文章 |
| `[Inferred]` 合理推断 | 从观察/报告行为中合理推导的系统组件 | 必须存在工具路由器来派发 function call；推理控制器必须门控计算预算 |
| `[Speculative]` 纯推测 | 纯架构假设 / 玩具实现，仅供学习 | 具体注意力机制（CSA/GCA）、MoE 专家数、隐藏维度、层数 |

---

## Gemini 的公开信息

### 能力（来自官方文档、API 和 Google AI Studio）

**[Observed]** 通过 API 和 Google AI Studio 可直接观察：
- 原生多模态输入（文本、图像、音频、视频）在单一 API 调用中
- Function calling / 结构化工具使用，支持用户定义 schema
- 思考级别 / 推理预算控制（low/medium/high）作为 API 参数暴露
- 长上下文窗口（1M+ tokens）
- 结构化输出（JSON 模式）
- 自主工具编排（规划→执行→校验循环）

**[Reported]** 来自 Google 官方来源：
- Gemini 技术报告（多个版本）
- Google AI 博客关于 Gemini 架构的文章
- 多模态处理通过视觉编码器 + 音频编码器 + 文本分词器
- MoE 是行业已知方向，但 Google 未确认 Gemini 的具体 MoE 架构
- 后训练 RLHF 对齐

---

## 推断的系统级架构

基于可观察行为，我们推断以下**系统级组件**（未必是模型内部层）：

### 行为数据流

```
用户输入
    │
    ▼
[Inferred]  多模态前端
[Observed]  在单一 API 调用中接受文本 + 图像 + 音频 + 视频
[Reported]  视觉编码器 + 音频编码器 + 文本分词器管线
    │
    ├─ 文本分词器 / embedding
    ├─ 图像编码器 / patch projector
    ├─ 音频编码器 / 时序 projector
    └─ 视频编码器 / 帧时序 projector
    │
    ▼
[Inferred]  统一 Token 序列
[Observed]  所有模态合并到单一上下文窗口
[Inferred]  模态类型 embedding + 位置/时序编码
    │
    ▼
[Inferred]  长上下文 Transformer 核心
[Speculative] 局部/滑动窗口注意力
[Speculative] 全局/压缩记忆注意力
[Speculative] 稀疏 MoE FFN
[Speculative] 共享密集 FFN
[Observed]  1M+ token 上下文支持
    │
    ▼
[Observed]  推理预算控制器
[Observed]  thinking_level (low/medium/high) 在 API 中暴露
[Inferred]  基于预算的动态计算分配
    │
    ▼
[Observed]  工具使用运行时
[Observed]  Function calling 支持结构化 JSON schema
[Observed]  自主工具编排（规划→执行）
[Inferred]  规划器 + function call 渲染器 + 观察注入
    │
    ▼
[Inferred]  安全 / 策略运行时
[Reported]  RLHF 后训练对齐
[Observed]  安全过滤器在 API 行为中可观察
    │
    ▼
输出
```

---

## 项目结构

```
opengemini/
├── src/
│   ├── __init__.py
│   ├── model.py                        # ToyGeminiBackbone + GeminiStyleAgentRuntime
│   └── layers/
│       ├── __init__.py
│       ├── common.py                   # [Inferred] LayerNorm
│       ├── embedding.py                # [Inferred] MultimodalFrontend + UnifiedTokenEncoder
│       ├── indexer.py                  # [Speculative] LightningIndexer (top-k 打分)
│       ├── csa_attention.py            # [Speculative] CompressedSparseAttention (4×)
│       ├── hca_attention.py            # [Speculative] GlobalCompressedAttention (128×)
│       ├── mhc_residual.py             # [Speculative] ManifoldConstrainedResidual
│       ├── moe.py                      # [Speculative] SparseMoELayer (naive 参考实现)
│       ├── thinking_ctrl.py            # [Inferred] ReasoningBudgetController
│       └── tool_strategy.py            # [Inferred] ToolRuntime (规划+校验)
├── demo/
│   └── infer_demo.py                   # 前向传播验证
├── docs/
│   ├── ARCHITECTURE.md
│   └── ascii_arch.txt
├── assets/
├── requirements.txt
├── .gitignore
├── LICENSE
└── opengemini.txt
```

---

## 组件证据映射

| 组件 | 文件 | 证据级别 | 理由 |
|------|------|---------|------|
| 多模态前端 | `embedding.py` | [Inferred] + [Observed] | Gemini API 接受多模态输入；文/图/音/视频处理管线可推断 |
| 统一 Token 编码 | `embedding.py` | [Inferred] | 任何多模态模型都需要对统一序列做类型+位置编码 |
| 压缩稀疏注意力 | `csa_attention.py` | [Speculative] | 纯假设；无证据表明 Gemini 使用此机制 |
| 全局压缩注意力 | `hca_attention.py` | [Speculative] | 纯假设；128× 压缩率为任意选择 |
| Lightning Indexer | `indexer.py` | [Speculative] | Top-k 选择是常见技术但属此 toy 设计的特定选择 |
| 流形约束残差 | `mhc_residual.py` | [Speculative] | 有趣的数学构造，无证据 Gemini 使用 Sinkhorn-Knopp 残差 |
| 稀疏 MoE 层 | `moe.py` | [Speculative] | MoE 是行业常用但 Gemini 的具体 MoE 结构未被确认 |
| 推理预算控制器 | `thinking_ctrl.py` | [Inferred] + [Observed] | thinking_level 在 API 中暴露；预算机制可推断 |
| 工具运行时 | `tool_strategy.py` | [Inferred] + [Observed] | Function calling 可观察；规划+校验结构可推断 |
| LayerNorm | `common.py` | [Inferred] | Transformer 架构中普遍存在 |

---

## 架构拆分：ToyGeminiBackbone + GeminiStyleAgentRuntime

模型拆分为两个独立关注点：

### ToyGeminiBackbone（纯模型前向）
```
MultimodalFrontend → UnifiedTokenEncoder → [注意力块 + 残差] → SparseMoE → Norm → LM Head
```
仅负责：text/image/audio/video → hidden → logits。所有具体架构选择均为 `[Speculative]`。

### GeminiStyleAgentRuntime（系统层）
```
ReasoningBudgetController → ToolRuntime (plan + verify + integrate)
```
处理思考级别、工具调用和结果整合，位于系统层而非嵌入 Transformer 主干内部。

---

## 本项目不是什么

- **不是** Gemini 内部模型的泄露或逆向源码
- **不是** Google 训练管线或权重的复现
- **不是** 声称 Gemini 使用 CSA/HCA、256 专家 MoE 或 mHC 残差
- **不是** 生产模型 — 纯教育性玩具实现

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
# 运行推理演示（验证玩具模型前向传播）
python demo/infer_demo.py
```

---

## 参考文献

### Google 官方来源（公开）
- Gemini Technical Report (Google DeepMind)
- Google AI Blog 关于 Gemini 的文章
- Gemini API 文档
- Google AI Studio

### 外部报道
- 第三方对 Gemini 模型的基准测试与评估
- 科技媒体对 Gemini 能力的报道
