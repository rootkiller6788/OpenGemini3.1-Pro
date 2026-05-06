# src/model.py
import torch.nn as nn

from .layers import (
    LayerNorm,
    MultimodalFrontend,
    UnifiedTokenEncoder,
    CompressedSparseAttention,
    GlobalCompressedAttention,
    ManifoldConstrainedResidual,
    SparseMoELayer,
    ReasoningBudgetController,
    ToolRuntime,
)


class ToyGeminiBackbone(nn.Module):
    """[Speculative] Gemini 风格模型主干 — 多模态输入 → hidden → logits

    仅负责从多模态输入到 logits 的纯模型前向传播。
    所有具体架构选择（层数、维度、注意力类型等）均为纯推测性 toy 实现。
    """

    def __init__(
        self,
        vocab_size=50000,
        dim=8192,
        heads=64,
        num_layers=40,
    ):
        super().__init__()
        self.frontend = MultimodalFrontend(vocab_size, dim)
        self.encoder = UnifiedTokenEncoder(dim)

        self.attn_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.attn_blocks.append(
                    HybridAttentionBlock(dim, heads, attn_type="csa")
                )
            else:
                self.attn_blocks.append(
                    HybridAttentionBlock(dim, heads, attn_type="gca")
                )

        self.residual = ManifoldConstrainedResidual(dim)
        self.moe = SparseMoELayer(dim)
        self.norm_out = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        text_ids,
        img_feat=None,
        audio_feat=None,
        video_feat=None,
        type_ids=None,
        pos_ids=None,
        attention_mask=None,
    ):
        x = self.frontend(text_ids, img_feat, audio_feat, video_feat)
        x = self.encoder(x, type_ids, pos_ids)

        for blk in self.attn_blocks:
            x = blk(x, attention_mask)
            x = self.residual(x)

        x = x + self.moe(x)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits


class HybridAttentionBlock(nn.Module):
    """混合注意力块 — [Speculative] CSA 与 GCA 交替使用"""

    def __init__(self, dim, heads, attn_type="csa"):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        if attn_type == "csa":
            self.attn = CompressedSparseAttention(dim, heads)
        elif attn_type == "gca":
            self.attn = GlobalCompressedAttention(dim, heads)
        else:
            raise ValueError("attn_type must be 'csa' or 'gca'")

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x


class GeminiStyleAgentRuntime(nn.Module):
    """[Inferred] Gemini 风格 Agent 运行时 — 系统层组件

    负责推理预算控制、工具调用、安全策略等系统级行为，
    不嵌入 Transformer backcone 内部。拆分系统行为与纯模型 forward。
    """

    def __init__(self, dim, num_tools=16):
        super().__init__()
        self.reasoning = ReasoningBudgetController(dim)
        self.tool = ToolRuntime(dim, num_tools)

    def forward(self, x, thinking_level="medium"):
        x = self.reasoning(x, level=thinking_level)
        tool_results = self.tool(x)
        x = self.tool.integrate_results(x, tool_results)
        return x, tool_results
