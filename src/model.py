# src/model.py
import torch.nn as nn

from .layers import (
    LayerNorm,
    MultiModalUnifiedProcessor,
    UnifiedMultimodalEmbedding,
    CSA,
    HCA,
    ManifoldConstrainedResidual,
    SparseMoEGemini,
    ThinkingLevelController,
    NativeToolStrategyLayer,
)


class HybridAttentionBlock(nn.Module):
    """混合注意力块（CSA+HCA交替）"""

    def __init__(self, dim, heads, use_csa=True, use_hca=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        if use_csa:
            self.attn = CSA(dim, heads)
        elif use_hca:
            self.attn = HCA(dim, heads)
        else:
            raise ValueError("must be CSA or HCA")

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x


class Gemini31ProFull(nn.Module):
    """Gemini 3.1 Pro 完整主模型"""

    def __init__(
        self,
        vocab_size=50000,
        dim=8192,
        heads=64,
        num_layers=40,
    ):
        super().__init__()

        self.processor = MultiModalUnifiedProcessor()
        self.embedding = UnifiedMultimodalEmbedding(vocab_size, dim)

        self.attn_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.attn_blocks.append(
                    HybridAttentionBlock(dim, heads, use_csa=True)
                )
            else:
                self.attn_blocks.append(
                    HybridAttentionBlock(dim, heads, use_hca=True)
                )

        self.mhc = ManifoldConstrainedResidual(dim)
        self.moe = SparseMoEGemini(dim, num_experts=256, shared_experts=1, top_k=8)
        self.thinking_ctrl = ThinkingLevelController(dim)
        self.tool_strategy = NativeToolStrategyLayer(dim)

        self.norm_out = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        text_ids,
        img_feat,
        audio_feat,
        video_feat,
        modality_type_ids,
        pos_ids,
        attention_mask=None,
        thinking_level="medium",
    ):
        x = self.processor(text_ids, img_feat, audio_feat, video_feat)
        x = self.embedding(x, modality_type_ids, pos_ids)

        for blk in self.attn_blocks:
            x = blk(x, attention_mask)
            x = self.mhc(x)

        x = x + self.moe(x)
        x = self.thinking_ctrl(x, level=thinking_level)

        tool_results = self.tool_strategy(x)
        x = self.tool_strategy.integrate_results(x, tool_results)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits, tool_results
