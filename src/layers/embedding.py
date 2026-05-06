# src/layers/embedding.py
import torch
import torch.nn as nn


class MultimodalFrontend(nn.Module):
    """[Speculative] 多模态前端 — 将文本/图像/音频/视频映射到统一维度空间

    当前仅作为 toy 实现：文本通过 Embedding 查表，非文本模态通过
    线性投影映射到统一 hidden dimension，然后拼接为单一序列。
    真实的 Gemini 多模态处理流程未公开。
    """

    def __init__(self, vocab_size=50000, dim=8192):
        super().__init__()
        self.dim = dim
        self.text_emb = nn.Embedding(vocab_size, dim)
        self.image_projector = nn.Sequential(
            nn.Linear(2048, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(1024, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.video_projector = nn.Sequential(
            nn.Linear(2048, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, text_ids, img_feat=None, audio_feat=None, video_feat=None):
        text_x = self.text_emb(text_ids)

        parts = [text_x]
        if img_feat is not None:
            parts.append(self.image_projector(img_feat).unsqueeze(1))
        if audio_feat is not None:
            parts.append(self.audio_projector(audio_feat).unsqueeze(1))
        if video_feat is not None:
            parts.append(self.video_projector(video_feat).unsqueeze(1))

        x = torch.cat(parts, dim=1)
        return x


class UnifiedTokenEncoder(nn.Module):
    """[Inferred] 统一 Token 编码 — 模态类型 + 位置编码叠加

    在 MultimodalFrontend 输出之上添加 type embedding（区分文本/图像/音频/视频）
    和 position encoding。任何多模态模型都需要类似机制。
    """

    def __init__(self, dim=8192):
        super().__init__()
        self.type_emb = nn.Embedding(4, dim)
        self.pos_emb = nn.Embedding(1024 * 1024, dim)

    def forward(self, x, type_ids, pos_ids):
        x = x + self.type_emb(type_ids)
        x = x + self.pos_emb(pos_ids)
        return x
