# src/layers/embedding.py
import torch
import torch.nn as nn


class MultiModalUnifiedProcessor(nn.Module):
    """原生多模态输入处理器（文本/图像/音频/视频四路）"""

    def __init__(self):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.GELU(),
            nn.Linear(8192, 8192),
        )
        self.audio_enc = nn.Sequential(
            nn.Linear(1024, 8192),
            nn.GELU(),
            nn.Linear(8192, 8192),
        )
        self.video_enc = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.GELU(),
            nn.Linear(8192, 8192),
        )

    def forward(self, text_ids, img_feat, audio_feat, video_feat):
        B, T_txt = text_ids.shape

        img = self.img_enc(img_feat).unsqueeze(1)
        audio = self.audio_enc(audio_feat).unsqueeze(1)
        video = self.video_enc(video_feat).unsqueeze(1)

        x = torch.cat([text_ids, img, audio, video], dim=1)
        return x


class UnifiedMultimodalEmbedding(nn.Module):
    """统一多模态嵌入层（单Token空间全模态映射）"""

    def __init__(self, vocab_size, dim=8192):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, dim)
        self.type_emb = nn.Embedding(4, dim)
        self.pos_emb = nn.Embedding(1024 * 1024, dim)

    def forward(self, token_ids, modality_type_ids, pos_ids):
        x = (
            self.text_emb(token_ids)
            + self.type_emb(modality_type_ids)
            + self.pos_emb(pos_ids)
        )
        return x
