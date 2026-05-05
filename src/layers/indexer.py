# src/layers/indexer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """Lightning Indexer – CSA专用：关键信息打分+筛选"""

    def __init__(self, dim, topk=512):
        super().__init__()
        self.topk = topk
        self.score_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, compressed_kv):
        score = self.score_net(compressed_kv).squeeze(-1)
        top_val, top_idx = torch.topk(score, k=self.topk, dim=1)
        return top_idx, F.softmax(top_val, dim=-1)
