# src/layers/thinking_ctrl.py
import torch
import torch.nn as nn


class ThinkingLevelController(nn.Module):
    """Thinking Level 控制模块（Low/Medium/High三档）"""

    def __init__(self, dim):
        super().__init__()
        self.level_emb = nn.Embedding(3, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, level="medium"):
        if level == "low":
            lid = torch.tensor([0], device=x.device)
        elif level == "high":
            lid = torch.tensor([2], device=x.device)
        else:
            lid = torch.tensor([1], device=x.device)

        level_vec = self.level_emb(lid).unsqueeze(0).unsqueeze(0)
        x = x + self.proj(level_vec)
        return x
