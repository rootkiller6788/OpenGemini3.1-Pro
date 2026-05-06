# src/layers/thinking_ctrl.py
import torch
import torch.nn as nn


class ReasoningBudgetController(nn.Module):
    """[Inferred] 推理预算控制器 — Low/Medium/High 三档

    Gemini API 暴露 thinking_level / thinking_budget 参数 ([Observed])，
    可推断存在动态推理资源分配机制。具体实现方式未知，
    当前用可学习 level embedding 做 toy 实现。
    """

    def __init__(self, dim):
        super().__init__()
        self.level_emb = nn.Embedding(3, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, level="medium"):
        if level == "low":
            lid = torch.tensor([0], device=x.device, dtype=torch.long)
        elif level == "high":
            lid = torch.tensor([2], device=x.device, dtype=torch.long)
        else:
            lid = torch.tensor([1], device=x.device, dtype=torch.long)

        level_vec = self.level_emb(lid).unsqueeze(0).unsqueeze(0)
        x = x + self.proj(level_vec)
        return x
