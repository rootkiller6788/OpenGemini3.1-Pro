# src/layers/tool_strategy.py
import torch
import torch.nn as nn


class NativeToolStrategyLayer(nn.Module):
    """原生工具策略层（规划→联动→自校验）"""

    def __init__(self, dim):
        super().__init__()
        self.plan_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 16),
        )
        self.verify_net = nn.Sequential(
            nn.Linear(dim + 16, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.proj = nn.Linear(16, dim)

    def forward(self, x):
        plan_logits = self.plan_net(x)
        verify_in = torch.cat([x, plan_logits], dim=-1)
        verify_prob = torch.sigmoid(self.verify_net(verify_in))
        return plan_logits, verify_prob

    def integrate_results(self, x, tool_results):
        plan_logits, _ = tool_results
        x = x + self.proj(plan_logits)
        return x
