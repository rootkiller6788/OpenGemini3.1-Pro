# src/layers/tool_strategy.py
import torch
import torch.nn as nn


class ToolRuntime(nn.Module):
    """[Inferred] 工具运行时 — 规划 + 自校验两阶段

    从可观察的 Gemini API 行为推断：Gemini 支持 function calling，
    可以自主规划调用工具并校验结果。内部实现细节未知，
    当前为 toy 架构假设。
    """

    def __init__(self, dim, num_tools=16):
        super().__init__()
        self.num_tools = num_tools
        self.plan_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_tools),
        )
        self.verify_net = nn.Sequential(
            nn.Linear(dim + num_tools, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.proj = nn.Linear(num_tools, dim)

    def forward(self, x):
        plan_logits = self.plan_net(x)
        verify_in = torch.cat([x, plan_logits], dim=-1)
        verify_prob = torch.sigmoid(self.verify_net(verify_in))
        return plan_logits, verify_prob

    def integrate_results(self, x, tool_results):
        plan_logits, _ = tool_results
        x = x + self.proj(plan_logits)
        return x
