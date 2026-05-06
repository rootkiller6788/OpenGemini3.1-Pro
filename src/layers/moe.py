# src/layers/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoELayer(nn.Module):
    """[Speculative] 稀疏 MoE 层 — 1 共享 + N 路由专家

    MoE 是行业常用技术，但 Gemini 是否使用 MoE、专家数量等均未公开。
    当前为 naive 参考实现：每个 expert 对完整 batch 执行前向计算，
    即使 mask 为空也会算出，未做负载均衡或容量限制。
    """

    def __init__(self, dim, num_experts=256, shared_experts=1, top_k=8):
        super().__init__()
        self.num_experts = num_experts
        self.shared_experts = shared_experts
        self.top_k = top_k

        self.shared_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        B, T, D = x.shape

        x_shared = self.shared_mlp(x)

        gate_logits = self.gate(x)
        top_val, top_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(top_val, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            ei = top_idx[:, :, k]
            wk = weights[:, :, k: k + 1]
            for e in range(self.num_experts):
                msk = (ei == e).float().unsqueeze(-1)
                out += msk * wk * self.experts[e](x)

        return x_shared + out
