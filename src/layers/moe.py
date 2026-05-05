# src/layers/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoEGemini(nn.Module):
    """稀疏MoE专家层（1共享 + 256路由，每Token激活8个）"""

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
            wk = weights[:, :, k : k + 1]
            for e in range(self.num_experts):
                msk = (ei == e).float().unsqueeze(-1)
                out += msk * wk * self.experts[e](x)

        return x_shared + out
