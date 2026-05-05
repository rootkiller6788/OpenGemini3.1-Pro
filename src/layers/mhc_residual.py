# src/layers/mhc_residual.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldConstrainedResidual(nn.Module):
    """mHC：流形约束残差连接（Manifold Constrained Residual）"""

    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.dim = dim

        self.W = nn.Linear(dim, n_streams * n_streams)
        self.b = nn.Parameter(torch.zeros(n_streams, n_streams))

        self.pre = nn.Linear(dim, n_streams)
        self.post = nn.Linear(n_streams, dim)

    def forward(self, x):
        B, T, D = x.shape
        W_mat = F.softplus(self.W(x).view(-1, T, self.n_streams, self.n_streams))
        for _ in range(5):
            W_mat = W_mat / W_mat.sum(dim=-1, keepdim=True)
            W_mat = W_mat / W_mat.sum(dim=-2, keepdim=True)

        x_stream = self.pre(x).unsqueeze(-1)
        x_stream = torch.matmul(W_mat, x_stream).squeeze(-1)
        x_out = self.post(x_stream)
        return x + x_out
