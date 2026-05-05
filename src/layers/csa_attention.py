# src/layers/csa_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .indexer import LightningIndexer


class CSA(nn.Module):
    """CSA：压缩稀疏注意力（Compressed Sparse Attention）"""

    def __init__(self, dim, heads, compression_rate=4, topk=512):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.cr = compression_rate
        self.topk = topk

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.k_compressor = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=compression_rate,
            stride=compression_rate,
        )
        self.v_compressor = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=compression_rate,
            stride=compression_rate,
        )

        self.indexer = LightningIndexer(dim, topk=topk)

    def forward(self, x, mask):
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)

        k_comp = self.k_compressor(k.reshape(-1, D, T)).reshape(B, self.heads, D, -1)
        v_comp = self.v_compressor(v.reshape(-1, D, T)).reshape(B, self.heads, D, -1)

        B, H, D, Tc = k_comp.shape
        k_comp_flat = k_comp.transpose(1, 2).reshape(B, D, H * Tc).transpose(1, 2)
        v_comp_flat = v_comp.transpose(1, 2).reshape(B, D, H * Tc).transpose(1, 2)
        top_idx, top_w = self.indexer(k_comp_flat)

        k_top = k_comp_flat.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        v_top = v_comp_flat.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))

        k_top = k_top.reshape(B, self.heads, self.topk, self.hd).transpose(2, 1)
        v_top = v_top.reshape(B, self.heads, self.topk, self.hd).transpose(2, 1)

        attn = (q @ k_top.transpose(-2, -1)) / (self.hd**0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_top).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
