# src/layers/hca_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalCompressedAttention(nn.Module):
    """[Speculative] 全局压缩注意力 — Conv1d 重度压缩 KV ×128，全量注意力捕获全局结构

    当前为 naive 参考实现，未针对长上下文优化。
    没有官方证据表明 Gemini 使用这种特定的压缩注意力机制。
    """

    def __init__(self, dim, heads, compression_rate=128):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.cr = compression_rate

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.k_compressor = nn.Conv1d(
            dim, dim, kernel_size=compression_rate, stride=compression_rate
        )
        self.v_compressor = nn.Conv1d(
            dim, dim, kernel_size=compression_rate, stride=compression_rate
        )

    def forward(self, x, mask):
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.hd).transpose(1, 2)

        k_comp = self.k_compressor(k.reshape(-1, D, T)).reshape(B, self.heads, D, -1)
        v_comp = self.v_compressor(v.reshape(-1, D, T)).reshape(B, self.heads, D, -1)

        attn = (q @ k_comp.transpose(-2, -1)) / (self.hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_comp).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
