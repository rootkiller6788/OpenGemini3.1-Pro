# demo/infer_demo.py
"""Gemini-style architecture inference demo"""

import torch
from src.model import ToyGeminiBackbone, GeminiStyleAgentRuntime


def main():
    dim = 8192
    backbone = ToyGeminiBackbone(
        vocab_size=50000,
        dim=dim,
        heads=64,
        num_layers=2,  # small for demo
    )
    agent = GeminiStyleAgentRuntime(dim, num_tools=16)

    backbone.eval()
    agent.eval()

    B = 1
    T = 64

    text_ids = torch.randint(0, 50000, (B, T))
    img_feat = torch.randn(B, 2048)
    audio_feat = torch.randn(B, 1024)
    video_feat = torch.randn(B, 2048)

    total_t = T + 3  # text + image + audio + video
    type_ids = torch.cat(
        [torch.zeros(B, T), torch.ones(B, 1), torch.full((B, 1), 2), torch.full((B, 1), 3)],
        dim=1,
    ).long()
    pos_ids = torch.arange(total_t).unsqueeze(0).expand(B, -1)

    print("Running backbone forward pass...")
    with torch.no_grad():
        logits = backbone(
            text_ids, img_feat, audio_feat, video_feat,
            type_ids, pos_ids,
        )
    print(f"Backbone logits shape: {logits.shape}")

    print("Running agent runtime forward pass...")
    with torch.no_grad():
        x_hidden = torch.randn(1, total_t, dim)
        x_out, tool_results = agent(x_hidden, thinking_level="medium")
    print(f"Agent output shape: {x_out.shape}")
    print(f"Tool plan shape: {tool_results[0].shape}")
    print(f"Verify prob shape: {tool_results[1].shape}")


if __name__ == "__main__":
    main()
