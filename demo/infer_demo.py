# demo/infer_demo.py
"""Gemini 3.1 Pro 推理演示"""

import torch
from src.model import Gemini31ProFull


def main():
    model = Gemini31ProFull(
        vocab_size=50000,
        dim=8192,
        heads=64,
        num_layers=40,
    )
    model.eval()

    B = 1
    text_ids = torch.randn(B, 512, 8192) if hasattr(model.processor, 'img_enc') else torch.zeros(B, 512).long()
    img_feat = torch.randn(B, 2048)
    audio_feat = torch.randn(B, 1024)
    video_feat = torch.randn(B, 2048)
    modality_type_ids = torch.cat(
        [torch.zeros(B, 512), torch.ones(B, 1), torch.full((B, 1), 2), torch.full((B, 1), 3)], dim=1
    ).long()
    pos_ids = torch.arange(515).unsqueeze(0).expand(B, -1)

    print("Model loaded. Running forward pass...")
    with torch.no_grad():
        logits, tool_results = model(
            text_ids, img_feat, audio_feat, video_feat,
            modality_type_ids, pos_ids,
            thinking_level="medium",
        )
    print(f"Output logits shape: {logits.shape}")
    print(f"Tool plan logits shape: {tool_results[0].shape}")
    print(f"Tool verify prob shape: {tool_results[1].shape}")


if __name__ == "__main__":
    main()
