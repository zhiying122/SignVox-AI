"""
輕量化 Transformer 手語分類器。

架構：
    Input [B, T, N_features=675]
    → Linear Projection [B, T, d_model=128]
    → Positional Encoding（可學習式）
    → TransformerEncoder × 4 層
        每層：MultiHeadAttention(heads=4, d_k=32) + FFN(d_ff=256)
    → CLS Token Pooling [B, d_model]
    → Linear [B, num_classes]
    → Softmax → Confidence_Score [B, num_classes]

參數量估算（d_model=128, heads=4, layers=4, num_classes=100）：
    Projection: 675*128 ≈ 86K
    Per Layer: Attn(128*128*4) + FFN(128*256*2) ≈ 131K
    4 Layers: ≈ 524K
    CLS + FC: 128*100 ≈ 13K
    總計 ≈ 0.6M 參數（遠低於 5M 上限）
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """可學習式位置編碼，形狀 [1, max_seq_len, d_model]"""

    def __init__(self, d_model: int, max_seq_len: int = 150):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]，截取至實際序列長度
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class LightweightTransformerClassifier(nn.Module):
    """
    輕量化 Transformer 手語分類器（≤ 5M 參數）。
    """

    def __init__(
        self,
        input_dim: int = 675,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        num_classes: int = 100,
        dropout: float = 0.1,
        max_seq_len: int = 150,
    ):
        super().__init__()

        # Linear Projection：input_dim → d_model
        self.projection = nn.Linear(input_dim, d_model)

        # 可學習式位置編碼（+1 是為了 CLS Token 佔用一個位置）
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len + 1)

        # CLS Token：[1, 1, d_model]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 輸出分類層
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,    # [B, T, N_features]
        mask: torch.Tensor, # [B, T]，1=有效幀，0=padding
    ) -> torch.Tensor:      # [B, num_classes]
        B = x.size(0)

        # Linear Projection
        x = self.projection(x)  # [B, T, d_model]

        # 前置 CLS Token：[B, 1, d_model]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)          # [B, T+1, d_model]

        # Positional Encoding
        x = self.pos_encoding(x)  # [B, T+1, d_model]

        # 建立 src_key_padding_mask：True=需遮蔽（padding 位置）
        # mask: [B, T]，1=有效，0=padding → 轉換為 True=padding
        # CLS Token 永遠有效（False），接在前面
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)  # [B, 1]
        padding_mask = (mask == 0)                                         # [B, T]
        src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1) # [B, T+1]

        # TransformerEncoder
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # [B, T+1, d_model]

        # 取 CLS Token 輸出（index 0）
        cls_out = out[:, 0, :]  # [B, d_model]

        # 分類 + Softmax
        logits = self.classifier(cls_out)       # [B, num_classes]
        return torch.softmax(logits, dim=-1)    # [B, num_classes]
