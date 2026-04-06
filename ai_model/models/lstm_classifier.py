"""
BiLSTM 手語分類器。

架構：
    Input [B, T, N_features=675]
    → Linear Projection [B, T, hidden_dim]
    → BiLSTM Layer 1 [B, T, hidden_dim*2]
    → Dropout(0.3)
    → BiLSTM Layer 2 [B, T, hidden_dim*2]
    → Dropout(0.3)
    → Masked Mean Pooling [B, hidden_dim*2]
    → Linear [B, num_classes]
    → Softmax → Confidence_Score [B, num_classes]
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    雙層雙向 LSTM 手語分類器。

    參數量估算（hidden=256, num_classes=100）：
        Layer1: 4 * (256+256) * 256 * 2 ≈ 1.0M
        Layer2: 4 * (512+256) * 256 * 2 ≈ 1.6M
        FC: 512 * 100 ≈ 51K
        總計 ≈ 2.7M 參數
    """

    def __init__(
        self,
        input_dim: int = 675,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Linear Projection：降維，讓 LSTM 輸入維度可調
        self.projection = nn.Linear(input_dim, hidden_dim)

        # 兩層 BiLSTM，每層獨立建立以便在中間插入 Dropout
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(p=dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  # BiLSTM 輸出為 hidden_dim*2
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(p=dropout)

        # 輸出分類層
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,    # [B, T, N_features]
        mask: torch.Tensor, # [B, T]，1=有效幀，0=padding
    ) -> torch.Tensor:      # [B, num_classes]
        # Linear Projection
        x = self.projection(x)  # [B, T, hidden_dim]

        # BiLSTM Layer 1
        x, _ = self.lstm1(x)    # [B, T, hidden_dim*2]
        x = self.dropout1(x)

        # BiLSTM Layer 2
        x, _ = self.lstm2(x)    # [B, T, hidden_dim*2]
        x = self.dropout2(x)

        # Masked Mean Pooling：只對有效幀（mask=1）取平均
        # mask: [B, T] → [B, T, 1]
        mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
        masked = x * mask_expanded                  # [B, T, hidden_dim*2]
        sum_valid = masked.sum(dim=1)               # [B, hidden_dim*2]
        count_valid = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]，避免除以零
        pooled = sum_valid / count_valid            # [B, hidden_dim*2]

        # 分類 + Softmax
        logits = self.classifier(pooled)            # [B, num_classes]
        return torch.softmax(logits, dim=-1)        # [B, num_classes]
