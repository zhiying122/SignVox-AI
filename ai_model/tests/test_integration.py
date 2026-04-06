"""
整合測試：驗證完整流程
KeypointPreprocessor → GestureDataset → 模型 → GesturePredictor
"""
import numpy as np
import pytest
import torch

from ai_model.data_pipeline.preprocessor import KeypointPreprocessor
from ai_model.models.lstm_classifier import BiLSTMClassifier
from ai_model.models.transformer_classifier import LightweightTransformerClassifier


NUM_CLASSES = 5
T = 30  # 序列長度


def test_end_to_end_pipeline(make_keypoint_matrix):
    """
    驗證完整流程：
    1. 生成假 keypoint 資料 [T, 75, 3]
    2. KeypointPreprocessor 前處理 → [T, 675]
    3. BiLSTMClassifier forward pass → [B, 5]，Softmax 加總為 1.0
    4. LightweightTransformerClassifier forward pass → [B, 5]，Softmax 加總為 1.0
    """
    # 1. 生成假資料
    keypoint_matrix = make_keypoint_matrix(T=T)
    assert keypoint_matrix.shape == (T, 75, 3)

    # 2. 前處理
    preprocessor = KeypointPreprocessor()
    feature_matrix, metadata = preprocessor.fit_transform(keypoint_matrix)

    assert feature_matrix.shape == (T, 675), (
        f"前處理輸出 shape 應為 [{T}, 675]，實際為 {feature_matrix.shape}"
    )
    assert metadata["is_valid"] is True

    # 準備 tensor：加 batch 維度
    x = torch.from_numpy(feature_matrix).unsqueeze(0)  # [1, T, 675]
    mask = torch.ones(1, T, dtype=torch.float32)        # [1, T]

    # 3. BiLSTMClassifier
    lstm_model = BiLSTMClassifier(
        input_dim=675,
        hidden_dim=32,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )
    lstm_model.eval()
    with torch.no_grad():
        lstm_out = lstm_model(x, mask)

    assert lstm_out.shape == (1, NUM_CLASSES), (
        f"BiLSTM 輸出 shape 應為 [1, {NUM_CLASSES}]，實際為 {lstm_out.shape}"
    )
    assert torch.allclose(lstm_out.sum(dim=-1), torch.ones(1), atol=1e-5), (
        f"BiLSTM Softmax 加總應為 1.0，實際為 {lstm_out.sum(dim=-1).item()}"
    )

    # 4. LightweightTransformerClassifier
    transformer_model = LightweightTransformerClassifier(
        input_dim=675,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        num_classes=NUM_CLASSES,
        dropout=0.0,
        max_seq_len=150,
    )
    transformer_model.eval()
    with torch.no_grad():
        transformer_out = transformer_model(x, mask)

    assert transformer_out.shape == (1, NUM_CLASSES), (
        f"Transformer 輸出 shape 應為 [1, {NUM_CLASSES}]，實際為 {transformer_out.shape}"
    )
    assert torch.allclose(transformer_out.sum(dim=-1), torch.ones(1), atol=1e-5), (
        f"Transformer Softmax 加總應為 1.0，實際為 {transformer_out.sum(dim=-1).item()}"
    )
