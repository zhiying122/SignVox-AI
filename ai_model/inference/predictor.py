"""
inference/predictor.py

GesturePredictor：推論 API，供 frontend/ 呼叫。
這是 ai_model/ 對外的唯一公開介面。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ai_model.data_pipeline.preprocessor import KeypointPreprocessor
from ai_model.models.lstm_classifier import BiLSTMClassifier
from ai_model.models.transformer_classifier import LightweightTransformerClassifier

logger = logging.getLogger(__name__)


class GesturePredictor:
    """
    推論 API，供 frontend/ 呼叫。
    這是 ai_model/ 對外的唯一公開介面。
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "auto",
    ):
        """
        載入模型權重，初始化時間 ≤ 3 秒。
        device="auto" 時自動偵測 CUDA 可用性。

        Raises
        ------
        FileNotFoundError
            若 model_path 不存在。
        """
        # 解析 device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 載入 YAML 設定檔
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config.get("model", {})
        architecture = model_cfg.get("architecture", "lstm").lower()

        # 依架構初始化模型
        if architecture == "lstm":
            self.model = BiLSTMClassifier(
                input_dim=model_cfg.get("input_dim", 675),
                hidden_dim=model_cfg.get("hidden_dim", 256),
                num_layers=model_cfg.get("num_layers", 2),
                num_classes=model_cfg.get("num_classes", 100),
                dropout=model_cfg.get("dropout", 0.3),
            )
        elif architecture in ("transformer", "lightweight_transformer"):
            self.model = LightweightTransformerClassifier(
                input_dim=model_cfg.get("input_dim", 675),
                d_model=model_cfg.get("d_model", 128),
                num_heads=model_cfg.get("num_heads", 4),
                num_layers=model_cfg.get("num_layers", 4),
                d_ff=model_cfg.get("d_ff", 256),
                num_classes=model_cfg.get("num_classes", 100),
                dropout=model_cfg.get("dropout", 0.1),
                max_seq_len=model_cfg.get("max_seq_len", 150),
            )
        else:
            raise ValueError(f"不支援的架構：{architecture}，請使用 'lstm' 或 'transformer'")

        # 載入模型權重
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型檔案不存在：{model_path}")

        state = torch.load(model_path, map_location=self.device)
        # 支援直接存 state_dict 或包在 checkpoint dict 中
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # 初始化前處理器
        self.preprocessor = KeypointPreprocessor()

        # max_seq_len（padding/截斷用）
        data_cfg = self.config.get("data", {})
        self.max_seq_len = data_cfg.get("max_seq_len", model_cfg.get("max_seq_len", 150))

        # 載入 label 映射
        self.label_map: dict[int, str] = self._load_label_map(model_cfg)

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def predict(self, keypoint_matrix: np.ndarray) -> dict:
        """
        單樣本推論。

        Parameters
        ----------
        keypoint_matrix : np.ndarray, shape [T, 75, 3]

        Returns
        -------
        dict with keys: gesture_label, confidence, inference_time_ms,
                        timestamp, keypoint_hash

        Raises
        ------
        ValueError
            若輸入 shape 不符合 [T, 75, 3]。
        """
        # 驗證輸入 shape
        if (
            keypoint_matrix.ndim != 3
            or keypoint_matrix.shape[1] != 75
            or keypoint_matrix.shape[2] != 3
        ):
            raise ValueError(
                f"輸入 shape 必須為 [T, 75, 3]，實際收到 {keypoint_matrix.shape}，"
                f"預期 (T, 75, 3)"
            )

        start_time = time.perf_counter()

        # 計算 keypoint_hash（MD5 of keypoint_matrix bytes）
        keypoint_hash = hashlib.md5(keypoint_matrix.tobytes()).hexdigest()

        # 前處理
        feature_matrix, _ = self.preprocessor.fit_transform(keypoint_matrix)
        # feature_matrix: [T, 675]

        # Padding / 截斷至 max_seq_len
        T = feature_matrix.shape[0]
        if T >= self.max_seq_len:
            features = feature_matrix[: self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - T
            features = np.concatenate(
                [feature_matrix, np.zeros((pad_len, feature_matrix.shape[1]), dtype=np.float32)],
                axis=0,
            )
            mask = np.concatenate(
                [np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)]
            )

        # 轉為 tensor，加 batch 維度
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)   # [1, max_seq_len, 675]
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)        # [1, max_seq_len]

        # 推論
        with torch.no_grad():
            probs = self.model(x, m)  # [1, num_classes]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 處理 NaN 輸出
        if torch.isnan(probs).any():
            logger.warning("推論輸出包含 NaN，回傳 confidence=0.0")
            gesture_label = self.label_map.get(0, "class_0")
            confidence = 0.0
        else:
            pred_idx = int(probs.argmax(dim=-1).item())
            confidence = float(probs[0, pred_idx].item())
            gesture_label = self.label_map.get(pred_idx, f"class_{pred_idx}")

        return {
            "gesture_label": gesture_label,
            "confidence": confidence,
            "inference_time_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat(),
            "keypoint_hash": keypoint_hash,
        }

    def predict_batch(self, keypoint_matrices: list[np.ndarray]) -> list[dict]:
        """批次推論，回傳與輸入等長的結果列表。"""
        return [self.predict(km) for km in keypoint_matrices]

    def serialize_result(self, result: dict) -> str:
        """將推論結果序列化為 JSON 字串。"""
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def deserialize_result(json_str: str) -> dict:
        """從 JSON 字串還原推論結果。"""
        return json.loads(json_str)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _load_label_map(self, model_cfg: dict) -> dict[int, str]:
        """
        載入 label 映射。

        優先順序：
        1. config 中的 label_map_path → 載入外部 JSON/YAML
        2. config 中的 label_map（內嵌 dict）
        3. 以 class_0, class_1... 作為預設標籤
        """
        # 1. 外部 label_map_path
        label_map_path = model_cfg.get("label_map_path") or self.config.get("label_map_path")
        if label_map_path:
            p = Path(label_map_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f) if p.suffix == ".json" else yaml.safe_load(f)
                # 支援 {int: str} 或 {str: str}
                return {int(k): v for k, v in raw.items()}
            else:
                logger.warning(f"label_map_path 不存在：{label_map_path}，使用預設標籤")

        # 2. 內嵌 label_map
        inline = model_cfg.get("label_map") or self.config.get("label_map")
        if inline and isinstance(inline, dict):
            return {int(k): v for k, v in inline.items()}

        # 3. 預設標籤
        num_classes = model_cfg.get("num_classes", 100)
        return {i: f"class_{i}" for i in range(num_classes)}
