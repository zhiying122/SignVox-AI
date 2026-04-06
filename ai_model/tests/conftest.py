"""
共用 pytest fixtures，供 ai_model/tests/ 下所有測試使用。
"""
import numpy as np
import pandas as pd
import pytest

# 設定全域隨機種子
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


@pytest.fixture
def make_keypoint_matrix():
    """
    生成 shape [T, 75, 3] 的隨機 float32 ndarray。

    用法：
        matrix = make_keypoint_matrix(T=30)
        matrix = make_keypoint_matrix(T=50, n_joints=75)
    """
    def _factory(T: int, n_joints: int = 75) -> np.ndarray:
        rng = np.random.default_rng(GLOBAL_SEED)
        return rng.random((T, n_joints, 3)).astype(np.float32)

    return _factory


@pytest.fixture
def make_annotation_csv(tmp_path):
    """
    生成假標註 CSV 檔案，包含 sample_id, gesture_label,
    start_frame, end_frame, quality_flag 欄位。

    用法：
        csv_path = make_annotation_csv(n_samples=20, n_classes=5)
    """
    def _factory(n_samples: int, n_classes: int) -> str:
        rng = np.random.default_rng(GLOBAL_SEED)
        labels = [f"gesture_{i % n_classes}" for i in range(n_samples)]
        data = {
            "sample_id": [f"sample_{i:04d}" for i in range(n_samples)],
            "gesture_label": labels,
            "start_frame": rng.integers(0, 10, size=n_samples).tolist(),
            "end_frame": rng.integers(20, 150, size=n_samples).tolist(),
            "quality_flag": rng.integers(0, 3, size=n_samples).tolist(),
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "annotations.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    return _factory
