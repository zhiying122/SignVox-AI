"""
data_pipeline/preprocessor.py

KeypointPreprocessor：將原始 Keypoint_Matrix [T, 75, 3] 轉換為
訓練用特徵矩陣 [T, 675]（座標 + 速度 + 加速度各 225 維）。
"""
from __future__ import annotations

import numpy as np


N_JOINTS = 75          # 左手21 + 右手21 + 姿態33
N_FEATURES = N_JOINTS * 9  # 675 = 75 × (3座標 + 3速度 + 3加速度)

LEFT_WRIST_IDX = 0
RIGHT_WRIST_IDX = 21


class KeypointPreprocessor:
    """
    將原始 Keypoint_Matrix 轉換為訓練用特徵矩陣。

    輸入：[T, N_joints, 3]（N_joints = 75：左手21 + 右手21 + 姿態33）
    輸出：[T, N_features]（N_features = 75*3 座標 + 75*3 速度 + 75*3 加速度 = 675）
    """

    def __init__(self, missing_threshold: float = 0.3):
        self.missing_threshold = missing_threshold

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        keypoint_matrix: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        完整前處理流程。

        Parameters
        ----------
        keypoint_matrix : np.ndarray, shape [T, 75, 3]

        Returns
        -------
        feature_matrix : np.ndarray, shape [T, 675]
        metadata : dict
            missing_ratio (float), is_valid (bool), fill_count (int)

        Raises
        ------
        ValueError
            若輸入 shape 不為 [T, 75, 3]。
        """
        if (
            keypoint_matrix.ndim != 3
            or keypoint_matrix.shape[1] != N_JOINTS
            or keypoint_matrix.shape[2] != 3
        ):
            raise ValueError(
                f"輸入 shape 必須為 [T, 75, 3]，實際收到 {keypoint_matrix.shape}"
            )

        matrix = keypoint_matrix.astype(np.float32, copy=True)

        # 1. 插值填補缺失值
        filled, missing_ratio = self._interpolate_missing(matrix)

        # 計算填補數量（原始缺失位置數）
        missing_mask = _missing_mask(matrix)
        fill_count = int(missing_mask.sum())

        # 2. 腕關節正規化
        normalized = self._normalize_to_wrist(filled)

        # 3. 計算速度與加速度，得到 [T, 75, 9]
        combined = self._compute_velocity_acceleration(normalized)

        # 4. 攤平為 [T, 675]
        T = combined.shape[0]
        feature_matrix = combined.reshape(T, N_FEATURES).astype(np.float32)

        is_valid = missing_ratio <= self.missing_threshold

        metadata = {
            "missing_ratio": float(missing_ratio),
            "is_valid": bool(is_valid),
            "fill_count": fill_count,
        }

        return feature_matrix, metadata

    def serialize(self, feature_matrix: np.ndarray, path: str) -> None:
        """將特徵矩陣序列化為 .npy 格式。"""
        np.save(path, feature_matrix)

    def deserialize(self, path: str) -> np.ndarray:
        """
        從 .npy 格式載入特徵矩陣。

        Raises
        ------
        ValueError
            若載入的陣列不是 2D 或最後一維不為 675。
        """
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[-1] != N_FEATURES:
            raise ValueError(
                f"格式不符：預期 2D array，shape[-1] == {N_FEATURES}，"
                f"實際收到 shape={arr.shape}，ndim={arr.ndim}"
            )
        return arr

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _interpolate_missing(
        self,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        對 NaN 與零值（視為缺失）進行線性插值。

        Parameters
        ----------
        matrix : np.ndarray, shape [T, 75, 3]

        Returns
        -------
        filled_matrix : np.ndarray, shape [T, 75, 3]
        missing_ratio : float  缺失幀比例（以整幀為單位）
        """
        T, J, C = matrix.shape
        filled = matrix.copy()

        # 判斷缺失：NaN 或全零（整個關節點的 3 個座標都是 0）
        # 以「幀」為單位計算缺失率：若某幀所有關節點都缺失，視為缺失幀
        frame_missing = _missing_mask(matrix)  # [T, J, C] bool

        # 缺失率：以「幀中至少有一個關節點缺失」計算
        frames_with_missing = np.any(frame_missing.reshape(T, -1), axis=1)  # [T]
        missing_ratio = float(frames_with_missing.mean())

        # 對每個關節點的每個座標維度做線性插值
        for j in range(J):
            for c in range(C):
                col = filled[:, j, c]
                bad = frame_missing[:, j, c]

                if not np.any(bad):
                    continue

                if np.all(bad):
                    # 整列都缺失，填 0
                    filled[:, j, c] = 0.0
                    continue

                # 找到有效索引
                good_idx = np.where(~bad)[0]
                good_val = col[good_idx]

                # 線性插值（含邊界外推用最近有效值）
                filled[:, j, c] = np.interp(
                    np.arange(T),
                    good_idx,
                    good_val,
                )

        return filled, missing_ratio

    def _normalize_to_wrist(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        以腕關節為原點進行相對座標轉換。

        - 左手 [0:21]  減去 index 0（左手腕）
        - 右手 [21:42] 減去 index 21（右手腕）
        - 姿態 [42:75] 保持不變

        Parameters
        ----------
        matrix : np.ndarray, shape [T, 75, 3]

        Returns
        -------
        normalized : np.ndarray, shape [T, 75, 3]
        """
        normalized = matrix.copy()

        # 左手腕座標 [T, 1, 3]
        left_wrist = matrix[:, LEFT_WRIST_IDX : LEFT_WRIST_IDX + 1, :]
        normalized[:, 0:21, :] = matrix[:, 0:21, :] - left_wrist

        # 右手腕座標 [T, 1, 3]
        right_wrist = matrix[:, RIGHT_WRIST_IDX : RIGHT_WRIST_IDX + 1, :]
        normalized[:, 21:42, :] = matrix[:, 21:42, :] - right_wrist

        # 姿態關節點不做正規化
        # normalized[:, 42:75, :] 保持不變

        return normalized

    def _compute_velocity_acceleration(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        """
        計算速度與加速度向量。

        Parameters
        ----------
        coords : np.ndarray, shape [T, 75, 3]

        Returns
        -------
        combined : np.ndarray, shape [T, 75, 9]
            沿最後一維 concatenate [coords, velocity, acceleration]
        """
        T, J, C = coords.shape

        # 速度：diff 後第一幀補零
        velocity = np.zeros_like(coords)
        velocity[1:] = np.diff(coords, axis=0)

        # 加速度：diff(velocity) 後第一幀補零
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = np.diff(velocity, axis=0)

        # 沿最後一維拼接 → [T, 75, 9]
        combined = np.concatenate([coords, velocity, acceleration], axis=-1)

        return combined.astype(np.float32)


# ------------------------------------------------------------------
# 模組內部工具函式
# ------------------------------------------------------------------

def _missing_mask(matrix: np.ndarray) -> np.ndarray:
    """
    回傳布林遮罩，True 表示該位置為缺失（NaN 或零值）。

    零值判斷：以整個關節點（3 個座標）全為 0 視為缺失，
    再廣播回各座標維度。

    Parameters
    ----------
    matrix : np.ndarray, shape [T, J, C]

    Returns
    -------
    mask : np.ndarray, shape [T, J, C], dtype=bool
    """
    nan_mask = np.isnan(matrix)

    # 整個關節點的 3 個座標都是 0 → 視為缺失
    zero_joint = np.all(matrix == 0.0, axis=-1, keepdims=True)  # [T, J, 1]
    zero_mask = np.broadcast_to(zero_joint, matrix.shape)

    return nan_mask | zero_mask
