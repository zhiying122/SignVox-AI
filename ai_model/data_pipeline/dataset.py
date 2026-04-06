"""
data_pipeline/dataset.py

GestureDataset：PyTorch Dataset，支援可變長度序列的 padding 與 masking。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data

FEATURE_DIM = 675  # 75 關節 × 9（座標 + 速度 + 加速度）

REQUIRED_CSV_COLUMNS = {
    "sample_id",
    "gesture_label",
    "start_frame",
    "end_frame",
    "quality_flag",
}


class GestureDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset，支援可變長度序列的 padding 與 masking。

    Parameters
    ----------
    feature_dir : str
        存放 .npy 特徵檔案的目錄路徑。
    annotation_csv : str
        標註 CSV 檔案路徑，需包含欄位：
        sample_id, gesture_label, start_frame, end_frame, quality_flag
    max_seq_len : int
        序列最大長度，超過時截斷，不足時補零。預設 150。
    augment : bool
        是否啟用資料增強（保留供未來擴充）。預設 False。

    Raises
    ------
    FileNotFoundError
        若有任何 sample_id 對應的 .npy 檔案不存在，列出所有缺失的 sample_id。
    ValueError
        若 CSV 缺少必要欄位。
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_csv: str,
        max_seq_len: int = 150,
        augment: bool = False,
    ) -> None:
        self.feature_dir = feature_dir
        self.max_seq_len = max_seq_len
        self.augment = augment

        # 讀取標註 CSV
        df = pd.read_csv(annotation_csv)

        # 驗證必要欄位
        missing_cols = REQUIRED_CSV_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"標註 CSV 缺少必要欄位：{sorted(missing_cols)}"
            )

        self.annotations = df.reset_index(drop=True)

        # 驗證所有 .npy 特徵檔案存在
        missing_files = [
            sid
            for sid in self.annotations["sample_id"]
            if not os.path.isfile(os.path.join(feature_dir, f"{sid}.npy"))
        ]
        if missing_files:
            raise FileNotFoundError(
                f"以下 sample_id 對應的特徵檔案不存在：{missing_files}"
            )

        # 建立 gesture_label → int 映射（依字母排序確保可重現性）
        unique_labels = sorted(self.annotations["gesture_label"].unique())
        self.label_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.idx_to_label: dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }

    # ------------------------------------------------------------------
    # Dataset 介面
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """
        載入並回傳單一樣本。

        Returns
        -------
        dict with keys:
            'features' : torch.FloatTensor [max_seq_len, 675]
            'label'    : int
            'mask'     : torch.FloatTensor [max_seq_len]  (1=有效, 0=padding)
            'sample_id': str
        """
        row = self.annotations.iloc[idx]
        sample_id: str = str(row["sample_id"])
        gesture_label: str = str(row["gesture_label"])

        # 載入特徵矩陣 [T, 675]
        npy_path = os.path.join(self.feature_dir, f"{sample_id}.npy")
        features: np.ndarray = np.load(npy_path).astype(np.float32)

        T = features.shape[0]

        # 截斷或 padding
        if T >= self.max_seq_len:
            # 截斷
            features = features[: self.max_seq_len]
            valid_len = self.max_seq_len
        else:
            # padding 補零
            pad_len = self.max_seq_len - T
            features = np.concatenate(
                [features, np.zeros((pad_len, FEATURE_DIM), dtype=np.float32)],
                axis=0,
            )
            valid_len = T

        # mask：有效幀為 1.0，padding 幀為 0.0
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:valid_len] = 1.0

        return {
            "features": torch.from_numpy(features),   # [max_seq_len, 675]
            "label": self.label_to_idx[gesture_label],
            "mask": torch.from_numpy(mask),            # [max_seq_len]
            "sample_id": sample_id,
        }

    # ------------------------------------------------------------------
    # collate_fn
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        動態 padding 至批次內最長序列。

        Parameters
        ----------
        batch : list[dict]
            每個元素為 __getitem__ 的回傳值。

        Returns
        -------
        dict with keys:
            'features'   : torch.FloatTensor [B, T_max, 675]
            'labels'     : torch.LongTensor  [B]
            'masks'      : torch.FloatTensor [B, T_max]
            'sample_ids' : list[str]
        """
        # 找出批次內各樣本的有效長度（mask 中 1.0 的數量）
        valid_lens = [int(item["mask"].sum().item()) for item in batch]
        t_max = max(valid_lens)

        B = len(batch)
        feat_dim = batch[0]["features"].shape[-1]

        features_out = torch.zeros(B, t_max, feat_dim, dtype=torch.float32)
        masks_out = torch.zeros(B, t_max, dtype=torch.float32)
        labels_out = torch.zeros(B, dtype=torch.long)
        sample_ids: list[str] = []

        for i, item in enumerate(batch):
            vl = valid_lens[i]
            features_out[i, :vl, :] = item["features"][:vl]
            masks_out[i, :vl] = 1.0
            labels_out[i] = item["label"]
            sample_ids.append(item["sample_id"])

        return {
            "features": features_out,
            "labels": labels_out,
            "masks": masks_out,
            "sample_ids": sample_ids,
        }
