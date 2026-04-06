"""
trainer/train.py

GestureTrainer：統一訓練管線，支援 LSTM 與 Transformer 架構切換。
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import yaml
from sklearn.metrics import confusion_matrix, f1_score

from ai_model.data_pipeline.dataset import GestureDataset
from ai_model.models.lstm_classifier import BiLSTMClassifier
from ai_model.models.transformer_classifier import LightweightTransformerClassifier
from ai_model.trainer.registry import ModelRegistry

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = {"experiment", "model", "training", "data", "paths"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subject_independent_split(
    annotations,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    依受試者獨立切分索引。
    sample_id 格式如 "subject_01_sample_001"，前綴為 "subject_01"。
    若無法識別受試者，退化為隨機切分。
    """
    rng = np.random.default_rng(seed)

    def _extract_subject(sample_id: str) -> str:
        parts = sample_id.split("_")
        # 嘗試識別 "subject_XX" 前綴
        if len(parts) >= 2 and parts[0].lower() in ("subject", "sub", "s"):
            return "_".join(parts[:2])
        return sample_id  # 退化：用 sample_id 本身

    sample_ids = annotations["sample_id"].tolist()
    subjects = [_extract_subject(str(sid)) for sid in sample_ids]
    unique_subjects = list(dict.fromkeys(subjects))  # 保持順序去重

    if len(unique_subjects) <= 1:
        # 退化為隨機切分
        indices = np.arange(len(annotations))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val :].tolist()
        return train_idx, val_idx, test_idx

    # 受試者獨立切分
    rng.shuffle(unique_subjects)
    n_train_subj = max(1, int(len(unique_subjects) * train_ratio))
    n_val_subj = max(1, int(len(unique_subjects) * val_ratio))

    train_subjects = set(unique_subjects[:n_train_subj])
    val_subjects = set(unique_subjects[n_train_subj : n_train_subj + n_val_subj])
    # 剩餘歸測試集

    train_idx, val_idx, test_idx = [], [], []
    for i, subj in enumerate(subjects):
        if subj in train_subjects:
            train_idx.append(i)
        elif subj in val_subjects:
            val_idx.append(i)
        else:
            test_idx.append(i)

    # 若某集合為空，退化為隨機切分
    if not train_idx or not val_idx or not test_idx:
        indices = np.arange(len(annotations))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val :].tolist()

    return train_idx, val_idx, test_idx


class GestureTrainer:
    """統一訓練管線，支援 LSTM 與 Transformer 架構切換。"""

    def __init__(self, config_path: str) -> None:
        """從 YAML 設定檔載入所有超參數，驗證必要欄位。"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: dict = yaml.safe_load(f)

        # 驗證必要欄位
        missing = REQUIRED_CONFIG_KEYS - set(self.config.keys())
        if missing:
            raise KeyError(f"設定檔缺少必要欄位：{sorted(missing)}")

        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 設定隨機種子
        seed = self.config["experiment"].get("seed", 42)
        _set_seed(seed)

        # 產生 experiment_id（時間戳記）
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 初始化為 None，等待 setup() 呼叫
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None
        self._num_classes: int = 0

        logger.info("GestureTrainer 初始化完成，experiment_id=%s", self.experiment_id)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """初始化資料集、模型、優化器、排程器。"""
        cfg_model = self.config["model"]
        cfg_train = self.config["training"]
        cfg_data = self.config["data"]

        # ---- 建立完整 GestureDataset（全量），再切分 ----
        full_dataset = GestureDataset(
            feature_dir=cfg_data["feature_dir"],
            annotation_csv=cfg_data["annotation_csv"],
            max_seq_len=cfg_data.get("max_seq_len", 150),
            augment=False,
        )

        self._num_classes = len(full_dataset.label_to_idx)
        annotations = full_dataset.annotations
        seed = self.config["experiment"].get("seed", 42)

        train_idx, val_idx, test_idx = _subject_independent_split(
            annotations,
            train_ratio=cfg_data.get("train_ratio", 0.8),
            val_ratio=cfg_data.get("val_ratio", 0.1),
            seed=seed,
        )

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

        batch_size = cfg_train.get("batch_size", 32)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=GestureDataset.collate_fn,
            drop_last=False,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GestureDataset.collate_fn,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GestureDataset.collate_fn,
        )

        # ---- 初始化模型 ----
        arch = cfg_model.get("architecture", "lstm").lower()
        if arch == "lstm":
            self.model = BiLSTMClassifier(
                input_dim=cfg_model.get("input_dim", 675),
                hidden_dim=cfg_model.get("hidden_dim", 256),
                num_layers=cfg_model.get("num_layers", 2),
                num_classes=self._num_classes,
                dropout=cfg_model.get("dropout", 0.3),
            )
        elif arch == "transformer":
            self.model = LightweightTransformerClassifier(
                input_dim=cfg_model.get("input_dim", 675),
                d_model=cfg_model.get("d_model", 128),
                num_heads=cfg_model.get("num_heads", 4),
                num_layers=cfg_model.get("num_layers", 4),
                d_ff=cfg_model.get("d_ff", 256),
                num_classes=self._num_classes,
                dropout=cfg_model.get("dropout", 0.1),
                max_seq_len=cfg_data.get("max_seq_len", 150),
            )
        else:
            raise ValueError(f"不支援的架構：{arch}，請使用 'lstm' 或 'transformer'")

        self.model = self.model.to(self.device)

        # ---- 損失函數 ----
        balance_strategy = cfg_train.get("balance_strategy", "")
        if balance_strategy == "weighted_loss":
            class_weights = self._compute_class_weights(full_dataset, train_idx)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ---- 優化器 ----
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg_train.get("learning_rate", 1e-3),
            weight_decay=cfg_train.get("weight_decay", 1e-4),
        )

        # ---- 排程器 ----
        epochs = cfg_train.get("epochs", 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        logger.info(
            "setup 完成：架構=%s，類別數=%d，訓練樣本=%d，驗證樣本=%d，測試樣本=%d",
            arch, self._num_classes, len(train_idx), len(val_idx), len(test_idx),
        )

    def _compute_class_weights(
        self,
        full_dataset: GestureDataset,
        train_idx: list[int],
    ) -> torch.Tensor:
        """計算訓練集各類別的反頻率權重。"""
        labels = [full_dataset.annotations.iloc[i]["gesture_label"] for i in train_idx]
        label_indices = [full_dataset.label_to_idx[lbl] for lbl in labels]
        counts = np.bincount(label_indices, minlength=self._num_classes).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)  # 避免除以零
        weights = 1.0 / counts
        weights = weights / weights.sum() * self._num_classes  # 正規化
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # _train_epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        """單一 epoch 訓練，回傳 {loss, accuracy, f1}。"""
        assert self.model is not None and self.train_loader is not None
        self.model.train()

        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in self.train_loader:
            features = batch["features"].to(self.device)   # [B, T, 675]
            labels = batch["labels"].to(self.device)       # [B]
            masks = batch["masks"].to(self.device)         # [B, T]

            self.optimizer.zero_grad()

            # 模型輸出已是 Softmax 機率，需轉為 log 機率再用 NLLLoss
            # 等效於直接用 CrossEntropyLoss(log_softmax(logits))
            probs = self.model(features, masks)            # [B, num_classes]
            log_probs = torch.log(probs.clamp(min=1e-9))  # 避免 log(0)

            loss = nn.functional.nll_loss(
                log_probs,
                labels,
                weight=self.criterion.weight if hasattr(self.criterion, "weight") else None,
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = probs.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        n = len(all_labels)
        avg_loss = total_loss / max(n, 1)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    # ------------------------------------------------------------------
    # _validate
    # ------------------------------------------------------------------

    def _validate(self) -> dict:
        """驗證集評估，回傳 {loss, accuracy, f1, confusion_matrix}。"""
        assert self.model is not None and self.val_loader is not None
        self.model.eval()

        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                masks = batch["masks"].to(self.device)

                probs = self.model(features, masks)
                log_probs = torch.log(probs.clamp(min=1e-9))

                loss = nn.functional.nll_loss(
                    log_probs,
                    labels,
                    weight=self.criterion.weight if hasattr(self.criterion, "weight") else None,
                )
                total_loss += loss.item() * features.size(0)
                preds = probs.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        n = len(all_labels)
        avg_loss = total_loss / max(n, 1)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self._num_classes)))

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": cm,
        }

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """完整訓練迴圈，含 Early Stopping、日誌記錄、最佳模型儲存。"""
        assert self.model is not None, "請先呼叫 setup()"

        cfg_train = self.config["training"]
        cfg_paths = self.config["paths"]
        epochs = cfg_train.get("epochs", 100)
        patience = cfg_train.get("early_stopping_patience", 10)

        log_dir = cfg_paths.get("log_dir", "experiments/logs/")
        checkpoint_dir = cfg_paths.get("checkpoint_dir", "experiments/checkpoints/")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{self.experiment_id}.jsonl")

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0
        best_ckpt_dir = os.path.join(checkpoint_dir, self.experiment_id)
        best_pt_path = os.path.join(best_ckpt_dir, "best_model.pt")

        final_metrics: dict = {}

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()

            if self.scheduler is not None:
                self.scheduler.step()

            # 寫入日誌
            log_entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "f1_score": val_metrics["f1"],
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | acc=%.4f | f1=%.4f",
                epoch, epochs,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["accuracy"], val_metrics["f1"],
            )

            # 儲存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                no_improve_count = 0
                final_metrics = {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "best_epoch": best_epoch,
                }
                self.save_checkpoint(best_ckpt_dir, final_metrics)
                logger.info("儲存最佳模型（epoch=%d，val_loss=%.4f）", epoch, best_val_loss)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        "Early Stopping：連續 %d epoch 驗證損失未改善，於 epoch %d 停止",
                        patience, epoch,
                    )
                    break

        # 匯出 ONNX
        onnx_path = os.path.join(best_ckpt_dir, "model.onnx")
        try:
            self.export_onnx(best_pt_path, onnx_path)
        except Exception as e:
            logger.warning("ONNX 匯出失敗：%s", e)

        # 註冊實驗
        registry_path = self.config["paths"].get("registry", "experiments/model_registry.json")
        registry = ModelRegistry(registry_path)
        registry.register(
            experiment_id=self.experiment_id,
            model_path=best_pt_path,
            onnx_path=onnx_path,
            metrics=final_metrics,
            config={"architecture": self.config["model"].get("architecture", "")},
        )

        logger.info("訓練完成，最終指標：%s", final_metrics)
        return final_metrics

    # ------------------------------------------------------------------
    # save_checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metrics: dict) -> None:
        """
        儲存模型權重（best_model.pt）、設定檔（config.yaml）、指標（metrics.json）。
        path 為目錄路徑（以 experiment_id 命名）。
        """
        os.makedirs(path, exist_ok=True)

        # 儲存模型權重
        pt_path = os.path.join(path, "best_model.pt")
        torch.save(self.model.state_dict(), pt_path)

        # 複製設定檔
        config_dst = os.path.join(path, "config.yaml")
        shutil.copy2(self.config_path, config_dst)

        # 儲存指標
        metrics_path = os.path.join(path, "metrics.json")
        # confusion_matrix 是 ndarray，需轉換
        serializable_metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # export_onnx
    # ------------------------------------------------------------------

    def export_onnx(self, pt_path: str, onnx_path: str) -> None:
        """
        匯出 ONNX 並以 onnxruntime 驗證數值一致性（L∞ ≤ 1e-5）。
        不一致時記錄 warning，不中止流程。
        """
        assert self.model is not None

        cfg_data = self.config["data"]
        max_seq_len = cfg_data.get("max_seq_len", 150)
        input_dim = self.config["model"].get("input_dim", 675)

        # 載入最佳權重
        state_dict = torch.load(pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 建立虛擬輸入
        dummy_features = torch.randn(1, max_seq_len, input_dim, device=self.device)
        dummy_mask = torch.ones(1, max_seq_len, device=self.device)

        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

        torch.onnx.export(
            self.model,
            (dummy_features, dummy_mask),
            onnx_path,
            input_names=["features", "mask"],
            output_names=["output"],
            dynamic_axes={
                "features": {0: "batch_size", 1: "seq_len"},
                "mask": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info("ONNX 模型已匯出至 %s", onnx_path)

        # 驗證數值一致性
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feat_np = dummy_features.cpu().numpy()
            mask_np = dummy_mask.cpu().numpy()

            with torch.no_grad():
                pt_out = self.model(dummy_features, dummy_mask).cpu().numpy()

            ort_out = sess.run(
                None,
                {"features": feat_np, "mask": mask_np},
            )[0]

            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            if max_diff > 1e-5:
                logger.warning(
                    "ONNX 與 PyTorch 輸出不一致（L∞=%.2e > 1e-5），請檢查模型匯出設定",
                    max_diff,
                )
            else:
                logger.info("ONNX 數值一致性驗證通過（L∞=%.2e）", max_diff)

        except ImportError:
            logger.warning("onnxruntime 未安裝，跳過數值一致性驗證")
        except Exception as e:
            logger.warning("ONNX 數值一致性驗證失敗：%s", e)
