"""
ModelRegistry：維護 model_registry.json，追蹤所有實驗版本。
"""

import json
import os
from datetime import datetime


class ModelRegistry:
    """維護 model_registry.json，追蹤所有實驗版本"""

    def __init__(self, registry_path: str = "experiments/model_registry.json"):
        self.registry_path = registry_path
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"experiments": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(
        self,
        experiment_id: str,
        model_path: str,
        onnx_path: str,
        metrics: dict,
        config: dict,
    ) -> None:
        """新增或覆蓋實驗記錄，並立即寫回 JSON 檔案。"""
        record = {
            "experiment_id": experiment_id,
            "architecture": config.get("architecture", ""),
            "training_date": datetime.now().isoformat(),
            "model_path": model_path,
            "onnx_path": onnx_path,
        }
        record.update(metrics)

        # 若 experiment_id 已存在則覆蓋
        experiments = self._data["experiments"]
        for i, exp in enumerate(experiments):
            if exp["experiment_id"] == experiment_id:
                experiments[i] = record
                self._save()
                return

        experiments.append(record)
        self._save()

    def get_best_model(self, metric: str = "val_accuracy") -> dict | None:
        """依指定指標排序，回傳最佳記錄；無記錄時回傳 None。"""
        experiments = self._data["experiments"]
        valid = [e for e in experiments if metric in e]
        if not valid:
            return None
        return max(valid, key=lambda e: e[metric])

    def list_experiments(self) -> list[dict]:
        """回傳所有實驗記錄，依 training_date 降序排列。"""
        return sorted(
            self._data["experiments"],
            key=lambda e: e.get("training_date", ""),
            reverse=True,
        )
