# 實作計畫：sign-language-ai-communication（ai_model/ 模組）

## 概覽

本計畫聚焦於資工A負責的 `ai_model/` 核心模組，依序實作資料前處理管線、雙架構分類模型、訓練管線、推論 API，並以 Hypothesis 屬性測試驗證 18 個正確性屬性。

## 任務

- [x] 1. 建立專案骨架與目錄結構
  - 建立 `ai_model/` 下的子目錄：`data_pipeline/`、`models/`、`trainer/`、`inference/`、`configs/`、`tests/`、`experiments/checkpoints/`、`experiments/logs/`
  - 在每個子目錄建立 `__init__.py`（含必要的公開匯出）
  - 建立 `ai_model/__init__.py`，匯出 `GesturePredictor` 作為對外唯一介面
  - 建立 `ai_model/configs/default_lstm.yaml` 與 `ai_model/configs/default_transformer.yaml`（依設計文件範例）
  - 建立 `ai_model/tests/conftest.py`，包含共用 fixtures（隨機種子設定、假 Keypoint_Matrix 生成器、假標註 CSV 生成器）
  - _需求：1.1, 5.1, 5.6_

- [x] 2. 實作 data_pipeline/preprocessor.py
  - [x] 2.1 實作 `KeypointPreprocessor` 類別
    - 實作 `_interpolate_missing()`：對 NaN/零值進行線性插值，回傳 `(filled_matrix, missing_ratio)`
    - 實作 `_normalize_to_wrist()`：以左手腕（index 0）與右手腕（index 21）為原點進行相對座標轉換
    - 實作 `_compute_velocity_acceleration()`：以 `np.diff` 計算速度與加速度，回傳 `[T, 75, 9]`
    - 實作 `fit_transform()`：串接上述步驟，輸出 `(feature_matrix [T, 675], metadata)`；缺失率 > 30% 時設定 `metadata["is_valid"] = False`
    - 實作 `serialize()` / `deserialize()`：以 `np.save` / `np.load` 處理 `.npy` 格式，格式不符時拋出 `ValueError`
    - _需求：1.1, 1.2, 1.3, 1.4, 1.5, 8.2, 8.3, 8.4_

  - [ ]* 2.2 撰寫 Property 1 屬性測試：前處理輸出形狀不變量
    - **Property 1：前處理輸出形狀不變量**
    - **Validates: Requirements 1.1, 1.5**

  - [ ]* 2.3 撰寫 Property 2 屬性測試：插值後無缺失值
    - **Property 2：插值後無缺失值**
    - **Validates: Requirements 1.2**

  - [ ]* 2.4 撰寫 Property 3 屬性測試：腕關節正規化後座標為零
    - **Property 3：腕關節正規化後座標為零**
    - **Validates: Requirements 1.3**

  - [ ]* 2.5 撰寫 Property 18 屬性測試：缺失率超限樣本被正確排除
    - **Property 18：缺失率超限樣本被正確排除**
    - **Validates: Requirements 1.4**

  - [ ]* 2.6 撰寫 Property 16 屬性測試：特徵矩陣序列化 Round-Trip
    - **Property 16：特徵矩陣序列化 Round-Trip**
    - **Validates: Requirements 8.2, 8.3**

- [x] 3. 實作 data_pipeline/dataset.py
  - [x] 3.1 實作 `GestureDataset` 類別
    - 實作 `__init__()`：讀取標註 CSV，驗證所有 `sample_id` 對應的 `.npy` 特徵檔案存在，缺失時拋出 `FileNotFoundError` 並列出缺失清單
    - 實作 `__getitem__()`：載入特徵矩陣，截斷或 padding 至 `max_seq_len`，回傳 `{features, label, mask, sample_id}`
    - 實作 `collate_fn()`：動態 padding 至批次內最長序列，生成對應 mask 張量
    - _需求：2.1, 2.2, 3.5_

  - [ ]* 3.2 撰寫 Property 6 屬性測試：Padding/Masking 正確性
    - **Property 6：Padding/Masking 正確性**
    - **Validates: Requirements 3.5**

- [x] 4. 實作 models/lstm_classifier.py
  - [x] 4.1 實作 `BiLSTMClassifier` 類別
    - 實作 `__init__()`：建立 Linear Projection、兩層 BiLSTM、Dropout、輸出 Linear 層
    - 實作 `forward()`：執行前向傳播，使用 mask 進行 Masked Mean Pooling，輸出 Softmax 機率分布 `[B, num_classes]`
    - _需求：3.1, 3.2, 3.5, 3.6_

  - [ ]* 4.2 撰寫 Property 5 屬性測試（LSTM 部分）：模型輸出為有效機率分布
    - **Property 5：模型輸出為有效機率分布（BiLSTMClassifier）**
    - **Validates: Requirements 3.2, 6.7**

  - [ ]* 4.3 撰寫 Property 7 屬性測試（LSTM 部分）：模型可用任意合法設定建立並前向傳播
    - **Property 7：模型可用任意合法設定建立並前向傳播（BiLSTMClassifier）**
    - **Validates: Requirements 3.1, 4.1**

  - [ ]* 4.4 撰寫單元測試：Dropout 訓練/推論模式切換
    - 驗證 `model.train()` 與 `model.eval()` 模式下 Dropout 行為差異
    - _需求：3.6_

- [x] 5. 實作 models/transformer_classifier.py
  - [x] 5.1 實作 `LearnablePositionalEncoding` 類別
    - 實作 `__init__()`：建立形狀為 `[1, max_seq_len, d_model]` 的可學習參數
    - 實作 `forward()`：將位置編碼加至輸入張量
    - _需求：4.4_

  - [x] 5.2 實作 `LightweightTransformerClassifier` 類別
    - 實作 `__init__()`：建立 Linear Projection、LearnablePositionalEncoding、TransformerEncoder（4 層）、CLS Token、輸出 Linear 層
    - 實作 `forward()`：處理 padding mask（True=需遮蔽），輸出 Softmax 機率分布 `[B, num_classes]`
    - _需求：4.1, 4.2, 4.3, 4.4_

  - [ ]* 5.3 撰寫 Property 5 屬性測試（Transformer 部分）：模型輸出為有效機率分布
    - **Property 5：模型輸出為有效機率分布（LightweightTransformerClassifier）**
    - **Validates: Requirements 3.2, 6.7**

  - [ ]* 5.4 撰寫 Property 7 屬性測試（Transformer 部分）：模型可用任意合法設定建立並前向傳播
    - **Property 7：模型可用任意合法設定建立並前向傳播（LightweightTransformerClassifier）**
    - **Validates: Requirements 3.1, 4.1**

  - [ ]* 5.5 撰寫 Property 8 屬性測試：Transformer 參數量上限
    - **Property 8：Transformer 參數量上限**
    - **Validates: Requirements 4.3**

  - [ ]* 5.6 撰寫單元測試：位置編碼輸出形狀驗證
    - 驗證 `LearnablePositionalEncoding` 輸出形狀與輸入一致
    - _需求：4.4_

- [x] 6. 檢查點 — 確認模型與資料管線基礎正確
  - 確保所有測試通過，如有問題請向使用者提問。

- [x] 7. 實作 trainer/registry.py
  - [x] 7.1 實作 `ModelRegistry` 類別
    - 實作 `__init__()`：若 `model_registry.json` 不存在則初始化空結構
    - 實作 `register()`：新增實驗記錄，確保 `experiment_id` 唯一
    - 實作 `get_best_model()`：依指定指標排序，回傳最佳記錄
    - 實作 `list_experiments()`：回傳所有實驗記錄列表
    - _需求：7.4, 7.5_

  - [ ]* 7.2 撰寫 Property 15 屬性測試：實驗記錄完整性
    - **Property 15：實驗記錄完整性**
    - **Validates: Requirements 7.4, 7.5**

- [x] 8. 實作 trainer/train.py
  - [x] 8.1 實作 `GestureTrainer` 類別核心
    - 實作 `__init__()`：從 YAML 載入設定，驗證必要欄位（缺少時拋出 `KeyError`）
    - 實作 `setup()`：依 `model.architecture` 設定初始化對應模型、`GestureDataset`、`DataLoader`、Adam 優化器、Cosine Annealing 排程器
    - 實作 `_train_epoch()`：單一 epoch 訓練，回傳 `{loss, accuracy, f1}`
    - 實作 `_validate()`：驗證集評估，回傳 `{loss, accuracy, f1, confusion_matrix}`
    - _需求：2.3, 2.4, 5.1, 5.2, 5.4_

  - [x] 8.2 實作 `GestureTrainer` 訓練迴圈與模型儲存
    - 實作 `train()`：完整訓練迴圈，含 Early Stopping（patience=10）、每 epoch 寫入日誌（`train_loss`, `val_loss`, `accuracy`, `f1_score`）、儲存最佳模型
    - 實作 `save_checkpoint()`：儲存 `.pt` 權重 + 超參數設定 + 訓練指標，目錄以 `experiment_id`（時間戳記）命名
    - 呼叫 `ModelRegistry.register()` 完成實驗記錄
    - _需求：5.2, 5.3, 5.5, 5.6, 7.1, 7.4, 7.5_

  - [x] 8.3 實作 `GestureTrainer.export_onnx()`
    - 使用 `torch.onnx.export` 匯出模型
    - 以 `onnxruntime` 載入並對隨機輸入驗證數值一致性（L∞ ≤ 1e-5），不一致時記錄警告日誌（不中止流程）
    - _需求：7.2, 7.3_

  - [ ]* 8.4 撰寫 Property 4 屬性測試：受試者獨立切分不變量
    - **Property 4：受試者獨立切分不變量**
    - **Validates: Requirements 2.3**

  - [ ]* 8.5 撰寫 Property 9 屬性測試：訓練日誌完整性
    - **Property 9：訓練日誌完整性**
    - **Validates: Requirements 5.2**

  - [ ]* 8.6 撰寫 Property 10 屬性測試：相同種子的訓練可重現性
    - **Property 10：相同種子的訓練可重現性**
    - **Validates: Requirements 5.6**

  - [ ]* 8.7 撰寫 Property 14 屬性測試：ONNX 與 PyTorch 數值一致性
    - **Property 14：ONNX 與 PyTorch 數值一致性**
    - **Validates: Requirements 7.3**

  - [ ]* 8.8 撰寫單元測試：YAML 超參數載入
    - 驗證合法 YAML 正確載入，缺少必要欄位時拋出 `KeyError`
    - _需求：5.1_

  - [ ]* 8.9 撰寫單元測試：Early Stopping 觸發條件
    - 模擬驗證損失連續 10 epoch 未改善，驗證訓練提前終止
    - _需求：5.3_

  - [ ]* 8.10 撰寫單元測試：Cosine Annealing 學習率曲線
    - 驗證學習率在訓練過程中依 Cosine Annealing 規律變化
    - _需求：5.4_

  - [ ]* 8.11 撰寫單元測試：混淆矩陣報告輸出
    - 驗證 `_validate()` 回傳的 `confusion_matrix` 形狀與類別數一致
    - _需求：5.5_

  - [ ]* 8.12 撰寫單元測試：模型儲存目錄結構
    - 驗證 `save_checkpoint()` 建立正確的目錄結構與必要檔案
    - _需求：7.1_

  - [ ]* 8.13 撰寫單元測試：ONNX 匯出檔案存在性
    - 驗證 `export_onnx()` 執行後 `.onnx` 檔案確實存在於指定路徑
    - _需求：7.2_

  - [ ]* 8.14 撰寫單元測試：CSV 標註檔載入格式驗證
    - 驗證合法 CSV 正確載入，缺少必要欄位時拋出 `ValueError`
    - _需求：2.1_

  - [ ]* 8.15 撰寫單元測試：類別平衡策略切換
    - 驗證 `balance_strategy: "smote"` 與 `"weighted_loss"` 均可正常初始化
    - _需求：2.4_

  - [ ]* 8.16 撰寫單元測試：資料集統計報告欄位完整性
    - 驗證統計報告包含各類別樣本數、總時長與品質分布欄位
    - _需求：2.5_

- [x] 9. 檢查點 — 確認訓練管線正確
  - 確保所有測試通過，如有問題請向使用者提問。

- [x] 10. 實作 inference/predictor.py
  - [x] 10.1 實作 `GesturePredictor` 類別
    - 實作 `__init__()`：依 `device="auto"` 自動偵測 CUDA，載入 `.pt` 模型權重與 YAML 設定，初始化 `KeypointPreprocessor`；模型檔案不存在時拋出 `FileNotFoundError`
    - 實作 `predict()`：驗證輸入 shape 為 `[T, 75, 3]`（不符時拋出含實際/預期 shape 的 `ValueError`），執行前處理與推論，回傳含 `gesture_label`、`confidence`、`inference_time_ms`、`timestamp`（ISO 8601）、`keypoint_hash`（MD5）的字典；推論輸出 NaN 時回傳 `confidence=0.0`
    - 實作 `predict_batch()`：迭代呼叫 `predict()`，回傳等長結果列表
    - 實作 `serialize_result()` / `deserialize_result()`：JSON 序列化與反序列化
    - _需求：6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 8.1, 8.5_

  - [ ]* 10.2 撰寫 Property 11 屬性測試：推論 API 回傳結構完整性
    - **Property 11：推論 API 回傳結構完整性**
    - **Validates: Requirements 6.1**

  - [ ]* 10.3 撰寫 Property 12 屬性測試：非法輸入拋出描述性例外
    - **Property 12：非法輸入拋出描述性例外**
    - **Validates: Requirements 6.5**

  - [ ]* 10.4 撰寫 Property 13 屬性測試：批次推論結果數量一致性
    - **Property 13：批次推論結果數量一致性**
    - **Validates: Requirements 6.6**

  - [ ]* 10.5 撰寫 Property 17 屬性測試：推論結果 JSON 序列化 Round-Trip
    - **Property 17：推論結果 JSON 序列化 Round-Trip**
    - **Validates: Requirements 8.1, 8.5**

- [x] 11. 整合串接與最終驗證
  - [x] 11.1 更新 `ai_model/__init__.py`，確認 `GesturePredictor` 可從套件根目錄直接匯入
    - 驗證 `from ai_model.inference.predictor import GesturePredictor` 呼叫路徑正確
    - _需求：6.1_

  - [x] 11.2 驗證端對端流程：`KeypointPreprocessor` → `GestureDataset` → 模型 → `GesturePredictor`
    - 以 `conftest.py` 中的假資料生成器建立完整流程的整合測試
    - _需求：1.1, 3.2, 6.1_

- [x] 12. 最終檢查點 — 確認所有測試通過
  - 確保所有測試通過，如有問題請向使用者提問。

## 備註

- 標記 `*` 的子任務為選填，可跳過以加速 MVP 開發
- 每個屬性測試須加上 `@settings(max_examples=100)` 並在註解中標明對應的 Property 編號
- 所有屬性測試放置於 `ai_model/tests/` 對應的測試檔案中
- 實作語言：Python（PyTorch 框架）
