# 需求文件

## 簡介

「聽見你的手勢」是一套基於邊緣運算的 AI 雙向無障礙溝通系統（SignVox-AI），旨在消弭聽障人士與服務人員之間的溝通斷層。系統僅需一般 Web Camera 與顯示螢幕，無需穿戴式裝置，即可在邊緣端完成即時手語辨識、語音合成、語音轉文字與語意精煉，實現零接觸、低門檻的雙向溝通。

本文件涵蓋整個系統的功能需求，並特別聚焦於資工A負責的核心 AI 模型訓練管線（`ai_model/` 模組）。

---

## 詞彙表

- **System**：SignVox-AI 整體系統
- **S2S_Module**：手語轉語音模組（Sign-to-Speech）
- **S2T_Module**：語音轉文字/手語模組（Speech-to-Text/Sign）
- **Landmark_Extractor**：MediaPipe 關鍵節點萃取器，負責從影像中提取手部與姿態特徵點
- **Signal_Processor**：訊號處理模組，負責卡爾曼濾波與滑動視窗切割（電機負責）
- **Gesture_Classifier**：手語分類模型，接收特徵矩陣並輸出手語語意標籤
- **Model_Trainer**：AI 模型訓練管線，負責資料集管理、模型訓練與超參數微調
- **Data_Pipeline**：資料前處理管線，負責將原始關節點座標轉換為訓練用特徵矩陣
- **TTS_Engine**：文字轉語音引擎（OpenAI TTS API）
- **STT_Engine**：語音轉文字引擎（Whisper）
- **LLM_Refiner**：大型語言模型語意精煉器，負責摘要與白話文改寫
- **Dashboard**：企業端營運戰情室（Streamlit，經管負責）
- **Frontend**：前端 GUI 介面（資工B負責）
- **Keypoint_Matrix**：由 Landmark_Extractor 輸出的關節點座標矩陣，形狀為 `[T, N_joints, 3]`
- **Gesture_Label**：手語動作的語意標籤字串
- **Inference_API**：模型推論 API，供 Frontend 呼叫
- **Confidence_Score**：模型輸出的預測信心分數（0.0 ~ 1.0）

---

## 需求

### 需求一：關節點特徵萃取與資料前處理

**使用者故事：** 身為資工A，我希望能從 MediaPipe 輸出的原始關節點座標建立標準化的特徵矩陣，以便後續模型訓練使用。

#### 驗收標準

1. THE Data_Pipeline SHALL 接收形狀為 `[T, N_joints, 3]` 的 Keypoint_Matrix 作為輸入，其中 T 為時間步數、N_joints 為關節點數量（手部 21 點 × 2 + 姿態 33 點）。
2. WHEN 輸入的 Keypoint_Matrix 包含缺失值（NaN 或零值），THE Data_Pipeline SHALL 以線性插值填補缺失幀，並記錄填補比例。
3. THE Data_Pipeline SHALL 對每個關節點座標進行正規化，以腕關節為原點進行相對座標轉換，消除絕對位置差異。
4. WHEN 填補後缺失幀比例超過 30%，THE Data_Pipeline SHALL 將該樣本標記為低品質並排除於訓練集之外。
5. THE Data_Pipeline SHALL 輸出形狀為 `[T, N_features]` 的特徵矩陣，其中 N_features 包含座標值、速度向量與加速度向量。
6. FOR ALL 有效輸入樣本，THE Data_Pipeline SHALL 在 100ms 內完成單一樣本的前處理。

---

### 需求二：資料集管理與標註

**使用者故事：** 身為資工A，我希望能有系統地管理手語資料集與標註，以確保訓練資料的品質與一致性。

#### 驗收標準

1. THE Model_Trainer SHALL 支援以 CSV 格式載入資料集標註檔，欄位包含 `sample_id`、`gesture_label`、`start_frame`、`end_frame`、`quality_flag`。
2. WHEN 載入資料集時，THE Model_Trainer SHALL 驗證所有 `sample_id` 對應的特徵檔案存在，IF 有缺失，THEN THE Model_Trainer SHALL 輸出缺失清單並中止載入。
3. THE Model_Trainer SHALL 依照 8:1:1 的比例自動切分訓練集、驗證集與測試集，並確保同一受試者的樣本不跨集合出現（受試者獨立切分）。
4. WHERE 資料集存在類別不平衡，THE Model_Trainer SHALL 支援過採樣（SMOTE）或加權損失函數兩種平衡策略，並允許使用者透過設定檔選擇。
5. THE Model_Trainer SHALL 產生資料集統計報告，包含各類別樣本數、總時長與品質分布。

---

### 需求三：LSTM 手語分類模型

**使用者故事：** 身為資工A，我希望能訓練一個基於 LSTM 的手語分類模型，以準確辨識連續手語動作的語意。

#### 驗收標準

1. THE Gesture_Classifier SHALL 實作雙層雙向 LSTM 架構，隱藏層維度可透過設定檔調整（預設 256）。
2. WHEN 接收形狀為 `[batch, T, N_features]` 的特徵矩陣，THE Gesture_Classifier SHALL 輸出各類別的 Confidence_Score 向量。
3. THE Gesture_Classifier SHALL 在驗證集上達到 Top-1 準確率 ≥ 85%。
4. THE Gesture_Classifier SHALL 在驗證集上達到巨觀平均 F1-Score ≥ 0.80。
5. WHEN 輸入序列長度不一致，THE Gesture_Classifier SHALL 透過 padding 與 masking 機制處理可變長度輸入。
6. THE Gesture_Classifier SHALL 支援 dropout 正則化（預設 0.3），以防止過擬合。

---

### 需求四：輕量化 Transformer 手語分類模型

**使用者故事：** 身為資工A，我希望能訓練一個輕量化 Transformer 模型作為 LSTM 的替代方案，以在邊緣裝置上取得更佳的推論效能。

#### 驗收標準

1. THE Gesture_Classifier SHALL 支援輕量化 Transformer 架構，包含多頭自注意力機制（Multi-Head Self-Attention），注意力頭數可透過設定檔調整（預設 4 頭）。
2. THE Gesture_Classifier SHALL 在邊緣裝置（無 GPU）上的單次推論延遲 ≤ 200ms。
3. THE Gesture_Classifier 的 Transformer 模型參數量 SHALL ≤ 5M 個參數。
4. WHEN 使用 Transformer 架構，THE Gesture_Classifier SHALL 加入位置編碼（Positional Encoding）以保留時序資訊。
5. THE Gesture_Classifier SHALL 在驗證集上達到與 LSTM 相當的準確率（Top-1 ≥ 85%）。

---

### 需求五：模型訓練流程與超參數微調

**使用者故事：** 身為資工A，我希望能有可重現的模型訓練流程與超參數管理機制，以系統化地提升模型效能。

#### 驗收標準

1. THE Model_Trainer SHALL 透過 YAML 設定檔管理所有超參數，包含學習率、批次大小、訓練輪數、模型架構選擇。
2. THE Model_Trainer SHALL 在每個 epoch 結束後記錄訓練損失、驗證損失、準確率與 F1-Score 至日誌檔案。
3. WHEN 驗證損失連續 10 個 epoch 未改善，THE Model_Trainer SHALL 觸發早停機制並儲存最佳模型權重。
4. THE Model_Trainer SHALL 支援學習率排程（Cosine Annealing），以改善訓練收斂。
5. THE Model_Trainer SHALL 在訓練結束後輸出混淆矩陣與各類別精確率/召回率報告。
6. FOR ALL 訓練實驗，THE Model_Trainer SHALL 記錄完整的超參數設定與隨機種子，確保實驗可重現。

---

### 需求六：模型推論 API

**使用者故事：** 身為資工A，我希望能提供標準化的推論 API，讓資工B的前端模組能直接呼叫模型進行即時手語辨識。

#### 驗收標準

1. THE Inference_API SHALL 提供 `predict(keypoint_matrix: np.ndarray) -> dict` 介面，回傳包含 `gesture_label`、`confidence` 與 `inference_time_ms` 的字典。
2. WHEN 載入模型權重，THE Inference_API SHALL 在 3 秒內完成模型初始化。
3. THE Inference_API SHALL 在配備 GPU 的環境下，單次推論延遲 ≤ 50ms。
4. THE Inference_API SHALL 在僅有 CPU 的環境下，單次推論延遲 ≤ 200ms。
5. IF 輸入的 Keypoint_Matrix 形狀不符合預期，THEN THE Inference_API SHALL 回傳包含錯誤描述的例外，而非靜默失敗。
6. THE Inference_API SHALL 支援批次推論（batch inference），允許一次輸入多個樣本。
7. FOR ALL 推論結果，THE Inference_API 的 Confidence_Score SHALL 介於 0.0 至 1.0 之間（包含端點）。

---

### 需求七：模型序列化與版本管理

**使用者故事：** 身為資工A，我希望能有系統地儲存與管理不同版本的模型，以便追蹤實驗進度與回滾至最佳版本。

#### 驗收標準

1. THE Model_Trainer SHALL 以 PyTorch `.pt` 格式儲存模型權重，並同時儲存對應的超參數設定檔與訓練指標摘要。
2. THE Model_Trainer SHALL 支援將模型匯出為 ONNX 格式，以利跨平台部署。
3. WHEN 匯出 ONNX 模型後，THE Model_Trainer SHALL 驗證 ONNX 模型的推論結果與 PyTorch 模型的推論結果差異 ≤ 1e-5（數值一致性）。
4. THE Model_Trainer SHALL 為每次訓練實驗產生唯一的實驗 ID（基於時間戳記），並以此 ID 組織模型檔案目錄。
5. FOR ALL 儲存的模型，THE Model_Trainer SHALL 維護一份 `model_registry.json`，記錄各版本的實驗 ID、訓練日期、驗證準確率與模型路徑。

---

### 需求八：手語辨識解析器與序列化（Parser/Serializer）

**使用者故事：** 身為資工A，我希望能將手語辨識結果序列化為標準格式，並能從儲存格式還原，以確保資料交換的正確性。

#### 驗收標準

1. WHEN 產生手語辨識結果，THE Gesture_Classifier SHALL 將結果序列化為 JSON 格式，包含 `gesture_label`、`confidence`、`timestamp` 與 `keypoint_hash`。
2. THE Data_Pipeline SHALL 支援將特徵矩陣序列化為 `.npy` 格式，並能從 `.npy` 格式還原為等價的特徵矩陣。
3. FOR ALL 有效的特徵矩陣，THE Data_Pipeline 序列化後再反序列化 SHALL 產生數值相等的矩陣（往返特性，round-trip property）。
4. WHEN 反序列化格式不符合預期的 `.npy` 檔案，THE Data_Pipeline SHALL 回傳描述性錯誤，而非靜默失敗。
5. THE Model_Trainer SHALL 支援從 JSON 格式的標註檔解析手語標籤，並能將解析結果重新序列化為等價的 JSON（往返特性）。

---

### 需求九：S2T 模組整合需求（語音轉文字）

**使用者故事：** 身為系統整合者，我希望 STT 與 LLM 精煉功能能正確運作，以完成雙向溝通的另一方向。

#### 驗收標準

1. WHEN 服務人員對麥克風說話，THE STT_Engine SHALL 在語音結束後 2 秒內輸出對應的文字轉錄結果。
2. THE LLM_Refiner SHALL 接收 STT_Engine 的輸出，並將其改寫為適合聽障人士閱讀的白話文字幕。
3. WHEN STT_Engine 無法辨識語音內容，THE STT_Engine SHALL 回傳空字串並記錄辨識失敗事件，而非拋出例外中斷系統。
4. THE System SHALL 在顯示器上以動態字幕形式呈現 LLM_Refiner 的輸出結果。

---

### 需求十：系統效能與邊緣運算需求

**使用者故事：** 身為系統部署者，我希望整個系統能在邊緣裝置上流暢運行，無需依賴雲端運算資源。

#### 驗收標準

1. THE System SHALL 在配備一般消費級 GPU（如 NVIDIA RTX 3060 或同等級）的邊緣裝置上，以端對端延遲 ≤ 500ms 完成手語辨識並輸出語音。
2. WHILE 系統運行中，THE System SHALL 維持 CPU 使用率 ≤ 80%，以確保系統穩定性。
3. THE System SHALL 在不依賴網際網路連線的情況下完成手語辨識與語音合成的核心功能（離線模式）。
4. WHERE 網際網路連線可用，THE System SHALL 支援呼叫 OpenAI TTS API 以獲得更自然的語音合成品質。
5. THE System SHALL 支援在 Windows 10/11 與 Ubuntu 20.04 以上的作業系統上運行。
