# 聽見你的手勢 — SignVox-AI

> 基於邊緣運算之 AI 雙向無障礙溝通系統

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 一、專題簡介

聽障人士在公家機關洽公、銀行臨櫃辦理業務或大型醫院就診時，常面臨「溝通斷層」。實體手語翻譯員極度匱乏且預約成本高昂；手語翻譯手套等硬體設備造價昂貴、穿戴不便，難以在公共場所大規模普及。

**SignVox-AI** 是一套「純軟體」的邊緣運算智動化資訊系統，主打「零接觸、低門檻」：

- 僅需一般 **Web Camera** 與顯示螢幕
- 無需任何穿戴式裝置或特殊硬體
- 在邊緣端（Edge）完成全部 AI 運算，保護隱私
- 實現聽障人士與服務人員的**即時雙向溝通**

落實聯合國永續發展目標：**SDG 10（減少不平等）** 與 **SDG 3（健康與福祉）**。

---

## 二、系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                        SignVox-AI                           │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  模組一：S2S          │    │  模組二：S2T              │  │
│  │  手語 → 語音          │    │  語音 → 文字/字幕         │  │
│  │                      │    │                          │  │
│  │  Camera              │    │  Microphone              │  │
│  │    ↓                 │    │    ↓                     │  │
│  │  MediaPipe           │    │  Whisper STT             │  │
│  │  (關鍵節點萃取)       │    │    ↓                     │  │
│  │    ↓                 │    │  LLM Refiner             │  │
│  │  Signal Processor    │    │  (白話文改寫)             │  │
│  │  (卡爾曼濾波/視窗)    │    │    ↓                     │  │
│  │    ↓                 │    │  動態字幕顯示             │  │
│  │  LSTM/Transformer    │    │                          │  │
│  │  (手語分類)           │    └──────────────────────────┘  │
│  │    ↓                 │                                   │
│  │  TTS Engine          │    ┌──────────────────────────┐  │
│  │  (語音合成)           │    │  Dashboard               │  │
│  └──────────────────────┘    │  (營運戰情室 / ROI)       │  │
│                              └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 模組一：手語轉語音（Sign-to-Speech, S2S）

| 步驟 | 元件 | 說明 |
|------|------|------|
| 1 | MediaPipe | 萃取手部 21 個關鍵節點 + 身體姿態特徵 |
| 2 | Signal Processor | 卡爾曼濾波平滑化 + 滑動視窗切割有效手勢區間 |
| 3 | LSTM / Transformer | 時序特徵分析，輸出手語語意標籤 |
| 4 | TTS Engine | 文字轉自然語音（OpenAI TTS API / 離線引擎） |

### 模組二：語音轉文字（Speech-to-Text/Sign, S2T）

| 步驟 | 元件 | 說明 |
|------|------|------|
| 1 | Whisper STT | 即時語音辨識，語音結束後 2 秒內輸出文字 |
| 2 | LLM Refiner | 摘要與白話文改寫，適合聽障人士閱讀 |
| 3 | Frontend GUI | 高對比動態字幕顯示 |

---

## 三、核心技術

### 無接觸式邊緣視覺辨識
全面捨棄昂貴感測硬體，採用 MediaPipe 輕量化人體特徵點抓取技術，將運算資源消耗降至最低，可在一般桌上型電腦或邊緣運算盒（Jetson Nano）上流暢運行。

### 動態時序分析
有別於傳統靜態圖片分類，採用時間序列分析（LSTM / Transformer），精準捕捉手語中的「方向性」、「速度」與「連續動作變化」，讓翻譯結果從「單字拼湊」進化為「流暢的自然手語語意」。

### 訊號清洗與雜訊過濾
- **卡爾曼濾波（Kalman Filter）**：平滑化 3D 座標點，解決攝影機捕捉時的物理抖動
- **資料滑動視窗（Data Windowing）**：精準切割有效手語動作區間，大幅提升推論準確率與強健性

---

## 四、商業模式

### B2B / B2G 軟體授權
| 目標客群 | 說明 |
|----------|------|
| 政府機關 | 戶政事務所、區公所 |
| 醫療院所 | 批價掛號櫃台、急診 |
| 金融機構 | 銀行臨櫃服務 |

**收費機制**：買斷制 + 年度維護合約，或基於使用流量（翻譯時數）的訂閱制。

### 企業端營運戰情室
- 即時視覺化每日使用次數、總翻譯時數
- 自動換算「節省之手語翻譯費（ROI）」
- 生成 ESG 永續報告書量化實績（社會責任 S 指標）

---

## 五、專案結構

```
SignVox-AI/
├── ai_model/                   # 資工A：核心 AI 模型訓練管線
│   ├── data_pipeline/          # 資料前處理（特徵萃取、正規化）
│   ├── models/                 # LSTM / Transformer 模型定義
│   ├── trainer/                # 訓練流程、超參數管理
│   ├── inference/              # 推論 API（供前端呼叫）
│   ├── configs/                # YAML 超參數設定檔
│   └── experiments/            # 實驗結果與模型版本管理
│
├── signal_processing/          # 電機：訊號處理與時序資料清洗
│   ├── kalman_filter.py        # 卡爾曼濾波器
│   └── data_windowing.py       # 滑動視窗切割
│
├── frontend/                   # 資工B：前端 GUI 介面
│   ├── main_window.py          # 主視窗（霧面玻璃風格）
│   ├── camera_feed.py          # 即時影像串流
│   └── subtitle_display.py     # 動態字幕顯示
│
├── dashboard/                  # 經管：企業端營運戰情室
│   ├── app.py                  # Streamlit 主應用
│   ├── database.py             # SQLite 資料庫操作
│   └── roi_calculator.py       # ROI 計算邏輯
│
├── docs/                       # 文件
├── requirements.txt            # Python 依賴套件
└── README.md
```

---

## 六、團隊分工

| 角色 | 負責模組 | 核心任務 |
|------|----------|----------|
| **資工A** | `ai_model/` | LSTM/Transformer 模型訓練、資料管線、推論 API |
| **資工B** | `frontend/` | GUI 介面開發、MediaPipe 整合、STT/LLM API 串接 |
| **電機** | `signal_processing/` | 卡爾曼濾波、滑動視窗、訊號清洗 |
| **經管** | `dashboard/` | Streamlit 儀表板、SQLite、ROI 計算、TTS API |

---

## 七、效能指標

| 指標 | 目標值 |
|------|--------|
| 端對端延遲（含 GPU） | ≤ 500ms |
| 模型推論延遲（GPU） | ≤ 50ms |
| 模型推論延遲（CPU） | ≤ 200ms |
| 手語辨識 Top-1 準確率 | ≥ 85% |
| 手語辨識 F1-Score（巨觀平均） | ≥ 0.80 |
| Transformer 模型參數量 | ≤ 5M |
| STT 轉錄延遲 | ≤ 2 秒 |
| CPU 使用率上限 | ≤ 80% |

---

## 八、環境需求

### 硬體
- CPU：Intel Core i5 / AMD Ryzen 5 以上
- GPU：NVIDIA RTX 3060 或同等級（建議，非必要）
- RAM：16GB 以上
- 攝影機：一般 USB Web Camera（720p 以上）

### 軟體
- OS：Windows 10/11 或 Ubuntu 20.04+
- Python：3.10+
- CUDA：11.8+（使用 GPU 時）

---

## 九、安裝與執行

```bash
# 1. 複製專案
git clone https://github.com/your-org/signvox-ai.git
cd signvox-ai

# 2. 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 執行主系統（資工B 完成後）
python frontend/main_window.py

# 5. 執行儀表板（經管完成後）
streamlit run dashboard/app.py
```

---

## 十、依賴套件

```
mediapipe          # 關鍵節點萃取
opencv-python      # 影像處理
torch              # 深度學習框架
torchvision        # 視覺工具
onnx               # 模型跨平台匯出
onnxruntime        # ONNX 推論引擎
numpy              # 數值運算
pandas             # 資料處理
scikit-learn       # 資料切分、評估指標
imbalanced-learn   # SMOTE 過採樣
pyyaml             # YAML 設定檔解析
streamlit          # 儀表板框架
PyQt5              # GUI 框架
openai             # TTS / LLM API
whisper            # 語音辨識
```

---

## 十一、授權

MIT License © 2025 SignVox-AI Team
