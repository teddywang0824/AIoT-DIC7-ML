# CRISP-DM Linear Regression Simulator

🌟 **[點此進入線上即時展示 (Live Demo)](https://teddywang0824-aiot-dic7-ml-app-qfujza.streamlit.app/)** 🌟

此專案為一個基於 Streamlit 的互動式資料科學應用程式。其主要目標為引導使用者體驗標準的 **CRISP-DM (Cross-Industry Standard Process for Data Mining) 六大階段**，並透過模擬帶有擾動雜訊的數據特徵，示範基礎線性迴歸模型 (Linear Regression) 的運作與表現。

## ✨ 專案亮點與功能

- **互動式拉桿控制**：可透過拉桿自由定義樣本數量 ($n$)、隨機種子 (Seed) 與 系統雜訊的變異數 (Variance)。
- **自動生成的虛擬數據**：內部生成 $x \in [-100, 100]$ 的自變數，並搭配隨機生成的偏移斜率與截距組成完美的 $y = ax + b + noise$ 關係。
- **CRISP-DM 完整流程展示**：以六個視覺化區塊從頭到尾拆解資料專案：
  1. 商業理解 (Business Understanding)
  2. 資料理解 (Data Understanding)
  3. 資料準備 (Data Preparation - `StandardScaler` 與 `train_test_split`)
  4. 建立模型 (Modeling - 訓練 `LinearRegression`)
  5. 評估 (Evaluation - $MSE, RMSE, R^2$ 計算與 Plotly 互動式圖表)
  6. 部署 (Deployment - 單點即時預測與打包下載模型 `.pkl`)

## 🛠 執行環境與安裝

本專案強烈建議透過 [`uv`](https://github.com/astral-sh/uv) 進行極速虛擬環境建置管理。

### 1. 建立虛擬環境與安裝依賴套件
確保您已安裝 `uv` 以及 Python 3 之後，在此專案目錄下執行：
```bash
python -m uv venv
python -m uv pip install --python .\.venv streamlit numpy pandas scikit-learn plotly joblib
```

### 2. 啟動應用程式
完成環境安裝後，請先啟動虛擬環境再執行 Streamlit：
```bash
# Windows 用戶請輸入：
.\.venv\Scripts\activate

# 啟動 Streamlit
streamlit run app.py
```
執行後，您的瀏覽器即會跳出這個專案的全互動式操作頁面！
