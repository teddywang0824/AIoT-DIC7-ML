import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def generate_data(n, seed, noise_var):
    """根據指定的樣本數、隨機種子與變異數，生成模擬的線性數據。"""
    np.random.seed(seed)
    
    x = np.random.uniform(-100, 100, n)
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-50, 50)
    
    noise_mean = np.random.uniform(-10, 10)
    noise_std = np.sqrt(noise_var)
    noise = np.random.normal(noise_mean, noise_std, n)
    
    y = a * x + b + noise
    
    df = pd.DataFrame({'x': x, 'y': y})
    params = {'a': a, 'b': b, 'noise_mean': noise_mean, 'noise_var': noise_var}
    
    return df, params


st.set_page_config(page_title="CRISP-DM 線性迴歸模型", layout="wide")
st.title("📊 Data Generator & CRISP-DM 流程展示")

# 側邊欄設定
st.sidebar.header("環境參數設定")
n = st.sidebar.slider("n (樣本數)", min_value=100, max_value=1000, value=500, step=1)
seed = st.sidebar.slider("隨機種子 (Seed)", min_value=0, max_value=1000, value=42, step=1)
noise_var = st.sidebar.slider("誤差變異數 (Variance)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

# 使用 session_state 記錄資料是否已生成，避免按鈕重置導致流程消失
if "generated" not in st.session_state:
    st.session_state["generated"] = False

if st.sidebar.button("產生數據 (Generate Data)"):
    st.session_state["generated"] = True

if st.session_state["generated"]:
    with st.spinner("正在生成資料並執行分析流程..."):
        df, params = generate_data(n, seed, noise_var)
        
    # --- 階段 1：商業理解 (Business Understanding) ---
    st.header("1. 商業理解 (Business Understanding)")
    st.write("此階段確立專案目標：我們希望從帶有雜訊的隨機資料中，透過建立機器學習模型來找出潛在的真實線性關係 $y = ax + b$。")
    
    # --- 階段 2：資料理解 (Data Understanding) ---
    st.header("2. 資料理解 (Data Understanding)")
    st.write("探索與分析原始資料的結構與特徵：下表為系統生成的模擬資料集（節錄前 5 筆），包含特徵 $x$ 與目標變數 $y$。")
    st.dataframe(df.head())
    st.caption(f"**資料總筆數：** {len(df)}")
    
    # --- 階段 3：資料準備 (Data Preparation) ---
    st.header("3. 資料準備 (Data Preparation)")
    st.write("資料清洗與特徵工程：將資料依 `8:2` 切分為訓練集與測試集，並使用 `StandardScaler` 進行特徵標準化。")
    X = df[['x']]
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col_dp1, col_dp2 = st.columns(2)
    col_dp1.info(f"**訓練集大小：** {X_train.shape[0]} 筆")
    col_dp2.info(f"**測試集大小：** {X_test.shape[0]} 筆")
    
    # --- 階段 4：建立模型 (Modeling) ---
    st.header("4. 建立模型 (Modeling)")
    st.write("在此階段，我們選擇了 `LinearRegression` 演算法，並使用標準化後的訓練集進行模型擬合。")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    st.success("✔️ 線性迴歸模型訓練完成！")
    
    # --- 階段 5：評估 (Evaluation) ---
    st.header("5. 模型評估 (Evaluation)")
    st.write("評估模型在未知測試集上的表現，並比較模型學習到的參數與真實參數的差異。")
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("效能指標 (基於測試集)")
    # 使用 st.columns 讓指標並排顯示
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("均方誤差 ($MSE$)", f"{mse:.4f}")
    m_col2.metric("均方根誤差 ($RMSE$)", f"{rmse:.4f}")
    m_col3.metric("決定係數 ($R^2$)", f"{r2:.4f}")
    
    st.subheader("參數比較 (真實 vs 預測)")
    # 計算還原標準化後的線性參數：y = (a_scaled / s_x) * x - a_scaled * u_x / s_x + b_scaled
    s_x = scaler.scale_[0]
    u_x = scaler.mean_[0]
    learned_a = model.coef_[0] / s_x
    learned_b = model.intercept_ - (model.coef_[0] * u_x / s_x)
    
    comp_df = pd.DataFrame({
        "參數": ["a (斜率)", "b (截距)"],
        "真實數值": [params['a'], params['b']],
        "模型學習數值": [learned_a, learned_b],
        "絕對誤差": [abs(params['a'] - learned_a), abs(params['b'] - learned_b)]
    })
    st.table(comp_df)
    
    st.subheader("視覺化：迴歸線與資料點分佈")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['x'], y=y_test, mode='markers', name='測試資料', marker=dict(color='orange', opacity=0.8)))
    fig.add_trace(go.Scatter(x=X_train['x'], y=y_train, mode='markers', name='訓練資料', marker=dict(color='blue', opacity=0.3)))
    
    x_range = np.linspace(X['x'].min(), X['x'].max(), 100)
    y_true_line = params['a'] * x_range + params['b']
    fig.add_trace(go.Scatter(x=x_range, y=y_true_line, mode='lines', name='真實線性關係', line=dict(color='green', dash='dash')))
    
    y_learned_line = learned_a * x_range + learned_b
    fig.add_trace(go.Scatter(x=x_range, y=y_learned_line, mode='lines', name='模型預測迴歸線', line=dict(color='purple')))
    
    fig.update_layout(title="線性迴歸線 vs 真實關係線", xaxis_title="特徵 ($x$)", yaxis_title="目標變數 ($y$)")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 階段 6：部署 (Deployment) ---
    st.header("6. 部署 (Deployment)")
    st.write("將訓練完成的最終模型投入實際應用環境。可進行單點預測，並將模型匯出成檔案 (`.pkl`) 供其他系統使用。")
    
    st.subheader("📝 6.1 即時單筆預測")
    x_input = st.number_input("請輸入特徵 $x$ 的數值，進行即時預測：", value=0.0, step=1.0)
    # 使用 DataFrame 以避免 feature names 警告
    x_input_df = pd.DataFrame({'x': [x_input]})
    x_input_scaled = scaler.transform(x_input_df)
    y_pred_single = model.predict(x_input_scaled)[0]
    st.info(f"當 $x = {x_input:.2f}$ 時，模型的預測結果 $\hat{{y}} = $ **{y_pred_single:.4f}**")
    
    st.subheader("📦 6.2 匯出模型與下載")
    file_path = "crisp_dm_linear_regression.pkl"
    if st.button("將模型與縮放器儲存為 .pkl"):
        pipeline_data = {"model": model, "scaler": scaler}
        joblib.dump(pipeline_data, file_path)
        st.success(f"✔️ 線性模型與參數已成功封裝，並儲存至 `{file_path}`！")
        
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 點擊下載模型檔案 (crisp_dm_linear_regression.pkl)",
                data=f,
                file_name=file_path,
                mime="application/octet-stream"
            )
else:
    st.info("👈 請至左側側邊欄設定參數，並點擊「產生數據 (Generate Data)」按鈕以開始執行 CRISP-DM 流程。")
