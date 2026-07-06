import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil
from matplotlib.font_manager import fontManager

# ===================== 中文字体设置 =====================
def set_chinese_font():
    font_candidates = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    available = [f.name for f in fontManager.ttflist]
    chosen = None
    for font in font_candidates:
        if font in available:
            chosen = font
            break
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ===================== 页面配置 =====================
st.set_page_config(page_title="合肥PM2.5预测", layout="wide")
st.title("🌿 合肥市 PM2.5 月度预测系统")
st.markdown("基于 XGBoost 模型，自动预测下月 PM2.5。支持手动添加最新数据并即时更新预测。")

# ===================== 文件路径 =====================
ORIGINAL_CSV = 'hefei_air_quality.csv'
FACTORY_BACKUP = 'hefei_air_quality_factory_backup.csv'   # 出厂备份，永不修改

# 第一次运行时创建出厂备份（如果不存在）
if not os.path.exists(FACTORY_BACKUP):
    shutil.copy(ORIGINAL_CSV, FACTORY_BACKUP)

# ===================== 模型加载（缓存） =====================
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    feat_cols = joblib.load('feat_cols.pkl')
    return model, feat_cols

model, feat_cols = load_model()

# ===================== 数据加载 =====================
def load_data(filepath=ORIGINAL_CSV):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, TypeError):
            continue
    if df is None:
        st.error("无法读取数据文件，请检查编码格式")
        st.stop()
    df['日期'] = pd.to_datetime(df['月份'].astype(str), format='%Y-%m', errors='coerce')
    df.dropna(subset=['日期'], inplace=True)
    df.set_index('日期', inplace=True)
    df.sort_index(inplace=True)
    return df

# 会话独立数据框（每次启动从当前文件加载）
if 'df' not in st.session_state:
    st.session_state.df = load_data(ORIGINAL_CSV)

# ===================== 辅助函数 =====================
def get_lag_pm25(df, months_back):
    last_date = df.index[-1]
    target_date = last_date - pd.DateOffset(months=months_back)
    if target_date in df.index:
        return df.loc[target_date, 'PM2.5']
    else:
        return df['PM2.5'].iloc[-months_back]

def make_prediction(df):
    last_date = df.index[-1]
    if last_date.month == 12:
        predict_year = last_date.year + 1
        predict_month = 1
    else:
        predict_year = last_date.year
        predict_month = last_date.month + 1

    last_row = df.iloc[-1]

    features = {}
    features['year'] = predict_year
    features['month'] = predict_month
    features['quarter'] = (predict_month - 1) // 3 + 1
    features['month_sin'] = np.sin(2 * np.pi * predict_month / 12)
    features['month_cos'] = np.cos(2 * np.pi * predict_month / 12)

    features['pm25_lag_1'] = last_row['PM2.5']
    features['pm25_lag_2'] = get_lag_pm25(df, 1)
    features['pm25_lag_3'] = get_lag_pm25(df, 2)
    features['pm25_lag_6'] = get_lag_pm25(df, 5)
    features['pm25_lag_12'] = get_lag_pm25(df, 11)

    vals = [get_lag_pm25(df, 2), get_lag_pm25(df, 1), last_row['PM2.5']]
    features['pm25_roll3_mean'] = np.mean(vals)
    features['pm25_roll3_std'] = np.std(vals)

    for col in ['PM10', 'NO2', 'CO', 'SO2', 'O3']:
        features[f'{col}_lag1'] = last_row[col]

    X_pred = pd.DataFrame([features])[feat_cols]
    pred = model.predict(X_pred)[0]
    return pred, predict_year, predict_month

# ===================== 侧边栏：操作区 =====================
with st.sidebar:
    st.header("📝 手动添加最新数据")
    last_date = st.session_state.df.index[-1]
    default_year = last_date.year if last_date.month < 12 else last_date.year + 1
    default_month = last_date.month + 1 if last_date.month < 12 else 1
    today = datetime.today()
    if today.year > default_year or (today.year == default_year and today.month >= default_month):
        default_year, default_month = today.year, today.month

    upd_year = st.number_input("年份", min_value=2013, max_value=2030, value=default_year)
    upd_month = st.number_input("月份", min_value=1, max_value=12, value=default_month)

    col_a, col_b = st.columns(2)
    with col_a:
        aqi_val = st.number_input("AQI", min_value=0.0, value=50.0, step=1.0)
        pm25_val = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=30.0, step=0.1)
        pm10_val = st.number_input("PM10 (μg/m³)", min_value=0.0, value=50.0, step=0.1)
        no2_val = st.number_input("NO2 (μg/m³)", min_value=0.0, value=20.0, step=0.1)
    with col_b:
        co_val = st.number_input("CO (mg/m³)", min_value=0.0, value=0.6, step=0.01, format="%.3f")
        so2_val = st.number_input("SO2 (μg/m³)", min_value=0.0, value=6.0, step=0.1)
        o3_val = st.number_input("O3 (μg/m³)", min_value=0.0, value=100.0, step=0.1)

    # ===== 提交按钮：仅更新内存 =====
    if st.button("提交并更新预测"):
        new_row = {
            '月份': f"{upd_year}-{upd_month:02d}",
            'AQI': aqi_val, 'PM2.5': pm25_val, 'PM10': pm10_val,
            'NO2': no2_val, 'CO': co_val, 'SO2': so2_val, 'O3': o3_val,
            '范围': '', '质量等级': ''
        }
        df_session = st.session_state.df.copy()
        date_str = f"{upd_year}-{upd_month:02d}"
        mask = df_session['月份'] == date_str
        if mask.any():
            df_session.loc[mask, list(new_row.keys())] = list(new_row.values())
            st.success(f"已更新 {date_str} 的数据（仅会话）")
        else:
            df_new = pd.concat([df_session, pd.DataFrame([new_row])], ignore_index=True)
            df_session = df_new
            st.success(f"已临时添加 {date_str} 的数据")
        df_session['日期'] = pd.to_datetime(df_session['月份'].astype(str), format='%Y-%m')
        df_session.set_index('日期', inplace=True)
        df_session.sort_index(inplace=True)
        st.session_state.df = df_session
        st.rerun()

    st.divider()
    st.header("💾 数据管理")

    # 三个功能按钮
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("💿 保存到本地"):
            st.session_state.df.to_csv(ORIGINAL_CSV, index=False, encoding='utf-8-sig')
            st.success("当前会话数据已保存到本地文件。")

    with col_btn2:
        if st.button("🔄 重置会话"):
            st.session_state.df = load_data(ORIGINAL_CSV)
            st.success("已从本地文件重新加载数据。")
            st.rerun()

    with col_btn3:
        if st.button("⚙️ 恢复出厂设置"):
            shutil.copy(FACTORY_BACKUP, ORIGINAL_CSV)
            st.session_state.df = load_data(ORIGINAL_CSV)
            st.success("已恢复至出厂初始数据。")
            st.rerun()

# ===================== 主界面 =====================
df = st.session_state.df

st.subheader("📈 合肥市 PM2.5 历史趋势")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df.index, df['PM2.5'], color='darkgreen', linewidth=1)
ax.axvline(x=df.index[-1], color='red', linestyle='--', label=f'最新数据 {df.index[-1].strftime("%Y-%m")}')
ax.set_ylabel('PM2.5 (μg/m³)')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("🔮 下月预测")
pred, p_year, p_month = make_prediction(df)
col1, col2, col3 = st.columns(3)
col1.metric("📅 预测月份", f"{p_year}年{p_month}月")
col2.metric("🌁 预测 PM2.5", f"{pred:.1f} μg/m³")
if pred < 35:
    level, color = "优", "green"
elif pred < 75:
    level, color = "良", "orange"
else:
    level, color = "轻度污染及以上", "red"
col3.markdown(f"<h4 style='color:{color};'>空气质量等级：{level}</h4>", unsafe_allow_html=True)

st.subheader("🔍 特征重要性 Top10")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh(range(len(indices)), importances[indices])
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([feat_cols[i] for i in indices])
    ax2.set_xlabel('Importance')
    ax2.set_title('XGBoost 特征重要性')
    st.pyplot(fig2)

st.markdown("---")
st.caption("合肥市 PM2.5 月度预测系统 | 课程设计项目 | 数据来源：真气网 2013-2026")