import streamlit as st
from prophet import Prophet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 设置网页信息
st.set_page_config(page_title="黄金价格预测App", layout="centered")
st.title("黄金价格预测（加入全球股市影响）")
st.markdown("AI 模型使用 Facebook Prophet，结合全球股市影响进行预测")

# 用户选择预测天数
period = st.selectbox("选择预测天数", [7, 14, 30])

# 获取数据
@st.cache_data
def get_all_data():
    # 黄金 ETF（GLD）
    gold = yf.download("GLD", start="2015-01-01")[['Close']].rename(columns={"Close": "gold"})
    # S&P500
    sp500 = yf.download("^GSPC", start="2015-01-01")[['Close']].rename(columns={"Close": "sp500"})
    # 日经225
    nikkei = yf.download("^N225", start="2015-01-01")[['Close']].rename(columns={"Close": "nikkei"})
    # DAX
    dax = yf.download("^GDAXI", start="2015-01-01")[['Close']].rename(columns={"Close": "dax"})

    # 合并
    df = gold.join([sp500, nikkei, dax], how='inner')
    df = df.reset_index().rename(columns={"Date": "ds"})
    df = df.rename(columns={"gold": "y"})  # Prophet 需要目标列名为 y
    return df

df = get_all_data()

# 显示历史黄金 vs 股市走势
st.subheader("黄金 & 全球股市历史趋势")
st.line_chart(df.set_index("ds")[["y", "sp500", "nikkei", "dax"]])

# 建立 Prophet 模型 + 外部变量
model = Prophet(daily_seasonality=True)
model.add_regressor("sp500")
model.add_regressor("nikkei")
model.add_regressor("dax")

# 拟合模型
model.fit(df)

# 未来数据框
future = model.make_future_dataframe(periods=period)

# 需要未来股市数据（我们用最近值填充）
latest_sp500 = df["sp500"].iloc[-1]
latest_nikkei = df["nikkei"].iloc[-1]
latest_dax = df["dax"].iloc[-1]

future["sp500"] = df["sp500"].tolist() + [latest_sp500]*period
future["nikkei"] = df["nikkei"].tolist() + [latest_nikkei]*period
future["dax"] = df["dax"].tolist() + [latest_dax]*period

# 预测
forecast = model.predict(future)

# 显示预测图
st.subheader("预测图表")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# 显示预测表格
forecast_tail = forecast[['ds', 'yhat']].tail(period).rename(columns={"ds": "日期", "yhat": "预测金价"})
forecast_tail["预测金价"] = forecast_tail["预测金价"].round(2)
st.subheader(f"未来 {period} 天预测数据（单位：美元）")
st.dataframe(forecast_tail)

# 中文总结
start_price = forecast_tail["预测金价"].iloc[0]
end_price = forecast_tail["预测金价"].iloc[-1]
change = end_price - start_price

st.subheader("AI预测总结：")
if change > 1:
    trend = "上涨"
elif change < -1:
    trend = "下跌"
else:
    trend = "震荡整理"

st.markdown(f"根据AI预测，全球股市近期趋势对黄金市场影响已考虑，"
            f"预计未来 {period} 天黄金价格将 **{trend}**，从约 **{start_price:.2f} 美元** 变动到 **{end_price:.2f} 美元**。")
