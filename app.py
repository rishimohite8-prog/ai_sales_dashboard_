import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📊 AI Sales Analyzer", layout="wide")

st.title("🤖 AI Sales Data Analyzer & Forecast Dashboard")

uploaded_file = st.file_uploader("📂 Upload Sales CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Cleaning
    st.subheader("🧹 Data Cleaning")
    df = df.dropna()
    st.success("Missing values removed successfully!")

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")

    # Visualization
    st.subheader("📈 Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Sales'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    # Forecast
    st.subheader("🔮 Sales Forecast (AI)")
    df['DayIndex'] = np.arange(len(df))
    X = df[['DayIndex']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)

    future_days = 30
    future_index = np.arange(len(df), len(df) + future_days).reshape(-1,1)
    predictions = model.predict(future_index)

    fig2, ax2 = plt.subplots()
    ax2.plot(df['Date'], df['Sales'])
    future_dates = pd.date_range(df['Date'].iloc[-1], periods=future_days)
    ax2.plot(future_dates, predictions)
    ax2.set_title("Next 30 Days Sales Prediction")
    st.pyplot(fig2)

    # AI Insights
    st.subheader("🧠 AI Insights")
    growth = (predictions[-1] - df['Sales'].iloc[-1]) / df['Sales'].iloc[-1] * 100

    if growth > 0:
        st.success(f"📈 Expected Growth: {growth:.2f}% in next 30 days.")
    else:
        st.warning(f"📉 Possible Decline: {growth:.2f}% in next 30 days.")

else:
    st.info("👆 Upload a CSV file with columns: Date, Sales")
