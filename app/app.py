import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor")
st.caption("Educational project. Not financial advice.")


@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", progress=False)
    if df.empty:
        return df
    df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def add_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    for lag in [1, 2, 3, 5]:
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)
    for w in [10, 20, 50]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volume_change"] = df["Volume"].pct_change()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))
    return df


def make_X(df):
    X = pd.DataFrame(index=df.index)
    for c in ["Return", "Return_lag1", "Return_lag2", "Return_lag3",
              "Return_lag5", "Volatility_10", "Volume_change", "RSI_14"]:
        X[c] = df[c]
    for c in ["SMA_10", "SMA_20", "SMA_50", "EMA_12"]:
        X[f"Close_vs_{c}"] = df["Close"] / df[c] - 1
    return X.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=3600)
def predict_next_day(df):
    X = make_X(df)
    y = df["Close"].shift(-1) / df["Close"] - 1
    train = X.notna().all(axis=1) & y.notna()
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                         subsample=0.8, random_state=42)
    model.fit(X[train], y[train])
    last = X.iloc[[-1]]
    if last.isna().any(axis=1).iloc[0]:
        return None
    return float(df["Close"].iloc[-1] * (1 + model.predict(last)[0]))


ticker = st.text_input("Enter a ticker (for example TCS.NS)", "TCS.NS")
raw = load_data(ticker)

if raw.empty:
    st.error("No data found for that ticker.")
else:
    df = add_features(raw)
    last_close = float(df["Close"].iloc[-1])
    estimate = predict_next_day(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Last close", f"{last_close:,.2f}")
    c2.metric("Baseline (tomorrow = today)", f"{last_close:,.2f}")
    if estimate is not None:
        c3.metric("XGBoost next-day estimate", f"{estimate:,.2f}",
                  f"{estimate - last_close:+.2f}")

    st.warning(
        "In my walk-forward tests on TCS (2021 to 2026), this model did not "
        "beat the simple baseline in any period. Treat the estimate as a "
        "demonstration, not a trading signal."
    )

    st.subheader(f"{ticker} closing price")
    st.line_chart(df[["Close", "SMA_20", "SMA_50"]])

    left, right = st.columns(2)
    left.subheader("RSI (14)")
    left.line_chart(df["RSI_14"])
    right.subheader("Daily returns")
    right.line_chart(df["Return"])