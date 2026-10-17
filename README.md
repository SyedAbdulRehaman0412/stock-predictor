# Stock Price Predictor

A machine learning project that predicts the next-day closing price of Indian stocks (tested mainly on TCS.NS), with a Streamlit web app.

## What I built
- Downloaded 2020-2026 price data with yfinance
- Exploratory analysis: moving averages, daily returns, volume
- Feature engineering: lagged returns, SMA/EMA, volatility, RSI
- Models: Linear Regression, Random Forest, XGBoost
- Time-based split and walk-forward validation (5 folds, 2021-2026)
- Streamlit app showing charts, indicators and a next-day estimate

## Key finding
No model beat the naive baseline ("tomorrow = today") in any of the 5 walk-forward periods.
Predicting returns instead of raw prices fixed the Random Forest's problem with new price ranges
(MAE 48.03 -> 32.45), but daily prices remain mostly noise.

## Run it
pip install -r requirements.txt
streamlit run app/app.py

## Disclaimer
Educational project. Not financial advice.