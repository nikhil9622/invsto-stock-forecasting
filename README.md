# Stock Price Forecasting with ARIMA and XGBoost

A comprehensive stock price forecasting project comparing time series (ARIMA) and machine learning (XGBoost) approaches with backtesting capabilities.

## 📊 Project Overview

This project implements and compares two forecasting approaches:
- **ARIMA**: Statistical time series model with auto-parameter selection
- **XGBoost**: Gradient boosting with engineered technical features

**Dataset**: Yahoo Finance (via yfinance)  
**Tickers**: AAPL, MSFT, GOOGL, AMZN, TSLA  
**Period**: 2015-01-01 to 2025-11-23

## 🏗️ Project Structure

```
invsto-stock-forecasting/
├─ notebooks/           # Jupyter notebooks for EDA and analysis
├─ src/                 # Source code modules
│  ├─ data_download.py  # Download stock data from Yahoo Finance
│  ├─ data_cleaning.py  # Clean and preprocess data
│  ├─ features.py       # Feature engineering
│  ├─ train_arima.py    # ARIMA model training
│  ├─ train_xgb.py      # XGBoost model training
│  └─ backtest.py       # Trading strategy backtesting
├─ data/                # Data storage
│  ├─ raw/              # Raw downloaded data
│  └─ processed/        # Cleaned and feature-engineered data
├─ models/              # Saved trained models
├─ report/              # Final report and visualizations
├─ requirements.txt     # Python dependencies
└─ README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
python src/data_download.py --tickers AAPL MSFT GOOGL AMZN TSLA --start 2015-01-01 --end 2025-11-23 --out data/raw/
```

### 3. Clean and Process Data

```bash
python src/data_cleaning.py --ticker AAPL --in data/raw/AAPL.csv --out data/processed/AAPL.parquet
```

### 4. Engineer Features

```bash
python src/features.py --in data/processed/AAPL.parquet --out data/processed/AAPL_features.parquet
```

### 5. Train Models

```bash
# ARIMA
python src/train_arima.py --in data/processed/AAPL.parquet --ticker AAPL --out models/arima_AAPL.pkl

# XGBoost
python src/train_xgb.py --in data/processed/AAPL_features.parquet --ticker AAPL --out models/xgb_AAPL.pkl
```

### 6. Backtest Strategy

```bash
python src/backtest.py --in data/processed/AAPL_features.parquet --model_xgb models/xgb_AAPL.pkl --model_arima models/arima_AAPL.pkl --ticker AAPL
```

## 📈 Features

### Data Processing
- Automatic handling of missing values
- Business day reindexing
- Adjusted close price normalization
- Log returns computation

### Feature Engineering
- Lagged prices (1, 2, 5 days)
- Moving averages (5, 10, 20, 50 days)
- Rolling volatility
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume features

### Models
- **ARIMA**: Auto-parameter selection with pmdarima
- **XGBoost**: Hyperparameter tuning with TimeSeriesSplit CV

### Backtesting
- Long/short trading signals
- Transaction cost simulation
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Walk-forward validation

## 📊 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Annualized return & volatility
- Sharpe ratio
- Maximum drawdown

## 🔗 Data Sources

- **Primary**: [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API
- **Alternative**: [Kaggle Stock Market Datasets](https://www.kaggle.com/datasets)
- **Alternative**: [Alpha Vantage API](https://www.alphavantage.co/)

## 📝 Report

The final report (`report/report.pdf`) includes:
- Executive summary
- Dataset description
- EDA highlights with visualizations
- Feature engineering methodology
- Model comparison and results
- Trading strategy performance
- Limitations and future work

## 🛠️ Technologies

- **Python 3.8+**
- **Data**: pandas, numpy, dask
- **Visualization**: matplotlib, seaborn, plotly, mplfinance
- **Time Series**: statsmodels, pmdarima
- **ML**: scikit-learn, xgboost
- **Technical Analysis**: ta

## 📄 License

MIT License

## 👥 Author

Created for stock price forecasting analysis and comparison study.
