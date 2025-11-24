# Stock Price Forecasting Project Report

**Author**: [Your Name]  
**Date**: November 24, 2025  
**Project**: Stock Price Forecasting with ARIMA and XGBoost

---

## Executive Summary

This project implements and compares two forecasting approaches for stock price prediction: statistical time series modeling (ARIMA) and machine learning (XGBoost). We analyzed five major tech stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) from 2015-2025 and evaluated model performance using multiple metrics and backtesting strategies.

**Key Findings**:
- [Model performance summary - which performed better]
- [Trading strategy results - profitability]
- [Best ticker for prediction]

---

## 1. Introduction

### 1.1 Objective
Develop and compare forecasting models to predict next-day stock prices and evaluate their effectiveness through backtested trading strategies.

### 1.2 Motivation
Stock price prediction is challenging but valuable for:
- Investment decision support
- Risk management
- Portfolio optimization
- Understanding market dynamics

---

## 2. Dataset

### 2.1 Data Source
- **Source**: Yahoo Finance via yfinance library
- **Tickers**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **Period**: January 1, 2015 to November 23, 2025
- **Features**: Open, High, Low, Close, Adjusted Close, Volume
- **Total Records**: ~2,700 trading days per ticker

### 2.2 Data Quality
- Missing values: [X%] - handled via interpolation
- Outliers: Checked using Z-score method
- Stationarity: Original series non-stationary, differencing applied

**Links**:
- yfinance: https://github.com/ranaroussi/yfinance
- Alternative: Kaggle datasets

---

## 3. Exploratory Data Analysis

### 3.1 Price Trends
[Insert: Time series plot of Adj Close for all tickers]

**Observations**:
- Strong upward trends across all stocks
- TSLA shows highest volatility
- COVID-19 impact visible (March 2020 dip)

### 3.2 Volatility Analysis
[Insert: Rolling volatility plot]

**Findings**:
- Average annualized volatility ranges from X% (MSFT) to Y% (TSLA)
- Volatility clusters during market events

### 3.3 Returns Distribution
[Insert: Histogram with normal distribution overlay]

**Statistics**:
| Ticker | Mean Return | Std Dev | Skewness | Kurtosis |
|--------|-------------|---------|----------|----------|
| AAPL   | X%          | Y%      | Z        | K        |
| MSFT   | X%          | Y%      | Z        | K        |
| ...    | ...         | ...     | ...      | ...      |

**Key Insight**: Fat tails (high kurtosis) indicate non-normal distribution, justifying both statistical and ML approaches.

### 3.4 Correlation Analysis
[Insert: Correlation heatmap]

**Findings**:
- High correlation (0.6-0.8) among tech stocks
- TSLA shows slightly lower correlation with others

### 3.5 ACF/PACF Analysis
[Insert: ACF and PACF plots for AAPL]

**ARIMA Parameters Suggestion**:
- Significant autocorrelation up to lag X
- PACF cuts off at lag Y
- Suggested starting point: ARIMA(p,1,q)

---

## 4. Feature Engineering

### 4.1 Features Created
**Total Features**: 50+

**Categories**:
1. **Lagged Prices**: lag_1, lag_2, lag_3, lag_5, lag_10
2. **Returns**: pct_change_1, pct_change_3, pct_change_5, pct_change_10
3. **Moving Averages**: ma_5, ma_10, ma_20, ma_50, ma_200, ema_12, ema_26
4. **Volatility**: volatility_5, volatility_10, volatility_20
5. **Technical Indicators**:
   - RSI (14-day)
   - MACD, MACD Signal, MACD Difference
   - Bollinger Bands (upper, lower, width, position)
6. **Volume Features**: volume_change, volume_ma_20, volume_ratio
7. **Price Range**: high_low_range, close_open_diff
8. **Momentum**: momentum_5, momentum_10, momentum_20

### 4.2 Feature Selection
[Insert: Feature importance chart from XGBoost]

**Top 10 Features**:
1. lag_1
2. ma_20
3. rsi_14
4. ...

**Reasoning**: 
- Lagged values capture recent trends
- Moving averages smooth noise
- Technical indicators capture market sentiment

---

## 5. Modeling Approach

### 5.1 ARIMA Model

**Methodology**:
- Auto-parameter selection using `pmdarima.auto_arima()`
- Searched parameter space: p=[0,5], q=[0,5]
- Differencing: d=1 (to achieve stationarity)
- Train/test split: 80/20 (time-aware)

**Best Parameters** (per ticker):
| Ticker | Order (p,d,q) | AIC     |
|--------|---------------|---------|
| AAPL   | (X, 1, Y)     | ZZZZ.Z  |
| MSFT   | (X, 1, Y)     | ZZZZ.Z  |
| ...    | ...           | ...     |

### 5.2 XGBoost Model

**Methodology**:
- Regression objective: `reg:squarederror`
- Cross-validation: TimeSeriesSplit (3 folds)
- Hyperparameter tuning: GridSearchCV

**Hyperparameters Tested**:
- n_estimators: [100, 200, 300]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.05, 0.1]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]

**Best Parameters** (AAPL example):
- n_estimators: XXX
- max_depth: X
- learning_rate: 0.XX
- subsample: 0.X
- colsample_bytree: 0.X

---

## 6. Model Evaluation

### 6.1 Performance Metrics

**ARIMA Results**:
| Ticker | RMSE    | MAE     | MAPE   |
|--------|---------|---------|--------|
| AAPL   | $X.XX   | $Y.YY   | Z.ZZ%  |
| MSFT   | $X.XX   | $Y.YY   | Z.ZZ%  |
| GOOGL  | $X.XX   | $Y.YY   | Z.ZZ%  |
| AMZN   | $X.XX   | $Y.YY   | Z.ZZ%  |
| TSLA   | $X.XX   | $Y.YY   | Z.ZZ%  |

**XGBoost Results**:
| Ticker | RMSE    | MAE     | MAPE   |
|--------|---------|---------|--------|
| AAPL   | $X.XX   | $Y.YY   | Z.ZZ%  |
| MSFT   | $X.XX   | $Y.YY   | Z.ZZ%  |
| GOOGL  | $X.XX   | $Y.YY   | Z.ZZ%  |
| AMZN   | $X.XX   | $Y.YY   | Z.ZZ%  |
| TSLA   | $X.XX   | $Y.YY   | Z.ZZ%  |

**Winner**: [ARIMA / XGBoost] performed better with [XX%] lower MAPE on average.

### 6.2 Prediction Visualization
[Insert: Actual vs Predicted plots for both models]

**Observations**:
- Both models capture overall trends
- XGBoost shows [better/worse] performance on volatility spikes
- ARIMA struggles with [specific pattern]

### 6.3 Residual Analysis
[Insert: Residual plots and ACF of residuals]

**ARIMA Residuals**:
- Mean: ~0 (unbiased)
- No significant autocorrelation (white noise)
- Slight heteroscedasticity

---

## 7. Backtesting & Trading Strategy

### 7.1 Strategy Design
**Type**: Long-only  
**Signal**: If predicted_price > current_price * (1 + threshold) → go long, else flat  
**Parameters**:
- Threshold: 1%
- Transaction cost: 0.05% per trade

### 7.2 Backtest Results

**Market (Buy & Hold)**:
| Ticker | Total Return | Annual Return | Volatility | Sharpe | Max DD   |
|--------|--------------|---------------|------------|--------|----------|
| AAPL   | XXX%         | YY%           | ZZ%        | A.AA   | -BB%     |
| ...    | ...          | ...           | ...        | ...    | ...      |

**XGBoost Strategy**:
| Ticker | Total Return | Annual Return | Volatility | Sharpe | Max DD   | Win Rate |
|--------|--------------|---------------|------------|--------|----------|----------|
| AAPL   | XXX%         | YY%           | ZZ%        | A.AA   | -BB%     | CC%      |
| ...    | ...          | ...           | ...        | ...    | ...      | ...      |

**Excess Returns**: Strategy vs Market
[Insert: Bar chart comparing returns]

### 7.3 Cumulative Returns
[Insert: Cumulative return plot - Strategy vs Market]

**Key Findings**:
- Strategy [outperformed/underperformed] buy-and-hold by X%
- Best performance on [TICKER]
- Transaction costs reduced returns by Y%

---

## 8. Discussion

### 8.1 Model Comparison
**ARIMA Strengths**:
- Simple, interpretable
- Fast training
- Good for stable trends

**ARIMA Weaknesses**:
- Limited feature incorporation
- Struggles with complex patterns
- Sensitive to outliers

**XGBoost Strengths**:
- Incorporates multiple features
- Captures non-linear patterns
- Better handling of volatility

**XGBoost Weaknesses**:
- Requires feature engineering
- Slower training
- Risk of overfitting

### 8.2 Business Implications
1. **Investment Strategy**: [Which model to use when]
2. **Risk Management**: [How to use predictions]
3. **Portfolio Optimization**: [Diversification insights]

### 8.3 Limitations
1. **Assumption of Efficient Markets**: Prices may already reflect available information
2. **Transaction Costs**: Real-world costs may be higher
3. **Market Impact**: Large trades affect prices
4. **Regime Changes**: Models trained on past data may not capture future regimes
5. **Overfitting Risk**: XGBoost may overfit to training period

---

## 9. Future Work

### 9.1 Model Improvements
1. **Ensemble Methods**: Combine ARIMA and XGBoost predictions
2. **Deep Learning**: Test LSTM/GRU neural networks
3. **Sentiment Analysis**: Incorporate news/social media sentiment
4. **Multi-step Forecasting**: Predict beyond next day

### 9.2 Feature Engineering
1. **Alternative Data**: Options flow, insider trading
2. **Macro Indicators**: Interest rates, GDP, unemployment
3. **Sector Analysis**: Industry-specific features

### 9.3 Strategy Enhancement
1. **Position Sizing**: Kelly criterion, risk parity
2. **Stop-Loss**: Dynamic stop-loss rules
3. **Multi-Asset**: Portfolio-level optimization
4. **Regime Detection**: Adapt strategy to market conditions

---

## 10. Conclusion

This project successfully implemented and compared ARIMA and XGBoost models for stock price forecasting. [Summary of which model performed better and why].

**Key Takeaways**:
1. [Main finding 1]
2. [Main finding 2]
3. [Main finding 3]

**Practical Value**:
- Demonstrates end-to-end ML pipeline
- Shows importance of feature engineering
- Highlights realistic backtesting with transaction costs

While stock prediction remains challenging, this analysis provides a solid framework for:
- Understanding time series forecasting
- Comparing statistical vs ML approaches
- Building realistic trading strategies

---

## References

1. yfinance: https://github.com/ranaroussi/yfinance
2. XGBoost: Chen & Guestrin (2016)
3. ARIMA: Box & Jenkins (1970)
4. pmdarima: https://alkaline-ml.com/pmdarima/
5. Technical Analysis Library: https://github.com/bukosabino/ta

---

## Appendix

### A. Code Repository
GitHub: [Your repository URL]

### B. File Structure
```
invsto-stock-forecasting/
├── notebooks/     # EDA and analysis
├── src/           # Source code
├── data/          # Raw and processed data
├── models/        # Trained models
└── report/        # This report
```

### C. Requirements
See `requirements.txt` for complete package list.

### D. Execution Commands
See `QUICKSTART.md` for step-by-step instructions.

---

**End of Report**
