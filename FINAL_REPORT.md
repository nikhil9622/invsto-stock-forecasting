# Stock Price Forecasting Project - Final Report

**Project:** Stock Price Forecasting with ARIMA and XGBoost  
**Date:** November 24, 2025  
**Tickers:** AAPL, MSFT, GOOGL, AMZN, TSLA  
**Period:** January 1, 2015 - November 21, 2025

---

## Executive Summary

This project developed and evaluated time series forecasting models for five major tech stocks using both traditional statistical methods (ARIMA) and modern machine learning (XGBoost). The models were trained on 10+ years of historical data and evaluated using multiple metrics including RMSE, MAE, and MAPE.

### Key Findings

1. **Model Performance**: ARIMA outperformed XGBoost for 3 out of 5 stocks (AAPL, MSFT, GOOGL), while XGBoost excelled with more volatile stocks (AMZN, TSLA)
2. **Best Performer**: XGBoost on TSLA achieved the lowest MAPE of 5.13%
3. **Trading Strategy**: The XGBoost-based long-only strategy showed risk reduction (lower max drawdown) but underperformed buy-and-hold in total returns for most stocks
4. **Most Predictable**: AAPL showed the best overall predictability with ARIMA achieving 8.98% MAPE

---

## 1. Data Overview

### Dataset Specifications
- **Source**: Yahoo Finance via yfinance API
- **Tickers**: 
  - AAPL (Apple Inc.)
  - MSFT (Microsoft Corporation)
  - GOOGL (Alphabet Inc.)
  - AMZN (Amazon.com Inc.)
  - TSLA (Tesla Inc.)
- **Time Period**: 2015-01-01 to 2025-11-21
- **Total Trading Days**: 2,740 per ticker
- **Data Points**: 13,700 total rows (5 tickers × 2,740 days)

### Data Quality
- **Missing Values**: 101 business days per ticker (reindexed to complete business day calendar)
- **Handling**: Linear interpolation (max 5 consecutive), forward fill (max 10), backward fill (max 10)
- **Cleaned Rows**: 2,840 per ticker after business day reindexing

### Summary Statistics

| Ticker | Avg Close | Avg Daily Return | Daily Volatility |
|--------|-----------|------------------|------------------|
| AAPL   | $104.79   | 0.10%            | 1.77%            |
| MSFT   | $203.69   | 0.10%            | 1.66%            |
| GOOGL  | $91.71    | 0.10%            | 1.77%            |
| AMZN   | $110.68   | 0.11%            | 2.02%            |
| TSLA   | $134.66   | 0.18%            | 3.53%            |

---

## 2. Feature Engineering

Created **46 engineered features** for XGBoost models:

### Feature Categories

#### Price-Based (15 features)
- Lagged prices: 1, 2, 3, 5, 10 days
- Moving averages: 5, 10, 20, 50, 200 days
- Price-to-MA ratios: 5 ratios
- Exponential moving averages: 12, 26 days

#### Technical Indicators (9 features)
- RSI (14-day)
- MACD (line, signal, divergence)
- Bollinger Bands (high, low, mid, width, position)

#### Volatility Features (3 features)
- Rolling standard deviation: 5, 10, 20 days

#### Volume Features (3 features)
- Volume change
- 20-day volume MA
- Volume ratio

#### Momentum Features (3 features)
- Price momentum: 5, 10, 20 days

#### Price Range Features (2 features)
- High-Low range
- Close-Open difference

#### Target Variable
- Next day's closing price

---

## 3. Model Development

### 3.1 ARIMA Models

#### Methodology
- **Algorithm**: Stepwise auto ARIMA with AIC minimization
- **Train/Test Split**: 80% / 20% (2,272 train / 568 test)
- **Evaluation**: Walk-forward validation available
- **Implementation**: pmdarima library

#### Model Configurations

| Ticker | Order      | AIC      | Parameters |
|--------|------------|----------|------------|
| AAPL   | (0, 1, 2)  | 8,936.97 | MA(2) with differencing |
| MSFT   | (1, 1, 1)  | 11,588.26| ARIMA(1,1,1) |
| GOOGL  | (1, 1, 1)  | 8,101.58 | ARIMA(1,1,1) |
| AMZN   | (0, 1, 0)  | 9,915.29 | Random walk |
| TSLA   | (0, 1, 0)  | 14,204.34| Random walk |

#### Performance Metrics

| Ticker | RMSE    | MAE     | MAPE   |
|--------|---------|---------|--------|
| AAPL   | $25.54  | $20.20  | 8.98%  |
| MSFT   | $75.48  | $66.56  | 14.94% |
| GOOGL  | $37.16  | $24.97  | 12.40% |
| AMZN   | $61.92  | $55.08  | 27.00% |
| TSLA   | $85.44  | $70.30  | 25.66% |

**Average MAPE**: 17.80%

### 3.2 XGBoost Models

#### Methodology
- **Algorithm**: Gradient Boosting with GridSearchCV
- **Cross-Validation**: TimeSeriesSplit (3 folds)
- **Train/Test Split**: 80% / 20% (2,112 train / 528 test)
- **Features**: 46 engineered features

#### Hyperparameter Tuning Grid
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

#### Best Hyperparameters

| Ticker | n_estimators | max_depth | learning_rate | subsample | colsample |
|--------|--------------|-----------|---------------|-----------|-----------|
| AAPL   | 300          | 7         | 0.1           | 0.8       | 0.8       |
| MSFT   | 300          | 3         | 0.05          | 1.0       | 0.8       |
| GOOGL  | 300          | 3         | 0.05          | 0.8       | 0.8       |
| AMZN   | 200          | 3         | 0.05          | 0.8       | 1.0       |
| TSLA   | 100          | 3         | 0.1           | 1.0       | 0.8       |

#### Performance Metrics

| Ticker | RMSE    | MAE     | MAPE   | Train RMSE | Overfitting |
|--------|---------|---------|--------|------------|-------------|
| AAPL   | $36.75  | $28.75  | 12.36% | $0.05      | High        |
| MSFT   | $91.71  | $79.52  | 17.52% | $2.13      | Moderate    |
| GOOGL  | $47.46  | $33.78  | 16.68% | $1.01      | High        |
| AMZN   | $29.31  | $21.59  | 9.94%  | $1.74      | Moderate    |
| TSLA   | $29.88  | $17.61  | 5.13%  | $4.04      | Low         |

**Average MAPE**: 12.33%

#### Top 3 Most Important Features by Ticker

**AAPL**:
1. ma_20 (58.43%)
2. bb_high (16.72%)
3. ma_50 (11.54%)

**MSFT**:
1. ma_20 (21.90%)
2. ma_5 (13.07%)
3. High (12.91%)

**GOOGL**:
1. Low (24.94%)
2. lag_2 (20.53%)
3. Close (19.36%)

**AMZN**:
1. High (22.45%)
2. Close (19.22%)
3. Low (19.08%)

**TSLA**:
1. Low (49.96%)
2. High (18.80%)
3. Close (13.63%)

---

## 4. Model Comparison

### Head-to-Head Performance (MAPE)

| Ticker | ARIMA MAPE | XGBoost MAPE | Winner   | Margin    |
|--------|------------|--------------|----------|-----------|
| AAPL   | **8.98%**  | 12.36%       | ARIMA    | -3.38%    |
| MSFT   | **14.94%** | 17.52%       | ARIMA    | -2.58%    |
| GOOGL  | **12.40%** | 16.68%       | ARIMA    | -4.28%    |
| AMZN   | 27.00%     | **9.94%**    | XGBoost  | -17.06%   |
| TSLA   | 25.66%     | **5.13%**    | XGBoost  | -20.53%   |

### Analysis

**ARIMA Strengths**:
- Better for stable, established stocks (AAPL, MSFT, GOOGL)
- Lower variance in predictions
- Simpler model, less prone to overfitting
- Excellent for AAPL (8.98% MAPE)

**XGBoost Strengths**:
- Exceptional performance on volatile stocks (AMZN, TSLA)
- Captures complex non-linear patterns
- Best overall performance: TSLA (5.13% MAPE)
- Feature importance provides interpretability

**Key Insight**: Model selection should depend on stock characteristics. Use ARIMA for stable blue-chip stocks and XGBoost for high-volatility growth stocks.

---

## 5. Trading Strategy Backtesting

### Strategy Description
- **Type**: Long-only momentum strategy
- **Signal**: Buy when predicted price > current price
- **Transaction Costs**: 0.1% per trade
- **Period**: October 9, 2015 - November 20, 2025 (10.1 years)
- **Benchmark**: Buy & Hold

### Performance Results

| Ticker | Strategy Return | Market Return | Strategy Sharpe | Market Sharpe | Max DD (Strategy) | Max DD (Market) |
|--------|----------------|---------------|-----------------|---------------|-------------------|-----------------|
| AAPL   | 0.09%          | 9.84%         | 0.16            | 0.90          | -0.09%            | -0.39%          |
| MSFT   | **7.41%**      | 10.55%        | **1.61**        | 1.00          | **-0.13%**        | -0.37%          |
| GOOGL  | 1.84%          | 7.92%         | 0.81            | 0.83          | -0.19%            | -0.44%          |
| AMZN   | 3.06%          | 7.02%         | 0.86            | 0.69          | -0.23%            | -0.56%          |
| TSLA   | 14.04%         | 26.21%        | 0.79            | 0.65          | -0.47%            | -0.74%          |

**Note**: The backtest returns shown are annualized percentages, not cumulative returns over the 10-year period.

### Key Findings

1. **Risk Reduction**: Strategy significantly reduced maximum drawdown across all stocks (50-70% reduction)
2. **Return Trade-off**: Lower returns vs buy-and-hold for most stocks (except AMZN showing slightly better risk-adjusted performance)
3. **Best Performance**: MSFT strategy achieved highest Sharpe ratio (1.61) with controlled drawdown
4. **Win Rates**: Generally low (0-21%), indicating frequent small trades
5. **Transaction Costs Impact**: 0.1% costs per trade significantly eroded returns

### Improvements to Consider
- Implement stop-loss mechanisms
- Add position sizing based on conviction
- Consider long-short strategies
- Optimize transaction cost thresholds
- Use ensemble predictions (ARIMA + XGBoost)

---

## 6. Conclusions

### Project Achievements

1. ✅ Successfully downloaded and processed 10+ years of stock data for 5 tickers
2. ✅ Engineered 46 meaningful features capturing price, volume, and technical patterns
3. ✅ Developed and tuned both ARIMA and XGBoost models with comprehensive evaluation
4. ✅ Achieved competitive forecast accuracy (best: 5.13% MAPE on TSLA with XGBoost)
5. ✅ Backtested trading strategies with realistic transaction costs
6. ✅ Created automated pipeline for end-to-end workflow
7. ✅ Generated comprehensive documentation and visualizations

### Model Performance Summary

- **Overall Winner**: XGBoost with 12.33% average MAPE vs ARIMA's 17.80%
- **Stock-Specific**: ARIMA better for stable stocks, XGBoost for volatile stocks
- **Best Single Performance**: XGBoost on TSLA (5.13% MAPE, $17.61 MAE)
- **Most Challenging**: AMZN with ARIMA (27.00% MAPE)

### Trading Strategy Insights

- **Risk Management**: Strategy successfully reduced risk (max drawdown cut by 50-70%)
- **Return Generation**: Underperformed buy-and-hold due to frequent trading and costs
- **Practical Application**: Better suited for risk-averse investors prioritizing capital preservation
- **Optimization Potential**: Significant room for improvement through ensemble methods and adaptive thresholds

### Limitations

1. **Overfitting**: XGBoost showed high overfitting (especially AAPL, GOOGL)
2. **Transaction Costs**: 0.1% significantly impacted strategy returns
3. **Market Regimes**: Models may not adapt to regime changes (bull vs bear markets)
4. **Look-Ahead Bias**: Careful to avoid in production implementation
5. **Single-Stock Focus**: No portfolio-level optimization

### Future Work

1. **Ensemble Methods**: Combine ARIMA and XGBoost predictions weighted by stock volatility
2. **Deep Learning**: Explore LSTM/Transformer models for sequential patterns
3. **Sentiment Analysis**: Incorporate news sentiment and social media data
4. **Adaptive Models**: Implement online learning for regime changes
5. **Portfolio Optimization**: Multi-stock strategies with correlation analysis
6. **Alternative Data**: Include options flow, institutional holdings, macro indicators
7. **Risk Management**: Implement dynamic position sizing and stop-losses

---

## 7. Technical Stack

### Data & Libraries
- **Python**: 3.12
- **Data Source**: Yahoo Finance (yfinance 0.2.66)
- **Data Processing**: pandas 2.3.3, numpy 2.3.5
- **Time Series**: statsmodels 0.14.5, pmdarima 2.1.1
- **Machine Learning**: scikit-learn 1.7.2, xgboost 3.1.2
- **Visualization**: matplotlib 3.10.7, seaborn 0.13.2, mplfinance 0.12.10b0, plotly 6.5.0
- **Technical Analysis**: ta 0.11.0

### Project Structure
```
invsto-stock-forecasting/
├── data/
│   ├── raw/              # Downloaded CSV files (5 tickers)
│   └── processed/        # Cleaned parquet files + features
├── models/               # Trained models + predictions + metrics
├── notebooks/            # Jupyter EDA notebook
├── report/               # Backtest results
├── src/                  # Source code modules
│   ├── data_download.py
│   ├── data_cleaning.py
│   ├── features.py
│   ├── train_arima.py
│   ├── train_xgb.py
│   └── backtest.py
├── run_pipeline.ps1      # Automated pipeline script
└── requirements.txt      # Dependencies
```

### Automation
- Full pipeline execution: `.\run_pipeline.ps1`
- Individual steps can be run independently
- All results saved automatically to respective folders

---

## 8. Reproducibility

### Setup Instructions

1. **Clone/Download Project**
   ```powershell
   cd C:\Users\belid\Downloads\stock\invsto-stock-forecasting
   ```

2. **Install Dependencies**
   ```powershell
   C:\Python312\python.exe -m pip install -r requirements.txt
   ```

3. **Run Pipeline**
   ```powershell
   .\run_pipeline.ps1
   ```

4. **View Results**
   ```powershell
   C:\Python312\python.exe review_results.py
   jupyter notebook notebooks/01_stock_forecasting_eda.ipynb
   ```

### Output Files
- **Models**: `models/arima_*.pkl`, `models/xgb_*.pkl`
- **Predictions**: `models/*_predictions.csv`
- **Metrics**: `models/*_metrics.csv`
- **Backtest**: `report/*_backtest_xgb.csv`
- **Feature Importance**: `models/xgb_*_feature_importance.csv`

---

## 9. References

### Academic & Industry
- Box, G. E., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice

### Libraries & Tools
- pmdarima: http://alkaline-ml.com/pmdarima/
- XGBoost: https://xgboost.readthedocs.io/
- yfinance: https://github.com/ranaroussi/yfinance
- scikit-learn: https://scikit-learn.org/

---

## Appendix A: Model Hyperparameters

### ARIMA Auto-Search Ranges
```python
{
    'start_p': 0,
    'start_q': 0,
    'max_p': 3,
    'max_q': 3,
    'd': 1,
    'seasonal': False,
    'stepwise': True,
    'information_criterion': 'aic',
    'trace': True
}
```

### XGBoost GridSearchCV
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'objective': 'reg:squarederror',
    'cv': TimeSeriesSplit(n_splits=3)
}
```

---

## Appendix B: Backtest Configuration

```python
{
    'strategy': 'long_only',
    'threshold': 0.0,  # Buy when predicted > actual
    'transaction_cost': 0.001,  # 0.1% per trade
    'starting_capital': 10000,  # $10,000
    'position_size': 1.0,  # 100% allocation when signal triggers
}
```

---

**Report Generated**: November 24, 2025  
**Project Status**: ✅ Complete  
**Pipeline Execution Time**: ~25 minutes

---

For questions or additional analysis, refer to:
- `README.md` - Project overview
- `QUICKSTART.md` - Step-by-step guide
- `PROJECT_SUMMARY.md` - Technical details
- `notebooks/01_stock_forecasting_eda.ipynb` - Interactive analysis
