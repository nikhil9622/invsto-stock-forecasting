# Quick Start Guide

This guide will help you get started with the stock forecasting project.

## Prerequisites

1. **Python 3.8+** installed
2. **pip** package manager

## Installation

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install all required packages including:
- yfinance (data download)
- pandas, numpy (data processing)
- matplotlib, seaborn, mplfinance (visualization)
- statsmodels, pmdarima (ARIMA)
- scikit-learn, xgboost (machine learning)
- ta (technical indicators)

### 2. Verify Installation

```powershell
python -c "import yfinance; import xgboost; import pmdarima; print('All packages installed successfully!')"
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Execute the automated pipeline script:

**Windows (PowerShell):**
```powershell
.\run_pipeline.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. Download data for AAPL, MSFT, GOOGL, AMZN, TSLA (2015-2025)
2. Clean and preprocess data
3. Engineer features
4. Train ARIMA models
5. Train XGBoost models
6. Backtest trading strategies

**Estimated time**: 15-30 minutes depending on your system

### Option 2: Run Step-by-Step

#### Step 1: Download Data
```powershell
python src/data_download.py --tickers AAPL MSFT GOOGL AMZN TSLA --start 2015-01-01 --end 2025-11-23 --out data/raw/
```

#### Step 2: Clean Data (example for AAPL)
```powershell
python src/data_cleaning.py --ticker AAPL --in data/raw/AAPL.csv --out data/processed/AAPL.parquet
```

Repeat for each ticker: MSFT, GOOGL, AMZN, TSLA

#### Step 3: Create Features
```powershell
python src/features.py --in data/processed/AAPL.parquet --out data/processed/AAPL_features.parquet
```

#### Step 4: Train ARIMA Model
```powershell
python src/train_arima.py --in data/processed/AAPL.parquet --ticker AAPL --out models/arima_AAPL.pkl
```

#### Step 5: Train XGBoost Model
```powershell
python src/train_xgb.py --in data/processed/AAPL_features.parquet --ticker AAPL --out models/xgb_AAPL.pkl
```

#### Step 6: Backtest Strategy
```powershell
python src/backtest.py --in data/processed/AAPL_features.parquet --model_xgb models/xgb_AAPL.pkl --ticker AAPL
```

### Option 3: Exploratory Data Analysis

Run the Jupyter notebook for interactive analysis:

```powershell
jupyter notebook notebooks/01_stock_forecasting_eda.ipynb
```

Or use VS Code Jupyter extension to open the notebook directly.

## Output Files

After running the pipeline, you'll find:

### Models
- `models/arima_<TICKER>.pkl` - Trained ARIMA models
- `models/xgb_<TICKER>.pkl` - Trained XGBoost models

### Predictions
- `models/arima_<TICKER>_predictions.csv` - ARIMA predictions
- `models/xgb_<TICKER>_predictions.csv` - XGBoost predictions

### Metrics
- `models/arima_<TICKER>_metrics.csv` - ARIMA performance metrics
- `models/xgb_<TICKER>_metrics.csv` - XGBoost performance metrics

### Backtesting
- `report/<TICKER>_backtest_xgb.csv` - Trading strategy results
- `report/<TICKER>_backtest_xgb_metrics.csv` - Strategy performance metrics

### Feature Importance
- `models/xgb_<TICKER>_feature_importance.csv` - Most important features

## Common Issues & Solutions

### Issue: Package Installation Fails

**Solution**: Upgrade pip and try again
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: yfinance Download Fails

**Solution**: Check internet connection and try with a single ticker first
```powershell
python src/data_download.py --tickers AAPL --start 2020-01-01 --end 2025-11-23 --out data/raw/
```

### Issue: Out of Memory Error

**Solution**: Process one ticker at a time or reduce date range
```powershell
# Process smaller date range
python src/data_download.py --tickers AAPL --start 2020-01-01 --end 2025-11-23 --out data/raw/
```

### Issue: ARIMA Training is Slow

**Solution**: Disable hyperparameter tuning for faster training
```powershell
# Edit train_arima.py and reduce max_p, max_q values
# Or skip walk-forward validation
```

## Next Steps

1. **Review Metrics**: Check `models/*_metrics.csv` files to compare ARIMA vs XGBoost performance

2. **Analyze Backtests**: Review `report/*_backtest_xgb_metrics.csv` for trading strategy performance

3. **Visualize Results**: Run the Jupyter notebook to create plots and visualizations

4. **Fine-tune Models**:
   - Adjust XGBoost hyperparameters in `src/train_xgb.py`
   - Modify feature engineering in `src/features.py`
   - Change trading strategy thresholds in `src/backtest.py`

5. **Generate Report**: Create `report/report.pdf` with your findings

## Tips for Success

- **Start Small**: Test with one ticker (AAPL) before running all five
- **Monitor Progress**: Watch console output for any errors or warnings
- **Check Data Quality**: Verify downloaded data has no missing values
- **Compare Models**: Use metrics to determine which model performs better
- **Realistic Expectations**: Stock prediction is challenging; focus on learning the process

## Advanced Options

### Run with Walk-Forward Validation
```powershell
python src/train_arima.py --in data/processed/AAPL.parquet --ticker AAPL --out models/arima_AAPL.pkl --walk_forward
```

### Skip Hyperparameter Tuning (Faster)
```powershell
python src/train_xgb.py --in data/processed/AAPL_features.parquet --ticker AAPL --out models/xgb_AAPL.pkl --no_tuning
```

### Custom Date Range
```powershell
python src/data_download.py --tickers AAPL --start 2020-01-01 --end 2024-12-31 --out data/raw/
```

### Long-Short Strategy
```powershell
python src/backtest.py --in data/processed/AAPL_features.parquet --model_xgb models/xgb_AAPL.pkl --ticker AAPL --strategy long_short
```

## Support

For issues or questions:
1. Check error messages carefully
2. Review the module-specific help: `python src/data_download.py --help`
3. Verify Python version: `python --version` (should be 3.8+)
4. Check package versions: `pip list`

Happy forecasting! 🚀📈
