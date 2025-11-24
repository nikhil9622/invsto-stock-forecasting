#!/bin/bash
# Stock Forecasting Quick Start Script (Bash version)
# Run this script to execute the entire pipeline

echo "========================================"
echo "Stock Forecasting Pipeline - Quick Start"
echo "========================================"
echo ""

# Configuration
TICKERS=("AAPL" "MSFT" "GOOGL" "AMZN" "TSLA")
START_DATE="2015-01-01"
END_DATE="2025-11-23"

# Step 1: Download Data
echo "[1/6] Downloading stock data..."
python src/data_download.py --tickers ${TICKERS[@]} --start $START_DATE --end $END_DATE --out data/raw/
if [ $? -ne 0 ]; then echo "Error downloading data!"; exit 1; fi

# Step 2: Clean Data for each ticker
echo ""
echo "[2/6] Cleaning and preprocessing data..."
for ticker in "${TICKERS[@]}"; do
    echo "  Processing $ticker..."
    python src/data_cleaning.py --ticker $ticker --in "data/raw/$ticker.csv" --out "data/processed/$ticker.parquet"
    if [ $? -ne 0 ]; then echo "Error cleaning $ticker!"; exit 1; fi
done

# Step 3: Feature Engineering
echo ""
echo "[3/6] Creating features..."
for ticker in "${TICKERS[@]}"; do
    echo "  Creating features for $ticker..."
    python src/features.py --in "data/processed/$ticker.parquet" --out "data/processed/${ticker}_features.parquet"
    if [ $? -ne 0 ]; then echo "Error creating features for $ticker!"; exit 1; fi
done

# Step 4: Train ARIMA Models
echo ""
echo "[4/6] Training ARIMA models..."
for ticker in "${TICKERS[@]}"; do
    echo "  Training ARIMA for $ticker..."
    python src/train_arima.py --in "data/processed/$ticker.parquet" --ticker $ticker --out "models/arima_$ticker.pkl"
    if [ $? -ne 0 ]; then echo "Error training ARIMA for $ticker!"; exit 1; fi
done

# Step 5: Train XGBoost Models
echo ""
echo "[5/6] Training XGBoost models..."
for ticker in "${TICKERS[@]}"; do
    echo "  Training XGBoost for $ticker..."
    python src/train_xgb.py --in "data/processed/${ticker}_features.parquet" --ticker $ticker --out "models/xgb_$ticker.pkl"
    if [ $? -ne 0 ]; then echo "Error training XGBoost for $ticker!"; exit 1; fi
done

# Step 6: Backtest Strategies
echo ""
echo "[6/6] Backtesting trading strategies..."
for ticker in "${TICKERS[@]}"; do
    echo "  Backtesting $ticker..."
    python src/backtest.py --in "data/processed/${ticker}_features.parquet" --model_xgb "models/xgb_$ticker.pkl" --ticker $ticker
    if [ $? -ne 0 ]; then echo "Error backtesting $ticker!"; exit 1; fi
done

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Models: models/"
echo "  - Predictions: models/*_predictions.csv"
echo "  - Metrics: models/*_metrics.csv"
echo "  - Backtest: report/*_backtest_xgb.csv"
echo ""
echo "Next steps:"
echo "  1. Review model metrics in models/*_metrics.csv"
echo "  2. Check backtest results in report/"
echo "  3. Run the Jupyter notebook for visualizations"
echo "  4. Generate final report"
