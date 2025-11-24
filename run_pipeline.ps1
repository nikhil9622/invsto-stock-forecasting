# Stock Forecasting Quick Start Script
# Run this script to execute the entire pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stock Forecasting Pipeline - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$TICKERS = @("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA")
$START_DATE = "2015-01-01"
$END_DATE = "2025-11-23"

# Use Python 3.12 (where packages are installed)
$PYTHON = "C:\Python312\python.exe"

# Step 1: Download Data
Write-Host "[1/6] Downloading stock data..." -ForegroundColor Yellow
& $PYTHON src/data_download.py --tickers $TICKERS --start $START_DATE --end $END_DATE --out data/raw/
if ($LASTEXITCODE -ne 0) { Write-Host "Error downloading data!" -ForegroundColor Red; exit }

# Step 2: Clean Data for each ticker
Write-Host "`n[2/6] Cleaning and preprocessing data..." -ForegroundColor Yellow
foreach ($ticker in $TICKERS) {
    Write-Host "  Processing $ticker..." -ForegroundColor Gray
    & $PYTHON src/data_cleaning.py --ticker $ticker --in "data/raw/$ticker.csv" --out "data/processed/$ticker.parquet"
    if ($LASTEXITCODE -ne 0) { Write-Host "Error cleaning $ticker!" -ForegroundColor Red; exit }
}

# Step 3: Feature Engineering
Write-Host "`n[3/6] Creating features..." -ForegroundColor Yellow
foreach ($ticker in $TICKERS) {
    Write-Host "  Creating features for $ticker..." -ForegroundColor Gray
    & $PYTHON src/features.py --in "data/processed/$ticker.parquet" --out "data/processed/${ticker}_features.parquet"
    if ($LASTEXITCODE -ne 0) { Write-Host "Error creating features for $ticker!" -ForegroundColor Red; exit }
}

# Step 4: Train ARIMA Models
Write-Host "`n[4/6] Training ARIMA models..." -ForegroundColor Yellow
foreach ($ticker in $TICKERS) {
    Write-Host "  Training ARIMA for $ticker..." -ForegroundColor Gray
    & $PYTHON src/train_arima.py --in "data/processed/$ticker.parquet" --ticker $ticker --out "models/arima_$ticker.pkl"
    if ($LASTEXITCODE -ne 0) { Write-Host "Error training ARIMA for $ticker!" -ForegroundColor Red; exit }
}

# Step 5: Train XGBoost Models
Write-Host "`n[5/6] Training XGBoost models..." -ForegroundColor Yellow
foreach ($ticker in $TICKERS) {
    Write-Host "  Training XGBoost for $ticker..." -ForegroundColor Gray
    & $PYTHON src/train_xgb.py --in "data/processed/${ticker}_features.parquet" --ticker $ticker --out "models/xgb_$ticker.pkl"
    if ($LASTEXITCODE -ne 0) { Write-Host "Error training XGBoost for $ticker!" -ForegroundColor Red; exit }
}

# Step 6: Backtest Strategies
Write-Host "`n[6/6] Backtesting trading strategies..." -ForegroundColor Yellow
foreach ($ticker in $TICKERS) {
    Write-Host "  Backtesting $ticker..." -ForegroundColor Gray
    & $PYTHON src/backtest.py --in "data/processed/${ticker}_features.parquet" --model_xgb "models/xgb_$ticker.pkl" --ticker $ticker
    if ($LASTEXITCODE -ne 0) { Write-Host "Error backtesting $ticker!" -ForegroundColor Red; exit }
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Pipeline Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - Models: models/" -ForegroundColor White
Write-Host "  - Predictions: models/*_predictions.csv" -ForegroundColor White
Write-Host "  - Metrics: models/*_metrics.csv" -ForegroundColor White
Write-Host "  - Backtest: report/*_backtest_xgb.csv" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review model metrics in models/*_metrics.csv" -ForegroundColor White
Write-Host "  2. Check backtest results in report/" -ForegroundColor White
Write-Host "  3. Run the Jupyter notebook for visualizations" -ForegroundColor White
Write-Host "  4. Generate final report" -ForegroundColor White
