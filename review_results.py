"""
Review Results Script
Quick summary of model performance and backtest results
"""

import pandas as pd
import glob

print("="*60)
print("STOCK FORECASTING PROJECT - RESULTS SUMMARY")
print("="*60)

# XGBoost Model Performance
print("\n📊 XGBOOST MODEL PERFORMANCE")
print("-" * 60)
files = glob.glob('models/xgb_*_metrics.csv')
for f in sorted(files):
    ticker = f.split('_')[1]
    df = pd.read_csv(f)
    print(f"\n{ticker}:")
    print(f"  Test RMSE: ${df['test_rmse'].values[0]:.2f}")
    print(f"  Test MAE:  ${df['test_mae'].values[0]:.2f}")
    print(f"  Test MAPE: {df['test_mape'].values[0]:.2f}%")

# ARIMA Model Performance
print("\n\n📈 ARIMA MODEL PERFORMANCE")
print("-" * 60)
files = glob.glob('models/arima_*_metrics.csv')
for f in sorted(files):
    ticker = f.split('_')[1]
    df = pd.read_csv(f)
    print(f"\n{ticker}:")
    print(f"  Model: {df['model'].values[0]} {df['order'].values[0]}")
    print(f"  Test RMSE: ${df['rmse'].values[0]:.2f}")
    print(f"  Test MAE:  ${df['mae'].values[0]:.2f}")
    print(f"  Test MAPE: {df['mape'].values[0]:.2f}%")

# Trading Strategy Performance
print("\n\n💰 TRADING STRATEGY PERFORMANCE (XGBoost-based)")
print("-" * 60)
files = glob.glob('report/*_backtest_xgb_metrics.csv')
for f in sorted(files):
    ticker = f.split('\\')[-1].split('_')[0]
    df = pd.read_csv(f)
    print(f"\n{ticker}:")
    print(f"  Strategy Total Return:  {df['strategy_total_return'].values[0]:>8.2f}%")
    print(f"  Market Total Return:    {df['market_total_return'].values[0]:>8.2f}%")
    print(f"  Strategy Sharpe Ratio:  {df['strategy_sharpe'].values[0]:>8.2f}")
    print(f"  Max Drawdown:           {df['strategy_max_drawdown'].values[0]:>8.2f}%")
    print(f"  Win Rate:               {df['strategy_win_rate'].values[0]:>8.2f}%")

# Summary comparison
print("\n\n🏆 MODEL COMPARISON (Test MAPE - Lower is Better)")
print("-" * 60)
comparison = []
for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
    arima_df = pd.read_csv(f'models/arima_{ticker}_metrics.csv')
    xgb_df = pd.read_csv(f'models/xgb_{ticker}_metrics.csv')
    arima_mape = arima_df['mape'].values[0]
    xgb_mape = xgb_df['test_mape'].values[0]
    winner = "XGBoost" if xgb_mape < arima_mape else "ARIMA"
    comparison.append({
        'Ticker': ticker,
        'ARIMA MAPE': f'{arima_mape:.2f}%',
        'XGBoost MAPE': f'{xgb_mape:.2f}%',
        'Winner': winner
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

print("\n" + "="*60)
print("✅ All results reviewed successfully!")
print("="*60)
