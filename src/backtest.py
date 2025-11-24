"""
Backtesting Module
Backtests trading strategies based on model predictions
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib


def calculate_returns(actual_prices, predicted_prices, strategy='long_only', 
                     threshold=0.01, transaction_cost=0.0005):
    """
    Calculate strategy returns based on predictions
    
    Parameters:
    -----------
    actual_prices : pd.Series
        Actual prices
    predicted_prices : pd.Series or np.array
        Predicted prices
    strategy : str
        'long_only' or 'long_short'
    threshold : float
        Minimum predicted change to take position
    transaction_cost : float
        Transaction cost as fraction (e.g., 0.0005 = 0.05%)
    
    Returns:
    --------
    pd.DataFrame with positions, returns, cumulative returns
    """
    df = pd.DataFrame({
        'actual': actual_prices,
        'predicted': predicted_prices
    }, index=actual_prices.index)
    
    # Calculate actual returns
    df['actual_return'] = df['actual'].pct_change()
    
    # Generate signals
    df['predicted_change'] = (df['predicted'] - df['actual']) / df['actual']
    
    if strategy == 'long_only':
        # Go long if predicted increase > threshold, else flat
        df['position'] = np.where(df['predicted_change'] > threshold, 1, 0)
    
    elif strategy == 'long_short':
        # Long if predicted increase, short if predicted decrease
        df['position'] = np.where(df['predicted_change'] > threshold, 1,
                                  np.where(df['predicted_change'] < -threshold, -1, 0))
    
    # Calculate strategy returns
    df['strategy_return'] = df['position'].shift(1) * df['actual_return']
    
    # Apply transaction costs when position changes
    df['position_change'] = df['position'].diff().abs()
    df['transaction_costs'] = df['position_change'] * transaction_cost
    df['strategy_return_net'] = df['strategy_return'] - df['transaction_costs']
    
    # Cumulative returns
    df['cumulative_market'] = (1 + df['actual_return'].fillna(0)).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_return_net'].fillna(0)).cumprod()
    
    return df


def calculate_metrics(returns_series):
    """
    Calculate performance metrics
    
    Parameters:
    -----------
    returns_series : pd.Series
        Series of returns
    
    Returns:
    --------
    dict with performance metrics
    """
    # Remove NaN
    returns = returns_series.dropna()
    
    # Annualization factor (252 trading days)
    ann_factor = 252
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized return
    n_periods = len(returns)
    n_years = n_periods / ann_factor
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(ann_factor)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': len(returns)
    }


def backtest_strategy(input_path, model_path, ticker, output_path=None, 
                     strategy='long_only', threshold=0.01, transaction_cost=0.0005):
    """
    Backtest a trading strategy using model predictions
    
    Parameters:
    -----------
    input_path : str
        Path to feature data
    model_path : str
        Path to trained model
    ticker : str
        Ticker symbol
    output_path : str
        Path to save results
    strategy : str
        Trading strategy type
    threshold : float
        Signal threshold
    transaction_cost : float
        Transaction cost
    """
    print(f"Backtesting {strategy} strategy for {ticker}...")
    
    # Load data
    df = pd.read_parquet(input_path)
    
    # Load model and get predictions
    model = joblib.load(model_path)
    
    # Prepare features
    X = df.drop('target', axis=1)
    y_actual = df['target']
    
    # Get predictions
    y_pred = model.predict(X)
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate returns
    results = calculate_returns(
        actual_prices=y_actual,
        predicted_prices=y_pred,
        strategy=strategy,
        threshold=threshold,
        transaction_cost=transaction_cost
    )
    
    # Calculate metrics
    market_metrics = calculate_metrics(results['actual_return'])
    strategy_metrics = calculate_metrics(results['strategy_return_net'])
    
    print(f"\n  Market (Buy & Hold) Performance:")
    print(f"    Total Return: {market_metrics['total_return']*100:.2f}%")
    print(f"    Annualized Return: {market_metrics['annualized_return']*100:.2f}%")
    print(f"    Annualized Volatility: {market_metrics['annualized_volatility']*100:.2f}%")
    print(f"    Sharpe Ratio: {market_metrics['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {market_metrics['max_drawdown']*100:.2f}%")
    
    print(f"\n  Strategy Performance:")
    print(f"    Total Return: {strategy_metrics['total_return']*100:.2f}%")
    print(f"    Annualized Return: {strategy_metrics['annualized_return']*100:.2f}%")
    print(f"    Annualized Volatility: {strategy_metrics['annualized_volatility']*100:.2f}%")
    print(f"    Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {strategy_metrics['max_drawdown']*100:.2f}%")
    print(f"    Win Rate: {strategy_metrics['win_rate']*100:.2f}%")
    
    # Calculate excess return
    excess_return = strategy_metrics['annualized_return'] - market_metrics['annualized_return']
    print(f"\n  Excess Return: {excess_return*100:.2f}%")
    
    if output_path:
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results.to_csv(output_path)
        print(f"\n✓ Backtest results saved to {output_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'ticker': [ticker],
            'strategy': [strategy],
            'threshold': [threshold],
            'transaction_cost': [transaction_cost],
            'market_total_return': [market_metrics['total_return']],
            'market_annual_return': [market_metrics['annualized_return']],
            'market_volatility': [market_metrics['annualized_volatility']],
            'market_sharpe': [market_metrics['sharpe_ratio']],
            'market_max_drawdown': [market_metrics['max_drawdown']],
            'strategy_total_return': [strategy_metrics['total_return']],
            'strategy_annual_return': [strategy_metrics['annualized_return']],
            'strategy_volatility': [strategy_metrics['annualized_volatility']],
            'strategy_sharpe': [strategy_metrics['sharpe_ratio']],
            'strategy_max_drawdown': [strategy_metrics['max_drawdown']],
            'strategy_win_rate': [strategy_metrics['win_rate']],
            'excess_return': [excess_return]
        })
        
        metrics_path = output_path.replace('.csv', '_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Metrics saved to {metrics_path}")
    
    return results, market_metrics, strategy_metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategy')
    parser.add_argument('--in', dest='input', type=str, required=True,
                        help='Input parquet file path (with features)')
    parser.add_argument('--model_xgb', type=str,
                        help='Path to XGBoost model')
    parser.add_argument('--model_arima', type=str,
                        help='Path to ARIMA model')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker symbol')
    parser.add_argument('--strategy', type=str, default='long_only',
                        choices=['long_only', 'long_short'],
                        help='Trading strategy')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Signal threshold (default: 0.01 = 1%)')
    parser.add_argument('--cost', type=float, default=0.0005,
                        help='Transaction cost (default: 0.0005 = 0.05%)')
    
    args = parser.parse_args()
    
    # Backtest XGBoost if provided
    if args.model_xgb:
        output_path = f"report/{args.ticker}_backtest_xgb.csv"
        backtest_strategy(
            input_path=args.input,
            model_path=args.model_xgb,
            ticker=args.ticker,
            output_path=output_path,
            strategy=args.strategy,
            threshold=args.threshold,
            transaction_cost=args.cost
        )
    
    # Note: ARIMA backtesting would require different approach
    # as it doesn't use features the same way
    if args.model_arima:
        print("\nNote: ARIMA backtesting requires walk-forward approach.")
        print("Use train_arima.py with --walk_forward flag for ARIMA evaluation.")


if __name__ == "__main__":
    main()
