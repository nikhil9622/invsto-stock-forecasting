"""
ARIMA Model Training Module
Trains ARIMA models for time series forecasting
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_arima_model(input_path, ticker, output_path, test_size=0.2):
    """
    Train ARIMA model with auto parameter selection
    
    Parameters:
    -----------
    input_path : str
        Path to cleaned parquet file
    ticker : str
        Ticker symbol
    output_path : str
        Path to save model
    test_size : float
        Fraction of data for testing
    """
    print(f"Training ARIMA model for {ticker}...")
    
    # Read data
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")
    
    # Use Close price for modeling
    series = df['Close']
    
    # Split into train and test
    split_idx = int(len(series) * (1 - test_size))
    train = series[:split_idx]
    test = series[split_idx:]
    
    print(f"  Train size: {len(train)} ({train.index.min()} to {train.index.max()})")
    print(f"  Test size: {len(test)} ({test.index.min()} to {test.index.max()})")
    
    # Auto ARIMA to find best parameters
    print("\n  Finding optimal ARIMA parameters...")
    model = auto_arima(
        train,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=False,
        d=None,  # Let auto_arima determine differencing
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    
    print(f"\n  Best model: ARIMA{model.order}")
    print(f"  AIC: {model.aic():.2f}")
    
    # Forecast on test set
    print("\n  Forecasting on test set...")
    n_periods = len(test)
    forecast = model.predict(n_periods=n_periods)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = mean_absolute_percentage_error(test, forecast)
    
    print(f"\n  Test Set Performance:")
    print(f"    RMSE: ${rmse:.4f}")
    print(f"    MAE: ${mae:.4f}")
    print(f"    MAPE: {mape:.4f}%")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    print(f"\n✓ Model saved to {output_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': test,
        'predicted': forecast
    }, index=test.index)
    
    pred_path = output_path.replace('.pkl', '_predictions.csv')
    predictions_df.to_csv(pred_path)
    print(f"✓ Predictions saved to {pred_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'ticker': [ticker],
        'model': ['ARIMA'],
        'order': [str(model.order)],
        'aic': [model.aic()],
        'rmse': [rmse],
        'mae': [mae],
        'mape': [mape]
    })
    
    metrics_path = output_path.replace('.pkl', '_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to {metrics_path}")
    
    return model, predictions_df, metrics_df


def walk_forward_validation(input_path, ticker, output_path, initial_train_size=0.7):
    """
    Perform walk-forward validation for more realistic evaluation
    
    Parameters:
    -----------
    input_path : str
        Path to cleaned parquet file
    ticker : str
        Ticker symbol
    output_path : str
        Path to save results
    initial_train_size : float
        Initial training fraction
    """
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation for {ticker}")
    print(f"{'='*60}")
    
    # Read data
    df = pd.read_parquet(input_path)
    series = df['Close']
    
    # Initial split
    split_idx = int(len(series) * initial_train_size)
    
    predictions = []
    actuals = []
    dates = []
    
    print(f"  Initial train size: {split_idx}")
    print(f"  Walk-forward steps: {len(series) - split_idx}")
    
    for i in range(split_idx, len(series)):
        # Train on all data up to current point
        train = series[:i]
        
        # Fit model (using simpler parameters for speed)
        try:
            model = auto_arima(
                train,
                start_p=1, start_q=1,
                max_p=3, max_q=3,
                seasonal=False,
                d=None,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            # Forecast next step
            forecast = model.predict(n_periods=1)[0]
            
            predictions.append(forecast)
            actuals.append(series.iloc[i])
            dates.append(series.index[i])
            
            if (i - split_idx + 1) % 50 == 0:
                print(f"  Progress: {i - split_idx + 1}/{len(series) - split_idx} steps")
                
        except Exception as e:
            print(f"  Error at step {i - split_idx + 1}: {str(e)}")
            continue
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    print(f"\n  Walk-Forward Performance:")
    print(f"    RMSE: ${rmse:.4f}")
    print(f"    MAE: ${mae:.4f}")
    print(f"    MAPE: {mape:.4f}%")
    
    # Save results
    results_df = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions
    }, index=dates)
    
    wf_path = output_path.replace('.pkl', '_walkforward.csv')
    results_df.to_csv(wf_path)
    print(f"\n✓ Walk-forward results saved to {wf_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ARIMA model')
    parser.add_argument('--in', dest='input', type=str, required=True,
                        help='Input parquet file path (cleaned data)')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker symbol')
    parser.add_argument('--out', type=str, required=True,
                        help='Output model file path')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--walk_forward', action='store_true',
                        help='Also perform walk-forward validation')
    
    args = parser.parse_args()
    
    # Train model
    train_arima_model(
        input_path=args.input,
        ticker=args.ticker,
        output_path=args.out,
        test_size=args.test_size
    )
    
    # Walk-forward validation if requested
    if args.walk_forward:
        walk_forward_validation(
            input_path=args.input,
            ticker=args.ticker,
            output_path=args.out
        )


if __name__ == "__main__":
    main()
