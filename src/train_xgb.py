"""
XGBoost Model Training Module
Trains XGBoost regression models for stock price prediction
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_xgboost_model(input_path, ticker, output_path, test_size=0.2, tune_hyperparams=True):
    """
    Train XGBoost model with optional hyperparameter tuning
    
    Parameters:
    -----------
    input_path : str
        Path to feature-engineered parquet file
    ticker : str
        Ticker symbol
    output_path : str
        Path to save model
    test_size : float
        Fraction of data for testing
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    """
    print(f"Training XGBoost model for {ticker}...")
    
    # Read data
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"  Features: {len(X.columns)}")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")
    
    # Time-aware split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n  Train size: {len(X_train)} ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Test size: {len(X_test)} ({X_test.index.min()} to {X_test.index.max()})")
    
    if tune_hyperparams:
        print("\n  Performing hyperparameter tuning...")
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Base model
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score (RMSE): ${np.sqrt(-grid_search.best_score_):.4f}")
        
    else:
        print("\n  Training with default parameters...")
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    
    print(f"\n  Training Set Performance:")
    print(f"    RMSE: ${train_rmse:.4f}")
    print(f"    MAE: ${train_mae:.4f}")
    print(f"    MAPE: {train_mape:.4f}%")
    
    print(f"\n  Test Set Performance:")
    print(f"    RMSE: ${test_rmse:.4f}")
    print(f"    MAE: ${test_mae:.4f}")
    print(f"    MAPE: {test_mape:.4f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    print(f"\n✓ Model saved to {output_path}")
    
    # Save feature importance
    fi_path = output_path.replace('.pkl', '_feature_importance.csv')
    feature_importance.to_csv(fi_path, index=False)
    print(f"✓ Feature importance saved to {fi_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test
    }, index=X_test.index)
    
    pred_path = output_path.replace('.pkl', '_predictions.csv')
    predictions_df.to_csv(pred_path)
    print(f"✓ Predictions saved to {pred_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'ticker': [ticker],
        'model': ['XGBoost'],
        'train_rmse': [train_rmse],
        'train_mae': [train_mae],
        'train_mape': [train_mape],
        'test_rmse': [test_rmse],
        'test_mae': [test_mae],
        'test_mape': [test_mape]
    })
    
    metrics_path = output_path.replace('.pkl', '_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to {metrics_path}")
    
    return model, predictions_df, metrics_df, feature_importance


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--in', dest='input', type=str, required=True,
                        help='Input parquet file path (with features)')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker symbol')
    parser.add_argument('--out', type=str, required=True,
                        help='Output model file path')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--no_tuning', action='store_true',
                        help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    train_xgboost_model(
        input_path=args.input,
        ticker=args.ticker,
        output_path=args.out,
        test_size=args.test_size,
        tune_hyperparams=not args.no_tuning
    )


if __name__ == "__main__":
    main()
