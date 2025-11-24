"""
Feature Engineering Module
Creates technical features for machine learning models
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def create_features(input_path, output_path):
    """
    Create features for machine learning models
    
    Features:
    - Lagged prices
    - Returns
    - Moving averages
    - Volatility
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Volume features
    - Target (next day price)
    
    Parameters:
    -----------
    input_path : str
        Path to cleaned parquet file
    output_path : str
        Path to save feature-engineered parquet file
    """
    print(f"Creating features from {input_path}...")
    
    # Read data
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")
    
    # Base features (use Close instead of Adj Close)
    df_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Add Return if not present (for volatility calculations)
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()
    
    # 1. Lagged prices
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'lag_{lag}'] = df['Close'].shift(lag)
    
    # 2. Returns
    for period in [1, 3, 5, 10]:
        df_features[f'pct_change_{period}'] = df['Close'].pct_change(periods=period)
    
    # 3. Moving averages
    for window in [5, 10, 20, 50, 200]:
        df_features[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
        # Price relative to MA
        df_features[f'price_to_ma_{window}'] = df['Close'] / df_features[f'ma_{window}']
    
    # 4. Exponential moving averages
    for span in [12, 26]:
        df_features[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # 5. Volatility (rolling standard deviation)
    for window in [5, 10, 20]:
        df_features[f'volatility_{window}'] = df['Return'].rolling(window=window).std()
    
    # 6. Technical indicators
    
    # RSI (Relative Strength Index)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df_features['rsi_14'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df_features['macd'] = macd.macd()
    df_features['macd_signal'] = macd.macd_signal()
    df_features['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df_features['bb_high'] = bb.bollinger_hband()
    df_features['bb_low'] = bb.bollinger_lband()
    df_features['bb_mid'] = bb.bollinger_mavg()
    df_features['bb_width'] = bb.bollinger_wband()
    df_features['bb_position'] = (df['Close'] - df_features['bb_low']) / (df_features['bb_high'] - df_features['bb_low'])
    
    # 7. Volume features
    df_features['volume_change'] = df['Volume'].pct_change()
    df_features['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    df_features['volume_ratio'] = df['Volume'] / df_features['volume_ma_20']
    
    # 8. Price range features
    df_features['high_low_range'] = df['High'] - df['Low']
    df_features['close_open_diff'] = df['Close'] - df['Open']
    
    # 9. Momentum features
    for period in [5, 10, 20]:
        df_features[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
    
    # 10. Target variable (next day's Close price)
    df_features['target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values
    rows_before = len(df_features)
    df_features = df_features.dropna()
    rows_after = len(df_features)
    print(f"  Dropped {rows_before - rows_after} rows with NaN values")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df_features.to_parquet(output_path)
    print(f"✓ Saved {len(df_features)} rows with {len(df_features.columns)} features to {output_path}")
    
    # Show feature summary
    print(f"\n  Feature Summary:")
    print(f"    Total features: {len(df_features.columns) - 1}")  # excluding target
    print(f"    Date range: {df_features.index.min()} to {df_features.index.max()}")
    print(f"\n  Feature columns:")
    for i, col in enumerate(df_features.columns, 1):
        print(f"    {i}. {col}")


def main():
    parser = argparse.ArgumentParser(description='Create features for ML models')
    parser.add_argument('--in', dest='input', type=str, required=True,
                        help='Input parquet file path (cleaned data)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output parquet file path (with features)')
    
    args = parser.parse_args()
    
    create_features(
        input_path=args.input,
        output_path=args.out
    )


if __name__ == "__main__":
    main()
