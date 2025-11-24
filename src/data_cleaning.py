"""
Data Cleaning Module
Cleans and preprocesses stock data for modeling
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def clean_stock_data(input_path, output_path, ticker):
    """
    Clean and preprocess stock data
    
    Steps:
    1. Parse dates and set as index
    2. Sort by date
    3. Remove duplicates
    4. Reindex to business days
    5. Handle missing values
    6. Add basic return columns
    
    Parameters:
    -----------
    input_path : str
        Path to raw CSV file
    output_path : str
        Path to save cleaned parquet file
    ticker : str
        Ticker symbol for logging
    """
    print(f"Cleaning data for {ticker}...")
    
    # Read data
    df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')
    print(f"  Loaded {len(df)} rows")
    
    # Sort by date
    df = df.sort_index()
    
    # Remove duplicate dates
    duplicates = df.index.duplicated()
    if duplicates.sum() > 0:
        print(f"  Removed {duplicates.sum()} duplicate dates")
        df = df[~duplicates]
    
    # Reindex to business days
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(date_range)
    print(f"  Reindexed to {len(df)} business days")
    
    # Check missing values
    missing_before = df.isnull().sum()
    print(f"  Missing values before cleaning:")
    for col in missing_before[missing_before > 0].index:
        print(f"    {col}: {missing_before[col]}")
    
    # Handle missing values
    # For price columns: interpolate small gaps, forward fill larger ones
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in numeric_cols:
        if col in df.columns:
            # Interpolate (limit to 5 consecutive missing values)
            df[col] = df[col].interpolate(method='linear', limit=5)
            
            # Forward fill remaining
            df[col] = df[col].ffill(limit=10)
            
            # Backward fill any remaining at the start
            df[col] = df[col].bfill(limit=10)
    
    # Check missing values after cleaning
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(f"  Missing values after cleaning:")
        for col in missing_after[missing_after > 0].index:
            print(f"    {col}: {missing_after[col]}")
        
        # Drop rows with remaining missing values
        rows_before = len(df)
        df = df.dropna()
        print(f"  Dropped {rows_before - len(df)} rows with missing values")
    
    # Add return columns using Close price
    if 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()
        df['LogReturn'] = np.log(df['Close']).diff()
        print(f"  Added Return and LogReturn columns")
    
    # Drop first row (NaN returns)
    df = df.iloc[1:]
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path)
    print(f"✓ Saved {len(df)} rows to {output_path}")
    
    # Summary statistics
    print(f"\n  Summary for {ticker}:")
    print(f"    Date range: {df.index.min()} to {df.index.max()}")
    print(f"    Total rows: {len(df)}")
    if 'Close' in df.columns:
        print(f"    Avg Close: ${df['Close'].mean():.2f}")
        print(f"    Avg Daily Return: {df['Return'].mean()*100:.4f}%")
        print(f"    Daily Volatility: {df['Return'].std()*100:.4f}%")


def main():
    parser = argparse.ArgumentParser(description='Clean stock data')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker symbol')
    parser.add_argument('--in', dest='input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--out', type=str, required=True,
                        help='Output parquet file path')
    
    args = parser.parse_args()
    
    clean_stock_data(
        input_path=args.input,
        output_path=args.out,
        ticker=args.ticker
    )


if __name__ == "__main__":
    main()
