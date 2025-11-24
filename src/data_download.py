"""
Data Download Module
Downloads stock data from Yahoo Finance using yfinance
"""

import argparse
import os
from pathlib import Path
import yfinance as yf
import pandas as pd
from tqdm import tqdm


def download_stock_data(tickers, start_date, end_date, output_dir):
    """
    Download stock data for multiple tickers from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    output_dir : str
        Directory to save the downloaded CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Download all tickers at once (faster with multi-threading)
    try:
        df = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker', 
            threads=True,
            progress=True
        )
        
        # Save each ticker to separate CSV
        for ticker in tqdm(tickers, desc="Saving CSV files"):
            try:
                if len(tickers) == 1:
                    # Single ticker case
                    ticker_df = df
                else:
                    # Multi-ticker case
                    ticker_df = df[ticker]
                
                # Drop rows with all NaN values
                ticker_df = ticker_df.dropna(how='all')
                
                if len(ticker_df) > 0:
                    output_path = os.path.join(output_dir, f"{ticker}.csv")
                    ticker_df.to_csv(output_path)
                    print(f"✓ Saved {ticker}: {len(ticker_df)} rows to {output_path}")
                else:
                    print(f"✗ Warning: {ticker} has no data")
                    
            except Exception as e:
                print(f"✗ Error saving {ticker}: {str(e)}")
                
    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        return
    
    print(f"\n✓ Download complete! Files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download stock data from Yahoo Finance')
    parser.add_argument('--tickers', nargs='+', required=True,
                        help='List of ticker symbols (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--out', type=str, default='data/raw/',
                        help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    download_stock_data(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.out
    )


if __name__ == "__main__":
    main()
