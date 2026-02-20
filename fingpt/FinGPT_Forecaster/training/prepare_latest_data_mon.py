"""
FinGPT Forecaster - Data Preparation Script
============================================
This script fetches the latest market data (stock prices, news, financials)
for training the FinGPT Forecaster model.

Prerequisites:
- Set environment variables: FINNHUB_API_KEY, OPENAI_API_KEY
- pip install finnhub-python yfinance pandas openai

Usage:
    python prepare_latest_data.py --start_date 2024-01-01 --end_date 2024-11-01
"""

import os
import re
import csv
import math
import time
import json
import random
import argparse
import finnhub
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple

# ============================================================================
# MONKEYPATCH: Fix yfinance SSL error by forcing compatible chrome version
# ============================================================================
try:
    import yfinance.data
    from curl_cffi import requests as crequests  # noqa: F401

    _original_session_cls = yfinance.data.requests.Session

    def _patched_session_cls(**kwargs):
        # yfinance defaults to "chrome", but some environments require chrome110.
        if kwargs.get("impersonate") == "chrome":
            kwargs["impersonate"] = "chrome110"
        return _original_session_cls(**kwargs)

    yfinance.data.requests.Session = _patched_session_cls
    print("  [System] Applied yfinance SSL patch (using chrome110)")
except Exception as e:
    print(f"  [System] Could not apply yfinance SSL patch: {e}")

# Load environment variables from .env file
load_dotenv()
load_dotenv("../.env")  # Also check parent directory


# ============================================================================
# CONFIGURATION
# ============================================================================

# DOW 30 Companies (you can customize this list)
DOW_30 = [
    "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE",
    "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW"
]

# Popular tech stocks (alternative list)
TECH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM"
]


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def bin_mapping(ret):
    """Map return percentage to a bin label (e.g., U1 = up 0-1%, D3 = down 2-3%)"""
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    return up_down + (str(integer) if integer <= 5 else '5+')


class WeekWindowStrategy:
    """Strategy pattern for week definitions used by training data prep."""

    def build_return_rows(self, stock_data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_news_window(self, row_start: datetime, row_end: datetime) -> Tuple[str, str]:
        raise NotImplementedError


class FriToFriWeekStrategy(WeekWindowStrategy):
    """Legacy weekly setup: close-to-close calendar weeks."""

    def build_return_rows(self, stock_data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        weekly_data = stock_data[price_col].resample('W').ffill()
        weekly_returns = weekly_data.pct_change()[1:]
        weekly_start_prices = weekly_data[:-1]
        weekly_end_prices = weekly_data[1:]
        weekly_df = pd.DataFrame({
            'Start Date': weekly_start_prices.index,
            'Start Price': weekly_start_prices.values,
            'End Date': weekly_end_prices.index,
            'End Price': weekly_end_prices.values,
            'Weekly Returns': weekly_returns.values
        })
        return weekly_df

    def get_news_window(self, row_start: datetime, row_end: datetime) -> Tuple[str, str]:
        return row_start.strftime('%Y-%m-%d'), row_end.strftime('%Y-%m-%d')


class MonToFriPreOpenStrategy(WeekWindowStrategy):
    """
    New setup: train/predict for Monday-open to Friday-close horizon.
    News window includes previous weekend through week close.
    """

    @staticmethod
    def _previous_saturday(dt: datetime) -> datetime:
        days_since_saturday = (dt.weekday() - 5) % 7
        return dt - timedelta(days=days_since_saturday)

    def build_return_rows(self, stock_data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        if stock_data.empty:
            return pd.DataFrame(columns=['Start Date', 'Start Price', 'End Date', 'End Price', 'Weekly Returns'])

        # Use Open -> Close to match Monday pre-open deployment intent.
        open_col = 'Open' if 'Open' in stock_data.columns else price_col
        close_col = 'Close' if 'Close' in stock_data.columns else price_col

        rows: List[dict] = []
        df = stock_data[[open_col, close_col]].copy()
        df = df.dropna()
        if df.empty:
            return pd.DataFrame(columns=['Start Date', 'Start Price', 'End Date', 'End Price', 'Weekly Returns'])

        # Group by week start (Monday)
        week_start = df.index.to_series().apply(lambda x: x - timedelta(days=x.weekday()))
        grouped = df.groupby(week_start)

        for _, week_df in grouped:
            week_df = week_df.sort_index()
            if len(week_df) < 4:
                # Skip incomplete/holiday-heavy weeks
                continue

            start_idx = week_df.index[0]
            end_idx = week_df.index[-1]
            start_price = float(week_df.iloc[0][open_col])
            end_price = float(week_df.iloc[-1][close_col])
            if start_price == 0:
                continue

            rows.append({
                'Start Date': start_idx,
                'Start Price': start_price,
                'End Date': end_idx,
                'End Price': end_price,
                'Weekly Returns': (end_price - start_price) / start_price
            })

        if not rows:
            return pd.DataFrame(columns=['Start Date', 'Start Price', 'End Date', 'End Price', 'Weekly Returns'])
        return pd.DataFrame(rows)

    def get_news_window(self, row_start: datetime, row_end: datetime) -> Tuple[str, str]:
        from_dt = self._previous_saturday(row_start)
        return from_dt.strftime('%Y-%m-%d'), row_end.strftime('%Y-%m-%d')


def _get_week_strategy(week_mode: str) -> WeekWindowStrategy:
    if week_mode == "mon_fri_preopen":
        return MonToFriPreOpenStrategy()
    return FriToFriWeekStrategy()


def get_returns(stock_symbol: str, start_date: str, end_date: str, week_mode: str = "mon_fri_preopen") -> pd.DataFrame:
    """
    Download stock data and calculate weekly returns.
    
    Args:
        stock_symbol: Ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with weekly price data and returns
    """
    print(f"  Downloading stock data for {stock_symbol}...")
    stock_data = yf.download(
        stock_symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )
    
    if len(stock_data) == 0:
        raise ValueError(f"No stock data found for {stock_symbol}")
    
    # Handle both old ('Adj Close') and new ('Close' with auto_adjust=True) yfinance versions
    if 'Adj Close' in stock_data.columns:
        price_col = 'Adj Close'
    elif 'Close' in stock_data.columns:
        price_col = 'Close'
    else:
        raise ValueError(f"Could not find price column in data for {stock_symbol}")
    
    # Handle multi-level columns (when downloading single stock, yfinance may return flat or multi-level)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data.droplevel(1, axis=1)
        price_col = 'Close' if 'Close' in stock_data.columns else 'Adj Close'
    
    strategy = _get_week_strategy(week_mode)
    weekly_df = strategy.build_return_rows(stock_data, price_col)
    
    weekly_df['Bin Label'] = weekly_df['Weekly Returns'].map(bin_mapping)
    return weekly_df


def get_news(
    finnhub_client,
    symbol: str,
    data: pd.DataFrame,
    rate_limit_delay: float = 0.25,
    week_mode: str = "mon_fri_preopen"
) -> pd.DataFrame:
    """
    Fetch company news for each week in the data.
    
    Args:
        finnhub_client: Finnhub API client
        symbol: Ticker symbol
        data: DataFrame with Start Date and End Date columns
        rate_limit_delay: Delay between API calls (seconds). 
                          Paid tier: 300 calls/min = 0.2s minimum, use 0.25s to be safe.
                          Free tier: 60 calls/min = 1.0s minimum, use 1.1s to be safe.
                          Set higher if still hitting limits.
    
    Returns:
        DataFrame with News column added
    """
    news_list = []
    
    strategy = _get_week_strategy(week_mode)
    for idx, row in data.iterrows():
        start_date, end_date = strategy.get_news_window(row['Start Date'], row['End Date'])
        print(f"    {symbol}: {start_date} - {end_date}")
        
        # Rate limiting - wait before each call
        time.sleep(rate_limit_delay)
        
        # Retry logic for rate limit errors
        max_retries = 3
        weekly_news = []
        
        for attempt in range(max_retries):
            try:
                weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
                weekly_news = [
                    {
                        "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                        "headline": n['headline'],
                        "summary": n['summary'],
                    } for n in weekly_news
                ]
                weekly_news.sort(key=lambda x: x['date'])
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "API limit" in error_str.lower():
                    # Rate limited - wait and retry
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"    Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"    Warning: Failed to fetch news for {symbol}: {e}")
                    break  # Non-rate-limit error, don't retry
                    
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    return data


def get_basics(finnhub_client, symbol: str, data: pd.DataFrame, start_date: str, always: bool = False) -> pd.DataFrame:
    """
    Fetch basic financial metrics for the company.
    
    Args:
        finnhub_client: Finnhub API client
        symbol: Ticker symbol
        data: DataFrame with date information
        start_date: Overall start date of the data
        always: If True, always include the latest available financials
    
    Returns:
        DataFrame with Basics column added
    """
    try:
        basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    except Exception as e:
        print(f"    Warning: Failed to fetch financials for {symbol}: {e}")
        data['Basics'] = [json.dumps({})] * len(data)
        return data
    
    if not basic_financials.get('series') or not basic_financials['series'].get('quarterly'):
        data['Basics'] = [json.dumps({})] * len(data)
        return data
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
            
    for i, row in data.iterrows():
        row_start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = start_date if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < row_start_date) or (last_start_date <= basic['period'] < row_start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
        
    data['Basics'] = final_basics
    return data


def prepare_data_for_company(
    finnhub_client, 
    symbol: str, 
    start_date: str, 
    end_date: str, 
    data_dir: str,
    with_basics: bool = True,
    rate_limit_delay: float = 1.1,
    week_mode: str = "mon_fri_preopen"
) -> pd.DataFrame:
    """
    Prepare complete dataset for a single company.
    
    Args:
        finnhub_client: Finnhub API client
        symbol: Ticker symbol
        start_date: Start date
        end_date: End date
        data_dir: Directory to save the data
        with_basics: Whether to include basic financials
        rate_limit_delay: Delay between API calls in seconds
    
    Returns:
        DataFrame with all data for the company
    """
    print(f"\nProcessing {symbol}...")
    
    try:
        data = get_returns(symbol, start_date, end_date, week_mode=week_mode)
        data = get_news(finnhub_client, symbol, data, rate_limit_delay, week_mode=week_mode)

        mode_suffix = "_monfri" if week_mode == "mon_fri_preopen" else ""
        
        if with_basics:
            data = get_basics(finnhub_client, symbol, data, start_date)
            output_file = f"{data_dir}/{symbol}_{start_date}_{end_date}{mode_suffix}.csv"
        else:
            data['Basics'] = [json.dumps({})] * len(data)
            output_file = f"{data_dir}/{symbol}_{start_date}_{end_date}{mode_suffix}_nobasics.csv"
        
        data.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"  ERROR processing {symbol}: {e}")
        return pd.DataFrame()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for FinGPT Forecaster')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default='dow30', 
                        choices=['dow30', 'tech', 'custom'],
                        help='Which stock list to use')
    parser.add_argument('--custom_symbols', type=str, default='',
                        help='Comma-separated list of custom symbols (use with --symbols custom)')
    parser.add_argument('--with_basics', action='store_true', default=True,
                        help='Include basic financials')
    parser.add_argument('--no_basics', action='store_true',
                        help='Exclude basic financials')
    parser.add_argument('--rate_limit_delay', type=float, default=0.25,
                        help='Delay between Finnhub API calls in seconds (default: 0.25s for paid tier: 300 calls/min)')
    parser.add_argument('--week_mode', type=str, default='mon_fri_preopen',
                        choices=['fri_fri', 'mon_fri_preopen'],
                        help='Week window mode: fri_fri (legacy) or mon_fri_preopen (Monday-open to Friday-close)')
    args = parser.parse_args()
    
    # Get API key
    finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
    if not finnhub_api_key:
        raise ValueError("Please set FINNHUB_API_KEY environment variable")
    
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    
    # Select stock list
    if args.symbols == 'dow30':
        symbols = DOW_30
    elif args.symbols == 'tech':
        symbols = TECH_STOCKS
    else:
        symbols = [s.strip().upper() for s in args.custom_symbols.split(',')]
    
    # Create data directory
    data_dir = f"./raw_data/{args.start_date}_{args.end_date}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    with_basics = not args.no_basics
    
    print("=" * 60)
    print("FinGPT Forecaster - Data Preparation")
    print("=" * 60)
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Symbols: {len(symbols)} companies")
    print(f"Include Financials: {with_basics}")
    print(f"Week Mode: {args.week_mode}")
    print(f"Output Directory: {data_dir}")
    print("=" * 60)
    
    # Process each company
    successful = 0
    failed = []
    
    for symbol in symbols:
        result = prepare_data_for_company(
            finnhub_client, symbol, args.start_date, args.end_date, 
            data_dir, with_basics, args.rate_limit_delay, args.week_mode
        )
        if not result.empty:
            successful += 1
        else:
            failed.append(symbol)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: {successful}/{len(symbols)} companies processed successfully")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Data saved to: {data_dir}")
    print("=" * 60)
    print("\nNext step: Run generate_labels.py to create GPT-4 labels")


if __name__ == "__main__":
    main()

