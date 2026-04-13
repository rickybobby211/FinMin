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
from io import StringIO
import finnhub
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# MONKEYPATCH: Fix yfinance SSL error by forcing compatible chrome version
# ============================================================================
try:
    import yfinance.data
    from curl_cffi import requests as crequests

    _original_session_cls = yfinance.data.requests.Session

    def _patched_session_cls(**kwargs):
        if kwargs.get('impersonate') == 'chrome':
            kwargs['impersonate'] = 'chrome110'
        return _original_session_cls(**kwargs)

    yfinance.data.requests.Session = _patched_session_cls
    print("    [System] Applied yfinance SSL patch (using chrome110)")
except Exception as e:
    print(f"    [System] Could not apply yfinance SSL patch: {e}")

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

MARKET_DATA_LOOKBACK_DAYS = 400


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def bin_mapping(ret):
    """Map return percentage to a bin label (e.g., U1 = up 0-1%, D3 = down 2-3%)"""
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    return up_down + (str(integer) if integer <= 5 else '5+')


def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded daily price data so downstream calculations stay consistent."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(1, axis=1)
        except Exception:
            pass

    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    elif "Adj Close" in df.columns:
        df = df.dropna(subset=["Adj Close"])

    return df.sort_index()


def market_snapshot_path(data_dir: str, symbol: str) -> Path:
    """Return the per-symbol daily market snapshot path."""
    safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol)
    return Path(data_dir) / "_market_data" / f"{safe_symbol}.csv"


def merge_market_snapshots(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge two daily OHLCV snapshots, keeping the newest duplicate rows."""
    frames = [df for df in (existing, new) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.isna()]
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_market_snapshot(data_dir: str, symbol: str) -> pd.DataFrame:
    """Load a previously persisted daily market snapshot if present."""
    snapshot_path = market_snapshot_path(data_dir, symbol)
    if not snapshot_path.exists():
        return pd.DataFrame()

    try:
        return normalize_price_df(pd.read_csv(snapshot_path))
    except Exception as e:
        print(f"  Warning: Failed to read market snapshot {snapshot_path}: {e}")
        return pd.DataFrame()


def save_market_snapshot(
    data_dir: str,
    symbol: str,
    market_data: pd.DataFrame,
    incremental_dir: str = ""
) -> pd.DataFrame:
    """Persist daily market data locally so label generation can reuse the same snapshot."""
    merged = normalize_price_df(market_data)

    if incremental_dir:
        merged = merge_market_snapshots(load_market_snapshot(incremental_dir, symbol), merged)

    merged = merge_market_snapshots(load_market_snapshot(data_dir, symbol), merged)
    if merged.empty:
        return merged

    snapshot_path = market_snapshot_path(data_dir, symbol)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_df = merged.reset_index().rename(columns={"index": "Date"})
    snapshot_df["Date"] = pd.to_datetime(snapshot_df["Date"]).dt.strftime("%Y-%m-%d")
    snapshot_df.to_csv(snapshot_path, index=False)
    print(f"  Saved market snapshot to {snapshot_path}")
    return merged


def resolve_market_symbol(symbol: str) -> str:
    """Return benchmark ETF used for the symbol's market context."""
    return "QQQ" if symbol in TECH_STOCKS else "SPY"


def market_data_start_date(start_date: str, lookback_days: int = MARKET_DATA_LOOKBACK_DAYS) -> str:
    """Extend market-data fetches backwards so long-window indicators have enough history."""
    return (pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")


def to_stooq_symbol(stock_symbol: str) -> str:
    """Map Yahoo-style symbols to Stooq format."""
    if stock_symbol == "^VIX":
        return "vix"
    return f"{stock_symbol.lower()}.us"


def download_from_stooq(stock_symbol: str) -> pd.DataFrame:
    """Fetch Stooq CSV with explicit HTTP validation for clearer failures."""
    stooq_symbol = to_stooq_symbol(stock_symbol)
    stooq_url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    response = requests.get(
        stooq_url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 FinGPT market-data backfill"},
    )
    response.raise_for_status()

    body = response.text.strip()
    if not body:
        raise ValueError("empty response body")
    if "Date,Open,High,Low,Close,Volume" not in body and "Date,Open,High,Low,Close" not in body:
        sample = body[:120].replace("\n", " ")
        raise ValueError(f"unexpected Stooq response: {sample}")

    return normalize_price_df(pd.read_csv(StringIO(body)))


def fetch_market_data(
    stock_symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 6,
    base_wait_seconds: float = 30.0,
    max_wait_seconds: float = 300.0,
) -> pd.DataFrame:
    """Download normalized daily OHLCV data from Yahoo or Stooq."""
    print(f"  Downloading stock data for {stock_symbol}...")
    stock_data = pd.DataFrame()
    last_error = None

    # Yahoo is best-effort: try once, then continue with Stooq fallback if it fails.
    try:
        stock_data = normalize_price_df(
            yf.download(stock_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        )
        if len(stock_data) == 0:
            last_error = ValueError("No stock data returned")
    except Exception as e:
        last_error = e

    if len(stock_data) == 0:
        print(f"    Yahoo failed for {stock_symbol}: {last_error}. Falling back to Stooq daily prices...")
        for attempt in range(1, max_retries + 1):
            try:
                stooq_df = download_from_stooq(stock_symbol)
                if stooq_df.empty or "Close" not in stooq_df.columns:
                    last_error = ValueError("Stooq returned empty/invalid data")
                else:
                    stooq_df = stooq_df[
                        (stooq_df.index >= pd.Timestamp(start_date))
                        & (stooq_df.index < pd.Timestamp(end_date))
                    ]
                    if stooq_df.empty:
                        last_error = ValueError("Stooq has no rows in requested date range")
                    else:
                        stock_data = stooq_df
                        break
            except Exception as e:
                last_error = e

            if attempt < max_retries:
                wait_seconds = min(base_wait_seconds * (2 ** (attempt - 1)), max_wait_seconds)
                wait_seconds += random.uniform(0, 3)
                print(
                    f"    Stooq fetch failed for {stock_symbol} (attempt {attempt}/{max_retries}): "
                    f"{last_error}. Retrying in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)

    if len(stock_data) == 0:
        raise ValueError(
            f"No stock data found for {stock_symbol} after Yahoo + Stooq retries. "
            f"Last error: {last_error}"
        )

    return stock_data


def get_returns(
    stock_symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 6,
    base_wait_seconds: float = 30.0,
    max_wait_seconds: float = 300.0
) -> pd.DataFrame:
    """
    Download stock data and calculate weekly returns.
    
    Args:
        stock_symbol: Ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with weekly price data and returns
    """
    stock_data = fetch_market_data(
        stock_symbol,
        market_data_start_date(start_date),
        end_date,
        max_retries=max_retries,
        base_wait_seconds=base_wait_seconds,
        max_wait_seconds=max_wait_seconds,
    )
    
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

    weekly_df = weekly_df[weekly_df['Start Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    weekly_df['Bin Label'] = weekly_df['Weekly Returns'].map(bin_mapping)
    return weekly_df, stock_data


def infer_symbols_from_data_dir(data_dir: str) -> list[str]:
    """Infer raw-data symbols from an existing directory."""
    symbols = set()
    for path in Path(data_dir).glob("*.csv"):
        name = path.name
        if name.startswith("_") or "gpt-4" in name or "nobasics" in name:
            continue
        if "_" not in name:
            continue
        symbol = name.split("_", 1)[0].strip().upper()
        if symbol:
            symbols.add(symbol)
    return sorted(symbols)


def expected_market_snapshot_symbols(symbols: list[str], include_company_symbols: bool = True) -> list[str]:
    """Return the full set of snapshot symbols needed for a dataset."""
    snapshot_symbols = set(symbols) if include_company_symbols else set()
    benchmark_symbols = {resolve_market_symbol(symbol) for symbol in symbols}
    snapshot_symbols.update(benchmark_symbols)
    snapshot_symbols.add("^VIX")
    return sorted(snapshot_symbols)


def snapshot_exists(data_dir: str, symbol: str) -> bool:
    """Return True when a local market snapshot already exists for the symbol."""
    snapshot_path = market_snapshot_path(data_dir, symbol)
    return snapshot_path.exists() and snapshot_path.stat().st_size > 0


def snapshot_has_required_coverage(data_dir: str, symbol: str, required_start: str) -> bool:
    """Return True when a snapshot exists and reaches far enough back in history."""
    if not snapshot_exists(data_dir, symbol):
        return False

    try:
        df = load_market_snapshot(data_dir, symbol)
        if df.empty:
            return False
        return df.index.min() <= pd.Timestamp(required_start)
    except Exception:
        return False


def write_market_snapshot_report(
    data_dir: str,
    expected_symbols: list[str],
    successful_symbols: list[str],
    failed_symbols: list[str],
    skipped_symbols: list[str],
) -> dict:
    """Persist a machine-readable and human-readable summary of snapshot coverage."""
    present_symbols = [symbol for symbol in expected_symbols if snapshot_exists(data_dir, symbol)]
    missing_symbols = [symbol for symbol in expected_symbols if symbol not in present_symbols]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_symbols": expected_symbols,
        "present_symbols": present_symbols,
        "missing_symbols": missing_symbols,
        "successful_symbols_this_run": successful_symbols,
        "failed_symbols_this_run": failed_symbols,
        "skipped_existing_symbols_this_run": skipped_symbols,
    }

    report_dir = Path(data_dir) / "_market_data"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_json = report_dir / "backfill_status.json"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    missing_txt = report_dir / "missing_symbols.txt"
    missing_txt.write_text("\n".join(missing_symbols) + ("\n" if missing_symbols else ""), encoding="utf-8")

    return report


def backfill_market_data(
    data_dir: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    incremental_dir: str = "",
    include_company_symbols: bool = True,
) -> dict:
    """Fetch and persist market-data snapshots without rebuilding news/basics files."""
    snapshot_symbols = expected_market_snapshot_symbols(
        symbols,
        include_company_symbols=include_company_symbols,
    )
    successful_symbols = []
    failed_symbols = []
    skipped_symbols = []

    print("\nBackfilling local market snapshots...")
    print(f"  Symbols: {', '.join(snapshot_symbols)}")
    print(f"  Range: {start_date} -> {end_date}")
    required_start = market_data_start_date(start_date)
    print(f"  Market-data lookback start: {required_start}")

    for symbol in snapshot_symbols:
        if snapshot_has_required_coverage(data_dir, symbol, required_start):
            print(f"  Skipping existing snapshot for {symbol}")
            skipped_symbols.append(symbol)
            continue
        if snapshot_exists(data_dir, symbol):
            print(f"  Refreshing existing snapshot for {symbol} to extend history")

        try:
            market_data = fetch_market_data(
                symbol,
                required_start,
                end_date,
                max_retries=2,
                base_wait_seconds=3.0,
                max_wait_seconds=10.0,
            )
            save_market_snapshot(data_dir, symbol, market_data, incremental_dir)
            successful_symbols.append(symbol)
        except Exception as e:
            print(f"  ERROR backfilling market snapshot for {symbol}: {e}")
            failed_symbols.append(symbol)

    report = write_market_snapshot_report(
        data_dir,
        snapshot_symbols,
        successful_symbols,
        failed_symbols,
        skipped_symbols,
    )
    return report


def log_close_to_close_samples(symbol: str, data: pd.DataFrame, sample_weeks: int = 3) -> None:
    """Print a few close-to-close weekly rows per symbol for quick spot checks."""
    if data.empty or sample_weeks <= 0:
        return

    total_rows = len(data)
    sample_weeks = min(sample_weeks, total_rows)

    if sample_weeks == 1:
        sample_indices = [total_rows - 1]
    else:
        # Evenly spread sampled weeks across the full date range.
        sample_indices = sorted(
            {
                round(i * (total_rows - 1) / (sample_weeks - 1))
                for i in range(sample_weeks)
            }
        )

    print(f"  Spot-check close-to-close weeks for {symbol}:")
    for idx in sample_indices:
        row = data.iloc[idx]
        print(
            "    "
            f"{row['Start Date'].strftime('%Y-%m-%d')} -> {row['End Date'].strftime('%Y-%m-%d')} | "
            f"start_close={row['Start Price']:.4f}, "
            f"end_close={row['End Price']:.4f}, "
            f"return={(row['Weekly Returns'] * 100):.3f}% ({row['Bin Label']})"
        )


def get_news(finnhub_client, symbol: str, data: pd.DataFrame, rate_limit_delay: float = 0.25) -> pd.DataFrame:
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
    
    for idx, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(f"    {symbol}: {start_date} - {end_date}")
        
        # Rate limiting - wait before each call
        time.sleep(rate_limit_delay)
        
        # Retry logic for rate limit errors
        max_retries = 3
        weekly_news = []
        
        for attempt in range(max_retries):
            try:
                company_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
                normalized_news = []
                skipped_invalid_datetime = 0

                for n in company_news:
                    raw_ts = n.get("datetime")
                    if raw_ts is None:
                        skipped_invalid_datetime += 1
                        continue

                    try:
                        ts = float(raw_ts)
                    except (TypeError, ValueError):
                        skipped_invalid_datetime += 1
                        continue

                    if ts <= 0:
                        skipped_invalid_datetime += 1
                        continue

                    # Finnhub timestamps are usually seconds, but tolerate millisecond payloads.
                    if ts > 1e12:
                        ts = ts / 1000.0

                    try:
                        parsed_date = datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
                    except (OverflowError, OSError, ValueError):
                        skipped_invalid_datetime += 1
                        continue

                    normalized_news.append(
                        {
                            "news_type": "company",
                            "date": parsed_date,
                            "headline": n.get('headline', ''),
                            "summary": n.get('summary', ''),
                            "source": n.get('source', ''),
                        }
                    )

                if skipped_invalid_datetime:
                    print(
                        f"    {symbol}: skipped {skipped_invalid_datetime} news items "
                        f"with invalid datetime in {start_date} - {end_date}"
                    )
                deduped = []
                seen_keys = set()
                for item in normalized_news:
                    key = (
                        item.get("date", ""),
                        (item.get("headline", "") or "").strip().lower(),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped.append(item)

                weekly_news = sorted(deduped, key=lambda x: x['date'])
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
    sample_weeks: int = 3,
    incremental_dir: str = ""
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
        incremental_dir: Path to existing data directory to incrementally update
    
    Returns:
        DataFrame with all data for the company
    """
    print(f"\nProcessing {symbol}...")
    
    old_df = None
    market_data = pd.DataFrame()
    fetch_start_date = start_date
    
    if incremental_dir:
        inc_path = Path(incremental_dir)
        if with_basics:
            matching_files = [f for f in inc_path.glob(f"{symbol}_*.csv") if "nobasics" not in f.name]
        else:
            matching_files = list(inc_path.glob(f"{symbol}_*_nobasics.csv"))
            
        if matching_files:
            # Use the first match
            try:
                old_df = pd.read_csv(matching_files[0])
                if not old_df.empty and 'End Date' in old_df.columns:
                    old_df['End Date'] = pd.to_datetime(old_df['End Date'])
                    old_df['Start Date'] = pd.to_datetime(old_df['Start Date'])
                    max_end = old_df['End Date'].max()
                    
                    # Ensure we don't start fetching after end_date
                    if max_end >= pd.Timestamp(end_date):
                        print(f"  Existing data is already up to date ({max_end.strftime('%Y-%m-%d')}).")
                        fetch_start_date = end_date
                    else:
                        # Overlap by 10 days to ensure we get the connecting week
                        overlap_start = max_end - pd.Timedelta(days=10)
                        fetch_start_date = overlap_start.strftime('%Y-%m-%d')
                        print(f"  Found existing data up to {max_end.strftime('%Y-%m-%d')}. Fetching new data from {fetch_start_date}...")
            except Exception as e:
                print(f"  Failed to read incremental file {matching_files[0]}: {e}")
                old_df = None

    try:
        # If fetch_start_date >= end_date, skip fetching completely
        if pd.Timestamp(fetch_start_date) >= pd.Timestamp(end_date):
            data = pd.DataFrame()
        else:
            try:
                data, market_data = get_returns(symbol, fetch_start_date, end_date)
            except ValueError as e:
                if "No stock data found" in str(e) and old_df is not None:
                    print(f"  No new prices found for {symbol} since {fetch_start_date}. Using old data.")
                    data = pd.DataFrame()
                else:
                    raise e

        if not market_data.empty or incremental_dir:
            save_market_snapshot(data_dir, symbol, market_data, incremental_dir)
        
        if not data.empty:
            log_close_to_close_samples(symbol, data, sample_weeks)
            data = get_news(finnhub_client, symbol, data, rate_limit_delay)
            
            if with_basics:
                # get_basics requires the overall start_date logic
                data = get_basics(finnhub_client, symbol, data, start_date)
            else:
                data['Basics'] = [json.dumps({})] * len(data)

        if old_df is not None:
            if not data.empty:
                data['End Date'] = pd.to_datetime(data['End Date'])
                data['Start Date'] = pd.to_datetime(data['Start Date'])
                combined = pd.concat([old_df, data], ignore_index=True)
                
                # Deduplicate on End Date, keeping the new data (last)
                combined = combined.drop_duplicates(subset=['End Date'], keep='last')
                combined = combined.sort_values('End Date').reset_index(drop=True)
                data = combined
            else:
                data = old_df
                
        if data.empty:
            print(f"  No data to save for {symbol}.")
            return data
            
        # Format dates as strings for CSV to maintain original format
        data['End Date'] = pd.to_datetime(data['End Date']).dt.strftime('%Y-%m-%d')
        data['Start Date'] = pd.to_datetime(data['Start Date']).dt.strftime('%Y-%m-%d')

        if with_basics:
            output_file = f"{data_dir}/{symbol}_{start_date}_{end_date}.csv"
        else:
            output_file = f"{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv"
        
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
    parser.add_argument('--start_date', type=str, default='', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='', help='End date (YYYY-MM-DD)')
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
    parser.add_argument('--sample_weeks', type=int, default=3,
                        help='Number of close-to-close weekly rows to print per symbol for spot checks')
    parser.add_argument('--incremental_from', type=str, default='',
                        help='Path to existing directory to incrementally update from (e.g. ./raw_data/2023-03-17_2026-03-17)')
    parser.add_argument('--backfill_market_data_dir', type=str, default='',
                        help='Only backfill _market_data snapshots for an existing raw_data directory')
    args = parser.parse_args()

    if args.backfill_market_data_dir:
        data_dir = args.backfill_market_data_dir
        dir_name = Path(data_dir).resolve().name
        if not args.start_date or not args.end_date:
            if "_" not in dir_name:
                raise ValueError(
                    "Please supply --start_date and --end_date when the backfill directory name "
                    "does not follow raw_data/<start>_<end>."
                )
            args.start_date, args.end_date = dir_name.split("_", 1)

        symbols = infer_symbols_from_data_dir(data_dir)
        if not symbols:
            raise ValueError(f"Could not infer any symbols from {data_dir}")

        print("=" * 60)
        print("FinGPT Forecaster - Market Data Backfill")
        print("=" * 60)
        print(f"Date Range: {args.start_date} to {args.end_date}")
        print(f"Raw Data Directory: {data_dir}")
        print(f"Inferred Symbols: {len(symbols)}")
        print(f"Incremental From: {args.incremental_from if args.incremental_from else 'None'}")
        print("=" * 60)

        report = backfill_market_data(
            data_dir,
            symbols,
            args.start_date,
            args.end_date,
            args.incremental_from,
            include_company_symbols=True,
        )

        print("\n" + "=" * 60)
        print(
            f"BACKFILL COMPLETE: {len(report['present_symbols'])}/{len(report['expected_symbols'])} "
            f"snapshots available locally"
        )
        if report["successful_symbols_this_run"]:
            print(f"Saved this run: {', '.join(report['successful_symbols_this_run'])}")
        if report["skipped_existing_symbols_this_run"]:
            print(f"Already present: {', '.join(report['skipped_existing_symbols_this_run'])}")
        if report["failed_symbols_this_run"]:
            print(f"Failed this run: {', '.join(report['failed_symbols_this_run'])}")
        if report["missing_symbols"]:
            print(f"Still missing: {', '.join(report['missing_symbols'])}")
        print(f"Market snapshots saved under: {Path(data_dir) / '_market_data'}")
        print(f"Backfill report: {Path(data_dir) / '_market_data' / 'backfill_status.json'}")
        print("=" * 60)
        return

    if not args.start_date or not args.end_date:
        raise ValueError("Please provide --start_date and --end_date")

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
        symbols = [s.strip().upper() for s in args.custom_symbols.split(',') if s.strip()]

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
    print(f"Output Directory: {data_dir}")
    print(f"Incremental From: {args.incremental_from if args.incremental_from else 'None'}")
    print("=" * 60)

    # Process each company
    successful = 0
    failed = []

    for symbol in symbols:
        result = prepare_data_for_company(
            finnhub_client, symbol, args.start_date, args.end_date,
            data_dir, with_basics, args.rate_limit_delay, args.sample_weeks, args.incremental_from
        )
        if not result.empty:
            successful += 1
        else:
            failed.append(symbol)

    context_report = backfill_market_data(
        data_dir,
        [symbol for symbol in symbols if symbol not in failed],
        args.start_date,
        args.end_date,
        args.incremental_from,
        include_company_symbols=False,
    )

    print("\n" + "=" * 60)
    print(f"COMPLETE: {successful}/{len(symbols)} companies processed successfully")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Context snapshots available locally: {len(context_report['present_symbols'])}/{len(context_report['expected_symbols'])}")
    if context_report["successful_symbols_this_run"]:
        print(f"Context snapshots saved this run: {', '.join(context_report['successful_symbols_this_run'])}")
    if context_report["missing_symbols"]:
        print(f"Context snapshots still missing: {', '.join(context_report['missing_symbols'])}")
    print(f"Data saved to: {data_dir}")
    print("=" * 60)
    print("\nNext step: Run generate_labels.py to create GPT-4 labels")


if __name__ == "__main__":
    main()

