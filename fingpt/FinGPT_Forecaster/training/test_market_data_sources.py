"""
Probe Yahoo Finance and Stooq market-data downloads in isolation.

Usage examples:
    python test_market_data_sources.py --symbols AAPL,AMZN,^VIX
    python test_market_data_sources.py --symbols AAPL --start 2023-03-17 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
import requests
import yfinance as yf


def apply_yfinance_ssl_patch() -> None:
    """Match the project workaround used by training scripts."""
    try:
        import yfinance.data

        original_session_cls = yfinance.data.requests.Session

        def patched_session_cls(**kwargs):
            if kwargs.get("impersonate") == "chrome":
                kwargs["impersonate"] = "chrome110"
            return original_session_cls(**kwargs)

        yfinance.data.requests.Session = patched_session_cls
        print("[System] Applied yfinance SSL patch (using chrome110)")
    except Exception as exc:
        print(f"[System] Could not apply yfinance SSL patch: {exc}")


def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index/columns the same way the training pipeline does."""
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


def to_stooq_symbol(symbol: str) -> str:
    """Map Yahoo-style symbols to Stooq format."""
    if symbol == "^VIX":
        return "vix"
    return f"{symbol.lower()}.us"


@dataclass
class ProbeResult:
    provider: str
    symbol: str
    ok: bool
    details: dict[str, Any]


def probe_yfinance(symbol: str, start: str, end: str) -> ProbeResult:
    """Run a single yfinance download probe."""
    details: dict[str, Any] = {"start": start, "end": end}
    try:
        raw_df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
        df = normalize_price_df(raw_df)
        details["raw_shape"] = tuple(raw_df.shape)
        details["raw_columns"] = [str(col) for col in raw_df.columns.tolist()]
        details["normalized_shape"] = tuple(df.shape)
        details["normalized_columns"] = [str(col) for col in df.columns.tolist()]
        if not df.empty:
            details["first_date"] = str(df.index.min().date())
            details["last_date"] = str(df.index.max().date())
            details["sample_close"] = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
        else:
            details["error"] = "normalized dataframe is empty"
        return ProbeResult("yfinance", symbol, not df.empty, details)
    except Exception as exc:
        details["error"] = repr(exc)
        return ProbeResult("yfinance", symbol, False, details)


def probe_stooq(symbol: str, start: str, end: str) -> ProbeResult:
    """Run a single Stooq CSV probe and capture raw HTTP diagnostics."""
    stooq_symbol = to_stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    details: dict[str, Any] = {"url": url, "start": start, "end": end}
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 FinGPT market-data probe"},
        )
        details["status_code"] = response.status_code
        details["content_type"] = response.headers.get("content-type")
        details["body_length"] = len(response.text)
        preview = response.text[:200].replace("\r", "\\r").replace("\n", "\\n")
        details["body_preview"] = preview
        response.raise_for_status()

        body = response.text.strip()
        if not body:
            raise ValueError("empty response body")

        raw_df = pd.read_csv(StringIO(body))
        df = normalize_price_df(raw_df)
        if not df.empty:
            df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

        details["raw_shape"] = tuple(raw_df.shape)
        details["raw_columns"] = [str(col) for col in raw_df.columns.tolist()]
        details["normalized_shape"] = tuple(df.shape)
        details["normalized_columns"] = [str(col) for col in df.columns.tolist()]
        if not df.empty:
            details["first_date"] = str(df.index.min().date())
            details["last_date"] = str(df.index.max().date())
            details["sample_close"] = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
        else:
            details["error"] = "normalized dataframe is empty after date filtering"
        return ProbeResult("stooq", symbol, not df.empty, details)
    except Exception as exc:
        details["error"] = repr(exc)
        return ProbeResult("stooq", symbol, False, details)


def print_result(result: ProbeResult) -> None:
    """Print a readable provider probe summary."""
    status = "OK" if result.ok else "FAIL"
    print(f"\n[{result.provider.upper()}] {result.symbol}: {status}")
    for key, value in result.details.items():
        print(f"  - {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Yahoo Finance and Stooq market data")
    parser.add_argument("--symbols", type=str, default="AAPL,AMZN,^VIX", help="Comma-separated symbol list")
    parser.add_argument("--start", type=str, default="2023-03-17", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-03-31", help="End date YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_yfinance_ssl_patch()
    symbols = [symbol.strip().upper() for symbol in re.split(r"[,\s]+", args.symbols) if symbol.strip()]

    print("=== Market Data Probe ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Range: {args.start} -> {args.end}")

    for symbol in symbols:
        print_result(probe_yfinance(symbol, args.start, args.end))
        print_result(probe_stooq(symbol, args.start, args.end))


if __name__ == "__main__":
    main()
