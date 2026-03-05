"""
Fetch company and/or market news from Finnhub for a user-defined period.

Usage examples:
  python fetch_finnhub_news.py --ticker AAPL --period 2025-01-01:2025-01-31
  python fetch_finnhub_news.py --ticker NVDA --period 2025-01-01_2025-01-31
"""

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import finnhub
from dotenv import load_dotenv

AUTO_MARKET_PAGES_LIMIT = 200


def parse_period(period: str) -> Tuple[str, str]:
    """
    Parse period input as YYYY-MM-DD:YYYY-MM-DD or YYYY-MM-DD_YYYY-MM-DD.
    """
    match = re.match(r"^(\d{4}-\d{2}-\d{2})[:_](\d{4}-\d{2}-\d{2})$", period.strip())
    if not match:
        raise ValueError(
            "Invalid --period format. Use YYYY-MM-DD:YYYY-MM-DD "
            "or YYYY-MM-DD_YYYY-MM-DD."
        )

    start_date, end_date = match.group(1), match.group(2)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError("Start date must be before or equal to end date.")

    return start_date, end_date


def sanitize_period_for_filename(period: str) -> str:
    """
    Keep filename-safe period string.
    """
    return period.replace(":", "_").replace("/", "-")


def _format_ts(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, UTC).strftime("%Y%m%d%H%M%S")


def _within_period(ts: Optional[int], start_dt: datetime, end_dt: datetime) -> bool:
    if ts is None:
        return False
    item_dt = datetime.fromtimestamp(ts, UTC)
    return start_dt <= item_dt <= end_dt


def fetch_company_news(
    finnhub_client: finnhub.Client, ticker: str, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch and normalize Finnhub company news.
    """
    raw_news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
    normalized_news = [
        {
            "news_type": "company",
            "date": _format_ts(item.get("datetime")),
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "id": item.get("id"),
            "related": item.get("related", ""),
        }
        for item in raw_news
    ]
    normalized_news.sort(key=lambda item: item["date"])
    return normalized_news


def fetch_market_news_proxy(
    finnhub_client: finnhub.Client,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """
    Fetch market news using large index ETFs (SPY, QQQ) as proxies for general market news.
    This allows fetching historical market news which the general_news endpoint doesn't support.
    """
    proxies = ["SPY", "QQQ"]
    market_news = []
    
    for proxy in proxies:
        proxy_news = fetch_company_news(finnhub_client, proxy, start_date, end_date)
        # Relabel them as market news
        for item in proxy_news:
            item["news_type"] = "market"
            item["category"] = "general"
            item["related"] = proxy  # Keep track of which proxy caught it
        market_news.extend(proxy_news)
        
    # Deduplicate by ID
    deduped = {item["id"]: item for item in market_news if item.get("id")}
    sorted_news = sorted(deduped.values(), key=lambda x: x["date"])
    
    return sorted_news


def fetch_market_news_general(
    finnhub_client: finnhub.Client,
    category: str,
    start_date: str,
    end_date: str,
    max_pages: Optional[int] = 20,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch market news by category and filter to the requested period.
    Finnhub market news endpoint is latest-based, so we paginate with min_id.
    """
    # Keep period checks in UTC.
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=UTC
    )

    page_limit = max_pages if max_pages is not None else AUTO_MARKET_PAGES_LIMIT

    collected: List[Dict[str, Any]] = []
    seen_ids = set()
    min_id: Optional[int] = None
    stop_reason = "page_limit_reached"
    pages_read = 0

    for page_idx in range(page_limit):
        if page_idx == 0 and min_id is None:
            batch = finnhub_client.general_news(category=category)
        else:
            batch = finnhub_client.general_news(category=category, min_id=min_id)

        if not batch:
            stop_reason = "empty_batch"
            if debug:
                print(
                    f"[market-debug] page={page_idx + 1} empty batch. stop pagination."
                )
            break

        pages_read += 1
        oldest_ts_in_batch: Optional[int] = None
        newest_ts_in_batch: Optional[int] = None
        ids_in_batch: List[int] = []
        matched_in_period = 0

        for item in batch:
            item_id = item.get("id")
            if item_id is not None:
                ids_in_batch.append(item_id)
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

            item_ts = item.get("datetime")
            if isinstance(item_ts, int):
                if oldest_ts_in_batch is None or item_ts < oldest_ts_in_batch:
                    oldest_ts_in_batch = item_ts
                if newest_ts_in_batch is None or item_ts > newest_ts_in_batch:
                    newest_ts_in_batch = item_ts

            if _within_period(item_ts, start_dt, end_dt):
                matched_in_period += 1
                collected.append(
                    {
                        "news_type": "market",
                        "category": category,
                        "date": _format_ts(item_ts),
                        "headline": item.get("headline", ""),
                        "summary": item.get("summary", ""),
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "id": item_id,
                        "related": item.get("related", ""),
                    }
                )

        if debug:
            oldest_txt = (
                datetime.fromtimestamp(oldest_ts_in_batch, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if oldest_ts_in_batch is not None
                else "n/a"
            )
            newest_txt = (
                datetime.fromtimestamp(newest_ts_in_batch, UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if newest_ts_in_batch is not None
                else "n/a"
            )
            print(
                "[market-debug] "
                f"page={page_idx + 1}/{page_limit}, batch={len(batch)}, "
                f"match={matched_in_period}, oldest={oldest_txt}, newest={newest_txt}, "
                f"id_min={min(ids_in_batch) if ids_in_batch else 'n/a'}, "
                f"id_max={max(ids_in_batch) if ids_in_batch else 'n/a'}"
            )

        # Stop when we've paged beyond the requested period.
        if oldest_ts_in_batch is not None:
            oldest_dt = datetime.fromtimestamp(oldest_ts_in_batch, UTC)
            if oldest_dt < start_dt:
                stop_reason = "crossed_start_date_boundary"
                if debug:
                    print(
                        "[market-debug] reached start_date boundary; "
                        "stop pagination."
                    )
                break

        if not ids_in_batch:
            stop_reason = "no_ids_in_batch"
            break

        next_min_id = min(ids_in_batch) - 1
        if min_id is not None and next_min_id >= min_id:
            stop_reason = "non_decreasing_min_id_or_cursor_ignored"
            if debug:
                print(
                    "[market-debug] cursor did not advance; Finnhub may be returning "
                    "same latest batch repeatedly for this account/endpoint."
                )
            break
        min_id = next_min_id

    collected.sort(key=lambda item: item["date"])
    # Final strict guard: keep only items fully within requested period.
    before_count = len(collected)
    collected = [
        item
        for item in collected
        if item.get("date", "") and start_date.replace("-", "") <= item["date"][:8] <= end_date.replace("-", "")
    ]
    removed_out_of_range = before_count - len(collected)
    if removed_out_of_range > 0:
        stop_reason = "out_of_period_filtered"

    first_market_date = collected[0].get("date", "") if collected else ""
    last_market_date = collected[-1].get("date", "") if collected else ""
    meta = {
        "pages_read": pages_read,
        "stop_reason": stop_reason,
        "first_market_date": first_market_date,
        "last_market_date": last_market_date,
        "removed_out_of_range": removed_out_of_range,
    }
    return collected, meta


def save_news(news: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist news to disk as JSON.
    """
    output_path.write_text(json.dumps(news, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Finnhub news for ticker and period."
    )
    parser.add_argument(
        "--ticker",
        required=False,
        type=str,
        help="Ticker symbol, e.g. AAPL (required for company news).",
    )
    parser.add_argument(
        "--period",
        required=True,
        type=str,
        help="Date range: YYYY-MM-DD:YYYY-MM-DD or YYYY-MM-DD_YYYY-MM-DD",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=str,
        help="Directory where output file is written (default: current directory).",
    )
    parser.add_argument(
        "--news-type",
        default="company",
        choices=["company", "market", "both"],
        help="Choose Finnhub endpoint output: company news, market news, or both.",
    )
    parser.add_argument(
        "--news-source",
        dest="news_type_legacy",
        choices=["company", "market", "both"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--market-strategy",
        default="general",
        choices=["proxy", "general"],
        help=(
            "How to fetch market news. 'proxy' uses SPY/QQQ for historical support. "
            "'general' uses the Finnhub general news endpoint (only recent days)."
        ),
    )
    parser.add_argument(
        "--market-category",
        default="general",
        type=str,
        help="Market news category (e.g. general, forex, crypto, merger).",
    )
    parser.add_argument(
        "--max-market-pages",
        default=20,
        type=int,
        help=(
            "Max pagination depth for market news endpoint. "
            "Use 0 or negative for adaptive pagination up to safety cap."
        ),
    )
    parser.add_argument(
        "--market-debug",
        action="store_true",
        help="Print per-page pagination/debug details for market news fetch.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    load_dotenv("../.env")

    args = parse_args()
    news_type = args.news_type_legacy or args.news_type
    ticker = args.ticker.upper().strip() if args.ticker else ""
    start_date, end_date = parse_period(args.period)

    if news_type in ("company", "both") and not ticker:
        raise ValueError("--ticker is required when --news-type is 'company' or 'both'.")

    finnhub_api_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_api_key:
        raise ValueError("Missing FINNHUB_API_KEY environment variable.")

    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    news: List[Dict[str, Any]] = []
    company_count = 0
    market_count = 0
    market_meta: Optional[Dict[str, Any]] = None

    if news_type in ("company", "both"):
        company_news = fetch_company_news(finnhub_client, ticker, start_date, end_date)
        company_count = len(company_news)
        news.extend(company_news)

    if news_type in ("market", "both"):
        if args.market_strategy == "proxy":
            market_news = fetch_market_news_proxy(
                finnhub_client, start_date, end_date
            )
        else:
            market_max_pages = (
                None if args.max_market_pages <= 0 else max(1, args.max_market_pages)
            )
            if args.market_debug and market_max_pages is None:
                print(
                    "[market-debug] adaptive pagination enabled "
                    f"(safety_cap={AUTO_MARKET_PAGES_LIMIT})."
                )
            market_news, market_meta = fetch_market_news_general(
                finnhub_client=finnhub_client,
                category=args.market_category,
                start_date=start_date,
                end_date=end_date,
                max_pages=market_max_pages,
                debug=args.market_debug,
            )
        market_count = len(market_news)
        news.extend(market_news)

    news.sort(key=lambda item: item.get("date", ""))

    period_for_name = sanitize_period_for_filename(args.period)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"news_finnhub_{period_for_name}.json"

    save_news(news, output_file)

    print(
        f"Saved {len(news)} news items to: {output_file} "
        f"(ticker={ticker}, company={company_count}, market={market_count}, "
        f"source={news_type}, period={start_date}..{end_date})"
    )
    if market_meta is not None:
        print(
            "Market summary: "
            f"pages_read={market_meta['pages_read']}, "
            f"first_date={market_meta['first_market_date'] or 'n/a'}, "
            f"last_date={market_meta['last_market_date'] or 'n/a'}, "
            f"stop_reason={market_meta['stop_reason']}, "
            f"filtered_out={market_meta['removed_out_of_range']}"
        )


if __name__ == "__main__":
    main()
