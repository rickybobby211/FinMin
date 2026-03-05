"""
Run FinGPT news filtering pipeline on a fetched Finnhub JSON file.

Pipeline (same core logic as generate_labels.py):
1) Optional cheap pre-filter (Gemini via OpenRouter) to remove noise
2) Main LLM ranking/selection with sentiment scores

Examples:
  python filter_finnhub_news.py \
    --input-file news_finnhub_2026-02-16_2026-02-24.json \
    --symbol IBM \
    --strategy llm \
    --backend deepseek \
    --model deepseek-reasoner \
    --pre-filter
"""

import argparse
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the exact filtering/ranking functions used in label generation.
from generate_labels import (
    MarketDataManager,
    pre_filter_news,
    rank_news_by_relevance,
    rank_news_with_llm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Finnhub news JSON with FinGPT pre-filter + LLM selection."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=str,
        help="Path to JSON file from fetch_finnhub_news.py",
    )
    parser.add_argument("--symbol", required=True, type=str, help="Ticker symbol, e.g. IBM")
    parser.add_argument(
        "--strategy",
        default="llm",
        choices=["relevant", "llm"],
        help="Selection strategy: relevant (heuristic) or llm (2-step pipeline).",
    )
    parser.add_argument(
        "--k",
        default=5,
        type=int,
        help="Max number of selected items to keep.",
    )
    parser.add_argument(
        "--news-type",
        default="all",
        choices=["all", "company", "market"],
        help="Filter input to company/market/all before selection.",
    )
    parser.add_argument(
        "--backend",
        default="deepseek",
        choices=["openai", "deepseek"],
        help="Main LLM backend (only used when --strategy llm).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-reasoner",
        type=str,
        help="Main LLM model name.",
    )
    parser.add_argument(
        "--pre-filter",
        action="store_true",
        help="Enable cheap pre-filter stage (Gemini via OpenRouter).",
    )
    parser.add_argument(
        "--pre-filter-model",
        default="google/gemini-2.0-flash-lite-001",
        type=str,
        help="Model used for pre-filter stage via OpenRouter.",
    )
    parser.add_argument(
        "--save-prefilter",
        action="store_true",
        help="Save pre-filter output (step 1) to a separate JSON file.",
    )
    parser.add_argument(
        "--prefilter-output-file",
        default="",
        type=str,
        help="Optional path for pre-filter output. Default: <input>_prefiltered.json",
    )
    parser.add_argument(
        "--stock-return",
        default=0.0,
        type=float,
        help="Weekly stock return (decimal, e.g. 0.03). Optional context for main ranking.",
    )
    parser.add_argument(
        "--market-return",
        default=0.0,
        type=float,
        help="Weekly market return (decimal, e.g. 0.01). Optional context for main ranking.",
    )
    parser.add_argument(
        "--market-name",
        default="Market",
        type=str,
        help="Market benchmark name used in prompt context.",
    )
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,
        help="Alpha = stock_return - market_return, used by main ranking prompt.",
    )
    parser.add_argument(
        "--vol-z",
        default=0.0,
        type=float,
        help="Volume Z-score used by main ranking prompt.",
    )
    parser.add_argument(
        "--auto-context",
        action="store_true",
        help="Auto-calculate stock_return/market_return/alpha/vol_z from ticker + period.",
    )
    parser.add_argument(
        "--context-start-date",
        default="",
        type=str,
        help="Optional YYYY-MM-DD start for auto context. Default: inferred from input news dates.",
    )
    parser.add_argument(
        "--context-end-date",
        default="",
        type=str,
        help="Optional YYYY-MM-DD end for auto context. Default: inferred from input news dates.",
    )
    parser.add_argument(
        "--benchmark",
        default="auto",
        type=str,
        help="Benchmark symbol for auto context: auto, SPY, QQQ, or custom ticker.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        type=str,
        help="Optional output JSON path. Default: <input>_filtered.json",
    )
    parser.add_argument(
        "--empty-reason-file",
        default="",
        type=str,
        help="Optional path to save empty-selection explanation text when LLM returns no items.",
    )
    parser.add_argument(
        "--save-step2-debug",
        action="store_true",
        help="Save raw step-2 LLM response and parsed IDs for debugging.",
    )
    parser.add_argument(
        "--incident-dedup",
        action="store_true",
        help="Enable post-selection incident deduplication (disabled by default).",
    )
    parser.add_argument(
        "--step2-debug-file",
        default="",
        type=str,
        help="Optional path for step-2 debug JSON. Default: <output>_step2_debug.json",
    )
    return parser.parse_args()


def build_main_client(backend: str) -> OpenAI:
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for backend=openai.")
        return OpenAI(api_key=api_key)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY for backend=deepseek.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def build_pre_filter_client(enabled: bool) -> OpenAI | None:
    if not enabled:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for --pre-filter.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_news(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in raw if isinstance(item, dict)]


def filter_news_type(news: List[Dict[str, Any]], news_type: str) -> List[Dict[str, Any]]:
    if news_type == "all":
        return news
    return [item for item in news if item.get("news_type") == news_type]


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_filtered.json")


def default_reason_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_empty_reason.txt")


def default_step2_debug_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_step2_debug.json")


def default_prefilter_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_prefiltered.json")


def parse_yyyymmdd(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str[:8], "%Y%m%d")
    except Exception:
        return None


def infer_context_period(news: List[Dict[str, Any]]) -> Tuple[str, str]:
    dates = []
    for item in news:
        item_date = item.get("date", "")
        dt = parse_yyyymmdd(item_date) if isinstance(item_date, str) else None
        if dt is not None:
            dates.append(dt)

    if not dates:
        raise ValueError(
            "Could not infer period from input news dates. "
            "Use --context-start-date and --context-end-date."
        )

    start_date = min(dates).strftime("%Y-%m-%d")
    end_date = max(dates).strftime("%Y-%m-%d")
    return start_date, end_date


def resolve_benchmark(symbol: str, benchmark_arg: str) -> Tuple[str, str]:
    if benchmark_arg and benchmark_arg.lower() != "auto":
        benchmark_symbol = benchmark_arg.upper()
        return benchmark_symbol, benchmark_symbol

    tech_stocks = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", "NFLX"}
    if symbol in tech_stocks:
        return "QQQ", "Nasdaq-100"
    return "SPY", "S&P 500"


def resolve_context(args: argparse.Namespace, symbol: str, source_news: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not args.auto_context:
        return {
            "stock_return": args.stock_return,
            "market_return": args.market_return,
            "market_name": args.market_name,
            "alpha": args.alpha,
            "vol_z": args.vol_z,
            "context_start_date": None,
            "context_end_date": None,
            "benchmark_symbol": None,
        }

    start_date, end_date = infer_context_period(source_news)
    if args.context_start_date:
        start_date = args.context_start_date
    if args.context_end_date:
        end_date = args.context_end_date

    benchmark_symbol, benchmark_name = resolve_benchmark(symbol, args.benchmark)
    manager = MarketDataManager()

    stock_return = manager.get_return(symbol, start_date, end_date)
    market_return = manager.get_return(benchmark_symbol, start_date, end_date)
    alpha = stock_return - market_return
    vol_z_raw = manager.get_volume_z_score(symbol, end_date)
    vol_z = float(vol_z_raw) if vol_z_raw is not None else 0.0

    market_name = args.market_name if args.market_name != "Market" else benchmark_name

    return {
        "stock_return": float(stock_return),
        "market_return": float(market_return),
        "market_name": market_name,
        "alpha": float(alpha),
        "vol_z": float(vol_z),
        "context_start_date": start_date,
        "context_end_date": end_date,
        "benchmark_symbol": benchmark_symbol,
    }


def explain_empty_selection(
    client: OpenAI,
    model: str,
    symbol: str,
    news: List[Dict[str, Any]],
    context: Dict[str, Any],
    k: int,
) -> str:
    news_text_lines = []
    for i, item in enumerate(news):
        date = item.get("date", "")
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        news_text_lines.append(f"ID {i}: [{date}] {headline} - {summary}")
    news_text = "\n".join(news_text_lines)

    system_prompt = (
        f"You are auditing a financial news selector for {symbol}. "
        "The previous selection returned an empty list. "
        "Explain WHY this happened based on the provided inputs. "
        "Be concrete, concise, and actionable."
    )

    user_prompt = (
        f"[MARKET DATA]\n"
        f"- Return: {context['stock_return']*100:.2f}%\n"
        f"- Alpha: {context['alpha']*100:.2f}%\n"
        f"- Volume Z: {context['vol_z']:.2f}\n"
        f"- Market: {context['market_name']}\n\n"
        f"[TASK]\n"
        f"Selection target was up to {k} items.\n"
        "Provide:\n"
        "1) Top 3 reasons empty selection is plausible\n"
        "2) Top 3 prompt/parser risks that could cause false empty output\n"
        "3) If you were forced to choose, give 1-3 candidate IDs\n\n"
        f"[NEWS]\n{news_text}"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return (completion.choices[0].message.content or "").strip()


class _CaptureCompletions:
    def __init__(self, base_completions, capture_store: Dict[str, Any]):
        self._base = base_completions
        self._store = capture_store

    def create(self, *args, **kwargs):
        response = self._base.create(*args, **kwargs)
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None
        self._store["raw_response"] = content
        self._store["request_kwargs"] = kwargs
        return response


class _CaptureChat:
    def __init__(self, base_chat, capture_store: Dict[str, Any]):
        self.completions = _CaptureCompletions(base_chat.completions, capture_store)


class CaptureClient:
    def __init__(self, base_client: OpenAI):
        self._store: Dict[str, Any] = {}
        self.chat = _CaptureChat(base_client.chat, self._store)

    @property
    def raw_response(self) -> str:
        value = self._store.get("raw_response")
        return value if isinstance(value, str) else ""


def extract_step2_ids(raw_response: str) -> List[int]:
    if not raw_response:
        return []
    text = raw_response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        ids: List[int] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "id" in item:
                    ids.append(int(item["id"]))
                elif isinstance(item, int):
                    ids.append(item)
        return ids
    except Exception:
        return [int(x) for x in re.findall(r"\d+", text)]


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\\s]", " ", lowered)
    lowered = re.sub(r"\\s+", " ", lowered).strip()
    return lowered


def _token_set(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "by",
        "with", "from", "as", "is", "are", "was", "were", "be", "this", "that",
        "today", "week", "stock", "shares",
    }
    return {tok for tok in _normalize_text(text).split() if tok not in stopwords and len(tok) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _signal_token_set(item: Dict[str, Any]) -> set[str]:
    # Stronger incident marker: uncommon tokens from headline+summary.
    text = f"{item.get('headline', '')} {item.get('summary', '')}"
    tokens = _token_set(text)
    generic = {
        "market", "markets", "company", "companies", "shares", "stock",
        "today", "week", "price", "business", "update", "report",
    }
    return {tok for tok in tokens if tok not in generic and len(tok) >= 5}


def _same_incident(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    # Exact duplicates first.
    if left.get("id") and left.get("id") == right.get("id"):
        return True
    if left.get("url") and left.get("url") == right.get("url"):
        return True

    # Avoid deduping unrelated stories across different days.
    left_day = str(left.get("date", ""))[:8]
    right_day = str(right.get("date", ""))[:8]
    if left_day and right_day and left_day != right_day:
        return False

    left_head = _normalize_text(str(left.get("headline", "")))
    right_head = _normalize_text(str(right.get("headline", "")))
    if not left_head or not right_head:
        return False

    seq_ratio = SequenceMatcher(None, left_head, right_head).ratio()
    if seq_ratio >= 0.82:
        return True

    left_tokens = _token_set(left_head)
    right_tokens = _token_set(right_head)
    if _jaccard(left_tokens, right_tokens) >= 0.68:
        return True

    # Catch paraphrased duplicates: same day + overlapping uncommon tokens.
    left_signal = _signal_token_set(left)
    right_signal = _signal_token_set(right)
    if len(left_signal & right_signal) >= 2:
        return True

    return False


def dedupe_same_incident(news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for item in news:
        if not isinstance(item, dict):
            continue
        if any(_same_incident(item, kept) for kept in deduped):
            continue
        deduped.append(item)
    return deduped


def main() -> None:
    load_dotenv()
    load_dotenv("../.env")

    args = parse_args()
    symbol = args.symbol.strip().upper()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_news = load_news(input_path)
    source_news = filter_news_type(all_news, args.news_type)
    if not source_news:
        raise ValueError(f"No news after --news-type={args.news_type} filter.")

    output_path = Path(args.output_file) if args.output_file else default_output_path(input_path)
    context = resolve_context(args, symbol, source_news)

    if args.strategy == "relevant":
        selected = rank_news_by_relevance(source_news, end_date=None)[: max(1, args.k)]
        prefiltered_count = len(source_news)
    else:
        main_client = build_main_client(args.backend)
        capture_client = CaptureClient(main_client)
        pre_client = build_pre_filter_client(args.pre_filter)

        target_news = source_news
        if pre_client is not None:
            target_news = pre_filter_news(
                source_news, symbol, pre_client, args.pre_filter_model
            )
            if args.save_prefilter:
                prefilter_path = (
                    Path(args.prefilter_output_file)
                    if args.prefilter_output_file
                    else default_prefilter_path(input_path)
                )
                prefilter_path.write_text(
                    json.dumps(target_news, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Pre-filter output saved to: {prefilter_path}")
        elif args.save_prefilter:
            print("Warning: --save-prefilter was set but --pre-filter is disabled; skipping pre-filter file.")

        prefiltered_count = len(target_news)
        selected = rank_news_with_llm(
            target_news,
            k=max(1, args.k),
            symbol=symbol,
            client=capture_client,
            model=args.model,
            stock_return=context["stock_return"],
            market_return=context["market_return"],
            market_name=context["market_name"],
            alpha=context["alpha"],
            vol_z=context["vol_z"],
            period_start_date=context.get("context_start_date"),
            period_end_date=context.get("context_end_date"),
        )
        step2_raw = capture_client.raw_response
        step2_ids = extract_step2_ids(step2_raw)
        if args.save_step2_debug:
            debug_path = (
                Path(args.step2_debug_file)
                if args.step2_debug_file
                else default_step2_debug_path(output_path)
            )
            debug_payload = {
                "symbol": symbol,
                "model": args.model,
                "strategy": args.strategy,
                "k": max(1, args.k),
                "target_news_count": len(target_news),
                "step2_raw_response": step2_raw,
                "step2_parsed_ids": step2_ids,
                "step2_selected_count": len(selected),
            }
            debug_path.write_text(
                json.dumps(debug_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Step-2 debug saved to: {debug_path}")

        if not selected:
            try:
                empty_reason = explain_empty_selection(
                    client=main_client,
                    model=args.model,
                    symbol=symbol,
                    news=target_news,
                    context=context,
                    k=max(1, args.k),
                )
            except Exception as exc:
                empty_reason = f"Failed to get empty-selection explanation: {exc}"

            reason_path = Path(args.empty_reason_file) if args.empty_reason_file else default_reason_path(output_path)
            reason_path.write_text(empty_reason + "\n", encoding="utf-8")
            raise RuntimeError(
                f"LLM selected 0 items. Explanation saved to: {reason_path}"
            )

    if args.incident_dedup:
        selected = dedupe_same_incident(selected)

    output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    context_info = ""
    if args.auto_context:
        context_info = (
            f", auto_context={context['context_start_date']}..{context['context_end_date']}, "
            f"benchmark={context['benchmark_symbol']}, "
            f"stock_return={context['stock_return']:.4f}, market_return={context['market_return']:.4f}, "
            f"alpha={context['alpha']:.4f}, vol_z={context['vol_z']:.2f}"
        )

    print(
        f"Saved {len(selected)} selected items to: {output_path} "
        f"(input={len(all_news)}, type_filtered={len(source_news)}, "
        f"after_prefilter={prefiltered_count}, strategy={args.strategy}{context_info})"
    )


if __name__ == "__main__":
    main()
