"""
FinGPT Forecaster - RunPod Serverless Handler
==============================================
This handler processes stock prediction requests on RunPod Serverless.

Deploy with: runpod deploy
"""

import sys
print("--- HANDLER STARTUP: Imports starting ---", flush=True)

import os
import re
import json
import time
import threading
import torch
import runpod
import finnhub
import pandas as pd
import transformers
import yfinance as yf
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Tuple
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("--- HANDLER STARTUP: Imports finished ---", flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
ADAPTER_ID = os.environ.get("ADAPTER_PATH")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32768"))
MIN_COMPLETION_TOKENS = int(os.environ.get("MIN_COMPLETION_TOKENS", "192"))
SAFE_MIN_COMPLETION_TOKENS = int(os.environ.get("SAFE_MIN_COMPLETION_TOKENS", "192"))
ANSWER_START_MARKER = "### ANSWER START"
INCLUDE_PROMPT_IN_RESPONSE = os.environ.get("INCLUDE_PROMPT_IN_RESPONSE", "0")
TECH_STOCKS = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", "NFLX"}
MARKET_NEWS_CATEGORY = os.environ.get("MARKET_NEWS_CATEGORY", "general")
MARKET_NEWS_MAX_PAGES = int(os.environ.get("MARKET_NEWS_MAX_PAGES", "3"))
MARKET_NEWS_MAX_AGE_DAYS = int(os.environ.get("MARKET_NEWS_MAX_AGE_DAYS", "3"))

B_INST, E_INST = "<|im_start|>", "<|im_end|>"
B_SYS, E_SYS = "system\n", "\n<|im_start|>user\n"

SYSTEM_PROMPT = """You are acting as a professional equity analyst.

You will be given:
- Company profile and basic financials
- Weekly historical news headlines and short descriptions
- Weekly price data
- Quant signals and technical indicators (RSI, MACD, SMA200, VIX, ATR, volume, mean reversion)

Your task:
1. Identify key positive developments from the news/financials.
2. Identify key negative developments.
3. Analyze price trend and momentum using the provided quant/technical signals.
4. Provide a next-week price direction prediction (UP or DOWN) with an estimated percentage change.
5. Provide confidence level (0–100%).

Constraints:
- Use ONLY the information given.
- Do NOT reference any future knowledge beyond the cutoff date.
- IF NO NEWS ARE PROVIDED: Be extremely cautious. Do not assume the current trend will continue blindly. Base your prediction more on valuation (P/E) and fundamental metrics (Profitability, Cash Flow) rather than just price momentum. A lack of news often leads to consolidation or sector-correlated movements.
- Treat SMA200 as long-term regime context. Short-term bearish signals against a strong bullish trend structure should usually reduce confidence rather than automatically flipping the base case to DOWN, unless news, flow, or relative performance have clearly deteriorated.

Your answer format must be as follows:

[Answer Start]
### ANSWER START

[Prediction]
Prediction: [UP/DOWN] by [Percentage]%
Confidence: [Percentage]%

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]
Analysis: ..."""

# Global model (loaded once, reused for all requests)
model = None
tokenizer = None
finnhub_client = None


# ============================================================================
# MARKET DATA (CACHED)
# ============================================================================

class MarketDataManager:
    """Singleton cache for market and price data."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MarketDataManager, cls).__new__(cls)
                    cls._instance.data = {}
        return cls._instance

    def get_data(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.data:
            print(f"Downloading market data for {symbol}...", flush=True)
            end_date = datetime.now().strftime("%Y-%m-%d")
            df = yf.download(symbol, start="2020-01-01", end=end_date, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.droplevel(1, axis=1)
                except Exception:
                    pass
            self.data[symbol] = df
        return self.data[symbol]

    def get_return(self, symbol: str, start_date: str, end_date: str) -> float:
        """Calculate return for a specific period."""
        try:
            df = self.get_data(symbol)
            price_col = "Close" if "Close" in df.columns else "Adj Close"
            series = df[price_col]
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            start_idx = series.index.get_indexer([start_dt], method="nearest")[0]
            end_idx = series.index.get_indexer([end_dt], method="nearest")[0]
            start_price = series.iloc[start_idx]
            end_price = series.iloc[end_idx]
            return (end_price - start_price) / start_price
        except Exception:
            return 0.0

    def get_vix_data(self, date_str: str) -> dict:
        """Get VIX data dict."""
        try:
            vix_df = self.get_data("^VIX")
            dt = pd.to_datetime(date_str)
            idx = vix_df.index.get_indexer([dt], method="pad")[0]
            if idx == -1:
                return {}
            vix_val = vix_df.iloc[idx]
            if isinstance(vix_val, pd.Series):
                vix_val = vix_val.item()
            desc = "High Fear" if vix_val > 30 else "Elevated Uncertainty" if vix_val > 20 else "Calm"
            return {"value": vix_val, "desc": desc}
        except Exception:
            return {}

    def get_volume_z_score(self, symbol: str, date_str: str) -> Optional[float]:
        """Calculate Volume Z-Score. Returns None if data is unavailable."""
        try:
            df = self.get_data(symbol)
            dt = pd.to_datetime(date_str)
            price_col = "Close" if "Close" in df.columns else "Adj Close"
            prices = df[price_col]
            if dt not in prices.index:
                idx_loc = prices.index.get_indexer([dt], method="pad")[0]
                if idx_loc == -1:
                    return None
                dt = prices.index[idx_loc]

            if "Volume" in df.columns:
                vol = df["Volume"].loc[:dt]
                if len(vol) >= 20:
                    vol_mean = vol.rolling(window=20).mean().iloc[-1]
                    vol_std = vol.rolling(window=20).std().iloc[-1]
                    current_vol = vol.iloc[-1]
                    return (current_vol - vol_mean) / vol_std if vol_std > 0 else None
            return None
        except Exception:
            return None

    def get_technical_data(self, symbol: str, date_str: str) -> dict:
        """Calculate technical indicators and return as dict."""
        try:
            df = self.get_data(symbol)
            price_col = "Close" if "Close" in df.columns else "Adj Close"
            prices = df[price_col]

            dt = pd.to_datetime(date_str)
            if dt not in prices.index:
                idx_loc = prices.index.get_indexer([dt], method="pad")[0]
                if idx_loc == -1:
                    return {}
                dt = prices.index[idx_loc]

            hist = prices.loc[:dt]
            if len(hist) < 200:
                return {}

            sma50 = hist.rolling(window=50).mean().iloc[-1]
            sma200 = hist.rolling(window=200).mean().iloc[-1]
            current_price = hist.iloc[-1]

            trend = "Bullish" if current_price > sma200 else "Bearish"
            if current_price > sma50 and current_price > sma200:
                trend = "Strong Uptrend"
            elif current_price < sma50 and current_price < sma200:
                trend = "Strong Downtrend"

            delta = hist.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            rsi_desc = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"

            weekly_close = prices.resample("W").last().loc[:dt]
            w_rsi_val = None
            w_rsi_desc = "N/A"
            if len(weekly_close) > 15:
                w_delta = weekly_close.diff()
                w_gain = (w_delta.where(w_delta > 0, 0)).rolling(window=14).mean()
                w_loss = (-w_delta.where(w_delta < 0, 0)).rolling(window=14).mean()
                w_rs = w_gain / w_loss
                w_rsi = 100 - (100 / (1 + w_rs))
                w_rsi_val = w_rsi.iloc[-1]
                w_rsi_desc = "Overbought" if w_rsi_val > 70 else "Oversold" if w_rsi_val < 30 else "Neutral"

            exp1 = hist.ewm(span=12, adjust=False).mean()
            exp2 = hist.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_desc = "Bullish Crossover" if macd.iloc[-1] > signal.iloc[-1] else "Bearish Crossover"

            vol_z_desc = "N/A"
            if "Volume" in df.columns:
                vol = df["Volume"].loc[:dt]
                if len(vol) >= 20:
                    vol_mean = vol.rolling(window=20).mean().iloc[-1]
                    vol_std = vol.rolling(window=20).std().iloc[-1]
                    vol_z = (vol.iloc[-1] - vol_mean) / vol_std if vol_std > 0 else None
                    if vol_z is not None:
                        if vol_z > 2.0:
                            vol_z_desc = f"HUGE (Z-Score: {vol_z:.1f})"
                        elif vol_z > 1.0:
                            vol_z_desc = f"High (Z-Score: {vol_z:.1f})"
                        else:
                            vol_z_desc = f"Normal (Z-Score: {vol_z:.1f})"

            high = df["High"].loc[:dt]
            low = df["Low"].loc[:dt]
            prev_close = hist.shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            atr_desc = "High" if atr > current_price * 0.03 else "Normal"

            dist_sma50 = (current_price - sma50) / sma50 * 100
            dist_sma200 = (current_price - sma200) / sma200 * 100
            reversion_desc = "Overextended" if abs(dist_sma50) > 15 else "Normal"
            if current_price > sma50 > sma200:
                trend_structure = "Bullish Stack"
            elif current_price < sma50 < sma200:
                trend_structure = "Bearish Stack"
            else:
                trend_structure = "Mixed Stack"

            return {
                "rsi_daily": rsi_val,
                "rsi_daily_desc": rsi_desc,
                "rsi_weekly": w_rsi_val,
                "rsi_weekly_desc": w_rsi_desc,
                "trend": trend,
                "macd": macd_desc,
                "vol_z_desc": vol_z_desc,
                "atr": atr,
                "atr_desc": atr_desc,
                "dist_sma50": dist_sma50,
                "dist_sma200": dist_sma200,
                "trend_structure": trend_structure,
                "reversion_desc": reversion_desc,
            }
        except Exception:
            return {}


market_manager = MarketDataManager()


def _get_market_context(symbol: str) -> Tuple[str, str]:
    market_symbol = "QQQ" if symbol.upper() in TECH_STOCKS else "SPY"
    market_name = "Nasdaq-100" if market_symbol == "QQQ" else "S&P 500"
    return market_symbol, market_name

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load model once at cold start."""
    global model, tokenizer, finnhub_client
    
    if model is not None:
        return  # Already loaded
    
    print("Loading model...", flush=True)
    
    try:
        # Get HuggingFace token
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: HF_TOKEN environment variable not set!", flush=True)
            raise ValueError("HF_TOKEN environment variable not set")
            
        if not ADAPTER_ID:
            print("ERROR: ADAPTER_PATH environment variable not set!", flush=True)
            raise ValueError("ADAPTER_PATH environment variable is missing. Please configure ADAPTER_PATH in RunPod environment variables.")
        
        print(f"Using Model ID: {MODEL_ID}", flush=True)
        print(f"Using Adapter ID: {ADAPTER_ID}", flush=True)

        print(f"Transformers version: {transformers.__version__}", flush=True)
        try:
            import bitsandbytes as bnb
            print(f"bitsandbytes version: {bnb.__version__}", flush=True)
        except Exception as bnb_error:
            print(f"bitsandbytes not available: {bnb_error}", flush=True)

        device_map = os.environ.get("DEVICE_MAP")
        if not device_map:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_kwargs = {
            "token": hf_token,
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config

        try:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        except TypeError as type_err:
            msg = (
                "Transformers version does not support 4-bit loading. "
                "Upgrade transformers/bitsandbytes or set "
                "ALLOW_FP16_FALLBACK=1 to attempt fp16 loading."
            )
            if os.environ.get("ALLOW_FP16_FALLBACK") == "1":
                print(f"WARNING: {msg}", flush=True)
                model_kwargs.pop("quantization_config", None)
                model_kwargs.pop("load_in_4bit", None)
                model_kwargs.pop("bnb_4bit_use_double_quant", None)
                model_kwargs.pop("bnb_4bit_quant_type", None)
                model_kwargs.pop("bnb_4bit_compute_dtype", None)
                base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
            else:
                raise RuntimeError(msg) from type_err
        print("Base model loaded.", flush=True)
        
        # Load LoRA adapter
        if "/" in ADAPTER_ID and not os.path.exists(ADAPTER_ID):
            model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=hf_token)
        else:
            model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        
        model = model.eval()
        print("Adapter loaded and merged.", flush=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        print("Tokenizer loaded.", flush=True)
        
        # Initialize Finnhub client
        finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
        if finnhub_api_key:
            finnhub_client = finnhub.Client(api_key=finnhub_api_key)
            print("Finnhub client initialized.", flush=True)
        
        print("Model loaded successfully!", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e


# ============================================================================
# DATA FETCHING
# ============================================================================

def n_weeks_before(date_string, n):
    d = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return d.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):
    """Fetch stock price data."""
    stock_data = yf.download(stock_symbol, steps[0], steps[-1], progress=False)
    if len(stock_data) == 0:
        raise ValueError(f"No stock data found for {stock_symbol}")
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        try:
            stock_data = stock_data.droplevel(1, axis=1)
        except Exception:
            pass

    dates, prices = [], []
    available_dates = stock_data.index.strftime('%Y-%m-%d').tolist()
    
    def _to_float(val):
        if hasattr(val, "iloc"):
            return float(val.iloc[0])
        return float(val)
    
    for step_date in steps[:-1]:
        for i, avail_date in enumerate(available_dates):
            if avail_date >= step_date:
                prices.append(_to_float(stock_data['Close'].iloc[i]))
                dates.append(datetime.strptime(avail_date, "%Y-%m-%d"))
                break
    
    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(_to_float(stock_data['Close'].iloc[-1]))
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })


def _parse_news_date(date_str: str) -> Optional[datetime]:
    """Parse finnhub news date strings into datetime; return None on failure."""
    if not date_str:
        return None
    try:
        # Support both YYYYMMDDHHMMSS and YYYY-MM-DD
        if "-" in date_str:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
    except Exception:
        return None


def _is_spam(item: dict) -> bool:
    """Check if a news item is spam/promotional."""
    summary = item.get("summary", "")
    if not summary:
        return False

    spam_phrases = [
        "Looking for stock market analysis and research with proves results?",
        "Zacks.com offers in-depth financial research",
        "Click here to read my analysis",
        "Click here to see why",
    ]

    for phrase in spam_phrases:
        if phrase in summary:
            return True
    return False


def _score_news_item(item: dict, end_date: Optional[str]) -> tuple:
    """
    Score news for relevance: prefer recent items (closest to end_date) and with richer text.
    Uses Strategy-like scoring so we can swap heuristics easily.
    """
    end_dt = _parse_news_date(end_date) if end_date else None
    news_dt = _parse_news_date(item.get("date", ""))

    # Recency: newer is better; if end_date known, distance to end_date (negative for closer)
    if news_dt and end_dt:
        recency_score = -(abs((end_dt - news_dt).total_seconds()))
    elif news_dt:
        recency_score = news_dt.timestamp()
    else:
        recency_score = float("-inf")

    # Information richness: longer headline+summary treated as more informative
    headline = item.get("headline") or ""
    summary = item.get("summary") or ""
    info_len = len(headline) + len(summary)

    # Keyword boosting: Prefer financial/business keywords to catch major deals
    keywords = ["partnership", "deal", "earnings", "revenue", "profit", "loss", "acquisition", "merger", "upgrade", "downgrade"]
    content = (headline + " " + summary).lower()
    keyword_score = sum(100 for kw in keywords if kw in content)

    return (recency_score, keyword_score + info_len)


def _normalize_news_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _signal_tokens(item: dict) -> set[str]:
    text = f"{item.get('headline', '')} {item.get('summary', '')}"
    tokens = _normalize_news_text(text).split()
    generic = {
        "stock", "stocks", "shares", "market", "markets", "today", "week",
        "company", "business", "update", "report",
    }
    return {tok for tok in tokens if len(tok) >= 5 and tok not in generic}


def _same_incident_light(left: dict, right: dict) -> bool:
    left_day = str(left.get("date", ""))[:8]
    right_day = str(right.get("date", ""))[:8]
    if left_day and right_day and left_day != right_day:
        return False
    shared = _signal_tokens(left) & _signal_tokens(right)
    return len(shared) >= 2


def rank_news_by_relevance(news: list, end_date: Optional[str]) -> list:
    """Return news sorted by heuristic relevance (recency + info richness)."""
    valid_news = [n for n in news if isinstance(n, dict)]
    return sorted(valid_news, key=lambda n: _score_news_item(n, end_date), reverse=True)


def rank_news_with_llm(
    news: list,
    k: int,
    symbol: str,
    client,
    model: str,
    stock_return: float = 0.0,
    market_return: float = 0.0,
    market_name: str = "Market",
    alpha: float = 0.0,
    vol_z: float = 0.0,
    period_start_date: Optional[str] = None,
    period_end_date: Optional[str] = None,
) -> list:
    """
    Use LLM to select the most impactful news for a specific stock.
    """
    if not news:
        return []

    # Prepare news list for LLM
    valid_news = [n for n in news if isinstance(n, dict) and not _is_spam(n)]
    cluster_representatives = []
    cluster_ids = []
    for idx, item in enumerate(valid_news):
        assigned_cluster = None
        for cluster_idx, rep_idx in enumerate(cluster_representatives):
            if _same_incident_light(item, valid_news[rep_idx]):
                assigned_cluster = cluster_idx
                break
        if assigned_cluster is None:
            cluster_representatives.append(idx)
            assigned_cluster = len(cluster_representatives) - 1
        cluster_ids.append(assigned_cluster)

    def _format_date_with_weekday(raw_date: str) -> str:
        try:
            if len(raw_date) >= 14:
                dt = datetime.strptime(raw_date[:14], "%Y%m%d%H%M%S")
            elif len(raw_date) >= 8:
                dt = datetime.strptime(raw_date[:8], "%Y%m%d")
            else:
                return raw_date
            return dt.strftime("%a %Y-%m-%d %H:%M")
        except Exception:
            return raw_date

    news_text = ""
    id_to_idx = {}
    for i, n in enumerate(valid_news):
        headline = n.get("headline", "No Headline")
        summary = n.get("summary", "No Summary")
        date_raw = n.get("date", "Unknown Date")
        date_fmt = _format_date_with_weekday(date_raw)
        source = n.get("source", "Unknown Source")
        news_type = n.get("news_type", "unknown")
        cluster = f"C{cluster_ids[i]}"
        raw_id = n.get("id")
        if isinstance(raw_id, int):
            external_id = str(raw_id)
        else:
            external_id = f"IDX_{i}"
        id_to_idx[external_id] = i
        news_text += (
            f"GLOBAL_ID={external_id} LOCAL_INDEX={i} [{date_fmt}] "
            f"[type={news_type}] [source={source}] [cluster={cluster}] "
            f"{headline} - {summary}\n"
        )

    contexts = []
    if vol_z is not None and vol_z > 2.0:
        contexts.append(f"EXTREME VOLUME (Z-Score: {vol_z:.1f})")
    if alpha < -0.02:
        contexts.append(f"SIGNIFICANT UNDERPERFORMANCE (Alpha: {alpha*100:.2f}%)")
    elif alpha > 0.02:
        contexts.append(f"SIGNIFICANT OUTPERFORMANCE (Alpha: {alpha*100:.2f}%)")

    context_header = " + ".join(contexts) if contexts else "NORMAL MARKET CONDITIONS"
    scenario_lines = [f"CONTEXT: {context_header}"]
    if alpha < -0.02:
        scenario_lines.append(
            "Task: Identify downside catalysts threatening the long-term business model "
            "(disruption/obsolescence/structural pressure), not generic negativity."
        )
    elif alpha > 0.02:
        scenario_lines.append(
            "Task: Identify strong positive catalysts "
            "(upgrades/partnerships/earnings beats/product traction)."
        )
    else:
        scenario_lines.append(
            "Task: If no major catalyst exists, choose representative context news instead of forced narratives."
        )
    if vol_z is not None and vol_z > 2.0:
        scenario_lines.append(
            "Priority: Find the specific trigger event behind the unusual trading volume."
        )
    scenario_instruction = "\n".join(scenario_lines)

    vol_z_str = f"{vol_z:.1f}" if vol_z is not None else "N/A"
    week_scope = (
        f"{period_start_date} to {period_end_date}"
        if period_start_date and period_end_date
        else "the provided period"
    )
    system_prompt = f"""You are a financial analyst selecting news for {symbol}.
You are analyzing news for the week of {week_scope}. Your goal is to explain price movement WITHIN THIS SPECIFIC WEEK.

[MARKET DATA]
- Return: {stock_return*100:.2f}%
- Alpha: {alpha*100:.2f}%
- Volume Z: {vol_z_str}

[INSTRUCTIONS]
{scenario_instruction}

[DIVERSITY & CAUSALITY RULES]
- Goal: explain causality, not headline popularity.
- Deduplicate event clusters: if multiple articles describe the same event, keep only the most information-dense one.
- Prioritize causal chain in this order:
  1) direct company catalysts, 2) precursor/warning signals, 3) sector/macro context.
- For k={k}, use slot budget as default:
  - company-specific: ~60% (about 3/5 when k=5)
  - sector/peer context: ~20% (about 1/5)
  - macro/policy context: ~20% (about 1/5)
- If no strong company catalysts exist, reallocate to sector/macro.
- If strong macro policy shock exists (e.g., tariffs/regulatory/geopolitics) and it can plausibly affect {symbol},
  reserve at least one slot for it.
- Prefer time-ordered evidence: precursor first, then catalyst, then consequence.

[OUTPUT RULES]
- Select exactly {k} items when at least {k} unique incidents exist; otherwise select all unique incidents.
- Never return 2 IDs from the same cluster label Cx.
- Assign a sentiment score (-1 to 1) relative to the stock price impact.
- Return a JSON list of objects, using GLOBAL_ID values from input.
- Include a concise "reason" for each selected item.
- Format strictly: [{{"id": "139187931", "score": -0.9, "reason": "COBOL disruption catalyst"}}, ...]
"""

    try:
        print(f"    [DeepSeek] Ranking {len(valid_news)} news items for {symbol}...", flush=True)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": news_text}
            ],
            temperature=0.0,
            max_tokens=700,
        )
        response = completion.choices[0].message.content.strip()
        print(f"    [DeepSeek] Response received: {response}", flush=True)

        # Parse JSON response
        try:
            # Try parsing as pure JSON first
            if "```json" in response:
                response_clean = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response_clean = response.split("```")[1].split("```")[0].strip()
            else:
                response_clean = response
            selected_items = json.loads(response_clean)

            # Strict parsing: require object entries with id and score.
            result_news = []
            used_ids = set()
            used_clusters = set()
            for item in selected_items:
                if isinstance(item, dict) and "id" in item:
                    score = item.get("score", 0)
                    reason = item.get("reason", "")
                    raw_id = str(item["id"]).strip()
                    idx = None
                    if raw_id in id_to_idx:
                        idx = id_to_idx[raw_id]
                    elif raw_id.isdigit() and raw_id in id_to_idx:
                        idx = id_to_idx[raw_id]
                    elif raw_id.startswith("IDX_") and raw_id[4:].isdigit():
                        local_idx = int(raw_id[4:])
                        if 0 <= local_idx < len(valid_news):
                            idx = local_idx
                    elif raw_id.isdigit():
                        local_idx = int(raw_id)
                        if 0 <= local_idx < len(valid_news):
                            idx = local_idx
                    if idx is None or idx >= len(valid_news):
                        continue
                    score = item.get("score", 0)
                    cluster_id = cluster_ids[idx]
                    if idx in used_ids or cluster_id in used_clusters:
                        continue
                    news_obj = valid_news[idx].copy()
                    news_obj["sentiment_score"] = score
                    if isinstance(reason, str) and reason.strip():
                        news_obj["selection_reason"] = reason.strip()
                    result_news.append(news_obj)
                    used_ids.add(idx)
                    used_clusters.add(cluster_id)

            return result_news[:k]

        except json.JSONDecodeError as e:
            raise RuntimeError("LLM response is not valid JSON in rank_news_with_llm.") from e
    except Exception as e:
        raise RuntimeError(f"LLM ranking failed in rank_news_with_llm: {e}") from e


def get_news(symbol, data, week_mode: str = "fri_fri"):
    """Fetch company + market news from Finnhub and rank with LLM (DeepSeek)."""
    if finnhub_client is None:
        data['News'] = [json.dumps([])] * len(data)
        return data

    # Setup DeepSeek client
    llm_client = None
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        try:
            llm_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            print("    [DeepSeek] Client initialized successfully", flush=True)
        except Exception as e:
            print(f"    [DeepSeek] Failed to init client: {e}", flush=True)
    else:
        print("    [DeepSeek] Warning: DEEPSEEK_API_KEY not found in environment variables", flush=True)

    # Enforce Strategy: LLM-based news ranking is mandatory for predictions.
    # We explicitly disallow falling back to heuristic relevance-only ranking,
    # to keep training/inference behaviour aligned.
    if llm_client is None:
        raise RuntimeError(
            "DeepSeek news ranking (rank_news_with_llm) is required for predictions, "
            "but the client could not be initialized. "
            "Set DEEPSEEK_API_KEY and ensure the DeepSeek API is reachable."
        )

    market_symbol = "QQQ" if symbol in TECH_STOCKS else "SPY"
    market_name = "Nasdaq-100" if market_symbol == "QQQ" else "S&P 500"

    def _fetch_market_news_target_week(start_date: str) -> list:
        """
        Fetch latest market/general news for the target week only.
        No date filtering is applied; we page backward with min_id up to
        MARKET_NEWS_MAX_PAGES and rely on Finnhub's own historical limits.
        Hard stop: never include items older than MARKET_NEWS_MAX_AGE_DAYS.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        prev_sunday = (start_dt - timedelta(days=((start_dt.weekday() + 1) % 7))).date()
        prev_saturday = prev_sunday - timedelta(days=1)
        cutoff_dt = datetime.now() - timedelta(days=max(0, MARKET_NEWS_MAX_AGE_DAYS))
        min_id = None
        seen_ids = set()
        collected = []

        for page_idx in range(max(1, MARKET_NEWS_MAX_PAGES)):
            if page_idx == 0 and min_id is None:
                batch = finnhub_client.general_news(category=MARKET_NEWS_CATEGORY)
            else:
                batch = finnhub_client.general_news(category=MARKET_NEWS_CATEGORY, min_id=min_id)

            if not batch:
                break

            ids_in_batch = []
            oldest_ts = None
            for item in batch:
                item_id = item.get("id")
                if item_id is not None:
                    ids_in_batch.append(item_id)
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)

                ts = item.get("datetime")
                if isinstance(ts, int):
                    dt = datetime.fromtimestamp(ts)
                    if oldest_ts is None or ts < oldest_ts:
                        oldest_ts = ts
                    include_item = dt >= cutoff_dt
                    if week_mode == "mon_fri_preopen":
                        # For Monday pre-open mode, include weekend only (Saturday + Sunday).
                        include_item = prev_saturday <= dt.date() <= prev_sunday

                    if include_item:
                        collected.append(
                            {
                                "news_type": "market",
                                "date": dt.strftime("%Y%m%d%H%M%S"),
                                "headline": item.get("headline", ""),
                                "summary": item.get("summary", ""),
                                "source": item.get("source", ""),
                                "category": MARKET_NEWS_CATEGORY,
                            }
                        )

            if oldest_ts is not None:
                oldest_dt = datetime.fromtimestamp(oldest_ts)
                if week_mode == "mon_fri_preopen":
                    if oldest_dt.date() < prev_saturday:
                        break
                elif oldest_dt < cutoff_dt:
                    break

            if not ids_in_batch:
                break
            next_min_id = min(ids_in_batch) - 1
            if min_id is not None and next_min_id >= min_id:
                break
            min_id = next_min_id

        return collected

    news_list = []
    total_rows = len(data)
    for row_idx, (_, row) in enumerate(data.iterrows()):
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        is_target_week = row_idx == (total_rows - 1)
        
        try:
            time.sleep(0.5)  # Rate limit
            company_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)

            # Strategy: market news endpoint is only used for the most recent target week.
            # Historical context remains company-news based.
            market_news = _fetch_market_news_target_week(start_date) if is_target_week else []

            # Pre-process and merge company + market news
            processed_news = []
            for n in company_news:
                # Convert timestamp to YYYYMMDDHHMMSS
                dt_str = datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S')
                processed_news.append({
                    "news_type": "company",
                    "date": dt_str,
                    "headline": n.get('headline', ''),
                    "summary": n.get('summary', ''),
                    "source": n.get('source', ''),
                })

            processed_news.extend(market_news)

            # Deduplicate by normalized (date, headline) key.
            deduped = []
            seen_keys = set()
            for item in processed_news:
                key = (
                    item.get("date", ""),
                    (item.get("headline", "") or "").strip().lower(),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(item)

            stock_return = (
                (row['End Price'] - row['Start Price']) / row['Start Price']
                if row['Start Price'] else 0.0
            )
            market_return = market_manager.get_return(market_symbol, start_date, end_date)
            alpha = stock_return - market_return
            vol_z_val = market_manager.get_volume_z_score(symbol, end_date)
            vol_z = float(vol_z_val) if vol_z_val is not None else 0.0

            # Use LLM selection
            selected_news = rank_news_with_llm(
                deduped, 5, symbol, llm_client, "deepseek-chat",
                stock_return=stock_return,
                market_return=market_return,
                market_name=market_name,
                alpha=alpha,
                vol_z=vol_z,
                period_start_date=start_date,
                period_end_date=end_date,
            )
            
            # Sort chronologically for the model (oldest to newest)
            selected_news.sort(key=lambda x: x['date'])
            
            weekly_news = selected_news
            
        except Exception as e:
            print(f"News fetch error: {e}")
            weekly_news = []
        
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    return data


def get_company_prompt(symbol):
    """Get company introduction."""
    if finnhub_client is None:
        return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."
    
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        if not profile:
            return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."
        
        template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}.\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}."
        return template.format(**profile)
    except:
        return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."


def get_current_basics(symbol, curday):
    """Fetch basic financials."""
    if finnhub_client is None:
        return None

    try:
        basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
        if not basic_financials.get('series'):
             return None
        
        final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
        
        for metric, value_list in basic_financials['series']['quarterly'].items():
            for value in value_list:
                basic_dict[value['period']].update({metric: value['v']})

        for k, v in basic_dict.items():
            v.update({'period': k})
            basic_list.append(v)
            
        basic_list.sort(key=lambda x: x['period'])
        
        # Find latest basics before curday
        target_basic = None
        for basic in basic_list[::-1]:
            if basic['period'] <= curday:
                target_basic = basic
                break
                
        return target_basic
    except Exception as e:
        print(f"Basics fetch error: {e}")
        return None


@dataclass(frozen=True)
class PromptContext:
    ticker: str
    curday: str
    n_weeks: int
    data: pd.DataFrame
    market_symbol: str
    market_name: str
    use_basics: bool
    use_quant_signals: bool
    week_mode: str


class PromptBuilder:
    """Builder pattern to keep prompt assembly consistent."""
    def __init__(self, context: PromptContext):
        self.context = context
        self.parts = []

    def add_company_intro(self):
        self.parts.append(get_company_prompt(self.context.ticker))
        return self

    def add_history(self):
        for _, row in self.context.data.iterrows():
            if self.context.use_quant_signals:
                quant_block = _build_quant_signals_block(
                    self.context.ticker,
                    row,
                    self.context.market_symbol,
                    self.context.market_name,
                )
                self.parts.append("\n" + quant_block)
                self.parts.append(_format_news_block(row))
            else:
                start_date = row["Start Date"].strftime("%Y-%m-%d")
                end_date = row["End Date"].strftime("%Y-%m-%d")
                term = "increased" if row["End Price"] > row["Start Price"] else "decreased"
                self.parts.append(
                    f"\nFrom {start_date} to {end_date}, {self.context.ticker}'s stock price {term} "
                    f"from {row['Start Price']:.2f} to {row['End Price']:.2f}. "
                    "Company news during this period are listed below:\n\n"
                )
                self.parts.append(_format_news_block(row, no_news_text="No news reported."))
        return self

    def add_basics(self):
        if not self.context.use_basics:
            return self
        basics = get_current_basics(self.context.ticker, self.context.curday)
        if basics:
            basics_str = (
                "Some recent basic financials of {}, reported at {}, are presented below:\n\n"
                "[Basic Financials]:\n\n".format(self.context.ticker, basics["period"])
                + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != "period")
            )
            self.parts.append("\n" + basics_str)
        else:
            self.parts.append("\n[Basic Financials]:\n\nNo basic financial reported.")
        return self

    def add_instruction(self):
        curday_dt = datetime.strptime(self.context.curday, "%Y-%m-%d")
        if self.context.week_mode in ["mon_fri", "mon_fri_preopen"]:
            end_dt = curday_dt + timedelta(days=4)
        else:
            end_dt = curday_dt + timedelta(days=7)
        end_date_str = end_dt.strftime("%Y-%m-%d")
        period = f"{self.context.curday} to {end_date_str}"
        
        instruction = (
            f"Based on all the information before {self.context.curday}, let's first analyze the "
            f"positive developments and potential concerns for {self.context.ticker}. Come up with "
            "2-4 most important factors respectively and keep them concise. Integrate both "
            "quantitative signals and news to explain the movement. "
            f"Then make your prediction for next week ({period}). Provide a summary analysis "
            "to support your prediction. Before writing the analysis, cross-reference the "
            "[QUANT SIGNALS] with the [NEWS]. If your prediction conflicts with the technicals "
            "(e.g., predicting 'Up' when Trend is 'Strong Downtrend'), explain this divergence "
            "logically rather than ignoring the data."
        )
        self.parts.append("\n" + instruction)
        return self

    def build(self):
        return "\n".join(part for part in self.parts if part).strip()


def _format_news_block(row: pd.Series, no_news_text: str = "No relative news reported.") -> str:
    try:
        news = json.loads(row["News"])
    except Exception:
        news = []
    formatted = []
    if isinstance(news, list):
        for n in news[:5]:
            headline = (n.get("headline") or "").strip()
            summary = (n.get("summary") or "").strip()
            if headline and summary:
                formatted.append(f"[Headline]: {headline}\n[Summary]: {summary}\n")
    return "\n".join(formatted) if formatted else no_news_text


def _build_quant_signals_block(ticker: str, row: pd.Series, market_symbol: str, market_name: str) -> str:
    start_date = row["Start Date"].strftime("%Y-%m-%d")
    end_date = row["End Date"].strftime("%Y-%m-%d")
    start_price = float(row["Start Price"])
    end_price = float(row["End Price"])
    term = "increased" if end_price > start_price else "decreased"

    stock_return = row.get("Weekly Returns")
    if stock_return is None:
        stock_return = (end_price - start_price) / start_price if start_price else 0.0
    market_return = market_manager.get_return(market_symbol, start_date, end_date)
    alpha = stock_return - market_return
    alpha_pct = alpha * 100
    alpha_sign = "+" if alpha >= 0 else ""
    stock_pct = abs(stock_return) * 100

    ta_data = market_manager.get_technical_data(ticker, end_date)
    vix_data = market_manager.get_vix_data(end_date)
    vix_str = f"{vix_data.get('value', 0):.2f} ({vix_data.get('desc', 'N/A')})" if vix_data else "N/A"

    head = "[QUANT SIGNALS - STRUCTURAL]:\n"
    head += f"- Price Move: {ticker} {term} by {stock_pct:.2f}% ({start_price:.2f} -> {end_price:.2f})\n"
    head += f"- Alpha vs {market_name}: {alpha_sign}{alpha_pct:.2f}% ({'Outperformed' if alpha >= 0 else 'Underperformed'})\n"
    head += f"- Market Context (VIX): {vix_str}\n"
    head += f"- Volume Status: {ta_data.get('vol_z_desc', 'N/A')}\n"

    atr_val = ta_data.get("atr")
    atr_str = f"{atr_val:.2f}" if atr_val is not None else "N/A"
    head += f"- Volatility (ATR): {atr_str} ({ta_data.get('atr_desc', 'N/A')})\n"

    rsi_weekly_val = ta_data.get("rsi_weekly")
    rsi_weekly_str = f"{rsi_weekly_val:.1f}" if rsi_weekly_val is not None else "N/A"
    head += f"- Weekly RSI (Trend Strength): {rsi_weekly_str} ({ta_data.get('rsi_weekly_desc', 'N/A')})\n\n"

    head += "[TECHNICALS - DAILY]:\n"
    rsi_daily_val = ta_data.get("rsi_daily")
    rsi_daily_str = f"{rsi_daily_val:.1f}" if rsi_daily_val is not None else "N/A"
    head += f"- Daily RSI: {rsi_daily_str} ({ta_data.get('rsi_daily_desc', 'N/A')})\n"
    head += f"- Trend: {ta_data.get('trend', 'N/A')}\n"
    dist_sma200_val = ta_data.get("dist_sma200")
    dist_sma200_str = f"{dist_sma200_val:+.1f}%" if dist_sma200_val is not None else "N/A"
    head += f"- Long-Term Trend: {dist_sma200_str} vs SMA200 ({ta_data.get('trend_structure', 'N/A')})\n"
    head += f"- MACD: {ta_data.get('macd', 'N/A')}\n"

    dist_sma50_val = ta_data.get("dist_sma50")
    dist_sma50_str = f"{dist_sma50_val:.1f}%" if dist_sma50_val is not None else "N/A"
    head += f"- Mean Reversion: {dist_sma50_str} from SMA50 ({ta_data.get('reversion_desc', 'N/A')})\n\n"
    head += "[NEWS]:\n"
    return head


def construct_prompt(ticker, curday, n_weeks, use_basics=True, use_quant_signals=True, week_mode="fri_fri"):
    """Build the full prompt for the model."""
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]

    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data, week_mode=week_mode)
    data["Weekly Returns"] = data.apply(
        lambda row: (row["End Price"] - row["Start Price"]) / row["Start Price"] if row["Start Price"] else 0.0,
        axis=1,
    )

    market_symbol, market_name = _get_market_context(ticker)
    context = PromptContext(
        ticker=ticker,
        curday=curday,
        n_weeks=n_weeks,
        data=data,
        market_symbol=market_symbol,
        market_name=market_name,
        use_basics=use_basics,
        use_quant_signals=use_quant_signals,
        week_mode=week_mode,
    )

    prompt = (
        PromptBuilder(context)
        .add_company_intro()
        .add_history()
        .add_basics()
        .add_instruction()
        .build()
    )

    full_prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST + "assistant\n"
    return full_prompt


# ============================================================================
# INFERENCE
# ============================================================================

def predict(ticker, prediction_date=None, n_weeks=3, use_basics=True, use_quant_signals=True, week_mode="fri_fri", temperature=0.7):
    """Generate stock prediction."""
    load_model()  # Ensure model is loaded
    
    if prediction_date is None:
        prediction_date = date.today().strftime("%Y-%m-%d")
    
    config = _build_generation_config(temperature)
    prompt = construct_prompt(ticker, prediction_date, n_weeks, use_basics, use_quant_signals, week_mode)

    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    input_len = inputs["input_ids"].shape[-1]
    context_limit = _get_context_limit()
    available_tokens = context_limit - input_len - 1
    if available_tokens < config.min_completion_tokens:
        raise ValueError(
            "Prompt too long for a complete response. "
            "Reduce history/news or increase context window."
        )

    max_new_tokens = min(config.max_new_tokens, available_tokens)
    min_completion_tokens = min(config.min_completion_tokens, SAFE_MIN_COMPLETION_TOKENS)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    attempts = [
        (config.temperature, min_completion_tokens),
        (config.temperature, min(min_completion_tokens, 160)),
        (config.temperature, min(min_completion_tokens, 128)),
    ]

    answer = ""
    for attempt_idx, (attempt_temp, attempt_min_tokens) in enumerate(attempts, start=1):
        answer = _generate_answer(
            inputs,
            max_new_tokens,
            attempt_min_tokens,
            attempt_temp,
            input_len,
        )
        if _is_usable_response(answer):
            break
        print(
            f"Warning: generation attempt {attempt_idx} returned incomplete or artifact-heavy output.",
            flush=True,
        )

    if not _is_complete_response(answer):
        print("Warning: response missing Prediction/Confidence after retries.", flush=True)

    return answer, prompt


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    min_completion_tokens: int
    temperature: float


def _build_generation_config(temperature):
    return GenerationConfig(
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        min_completion_tokens=MIN_COMPLETION_TOKENS,
        temperature=float(temperature),
    )


def _get_context_limit():
    max_len = getattr(tokenizer, "model_max_length", None)
    if not max_len or max_len > 100000:
        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not max_len or max_len > 100000:
        max_len = 8192
    return int(max_len)


def _generate_answer(inputs, max_new_tokens, min_new_tokens, temperature, input_len):
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if temperature > 0:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
        )
    else:
        generate_kwargs.update(do_sample=False)

    with torch.no_grad():
        res = model.generate(**generate_kwargs)

    output_ids = res[0]
    answer = _extract_assistant_response(output_ids, input_len)
    torch.cuda.empty_cache()
    return answer


def _extract_assistant_response(output_ids, input_len):
    """
    Chain-of-responsibility: try extraction methods in order.
    1) Decode only generated tokens (preferred).
    2) Decode full output and split on assistant markers (fallback).
    """
    extractors = [
        lambda: _decode_generated_tokens(output_ids, input_len),
        lambda: _decode_by_marker(tokenizer.decode(output_ids, skip_special_tokens=False)),
    ]

    for extract in extractors:
        candidate = _clean_answer(extract())
        if candidate:
            return candidate

    return ""


def _decode_generated_tokens(output_ids, input_len):
    if output_ids is None or input_len is None:
        return ""
    if input_len >= len(output_ids):
        return ""
    return tokenizer.decode(output_ids[input_len:], skip_special_tokens=False)


def _decode_by_marker(text):
    for marker in ("<|im_end|>assistant", "<|im_start|>assistant"):
        if marker in text:
            return text.split(marker)[-1]
    return text


def _clean_answer(text):
    if not text:
        return ""
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    cleaned = text.strip()
    lower = cleaned.lower()

    if ANSWER_START_MARKER.lower() in lower:
        marker_idx = lower.rfind(ANSWER_START_MARKER.lower())
        cleaned = cleaned[marker_idx + len(ANSWER_START_MARKER):].strip()
        lower = cleaned.lower()

    if lower.startswith("system") and "assistant" in lower:
        last_idx = lower.rfind("assistant")
        cleaned = cleaned[last_idx + len("assistant"):].strip()

    cleaned = _truncate_artifact_tail(cleaned)
    return cleaned.strip()


def _truncate_artifact_tail(text):
    if not text:
        return ""

    artifact_patterns = [
        r"<\s*/?\s*tool_call\b[^>\n]*>",
        r"<\s*/?\s*function\b[^>\n]*>",
        r"<\s*/?\s*assistant_response\b[^>\n]*>",
        r"<\s*/?\s*analysis\b[^>\n]*>",
        r"<\s*/?\s*response\b[^>\n]*>",
    ]

    for pattern in artifact_patterns:
        artifact_match = re.search(pattern, text, flags=re.IGNORECASE)
        if artifact_match:
            text = text[:artifact_match.start()]
            break

    return text.strip()


def _env_flag(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_complete_response(answer):
    if not answer:
        return False
    lowered = answer.lower()
    return "prediction:" in lowered and "confidence:" in lowered


def _is_usable_response(answer):
    if not _is_complete_response(answer):
        return False
    if "<" in answer and ">" in answer:
        return False
    if len(answer.strip()) < 40:
        return False
    return True


# ============================================================================
# RUNPOD HANDLER
# ============================================================================

def handler(event):
    """
    RunPod Serverless handler function.
    """
    try:
        input_data = event.get("input", {})
        
        ticker = input_data.get("ticker")
        if not ticker:
            return {"error": "Missing required field: ticker"}
        
        prediction_date = input_data.get("date")
        n_weeks = input_data.get("n_weeks", 3)
        use_basics = input_data.get("use_basics", True)
        use_quant_signals = input_data.get("use_quant_signals", True)
        week_mode = input_data.get("week_mode", "fri_fri")
        
        temperature = input_data.get("temperature")
        if temperature is None:
            return {"error": "Missing required field: temperature"}
        
        # Generate prediction
        prediction, prompt = predict(
            ticker=ticker.upper(),
            prediction_date=prediction_date,
            n_weeks=n_weeks,
            use_basics=use_basics,
            use_quant_signals=use_quant_signals,
            week_mode=week_mode,
            temperature=temperature
        )
        
        response = {
            "ticker": ticker.upper(),
            "date": prediction_date or date.today().strftime("%Y-%m-%d"),
            "prediction": prediction,
            "adapter_used": ADAPTER_ID
        }
        if _env_flag(INCLUDE_PROMPT_IN_RESPONSE):
            response["prompt"] = prompt
        return response
    
    except Exception as e:
        return {"error": str(e)}


# Start the serverless worker
print("--- HANDLER STARTUP: Starting RunPod Listener ---", flush=True)
runpod.serverless.start({"handler": handler})
