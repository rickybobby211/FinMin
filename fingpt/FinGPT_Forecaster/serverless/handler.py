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
ADAPTER_ID = os.environ.get("ADAPTER_PATH", "rickson21/qwen2.5-32b-finmin-v1")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32768"))
MIN_COMPLETION_TOKENS = int(os.environ.get("MIN_COMPLETION_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
ANSWER_START_MARKER = "### ANSWER START"
INCLUDE_PROMPT_IN_RESPONSE = os.environ.get("INCLUDE_PROMPT_IN_RESPONSE", "0")
TECH_STOCKS = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", "NFLX"}

B_INST, E_INST = "<|im_start|>", "<|im_end|>"
B_SYS, E_SYS = "system\n", "\n<|im_start|>user\n"

SYSTEM_PROMPT = """You are acting as a professional equity analyst.

You will be given:
- Company profile and basic financials
- Weekly historical news headlines and short descriptions
- Weekly price data
- Quant signals and technical indicators (RSI, MACD, VIX, ATR, volume, mean reversion)

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
            reversion_desc = "Overextended" if abs(dist_sma50) > 15 else "Normal"

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
    
    dates, prices = [], []
    available_dates = stock_data.index.strftime('%Y-%m-%d').tolist()
    
    for step_date in steps[:-1]:
        for i, avail_date in enumerate(available_dates):
            if avail_date >= step_date:
                prices.append(float(stock_data['Close'].iloc[i]))
                dates.append(datetime.strptime(avail_date, "%Y-%m-%d"))
                break
    
    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(float(stock_data['Close'].iloc[-1]))
    
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


def rank_news_by_relevance(news: list, end_date: Optional[str]) -> list:
    """Return news sorted by heuristic relevance (recency + info richness)."""
    valid_news = [n for n in news if isinstance(n, dict)]
    return sorted(valid_news, key=lambda n: _score_news_item(n, end_date), reverse=True)


def rank_news_with_llm(news: list, k: int, symbol: str, client, model: str) -> list:
    """
    Use LLM to select the most impactful news for a specific stock.
    """
    if not news:
        return []
    
    # Prepare news list for LLM
    news_text = ""
    valid_news = [n for n in news if isinstance(n, dict)]
    for i, n in enumerate(valid_news):
        headline = n.get('headline', 'No Headline')
        summary = n.get('summary', 'No Summary')
        date_str = n.get('date', 'Unknown Date') # Renamed to avoid shadowing datetime.date
        news_text += f"ID {i}: [{date_str}] {headline} - {summary}\n"

    system_prompt = f"""You are a financial analyst specializing in {symbol}. 
Your task is to select the {k} most important news items from the provided list that are likely to have the biggest impact on {symbol}'s stock price direction.
Focus on:
1. Earnings releases and financial guidance
2. Major product launches or regulatory approvals
3. Mergers, acquisitions, and strategic partnerships
4. Macroeconomic events directly affecting the sector
5. Significant legal or regulatory actions

Ignore generic market commentary or minor fluff unless it's the only info available.
Return ONLY the IDs of the selected news items as a JSON list, e.g., [0, 4, 12]. Do not include any other text."""

    try:
        print(f"    [DeepSeek] Ranking {len(valid_news)} news items for {symbol}...", flush=True)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": news_text}
            ],
            temperature=0.0,
            max_tokens=100
        )
        response = completion.choices[0].message.content.strip()
        print(f"    [DeepSeek] Response received: {response}", flush=True)
        
        # Extract IDs using regex to be robust
        ids = [int(x) for x in re.findall(r'\d+', response)]
        print(f"    [DeepSeek] Parsed IDs: {ids}", flush=True)
        
        # Return selected news items in order
        selected = [valid_news[i] for i in ids if i < len(valid_news)]
        return selected[:k]
    except Exception as e:
        print(f"    LLM News Selection Failed: {e}. Falling back to relevance scoring.")
        return rank_news_by_relevance(news, None)[:k]


def get_news(symbol, data):
    """Fetch company news from Finnhub and rank with LLM (DeepSeek) if available."""
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

    news_list = []
    for _, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        
        try:
            time.sleep(0.5)  # Rate limit
            weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            
            # Pre-process news for our format
            processed_news = []
            for n in weekly_news:
                # Convert timestamp to YYYYMMDDHHMMSS
                dt_str = datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S')
                processed_news.append({
                    "date": dt_str,
                    "headline": n.get('headline', ''),
                    "summary": n.get('summary', '')
                })

            if llm_client:
                # Use LLM selection
                selected_news = rank_news_with_llm(processed_news, 5, symbol, llm_client, "deepseek-chat")
            else:
                # Use fallback relevance scoring
                selected_news = rank_news_by_relevance(processed_news, end_date)[:5]
            
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
        period = f"{self.context.curday} to {n_weeks_before(self.context.curday, -1)}"
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
    head += f"- MACD: {ta_data.get('macd', 'N/A')}\n"

    dist_sma50_val = ta_data.get("dist_sma50")
    dist_sma50_str = f"{dist_sma50_val:.1f}%" if dist_sma50_val is not None else "N/A"
    head += f"- Mean Reversion: {dist_sma50_str} from SMA50 ({ta_data.get('reversion_desc', 'N/A')})\n\n"
    head += "[NEWS]:\n"
    return head


def construct_prompt(ticker, curday, n_weeks, use_basics=True, use_quant_signals=True):
    """Build the full prompt for the model."""
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]

    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data)
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

def predict(ticker, prediction_date=None, n_weeks=3, use_basics=True, use_quant_signals=True):
    """Generate stock prediction."""
    load_model()  # Ensure model is loaded
    
    if prediction_date is None:
        prediction_date = date.today().strftime("%Y-%m-%d")
    
    config = _build_generation_config()
    prompt = construct_prompt(ticker, prediction_date, n_weeks, use_basics, use_quant_signals)

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
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    answer = _generate_answer(
        inputs,
        max_new_tokens,
        config.min_completion_tokens,
        config.temperature,
        input_len,
    )

    if not _is_complete_response(answer):
        print("Warning: response missing Prediction/Confidence.", flush=True)

    return answer, prompt


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    min_completion_tokens: int
    temperature: float


def _build_generation_config():
    return GenerationConfig(
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        min_completion_tokens=MIN_COMPLETION_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )


def _get_context_limit():
    max_len = getattr(tokenizer, "model_max_length", None)
    if not max_len or max_len > 100000:
        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not max_len or max_len > 100000:
        max_len = 8192
    return int(max_len)


def _generate_answer(inputs, max_new_tokens, min_new_tokens, temperature, input_len):
    with torch.no_grad():
        res = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

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

    return cleaned


def _env_flag(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_complete_response(answer):
    if not answer:
        return False
    lowered = answer.lower()
    return "prediction:" in lowered and "confidence:" in lowered


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
        
        # Generate prediction
        prediction, prompt = predict(
            ticker=ticker.upper(),
            prediction_date=prediction_date,
            n_weeks=n_weeks,
            use_basics=use_basics,
            use_quant_signals=use_quant_signals
        )
        
        response = {
            "ticker": ticker.upper(),
            "date": prediction_date or date.today().strftime("%Y-%m-%d"),
            "prediction": prediction
        }
        if _env_flag(INCLUDE_PROMPT_IN_RESPONSE):
            response["prompt"] = prompt
        return response
    
    except Exception as e:
        return {"error": str(e)}


# Start the serverless worker
print("--- HANDLER STARTUP: Starting RunPod Listener ---", flush=True)
runpod.serverless.start({"handler": handler})
