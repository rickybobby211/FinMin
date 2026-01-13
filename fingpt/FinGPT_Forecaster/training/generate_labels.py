"""
FinGPT Forecaster - Label Generation Script (PARALLEL OPTIMIZED)
=================================================================
This script uses LLMs to generate training labels (analysis and predictions)
for the prepared market data.

Supported backends:
- OpenAI (gpt-3.5-turbo, gpt-4, gpt-4o-mini) - requires OPENAI_API_KEY
- DeepSeek (deepseek-chat) - requires DEEPSEEK_API_KEY (cheap & good!)
- Ollama (local, FREE) - requires Ollama running locally

Prerequisites:
- For OpenAI: Set environment variable OPENAI_API_KEY
- For DeepSeek: Set environment variable DEEPSEEK_API_KEY
- For Ollama: Install Ollama and run `ollama pull llama3.1` or `ollama pull mistral`
- Run prepare_latest_data.py first to generate raw data

Usage:
    # DeepSeek - Cheap & Good (recommended) with parallel processing
    python generate_labels.py --data_dir ./raw_data/2024-01-01_2024-11-01 --backend deepseek --parallel 5
    
    # OpenAI option
    python generate_labels.py --data_dir ./raw_data/2024-01-01_2024-11-01 --model gpt-4o-mini --parallel 5
    
    # FREE local option with Ollama
    python generate_labels.py --data_dir ./raw_data/2024-01-01_2024-11-01 --backend ollama --model llama3.1
"""

import os
import re
import csv
import json
import random
from datetime import datetime
import argparse
import finnhub
import pandas as pd
import requests
import threading
import yfinance as yf
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from tqdm import tqdm  # Better progress bars

# ============================================================================
# MONKEYPATCH: Fix yfinance SSL error by forcing compatible chrome version
# ============================================================================
try:
    import yfinance.data
    from curl_cffi import requests as crequests
    
    # Original Session class
    _original_session_cls = yfinance.data.requests.Session
    
    def _patched_session_cls(**kwargs):
        # Force chrome110 if chrome is requested (default in yfinance)
        # chrome110 works on this system, chrome fails with OPENSSL_internal
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

class MarketDataManager:
    """Manages downloading and caching of market index data."""
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
            print(f"    Downloading market data for {symbol}...")
            # Download plenty of history to cover all backtests
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Suppress FutureWarning by setting auto_adjust explicitly
            df = yf.download(symbol, start="2020-01-01", end=end_date, progress=False, auto_adjust=False)
            
            # Handle yfinance columns
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.droplevel(1, axis=1)
                except:
                    pass
            
            self.data[symbol] = df
        return self.data[symbol]

    def get_return(self, symbol: str, start_date: str, end_date: str) -> float:
        """Calculate return for a specific period."""
        try:
            df = self.get_data(symbol)
            price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
            series = df[price_col]
            
            # Find closest available dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get prices (using nearest valid trading days)
            start_idx = series.index.get_indexer([start_dt], method='nearest')[0]
            end_idx = series.index.get_indexer([end_dt], method='nearest')[0]
            
            start_price = series.iloc[start_idx]
            end_price = series.iloc[end_idx]
            
            return (end_price - start_price) / start_price
        except Exception as e:
            # print(f"Warning: Could not calc market return: {e}")
            return 0.0

    def get_volatility_data(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Get volatility data dict."""
        try:
            df = self.get_data(symbol)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            period_df = df.loc[mask]
            
            if period_df.empty:
                return {}
                
            high = period_df['High'].max()
            low = period_df['Low'].min()
            
            range_pct = (high - low) / low * 100
            return {"high": high, "low": low, "pct": range_pct}
        except:
            return {}
            
    def get_vix_data(self, date_str: str) -> dict:
        """Get VIX data dict."""
        try:
            vix_df = self.get_data("^VIX")
            dt = pd.to_datetime(date_str)
            idx = vix_df.index.get_indexer([dt], method='pad')[0]
            if idx == -1: return {}
            
            vix_val = vix_df.iloc[idx]
            if isinstance(vix_val, pd.Series): vix_val = vix_val.item()
            
            desc = "High Fear" if vix_val > 30 else "Elevated Uncertainty" if vix_val > 20 else "Calm"
            return {"value": vix_val, "desc": desc}
        except:
            return {}

    def get_volume_z_score(self, symbol: str, date_str: str) -> Optional[float]:
        """Calculate Volume Z-Score. Returns None if data is unavailable."""
        try:
            df = self.get_data(symbol)
            dt = pd.to_datetime(date_str)
            # Match date logic from technical summary
            price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
            prices = df[price_col]
            if dt not in prices.index:
                idx_loc = prices.index.get_indexer([dt], method='pad')[0]
                if idx_loc == -1: return None
                dt = prices.index[idx_loc]
                
            if 'Volume' in df.columns:
                vol = df['Volume'].loc[:dt]
                if len(vol) >= 20:
                    vol_mean = vol.rolling(window=20).mean().iloc[-1]
                    vol_std = vol.rolling(window=20).std().iloc[-1]
                    current_vol = vol.iloc[-1]
                    return (current_vol - vol_mean) / vol_std if vol_std > 0 else None
            return None
        except:
            return None

    def get_technical_data(self, symbol: str, date_str: str) -> dict:
        """Calculate technical indicators and return as dict."""
        try:
            df = self.get_data(symbol)
            price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
            prices = df[price_col]
            
            dt = pd.to_datetime(date_str)
            if dt not in prices.index:
                idx_loc = prices.index.get_indexer([dt], method='pad')[0]
                if idx_loc == -1: return {}
                dt = prices.index[idx_loc]
            
            hist = prices.loc[:dt]
            if len(hist) < 200: return {}
            
            # SMA & Trend
            sma50 = hist.rolling(window=50).mean().iloc[-1]
            sma200 = hist.rolling(window=200).mean().iloc[-1]
            current_price = hist.iloc[-1]
            
            trend = "Bullish" if current_price > sma200 else "Bearish"
            if current_price > sma50 and current_price > sma200: trend = "Strong Uptrend"
            elif current_price < sma50 and current_price < sma200: trend = "Strong Downtrend"
            
            # RSI (14)
            delta = hist.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            rsi_desc = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            
            # WEEKLY RSI
            weekly_close = prices.resample('W').last().loc[:dt]
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
            
            # MACD
            exp1 = hist.ewm(span=12, adjust=False).mean()
            exp2 = hist.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_desc = "Bullish Crossover" if macd.iloc[-1] > signal.iloc[-1] else "Bearish Crossover"
            
            # Volume Z-Score
            vol_z = None
            vol_z_desc = "N/A"
            if 'Volume' in df.columns:
                vol = df['Volume'].loc[:dt]
                if len(vol) >= 20:
                    vol_mean = vol.rolling(window=20).mean().iloc[-1]
                    vol_std = vol.rolling(window=20).std().iloc[-1]
                    vol_z = (vol.iloc[-1] - vol_mean) / vol_std if vol_std > 0 else None
                    if vol_z is not None:
                        if vol_z > 2.0: vol_z_desc = f"HUGE (Z-Score: {vol_z:.1f})"
                        elif vol_z > 1.0: vol_z_desc = f"High (Z-Score: {vol_z:.1f})"
                        else: vol_z_desc = f"Normal (Z-Score: {vol_z:.1f})"
            
            # ATR
            high = df['High'].loc[:dt]
            low = df['Low'].loc[:dt]
            prev_close = hist.shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            atr_desc = "High" if atr > current_price * 0.03 else "Normal"
            
            # SMA Distance
            dist_sma50 = (current_price - sma50) / sma50 * 100
            reversion_desc = "Overextended" if abs(dist_sma50) > 15 else "Normal"
            
            return {
                "rsi_daily": rsi_val, "rsi_daily_desc": rsi_desc,
                "rsi_weekly": w_rsi_val, "rsi_weekly_desc": w_rsi_desc,
                "trend": trend, "macd": macd_desc,
                "vol_z": vol_z, "vol_z_desc": vol_z_desc,
                "atr": atr, "atr_desc": atr_desc,
                "dist_sma50": dist_sma50, "reversion_desc": reversion_desc
            }
        except Exception as e:
            return {}

market_manager = MarketDataManager()

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]:
...

[Primary Driver]:
Technical | Flow | Narrative | Mixed | Unknown

Follow this analytical hierarchy:
1. Technical Reality: Technical indicators (RSI, MACD, SMA50) define the boundaries of what is possible. Do not predict a rally if the stock is extremely overextended without a massive catalyst.
2. Relative Performance: Always consider Alpha. If the stock is underperforming its index despite good news, look for hidden weaknesses.
3. News Catalyst: Use news to explain the 'Why', but never let news justify a lie about the 'What' (the technical data).
4. Dealing with "No News": If there are no specific company news but the price moved significantly, do NOT just say "absence of news". Instead, explicitly analyze the move as driven by market mechanics (e.g., "sector-level repricing", "institutional flows indicated by Volume Z-Score", "beta-driven momentum", or "technical breakout"). Avoid vague language; ascribe the move to market flows or technical structure if fundamental catalysts are missing.
5. Primary Driver: Conclude your analysis by categorizing the main cause of the move into one of these tags:
   - Technical: Move driven by chart patterns, oversold/bought conditions, or mean reversion.
   - Flow: Move driven by volume, sector rotation, or broad market correlation (beta).
   - Narrative: Move driven by specific company news, earnings, or macro events directly impacting the thesis.
   - Mixed: A combination of strong news and technical setup.
   - Unknown: No clear reason found.
"""


# ============================================================================
# PROMPT GENERATION FUNCTIONS
# ============================================================================

def get_company_prompt(finnhub_client, symbol: str) -> str:
    """Generate company introduction prompt."""
    if finnhub_client is None:
        return f"[Company Introduction]:\\n\\n{symbol} is a publicly traded company."
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        if not profile:
            return f"[Company Introduction]:\\n\\n{symbol} is a publicly traded company."
            
        company_template = (
            "[Company Introduction]:\\n\\n{name} is a leading entity in the {finnhubIndustry} sector. "
            "Incorporated and publicly traded since {ipo}, the company has established its reputation "
            "as one of the key players in the market. As of today, {name} has a market capitalization "
            "of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding."
            "\\n\\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. "
            "As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
        )
        return company_template.format(**profile)
    except Exception as e:
        print(f"    Warning: Could not fetch profile for {symbol}: {e}")
        return f"[Company Introduction]:\\n\\n{symbol} is a publicly traded company."


def get_prompt_by_row(symbol: str, row: pd.Series, market_return: float = 0.0, market_name: str = "Market") -> tuple:
    """Generate prompt components for a single row of data."""
    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    
    # Calculate Alpha
    stock_return = row['Weekly Returns']
    alpha = stock_return - market_return
    alpha_pct = alpha * 100
    alpha_sign = "+" if alpha >= 0 else ""
    stock_pct = abs(stock_return) * 100
    
    # Get Volatility
    vol_data = market_manager.get_volatility_data(symbol, start_date, end_date)
    
    # Get Technicals & VIX
    ta_data = market_manager.get_technical_data(symbol, end_date)
    vix_data = market_manager.get_vix_data(end_date)
    
    vix_str = f"{vix_data.get('value', 0):.2f} ({vix_data.get('desc', 'N/A')})" if vix_data else "N/A"
    
    head = f"[QUANT SIGNALS - STRUCTURAL]:\\n"
    head += f"- Price Move: {symbol} {term} by {stock_pct:.2f}% ({row['Start Price']:.2f} -> {row['End Price']:.2f})\\n"
    head += f"- Alpha vs {market_name}: {alpha_sign}{alpha_pct:.2f}% ({'Outperformed' if alpha >= 0 else 'Underperformed'})\\n"
    head += f"- Market Context (VIX): {vix_str}\\n"
    head += f"- Volume Status: {ta_data.get('vol_z_desc', 'N/A')}\\n"
    
    # ATR - handle None values
    atr_val = ta_data.get('atr')
    atr_str = f"{atr_val:.2f}" if atr_val is not None else "N/A"
    head += f"- Volatility (ATR): {atr_str} ({ta_data.get('atr_desc', 'N/A')})\\n"
    
    # Weekly RSI - handle None values
    rsi_weekly_val = ta_data.get('rsi_weekly')
    rsi_weekly_str = f"{rsi_weekly_val:.1f}" if rsi_weekly_val is not None else "N/A"
    head += f"- Weekly RSI (Trend Strength): {rsi_weekly_str} ({ta_data.get('rsi_weekly_desc', 'N/A')})\\n\\n"
    
    head += f"[TECHNICALS - DAILY]:\\n"
    
    # Daily RSI - handle None values
    rsi_daily_val = ta_data.get('rsi_daily')
    rsi_daily_str = f"{rsi_daily_val:.1f}" if rsi_daily_val is not None else "N/A"
    head += f"- Daily RSI: {rsi_daily_str} ({ta_data.get('rsi_daily_desc', 'N/A')})\\n"
    head += f"- Trend: {ta_data.get('trend', 'N/A')}\\n"
    head += f"- MACD: {ta_data.get('macd', 'N/A')}\\n"
    
    # Mean Reversion - handle None values
    dist_sma50_val = ta_data.get('dist_sma50')
    dist_sma50_str = f"{dist_sma50_val:.1f}%" if dist_sma50_val is not None else "N/A"
    head += f"- Mean Reversion: {dist_sma50_str} from SMA50 ({ta_data.get('reversion_desc', 'N/A')})\\n\\n"
    
    head += "[NEWS]:\\n"
    
    try:
        news_raw = json.loads(row["News"]) if isinstance(row["News"], str) else row["News"]
        if not isinstance(news_raw, list):
            news_raw = []
    except:
        news_raw = []
    
    # Filter and format news
    formatted_news = []
    for n in news_raw:
        if not isinstance(n, dict):
            continue
            
        # Skip spam
        if _is_spam(n):
            continue

        # Check if date exists and is before end_date
        news_date = n.get('date', '')
        if news_date:
            try:
                # Handle both 'YYYYMMDDHHMMSS' and 'YYYY-MM-DD' formats
                if len(news_date) >= 8:
                    date_str = news_date[:8] if len(news_date) >= 8 else news_date
                    if date_str > end_date.replace('-', ''):
                        continue
            except:
                pass  # If date parsing fails, include the news anyway
        
        headline = n.get('headline', '')
        summary = n.get('summary', '')
        
        if headline and summary:
            formatted_news.append(f"[Headline]: {headline}\\n[Summary]: {summary}\\n")
    
    basics = json.loads(row['Basics'])
    if basics:
        basics_str = f"Some recent basic financials of {symbol}, reported at {basics['period']}, are presented below:\\n\\n[Basic Financials]:\\n\\n"
        basics_str += "\\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics_str = "[Basic Financials]:\\n\\nNo basic financial reported."
    
    return head, formatted_news, basics_str, news_raw


def _is_spam(item: dict) -> bool:
    """Check if a news item is spam/promotional."""
    summary = item.get('summary', '')
    if not summary:
        return False
    
    # Common spam patterns
    spam_phrases = [
        "Looking for stock market analysis and research with proves results?",
        "Zacks.com offers in-depth financial research",
        "Click here to read my analysis",
        "Click here to see why"
    ]
    
    for phrase in spam_phrases:
        if phrase in summary:
            return True
            
    return False


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

    # Keyword boosting: Prefer financial/business keywords to catch major deals (e.g. Alibaba/Apple partnership)
    # even if they are shorter or slightly older.
    keywords = ["partnership", "deal", "earnings", "revenue", "profit", "loss", "acquisition", "merger", "upgrade", "downgrade"]
    content = (headline + " " + summary).lower()
    keyword_score = sum(100 for kw in keywords if kw in content)

    return (recency_score, keyword_score + info_len)


def rank_news_by_relevance(news: list, end_date: Optional[str]) -> list:
    """Return news sorted by heuristic relevance (recency + info richness)."""
    valid_news = [n for n in news if isinstance(n, dict)]
    return sorted(valid_news, key=lambda n: _score_news_item(n, end_date), reverse=True)


def rank_news_with_llm(news: list, k: int, symbol: str, client, model: str,
                       stock_return: float = 0.0, market_return: float = 0.0, market_name: str = "Market",
                       alpha: float = 0.0, vol_z: float = 0.0) -> list:
    """
    Use LLM to select the most impactful news for a specific stock.
    Includes price movement context (facit) to help the LLM find explaining news.
    """
    if not news:
        return []
    
    # Prepare news list for LLM
    news_text = ""
    valid_news = [n for n in news if isinstance(n, dict) and not _is_spam(n)]
    
    # [OBSOLETE] Pre-rank 200 limit removed to rely on pre-filtering or LLM context capacity.
    # If list is enormous without pre-filtering, DeepSeek can handle ~64k context.

    for i, n in enumerate(valid_news):
        headline = n.get('headline', 'No Headline')
        summary = n.get('summary', 'No Summary')
        date = n.get('date', 'Unknown Date')
        news_text += f"ID {i}: [{date}] {headline} - {summary}\\n"

    # Define scenario instruction dynamically
    scenario_instruction = ""
    
    # Priority 1: Extreme Volume (Something big happened)
    if vol_z is not None and vol_z > 2.0:
        scenario_instruction = f"""
CRITICAL CONTEXT: TRADING VOLUME WAS EXTREME (Z-Score: {vol_z:.1f}).
Something significant happened. You MUST prioritize news about Earnings, M&A, FDA approvals, or major contracts.
Ignore minor press releases. Focus on the catalyst for this volume."""
    
    # Priority 2: Significant Negative Alpha
    elif alpha < -0.02:
        scenario_instruction = f"""
CONTEXT: The stock SIGNIFICANTLY UNDERPERFORMED the market (Alpha: {alpha*100:.2f}%).
You must prioritize negative company-specific news (downgrades, lawsuits, missed earnings) that explains this drop.
If the summary is generic, score it low."""

    # Priority 3: Significant Positive Alpha
    elif alpha > 0.02:
        scenario_instruction = f"""
CONTEXT: The stock SIGNIFICANTLY OUTPERFORMED the market (Alpha: {alpha*100:.2f}%).
You must prioritize positive company-specific catalysts (upgrades, partnerships, beats)."""

    # Priority 4: Normal/Boring week
    else:
        scenario_instruction = """
CONTEXT: The stock moved in line with the market with normal volume.
Do not try to force a narrative if no major news exists. Pick representative news items that reflect the general sentiment.
It is acceptable to pick fewer items if nothing is relevant."""

    vol_z_str = f"{vol_z:.1f}" if vol_z is not None else "N/A"
    system_prompt = f"""You are a financial analyst selecting news for {symbol}.

[MARKET DATA]
- Return: {stock_return*100:.2f}%
- Alpha: {alpha*100:.2f}%
- Volume Z: {vol_z_str}

[INSTRUCTIONS]
{scenario_instruction}

[OUTPUT RULES]
- Select up to {k} items.
- Assign a sentiment score (-1 to 1) relative to the stock price impact.
- Return ONLY a JSON list: [{{"id": 0, "score": 0.8}}, ...]
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": news_text}
            ],
            temperature=0.0,
            max_tokens=200
        )
        response = completion.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Try parsing as pure JSON first
            import json
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            selected_items = json.loads(response)
            
            # Helper to get ID and score safely
            result_news = []
            for item in selected_items:
                if isinstance(item, dict) and 'id' in item:
                    idx = int(item['id'])
                    score = item.get('score', 0)
                    if idx < len(valid_news):
                        news_obj = valid_news[idx].copy()
                        news_obj['sentiment_score'] = score
                        result_news.append(news_obj)
                elif isinstance(item, int): # Fallback if LLM returns simple list
                    idx = item
                    if idx < len(valid_news):
                        result_news.append(valid_news[idx])

            return result_news[:k]
            
        except json.JSONDecodeError:
            # Fallback to regex if JSON fails
            import re
            ids = [int(x) for x in re.findall(r'\d+', response)]
            selected = [valid_news[i] for i in ids if i < len(valid_news)]
            return selected[:k]

    except Exception as e:
        # print(f"    LLM News Selection Failed: {e}. Falling back to relevance scoring.")
        return rank_news_by_relevance(news, None)[:k]


def pre_filter_news(news: list, symbol: str, client, model: str) -> list:
    """
    Stage 1: Cheap pre-filtering to remove noise using Gemini Flash Lite.
    Instead of a fixed percentage, we ask the model to aggressively filter out noise
    and keep only material news.
    """
    if not news or len(news) < 5:
        return news
        
    valid_news = [n for n in news if isinstance(n, dict) and not _is_spam(n)]
    if len(valid_news) < 5:
        return valid_news

    # Prepare simplified text
    news_text = ""
    for i, n in enumerate(valid_news):
        headline = n.get('headline', 'No Headline')
        date = n.get('date', '')
        news_text += f"ID {i}: [{date}] {headline}\n"

    system_prompt = f"""You are a strict financial news filter for {symbol}.
Your task is to FILTER OUT all noise, fluff, and irrelevant news.
Keep ONLY news items that are "material" and likely to move the stock price (e.g., Earnings, M&A, Regulatory, Major Products, Macro impacts).
Discard routine upgrades/downgrades unless they are from major banks with significant price target changes.
Discard generic market commentary.

Return ONLY a JSON object with the list of kept IDs: {{ "ids": [0, 5, 12] }}."""

    # Retry logic for Rate Limits (429)
    # Retry INDEFINITELY until success for Rate Limits to ensure filtering happens
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": news_text}
                ],
                temperature=0.0,
                max_tokens=500
            )
            response = completion.choices[0].message.content.strip()
            
            # Clean response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            import json
            try:
                data = json.loads(response)
                ids = data.get("ids", [])
                
                # Fallback if list is flat
                if isinstance(data, list):
                    ids = data
            except json.JSONDecodeError:
                # Fallback: regex search for numbers
                # Looks for patterns like "ids": [1, 2, 3] or just [1, 2, 3]
                import re
                ids = [int(x) for x in re.findall(r'\d+', response)]
                # Sanity check: if too many numbers (e.g. dates), be careful.
                # But since we asked for IDs only, usually fine.
                
            selected = []
            for i in ids:
                if isinstance(i, int) and i < len(valid_news):
                    selected.append(valid_news[i])
            
            if not selected:
                # If model filtered everything, it might mean everything was noise.
                # However, DeepSeek needs *something* to work with if possible, or maybe it's fine to have empty news.
                # Let's keep at least the top 1 most recent just in case, or trust the filter.
                # TRUST THE FILTER: If Gemini says it's all noise, we return empty list.
                print(f"    [Pre-Filter] {symbol}: All {len(valid_news)} items filtered as noise.")
                return []
                
            print(f"    [Pre-Filter] {symbol}: Kept {len(selected)}/{len(valid_news)} material items")
            return selected

        except Exception as e:
            # Handle Rate Limits with backoff - WAIT FOREVER
            if "429" in str(e):
                import time
                sleep_time = 10 + (random.random() * 5) # Wait at least 10-15 seconds
                # print(f"    [Pre-Filter] Rate limit (429). Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            
            # Other errors (non-429) -> Skip filter
            print(f"    [Pre-Filter] Failed (Non-429): {e}. Passing all news.")
            return valid_news


def sample_news(news: list, k: int = 5, strategy: str = "relevant", end_date: Optional[str] = None, 
                symbol: str = "", client = None, model: str = "",
                stock_return: float = 0.0, market_return: float = 0.0, market_name: str = "Market",
                alpha: float = 0.0, vol_z: float = 0.0,
                pre_filter_client = None, pre_filter_model: str = "") -> list:
    """
    Select up to k news items using a strategy:
    - "relevant": deterministic top-k by recency + information richness + keywords
    - "random": random sample
    - "llm": use LLM to pick most impactful news for the symbol
    """
    if len(news) <= k:
        return news

    if strategy == "random":
        return [news[i] for i in sorted(random.sample(range(len(news)), k))]
    
    if strategy == "llm" and client and symbol:
        # --- NEW: Stage 1 Pre-Filtering ---
        target_news = news
        if pre_filter_client and pre_filter_model:
            target_news = pre_filter_news(news, symbol, pre_filter_client, pre_filter_model)
        
        # --- Stage 2: DeepSeek/Main LLM Selection (Original Logic) ---
        return rank_news_with_llm(target_news, k, symbol, client, model, stock_return, market_return, market_name, alpha, vol_z)

    ranked = rank_news_by_relevance(news, end_date)
    return ranked[:k]


def format_news_items(news_items: list) -> list:
    """
    Format raw news dicts (or preformatted strings) into prompt-ready strings.
    """
    formatted = []
    for item in news_items:
        if isinstance(item, dict):
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            
            # Add sentiment score if available
            score_str = ""
            if "sentiment_score" in item:
                score = item["sentiment_score"]
                sign = "+" if score > 0 else ""
                score_str = f" (Sentiment: {sign}{score:.2f})"

            if headline or summary:
                formatted.append(f"[Headline]{score_str}: {headline}\\n[Summary]: {summary}\\n")
        elif isinstance(item, str):
            formatted.append(item)
    return formatted


def map_bin_label(bin_lb: str) -> str:
    """Map bin label to human-readable format."""
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
    else:
        lb = lb.replace('5', '4-5%')
    return lb


def get_all_prompts(
    finnhub_client, 
    symbol: str, 
    data_dir: str,
    min_past_weeks: int = 1, 
    max_past_weeks: int = 3, 
    with_basics: bool = True,
    news_strategy: str = "relevant",
    client = None,  # Pass LLM client down
    model: str = "", # Pass model name down
    parallel: int = 1, # Added parallel support
    pre_filter_client = None,
    pre_filter_model: str = ""
) -> list:
    """
    Generate all prompts for a symbol from the prepared data.
    
    Returns list of (prompt, prediction_label) tuples.
    """
    # Load data
    if with_basics:
        csv_file = list(Path(data_dir).glob(f"{symbol}_*_*.csv"))
        csv_file = [f for f in csv_file if 'nobasics' not in str(f) and 'gpt-4' not in str(f)]
    else:
        csv_file = list(Path(data_dir).glob(f"{symbol}_*_*_nobasics.csv"))
    
    if not csv_file:
        print(f"    No data file found for {symbol}")
        return []
    
    df = pd.read_csv(csv_file[0])
    company_prompt = get_company_prompt(finnhub_client, symbol)

    # Determine market index for context
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", "NFLX"]
    market_symbol = "QQQ" if symbol in tech_stocks else "SPY"
    market_name = "Nasdaq-100" if market_symbol == "QQQ" else "S&P 500"

    # --- STEP 1: Pre-process all rows (Lightweight & Sequential) ---
    # We gather all technicals, basics, and raw news first without calling LLM.
    print(f"    Pre-processing {len(df)} rows for {symbol}...")
    processed_rows = []
    
    for row_idx, row in df.iterrows():
        # Get returns and metrics for current row
        stock_ret = row['Weekly Returns']
        market_ret = market_manager.get_return(market_symbol, row['Start Date'], row['End Date'])
        alpha = stock_ret - market_ret
        vol_z = market_manager.get_volume_z_score(symbol, row['End Date'])
        
        # Get formatted data (no LLM calls here yet)
        head, news_formatted, basics, news_raw = get_prompt_by_row(symbol, row, market_ret, market_name)
        
        # Store everything needed for prompt construction
        processed_rows.append({
            "idx": row_idx,
            "head": head,
            "news_formatted": news_formatted, # Default formatted news
            "basics": basics,
            "news_raw": news_raw,
            "start_date": row['Start Date'],
            "end_date": row['End Date'],
            "stock_ret": stock_ret,
            "market_ret": market_ret,
            "alpha": alpha,
            "vol_z": vol_z,
            "bin_label": row['Bin Label']
        })

    # --- STEP 2: Construct Prompts (Heavy if using LLM for news) ---
    # This step is now parallelized!
    
    def _build_single_prompt_worker(p_row):
        # Determine history window
        curr_idx = p_row["idx"]
        
        # If not enough history, skip
        if curr_idx < min_past_weeks:
            return None

        # Randomly choose history length
        # Note: In parallel execution, random is thread-safe enough for this
        hist_len = min(random.choice(range(min_past_weeks, max_past_weeks + 1)), curr_idx)
        
        prompt_parts = []
        
        # Build history context
        for i in range(curr_idx - hist_len, curr_idx):
            hist_row = processed_rows[i]
            prompt_parts.append("\\n" + hist_row["head"])
            
            # NEWS SELECTION (The heavy part if using LLM)
            # We use the raw news from the historical row
            sampled_news = sample_news(
                hist_row["news_raw"],
                min(5, len(hist_row["news_raw"])),
                strategy=news_strategy,
                end_date=hist_row["end_date"],
                symbol=symbol,
                client=client,
                model=model,
                stock_return=hist_row["stock_ret"],
                market_return=hist_row["market_ret"],
                market_name=market_name,
                alpha=hist_row["alpha"],
                vol_z=hist_row["vol_z"],
                pre_filter_client=pre_filter_client,
                pre_filter_model=pre_filter_model
            )
            
            formatted = format_news_items(sampled_news)
            if formatted:
                prompt_parts.append("\\n".join(formatted))
            else:
                prompt_parts.append("No relative news reported.")

        # Add current row context (Target)
        # Note: The prompt asks to predict NEXT week, so we provide info UP TO start_date
        # p_row["basics"] and p_row["head"] contain info ending at p_row["end_date"]? 
        # Wait, the prompt structure in original code was:
        # prompt = company + history + basics + instruction
        # And history was built from prev_rows.
        
        prompt_str = ""
        prompt_str += "\\n".join(prompt_parts)
        
        prediction = map_bin_label(p_row["bin_label"])
        
        # Build final prompt string
        full_prompt = company_prompt + '\\n' + prompt_str + '\\n' + p_row["basics"]
        full_prompt += f"\\n\\nBased on all the information before {p_row['start_date']}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Integrate both quantitative signals and news to explain the movement. "
        full_prompt += f"Then let's assume your prediction for next week ({p_row['start_date']} to {p_row['end_date']}) is {prediction}. Provide a summary analysis to support your prediction. Before writing the analysis, cross-reference the [QUANT SIGNALS] with the [NEWS]. If the prediction is {prediction} but the technicals show a conflict (e.g., predicting 'Up' when Trend is 'Strong Downtrend'), explain this divergence logically (e.g., as a 'trend reversal' or 'short squeeze') rather than ignoring the data. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."

        return (full_prompt.strip(), prediction, p_row['start_date'], p_row['end_date'])

    all_prompts = []
    
    # Run parallel construction
    print(f"    Generating prompts (Parallel={parallel})...")
    if parallel > 1 and news_strategy == 'llm':
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Create futures
            futures = [executor.submit(_build_single_prompt_worker, row) for row in processed_rows]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"    {symbol} Prompts", unit="gen"):
                res = future.result()
                if res:
                    all_prompts.append(res)
    else:
        # Sequential fallback (or if not using LLM for news, it's fast enough)
        for row in tqdm(processed_rows, desc=f"    {symbol} Prompts", unit="gen"):
            res = _build_single_prompt_worker(row)
            if res:
                all_prompts.append(res)

    # Sort by date just to be clean (parallel execution might scramble order)
    all_prompts.sort(key=lambda x: x[2]) # Sort by start_date
    
    print(f"    DEBUG: Generated {len(all_prompts)} prompts for {symbol}.")
    return all_prompts


# ============================================================================
# GPT-4 QUERY FUNCTIONS
# ============================================================================

# Semaphore to control DeepSeek concurrency SEPARATELY from pre-filter
# Allow high concurrency for pre-filter (limited by Rate Limit retry) but
# allow user to control DeepSeek load if needed.
# Actually, the `parallel` arg controls the ThreadPool size, which limits EVERYTHING.
# To separate them, we would need two thread pools or a semaphore inside the worker.
# For now, let's trust the user's --parallel setting but ensure pre-filter waits.
csv_lock = threading.Lock()


def initialize_csv(filename: str):
    """Initialize CSV file with headers."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer", "prediction", "start_date", "end_date", "index", "primary_driver"])


def append_to_csv(filename: str, prompt: str, answer: str, prediction: str, start_date: str, end_date: str, index: int = 0, primary_driver: str = "Unknown"):
    """Append a row to the CSV file (thread-safe)."""
    with csv_lock:
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([prompt, answer, prediction, start_date, end_date, index, primary_driver])


def query_ollama(prompt: str, system_prompt: str, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> str:
    """Query local Ollama instance (FREE!)"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\\n\\nUser: {prompt}\\n\\nAssistant:",
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 1000}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise Exception(f"Ollama error: {e}")


def process_single_prompt(
    client,
    prompt_data: Tuple[int, str, str, str, str],  # (index, prompt, prediction, start_date, end_date)
    csv_file: str,
    model: str,
    backend: str,
    symbol: str,
    total: int
) -> Tuple[int, bool]:
    """
    Process a single prompt and save to CSV (thread-safe).
    Returns (index, success).
    """
    index, prompt, prediction, start_date, end_date = prompt_data
    
    # Retry logic
    for attempt in range(5):
        try:
            if backend == "ollama":
                answer = query_ollama(prompt, SYSTEM_PROMPT, model)
            else:
                # OpenAI / DeepSeek backend (same API format)
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4000 # Increased for reasoner output if needed, though split
                }
                
                # DeepSeek Reasoner does not support temperature/top_p
                if "reasoner" not in model:
                    kwargs["temperature"] = 0.4
                
                completion = client.chat.completions.create(**kwargs)
                answer = completion.choices[0].message.content
                
                # Clean <think> tags if present (DeepSeek Reasoner sometimes includes them or if we use specific format)
                # Note: DeepSeek API usually separates 'reasoning_content', but we ensure 'content' is clean just in case.
                if answer and "<think>" in answer:
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            
            # Extract Primary Driver using Regex
            primary_driver = "Unknown"
            try:
                # Matches [Primary Driver]: Technical (or similar)
                match = re.search(r"\[Primary Driver\]:\s*(.*)", answer, re.IGNORECASE)
                if match:
                    primary_driver = match.group(1).strip()
                    # Clean up if model added extra punctuation
                    primary_driver = primary_driver.split('\n')[0].strip()
            except:
                pass

            append_to_csv(csv_file, prompt, answer, prediction, str(start_date), str(end_date), index, primary_driver)
            return (index, True)
        except Exception as e:
            if attempt < 4:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"\\n    FAILED {symbol}-{index} after 5 retries: {e}")
                append_to_csv(csv_file, prompt, "", prediction, str(start_date), str(end_date), index, "Unknown")
                return (index, False)
    
    return (index, False)


def query_llm(
    client,  # OpenAI client or None for Ollama
    finnhub_client,
    symbol: str,
    data_dir: str,
    min_past_weeks: int = 1,
    max_past_weeks: int = 3,
    with_basics: bool = True,
    model: str = "gpt-3.5-turbo",
    backend: str = "openai",
    parallel: int = 1,  # Number of parallel requests
    news_strategy: str = "relevant",
    pre_filter_client = None,
    pre_filter_model: str = ""
):
    """
    Query LLM to generate analysis for all prompts of a symbol.
    Supports OpenAI API or local Ollama.
    Now with PARALLEL processing support!
    """
    suffix = "" if with_basics else "_nobasics"
    csv_file = f'{data_dir}/{symbol}{suffix}_gpt-4.csv'
    
    # Check for existing progress
    done_indices = set()
    if os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            if 'index' in existing_df.columns:
                done_indices = set(existing_df['index'].tolist())
            else:
                # Old format without index - count rows as done
                done_indices = set(range(len(existing_df)))
            print(f"    Resuming from {len(done_indices)} existing entries")
        except:
            initialize_csv(csv_file)
    else:
        initialize_csv(csv_file)

    prompts = get_all_prompts(
        finnhub_client,
        symbol,
        data_dir,
        min_past_weeks,
        max_past_weeks,
        with_basics,
        news_strategy,
        client=client,
        model=model,
        parallel=parallel,  # Pass parallel arg
        pre_filter_client=pre_filter_client,
        pre_filter_model=pre_filter_model
    )
    
    if not prompts:
        return
    
    # Filter out already done prompts
    prompts_with_index = [
        (i, prompt, prediction, start_date, end_date)
        for i, (prompt, prediction, start_date, end_date) in enumerate(prompts)
        if i not in done_indices
    ]
    
    total = len(prompts)
    remaining = len(prompts_with_index)
    
    if remaining == 0:
        print(f"    {symbol} - Already complete!")
        return
    
    print(f"    Processing {remaining}/{total} prompts (parallel={parallel})")

    if parallel <= 1:
        # Sequential mode (original behavior)
        for prompt_data in tqdm(prompts_with_index, desc=f"    {symbol}", unit="prompt"):
            index = prompt_data[0]
            process_single_prompt(client, prompt_data, csv_file, model, backend, symbol, total)
    else:
        # PARALLEL MODE - Much faster!
        # completed = len(done_indices)
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    process_single_prompt, client, prompt_data, csv_file, model, backend, symbol, total
                ): prompt_data[0]
                for prompt_data in prompts_with_index
            }
            
            # Use tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"    {symbol} (Parallel)", unit="prompt"):
                try:
                    result_index, success = future.result()
                    # status = "✓" if success else "✗"
                    # print(f"    {symbol} - {completed}/{total} {status}", end='\\r')
                except Exception as e:
                    print(f"\\n    Error processing {symbol}: {e}")
    
    print(f"    {symbol} - Complete!                    ")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate LLM labels for FinGPT Forecaster training')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing prepared data')
    parser.add_argument('--symbols', type=str, default='all',
                        help='Comma-separated symbols or "all" for all in data_dir')
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'ollama', 'deepseek'],
                        help='LLM backend: openai, deepseek (cheap & good!), or ollama (FREE local)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        help='Model name. OpenAI: gpt-4o-mini. DeepSeek: deepseek-chat. Ollama: llama3.1')
    parser.add_argument('--min_weeks', type=int, default=1, help='Minimum past weeks of context')
    parser.add_argument('--max_weeks', type=int, default=4, help='Maximum past weeks of context')
    parser.add_argument('--no_basics', action='store_true', help='Process nobasics files')
    parser.add_argument('--parallel', type=int, default=5,
                        help='Number of parallel API requests (default: 5, use 1 for sequential)')
    parser.add_argument('--news_strategy', type=str, default='relevant',
                        choices=['relevant', 'random', 'llm'],
                        help='How to pick news snippets: relevant (ranked), random, or llm (DeepSeek picks)')
    parser.add_argument('--pre_filter', action='store_true',
                        help='Use cheap model (Gemini/OpenRouter) to filter 80%% of news before DeepSeek')
    args = parser.parse_args()
    
    # Setup backend
    openai_client = None
    if args.backend == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        openai_client = OpenAI(api_key=openai_api_key)
    elif args.backend == "deepseek":
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY environment variable. Get it at: https://platform.deepseek.com/")
        openai_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        # Default to deepseek-reasoner if no model specified
        if args.model == "gpt-3.5-turbo":
            args.model = "deepseek-reasoner"
        print(f"Using DeepSeek API with model: {args.model}")
    elif args.backend == "ollama":
        # Test Ollama connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            available_models = [m['name'] for m in response.json().get('models', [])]
            if not available_models:
                print("WARNING: No models found in Ollama. Run: ollama pull llama3.1")
            else:
                print(f"Ollama models available: {', '.join(available_models)}")
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at localhost:11434. Is it running? Error: {e}")
    
    # Setup Pre-Filter Client (OpenRouter / Gemini)
    pre_filter_client = None
    pre_filter_model = "google/gemini-2.0-flash-lite-001"
    
    if args.pre_filter:
        print(f"[Pre-Filter] Enabled. Using OpenRouter ({pre_filter_model})")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
             # Fallback to DeepSeek Key if OpenRouter not set, but warn
            print("    WARNING: OPENROUTER_API_KEY not found for pre-filter.")
            print("    Please set OPENROUTER_API_KEY for free Gemini access.")
            raise ValueError("Missing OPENROUTER_API_KEY")
            
        pre_filter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )

    # Finnhub is optional (used for company profiles)
    finnhub_client = None
    finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
    if finnhub_api_key:
        finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    else:
        print("Note: FINNHUB_API_KEY not set. Company profiles will be skipped.")
    
    # Find symbols
    if args.symbols == 'all':
        csv_files = list(Path(args.data_dir).glob("*_*_*.csv"))
        csv_files = [f for f in csv_files if 'gpt-4' not in str(f)]
        symbols = list(set([f.stem.split('_')[0] for f in csv_files]))
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    with_basics = not args.no_basics
    
    print("=" * 60)
    print("FinGPT Forecaster - Label Generation (PARALLEL)")
    print("=" * 60)
    backend_note = " (FREE!)" if args.backend == "ollama" else " (Cheap & Good!)" if args.backend == "deepseek" else ""
    print(f"Backend: {args.backend.upper()}{backend_note}")
    print(f"Model: {args.model}")
    print(f"Parallel: {args.parallel} concurrent requests")
    print(f"Data Directory: {args.data_dir}")
    print(f"Symbols: {len(symbols)} companies")
    print(f"Context: {args.min_weeks}-{args.max_weeks} weeks")
    print("=" * 60)
    
    for symbol in sorted(symbols):
        print(f"\\nProcessing {symbol}...")
        query_llm(
            openai_client, finnhub_client, symbol, args.data_dir,
            args.min_weeks, args.max_weeks, with_basics, args.model, args.backend,
            args.parallel, args.news_strategy,
            pre_filter_client=pre_filter_client,
            pre_filter_model=pre_filter_model
        )
    
    print("\\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Labels saved to: {args.data_dir}/*_gpt-4.csv")
    print("=" * 60)
    print("\\nNext step: Run build_dataset.py to create training dataset")


if __name__ == "__main__":
    main()
