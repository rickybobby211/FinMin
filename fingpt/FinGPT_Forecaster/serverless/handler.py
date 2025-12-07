"""
FinGPT Forecaster - RunPod Serverless Handler
==============================================
This handler processes stock prediction requests on RunPod Serverless.

Deploy with: runpod deploy
"""

import os
import re
import json
import time
import torch
import runpod
import finnhub
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
# Use your custom trained adapter (set via environment variable or default)
# Options:
#   1. HuggingFace path: "your-username/fingpt-v3-float16" (RECOMMENDED for Serverless)
#   2. Official FinGPT: "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora" (default)
#   3. Local path: "/runpod-volume/fingpt-v3-float16_202512060944" (only works if volume mounted)
ADAPTER_ID = os.environ.get("ADAPTER_PATH", "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]
Prediction: Up by 2-3%
Analysis: ..."""

# Global model (loaded once, reused for all requests)
model = None
tokenizer = None
finnhub_client = None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load model once at cold start."""
    global model, tokenizer, finnhub_client
    
    if model is not None:
        return  # Already loaded
    
    print("Loading model...")
    
    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapter
    # If adapter is from HuggingFace, pass token for private repos
    if "/" in ADAPTER_ID and not os.path.exists(ADAPTER_ID):
        # HuggingFace path - pass token for private repos
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=hf_token)
    else:
        # Local path
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model = model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    
    # Initialize Finnhub client
    finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
    if finnhub_api_key:
        finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    
    print("Model loaded successfully!")


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


def get_news(symbol, data):
    """Fetch company news from Finnhub."""
    if finnhub_client is None:
        data['News'] = [json.dumps([])] * len(data)
        return data
    
    news_list = []
    for _, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        
        try:
            time.sleep(0.5)  # Rate limit
            weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            weekly_news = [
                {
                    "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                    "headline": n['headline'],
                    "summary": n['summary'],
                } for n in weekly_news[:10]  # Limit to 10 news items
            ]
            weekly_news.sort(key=lambda x: x['date'])
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


def construct_prompt(ticker, curday, n_weeks):
    """Build the full prompt for the model."""
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    
    # Get stock data and news
    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data)
    
    # Build prompt
    company_prompt = get_company_prompt(ticker)
    prompt = company_prompt + "\n"
    
    for _, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
        
        prompt += f"\nFrom {start_date} to {end_date}, {ticker}'s stock price {term} from {row['Start Price']:.2f} to {row['End Price']:.2f}. Company news during this period are listed below:\n\n"
        
        news = json.loads(row['News'])
        if news:
            for n in news[:5]:
                prompt += f"[Headline]: {n['headline']}\n[Summary]: {n['summary']}\n\n"
        else:
            prompt += "No news reported.\n"
    
    period = f"{curday} to {n_weeks_before(curday, -1)}"
    prompt += f"\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {ticker}. Come up with 2-4 most important factors respectively and keep them concise. Then make your prediction of the {ticker} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
    
    full_prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    return full_prompt


# ============================================================================
# INFERENCE
# ============================================================================

def predict(ticker, prediction_date=None, n_weeks=3):
    """Generate stock prediction."""
    load_model()  # Ensure model is loaded
    
    if prediction_date is None:
        prediction_date = date.today().strftime("%Y-%m-%d")
    
    # Build prompt
    prompt = construct_prompt(ticker, prediction_date, n_weeks)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Generate
    with torch.no_grad():
        res = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
    
    torch.cuda.empty_cache()
    
    return answer


# ============================================================================
# RUNPOD HANDLER
# ============================================================================

def handler(event):
    """
    RunPod Serverless handler function.
    
    Input format:
    {
        "input": {
            "ticker": "AAPL",
            "date": "2025-12-04",  # optional
            "n_weeks": 3           # optional
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        ticker = input_data.get("ticker")
        if not ticker:
            return {"error": "Missing required field: ticker"}
        
        prediction_date = input_data.get("date")
        n_weeks = input_data.get("n_weeks", 3)
        
        # Generate prediction
        prediction = predict(
            ticker=ticker.upper(),
            prediction_date=prediction_date,
            n_weeks=n_weeks
        )
        
        return {
            "ticker": ticker.upper(),
            "date": prediction_date or date.today().strftime("%Y-%m-%d"),
            "prediction": prediction
        }
    
    except Exception as e:
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
