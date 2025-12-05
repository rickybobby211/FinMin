"""
FinGPT Forecaster - Label Generation Script
============================================
This script uses LLMs to generate training labels (analysis and predictions)
for the prepared market data.

Supported backends:
- OpenAI (gpt-3.5-turbo, gpt-4, gpt-4o-mini) - requires OPENAI_API_KEY
- Ollama (local, FREE) - requires Ollama running locally

Prerequisites:
- For OpenAI: Set environment variable OPENAI_API_KEY
- For Ollama: Install Ollama and run `ollama pull llama3.1` or `ollama pull mistral`
- Run prepare_latest_data.py first to generate raw data

Usage:
    # Cheap OpenAI option (recommended)
    python generate_labels.py --data_dir ./raw_data/2024-01-01_2024-11-01 --model gpt-3.5-turbo
    
    # FREE local option with Ollama
    python generate_labels.py --data_dir ./raw_data/2024-01-01_2024-11-01 --backend ollama --model llama3.1
"""

import os
import re
import csv
import json
import random
import argparse
import finnhub
import pandas as pd
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
load_dotenv("../.env")  # Also check parent directory


# ============================================================================
# CONFIGURATION
# ============================================================================

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]:
...
"""


# ============================================================================
# PROMPT GENERATION FUNCTIONS
# ============================================================================

def get_company_prompt(finnhub_client, symbol: str) -> str:
    """Generate company introduction prompt."""
    if finnhub_client is None:
        return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        if not profile:
            return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."
            
        company_template = (
            "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. "
            "Incorporated and publicly traded since {ipo}, the company has established its reputation "
            "as one of the key players in the market. As of today, {name} has a market capitalization "
            "of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding."
            "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. "
            "As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
        )
        return company_template.format(**profile)
    except Exception as e:
        print(f"    Warning: Could not fetch profile for {symbol}: {e}")
        return f"[Company Introduction]:\n\n{symbol} is a publicly traded company."


def get_prompt_by_row(symbol: str, row: pd.Series) -> tuple:
    """Generate prompt components for a single row of data."""
    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    
    head = f"From {start_date} to {end_date}, {symbol}'s stock price {term} from {row['Start Price']:.2f} to {row['End Price']:.2f}. Company news during this period are listed below:\n\n"
    
    try:
        news = json.loads(row["News"]) if isinstance(row["News"], str) else row["News"]
        if not isinstance(news, list):
            news = []
    except:
        news = []
    
    # Filter and format news
    formatted_news = []
    for n in news:
        if not isinstance(n, dict):
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
        
        # Skip spam
        summary = n.get('summary', '')
        if summary and summary.startswith("Looking for stock market analysis and research with proves results?"):
            continue
        
        headline = n.get('headline', '')
        if headline and summary:
            formatted_news.append(f"[Headline]: {headline}\n[Summary]: {summary}\n")
    
    news = formatted_news

    basics = json.loads(row['Basics'])
    if basics:
        basics_str = f"Some recent basic financials of {symbol}, reported at {basics['period']}, are presented below:\n\n[Basic Financials]:\n\n"
        basics_str += "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics_str = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics_str


def sample_news(news: list, k: int = 5) -> list:
    """Randomly sample k news items."""
    if len(news) <= k:
        return news
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


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
    with_basics: bool = True
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

    prev_rows = []
    all_prompts = []

    for row_idx, row in df.iterrows():
        prompt = ""
        if len(prev_rows) >= min_past_weeks:
            idx = min(random.choice(range(min_past_weeks, max_past_weeks + 1)), len(prev_rows))
            for i in range(-idx, 0):
                prompt += "\n" + prev_rows[i][0]
                sampled_news = sample_news(prev_rows[i][1], min(5, len(prev_rows[i][1])))
                if sampled_news:
                    prompt += "\n".join(sampled_news)
                else:
                    prompt += "No relative news reported."

        head, news, basics = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, basics))
        
        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)

        if not prompt:
            continue

        prediction = map_bin_label(row['Bin Label'])
        
        # Build the full prompt (for GPT-4 to generate analysis)
        prompt = company_prompt + '\n' + prompt + '\n' + basics
        prompt += f"\n\nBased on all the information before {row['Start Date']}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. "
        prompt += f"Then let's assume your prediction for next week ({row['Start Date']} to {row['End Date']}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."

        all_prompts.append((prompt.strip(), prediction, row['Start Date'], row['End Date']))
    
    return all_prompts


# ============================================================================
# GPT-4 QUERY FUNCTIONS
# ============================================================================

def initialize_csv(filename: str):
    """Initialize CSV file with headers."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer", "prediction", "start_date", "end_date"])


def append_to_csv(filename: str, prompt: str, answer: str, prediction: str, start_date: str, end_date: str):
    """Append a row to the CSV file."""
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([prompt, answer, prediction, start_date, end_date])


def query_ollama(prompt: str, system_prompt: str, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> str:
    """Query local Ollama instance (FREE!)"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1000}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise Exception(f"Ollama error: {e}")


def query_llm(
    client,  # OpenAI client or None for Ollama
    finnhub_client,
    symbol: str,
    data_dir: str,
    min_past_weeks: int = 1,
    max_past_weeks: int = 3,
    with_basics: bool = True,
    model: str = "gpt-3.5-turbo",
    backend: str = "openai"
):
    """
    Query LLM to generate analysis for all prompts of a symbol.
    Supports OpenAI API or local Ollama.
    """
    suffix = "" if with_basics else "_nobasics"
    csv_file = f'{data_dir}/{symbol}{suffix}_gpt-4.csv'
    
    # Check for existing progress
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        pre_done = len(existing_df)
        print(f"    Resuming from {pre_done} existing entries")
    else:
        initialize_csv(csv_file)
        pre_done = 0

    prompts = get_all_prompts(finnhub_client, symbol, data_dir, min_past_weeks, max_past_weeks, with_basics)
    
    if not prompts:
        return
    
    print(f"    Processing {len(prompts)} prompts ({pre_done} already done)")

    for i, (prompt, prediction, start_date, end_date) in enumerate(prompts):
        if i < pre_done:
            continue

        print(f"    {symbol} - {i+1}/{len(prompts)}", end='\r')
        
        # Retry logic
        for attempt in range(5):
            try:
                if backend == "ollama":
                    answer = query_ollama(prompt, SYSTEM_PROMPT, model)
                else:
                    # OpenAI / DeepSeek backend (same API format)
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    answer = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"\n    Retry {attempt + 1}/5 for {symbol}-{i}: {e}")
                if attempt == 4:
                    answer = ""
        
        append_to_csv(csv_file, prompt, answer, prediction, str(start_date), str(end_date))
    
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
        # Default to deepseek-chat if no model specified
        if args.model == "gpt-3.5-turbo":
            args.model = "deepseek-chat"
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
    print("FinGPT Forecaster - Label Generation")
    print("=" * 60)
    backend_note = " (FREE!)" if args.backend == "ollama" else " (Cheap & Good!)" if args.backend == "deepseek" else ""
    print(f"Backend: {args.backend.upper()}{backend_note}")
    print(f"Model: {args.model}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Symbols: {len(symbols)} companies")
    print(f"Context: {args.min_weeks}-{args.max_weeks} weeks")
    print("=" * 60)
    
    for symbol in sorted(symbols):
        print(f"\nProcessing {symbol}...")
        query_llm(
            openai_client, finnhub_client, symbol, args.data_dir,
            args.min_weeks, args.max_weeks, with_basics, args.model, args.backend
        )
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Labels saved to: {args.data_dir}/*_gpt-4.csv")
    print("=" * 60)
    print("\nNext step: Run build_dataset.py to create training dataset")


if __name__ == "__main__":
    main()

