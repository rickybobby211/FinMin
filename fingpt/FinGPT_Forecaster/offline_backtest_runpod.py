import argparse
import requests
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# RunPod Configuration
# Set your own API key and endpoint ID
# Get API key from: https://www.runpod.io/console/user/settings
DEFAULT_RUNPOD_API_ID = os.environ.get("RUNPOD_API_ID", "YOUR_ENDPOINT_ID_HERE")
DEFAULT_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY_HERE")

def get_actual_movement(ticker, start_date, end_date):
    """
    Fetch actual stock movement from yfinance for verification.
    """
    try:
        # Extend end_date by a few days to ensure we get the data (yfinance end is exclusive)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        end_str = end_dt.strftime("%Y-%m-%d")
        
        data = yf.download(ticker, start=start_date, end=end_str, progress=False)
        
        if len(data) == 0:
            return None, None, None

        # Get close price closest to start_date
        # Handle potential MultiIndex columns from yfinance
        try:
            close_data = data['Close']
            if isinstance(close_data, pd.DataFrame):
                # If ticker was passed, try to access it
                if ticker in close_data.columns:
                    close_data = close_data[ticker]
                else:
                    # Otherwise assume it's the only column or take the first one
                    close_data = close_data.iloc[:, 0]
        except KeyError:
             print("Error: 'Close' column not found in data")
             return None, None, None

        start_price = close_data.iloc[0]
        
        # Get close price closest to end_date (but within range)
        # We want the price roughly 7 days after start.
        # Let's find the date in data closest to end_date
        
        # Simple approach: First and Last in the range [start, start+7]
        # Actually, let's filter specifically for the week
        
        # Use simple string slicing on the index if it's DatetimeIndex
        mask = (close_data.index >= start_date) & (close_data.index <= end_date)
        week_data = close_data.loc[mask]
        
        if len(week_data) < 2:
            # If not enough data points in the exact week, try taking next available
            return None, None, None
            
        week_start_price = float(week_data.iloc[0])
        week_end_price = float(week_data.iloc[-1])

        
        pct_change = ((week_end_price - week_start_price) / week_start_price) * 100
        direction = "UP" if pct_change >= 0 else "DOWN"
        
        return direction, float(pct_change), float(week_end_price), float(week_start_price)
        
    except Exception as e:
        print(f"Error fetching verification data: {e}")
        return None, None, None, None

def run_prediction_task(api_id, api_key, payload):
    base_url = f"https://api.runpod.ai/v2/{api_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Start Job
    run_url = f"{base_url}/run"
    try:
        response = requests.post(run_url, json=payload, headers=headers)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("id")
    except Exception as e:
        return {"error": f"Request failed: {e}"}

    if not job_id:
        return {"error": "No Job ID returned"}

    print(f"Job started: {job_id}...", end="", flush=True)

    # 2. Poll for completion
    status_url = f"{base_url}/status/{job_id}"
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 600: # 10 min timeout
            return {"error": "Timeout"}
            
        time.sleep(2)
        try:
            status_res = requests.get(status_url, headers=headers)
            status_data = status_res.json()
            status = status_data.get("status")
            
            if status == "COMPLETED":
                print(" Done.")
                return status_data.get("output", {})
            elif status == "FAILED":
                print(" Failed.")
                return {"error": "Job Failed", "details": status_data}
        except Exception as e:
            print(f" Polling error: {e}")
            time.sleep(5)

def parse_prediction(text):
    """
    Extract prediction (UP/DOWN) from the model's text response.
    """
    if not text or not isinstance(text, str):
        return "UNKNOWN"
        
    text_lower = text.lower()
    
    # Map synonyms
    synonyms = {
        "up": "UP",
        "increase": "UP",
        "rise": "UP",
        "gain": "UP",
        "growth": "UP",
        "positive": "UP",
        "bullish": "UP",
        "rally": "UP",
        "jump": "UP",
        
        "down": "DOWN",
        "decrease": "DOWN",
        "drop": "DOWN",
        "fall": "DOWN",
        "decline": "DOWN",
        "negative": "DOWN",
        "bearish": "DOWN",
        "loss": "DOWN",
        "plummet": "DOWN",
        "slump": "DOWN"
    }
    
    direction_pattern = r"\b(" + "|".join(synonyms.keys()) + r")\b"
    
    # Strategy 1: Look for explicit "Prediction" keyword (case insensitive)
    # We find all occurrences and check the context immediately following them.
    # We prioritize the LAST occurrence as it usually contains the final verdict.
    matches = list(re.finditer(r"prediction[:\s\*\*]+", text_lower))
    
    if matches:
        # Check the last occurrence first
        last_match = matches[-1]
        start_idx = last_match.end()
        # Look at the next 300 chars to find the direction word
        search_window = text_lower[start_idx : start_idx + 300]
        
        # Search for direction words (prioritize longer matches like 'increase' over 'up')
        found = re.search(direction_pattern, search_window)
        if found:
            word = found.group(1)
            return synonyms[word]

    # Strategy 2: Last Resort - Look for direction in the last 200 characters of the entire text
    # Ideally, the conclusion is at the very end.
    last_chars = text_lower[-300:] # Increased window for safety
    found = re.search(direction_pattern, last_chars)
    if found:
        word = found.group(1)
        return synonyms[word]
        
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser(description="Run Offline Backtest using RunPod FinGPT")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock Ticker")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=10, help="Number of weeks to test")
    parser.add_argument("--history_weeks", type=int, default=2, help="Weeks of history in prompt (reduce if truncation occurs)")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="RunPod API Key")
    parser.add_argument("--api_id", type=str, default=DEFAULT_RUNPOD_API_ID, help="RunPod Endpoint ID")
    
    args = parser.parse_args()
    
    # Generate weekly dates
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    test_dates = []
    for i in range(args.weeks):
        # Create a date for each week (e.g., every 7 days)
        d = start_dt + timedelta(days=7 * i)
        test_dates.append(d.strftime("%Y-%m-%d"))
        
    print(f"Starting backtest for {args.ticker}")
    print(f"Period: {test_dates[0]} to {test_dates[-1]}")
    print(f"Total test points: {len(test_dates)}")
    print("-" * 60)
    
    results = []
    correct_count = 0
    
    for test_date in test_dates:
        # 1. Define verification period (the week FOLLOWING the test_date)
        # The model predicts "Next Week" relative to input date.
        pred_start = test_date
        pred_end_dt = datetime.strptime(test_date, "%Y-%m-%d") + timedelta(days=7)
        pred_end = pred_end_dt.strftime("%Y-%m-%d")
        
        print(f"[{test_date}] Testing...", end=" ")
        
        # 2. Call RunPod
        payload = {
            "input": {
                "ticker": args.ticker,
                "date": test_date,
                "n_weeks": args.history_weeks,
                "use_basics": True
            }
        }
        
        output = run_prediction_task(args.api_id, args.api_key, payload)
        
        if "error" in output:
            print(f"Error: {output['error']}")
            continue
            
        prediction_text = output.get("prediction", "")
        full_prompt = output.get("prompt", "")
        predicted_dir = parse_prediction(prediction_text)
        
        # 3. Verify
        actual_dir, pct_change, end_price, start_price = get_actual_movement(args.ticker, pred_start, pred_end)
        
        is_correct = (predicted_dir == actual_dir)
        if is_correct:
            correct_count += 1
            
        mark = "[CORRECT]" if is_correct else "[WRONG]"
        if actual_dir is None:
            mark = "[UNKNOWN]"
            
        # Check if news were missing in prompt
        has_news = "[Headline]:" in full_prompt
        news_status = "" if has_news else " (NO NEWS)"
            
        print(f" Pred: {predicted_dir} | Act: {actual_dir} ({pct_change:.2f}%) {mark}{news_status}")
        
        results.append({
            "date": test_date,
            "prediction": predicted_dir,
            "actual": actual_dir,
            "pct_change": pct_change,
            "start_price": start_price,
            "end_price": end_price,
            "correct": is_correct,
            "has_news": has_news,
            "full_text": prediction_text,
            "prompt_input": json.dumps(payload),
            "full_prompt": full_prompt
        })
        
    # Summary
    if results:
        accuracy = (correct_count / len(results)) * 100
        
        # Calculate stats for weeks with/without news
        with_news = [r for r in results if r['has_news']]
        without_news = [r for r in results if not r['has_news']]
        
        acc_news = (len([r for r in with_news if r['correct']]) / len(with_news)) * 100 if with_news else 0
        acc_no_news = (len([r for r in without_news if r['correct']]) / len(without_news)) * 100 if without_news else 0
        
        print("-" * 60)
        print(f"Backtest Complete. Total Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
        print(f"  - With News:    {acc_news:.2f}% ({len([r for r in with_news if r['correct']])}/{len(with_news)})")
        print(f"  - Without News: {acc_no_news:.2f}% ({len([r for r in without_news if r['correct']])}/{len(without_news)})")
        
        # Save CSV
        filename = f"backtest_{args.ticker}_{args.start}_{len(results)}w.csv"
        pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
