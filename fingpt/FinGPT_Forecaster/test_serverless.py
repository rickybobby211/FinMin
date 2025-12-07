"""
Test script for RunPod Serverless FinGPT Forecaster API
"""
import requests
import json
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your RunPod endpoint ID (from the URL)
ENDPOINT_ID = "lfahu5e6q9pmfq"

# Your RunPod API Key (get from: https://www.runpod.io/console/user/settings)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")  # Replace with your actual key

# API URL (using runsync to wait for completion)
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_prediction(ticker="AAPL", date="2025-12-05", n_weeks=1):
    """Test the FinGPT Forecaster API."""
    
    print(f"\n{'='*60}")
    print("Testing FinGPT Forecaster API")
    print(f"{'='*60}")
    print(f"Ticker: {ticker}")
    print(f"Date: {date}")
    print(f"N Weeks: {n_weeks}")
    print(f"{'='*60}\n")
    
    # Prepare request
    payload = {
        "input": {
            "ticker": ticker,
            "date": date,
            "n_weeks": n_weeks
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # Send request (runsync waits for completion)
    print("Sending request to RunPod (waiting for completion)...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)  # 5 min timeout
        
        elapsed_time = time.time() - start_time
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            # Debug: Print full response to see structure
            print("\n🔍 Full response structure:")
            print(json.dumps(result, indent=2))
            print("\n" + "="*60)
            
            # Check if there's an error in the result
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                return
            
            # Extract output - RunPod Serverless wraps response in "output" field
            if "output" in result:
                output = result["output"]
            else:
                output = result
            
            print(f"\n{'='*60}")
            print("PREDICTION RESULT:")
            print(f"{'='*60}")
            print(f"Ticker: {output.get('ticker', 'N/A')}")
            print(f"Date: {output.get('date', 'N/A')}")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            print("\nPrediction:")
            print("-" * 60)
            
            # Try different possible keys for the prediction
            prediction = output.get('prediction') or output.get('result') or output.get('text') or str(output)
            print(prediction)
            print("-" * 60)
        
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.Timeout:
        print("❌ Request timed out (took longer than 5 minutes)")
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test with Apple stock
    test_prediction(ticker="AAPL", date="2025-12-06", n_weeks=3)
    
    # Uncomment to test other stocks:
    # test_prediction(ticker="MSFT", date="2025-12-06", n_weeks=3)
    # test_prediction(ticker="GOOGL", date="2025-12-06", n_weeks=3)

