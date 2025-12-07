"""
FinGPT Forecaster - Web UI for RunPod Serverless API
====================================================
A simple Flask web interface to interact with your FinGPT model on RunPod Serverless.

Usage:
    python web_ui.py

Requirements:
    pip install flask requests
"""

from flask import Flask, render_template_string, request, jsonify
import requests
import os

app = Flask(__name__)

# RunPod Serverless endpoint (get this after deploying on RunPod)
RUNPOD_API_URL = os.environ.get("RUNPOD_API_URL", "YOUR_RUNPOD_ENDPOINT_HERE")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY_HERE")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FinGPT Forecaster - Stock Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            opacity: 0.9;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            display: none;
        }
        .loading {
            text-align: center;
            color: #667eea;
            display: none;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
        .info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 FinGPT Forecaster</h1>
        
        <div class="info">
            <strong>ℹ️ How to use:</strong><br>
            1. Deploy your model on RunPod Serverless<br>
            2. Set RUNPOD_API_URL and RUNPOD_API_KEY environment variables<br>
            3. Enter a stock ticker (e.g., AAPL, MSFT, TSLA)<br>
            4. Get AI-powered stock predictions!
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="ticker">Stock Ticker:</label>
                <input type="text" id="ticker" name="ticker" placeholder="AAPL" required>
            </div>
            
            <div class="form-group">
                <label for="date">Prediction Date (optional):</label>
                <input type="date" id="date" name="date">
            </div>
            
            <div class="form-group">
                <label for="n_weeks">Number of Weeks (1-4):</label>
                <select id="n_weeks" name="n_weeks">
                    <option value="1">1 week</option>
                    <option value="2">2 weeks</option>
                    <option value="3" selected>3 weeks</option>
                    <option value="4">4 weeks</option>
                </select>
            </div>
            
            <button type="submit">Get Prediction</button>
        </form>
        
        <div class="loading" id="loading">⏳ Generating prediction...</div>
        <div class="error" id="error"></div>
        <div class="result" id="result"></div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const date = document.getElementById('date').value;
            const n_weeks = parseInt(document.getElementById('n_weeks').value);
            
            // Show loading, hide result/error
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').textContent = '';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        ticker: ticker,
                        date: date || null,
                        n_weeks: n_weeks
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = 'Error: ' + data.error;
                } else {
                    document.getElementById('result').textContent = data.prediction;
                    document.getElementById('result').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'Error: ' + error.message;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Forward prediction request to RunPod Serverless API."""
    data = request.json
    
    if not RUNPOD_API_URL or RUNPOD_API_URL == "YOUR_RUNPOD_ENDPOINT_HERE":
        return jsonify({
            "error": "RunPod API URL not configured. Set RUNPOD_API_URL environment variable."
        }), 500
    
    # Prepare request for RunPod
    payload = {
        "input": {
            "ticker": data.get("ticker"),
            "date": data.get("date"),
            "n_weeks": data.get("n_weeks", 3)
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        # Call RunPod Serverless API
        response = requests.post(
            RUNPOD_API_URL,
            json=payload,
            headers=headers,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle RunPod response format
            if "output" in result:
                output = result["output"]
                if "error" in output:
                    return jsonify({"error": output["error"]}), 500
                return jsonify({
                    "prediction": output.get("prediction", "No prediction generated"),
                    "ticker": output.get("ticker"),
                    "date": output.get("date")
                })
            else:
                return jsonify({"error": "Unexpected response format from RunPod"}), 500
        else:
            return jsonify({
                "error": f"RunPod API error: {response.status_code} - {response.text}"
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. The model might be taking too long."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("FinGPT Forecaster - Web UI")
    print("=" * 60)
    print(f"RunPod API URL: {RUNPOD_API_URL}")
    print(f"API Key configured: {'Yes' if RUNPOD_API_KEY != 'YOUR_RUNPOD_API_KEY_HERE' else 'No'}")
    print("=" * 60)
    print("\nOpen http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

