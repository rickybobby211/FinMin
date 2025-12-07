# FinGPT Forecaster - Deployment Guide

This guide explains how to deploy your trained FinGPT model and access it via API or Web UI.

## 🚀 Deployment Options

### Option 1: RunPod Serverless (Recommended for API)

**Best for:** Production API, automatic scaling, pay-per-use

#### Step 1: Prepare Your Trained Model

After training, your model is saved in:
```
finetuned_models/fingpt-v3-float16_YYYYMMDDHHMM/
```

#### Step 2: Upload Model to RunPod Volume (Optional)

If you want to use your custom trained model:

1. **On RunPod GPU Pod:**
   ```bash
   # Copy your trained model to persistent volume
   cp -r finetuned_models/fingpt-v3-float16_* /runpod-volume/
   ```

2. **Or upload via RunPod Web UI:**
   - Go to RunPod → Volumes
   - Upload your model folder

#### Step 3: Build and Deploy Docker Image

1. **Build the Docker image:**
   ```bash
   cd fingpt/FinGPT_Forecaster/serverless
   docker build -t fingpt-forecaster:latest .
   ```

2. **Push to Docker Hub (or RunPod Registry):**
   ```bash
   docker tag fingpt-forecaster:latest yourusername/fingpt-forecaster:latest
   docker push yourusername/fingpt-forecaster:latest
   ```

3. **Deploy on RunPod:**
   - Go to RunPod → Serverless
   - Create new endpoint
   - Use your Docker image: `yourusername/fingpt-forecaster:latest`
   - Set environment variables:
     - `HF_TOKEN`: Your HuggingFace token
     - `FINNHUB_API_KEY`: Your Finnhub API key
     - `ADAPTER_PATH`: `/runpod-volume/fingpt-v3-float16_YYYYMMDDHHMM` (if using custom model)

4. **Get your API endpoint URL** (looks like: `https://api.runpod.ai/v2/xxxxx`)

#### Step 4: Use the API

```python
import requests

response = requests.post(
    "YOUR_RUNPOD_ENDPOINT_URL",
    headers={"Authorization": "Bearer YOUR_RUNPOD_API_KEY"},
    json={
        "input": {
            "ticker": "AAPL",
            "date": "2025-01-15",
            "n_weeks": 3
        }
    }
)

print(response.json())
```

---

### Option 2: Gradio UI on RunPod GPU Pod

**Best for:** Interactive testing, demos, quick access

#### Step 1: Update app.py to Use Your Model

Edit `app.py` and change:
```python
model = PeftModel.from_pretrained(
    base_model,
    'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'  # Old
)
```

To:
```python
# Use your trained model
model = PeftModel.from_pretrained(
    base_model,
    './finetuned_models/fingpt-v3-float16_YYYYMMDDHHMM'  # Your model path
)
```

#### Step 2: Run on RunPod

```bash
cd /workspace/FinMin/fingpt/FinGPT_Forecaster

# Set environment variables
export HF_TOKEN="your_token"
export FINNHUB_API_KEY="your_key"

# Run Gradio
python app.py
```

Gradio will give you a public URL like: `https://xxxxx.gradio.live`

---

### Option 3: Web UI (Flask) - Local or Cloud Server

**Best for:** Custom UI, integration with other services

#### Step 1: Install Dependencies

```bash
pip install flask requests
```

#### Step 2: Set Environment Variables

```bash
export RUNPOD_API_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID"
export RUNPOD_API_KEY="your_runpod_api_key"
```

#### Step 3: Run Web UI

```bash
python web_ui.py
```

Open `http://localhost:5000` in your browser.

#### Step 4: Deploy to Cloud (Optional)

**Heroku:**
```bash
heroku create fingpt-forecaster
heroku config:set RUNPOD_API_URL="..." RUNPOD_API_KEY="..."
git push heroku master
```

**Railway/Render:**
- Connect your GitHub repo
- Set environment variables
- Deploy!

---

## 📋 Environment Variables

### For Serverless (handler.py):
- `HF_TOKEN`: HuggingFace access token
- `FINNHUB_API_KEY`: Finnhub API key
- `ADAPTER_PATH`: (Optional) Path to your custom trained model

### For Web UI (web_ui.py):
- `RUNPOD_API_URL`: Your RunPod Serverless endpoint URL
- `RUNPOD_API_KEY`: Your RunPod API key

---

## 🔧 Troubleshooting

### Model Not Loading
- Check that `HF_TOKEN` has access to Llama-2 (accept license on HuggingFace)
- Verify model path is correct
- Check GPU memory (need 24GB+ for float16)

### API Timeout
- Increase timeout in `web_ui.py` (default: 300s)
- Check RunPod endpoint logs
- Verify model is loaded correctly

### Rate Limiting
- RunPod Serverless has rate limits based on your plan
- Consider using GPU Pod for unlimited requests

---

## 💰 Cost Comparison

| Method | Cost | Best For |
|--------|------|----------|
| **Serverless** | Pay per request (~$0.01-0.05/request) | Production, variable load |
| **GPU Pod** | ~$0.50-2.00/hour | Development, constant use |
| **Web UI (Cloud)** | Free-$5/month | Simple hosting |

---

## 📚 Next Steps

1. **Monitor Usage:** Check RunPod dashboard for API usage
2. **Optimize:** Adjust `max_new_tokens` for faster responses
3. **Scale:** Add more endpoints for higher throughput
4. **Integrate:** Connect to trading bots, dashboards, etc.

---

## 🆘 Support

- RunPod Docs: https://docs.runpod.io
- FinGPT GitHub: https://github.com/AI4Finance-Foundation/FinGPT

