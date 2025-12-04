# FinGPT Forecaster - RunPod Serverless Deployment

Deploy FinGPT Forecaster as a serverless API. Pay only when requests are made!

## 💰 Pricing

| | GPU Pod (always on) | Serverless |
|---|---|---|
| **Idle cost** | ~$5/day | $0 |
| **Per request** | N/A | ~$0.02-0.05 |
| **Best for** | Heavy usage | Occasional use |

## 🚀 Quick Deploy (Recommended)

### Option 1: Use Pre-built Image

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:
   - **Docker Image**: `your-dockerhub/fingpt-forecaster:latest`
   - **GPU**: RTX 3090 or RTX 4090
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 3
4. Add Environment Variables:
   ```
   HF_TOKEN=your_huggingface_token
   FINNHUB_API_KEY=your_finnhub_key
   ```
5. Deploy!

### Option 2: Build Your Own Image

```bash
cd fingpt/FinGPT_Forecaster/serverless

# Build
docker build -t your-dockerhub/fingpt-forecaster:latest .

# Push to Docker Hub
docker login
docker push your-dockerhub/fingpt-forecaster:latest
```

## 📡 API Usage

### Request Format

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "ticker": "AAPL",
      "date": "2025-12-04",
      "n_weeks": 3
    }
  }'
```

### Response Format

```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "ticker": "AAPL",
    "date": "2025-12-04",
    "prediction": "[Positive Developments]:\n1. Tesla adding CarPlay support...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]\nPrediction: Up by 1-2%\nAnalysis: ..."
  }
}
```

### Python Client

```python
import runpod

runpod.api_key = "YOUR_RUNPOD_API_KEY"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Synchronous request
result = endpoint.run_sync({
    "input": {
        "ticker": "AAPL",
        "date": "2025-12-04",
        "n_weeks": 3
    }
})

print(result["prediction"])
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for Llama 2 |
| `FINNHUB_API_KEY` | No | For news (works without, just no news) |

### Using Custom Trained Model

Edit `handler.py`:

```python
# Change from official adapter:
ADAPTER_ID = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

# To your custom trained adapter:
ADAPTER_ID = "/runpod-volume/fingpt-2025"  # From network volume
# or
ADAPTER_ID = "your-hf-username/fingpt-custom"  # From HuggingFace
```

## 🔧 Advanced: Network Volume

For faster cold starts, pre-load the model to a network volume:

1. Create Network Volume in RunPod
2. Attach to a GPU Pod
3. Download model:
   ```bash
   python -c "
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   
   model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
   model = PeftModel.from_pretrained(model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
   model.save_pretrained('/runpod-volume/fingpt-model')
   "
   ```
4. Update `handler.py` to load from `/runpod-volume/fingpt-model`
5. Attach volume to Serverless endpoint

## 📊 Performance

| Metric | Value |
|--------|-------|
| Cold start | ~60-90 seconds |
| Warm request | ~10-20 seconds |
| Max concurrent | Configurable |

## 🐛 Troubleshooting

**"CUDA out of memory"**
- Use RTX 3090/4090 (24GB VRAM)

**"Model not found"**
- Check HF_TOKEN is set correctly
- Ensure you have Llama 2 access on HuggingFace

**Slow cold start**
- Use Network Volume with pre-loaded model
- Increase min workers to 1 (costs more but no cold start)

