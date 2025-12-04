# FinGPT Forecaster - Training Guide

This guide explains how to train the FinGPT Forecaster model on the latest market data.

## Overview

The training pipeline consists of 4 steps:

```
1. Prepare Data     →  2. Generate Labels  →  3. Build Dataset  →  4. Train Model
(fetch news/prices)    (GPT-4 analysis)       (format for Llama)    (LoRA fine-tune)
```

## Prerequisites

### API Keys Required

1. **Finnhub API Key** (Free tier available)
   - Sign up at: https://finnhub.io/register
   - Used for: company news, profiles, and financial metrics

2. **OpenAI API Key** (Paid - GPT-4 access required)
   - Get it at: https://platform.openai.com/api-keys
   - Used for: generating training labels
   - **Cost estimate**: ~$0.03 per sample × ~600 samples = ~$18 for DOW30

3. **Hugging Face Token** (Free)
   - Get it at: https://huggingface.co/settings/tokens
   - Request Llama 2 access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

4. **Weights & Biases** (Optional, for logging)
   - Sign up at: https://wandb.ai/

### Hardware Requirements

| Stage | Hardware | Time Estimate |
|-------|----------|---------------|
| Data Prep | CPU only | ~30 min for DOW30 |
| Label Gen | CPU only | ~2-3 hours for DOW30 |
| Training | GPU (24GB+ VRAM) | ~4-6 hours on RTX 4090 |

**Recommended**: RunPod with RTX 4090 (~$0.74/hr)

## Step-by-Step Instructions

### Step 1: Setup Environment

```bash
cd fingpt/FinGPT_Forecaster/training
pip install -r requirements_training.txt
```

Set your API keys:
```bash
export FINNHUB_API_KEY="your_finnhub_key"
export OPENAI_API_KEY="your_openai_key"
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # optional
```

### Step 2: Prepare Raw Data

Fetch stock prices, news, and financials for your date range:

```bash
# Example: Last 6 months of data
python prepare_latest_data.py \
    --start_date 2024-05-01 \
    --end_date 2024-11-01 \
    --symbols dow30
```

**Options:**
- `--symbols dow30` - DOW 30 companies (default)
- `--symbols tech` - Popular tech stocks
- `--symbols custom --custom_symbols "AAPL,MSFT,GOOGL"` - Custom list
- `--no_basics` - Skip financial metrics (faster, less context)

**Output:** `./raw_data/2024-05-01_2024-11-01/*.csv`

### Step 3: Generate GPT-4 Labels

Use GPT-4 to generate analysis and predictions for each data point:

```bash
python generate_labels.py \
    --data_dir ./raw_data/2024-05-01_2024-11-01 \
    --model gpt-4o-mini \
    --min_weeks 1 \
    --max_weeks 4
```

**Options:**
- `--model gpt-4` - Most accurate but expensive (~$0.06/sample)
- `--model gpt-4o-mini` - Good balance (~$0.001/sample) ✅ Recommended
- `--model gpt-4-turbo` - Fast and capable (~$0.02/sample)
- `--symbols "AAPL,MSFT"` - Process specific symbols only

**Output:** `./raw_data/2024-05-01_2024-11-01/*_gpt-4.csv`

### Step 4: Build Training Dataset

Convert labeled data to Llama2 format:

```bash
python build_dataset.py \
    --data_dir ./raw_data/2024-05-01_2024-11-01 \
    --output_name fingpt-forecaster-2024 \
    --train_ratio 0.8
```

**Output:** `./datasets/fingpt-forecaster-2024/`

### Step 5: Train the Model

#### Option A: Single GPU (RTX 4090)

```bash
cd ..  # Back to FinGPT_Forecaster directory
bash training/train_runpod.sh
```

Or manually:
```bash
python train_lora.py \
    --run_name my-fingpt-v1 \
    --base_model llama2 \
    --dataset fingpt-forecaster-2024 \
    --max_length 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --warmup_ratio 0.03 \
    --scheduler constant \
    --ds_config config.json
```

#### Option B: Multi-GPU (with DeepSpeed)

```bash
deepspeed --include localhost:0,1 train_lora.py \
    --run_name my-fingpt-v1 \
    --base_model llama2 \
    --dataset fingpt-forecaster-2024 \
    --max_length 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --ds_config config.json
```

**Output:** `./finetuned_models/my-fingpt-v1_YYYYMMDDHHMM/`

## Using Your Trained Model

After training, use your model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    './finetuned_models/my-fingpt-v1_YYYYMMDDHHMM'
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
```

Or update `app.py` to use your model:
```python
model = PeftModel.from_pretrained(
    base_model,
    './finetuned_models/my-fingpt-v1_YYYYMMDDHHMM'  # Your model path
)
```

## Uploading to Hugging Face

Share your model on Hugging Face:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./finetuned_models/my-fingpt-v1_YYYYMMDDHHMM",
    repo_id="your-username/fingpt-forecaster-custom",
    repo_type="model"
)
```

## Cost Estimates

| Component | Cost |
|-----------|------|
| Finnhub API | Free (60 calls/min) |
| GPT-4o-mini labels (600 samples) | ~$0.60 |
| GPT-4 labels (600 samples) | ~$36 |
| RunPod 4090 (6 hours) | ~$4.50 |
| **Total (budget)** | **~$5-10** |
| **Total (premium GPT-4)** | **~$40-45** |

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use 8-bit training: add `load_in_8bit=True` to model loading

### Slow Training

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`

### API Rate Limits

- Finnhub free tier: 60 calls/minute (script handles this)
- OpenAI: Adjust retry logic in `generate_labels.py` if needed

## Tips for Better Results

1. **More data = better model**: Use longer date ranges (6-12 months)
2. **Diverse stocks**: Include companies from different sectors
3. **Quality labels**: GPT-4 produces better labels than GPT-4-mini
4. **Longer context**: Use `--max_weeks 4` for more historical context
5. **Multiple epochs**: 5-8 epochs usually works well

## File Structure

```
training/
├── prepare_latest_data.py   # Step 1: Fetch raw data
├── generate_labels.py       # Step 2: GPT-4 labeling
├── build_dataset.py         # Step 3: Format dataset
├── train_runpod.sh          # Step 4: Training script
├── requirements_training.txt
└── README.md                # This file

raw_data/
└── 2024-05-01_2024-11-01/
    ├── AAPL_2024-05-01_2024-11-01.csv
    ├── AAPL_gpt-4.csv
    └── ...

datasets/
└── fingpt-forecaster-2024/
    ├── train/
    └── test/

finetuned_models/
└── my-fingpt-v1_202411281200/
    ├── adapter_config.json
    ├── adapter_model.bin
    └── ...
```

## Questions?

- GitHub Issues: https://github.com/AI4Finance-Foundation/FinGPT/issues
- Discord: https://discord.gg/trsr8SXpW5

