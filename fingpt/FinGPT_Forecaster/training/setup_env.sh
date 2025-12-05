#!/bin/bash
# FinGPT Training - Environment Setup
# Run this script once per terminal session: source setup_env.sh
# Or add these exports to your ~/.bashrc for permanent setup

# Required: OpenAI API key for generating training labels
# Get your key at: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_openai_api_key_here"

# Required: Finnhub API key for fetching news and financials  
# Get a free key at: https://finnhub.io/register
export FINNHUB_API_KEY="your_finnhub_api_key_here"

# Optional: Hugging Face token for accessing Llama-2 and uploading models
# Get your token at: https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token_here"

# Optional: Weights & Biases for training logging
export WANDB_API_KEY="your_wandb_api_key_here"

echo "Environment variables set!"
echo "  OPENAI_API_KEY: $(if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_openai_api_key_here" ]; then echo "✓ Set"; else echo "✗ Not set"; fi)"
echo "  FINNHUB_API_KEY: $(if [ -n "$FINNHUB_API_KEY" ] && [ "$FINNHUB_API_KEY" != "your_finnhub_api_key_here" ]; then echo "✓ Set"; else echo "✗ Not set"; fi)"
echo "  HF_TOKEN: $(if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then echo "✓ Set"; else echo "✗ Not set"; fi)"

