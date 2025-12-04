#!/bin/bash
# ============================================================================
# FinGPT Forecaster - Training Script for RunPod (RTX 4090)
# ============================================================================
# This script is optimized for a single RTX 4090 GPU (24GB VRAM)
#
# Usage:
#   1. Upload your dataset to ./datasets/
#   2. Set your WANDB_API_KEY (optional, for logging)
#   3. Run: bash train_runpod.sh
# ============================================================================

# Environment setup
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0

# Optional: Weights & Biases logging
# export WANDB_API_KEY="your_wandb_key"
# export WANDB_PROJECT="fingpt-forecaster"

# ============================================================================
# CONFIGURATION - Modify these as needed
# ============================================================================
DATASET_NAME="fingpt-forecaster-2024"       # Your dataset name in ./datasets/
RUN_NAME="fingpt-forecaster-4090-v1"        # Name for this training run
BASE_MODEL="llama2"                          # llama2 or chatglm2
MAX_LENGTH=4096                              # Max sequence length
BATCH_SIZE=1                                 # Batch size per device (1 for 4090)
GRAD_ACCUM=16                                # Gradient accumulation steps
LEARNING_RATE=5e-5                           # Learning rate
NUM_EPOCHS=5                                 # Number of training epochs
WARMUP_RATIO=0.03                            # Warmup ratio

# ============================================================================
# RUN TRAINING
# ============================================================================
echo "=============================================="
echo "FinGPT Forecaster Training"
echo "=============================================="
echo "Dataset: $DATASET_NAME"
echo "Run Name: $RUN_NAME"
echo "Base Model: $BASE_MODEL"
echo "Batch Size: $BATCH_SIZE x $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "=============================================="

python train_lora.py \
    --run_name "$RUN_NAME" \
    --base_model "$BASE_MODEL" \
    --dataset "$DATASET_NAME" \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --scheduler constant \
    --evaluation_strategy steps \
    --eval_steps 0.1 \
    --log_interval 10 \
    --ds_config config.json

echo "=============================================="
echo "Training Complete!"
echo "Model saved to: ./finetuned_models/${RUN_NAME}_*"
echo "=============================================="

