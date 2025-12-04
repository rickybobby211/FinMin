#!/bin/bash
# ============================================================================
# FinGPT Forecaster - Training Script for RTX 3080 (10GB VRAM)
# ============================================================================
# Optimized for limited VRAM using 8-bit quantization
#
# Usage:
#   1. Prepare your dataset first (Steps 1-3)
#   2. Run: bash train_local_3080.sh
# ============================================================================

# Environment setup
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
export CUDA_VISIBLE_DEVICES=0

# Optional: Weights & Biases logging
# export WANDB_API_KEY="your_wandb_key"
# export WANDB_PROJECT="fingpt-forecaster"

# ============================================================================
# CONFIGURATION - Optimized for RTX 3080 (10GB VRAM)
# ============================================================================
DATASET_NAME="fingpt-forecaster-2024"       # Your dataset name in ./datasets/
RUN_NAME="fingpt-forecaster-3080-v1"        # Name for this training run
BASE_MODEL="llama2"                          # llama2 or chatglm2
MAX_LENGTH=2048                              # Reduced from 4096 to save memory
BATCH_SIZE=1                                 # Must be 1 for 10GB VRAM
GRAD_ACCUM=32                                # Higher to compensate for batch_size=1
LEARNING_RATE=5e-5                           # Learning rate
NUM_EPOCHS=5                                 # Number of training epochs
WARMUP_RATIO=0.03                            # Warmup ratio

# ============================================================================
# RUN TRAINING (8-bit mode)
# ============================================================================
echo "=============================================="
echo "FinGPT Forecaster Training (RTX 3080 Mode)"
echo "=============================================="
echo "Dataset: $DATASET_NAME"
echo "Run Name: $RUN_NAME"
echo "Base Model: $BASE_MODEL (8-bit quantized)"
echo "Max Length: $MAX_LENGTH (reduced for VRAM)"
echo "Batch Size: $BATCH_SIZE x $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "=============================================="

python train_lora_8bit.py \
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
    --eval_steps 0.2 \
    --log_interval 10 \
    --from_pretrained_adapter  # Start from FinGPT's pre-trained model!

echo "=============================================="
echo "Training Complete!"
echo "Model saved to: ./finetuned_models/${RUN_NAME}_*"
echo "=============================================="

