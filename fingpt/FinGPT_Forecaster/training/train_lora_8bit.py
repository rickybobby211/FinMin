"""
FinGPT Forecaster - 8-bit Training Script for RTX 3080 (10GB VRAM)
===================================================================
This script uses 8-bit quantization to fit Llama-2-7b training on GPUs
with limited VRAM (10-12GB).

Usage:
    python train_lora_8bit.py --dataset fingpt-forecaster-2024 --base_model llama2
"""

import os
import re
import sys
import torch
import argparse
import datasets
from datetime import datetime
from functools import partial
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, skipping logging")


# ============================================================================
# CONFIGURATION
# ============================================================================

LORA_MODULES = {
    'chatglm2': ['query_key_value'],
    'llama2': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
}

MODEL_NAMES = {
    'chatglm2': 'THUDM/chatglm2-6b',
    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
}

# Pre-trained FinGPT adapter (trained on DOW30 2022-2023 data)
FINGPT_ADAPTER = 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tokenize(args, tokenizer, feature):
    """Tokenize a single training example."""
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), 
        padding=False,
        max_length=args.max_length, 
        truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), 
        padding=False,
        max_length=args.max_length, 
        truncation=True, 
        add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
    # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def load_dataset_from_disk(dataset_path):
    """Load dataset from local disk."""
    # Check common locations
    possible_paths = [
        dataset_path,
        f"./datasets/{dataset_path}",
        f"./data/{dataset_path}",
        f"../data/{dataset_path}",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            return datasets.load_from_disk(path)
    
    raise FileNotFoundError(f"Dataset not found. Tried: {possible_paths}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    print("\n" + "=" * 60)
    print("FinGPT Forecaster - 8-bit Training")
    print("=" * 60)
    
    # Get model name
    model_name = MODEL_NAMES.get(args.base_model)
    if not model_name:
        raise ValueError(f"Unknown base model: {args.base_model}")
    
    print(f"Base Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Max Length: {args.max_length}")
    print("=" * 60 + "\n")
    
    # ========================================================================
    # Load Model with 8-bit Quantization
    # ========================================================================
    print("Loading model with 8-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded. Device: {model.device}")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # ========================================================================
    # Load Tokenizer
    # ========================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("\nLoading dataset...")
    dataset = load_dataset_from_disk(args.dataset)
    
    if 'test' not in dataset:
        dataset = dataset.train_test_split(0.1, shuffle=True, seed=42)
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Tokenize
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        partial(tokenize, args, tokenizer),
        remove_columns=['prompt', 'answer', 'label', 'symbol', 'period']
    )
    
    # Filter out samples that exceed max length
    original_len = len(tokenized_dataset['train'])
    tokenized_dataset = tokenized_dataset.filter(lambda x: not x['exceed_max_length'])
    tokenized_dataset = tokenized_dataset.remove_columns(['exceed_max_length'])
    
    print(f"Filtered: {original_len} -> {len(tokenized_dataset['train'])} samples")
    
    # ========================================================================
    # Setup LoRA (or load pre-trained FinGPT adapter)
    # ========================================================================
    
    if args.from_pretrained_adapter and args.base_model == 'llama2':
        print("\n✅ Loading pre-trained FinGPT-Forecaster adapter...")
        print(f"   Adapter: {FINGPT_ADAPTER}")
        print("   This preserves knowledge from DOW30 2022-2023 training!")
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            FINGPT_ADAPTER,
            is_trainable=True  # Enable training on top of existing adapter
        )
        print("   Pre-trained adapter loaded successfully!")
    else:
        print("\nSetting up fresh LoRA adapter...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,                    # LoRA rank
            lora_alpha=16,          # LoRA alpha
            lora_dropout=0.1,       # Dropout
            target_modules=LORA_MODULES[args.base_model],
            bias='none',
        )
        
        model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    # ========================================================================
    # Training Arguments
    # ========================================================================
    current_time = datetime.now().strftime('%Y%m%d%H%M')
    output_dir = f'finetuned_models/{args.run_name}_{current_time}'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps if isinstance(args.eval_steps, int) else int(args.eval_steps * len(tokenized_dataset['train'])),
        eval_steps=args.eval_steps if isinstance(args.eval_steps, int) else int(args.eval_steps * len(tokenized_dataset['train'])),
        eval_strategy=args.evaluation_strategy,
        fp16=True,              # Use FP16 for speed
        optim="adamw_torch",    # Optimizer
        remove_unused_columns=False,
        report_to='wandb' if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY') else 'none',
        run_name=args.run_name,
        # Memory optimizations for RTX 3080
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # Reduce for Windows compatibility
    )
    
    # ========================================================================
    # Train
    # ========================================================================
    print("\nStarting training...")
    print(f"Output directory: {output_dir}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            padding=True,
            return_tensors="pt"
        ),
    )
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Train!
    trainer.train()
    
    # ========================================================================
    # Save Model
    # ========================================================================
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print("\nTo use your model:")
    print(f"  model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FinGPT Forecaster with 8-bit quantization')
    
    # Required
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name or path")
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2'])
    
    # Training params
    parser.add_argument("--run_name", default='fingpt-3080', type=str)
    parser.add_argument("--max_length", default=2048, type=int, help="Max sequence length (reduced for 10GB VRAM)")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_epochs", default=5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--scheduler", default='constant', type=str)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--eval_steps", default=0.2, type=float)
    parser.add_argument("--from_pretrained_adapter", action='store_true', 
                        help="Start from pre-trained FinGPT adapter instead of base Llama")
    
    args = parser.parse_args()
    
    # Login to wandb if available
    if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
        wandb.login()
    
    main(args)

