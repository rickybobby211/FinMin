"""
FinGPT Forecaster - Dataset Builder Script
==========================================
This script converts GPT-4 labeled data into Llama2 training format
and creates HuggingFace datasets for training.

Prerequisites:
- Run prepare_latest_data.py and generate_labels.py first

Usage:
    python build_dataset.py --data_dir ./raw_data/2024-01-01_2024-11-01 --output_name fingpt-forecaster-2024
"""

import os
import re
import json
import argparse
import datasets
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict


# ============================================================================
# CONFIGURATION
# ============================================================================

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]
Prediction: ...
Analysis: ..."""


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def gpt4_to_llama(symbol: str, data_dir: str, with_basics: bool = True, format_type: str = 'llama2') -> dict:
    """
    Convert GPT-4 labeled data to Llama2 training format.
    
    Args:
        symbol: Ticker symbol
        data_dir: Directory containing GPT-4 labeled CSV files
        with_basics: Whether to use files with basic financials
        format_type: 'llama2' (default) or 'qwen'
    
    Returns:
        Dictionary with prompts, answers, periods, labels, and symbols
    """
    suffix = "" if with_basics else "_nobasics"
    csv_file = f'{data_dir}/{symbol}{suffix}_gpt-4.csv'
    
    if not os.path.exists(csv_file):
        print(f"    Warning: No GPT-4 file found for {symbol}")
        return None
    
    df = pd.read_csv(csv_file)
    
    prompts, answers, periods, labels, symbols = [], [], [], [], []
    
    for i, row in df.iterrows():
        prompt, answer = row['prompt'], row['answer']
        
        if pd.isna(answer) or not answer.strip():
            continue
        
        # Extract period and label from the original prompt
        res = re.search(
            r"Then let's assume your prediction for next week \((.*)\) is ((?:up|down) by .*%)\.", 
            prompt
        )
        
        if not res:
            print(f"    Warning: Could not parse prompt {i} for {symbol}")
            continue
        
        period, label = res.group(1), res.group(2)
        
        # Transform the prompt: remove the "assume prediction" part
        prompt = re.sub(
            r"Then let's assume your prediction for next week \((.*)\) is (?:up|down) by (?:.*%)%. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.",
            f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction.",
            prompt
        )
        
        # Transform the answer: add explicit prediction line
        try:
            answer = re.sub(
                r"\[Prediction & Analysis\]:\s*",
                f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
                answer
            )
        except Exception as e:
            print(f"    Warning: Could not transform answer {i} for {symbol}: {e}")
            continue
        
        # Format based on model type
        if format_type == 'qwen':
            # Qwen/ChatML format
            # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            answer = f"{answer}<|im_end|>"
        else:
            # Llama2 format (default)
            prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
        
        prompts.append(prompt)
        answers.append(answer)
        periods.append(period)
        labels.append(label)
        symbols.append(symbol)
    
    if not prompts:
        return None
    
    return {
        "prompt": prompts,
        "answer": answers,
        "period": periods,
        "label": labels,
        "symbol": symbols,
    }


def create_dataset(
    data_dir: str,
    symbols: list = None,
    train_ratio: float = 0.8,
    with_basics: bool = True,
    format_type: str = 'llama2'
) -> DatasetDict:
    """
    Create a HuggingFace dataset from all processed symbols.
    
    Args:
        data_dir: Directory containing GPT-4 labeled CSV files
        symbols: List of symbols to include (None = all available)
        train_ratio: Ratio of data to use for training
        with_basics: Whether to use files with basic financials
        format_type: 'llama2' or 'qwen'
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    # Find all available symbols if not specified
    if symbols is None:
        suffix = "" if with_basics else "_nobasics"
        csv_files = list(Path(data_dir).glob(f"*{suffix}_gpt-4.csv"))
        symbols = [f.stem.replace(suffix + "_gpt-4", "") for f in csv_files]
    
    train_dataset_list = []
    test_dataset_list = []
    
    print(f"\nProcessing {len(symbols)} symbols...")
    
    for symbol in symbols:
        data_dict = gpt4_to_llama(symbol, data_dir, with_basics, format_type)
        
        if data_dict is None:
            continue
        
        dataset = Dataset.from_dict(data_dict)
        train_size = round(train_ratio * len(dataset))
        
        if train_size > 0 and train_size < len(dataset):
            train_dataset_list.append(dataset.select(range(train_size)))
            test_dataset_list.append(dataset.select(range(train_size, len(dataset))))
        elif train_size == len(dataset):
            train_dataset_list.append(dataset)
        
        print(f"    {symbol}: {len(dataset)} samples ({train_size} train, {len(dataset) - train_size} test)")
    
    if not train_dataset_list:
        raise ValueError("No valid data found!")
    
    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list) if test_dataset_list else train_dataset.select(range(0))
    
    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build training dataset for FinGPT Forecaster')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing GPT-4 labeled data')
    parser.add_argument('--output_name', type=str, required=True, help='Name for the output dataset')
    parser.add_argument('--output_dir', type=str, default='./datasets', help='Directory to save dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--symbols', type=str, default='all',
                        help='Comma-separated symbols or "all"')
    parser.add_argument('--no_basics', action='store_true', help='Use nobasics files')
    parser.add_argument('--format', type=str, default='llama2', choices=['llama2', 'qwen'],
                        help='Prompt format: llama2 (default) or qwen')
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols == 'all':
        symbols = None
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    with_basics = not args.no_basics
    
    print("=" * 60)
    print("FinGPT Forecaster - Dataset Builder")
    print("=" * 60)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Name: {args.output_name}")
    print(f"Train Ratio: {args.train_ratio}")
    print(f"Include Financials: {with_basics}")
    print(f"Format: {args.format}")
    print("=" * 60)
    
    # Create dataset
    dataset = create_dataset(
        args.data_dir,
        symbols=symbols,
        train_ratio=args.train_ratio,
        with_basics=with_basics,
        format_type=args.format
    )
    
    # Save dataset
    output_path = f"{args.output_dir}/{args.output_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    print("\n" + "=" * 60)
    print("DATASET CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    
    # Show sample
    print("\n--- Sample Training Entry ---")
    if len(dataset['train']) > 0:
        sample = dataset['train'][0]
        print(f"Symbol: {sample['symbol']}")
        print(f"Period: {sample['period']}")
        print(f"Label: {sample['label']}")
        print(f"Prompt length: {len(sample['prompt'])} chars")
        print(f"Answer length: {len(sample['answer'])} chars")
    
    print("\n" + "=" * 60)
    print("Next step: Run training with train_lora.py")
    print(f"  python train_lora.py --dataset {args.output_name} --base_model llama2")
    print("=" * 60)


if __name__ == "__main__":
    main()

