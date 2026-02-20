"""
Print all train/test weeks for one symbol from a saved HuggingFace dataset.

Example:
  /home/ricky/CursorProjects/FinMin/venv/bin/python print_symbol_weeks.py \
    --dataset_dir ./datasets/fingpt-qwen-2026-mon \
    --symbol TSM
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all period weeks for one symbol in train/test splits."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory created by build_dataset_mon.py",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Ticker symbol to print, e.g. TSM",
    )
    parser.add_argument(
        "--show_index",
        action="store_true",
        help="Prefix each printed row with running index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.strip().upper()

    ds = load_from_disk(args.dataset_dir)
    if "train" not in ds or "test" not in ds:
        print("ERROR: Dataset must contain 'train' and 'test' splits.")
        return 2

    results = defaultdict(list)
    for split in ("train", "test"):
        for row in ds[split]:
            if str(row.get("symbol", "")).upper() == symbol:
                results[split].append(str(row.get("period", "")))

    if not results["train"] and not results["test"]:
        print(f"No rows found for symbol: {symbol}")
        return 1

    print("=" * 80)
    print(f"Symbol: {symbol}")
    print("=" * 80)
    for split in ("train", "test"):
        periods = results[split]
        print(f"\n[{split.upper()}] count={len(periods)}")
        if not periods:
            print("  (none)")
            continue
        for i, period in enumerate(periods, start=1):
            if args.show_index:
                print(f"{i:03d}. {period}")
            else:
                print(period)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
