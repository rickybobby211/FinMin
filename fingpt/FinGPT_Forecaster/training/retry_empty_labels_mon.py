"""
Repair empty training labels and optionally rerun generation.

Workflow:
1) Scan files matching *_monfri_gpt-4.csv
2) Remove rows with empty "answer"
3) Optionally call generate_labels_mon.py for affected symbols only

Examples:
    python3 retry_empty_labels_mon.py --data_dir ./raw_data/2023-02-20_2026-02-16 --dry_run

    python3 retry_empty_labels_mon.py --data_dir ./raw_data/2023-02-20_2026-02-16 \
      --backend deepseek --model deepseek-reasoner --parallel 5 --run_generate
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class RepairResult:
    file_name: str
    symbol: str
    removed_rows: int
    total_rows: int
    missing_index_count: int
    missing_indices: List[int]
    removed_for_duplicate_dates: int
    removed_for_explicit_dates: int


class EmptyRowPolicy(ABC):
    """Strategy for deciding if a CSV row should be removed."""

    @abstractmethod
    def should_remove(self, row: Dict[str, str]) -> bool:
        pass


class EmptyAnswerPolicy(EmptyRowPolicy):
    """Remove rows where answer is empty/whitespace."""

    def should_remove(self, row: Dict[str, str]) -> bool:
        return not str(row.get("answer", "")).strip()


class CsvRepairService:
    """Repairs label CSV files using a configurable row policy."""

    def __init__(self, policy: EmptyRowPolicy, backup_suffix: str = ".bak") -> None:
        self.policy = policy
        self.backup_suffix = backup_suffix

    def _extract_present_indices(self, rows: List[Dict[str, str]]) -> List[int]:
        present: List[int] = []
        for row in rows:
            raw = str(row.get("index", "")).strip()
            if not raw:
                continue
            try:
                present.append(int(raw))
            except ValueError:
                continue
        return sorted(set(present))

    def _find_missing_indices(self, rows: List[Dict[str, str]], expected_rows: int) -> List[int]:
        present = self._extract_present_indices(rows)
        if expected_rows <= 0:
            return []
        expected = set(range(expected_rows))
        missing = expected - set(present)
        return sorted(missing)

    def _remove_duplicate_start_dates(self, rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
        counts: Dict[str, int] = {}
        for row in rows:
            date = str(row.get("start_date", "")).strip()
            if not date:
                continue
            counts[date] = counts.get(date, 0) + 1

        duplicated_dates = {d for d, c in counts.items() if c > 1}
        if not duplicated_dates:
            return rows, 0

        kept = [row for row in rows if str(row.get("start_date", "")).strip() not in duplicated_dates]
        removed = len(rows) - len(kept)
        return kept, removed

    def _remove_explicit_start_dates(self, rows: List[Dict[str, str]], drop_dates: Set[str]) -> tuple[List[Dict[str, str]], int]:
        if not drop_dates:
            return rows, 0
        kept = [row for row in rows if str(row.get("start_date", "")).strip() not in drop_dates]
        removed = len(rows) - len(kept)
        return kept, removed

    def repair_file(
        self,
        path: Path,
        expected_rows: int,
        dry_run: bool = False,
        drop_duplicate_start_dates: bool = False,
        drop_start_dates: Set[str] | None = None,
    ) -> RepairResult:
        symbol = path.name.split("_")[0].upper()
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        kept_rows: List[Dict[str, str]] = []
        removed_count = 0
        for row in rows:
            safe_row = {k: (v if v is not None else "") for k, v in row.items()}
            if self.policy.should_remove(safe_row):
                removed_count += 1
            else:
                kept_rows.append(safe_row)

        removed_for_duplicate_dates = 0
        removed_for_explicit_dates = 0

        if drop_duplicate_start_dates:
            kept_rows, removed_for_duplicate_dates = self._remove_duplicate_start_dates(kept_rows)

        clean_drop_dates = {d.strip() for d in (drop_start_dates or set()) if d.strip()}
        if clean_drop_dates:
            kept_rows, removed_for_explicit_dates = self._remove_explicit_start_dates(kept_rows, clean_drop_dates)

        total_removed = removed_count + removed_for_duplicate_dates + removed_for_explicit_dates

        if total_removed > 0 and not dry_run:
            backup_path = path.with_name(path.name + self.backup_suffix)
            if not backup_path.exists():
                shutil.copy2(path, backup_path)

            with open(path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)

        missing_indices = self._find_missing_indices(kept_rows, expected_rows)

        return RepairResult(
            file_name=path.name,
            symbol=symbol,
            removed_rows=removed_count,
            total_rows=len(rows),
            missing_index_count=len(missing_indices),
            missing_indices=missing_indices,
            removed_for_duplicate_dates=removed_for_duplicate_dates,
            removed_for_explicit_dates=removed_for_explicit_dates,
        )


class GenerationRunner:
    """Template Method for invoking label generation after repair."""

    def run(self, symbols: List[str], missing_by_symbol: Dict[str, List[int]], args: argparse.Namespace) -> int:
        if not symbols:
            print("No affected symbols to rerun.")
            return 0
        return self._run_impl(symbols, missing_by_symbol, args)

    @abstractmethod
    def _run_impl(self, symbols: List[str], missing_by_symbol: Dict[str, List[int]], args: argparse.Namespace) -> int:
        pass


class SubprocessGenerationRunner(GenerationRunner):
    """Runs generate_labels_mon.py in a subprocess."""

    def _run_impl(self, symbols: List[str], missing_by_symbol: Dict[str, List[int]], args: argparse.Namespace) -> int:
        script_path = Path(__file__).with_name("generate_labels_mon.py")
        symbols_arg = ",".join(sorted(set(symbols)))
        only_indices_chunks = []
        for symbol in sorted(set(symbols)):
            indices = missing_by_symbol.get(symbol, [])
            if not indices:
                continue
            only_indices_chunks.append(f"{symbol}:{','.join(str(i) for i in indices)}")
        only_indices_arg = ";".join(only_indices_chunks)

        cmd = [
            "python3",
            str(script_path),
            "--data_dir",
            str(args.data_dir),
            "--symbols",
            symbols_arg,
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--parallel",
            str(args.parallel),
            "--news_strategy",
            args.news_strategy,
            "--week_mode",
            args.week_mode,
        ]
        if only_indices_arg:
            cmd.extend(["--only_indices", only_indices_arg])
        if args.pre_filter:
            cmd.append("--pre_filter")

        if args.confirm_only_indices:
            print("\nTargeted retry payload (--only_indices):")
            print(only_indices_arg if only_indices_arg else "(empty)")
            if not args.yes:
                user_input = input("Continue and run generation? [y/N]: ").strip().lower()
                if user_input not in ("y", "yes"):
                    print("Aborted by user.")
                    return 0

        print("\nRunning generator for affected symbols:")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair empty answers and rerun label generation.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing label CSV files.")
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help='Optional comma-separated symbols to process (default: "all").',
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_monfri_gpt-4.csv",
        help="File pattern to repair (default: *_monfri_gpt-4.csv).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only report, do not modify files.")
    parser.add_argument("--run_generate", action="store_true", help="After repair, rerun generation for affected symbols.")
    parser.add_argument(
        "--drop_duplicate_start_dates",
        action="store_true",
        help="Remove all rows belonging to duplicate start_date values within each file.",
    )
    parser.add_argument(
        "--drop_start_dates",
        type=str,
        default="",
        help='Comma-separated start_date values to remove entirely (example: "2024-01-22,2025-06-09").',
    )
    parser.add_argument(
        "--confirm_only_indices",
        action="store_true",
        help="Print exact --only_indices payload before starting generation.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt (used with --confirm_only_indices).",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=155,
        help="Expected number of prompts (index range 0..N-1), default: 155.",
    )
    parser.add_argument("--backend", type=str, default="deepseek", choices=["openai", "deepseek", "ollama"])
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--news_strategy", type=str, default="llm", choices=["relevant", "random", "llm"])
    parser.add_argument("--pre_filter", action="store_true")
    parser.add_argument(
        "--week_mode",
        type=str,
        default="mon_fri_preopen",
        choices=["fri_fri", "mon_fri_preopen"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: Invalid directory: {data_dir}")
        return 2

    files = sorted(data_dir.glob(args.pattern))
    if not files:
        print(f"No matching files in {data_dir} for pattern: {args.pattern}")
        return 0

    if args.symbols != "all":
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        files = [p for p in files if p.name.split("_")[0].upper() in wanted]
        if not files:
            print(f"No matching files for symbols: {', '.join(sorted(wanted))}")
            return 0

    repair_service = CsvRepairService(policy=EmptyAnswerPolicy())
    explicit_drop_dates = {d.strip() for d in args.drop_start_dates.split(",") if d.strip()}
    results: List[RepairResult] = [
        repair_service.repair_file(
            file_path,
            expected_rows=args.expected_rows,
            dry_run=args.dry_run,
            drop_duplicate_start_dates=args.drop_duplicate_start_dates,
            drop_start_dates=explicit_drop_dates,
        )
        for file_path in files
    ]

    affected = [r for r in results if r.removed_rows > 0 or r.missing_index_count > 0]
    total_removed = sum(r.removed_rows for r in results)
    total_removed_duplicates = sum(r.removed_for_duplicate_dates for r in results)
    total_removed_explicit_dates = sum(r.removed_for_explicit_dates for r in results)
    total_missing = sum(r.missing_index_count for r in results)

    print("=" * 72)
    print("Empty label repair report")
    print("=" * 72)
    print(f"Directory: {data_dir}")
    print(f"Pattern:   {args.pattern}")
    print(f"Files:     {len(files)}")
    print(f"Removed:   {total_removed} rows")
    print(f"Removed duplicate dates: {total_removed_duplicates} rows")
    print(f"Removed explicit dates:  {total_removed_explicit_dates} rows")
    print(f"Missing:   {total_missing} index slots (0..{args.expected_rows - 1})")
    print("-" * 72)
    for r in results:
        print(
            f"{r.file_name}: removed={r.removed_rows}/{r.total_rows}, "
            f"dup_date_removed={r.removed_for_duplicate_dates}, "
            f"explicit_date_removed={r.removed_for_explicit_dates}, "
            f"missing_index={r.missing_index_count}"
        )

    if args.dry_run:
        print("\nDry run only. No files were modified.")
    elif total_removed > 0:
        print("\nBackups created next to modified files with suffix: .bak")

    affected_symbols = [r.symbol for r in affected]
    missing_by_symbol = {r.symbol: r.missing_indices for r in affected}
    if affected_symbols:
        print(f"Affected symbols: {', '.join(sorted(set(affected_symbols)))}")
    else:
        print("No empty answers or missing index gaps found.")

    if args.run_generate and not args.dry_run:
        runner = SubprocessGenerationRunner()
        return runner.run(affected_symbols, missing_by_symbol, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
