"""
Check training label CSV files for empty rows/fields.

Pattern example:
    [ticker]_monfri_gpt-4.csv

Usage:
    python check_empty_labels_mon.py --data_dir ./raw_data/2023-02-20_2026-02-16
"""

from __future__ import annotations

import argparse
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Issue:
    file: str
    symbol: str
    row_number: int
    row_index_value: str
    issue_type: str
    preview: str


class RowCheckStrategy(ABC):
    """Strategy interface for row-level checks."""

    issue_type: str

    @abstractmethod
    def check(self, row: Dict[str, str]) -> bool:
        """Return True if row violates this strategy."""


class FullyEmptyRowStrategy(RowCheckStrategy):
    issue_type = "fully_empty_row"

    def check(self, row: Dict[str, str]) -> bool:
        for value in row.values():
            if str(value).strip():
                return False
        return True


class EmptyAnswerStrategy(RowCheckStrategy):
    issue_type = "empty_answer"

    def check(self, row: Dict[str, str]) -> bool:
        return not str(row.get("answer", "")).strip()


class EmptyPromptStrategy(RowCheckStrategy):
    issue_type = "empty_prompt"

    def check(self, row: Dict[str, str]) -> bool:
        return not str(row.get("prompt", "")).strip()


class LabelFileAuditor:
    """Runs configured strategies against each CSV row."""

    def __init__(self, strategies: List[RowCheckStrategy]) -> None:
        self.strategies = strategies

    @staticmethod
    def _safe_str(value: object) -> str:
        return "" if value is None else str(value).strip()

    def _build_issue(
        self,
        path: Path,
        symbol: str,
        row_number: int,
        row: Dict[str, str],
        issue_type: str,
    ) -> Issue:
        row_index_value = self._safe_str(row.get("index", ""))
        preview_source = self._safe_str(row.get("prediction", "")) or self._safe_str(row.get("end_date", ""))
        preview = preview_source[:120]
        return Issue(
            file=path.name,
            symbol=symbol,
            row_number=row_number,
            row_index_value=row_index_value,
            issue_type=issue_type,
            preview=preview,
        )

    def audit_file(self, path: Path) -> List[Issue]:
        symbol = path.name.split("_")[0].upper()
        issues: List[Issue] = []

        try:
            with open(path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    row_number = row_idx + 2  # +1 for header, +1 for 0-index
                    clean_row = {k: (v if v is not None else "") for k, v in row.items()}
                    for strategy in self.strategies:
                        if strategy.check(clean_row):
                            issues.append(
                                self._build_issue(path, symbol, row_number, clean_row, strategy.issue_type)
                            )
        except Exception as exc:
            issues.append(
                Issue(
                    file=path.name,
                    symbol=symbol,
                    row_number=0,
                    row_index_value="",
                    issue_type="read_error",
                    preview=str(exc)[:120],
                )
            )

        return issues


def summarize(issues: List[Issue]) -> Dict[str, Dict[str, int]]:
    per_file: Dict[str, Dict[str, int]] = {}
    for issue in issues:
        file_stats = per_file.setdefault(issue.file, {})
        file_stats[issue.issue_type] = file_stats.get(issue.issue_type, 0) + 1
    return per_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check *_monfri_gpt-4.csv for empty label rows.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing label CSV files.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_monfri_gpt-4.csv",
        help="Glob pattern for files to inspect (default: *_monfri_gpt-4.csv).",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="",
        help="Optional path to write detailed issue report as CSV.",
    )
    parser.add_argument(
        "--fail_on_issues",
        action="store_true",
        help="Exit with code 1 if any issue is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: Invalid directory: {data_dir}")
        return 2

    files = sorted(data_dir.glob(args.pattern))
    if not files:
        print(f"No files found in {data_dir} with pattern: {args.pattern}")
        return 0

    auditor = LabelFileAuditor(
        strategies=[
            FullyEmptyRowStrategy(),
            EmptyAnswerStrategy(),
            EmptyPromptStrategy(),
        ]
    )

    all_issues: List[Issue] = []
    for file_path in files:
        all_issues.extend(auditor.audit_file(file_path))

    summary = summarize(all_issues)
    total_issues = len(all_issues)

    print("=" * 72)
    print("Training label audit")
    print("=" * 72)
    print(f"Directory: {data_dir}")
    print(f"Pattern:   {args.pattern}")
    print(f"Files:     {len(files)}")
    print(f"Issues:    {total_issues}")
    print("-" * 72)

    for file_path in files:
        name = file_path.name
        stats = summary.get(name, {})
        empty_answer = stats.get("empty_answer", 0)
        fully_empty = stats.get("fully_empty_row", 0)
        empty_prompt = stats.get("empty_prompt", 0)
        print(
            f"{name}: empty_answer={empty_answer}, fully_empty_row={fully_empty}, empty_prompt={empty_prompt}"
        )

    if total_issues:
        print("-" * 72)
        print("First 30 issues:")
        for issue in all_issues[:30]:
            idx = f" index={issue.row_index_value}" if issue.row_index_value else ""
            print(
                f"{issue.file} row={issue.row_number}{idx} type={issue.issue_type}"
                + (f" preview='{issue.preview}'" if issue.preview else "")
            )

    if args.report_csv:
        report_path = Path(args.report_csv).expanduser().resolve()
        with open(report_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "symbol", "row_number", "row_index_value", "issue_type", "preview"],
            )
            writer.writeheader()
            for issue in all_issues:
                writer.writerow(issue.__dict__)
        print(f"Detailed report saved to: {report_path}")

    if args.fail_on_issues and total_issues > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
