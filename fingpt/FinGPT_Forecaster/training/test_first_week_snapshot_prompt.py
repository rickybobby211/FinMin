import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import pandas as pd

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import generate_labels
import prepare_latest_data


def build_ohlcv_frame(start: str, end: str, price_start: float, price_step: float, volume_start: int) -> pd.DataFrame:
    """Create deterministic business-day OHLCV data for prompt-generation tests."""
    dates = pd.date_range(start=start, end=end, freq="B")
    rows = []
    for idx, dt in enumerate(dates):
        close = price_start + (idx * price_step)
        rows.append(
            {
                "Date": dt,
                "Adj Close": close,
                "Close": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Open": close * 0.995,
                "Volume": volume_start + (idx * 1000),
            }
        )
    return pd.DataFrame(rows)


class FirstWeekSnapshotPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        self.market_data_dir = self.data_dir / "_market_data"
        self.market_data_dir.mkdir(parents=True, exist_ok=True)

        self.nvda_daily = build_ohlcv_frame("2022-02-01", "2023-04-30", 100.0, 0.35, 1_000_000)
        self.qqq_daily = build_ohlcv_frame("2022-02-01", "2023-04-30", 250.0, 0.12, 2_000_000)
        self.vix_daily = build_ohlcv_frame("2022-02-01", "2023-04-30", 18.0, 0.01, 0)

        self._write_snapshot("NVDA", self.nvda_daily)
        self._write_snapshot("QQQ", self.qqq_daily)
        self._write_snapshot("^VIX", self.vix_daily)

        generate_labels.market_manager.data.clear()
        generate_labels.market_manager.data_sources.clear()
        generate_labels.market_manager.set_cache_dir(str(self.data_dir))

    def tearDown(self) -> None:
        generate_labels.market_manager.data.clear()
        generate_labels.market_manager.data_sources.clear()
        self.tmpdir.cleanup()

    def _write_snapshot(self, symbol: str, df: pd.DataFrame) -> None:
        safe_symbol = generate_labels.re.sub(r"[^A-Za-z0-9._-]+", "_", symbol)
        df.to_csv(self.market_data_dir / f"{safe_symbol}.csv", index=False)

    def test_first_week_prompt_uses_snapshot_lookback(self) -> None:
        with patch.object(prepare_latest_data, "fetch_market_data", return_value=self.nvda_daily.set_index("Date")):
            weekly_df, _ = prepare_latest_data.get_returns("NVDA", "2023-03-17", "2023-04-30")

        first_row = weekly_df.iloc[0].copy()
        first_row["News"] = "[]"
        first_row["Basics"] = "{}"
        self.assertEqual(str(first_row["Start Date"].date()), "2023-03-19")
        self.assertEqual(str(first_row["End Date"].date()), "2023-03-26")

        market_return = generate_labels.market_manager.get_return("QQQ", first_row["Start Date"], first_row["End Date"])
        head, _, _, _ = generate_labels.get_prompt_by_row("NVDA", first_row, market_return, "Nasdaq-100")

        self.assertNotIn("Market Context (VIX): N/A", head)
        self.assertNotIn("Volume Status: N/A", head)
        self.assertNotIn("Volatility (ATR): N/A", head)
        self.assertNotIn("Daily RSI: N/A", head)
        self.assertNotIn("Long-Term Trend: N/A vs SMA200", head)


if __name__ == "__main__":
    unittest.main()
