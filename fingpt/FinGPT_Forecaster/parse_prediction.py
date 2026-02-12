import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionRecord:
    """Strukturerat resultat för en prognosrad i CSV:en."""
    ticker: str
    date: str
    move: str          # t.ex. "UP by 2.5%"
    confidence: str    # t.ex. "75%"
    text: str          # resterande analystext


class PredictionParser:
    """Ansvarar för att parsa textfilen till PredictionRecord-objekt."""
    PRED_REGEX = re.compile(r'^\s*Prediction:(.*)', re.IGNORECASE)
    CONF_REGEX = re.compile(r'^\s*Confidence:(.*)', re.IGNORECASE)
    CONF_LEVEL_REGEX = re.compile(r'^\s*Confidence Level:(.*)', re.IGNORECASE)
    TICKER_REGEX = re.compile(r'Analysis\s+for\s+([A-Z0-9\.-]+)', re.IGNORECASE)
    DATE_REGEX = re.compile(r'Target Week:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})')

    def parse(self, content: str) -> list[PredictionRecord]:
        # Dela upp på block som separeras av "View Raw API Response"
        blocks = content.split("View Raw API Response")
        records: list[PredictionRecord] = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            ticker = self._extract_ticker(lines)
            date = self._extract_date(lines)
            move = self._extract_prediction(lines)
            confidence = self._extract_confidence(lines)

            # Hoppa block som inte har vare sig prediction eller confidence
            if not move and not confidence:
                continue

            text = self._extract_text(lines)
            records.append(PredictionRecord(ticker, date, move, confidence, text))

        return records

    def _clean(self, s: str) -> str:
        # Ta bort enkel markdown (** och `) m.m.
        s = re.sub(r'[*`]', '', s)
        return s.strip()

    def _extract_ticker(self, lines: list[str]) -> str:
        for line in lines:
            m = self.TICKER_REGEX.search(line)
            if m:
                return m.group(1).upper().strip()
        return ""

    def _extract_date(self, lines: list[str]) -> str:
        """Plockar ut Target Week-datumet om det finns."""
        for line in lines:
            m = self.DATE_REGEX.search(line)
            if m:
                return m.group(1)
        return ""

    def _extract_prediction(self, lines: list[str]) -> str:
        for line in lines:
            lower = line.lower()
            # Hoppa "Price Direction Prediction" och "Final Prediction" – ta den första "vanliga"
            if lower.startswith("price direction prediction"):
                continue
            if lower.startswith("final prediction"):
                continue

            m = self.PRED_REGEX.match(line)
            if m:
                return self._clean(m.group(1))
        return ""

    def _extract_confidence(self, lines: list[str]) -> str:
        # Först "Confidence:"
        for line in lines:
            m = self.CONF_REGEX.match(line)
            if m:
                return self._clean(m.group(1))

        # Fallback: "Confidence Level:"
        for line in lines:
            m = self.CONF_LEVEL_REGEX.match(line)
            if m:
                return self._clean(m.group(1))

        return ""

    def _extract_text(self, lines: list[str]) -> str:
        filtered: list[str] = []

        for line in lines:
            # Hoppa rena prediction-/confidence-rader och rubriker
            if self.PRED_REGEX.match(line):
                continue
            if self.CONF_REGEX.match(line) or self.CONF_LEVEL_REGEX.match(line):
                continue
            if line.strip().lower() in ("[prediction]", "**prediction**"):
                continue

            filtered.append(self._clean(line))

        # Lägg allt i en enda kolumn (csv-writer sköter quoting)
        return " ".join(filtered)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} INPUT_FILE [OUTPUT_CSV]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".csv")

    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    content = input_path.read_text(encoding="utf-8")

    parser = PredictionParser()
    records = parser.parse(content)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        # Använd semikolon som avskiljare så att Excel på svensk
        # Windows öppnar filen snyggt med en kolumn per fält.
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["ticker", "date", "move", "confidence", "text"])
        for r in records:
            writer.writerow([r.ticker, r.date, r.move, r.confidence, r.text])

    print(f"Wrote {len(records)} rows to {output_path}")


if __name__ == "__main__":
    main()