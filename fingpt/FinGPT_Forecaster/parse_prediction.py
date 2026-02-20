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
    # Fångar t.ex. "Prediction: ...", "**Prediction**: ...", "Predicted Movement: ..."
    PRED_REGEX = re.compile(
        r'^\s*(?:\*\*)?(?:prediction|predicted movement|next week prediction|summary prediction|price direction prediction|final prediction)(?:\*\*)?\s*:\s*(.*)$',
        re.IGNORECASE,
    )
    # Fångar t.ex. "Confidence: ...", "**Confidence**: ...", "Confidence Level: ..."
    CONF_REGEX = re.compile(
        r'^\s*(?:\*\*)?(?:confidence|confidence level)(?:\*\*)?\s*:\s*(.*)$',
        re.IGNORECASE,
    )
    CONF_LEVEL_REGEX = CONF_REGEX
    TICKER_REGEX = re.compile(r'Analysis\s+for\s+([A-Z0-9\.-]+)', re.IGNORECASE)
    DATE_REGEX = re.compile(r'Target Week:\s*(\d{4}-\d{2}-\d{2})')
    MOVE_FALLBACK_REGEX = re.compile(
        r'\b(UP|DOWN)\b\s+by\s+(?:approximately\s+)?([+-]?\d+(?:\.\d+)?%)',
        re.IGNORECASE,
    )
    CONF_FALLBACK_REGEX = re.compile(
        r'(\d+(?:\.\d+)?%)\s+confidence\b|\bconfidence\b[^%\n\r]*(\d+(?:\.\d+)?%)',
        re.IGNORECASE,
    )

    def parse(self, content: str) -> list[PredictionRecord]:
        # Dela upp på varje "Analysis for <TICKER>" så vi inte är beroende
        # av att "View Raw API Response" finns mellan alla sektioner.
        blocks = self._split_into_ticker_blocks(content)
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

    def _split_into_ticker_blocks(self, content: str) -> list[str]:
        lines = content.splitlines()
        blocks: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            # Start på ny ticker-sektion
            if self.TICKER_REGEX.search(line):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                if current:
                    current.append(line)

        if current:
            blocks.append(current)

        # Fallback om inga ticker-sektioner hittades
        if not blocks:
            return [content]

        return ["\n".join(block) for block in blocks]

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
            m = self.PRED_REGEX.match(line)
            if m:
                value = self._clean(m.group(1))
                if value:
                    return value

        # Fallback: leta rörelse var som helst i sektionen.
        for line in lines:
            m = self.MOVE_FALLBACK_REGEX.search(line)
            if m:
                direction = m.group(1).upper()
                magnitude = m.group(2)
                return f"{direction} by {magnitude}"

        return ""

    def _first_cleaned_match(self, lines: list[str], patterns: list[re.Pattern[str]]) -> str:
        for line in lines:
            for pattern in patterns:
                m = pattern.match(line)
                if m:
                    value = self._clean(m.group(1))
                    if value:
                        return value
        return ""

    def _extract_confidence_fallback(self, lines: list[str]) -> str:
        for line in lines:
            m = self.CONF_FALLBACK_REGEX.search(line)
            if not m:
                continue

            value = m.group(1) or m.group(2)
            if value:
                return value.strip()

        return ""

    def _extract_confidence(self, lines: list[str]) -> str:
        # Först explicita confidence-rader, sedan textfallback.
        value = self._first_cleaned_match(lines, [self.CONF_REGEX, self.CONF_LEVEL_REGEX])
        if value:
            return value

        return self._extract_confidence_fallback(lines)

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