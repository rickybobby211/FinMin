import streamlit as st
import requests
import json
import time
import csv
import io
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import yfinance as yf

# ============================================================================
# CONFIGURATION
# ============================================================================

# RunPod Configuration
RUNPOD_API_ID = "4fbwlg7yhbwu2u"  # TODO: Replace with your new Endpoint ID
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_API_ID}"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Trained tickers (Dow 30 subset used in this project)
TRAINED_TICKERS = [
    "AAPL",
    "AMZN",
    "CRM",
    "CSCO",
    "GOOGL",
    "IBM",
    "INTC",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    "TSM",
]

# Try to get API KEY from Streamlit secrets, environment variable, or fallback
try:
    API_KEY = st.secrets["RUNPOD_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local testing - Remember to remove before push!
    API_KEY = "YOUR_API_KEY_HERE"

# Page Config
st.set_page_config(
    page_title="FinGPT Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# APP LOGIC
# ============================================================================

def run_prediction(payload, headers):
    """Start prediction job and poll for status."""
    
    # 1. Start Job (Async)
    run_url = f"{BASE_URL}/run"
    response = requests.post(run_url, json=payload, headers=headers)
    
    if response.status_code != 200:
        return {"error": f"Failed to start job: {response.status_code}", "details": response.text}
    
    job_data = response.json()
    job_id = job_data.get("id")
    
    if not job_id:
         return {"error": "No Job ID returned", "details": job_data}
         
    # 2. Poll for completion
    status_url = f"{BASE_URL}/status/{job_id}"
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    while True:
        # Check timeout (e.g. 15 mins for cold starts)
        if time.time() - start_time > 900:
            return {"error": "Timeout waiting for prediction (RunPod took > 15m)"}
            
        time.sleep(3) # Wait between polls
        
        status_res = requests.get(status_url, headers=headers)
        if status_res.status_code != 200:
            continue
            
        status_data = status_res.json()
        status = status_data.get("status")
        
        if status == "COMPLETED":
            progress_bar.progress(100)
            status_text.text("Analysis Complete!")
            return status_data
            
        elif status == "FAILED":
            return {"error": "Job Failed on Server", "details": status_data}
            
        elif status == "IN_QUEUE":
            status_text.text("Job in Queue... (Scaling up GPU)")
            progress_bar.progress(10)
            
        elif status == "IN_PROGRESS":
            duration = time.time() - start_time
            status_text.text(f"AI Analyzing Market Data... ({int(duration)}s)")
            # Fake progress for user feedback
            prog = min(90, 10 + int(duration))
            progress_bar.progress(prog)


# Simple Builder + Command pattern to support batch predictions.
class PredictionPayloadBuilder:
    def __init__(self, prediction_date, n_weeks, use_basics):
        self.prediction_date = prediction_date
        self.n_weeks = n_weeks
        self.use_basics = use_basics

    def build(self, ticker):
        return {
            "input": {
                "ticker": ticker,
                "date": self.prediction_date.strftime("%Y-%m-%d"),
                "n_weeks": self.n_weeks,
                "use_basics": self.use_basics,
            }
        }


class PredictionJob:
    def __init__(self, ticker, payload_builder, headers):
        self.ticker = ticker
        self.payload_builder = payload_builder
        self.headers = headers

    def execute(self):
        payload = self.payload_builder.build(self.ticker)
        return run_prediction(payload, self.headers)


class PredictionResultRepository:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def append(self, prediction_date, ticker, prediction_text):
        file_path = self.base_dir / f"predictions_{prediction_date}.csv"
        is_new_file = not file_path.exists()

        try:
            with file_path.open("a", newline="", encoding="utf-8") as file_handle:
                writer = csv.writer(file_handle)
                if is_new_file:
                    writer.writerow(["prediction", "ticker", "date"])
                writer.writerow([prediction_text, ticker, prediction_date])
            return file_path, None
        except OSError as exc:
            return file_path, str(exc)


class RetryPolicy:
    def __init__(self, max_attempts=2, backoff_seconds=4):
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def should_retry(self, error_message, attempt_index):
        if attempt_index >= self.max_attempts - 1:
            return False
        if not error_message:
            return False
        retryable_tokens = (
            "Failed to start job",
            "Timeout waiting for prediction",
            "Network error",
        )
        return any(token in error_message for token in retryable_tokens)

    def wait(self, attempt_index):
        time.sleep(self.backoff_seconds * (attempt_index + 1))


class PredictionJobRunner:
    def __init__(self, retry_policy):
        self.retry_policy = retry_policy

    def run(self, job):
        for attempt_index in range(self.retry_policy.max_attempts):
            try:
                result = job.execute()
            except requests.RequestException as exc:
                result = {"error": f"Network error: {exc}"}
            except Exception as exc:
                result = {"error": f"Unexpected error: {exc}"}

            if "error" not in result:
                return result

            if not self.retry_policy.should_retry(result.get("error"), attempt_index):
                return result

            self.retry_policy.wait(attempt_index)

        return {"error": "Retry policy exhausted without result"}


# ============================================================================
# PRICE DATA (Strategy + Service pattern)
# ============================================================================


class PriceDataProvider:
    """Abstrakt provider för prisdata (Strategy‑mönster)."""

    def get_close_on_or_before(self, ticker: str, target_date: date) -> float | None:
        raise NotImplementedError


class YahooPriceDataProvider(PriceDataProvider):
    """Hämtar prisdata från Yahoo Finance via yfinance."""

    def get_close_on_or_before(self, ticker: str, target_date: date) -> float | None:
        # Hämta ett litet fönster runt dagen för att hantera helger/helgdagar.
        start = target_date - timedelta(days=7)
        end = target_date + timedelta(days=1)

        try:
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
            )
        except Exception:
            return None

        if data.empty or "Close" not in data:
            return None

        # Filtrera på dagar <= target_date och ta senaste stängningspris.
        closes = data["Close"]
        closes = closes[closes.index.date <= target_date]
        if closes.empty:
            return None

        return float(closes.iloc[-1])


@dataclass
class PriceComparisonResult:
    ticker: str
    date_a: date
    date_b: date
    price_a: float | None
    price_b: float | None
    abs_change: float | None
    pct_change: float | None


@dataclass
class UploadedPrediction:
    ticker: str
    date: str
    move: str
    confidence: str
    text: str
    direction: str | None
    expected_pct_change: float | None


class PredictionCsvParser:
    """Parsar predictions-CSV från parse_prediction.py."""

    REQUIRED_COLUMNS = {"ticker", "date", "move"}

    def parse(self, uploaded_file) -> tuple[list[UploadedPrediction], str | None]:
        if uploaded_file is None:
            return [], None

        try:
            raw_text = uploaded_file.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            return [], "Kunde inte läsa filen. Kontrollera att CSV-filen är UTF-8-kodad."

        if not raw_text.strip():
            return [], "CSV-filen är tom."

        try:
            sample = raw_text[:2048]
            delimiter = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        except csv.Error:
            # parse_prediction.py skriver semikolon, men vi tillater komma som fallback.
            delimiter = ";"

        reader = csv.DictReader(io.StringIO(raw_text), delimiter=delimiter)
        if not reader.fieldnames:
            return [], "CSV-filen saknar header."

        normalized_columns = {col.strip().lower() for col in reader.fieldnames}
        if not self.REQUIRED_COLUMNS.issubset(normalized_columns):
            return [], (
                "CSV-filen saknar obligatoriska kolumner. "
                "Förväntat minst: ticker, date, move."
            )

        records: list[UploadedPrediction] = []
        for row in reader:
            ticker = (row.get("ticker") or row.get("Ticker") or "").strip().upper()
            pred_date = (row.get("date") or row.get("Date") or "").strip()
            move = (row.get("move") or row.get("Move") or "").strip()
            confidence = (row.get("confidence") or row.get("Confidence") or "").strip()
            text = (row.get("text") or row.get("Text") or "").strip()

            if not ticker:
                continue

            direction = self._extract_direction(move)
            expected_pct_change = self._extract_expected_pct_change(move, direction)
            records.append(
                UploadedPrediction(
                    ticker=ticker,
                    date=pred_date,
                    move=move,
                    confidence=confidence,
                    text=text,
                    direction=direction,
                    expected_pct_change=expected_pct_change,
                )
            )

        if not records:
            return [], "Inga giltiga predictions hittades i CSV-filen."

        return records, None

    def _extract_direction(self, move_text: str) -> str | None:
        normalized = move_text.lower().strip()
        if not normalized:
            return None

        if any(token in normalized for token in ("up", "bullish", "increase", "higher")):
            return "UP"
        if any(
            token in normalized
            for token in ("down", "bearish", "decrease", "lower", "drop")
        ):
            return "DOWN"
        if "flat" in normalized or "sideways" in normalized:
            return "FLAT"
        return None

    def _extract_expected_pct_change(
        self, move_text: str, direction: str | None
    ) -> float | None:
        # Hanterar både "2%" och intervall som "1-2%".
        normalized = move_text.lower().replace(",", ".")
        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*%", normalized)
        if range_match:
            value = (float(range_match.group(1)) + float(range_match.group(2))) / 2
        else:
            single_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", normalized)
            value = float(single_match.group(1)) if single_match else None

        if value is None:
            return 0.0 if direction == "FLAT" else None

        if direction == "DOWN":
            return -abs(value)
        if direction == "UP":
            return abs(value)
        if direction == "FLAT":
            return 0.0
        return value


class PredictionComparisonService:
    """Kopplar faktiska prisresultat till uppladdade predictions."""

    def build_prediction_index(
        self, uploaded_predictions: list[UploadedPrediction]
    ) -> dict[str, UploadedPrediction]:
        index: dict[str, UploadedPrediction] = {}
        for prediction in uploaded_predictions:
            # Senaste raden per ticker vinner om dubbletter finns.
            index[prediction.ticker] = prediction
        return index

    def resolve_actual_direction(self, pct_change: float | None) -> str | None:
        if pct_change is None:
            return None
        if pct_change > 0:
            return "UP"
        if pct_change < 0:
            return "DOWN"
        return "FLAT"

    def direction_match(
        self, predicted_direction: str | None, actual_direction: str | None
    ) -> str:
        if predicted_direction is None or actual_direction is None:
            return "-"
        return "Ja" if predicted_direction == actual_direction else "Nej"

    def pct_gap(
        self, expected_pct_change: float | None, actual_pct_change: float | None
    ) -> str:
        if expected_pct_change is None or actual_pct_change is None:
            return "-"
        return f"{abs(actual_pct_change - expected_pct_change):.2f}%"


class PriceComparisonService:
    """Applikationstjänst som jämför pris mellan två datum."""

    def __init__(self, provider: PriceDataProvider):
        self.provider = provider

    def compare(
        self, tickers: list[str], date_a: date, date_b: date
    ) -> list[PriceComparisonResult]:
        results: list[PriceComparisonResult] = []

        for ticker in tickers:
            price_a = self.provider.get_close_on_or_before(ticker, date_a)
            price_b = self.provider.get_close_on_or_before(ticker, date_b)

            if price_a is None or price_b is None:
                abs_change = None
                pct_change = None
            else:
                abs_change = price_b - price_a
                pct_change = (abs_change / price_a) * 100 if price_a != 0 else None

            results.append(
                PriceComparisonResult(
                    ticker=ticker,
                    date_a=date_a,
                    date_b=date_b,
                    price_a=price_a,
                    price_b=price_b,
                    abs_change=abs_change,
                    pct_change=pct_change,
                )
            )

        return results


def render_price_comparison_section():
    """Streamlit‑vy för att jämföra kurs mellan två datum."""
    st.markdown("---")
    st.subheader("📊 Jämför kursrörelse mellan två datum")

    selected_tickers = st.multiselect(
        "Tickers att jämföra",
        options=TRAINED_TICKERS,
        default=["AAPL"],
        help="Välj en eller flera tickers du vill jämföra.",
        key="compare_tickers",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        date_a = st.date_input(
            "Datum A",
            value=date.today(),
            help="Startdatum för jämförelsen.",
            key="compare_date_a",
        )
    with col_b:
        date_b = st.date_input(
            "Datum B",
            value=date.today(),
            help="Slutdatum för jämförelsen.",
            key="compare_date_b",
        )

    uploaded_prediction_csv = st.file_uploader(
        "Ladda upp predictions CSV (från parse_prediction.py)",
        type=["csv"],
        help=(
            "CSV med kolumner som ticker, date, move, confidence, text. "
            "Den används för att jämföra prediction mot faktisk kursrörelse."
        ),
        key="prediction_compare_upload",
    )

    compare_btn = st.button(
        "📈 Hämta kursrörelse",
        type="primary",
        use_container_width=True,
        key="compare_button",
    )

    if not compare_btn:
        return

    if not selected_tickers:
        st.warning("Välj minst en ticker för att göra jämförelsen.")
        return

    if date_b < date_a:
        st.warning("Datum B måste vara samma som eller senare än datum A.")
        return

    provider = YahooPriceDataProvider()
    service = PriceComparisonService(provider)

    with st.spinner("Hämtar kursdata från Yahoo Finance..."):
        results = service.compare(selected_tickers, date_a, date_b)

    if not results:
        st.info("Hittade ingen kursdata för angivna inställningar.")
        return

    parser = PredictionCsvParser()
    prediction_rows, parse_error = parser.parse(uploaded_prediction_csv)
    if parse_error:
        st.warning(parse_error)

    prediction_compare_service = PredictionComparisonService()
    prediction_index = prediction_compare_service.build_prediction_index(prediction_rows)

    table_rows = []
    for r in results:
        if r.price_a is None or r.price_b is None:
            status = "Ingen data"
        else:
            status = ""

        prediction_row = prediction_index.get(r.ticker)
        actual_direction = prediction_compare_service.resolve_actual_direction(r.pct_change)
        predicted_direction = prediction_row.direction if prediction_row else None

        table_rows.append(
            {
                "Ticker": r.ticker,
                "Datum A": r.date_a.strftime("%Y-%m-%d"),
                "Pris A": f"{r.price_a:.2f}" if r.price_a is not None else "-",
                "Datum B": r.date_b.strftime("%Y-%m-%d"),
                "Pris B": f"{r.price_b:.2f}" if r.price_b is not None else "-",
                "Förändring": (
                    f"{r.abs_change:.2f}" if r.abs_change is not None else "-"
                ),
                "Förändring %": (
                    f"{r.pct_change:.2f}%" if r.pct_change is not None else "-"
                ),
                "Status": status,
                "Prediction Datum": prediction_row.date if prediction_row else "-",
                "Prediction Move": prediction_row.move if prediction_row else "-",
                "Confidence": prediction_row.confidence if prediction_row else "-",
                "Pred. Riktning": predicted_direction or "-",
                "Faktisk Riktning": actual_direction or "-",
                "Rätt Riktning": prediction_compare_service.direction_match(
                    predicted_direction,
                    actual_direction,
                ),
                "Avvikelse %": prediction_compare_service.pct_gap(
                    prediction_row.expected_pct_change if prediction_row else None,
                    r.pct_change,
                ),
            }
        )

    st.table(table_rows)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://github.com/AI4Finance-Foundation/FinGPT/raw/master/figs/logo.png", width=100)
        st.title("FinGPT Forecaster")
        st.markdown("---")

        tickers = st.multiselect(
            "Select Tickers (Trained Dow 30)",
            options=TRAINED_TICKERS,
            default=["AAPL"],
            help="Select one or more tickers for the chosen prediction date."
        )
        
        prediction_date = st.date_input(
            "Prediction Date", 
            value=date.today(),
            help="The date from which to make the prediction"
        )
        
        n_weeks = st.slider(
            "Context Weeks", 
            min_value=1, 
            max_value=4, 
            value=3,
            help="Number of past weeks of news/price data to analyze"
        )
        
        use_basics = st.checkbox(
            "Use Latest Basic Financials",
            value=True,
            help="Include quarterly financial metrics in the analysis."
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            """
            **FinGPT Forecaster** analyzes market news and price action to predict weekly stock movements.
            
            Powered by:
            - **RunPod Serverless** (GPU)
            - **Qwen-2.5-32B** (Fine-tuned)
            - **Finnhub** (News Data)
            """
        )

    # Main Content
    st.markdown('<h1 class="main-header">📈 Stock Market AI Analyst</h1>', unsafe_allow_html=True)
    st.markdown("Generate professional stock analysis and predictions using FinGPT.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        generate_btn = st.button("🚀 Generate Analysis", type="primary", use_container_width=True)

    if "prediction_results" not in st.session_state:
        st.session_state["prediction_results"] = []
    if "prediction_errors" not in st.session_state:
        st.session_state["prediction_errors"] = []

    if generate_btn:
        if not tickers:
            st.warning("Please select at least one ticker.")
            return

        with st.container():
            payload_builder = PredictionPayloadBuilder(prediction_date, n_weeks, use_basics)
            result_repository = PredictionResultRepository(RESULTS_DIR)
            job_runner = PredictionJobRunner(RetryPolicy())
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            jobs = [PredictionJob(ticker, payload_builder, headers) for ticker in tickers]
            st.session_state["prediction_results"] = []
            st.session_state["prediction_errors"] = []
            prediction_date_str = prediction_date.strftime("%Y-%m-%d")
            saved_path = None
            save_error = None
            overall_progress = st.progress(0)
            overall_status = st.empty()

            for idx, job in enumerate(jobs):
                if idx > 0:
                    st.markdown("---")

                overall_status.text(f"Running {idx + 1}/{len(jobs)}: {job.ticker}")
                result = job_runner.run(job)
                overall_progress.progress(int(((idx + 1) / len(jobs)) * 100))

                if "error" in result:
                    error_message = result["error"]
                    st.session_state["prediction_errors"].append({
                        "ticker": job.ticker,
                        "error": error_message,
                    })
                    st.error(f"❌ {job.ticker}: {error_message}")
                    if "details" in result:
                        st.json(result["details"])
                    continue

                output = result.get("output", result)

                # Check for internal error in output
                if isinstance(output, dict) and "error" in output:
                    st.error(f"❌ Model Error: {output['error']}")
                    continue

                raw_text = output.get("prediction", "No prediction text returned.")
                resolved_ticker = output.get("ticker", job.ticker)
                resolved_date = output.get("date", prediction_date_str)

                st.session_state["prediction_results"].append({
                    "ticker": resolved_ticker,
                    "date": resolved_date,
                    "prediction": raw_text,
                    "raw_result": result
                })

                saved_path, save_error = result_repository.append(
                    prediction_date=resolved_date,
                    ticker=resolved_ticker,
                    prediction_text=raw_text
                )

            if save_error:
                st.warning(f"Could not save results to disk: {save_error}")
            elif saved_path and st.session_state["prediction_results"]:
                st.success(
                    f"Saved {len(st.session_state['prediction_results'])} predictions to {saved_path}"
                )

    # Always allow on‑demand price comparison between two arbitrary dates.
    render_price_comparison_section()

    if st.session_state["prediction_errors"]:
        st.markdown("---")
        st.subheader("⚠️ Failed Tickers")
        for error_item in st.session_state["prediction_errors"]:
            st.write(f"{error_item['ticker']}: {error_item['error']}")

    if st.session_state["prediction_results"]:
        st.markdown("---")
        st.subheader("📌 Latest Predictions")

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["prediction", "ticker", "date"])

        for item in st.session_state["prediction_results"]:
            csv_writer.writerow([item["prediction"], item["ticker"], item["date"]])
            st.subheader(f"📊 Analysis for {item['ticker']}")
            st.caption(f"Target Week: {item['date']}")

            st.markdown(f"""
            <div class="prediction-box">
                {item['prediction'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Raw API Response"):
                st.json(item["raw_result"])

        st.download_button(
            label="Download predictions CSV",
            data=csv_buffer.getvalue(),
            file_name=f"predictions_{st.session_state['prediction_results'][0]['date']}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
