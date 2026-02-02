import streamlit as st
import requests
import json
import time
import csv
import io
from datetime import date
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# RunPod Configuration
RUNPOD_API_ID = "4fbwlg7yhbwu2u"  # TODO: Replace with your new Endpoint ID
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_API_ID}"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://github.com/AI4Finance-Foundation/FinGPT/raw/master/figs/logo.png", width=100)
        st.title("FinGPT Forecaster")
        st.markdown("---")

        # Trained tickers (Dow 30 subset used in this project)
        TRAINED_TICKERS = [
            "AAPL", "AMZN", "CRM", "CSCO", "GOOGL", "IBM",
            "INTC", "META", "MSFT", "NVDA", "TSLA", "TSM"
        ]

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
                st.success(f"Saved {len(st.session_state['prediction_results'])} predictions to {saved_path}")

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
