import streamlit as st
import requests
import json
import time
from datetime import date

# ============================================================================
# CONFIGURATION
# ============================================================================

# RunPod Configuration
RUNPOD_API_ID = "lfahu5e6q9pmfq"  # Your Endpoint ID
BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_API_ID}"

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


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://github.com/AI4Finance-Foundation/FinGPT/raw/master/figs/logo.png", width=100)
        st.title("FinGPT Forecaster")
        st.markdown("---")
        
        # Dow 30 Tickers
        DOW_30 = [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ]
        
        ticker = st.selectbox(
            "Select Ticker (Dow 30)",
            options=DOW_30,
            index=DOW_30.index("AAPL"), # Default to AAPL
            help="Select a company from the Dow Jones Industrial Average."
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
            - **Llama-2-7b** (Fine-tuned)
            - **Finnhub** (News Data)
            """
        )

    # Main Content
    st.markdown('<h1 class="main-header">📈 Stock Market AI Analyst</h1>', unsafe_allow_html=True)
    st.markdown("Generate professional stock analysis and predictions using FinGPT.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        generate_btn = st.button("🚀 Generate Analysis", type="primary", use_container_width=True)

    if generate_btn:
        with st.container():
            # Prepare payload
            payload = {
                "input": {
                    "ticker": ticker,
                    "date": prediction_date.strftime("%Y-%m-%d"),
                    "n_weeks": n_weeks,
                    "use_basics": use_basics
                }
            }
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            result = run_prediction(payload, headers)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                if "details" in result:
                    st.json(result["details"])
            else:
                output = result.get("output", result)
                
                # Check for internal error in output
                if isinstance(output, dict) and "error" in output:
                     st.error(f"❌ Model Error: {output['error']}")
                     return
                
                # Display Results
                st.markdown("---")
                
                raw_text = output.get('prediction', 'No prediction text returned.')
                
                st.subheader(f"📊 Analysis for {output.get('ticker', ticker)}")
                st.caption(f"Target Week: {output.get('date', prediction_date)}")
                
                st.markdown(f"""
                <div class="prediction-box">
                    {raw_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Raw API Response"):
                    st.json(result)

if __name__ == "__main__":
    main()
