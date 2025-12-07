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
API_URL = f"https://api.runpod.ai/v2/{RUNPOD_API_ID}/runsync"

# Try to get API KEY from Streamlit secrets, environment variable, or fallback (for local dev)
try:
    API_KEY = st.secrets["RUNPOD_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local testing if secrets.toml is missing (NOT RECOMMENDED FOR PRODUCTION)
    # You should create a .streamlit/secrets.toml file locally for testing
    API_KEY = "YOUR_API_KEY_HERE" 
    # st.warning("Using hardcoded API Key. Setup secrets for production.")

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
        
        # selected_ticker logic removed as we stick to Dow 30
        
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
        if not ticker:
            st.error("Please enter a ticker symbol.")
            return

        with st.spinner(f"🤖 Analyzing {ticker} market data... (Typical wait: 30-40s)"):
            
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
            
            try:
                start_time = time.time()
                response = requests.post(API_URL, json=payload, headers=headers, timeout=600)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle RunPod specific error fields
                    if "error" in result:
                        st.error(f"❌ API Error: {result['error']}")
                        st.json(result)
                        return

                    # Extract output
                    # RunPod sync endpoint returns structure: {"id": "...", "status": "COMPLETED", "output": {...}}
                    output = result.get("output", result)
                    
                    if not output or (isinstance(output, dict) and "error" in output):
                         err_msg = output.get("error", "Unknown error") if isinstance(output, dict) else "Empty response"
                         st.error(f"❌ Model Error: {err_msg}")
                         return

                    # Success!
                    st.success(f"✅ Analysis completed in {elapsed_time:.1f} seconds")
                    
                    # Display Results
                    st.markdown("---")
                    
                    # Parse the prediction text to make it look nicer
                    raw_text = output.get('prediction', 'No prediction text returned.')
                    
                    # Display Ticker & Date Header
                    st.subheader(f"📊 Analysis for {output.get('ticker', ticker)}")
                    st.caption(f"Target Week: {output.get('date', prediction_date)}")
                    
                    # Display the text in a nice box
                    st.markdown(f"""
                    <div class="prediction-box">
                        {raw_text.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Raw JSON expander for debugging
                    with st.expander("View Raw API Response"):
                        st.json(result)
                        
                else:
                    st.error(f"❌ Server Error ({response.status_code})")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The model is taking too long to respond.")
            except Exception as e:
                st.error(f"❌ Connection Error: {str(e)}")

if __name__ == "__main__":
    main()

