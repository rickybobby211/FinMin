"""
Test the Official FinGPT Forecaster Model
==========================================
Run: python test_official_fingpt.py

You can modify the TEST_PROMPT below with your own news/data.
"""
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Prompt format for Llama-2
B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]:
Prediction: ...
Analysis: ...
"""

# ============================================================
# MODIFY THIS PROMPT WITH YOUR OWN DATA
# ============================================================
TEST_PROMPT = """[Company Introduction]:
Apple Inc is a leading entity in the Technology sector. Incorporated and publicly traded since 1980-12-12, the company has established its reputation as one of the key players in the market.

From 2025-11-18 to 2025-11-25, AAPL stock price increased from 228.22 to 234.93. News during this period:

[Headline]: Apple reportedly developing smart glasses
[Summary]: Apple is working on augmented reality glasses that could launch in 2026, expanding beyond iPhone.

[Headline]: Strong iPhone 16 demand in China
[Summary]: iPhone 16 sales exceed expectations in Chinese market during November despite earlier concerns.

[Headline]: Warren Buffett increases Apple stake
[Summary]: Berkshire Hathaway disclosed additional purchases of Apple shares in Q3 2025.

Based on all the information, analyze the positive developments and potential concerns for AAPL. Then make your prediction of the AAPL stock price movement for next week (2025-11-25 to 2025-12-02). Provide a summary analysis to support your prediction.
"""


def main():
    print("=" * 70)
    print("Testing Official FinGPT Forecaster Model")
    print("=" * 70)
    
    # Load model with 8-bit quantization (fits on RTX 3080)
    print("\nLoading model (this takes ~30 seconds)...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        llm_int8_threshold=6.0
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        quantization_config=quantization_config,
        device_map='auto',
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    
    # Load official FinGPT adapter
    print("Loading FinGPT adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'
    )
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    model.eval()
    
    print("Model loaded!\n")
    
    # Generate response
    print("Generating prediction (this takes ~1-2 minutes)...")
    
    full_prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + TEST_PROMPT + E_INST
    
    inputs = tokenizer(full_prompt, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():
        res = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
    
    print("\n" + "=" * 70)
    print("FINGPT PREDICTION:")
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()

