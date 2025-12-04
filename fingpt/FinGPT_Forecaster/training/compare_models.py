"""
Compare your fine-tuned model vs the official FinGPT model.
"""
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Prompt format
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

TEST_PROMPT = """[Company Introduction]:
Apple Inc is a leading entity in the Technology sector. Incorporated and publicly traded since 1980-12-12, the company has established its reputation as one of the key players in the market.

From 2025-11-11 to 2025-11-18, AAPL stock price decreased from 225.12 to 228.22. News during this period:

[Headline]: Apple faces antitrust scrutiny in EU
[Summary]: European regulators are investigating Apple's App Store policies and potential anti-competitive practices.

[Headline]: Apple announces record services revenue
[Summary]: Apple's services division reports $25 billion quarterly revenue, beating analyst expectations.

[Headline]: iPhone production issues in China
[Summary]: Supply chain disruptions in China may affect iPhone availability during holiday season.

From 2025-11-18 to 2025-11-25, AAPL stock price increased from 228.22 to 234.93. News during this period:

[Headline]: Apple reportedly developing smart glasses
[Summary]: Apple is working on augmented reality glasses that could launch in 2026, expanding beyond iPhone.

[Headline]: Strong iPhone 16 demand in China
[Summary]: iPhone 16 sales exceed expectations in Chinese market during November despite earlier concerns.

[Headline]: Warren Buffett increases Apple stake
[Summary]: Berkshire Hathaway disclosed additional purchases of Apple shares in Q3 2025.

Based on all the information, analyze the positive developments and potential concerns for AAPL. Then make your prediction of the AAPL stock price movement for next week (2025-11-25 to 2025-12-02). Provide a summary analysis to support your prediction.
"""


def load_model(adapter_path, model_name="Your Fine-tuned Model"):
    """Load a model with the given adapter."""
    print(f"\nLoading {model_name}...")
    
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
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    model.eval()
    
    print(f"{model_name} loaded!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, min_tokens=250, max_tokens=600):
    """Generate a response from the model."""
    full_prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    
    inputs = tokenizer(full_prompt, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():
        res = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
    return answer


def main():
    print("=" * 70)
    print("MODEL COMPARISON: Your Model vs Official FinGPT")
    print("=" * 70)
    
    # Your fine-tuned model
    YOUR_MODEL_PATH = 'finetuned_models/fingpt-3080-test_202512011350'
    
    # Official FinGPT model
    OFFICIAL_MODEL_PATH = 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'
    
    # Load and test your model
    your_model, your_tokenizer = load_model(YOUR_MODEL_PATH, "Your Fine-tuned Model (118 samples)")
    
    print("\nGenerating response from YOUR model...")
    your_response = generate_response(your_model, your_tokenizer, TEST_PROMPT)
    
    # Free memory before loading next model
    del your_model
    torch.cuda.empty_cache()
    
    # Load and test official model
    official_model, official_tokenizer = load_model(OFFICIAL_MODEL_PATH, "Official FinGPT (600+ samples)")
    
    print("\nGenerating response from OFFICIAL FinGPT model...")
    official_response = generate_response(official_model, official_tokenizer, TEST_PROMPT)
    
    # Print comparison
    print("\n")
    print("=" * 70)
    print("YOUR MODEL (trained on 118 samples, 3 stocks)")
    print("=" * 70)
    print(your_response)
    
    print("\n")
    print("=" * 70)
    print("OFFICIAL FINGPT (trained on 600+ samples, 30 stocks)")
    print("=" * 70)
    print(official_response)
    
    print("\n")
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Your model response length: {len(your_response)} chars")
    print(f"Official model response length: {len(official_response)} chars")
    print()
    print("Things to compare:")
    print("1. Does your model follow the correct format?")
    print("2. Are the positive/negative factors relevant to the news?")
    print("3. Is the prediction coherent with the analysis?")
    print("4. Is the language quality similar?")


if __name__ == "__main__":
    main()

