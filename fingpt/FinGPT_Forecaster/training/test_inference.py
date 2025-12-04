"""
Test inference with the fine-tuned FinGPT model.
"""
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    print("Loading model...")
    
    # Load base model with 8-bit quantization
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
    
    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        'finetuned_models/fingpt-3080-test_202512011350'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'finetuned_models/fingpt-3080-test_202512011350'
    )
    model.eval()
    
    print("Model loaded! Generating response...")
    print()
    
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

    USER_PROMPT = """[Company Introduction]:
Apple Inc is a leading entity in the Technology sector.

From 2025-11-18 to 2025-11-25, AAPL stock price increased from 228.22 to 234.93. News during this period:

[Headline]: Apple reportedly developing smart glasses
[Summary]: Apple is working on augmented reality glasses that could launch in 2026.

[Headline]: Strong iPhone 16 demand in China
[Summary]: iPhone 16 sales exceed expectations in Chinese market during November.

Based on all the information, analyze the positive developments and potential concerns for AAPL. Then make your prediction of the AAPL stock price movement for next week (2025-11-25 to 2025-12-02). Provide a summary analysis to support your prediction.
"""

    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + USER_PROMPT + E_INST
    
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"PAD token id: {tokenizer.pad_token_id}")
    
    # Don't use eos_token_id to prevent early stopping
    with torch.no_grad():
        res = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            # Don't set eos_token_id - let it generate freely
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            min_new_tokens=200  # Force at least 200 tokens
        )
    
    print(f"Output length: {res.shape[1]} tokens")
    print(f"Generated: {res.shape[1] - inputs['input_ids'].shape[1]} new tokens")
    
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
    
    print()
    print("=" * 70)
    print("YOUR FINE-TUNED MODEL RESPONSE:")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    
    # Also show raw output for debugging
    if len(answer) < 50:
        print()
        print("DEBUG - Raw output (last 500 chars):")
        print(output[-500:] if len(output) > 500 else output)

if __name__ == "__main__":
    main()

