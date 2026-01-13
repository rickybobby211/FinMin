import csv
import os
import re
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("fingpt/.env")

SYSTEM_FIX_PROMPT = """You are a rigorous data auditor for financial training data. 
Your task is to rewrite the [Analysis] to match the [ACTUAL OUTCOME], but you MUST respect the [ORIGINAL PROMPT DATA].

RULES:
1. **ALIGNMENT:** The final prediction direction MUST match the [ACTUAL OUTCOME] (Label).
2. **FACTUAL ACCURACY:** You strictly CANNOT change the technical indicators provided in the PROMPT. 
   - If Prompt says "RSI: 70" (Overbought) and Label is "UP", do NOT say "Oversold". 
   - Instead, explain WHY it goes up despite being Overbought (e.g., "Momentum is too strong," "Short squeeze," "Market ignoring technicals").
3. **LOGIC:** Do not just flip words (Overbought -> Oversold) unless the PROMPT data actually supports it. Find a logical narrative that fits the TRUE data to the TRUE outcome.
"""

def get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY in .env or environment")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def detect_mismatch(prediction, answer):
    """Returns True if there is a semantic mismatch between label and text."""
    if not isinstance(answer, str) or not isinstance(prediction, str):
        return False
        
    pred = prediction.lower()
    
    # Extract conclusion from text
    match = re.search(r"Prediction for the upcoming week.*?:(.*?)$", answer, re.IGNORECASE | re.DOTALL)
    text_conclusion = match.group(1).lower() if match else answer[-200:].lower()
    
    # Clean text for regex
    text_clean = text_conclusion.replace(".", " ").replace(",", " ")

    is_label_up = "up" in pred or "increase" in pred
    is_label_down = "down" in pred or "decrease" in pred
    
    # Robust regex matching for whole words
    up_keywords = r'\b(up|increase|gain|bull|bullish|positive|higher)\b'
    down_keywords = r'\b(down|decrease|drop|bear|bearish|negative|lower)\b'

    is_text_up = re.search(up_keywords, text_clean)
    is_text_down = re.search(down_keywords, text_clean)

    if is_label_up and is_text_down and not is_text_up:
        return True
    if is_label_down and is_text_up and not is_text_down:
        return True
        
    return False

def fix_row(client, row, model="deepseek-chat"):
    """Calls LLM to fix the specific row."""
    prompt_content = row['prompt']
    wrong_answer = row['answer']
    correct_label = row['prediction']
    
    user_msg = f"""
[ORIGINAL PROMPT DATA]:
{prompt_content}

[OLD (WRONG) ANALYSIS]:
{wrong_answer}

[ACTUAL OUTCOME (LABEL)]:
{correct_label}

Please rewrite the analysis (Positive Developments, Potential Concerns, Prediction & Analysis) to support the ACTUAL OUTCOME.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_FIX_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3
        )
        fixed_answer = response.choices[0].message.content
        return fixed_answer, wrong_answer
    except Exception as e:
        print(f"Error fixing row: {e}")
        return None, None

def process_file(file_path, output_path, log_path, parallel=5):
    client = get_client()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        
    print(f"Loaded {len(reader)} rows from {file_path}")
    
    # Filter valid rows (remove empty/failed generations)
    valid_rows = [r for r in reader if r.get('answer') and r.get('prediction')]
    print(f"Found {len(valid_rows)} valid rows (removed empty/failed ones).")
    
    # Identify rows to fix
    indices_to_fix = []
    for i, row in enumerate(valid_rows):
        if detect_mismatch(row['prediction'], row['answer']):
            indices_to_fix.append(i)
            
    print(f"Found {len(indices_to_fix)} rows with logical inconsistencies to fix.")
    print(f"Processing fixes with {parallel} threads...")
    
    fixed_details = []
    
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_idx = {
            executor.submit(fix_row, client, valid_rows[idx]): idx 
            for idx in indices_to_fix
        }
        
        count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            original_label = valid_rows[idx]['prediction']
            start_date = valid_rows[idx].get('start_date', 'Unknown Date')
            
            new_answer, old_answer = future.result()
            count += 1
            if new_answer:
                valid_rows[idx]['answer'] = new_answer
                print(f"[{count}/{len(indices_to_fix)}] ✓ Fixed row (Date: {start_date}, Target: {original_label})")
                
                # Extract conclusion for logging comparison
                old_concl = old_answer[-150:].replace("\n", " ") if old_answer else "N/A"
                new_concl = new_answer[-150:].replace("\n", " ")
                
                fixed_details.append({
                    "index": idx,
                    "date": start_date,
                    "target_label": original_label,
                    "old_conclusion": old_concl,
                    "new_conclusion": new_concl,
                    "full_old": old_answer,
                    "full_new": new_answer
                })
            else:
                print(f"[{count}/{len(indices_to_fix)}] ✗ Failed to fix row")

    # Save corrected CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if valid_rows:
            writer = csv.DictWriter(f, fieldnames=valid_rows[0].keys(), quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(valid_rows)
            
    # Save Log File
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Fix Log for {file_path}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for item in sorted(fixed_details, key=lambda x: x['date']):
            f.write(f"Row Index: {item['index']} | Date: {item['date']} | Target: {item['target_label']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"OLD ENDING: ...{item['old_conclusion']}\n")
            f.write(f"NEW ENDING: ...{item['new_conclusion']}\n")
            f.write("\n")
            
    print(f"Saved {len(valid_rows)} cleaned rows to {output_path}")
    print(f"Detailed change log saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to CSV file")
    args = parser.parse_args()
    
    output = args.file.replace(".csv", "_corrected.csv")
    log_file = args.file.replace(".csv", "_fix_log.txt")
    process_file(args.file, output, log_file)
