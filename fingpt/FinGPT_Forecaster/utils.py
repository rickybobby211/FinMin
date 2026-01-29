import re
import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer


lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        # 'embed_tokens', 'lm_head',
    ],
    'qwen2.5-32b': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 
        'gate_proj', 'up_proj', 'down_proj'
    ]
}


def load_model(
    model_name,
    load_in_4bit=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
):
    """
    Factory function to load models with optional quantization.
    """
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        device_map="auto" if load_in_4bit else None
    )
    return model


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
    # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-chat-hf' # if from_remote else 'base_models/Llama-2-7b-chat-hf'
    elif name == 'qwen2.5-32b':
        return 'Qwen/Qwen2.5-32B-Instruct'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    
    dataset_names = [d for d in names.split(',')]
    dataset_list = []
    
    for name in dataset_names:
        rep = 1
        if not os.path.exists(name) and '/' not in name:
            rep = int(name.split('*')[1]) if '*' in name else 1
            name = ('FinGPT/fingpt-forecaster-' if from_remote else 'data/fingpt-forecaster-') + name.split('*')[0]
        tmp_dataset = datasets.load_dataset(name) if from_remote else datasets.load_from_disk(name)
    
        if 'test' not in tmp_dataset:
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)   
        dataset_list.extend([tmp_dataset] * rep)
    
    return dataset_list


def parse_answer(answer):
    
    # Attempt to match the original format
    match_res = re.match(r"^\s*\[Positive Developments\]:\s*(.*)\s*\[Potential Concerns\]:\s*(.*)\s*\[Prediction (&|and) Analysis\]:\s*(.*)\s*$", answer, flags=re.DOTALL)
    
    # If failed, attempt to match the markdown format (with bold headers)
    if not match_res:
        match_res = re.match(r"^\s*\*\*Positive Developments:?\*\*\s*(.*)\s*\*\*Potential Concerns:?\*\*\s*(.*)\s*\*\*Prediction (&|and) Analysis:?\*\*\s*(.*)\s*$", answer, flags=re.DOTALL)

    if not match_res:
        return None
    
    pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(4)
        
    # Attempt to separate Prediction and Analysis
    # 1. Original format: Prediction then Analysis
    match_res_pna = re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$', pna, flags=re.DOTALL)
    
    if match_res_pna:
        pred, anal = match_res_pna.group(1), match_res_pna.group(2)
    else:
        # 2. Markdown format: Analysis then Prediction (common in recent outputs)
        match_res_pna = re.match(r'^\s*\*\*Analysis:?\*\*\s*(.*)\s*\*\*Prediction:?\*\*\s*(.*)$', pna, flags=re.DOTALL)
        if match_res_pna:
            anal, pred = match_res_pna.group(1), match_res_pna.group(2)
        else:
            # 3. Markdown format: Prediction then Analysis
            match_res_pna = re.match(r'^\s*\*\*Prediction:?\*\*\s*(.*)\s*\*\*Analysis:?\*\*\s*(.*)$', pna, flags=re.DOTALL)
            if match_res_pna:
                pred, anal = match_res_pna.group(1), match_res_pna.group(2)
            else:
                return None
        
    # Parse Binary Prediction (Direction)
    pred_lower = pred.lower()
    if re.search(r'direction\**:\s*\**down', pred_lower) or re.search(r'down|decrease|decline', pred_lower):
        pred_bin = -1
    elif re.search(r'direction\**:\s*\**up', pred_lower) or re.search(r'up|increase', pred_lower):
        pred_bin = 1
    else:
        pred_bin = 0
            
    # Parse Prediction Margin (Percentage)
    # 1. Structured output: "Estimated Percentage Change: -3%"
    match_res = re.search(r'estimated percentage change\**:\s*\**([+-]?\d+(?:\.\d+)?)%', pred_lower)
    if match_res:
        pred_margin = float(match_res.group(1))
    else:
        # 2. Range: "1-2%"
        match_res = re.search(r'(\d+)-(\d+)%', pred)
        if match_res:
            match_res_1 = float(match_res.group(1))
            match_res_2 = float(match_res.group(2))
            pred_margin = pred_bin * ((match_res_1 + match_res_2) / 2)
        else:
            # 3. Single value: "more than 3%"
            match_res = re.search(r'(?:more than )?(\d+(?:\.\d+)?)%', pred)    
            if match_res:
                pred_margin = pred_bin * (float(match_res.group(1)) + 0.5)
            else:
                pred_margin = 0.
        
    return {
        "positive developments": pros.strip(),
        "potential concerns": cons.strip(),
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": anal.strip()
    }
    

def calc_rouge_score(references, answers):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
