from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import re
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils import *

# LoRA / QLoRA path
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,   
)

# Replace with your own api_key and project name
os.environ['WANDB_API_KEY'] = ''    # TODO: Replace with your environment variable
os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'


class GenerationEvalCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, base_model, tokenizer, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.ignore_until_epoch = ignore_until_epoch
        self.is_qwen = "qwen" in base_model.lower()
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return
            
        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = self.tokenizer
            generated_texts, reference_texts = [], []

            for feature in tqdm(self.eval_dataset):
                prompt = feature['prompt']
                gt = feature['answer']
                inputs = tokenizer(
                    prompt, return_tensors='pt',
                    padding=False, max_length=args.max_length
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                
                res = model.generate(
                    **inputs, 
                    use_cache=True
                )
                # Decode with special tokens kept to parse ChatML when needed
                full_output = tokenizer.decode(res[0], skip_special_tokens=False)
                if self.is_qwen:
                    if "<|im_start|>assistant" in full_output:
                        answer = full_output.split("<|im_start|>assistant")[-1]
                        answer = answer.replace("<|im_end|>", "").strip()
                    else:
                        answer = full_output.strip()
                else:
                    output = tokenizer.decode(res[0], skip_special_tokens=True)
                    answer = re.sub(r'.*\\[/INST\\]\\s*', '', output, flags=re.DOTALL)

                generated_texts.append(answer)
                reference_texts.append(gt)

                # print("GENERATED: ", answer)
                # print("REFERENCE: ", gt)

            metrics = calc_metrics(reference_texts, generated_texts)
            
            # Ensure wandb is initialized
            if wandb.run is None:
                wandb.init()
                
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()            


def main(args):
        
    model_name = parse_model_name(args.base_model, args.from_remote)
    
    # load model using factory from utils.py (QLoRA-friendly)
    model = load_model(
        model_name, 
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True
    )

    if args.local_rank == 0:
        print(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # load data
    if os.path.exists(args.dataset):
        dataset_fname = args.dataset
    else:
        dataset_fname = "./data/" + args.dataset

    # Direct loading for debugging/simplicity
    if not args.from_remote and os.path.exists(dataset_fname):
        print(f"Loading dataset directly from disk: {dataset_fname}")
        from datasets import load_from_disk
        raw_dataset = load_from_disk(dataset_fname)
        
        # Ensure it is a DatasetDict
        if isinstance(raw_dataset, datasets.Dataset):
            # If it's a single dataset, assume it's train and split it
             raw_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
             
        dataset_train = raw_dataset['train'].shuffle(seed=42)
        dataset_test = raw_dataset['test']
        print("Direct load successful.")
    else:
        dataset_list = load_dataset(dataset_fname, args.from_remote)
        dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
        dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    
    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    
    print(f"Loaded dataset sizes - Train: {len(original_dataset['train'])}, Test: {len(original_dataset['test'])}")
    if len(original_dataset['train']) > 0:
        print("First training example keys:", original_dataset['train'][0].keys())
        # print("First training example:", original_dataset['train'][0]) # Uncomment if needed

    num_eval_samples = min(50, len(original_dataset['test']))
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(num_eval_samples))
    
    dataset = original_dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}', # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        load_best_model_at_end=True,            # Load the best model when finished
        metric_for_best_model="eval_loss",      # Monitor Validation Loss
        greater_is_better=False,                # Lower loss is better
        save_strategy=args.evaluation_strategy, # Ensure save strategy matches eval strategy
        deepspeed=args.ds_config if not args.load_in_4bit else None, # Disable DeepSpeed if 4-bit quantization is used
        eval_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # use_cache is safely set via model.config for ChatGLM/Qwen/LLaMA
    model.config.use_cache = False
    
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    
    # Train
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], 
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True,
            return_tensors="pt"
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                base_model=args.base_model,
                tokenizer=tokenizer,
                ignore_until_epoch=round(0.3 * args.num_epochs)
            )
        ]
    )
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'qwen2.5-32b'])
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.1, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--load_in_4bit", action='store_true', help="Load model in 4-bit precision")    
    args = parser.parse_args()
    
    wandb.login()
    main(args)

