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

# Import utils from local file if present, otherwise define minimal utils
try:
    from utils import *
except ImportError:
    # Minimal utils needed if utils.py is missing
    def parse_model_name(name, from_remote=False):
        if name == 'chatglm2':
            return 'THUDM/chatglm2-6b'
        elif name == 'llama2':
            return 'meta-llama/Llama-2-7b-chat-hf'
        return name

    def load_dataset(path, from_remote=False):
        return datasets.load_from_disk(path)

    def tokenize(args, tokenizer, feature):
        prompt_ids = tokenizer.encode(
            feature['prompt'].strip(), padding=False,
            max_length=args.max_length, truncation=True
        )
        target_ids = tokenizer.encode(
            feature['answer'].strip(), padding=False,
            max_length=args.max_length, truncation=True,
            add_special_tokens=False
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

# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training, 
)

# Config
lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
}

# Pre-trained FinGPT adapter
FINGPT_ADAPTER = 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'


class GenerationEvalCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, ignore_until_epoch=0, use_wandb=False):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch
        self.use_wandb = use_wandb
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return
            
        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['tokenizer']
            generated_texts, reference_texts = [], []

            for feature in tqdm(self.eval_dataset, desc="Evaluating"):
                prompt = feature['prompt']
                gt = feature['answer']
                inputs = tokenizer(
                    prompt, return_tensors='pt',
                    padding=False, max_length=4096, truncation=True
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                
                with torch.no_grad():
                    res = model.generate(
                        **inputs, 
                        use_cache=True,
                        max_new_tokens=128
                    )
                output = tokenizer.decode(res[0], skip_special_tokens=True)
                # Try to extract just the answer part
                if '[/INST]' in output:
                    answer = output.split('[/INST]')[1].strip()
                else:
                    answer = output

                generated_texts.append(answer)
                reference_texts.append(gt)

            # Calculate metrics if available (optional)
            try:
                from utils import calc_metrics
                metrics = calc_metrics(reference_texts, generated_texts)
                print(f"\nEvaluation Metrics: {metrics}")
                
                # Log to wandb if enabled
                if self.use_wandb and wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except ImportError:
                print("\n⚠️  Metrics calculation skipped (rouge_score not available)")
                print("   Install with: pip install rouge-score")
            except Exception as e:
                print(f"\n⚠️  Metrics calculation failed: {e}")

            torch.cuda.empty_cache()            


def main(args):
    print(f"\nTraining with {args.base_model} (Float16)")
    
    model_name = parse_model_name(args.base_model, args.from_remote)
    
    # load model
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Load in Float16
        trust_remote_code=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # load data
    print(f"Loading dataset: {args.dataset}")
    if os.path.exists(args.dataset):
        dataset_list = [datasets.load_from_disk(args.dataset)]
    else:
        dataset_fname = "./datasets/" + args.dataset
        if not os.path.exists(dataset_fname):
             dataset_fname = "./data/" + args.dataset
        dataset_list = [datasets.load_from_disk(dataset_fname)]
    
    # Handle train/test split if not present in dataset
    if 'train' in dataset_list[0]:
        dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
        dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    else:
        # If just a raw dataset, split it
        full_ds = datasets.concatenate_datasets(dataset_list).shuffle(seed=42)
        split = full_ds.train_test_split(test_size=0.1)
        dataset_train = split['train']
        dataset_test = split['test']
    
    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    
    # Small subset for eval callback
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(min(10, len(original_dataset['test']))))
    
    print("Tokenizing...")
    dataset = original_dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    output_dir = f'finetuned_models/{args.run_name}_{formatted_time}'

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_steps=args.eval_steps if isinstance(args.eval_steps, int) else int(args.eval_steps * len(dataset['train'])),
        eval_steps=args.eval_steps if isinstance(args.eval_steps, int) else int(args.eval_steps * len(dataset['train'])),
        fp16=True,
        eval_strategy=args.evaluation_strategy, # Updated for transformers >= 4.41
        save_strategy=args.evaluation_strategy, # Ensure save strategy matches eval strategy
        load_best_model_at_end=True,            # Load the best model when finished
        metric_for_best_model="eval_loss",      # Monitor Validation Loss
        greater_is_better=False,                # Lower loss is better
        remove_unused_columns=False,
        report_to='wandb' if args.use_wandb else 'none',
        run_name=args.run_name,
        gradient_checkpointing=True 
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    
    # setup peft
    if args.from_pretrained_adapter and args.base_model == 'llama2':
        print(f"\nLoading pre-trained adapter: {FINGPT_ADAPTER}")
        model = PeftModel.from_pretrained(
            model, 
            FINGPT_ADAPTER,
            is_trainable=True
        )
    else:
        print("\nInitializing new LoRA adapter...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=lora_module_dict[args.base_model],
            bias='none',
        )
        model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    # Train
    # Optional: Add generation evaluation callback (requires rouge-score)
    callbacks = []
    try:
        import rouge_score  # Check if available
        callbacks.append(
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                ignore_until_epoch=round(0.3 * args.num_epochs),
                use_wandb=args.use_wandb
            )
        )
        print("✅ Generation evaluation callback enabled (will calculate ROUGE metrics)")
    except ImportError:
        print("ℹ️  Generation evaluation callback disabled (install rouge-score to enable)")
    except Exception as e:
        print(f"⚠️  Could not add evaluation callback: {e}")
    
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
        callbacks=callbacks if callbacks else None,
    )
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except:
            pass
    
    torch.cuda.empty_cache()
    print("\nStarting training...")
    trainer.train()

    # save model
    print(f"\nSaving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='fingpt-forecaster', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2'])
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=3, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=0, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--scheduler", default='constant', type=str)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.2, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--from_pretrained_adapter", action='store_true', help="Start from FinGPT pre-trained adapter")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases")
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.login()
        
    main(args)
