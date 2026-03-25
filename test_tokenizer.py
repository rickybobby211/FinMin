from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
max_len = getattr(tokenizer, "model_max_length", None)
print("tokenizer.model_max_length:", max_len)

# We can't easily load the 32B model on a small instance, but we can load its config
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print("config.max_position_embeddings:", getattr(config, "max_position_embeddings", None))
