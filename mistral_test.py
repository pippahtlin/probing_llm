import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.cuda.empty_cache()
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

print("loaded")