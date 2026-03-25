from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
cache_dir = "/net/scratch/pippalin2/hf_cache"
save_path = "mistral_two_senators_activations.pt"

device_dtype = torch.float16
num_layers = 32
num_heads = 32
hidden_size = 4096
head_dim = hidden_size // num_heads

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    dtype=device_dtype,
    device_map="auto",
    quantization_config=bnb_config,
)
model.eval()

# --------------------------------------------------
# Example small test set: replace with your real senators/prompts
# --------------------------------------------------
examples = [
    {
        "senator": "Bernie Sanders",
        "prompt": "Generate a statement by Bernie Sanders, a politician in the United States."
    },
    {
        "senator": "Ted Cruz",
        "prompt": "Generate a statement by Bernie Sanders, a politician in the United States."
    },
]

# --------------------------------------------------
# Storage for attention-head activations
# We'll capture the INPUT to each layer's o_proj:
# shape: [batch, seq_len, hidden_size]
# then reshape -> [batch, seq_len, num_heads, head_dim]
# --------------------------------------------------
captured_head_outputs = {}

def make_o_proj_pre_hook(layer_idx):
    def hook(module, inputs):
        # inputs is a tuple; inputs[0] is tensor passed into o_proj
        x = inputs[0].detach().cpu()   # [B, T, H]
        captured_head_outputs[layer_idx] = x
    return hook

hooks = []
for layer_idx in range(num_layers):
    hook = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
        make_o_proj_pre_hook(layer_idx)
    )
    hooks.append(hook)

all_records = []

for ex in examples:
    senator = ex["senator"]
    prompt = ex["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    captured_head_outputs.clear()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # ------------------------------------------
    # 1) Layer-wise hidden states
    # outputs.hidden_states has length 33:
    #   [embedding_output, layer1_output, ..., layer32_output]
    # We usually drop embedding output and keep only the 32 layers.
    # Each tensor shape: [B, T, 4096]
    # ------------------------------------------
    layer_hidden_avg = []
    for layer_idx in range(1, len(outputs.hidden_states)):  # skip embedding output at index 0
        h = outputs.hidden_states[layer_idx].detach().cpu()   # [B, T, H]
        h_avg = h.mean(dim=1).squeeze(0)                      # [H]
        layer_hidden_avg.append(h_avg)

    layer_hidden_avg = torch.stack(layer_hidden_avg, dim=0)   # [32, 4096]

    # ------------------------------------------
    # 2) Per-head activations
    # captured_head_outputs[layer_idx]: [B, T, 4096]
    # reshape -> [B, T, 32, 128]
    # average over tokens -> [32, 128]
    # ------------------------------------------
    head_avg_all_layers = []
    for layer_idx in range(num_layers):
        x = captured_head_outputs[layer_idx]                  # [B, T, 4096]
        B, T, H = x.shape
        x = x.view(B, T, num_heads, head_dim)                 # [B, T, 32, 128]
        x_avg = x.mean(dim=1).squeeze(0)                      # [32, 128]
        head_avg_all_layers.append(x_avg)

    head_avg_all_layers = torch.stack(head_avg_all_layers, dim=0)  # [32, 32, 128]

    # ------------------------------------------
    # Save one record per senator/prompt
    # ------------------------------------------
    record = {
        "senator": senator,
        "prompt": prompt,
        "input_ids": inputs["input_ids"].detach().cpu(),
        "attention_mask": inputs["attention_mask"].detach().cpu(),
        "layer_hidden_avg": layer_hidden_avg,         # [32, 4096]
        "head_avg": head_avg_all_layers,              # [32, 32, 128]
    }
    all_records.append(record)

torch.save(all_records, save_path)

for h in hooks:
    h.remove()

print(f"Saved {len(all_records)} records to {save_path}")
print("Example layer_hidden_avg shape:", all_records[0]["layer_hidden_avg"].shape)
print("Example head_avg shape:", all_records[0]["head_avg"].shape)