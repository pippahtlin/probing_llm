import torch
from dataclasses import dataclass

@dataclass
class Record:
    senator: str
    prompt: str
    dw1: float
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    layer_hidden_avg: torch.Tensor   # [32, 4096]
    head_avg: torch.Tensor           # [32, 32, 128]

def inspect_pt(file_path, num_items=5):
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    print(f"\n===== Inspecting: {file_path} =====")

    if isinstance(data, torch.Tensor):
        print("Type: Tensor")
        print("Shape:", data.shape)
        print("First 5 elements:")
        print(data.view(-1)[:num_items])

    elif isinstance(data, dict):
        print("Type: dict")
        keys = list(data.keys())
        print(f"Total keys: {len(keys)}")
        print("First 5 keys:")
        for k in keys[:num_items]:
            print(f"  - {k}")
            v = data[k]
            if isinstance(v, torch.Tensor):
                print(f"    shape: {v.shape}")
            else:
                print(f"    type: {type(v)}")
                # 如果想顺便看看内容，也可以加一句：
                # print(f"    value: {v}")

    elif isinstance(data, (list, tuple)):
        print(f"Type: {type(data)}")
        print("First 5 items:")
        for i, item in enumerate(data[:num_items]):
            print(f"[{i}] type: {type(item)}")
            print(item)

    else:
        print("Other type:", type(data))
        print(data)

# Example usage
file_path = "/home/pippalin2/probing_llm/mistral_dw_outputs/all_records.pt"
inspect_pt(file_path)