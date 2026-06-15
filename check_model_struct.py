import torch
from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.bfloat16)
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
model.model.text_model = get_peft_model(model.model.text_model, lora_config)

print("Type of model.model.text_model:", type(model.model.text_model))
# print("Peft modules:", list(model.model.text_model.named_modules())[:10])

# Try to find layers
def find_layers(m, depth=0):
    if depth > 10: return
    for name, child in m.named_children():
        if "layers" in name:
            print(f"Found layers at {name}: {type(child)}")
            return
        find_layers(child, depth + 1)

find_layers(model.model.text_model)
