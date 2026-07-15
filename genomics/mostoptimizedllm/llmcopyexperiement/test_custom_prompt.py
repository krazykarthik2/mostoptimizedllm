import os
import gc
import torch
import torch.nn as nn
from model import Gemma3EMLKANMLP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def generate_response(model, tokenizer, prompt, device, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Greedy decoding for maximum reasoning stability
            pad_token_id=tokenizer.eos_token_id
        )
    # Extract only the generated response text
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_evaluation(model_id, weights_path, device="cuda"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    custom_prompt = (
        "Hello! I am John and I have 5 apples. If I give 2 apples to Mary and buy 3 "
        "more apples from the store, how many apples do I have now? Explain your reasoning step-by-step."
    )
    print(f"\nCustom Evaluation Prompt:\n'{custom_prompt}'")
    
    # 1. Evaluate Fitted EML-KAN Model on GPU
    print("\n--- EVALUATING FITTED EML-KAN MODEL (GPU) ---")
    try:
        model_kan = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.bfloat16
        ).to(device)
        for i in range(model_kan.config.num_hidden_layers):
            model_kan.model.layers[i].mlp = Gemma3EMLKANMLP(model_kan.config).to(torch.bfloat16).to(device)
            
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model_kan.load_state_dict(state_dict)
        model_kan.eval()
        
        response_kan = generate_response(model_kan, tokenizer, custom_prompt, device)
        print(f"Fitted EML-KAN Response:\n{response_kan}")
    except Exception as e:
        print(f"Failed to run fitted EML-KAN model on GPU: {e}")
        response_kan = None
        
    # 2. Evaluate Compressed EML-KAN Model on CPU
    print("\n--- EVALUATING COMPRESSED EML-KAN MODEL (CPU) ---")
    try:
        # Transfer model to CPU and cast to float32
        model_cpu = model_kan.cpu().float()
        
        # Apply 50% magnitude pruning
        with torch.no_grad():
            for name, param in model_cpu.named_parameters():
                if "linear.weight" in name or "weight_eml" in name:
                    threshold = torch.quantile(torch.abs(param), 0.5)
                    mask = torch.abs(param) >= threshold
                    param.mul_(mask.float())
                    
        # Apply 8-bit dynamic quantization
        print("Quantizing CPU model...")
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        response_quant = generate_response(quantized_model, tokenizer, custom_prompt, "cpu")
        print(f"Quantized EML-KAN Response:\n{response_quant}")
    except Exception as e:
        print(f"Failed to run quantized EML-KAN model on CPU: {e}")
        response_quant = None
        
    # Comparative summary report
    print("\n" + "="*80)
    print("                      CUSTOM PROMPT EVALUATION REPORT")
    print("="*80)
    print("Prompt:")
    print(f"  '{custom_prompt}'")
    print("-"*80)
    print("Fitted EML-KAN (GPU) Response:")
    print(f"  {response_kan if response_kan else 'N/A'}")
    print("-"*80)
    print("Quantized EML-KAN (CPU) Response:")
    print(f"  {response_quant if response_quant else 'N/A'}")
    print("="*80)

if __name__ == "__main__":
    weights = "gemma3_eml_kan/model_state_tuned.pt"
    run_evaluation("google/gemma-3-1b-it", weights)
