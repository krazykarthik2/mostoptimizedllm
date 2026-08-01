import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from full_model_hybrid_polynomial_benchmark import QuantizableHybridPolynomialGemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

# ANSI Color Codes
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

MODEL_ID = "google/gemma-3-1b-it"
WEIGHTS_PATH = "genomics/mostoptimizedllm/llmcopyexperiement/gemma3_eml_kan/model_state_high_sparsity.pt"

def print_header():
    print("\n" + CYAN + BOLD + "=" * 75 + RESET)
    print(CYAN + BOLD + "   🚀 EML-KAN DP-COLLAPSED MODEL INTERACTIVE TERMINAL CHAT STUDIO" + RESET)
    print(DIM + "   Gemma-3-1b-it Structural Clone | Single-Spline Fused Edge KAN (60.17 t/s)" + RESET)
    print(CYAN + BOLD + "=" * 75 + RESET + "\n")
    print(GREEN + "Type your message below. Commands: " + BOLD + "/tokens" + RESET + GREEN + " (toggle raw tokens), " + BOLD + "/clear" + RESET + GREEN + " (reset chat), " + BOLD + "/quit" + RESET + GREEN + " (exit)\n" + RESET)

def main():
    print_header()
    
    print(YELLOW + "Loading Gemma-3 Tokenizer & Model Weights..." + RESET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    
    print(YELLOW + "Instantiating Compiled Fused EML-KAN Layers on GPU..." + RESET)
    for idx in range(model.config.num_hidden_layers):
        poly_mlp = QuantizableHybridPolynomialGemma3EMLKANMLP(model.config, idx, state_dict, num_components=1)
        model.model.layers[idx].mlp = poly_mlp
        
    model.to(torch.bfloat16).to("cuda")
    model.eval()
    print(GREEN + BOLD + "✓ Model Ready on NVIDIA L40S GPU!\n" + RESET)
    
    system_prompt = "You are an expert AI Research Assistant specialized in Deep Learning Architecture, Model Compression, and SciML."
    show_raw_tokens = False
    history = []
    
    while True:
        try:
            user_input = input(MAGENTA + BOLD + "User > " + RESET).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + YELLOW + "Exiting Chat Studio. Goodbye!" + RESET)
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "/quit" or user_input.lower() == "exit":
            print(YELLOW + "Exiting Chat Studio. Goodbye!" + RESET)
            break
        elif user_input.lower() == "/clear":
            history = []
            print(GREEN + "✓ Conversation history cleared.\n" + RESET)
            continue
        elif user_input.lower() == "/tokens":
            show_raw_tokens = not show_raw_tokens
            state_str = "ENABLED" if show_raw_tokens else "DISABLED"
            print(CYAN + f"✓ Special token inspection: {BOLD}{state_str}{RESET}\n")
            continue
            
        # Construct full conversation template with special tokens
        prompt_text = f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        for u, m in history:
            prompt_text += f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{m}<end_of_turn>\n"
        prompt_text += f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        if show_raw_tokens:
            print("\n" + RED + DIM + "[SPECIAL TOKENS PIPELINE]:" + RESET)
            print(RED + prompt_text + RESET)
            
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            torch.cuda.synchronize()
        dt = time.time() - t0
        
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        gen_tokens = len(gen_ids)
        tps = gen_tokens / dt if dt > 0 else 0
        
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        history.append((user_input, response_text))
        
        print("\n" + CYAN + BOLD + "EML-KAN Model > " + RESET + response_text)
        print(DIM + GREEN + f"[{gen_tokens} tokens generated in {dt:.2f}s | {tps:.2f} t/s]\n" + RESET)

if __name__ == "__main__":
    main()
