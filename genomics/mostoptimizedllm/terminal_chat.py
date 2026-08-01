import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(ROOT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "llmcopyexperiement"))

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
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "llmcopyexperiement", "gemma3_eml_kan", "model_state_high_sparsity.pt")

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
    
    system_prompt = (
        "You are a helpful, precise, and thoughtful AI assistant. "
        "When presented with logic puzzles, mathematical problems, or step-by-step tasks, "
        "reason through the problem carefully before providing your final answer. "
        "Keep your tone natural, helpful, and direct, avoiding unnecessary jargon unless explicitly asked."
    )
    show_raw_tokens = False
    use_greedy = True
    temperature = 0.7
    top_p = 0.9
    
    SYSTEM_MESSAGE = {"role": "system", "content": system_prompt}
    chat_history = [SYSTEM_MESSAGE]
    
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
            chat_history = [SYSTEM_MESSAGE]
            print(GREEN + "✓ Conversation history cleared back to SYSTEM_PROMPT.\n" + RESET)
            continue
        elif user_input.lower() == "/tokens":
            show_raw_tokens = not show_raw_tokens
            state_str = "ENABLED" if show_raw_tokens else "DISABLED"
            print(CYAN + f"✓ Special token inspection: {BOLD}{state_str}{RESET}\n")
            continue
        elif user_input.lower() == "/mode":
            use_greedy = not use_greedy
            mode_str = "GREEDY DECODING (do_sample=False)" if use_greedy else f"SAMPLING (do_sample=True, temp={temperature})"
            print(CYAN + f"✓ Decoding mode set to: {BOLD}{mode_str}{RESET}\n")
            continue
        elif user_input.lower().startswith("/temp"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    use_greedy = False
                    print(CYAN + f"✓ Temperature set to {temperature} (Sampling Enabled)\n" + RESET)
                except ValueError:
                    print(RED + "Invalid temperature value.\n" + RESET)
            else:
                print(CYAN + f"Current Temperature: {temperature}\n" + RESET)
            continue
            
        chat_history.append({"role": "user", "content": user_input})
        
        if show_raw_tokens:
            raw_prompt_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            print("\n" + RED + DIM + "[SPECIAL TOKENS PIPELINE]:" + RESET)
            print(RED + raw_prompt_text + RESET)
            
        model_inputs = tokenizer.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        if isinstance(model_inputs, torch.Tensor):
            input_ids = model_inputs
        else:
            input_ids = model_inputs.input_ids
            
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 300,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        if use_greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
            torch.cuda.synchronize()
        dt = time.time() - t0
        
        gen_ids = outputs[0][input_ids.shape[1]:]
        gen_tokens = len(gen_ids)
        tps = gen_tokens / dt if dt > 0 else 0
        
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        chat_history.append({"role": "model", "content": response_text})
        
        print("\n" + CYAN + BOLD + "EML-KAN Model > " + RESET + response_text)
        print(DIM + GREEN + f"[{gen_tokens} tokens generated in {dt:.2f}s | {tps:.2f} t/s]\n" + RESET)

if __name__ == "__main__":
    main()
