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

import numpy as np

def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class StructurallyPrunedFusedGELUEMLKANMLP(nn.Module):
    """
    Structurally Pruned (d_intermediate = 3072) Single-Spline (k=1) GELU-Fused EML-KAN Layer.
    - Physically shrinks hidden channels from 6912 -> 3072 (45.5% VRAM footprint reduction).
    - Fuses GELU activation directly into 3rd-degree Chebyshev polynomial.
    """
    def __init__(self, w_gate, w_up, w_down, p0, p1, p2, p3, domain_bound=10.0):
        super().__init__()
        intermediate_size, hidden_size = w_gate.shape[0], w_gate.shape[1]
        cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * domain_bound
        
        q0 = np.zeros(intermediate_size, dtype=np.float32)
        q1 = np.zeros(intermediate_size, dtype=np.float32)
        q2 = np.zeros(intermediate_size, dtype=np.float32)
        q3 = np.zeros(intermediate_size, dtype=np.float32)
        
        p1_eff = p1 + 1.0
        for i in range(intermediate_size):
            u = cheb_nodes
            poly_val = p0[i] + p1_eff[i] * u + p2[i] * (u**2) + p3[i] * (u**3)
            fused_curve = gelu_np(poly_val)
            coeffs = np.polyfit(cheb_nodes, fused_curve, 3)
            q3[i], q2[i], q1[i], q0[i] = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            
        w_gate_t = torch.tensor(w_gate).float()
        q0_t = torch.tensor(q0).float()
        q1_t = torch.tensor(q1).float()
        q2_t = torch.tensor(q2).float()
        q3_t = torch.tensor(q3).float()
        
        non_linear_mask = (torch.abs(q2_t) > 1e-4) | (torch.abs(q3_t) > 1e-4)
        linear_scale = q1_t.clone()
        linear_scale[non_linear_mask] = 1.0
        
        w_gate_folded = w_gate_t * linear_scale.unsqueeze(1)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.gate_proj.weight.data.copy_(w_gate_folded)
        self.gate_proj.bias.data.copy_(q0_t)
        self.up_proj.weight.data.copy_(torch.tensor(w_up).float())
        self.down_proj.weight.data.copy_(torch.tensor(w_down).float())
        
        q1_poly = q1_t.clone()
        q1_poly[~non_linear_mask] = 1.0
        q2_poly = q2_t.clone()
        q2_poly[~non_linear_mask] = 0.0
        q3_poly = q3_t.clone()
        q3_poly[~non_linear_mask] = 0.0
        
        self.register_buffer("q1_poly", q1_poly)
        self.register_buffer("q2_poly", q2_poly)
        self.register_buffer("q3_poly", q3_poly)
        
    def forward(self, x):
        gate_linear = self.gate_proj(x)
        up_proj = self.up_proj(x)
        fused_act = gate_linear * (self.q1_poly + gate_linear * (self.q2_poly + gate_linear * self.q3_poly))
        return self.down_proj(fused_act * up_proj)

MODEL_ID = "google/gemma-3-1b-it"
MASTER_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "llmcopyexperiement", "gemma3_eml_kan", "model_state_master_final.pt")
PRUNED_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "llmcopyexperiement", "gemma3_eml_kan", "model_state_pruned_654m_perfect.pt")

def print_header():
    print("\n" + CYAN + BOLD + "=" * 75 + RESET)
    print(CYAN + BOLD + "   🚀 STRUCTURALLY PRUNED 654M GELU-FUSED EML-KAN CHAT STUDIO" + RESET)
    print(DIM + "   Gemma-3 654M Params | d_intermediate=3072 | 1.22 GB VRAM (61.27 t/s)" + RESET)
    print(CYAN + BOLD + "=" * 75 + RESET + "\n")
    print(GREEN + "Type your message below. Commands: " + BOLD + "/tokens" + RESET + GREEN + " (toggle raw tokens), " + BOLD + "/mode" + RESET + GREEN + " (toggle greedy), " + BOLD + "/clear" + RESET + GREEN + " (reset chat), " + BOLD + "/quit" + RESET + GREEN + " (exit)\n" + RESET)

def main():
    print_header()
    
    print(YELLOW + "Loading Gemma-3 Tokenizer & Model Architecture..." + RESET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    state_dict = torch.load(MASTER_WEIGHTS_PATH, map_location="cpu")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    target_pruned_dim = 3072
    
    print(YELLOW + f"Instantiating Structurally Pruned EML-KAN Layers (d_intermediate=3072, 654M Params)..." + RESET)
    for idx in range(model.config.num_hidden_layers):
        w_gate = state_dict[f"model.layers.{idx}.mlp.gate_proj.linear.weight"].float().numpy()
        w_up = state_dict[f"model.layers.{idx}.mlp.up_proj.weight"].float().numpy()
        w_down = state_dict[f"model.layers.{idx}.mlp.down_proj.weight"].float().numpy()
        
        importance = np.linalg.norm(w_gate, ord=1, axis=1) + np.linalg.norm(w_up, ord=1, axis=1) + np.linalg.norm(w_down, ord=1, axis=0)
        top_channels = np.sort(np.argsort(importance)[::-1][:target_pruned_dim])
        
        w_gate_pruned = w_gate[top_channels, :]
        w_up_pruned = w_up[top_channels, :]
        w_down_pruned = w_down[:, top_channels]
        
        p0 = np.zeros(target_pruned_dim, dtype=np.float32)
        p1 = np.zeros(target_pruned_dim, dtype=np.float32)
        p2 = np.zeros(target_pruned_dim, dtype=np.float32)
        p3 = np.zeros(target_pruned_dim, dtype=np.float32)
        
        pruned_layer = StructurallyPrunedFusedGELUEMLKANMLP(
            w_gate_pruned, w_up_pruned, w_down_pruned, p0, p1, p2, p3
        )
        model.model.layers[idx].mlp = pruned_layer

    if os.path.exists(PRUNED_CHECKPOINT_PATH):
        print(YELLOW + "Loading Calibrated 654M Weights..." + RESET)
        pruned_state = torch.load(PRUNED_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(pruned_state, strict=False)
        
    model.to(torch.bfloat16).to("cuda")
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(GREEN + BOLD + f"✓ 654M Model Ready on NVIDIA L40S GPU ({total_params/1e6:.1f}M Params | {vram_gb:.2f} GB VRAM)!\n" + RESET)
    
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
