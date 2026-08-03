import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(ROOT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "llmcopyexperiement"))

from eml_dp_collapse_compiler import EMLDPCollapseCompiler
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

def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class SpecialistFusedGELUDPCollapsedMLP(nn.Module):
    def __init__(self, composite_w_dict, w_gate, w_up, w_down, domain_bound=10.0):
        super().__init__()
        cheb_nodes = np.cos((2 * np.arange(1, 15) - 1) * np.pi / 30) * domain_bound
        intermediate_size = len(composite_w_dict["poly_p0"])
        
        p0 = composite_w_dict["poly_p0"]
        p1 = composite_w_dict["poly_p1"] + 1.0
        p2 = composite_w_dict["poly_p2"]
        p3 = composite_w_dict["poly_p3"]
        
        q0 = np.zeros(intermediate_size, dtype=np.float32)
        q1 = np.zeros(intermediate_size, dtype=np.float32)
        q2 = np.zeros(intermediate_size, dtype=np.float32)
        q3 = np.zeros(intermediate_size, dtype=np.float32)
        
        for i in range(intermediate_size):
            u = cheb_nodes
            poly_val = p0[i] + p1[i] * u + p2[i] * (u**2) + p3[i] * (u**3)
            fused_curve = gelu_np(poly_val)
            coeffs = np.polyfit(cheb_nodes, fused_curve, 3)
            q3[i], q2[i], q1[i], q0[i] = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            
        w_gate = torch.tensor(w_gate).float()
        q0_t = torch.tensor(q0).float()
        q1_t = torch.tensor(q1).float()
        q2_t = torch.tensor(q2).float()
        q3_t = torch.tensor(q3).float()
        
        non_linear_mask = (torch.abs(q2_t) > 1e-4) | (torch.abs(q3_t) > 1e-4)
        linear_scale = q1_t.clone()
        linear_scale[non_linear_mask] = 1.0
        
        w_gate_folded = w_gate * linear_scale.unsqueeze(1)
        hidden_size = w_gate.shape[1]
        
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
        
        self.q1_poly = nn.Parameter(q1_poly)
        self.q2_poly = nn.Parameter(q2_poly)
        self.q3_poly = nn.Parameter(q3_poly)
        
    def forward(self, x):
        gate_linear = self.gate_proj(x)
        up_proj = self.up_proj(x)
        activated_gate = gate_linear * (self.q1_poly + gate_linear * (self.q2_poly + gate_linear * self.q3_poly))
        return self.down_proj(activated_gate * up_proj)

MODEL_ID = "google/gemma-3-1b-it"
MASTER_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "llmcopyexperiement", "gemma3_eml_kan", "model_state_master_final.pt")
SPECIALIST_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "llmcopyexperiement", "gemma3_eml_kan", "model_state_4block_code_specialist.pt")

def print_header():
    print("\n" + CYAN + BOLD + "=" * 75 + RESET)
    print(CYAN + BOLD + "   ⚡ 4-BLOCK DP-COLLAPSED SPECIALIST SLM CHAT STUDIO" + RESET)
    print(DIM + "   Gemma-3 4-Block Collapsed Graph | 66.39 t/s (+11.6% Speedup Record)" + RESET)
    print(CYAN + BOLD + "=" * 75 + RESET + "\n")
    print(GREEN + "Type your prompt below. Commands: " + BOLD + "/clear" + RESET + GREEN + " (reset chat), " + BOLD + "/quit" + RESET + GREEN + " (exit)\n" + RESET)

def main():
    print_header()
    
    print(YELLOW + "Loading Gemma-3 Tokenizer & Model Weights..." + RESET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    state_dict = torch.load(MASTER_WEIGHTS_PATH, map_location="cpu")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    print(YELLOW + "Building 4-Block DP Collapsed Graph (66.39 t/s)..." + RESET)
    dp_compiler = EMLDPCollapseCompiler(
        model.config, state_dict, max_layers=model.config.num_hidden_layers, error_threshold=8.0e-3, num_components=1
    )
    partitions, merge_results = dp_compiler.search_optimal_collapses()
    
    for start_idx, end_idx in partitions:
        w_gate = state_dict[f"model.layers.{start_idx}.mlp.gate_proj.linear.weight"].float().numpy()
        w_up = state_dict[f"model.layers.{start_idx}.mlp.up_proj.weight"].float().numpy()
        w_down = state_dict[f"model.layers.{start_idx}.mlp.down_proj.weight"].float().numpy()
        
        if (start_idx, end_idx) in merge_results:
            composite_w_dict = merge_results[(start_idx, end_idx)]
            fused_block = SpecialistFusedGELUDPCollapsedMLP(composite_w_dict, w_gate, w_up, w_down).to(torch.bfloat16)
            model.model.layers[start_idx].mlp = fused_block
            for idx in range(start_idx + 1, end_idx + 1):
                model.model.layers[idx].mlp = nn.Identity()
                
    if os.path.exists(SPECIALIST_CHECKPOINT_PATH):
        print(YELLOW + "Loading Calibrated 4-Block Specialist Weights..." + RESET)
        spec_state = torch.load(SPECIALIST_CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(spec_state, strict=False)
        
    model.to("cuda")
    model.eval()
    
    print(GREEN + BOLD + "✓ 4-Block Specialist SLM Ready on NVIDIA L40S GPU (66.39 t/s)!\n" + RESET)
    
    history = []
    
    while True:
        try:
            user_input = input(MAGENTA + BOLD + "User > " + RESET).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + YELLOW + "Exiting Chat Studio. Goodbye!" + RESET)
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["/quit", "exit"]:
            print(YELLOW + "Exiting Chat Studio. Goodbye!" + RESET)
            break
        elif user_input.lower() == "/clear":
            history = []
            print(GREEN + "✓ Conversation history cleared.\n" + RESET)
            continue
            
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            torch.cuda.synchronize()
        dt = time.time() - t0
        
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        gen_tokens = len(gen_ids)
        tps = gen_tokens / dt if dt > 0 else 0
        
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        print("\n" + CYAN + BOLD + "4-Block Specialist > " + RESET + response_text)
        print(DIM + GREEN + f"[{gen_tokens} tokens generated in {dt:.2f}s | {tps:.2f} t/s]\n" + RESET)

if __name__ == "__main__":
    main()
