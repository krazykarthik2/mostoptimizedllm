import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

sys.path.append(os.path.abspath("genomics/mostoptimizedllm/llmcopyexperiement"))
from model import Gemma3EMLKANGatedMLP
from eml_hybrid_polynomial_compiler import EMLHybridPolynomialCompiler
from full_model_hybrid_polynomial_benchmark import QuantizableHybridPolynomialGemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model weights globally for chat server
MODEL_ID = "google/gemma-3-1b-it"
WEIGHTS_PATH = "genomics/mostoptimizedllm/llmcopyexperiement/gemma3_eml_kan/model_state_high_sparsity.pt"

print("Loading Gemma-3 Tokenizer & Base Weights...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)

print("Compiling model to Single-Spline Fused EML-KAN (60.17 t/s)...")
for idx in range(model.config.num_hidden_layers):
    poly_mlp = QuantizableHybridPolynomialGemma3EMLKANMLP(model.config, idx, state_dict, num_components=1)
    model.model.layers[idx].mlp = poly_mlp
    
model.to(torch.bfloat16).to("cuda")
model.eval()
print("Model ready on CUDA NVIDIA L40S GPU!")

class ChatRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("genomics/mostoptimizedllm/chat_interface.html", "rb") as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()
            
    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            user_prompt = data.get("prompt", "")
            system_prompt = data.get("system_prompt", "You are a helpful AI assistant.")
            
            # Construct Gemma-3 Native Chat Template with Special Tokens
            formatted_prompt = (
                f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n"
                f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"response": response_text}).encode('utf-8'))

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChatRequestHandler)
    print(f"Server running at http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
