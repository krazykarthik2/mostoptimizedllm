import os
import json
import urllib.request
import urllib.parse
import torch
from model import Gemma3EMLKANMLP
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# 1. Real Tool Implementations
# ==============================================================================

def execute_calculator(expression):
    # Safe character filtering for basic math operations
    clean_expr = "".join(c for c in expression if c in "0123456789+-*/(). ")
    try:
        # Evaluate mathematical string
        result = eval(clean_expr)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def execute_web_search(query):
    # Real live Wikipedia API search query
    encoded_query = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return "No search results found."
            # Retrieve the snippet of the top result
            top_result = search_results[0]
            title = top_result.get("title", "")
            snippet = top_result.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            return f"Top Result: {title}. Summary: {snippet}"
    except Exception as e:
        return f"Search error: {e}"

# ==============================================================================
# 2. Agent Loop Controller
# ==============================================================================

def run_agent_turn(model, tokenizer, chat_history, device, max_new_tokens=150):
    inputs = tokenizer(chat_history, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return generated_text.strip()

def run_agent_loop(model, tokenizer, prompt, device):
    print("\n" + "="*80)
    print(f"User Prompt: '{prompt}'")
    print("="*80)
    
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # First Model Turn
    model_output = run_agent_turn(model, tokenizer, chat_prompt, device)
    print(f"Model: {model_output}")
    
    # Check for [TOOL_CALL] trigger in the output
    if "[TOOL_CALL]" in model_output:
        try:
            # Extract JSON block
            json_start = model_output.find("[TOOL_CALL]") + len("[TOOL_CALL]")
            # Look for tool response or end of string
            json_end = model_output.find("[TOOL_RESPONSE]")
            if json_end == -1:
                json_str = model_output[json_start:].strip()
            else:
                json_str = model_output[json_start:json_end].strip()
                
            tool_data = json.loads(json_str)
            tool_name = tool_data.get("tool")
            tool_args = tool_data.get("args", {})
            
            # Execute real tool call
            print(f"\n>>> INTERCEPTED TOOL CALL: {tool_name} with args: {tool_args}")
            if tool_name == "calculator":
                expr = tool_args.get("expression", "")
                print(f"    Executing Calculator: {expr}")
                tool_result = execute_calculator(expr)
            elif tool_name == "google_search" or tool_name == "web_search":
                q = tool_args.get("query", "")
                print(f"    Executing Live Wikipedia Search: {q}")
                tool_result = execute_web_search(q)
            else:
                tool_result = f"Unknown tool: {tool_name}"
                
            print(f"    Real Tool Output: {tool_result}")
            
            # Construct the response token history
            tool_response_text = f"\n[TOOL_RESPONSE] {tool_result}\n"
            # Append tool response to the history and resume generation
            new_chat_history = chat_prompt + model_output + tool_response_text
            
            print("\n>>> RESUMING GENERATION WITH REAL TOOL RESULTS...")
            final_output = run_agent_turn(model, tokenizer, new_chat_history, device)
            print(f"Model Final Answer:\n{final_output}")
            
        except Exception as e:
            print(f"\n[Error parsing tool call]: {e}")
    else:
        print("\n(No tool call requested by the model. Finished.)")

def main():
    model_id = "google/gemma-3-1b-it"
    weights_path = "gemma3_eml_kan/model_state_skills.pt"
    device = "cuda:0"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading EML-KAN model with weights: {weights_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].mlp = Gemma3EMLKANMLP(model.config).to(torch.bfloat16).to(device)
        
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Define basic test set
    test_set = [
        # Case 1: Calculator Math Task
        "Calculate (847 + 293) * 12 using the calculator tool.",
        # Case 2: Live Wikipedia Search Task
        "Search the web for the capital of France and tell me.",
        # Case 3: Logical/Probability Reasoning Task (No Tool Call)
        "If a box contains 3 red balls and 5 blue balls, and I take out 2 red balls, what is the probability of drawing a red ball next? Explain your reasoning step-by-step."
    ]
    
    for prompt in test_set:
        run_agent_loop(model, tokenizer, prompt, device)

if __name__ == "__main__":
    main()
