import os, gc, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from model import Gemma3EMLKANAttention, Gemma3EMLKANGatedMLP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

NAMES = ["John", "Alice", "Bob", "Emma", "David", "Mary", "James", "Sarah"]
ITEMS = ["apples", "oranges", "books", "pens", "coins"]
CODE_TASKS = [
    ("prime check", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
    ("reverse string", "def reverse_string(s):\n    return s[::-1]"),
    ("factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
    ("binary search", "def binary_search(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid]==t: return mid\n        elif arr[mid]<t: lo=mid+1\n        else: hi=mid-1\n    return -1"),
    ("gcd", "def gcd(a,b):\n    while b: a,b = b, a%b\n    return a"),
    ("fibonacci", "def fib(n):\n    a,b = 0,1\n    for _ in range(n): a,b = b,a+b\n    return a"),
    ("vowel count", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
]

def gen_data(n=600):
    examples = []
    for _ in range(n//3):
        n1,n2 = random.sample(NAMES,2)
        item = random.choice(ITEMS)
        s,g,b = random.randint(5,25), random.randint(1,4), random.randint(2,10)
        f = s-g+b
        examples.append((f"I am {n1} and I have {s} {item}. I give {g} to {n2} and buy {b} more. How many?",
            f"Step 1: Start with {s}. Step 2: Give {g}: {s}-{g}={s-g}. Step 3: Buy {b}: {s-g}+{b}={f}. Answer: {f}"))
    for _ in range(n//3):
        a,b = random.randint(2,30), random.randint(2,30)
        examples.append((f"What is {a}*{b}+{a+b}?", f"{a}*{b}={a*b}. {a*b}+{a+b}={a*b+a+b}. Answer: {a*b+a+b}"))
    for _ in range(n//3):
        desc,code = random.choice(CODE_TASKS)
        examples.append((f"Write Python to {desc}.", f"```python\n{code}\n```"))
    random.shuffle(examples)
    return examples

class SimpleDataset(Dataset):
    def __init__(self, examples, tokenizer, seq_len=384):
        self.examples = []
        all_tokens = []
        for p, a in examples:
            msgs = [{"role":"user","content":p},{"role":"model","content":a}]
            fmt = tokenizer.apply_chat_template(msgs, tokenize=False)
            all_tokens.extend(tokenizer.encode(fmt, add_special_tokens=False))
        shakespeare = os.path.join(os.path.dirname(__file__), "../MHNKAN/shakespeare.txt")
        if os.path.exists(shakespeare):
            with open(shakespeare, "r") as f:
                all_tokens.extend(tokenizer.encode(f.read(), add_special_tokens=False))
        for i in range(0, len(all_tokens)-seq_len, seq_len):
            self.examples.append(torch.tensor(all_tokens[i:i+seq_len], dtype=torch.long))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def gen(model, tok, prompt, device):
    msgs = [{"role":"user","content":prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False,
            repetition_penalty=1.2, no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def run(weights_path, save_path, max_steps=2000, lr=1e-4, kd_alpha=0.7, T=2.0, device="cuda"):
    model_id = "google/gemma-3-1b-it"
    tok = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # Teacher (frozen)
    print("Loading frozen teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    teacher.eval(); teacher.requires_grad_(False)

    # Student
    print("Loading student EML-KAN...")
    student = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    for i in range(config.num_hidden_layers):
        student.model.layers[i].self_attn = Gemma3EMLKANAttention(config).to(device).to(student.dtype)
        student.model.layers[i].mlp = Gemma3EMLKANGatedMLP(config).to(device).to(student.dtype)
    sd = torch.load(weights_path, map_location=device)
    ssd = student.state_dict()
    loadable = {k:v for k,v in sd.items() if k in ssd and ssd[k].shape==v.shape}
    ssd.update(loadable); student.load_state_dict(ssd)
    print(f"Loaded {len(loadable)} params.")

    # CRITICAL: freeze linear weights, ONLY train EML correction params (a,b,c,d,weight_eml)
    print("Freezing linear weights, training ONLY EML correction params...")
    student.requires_grad_(False)
    for name, param in student.named_parameters():
        if ".eml." in name or "norm" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Data
    examples = gen_data(600)
    dataset = SimpleDataset(examples, tok, seq_len=384)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr, weight_decay=0.01
    )
    warmup = min(100, max_steps//10)
    def lr_sched(step):
        if step < warmup: return float(step)/float(max(1,warmup))
        progress = float(step-warmup)/float(max(1,max_steps-warmup))
        return max(0.1, 0.5*(1+math.cos(math.pi*progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    print(f"\nTraining: {max_steps} steps, alpha={kd_alpha}, T={T}, lr={lr}")
    t0 = time.time()
    student.train()
    step = 0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps: break
            inputs = batch.to(device)
            targets = inputs[:,1:].contiguous()
            ids = inputs[:,:-1].contiguous()

            with torch.no_grad():
                t_logits = teacher(ids).logits.float()
            s_logits = student(ids).logits.float()

            kl = F.kl_div(F.log_softmax(t_logits/T,-1), F.log_softmax(s_logits/T,-1),
                          reduction="batchmean", log_target=True) * T*T
            ce = F.cross_entropy(s_logits.view(-1,s_logits.size(-1)), targets.view(-1))
            loss = kd_alpha*kl + (1-kd_alpha)*ce

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            step += 1

            if step % 200 == 0:
                print(f"  Step {step}/{max_steps} | loss={loss.item():.2f} kl={kl.item():.2f} ce={ce.item():.3f}")

            if step % 500 == 0:
                student.eval()
                print(f"\n  [Step {step}] Generation:")
                for q in ["I have 12 apples, give 3 buy 7. How many?","Write Python prime check.","If all Bloops are Razzies?"]:
                    print(f"  Q: {q[:50]}")
                    print(f"  A: {gen(student,tok,q,device)[:120]}\n")
                student.train()

            if step % 200 == 0: gc.collect(); torch.cuda.empty_cache()

    print(f"\nDone in {time.time()-t0:.0f}s")
    torch.save(student.state_dict(), os.path.join(save_path, "model_state_tuned.pt"))

    # Final eval
    student.eval()
    print("\n" + "="*80)
    print("  FINAL EVAL")
    print("="*80)
    for q in [
        "I have 12 apples. I give 3 to Alice and 5 to Bob. Then I buy 7 more. How many apples do I have? Think step by step.",
        "Write a Python function to check if a number is prime.",
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {gen(student,tok,q,device)}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", default="gemma3_eml_kan/model_state.pt")
    parser.add_argument("--save_path", default="gemma3_eml_kan")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kd_alpha", type=float, default=0.7)
    args = parser.parse_args()
    run(args.weights_path, args.save_path, args.max_steps, args.lr, args.kd_alpha)
