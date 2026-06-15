import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowBlock(nn.Module):
    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_size * 2) # For Scale and Shift (FiLM)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, cond):
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        res = x
        x = self.norm(x)
        x = x * (1 + gamma) + beta
        x = self.mlp(x)
        return x + res

class FlowHead(nn.Module):
    def __init__(self, hidden_size, num_waypoints=16, state_dim=4, head_hidden_size=2048):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.state_dim = state_dim
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        # Action Input Projection
        self.action_in = nn.Linear(num_waypoints * state_dim, head_hidden_size)
        
        # Conditioning: hidden_state (576) + time (512) = 1088
        cond_dim = hidden_size + 512
        
        # ResNet Blocks
        self.blocks = nn.ModuleList([
            FlowBlock(head_hidden_size, cond_dim),
            FlowBlock(head_hidden_size, cond_dim),
            FlowBlock(head_hidden_size, cond_dim),
            FlowBlock(head_hidden_size, cond_dim)
        ])
        
        self.out = nn.Linear(head_hidden_size, num_waypoints * state_dim)
        
    def forward(self, hidden_state, noisy_actions, tau):
        t_emb = self.time_mlp(tau.squeeze(-1) * 1000.0)
        cond = torch.cat([hidden_state, t_emb], dim=-1)
        
        x = self.action_in(noisy_actions)
        for block in self.blocks:
            x = block(x, cond)
            
        return self.out(x)

class SmolVLA(nn.Module):
    def __init__(self):
        super().__init__()
        # Load native SmolVLM-256M-Instruct base model
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16
        )
        
        # SmolVLM-256M text_model hidden size is 576
        self.hidden_size = self.vlm_model.config.text_config.hidden_size 
        
        # State Projection: 4 -> 1 token
        self.state_proj = nn.Linear(4, self.hidden_size)
        nn.init.normal_(self.state_proj.weight, mean=0, std=0.13)
        nn.init.zeros_(self.state_proj.bias)
        
        # Flow Head
        self.flow_head = FlowHead(self.hidden_size).to(dtype=torch.float32)
        
        # LoRA Configuration
        self.apply_lora()
        
        # Ensure critical modules are trainable
        self.state_proj.requires_grad_(True)
        self.flow_head.requires_grad_(True)
            
        # Native Multi-modal Connector (Projector)
        for param in self.vlm_model.model.connector.parameters():
            param.requires_grad = True
            
        # Unfreeze last 4 layers of the language model backbone
        layers = self.vlm_model.model.text_model.layers
        for i in range(len(layers) - 4, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

    def apply_lora(self):
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None 
        )
        self.vlm_model.model.text_model = get_peft_model(self.vlm_model.model.text_model, lora_config)

    def forward(self, vision_embeddings, state, input_ids, noisy_actions=None, tau=None):
        batch_size = vision_embeddings.shape[0]
        n_vision_tokens = vision_embeddings.shape[1] // self.hidden_size
        v_tokens = vision_embeddings.view(batch_size, n_vision_tokens, self.hidden_size).to(dtype=torch.bfloat16)
        s_tokens = self.state_proj(state.to(dtype=torch.float32)).unsqueeze(1).to(dtype=torch.bfloat16)
        w_tokens = self.vlm_model.model.text_model.get_input_embeddings()(input_ids)
        
        # Sequence: [VISION] [INSTRUCTIONS] [STATE]
        tokens = torch.cat([v_tokens, w_tokens, s_tokens], dim=1)
        
        outputs = self.vlm_model.model.text_model(inputs_embeds=tokens)
        last_hidden_state = outputs.last_hidden_state # [B, Seq_Len, hidden_size]
        
        # Use the last token's hidden state (the [STATE] token)
        hidden_state = last_hidden_state[:, -1, :].to(dtype=torch.float32)
        
        if noisy_actions is not None and tau is not None:
            return self.flow_head(hidden_state, noisy_actions.to(dtype=torch.float32), tau.to(dtype=torch.float32))
        
        return hidden_state

    @torch.no_grad()
    def predict_action(self, vision_embeddings, state, input_ids, num_steps=32):
        """Inference using Euler integration."""
        self.eval()
        device = vision_embeddings.device
        batch_size = vision_embeddings.shape[0]
        hidden_state = self.forward(vision_embeddings, state, input_ids)
        x = torch.randn(batch_size, 16 * 4, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.ones(batch_size, 1, device=device) * (i * dt)
            velocity = self.flow_head(hidden_state, x, tau)
            x = x + velocity * dt
        return x

if __name__ == "__main__":
    model = SmolVLA()
    print("Model initialized.")
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
