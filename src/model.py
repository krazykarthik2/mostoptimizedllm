import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class TrajectoryHead(nn.Module):
    def __init__(self, hidden_size, num_waypoints=16, state_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_waypoints * state_dim)
        )
        
    def forward(self, x):
        # x is hidden state [B, hidden_size]
        return self.mlp(x) # [B, 64]

class SmolVLA(nn.Module):
    def __init__(self, config_name="HuggingFaceTB/SmolLM2-360M", num_layers=24):
        super().__init__()
        # Load pretrained backbone
        self.backbone = LlamaModel.from_pretrained(config_name)
        
        # Slice layers if fewer are requested (24 active out of 32)
        if num_layers < len(self.backbone.layers):
            self.backbone.layers = nn.ModuleList([self.backbone.layers[i] for i in range(num_layers)])
            self.backbone.config.num_hidden_layers = num_layers
        
        # Projections
        self.hidden_size = self.backbone.config.hidden_size # 960
        
        # Modality Aligner: Projects 768-dim SigLIP output into the SmolLM2 language embedding space (960-dim)
        self.modality_aligner = nn.Linear(768, 8 * self.hidden_size)
        
        # State Projection: 4 -> 2 tokens (2 * 960)
        self.state_proj = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.hidden_size)
        )
        
        # Output Head
        self.trajectory_head = TrajectoryHead(self.hidden_size)
        
        # LoRA Configuration
        self.apply_lora()
        
        # Freezing
        self.freeze_layers(num_layers)

    def apply_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None 
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def freeze_layers(self, total_layers):
        # User: "Freeze Layers 0–19. Train Layers 20–23."
        # Note: self.backbone is now a PeftModel, it wraps the base model.
        # Base model layers are in self.backbone.base_model.model.layers
        
        # First, freeze everything in the backbone except LoRA
        for name, param in self.backbone.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        
        # Then, unfreeze layers 20-23
        # Layer indices are 0-based.
        layers = self.backbone.base_model.model.layers
        for i in range(20, min(24, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True
                
        # Projections and head must be trainable
        for param in self.modality_aligner.parameters():
            param.requires_grad = True
        for param in self.state_proj.parameters():
            param.requires_grad = True
        for param in self.trajectory_head.parameters():
            param.requires_grad = True

    def forward(self, vision_embeddings, state, input_ids):
        # vision_embeddings: [B, 768]
        # state: [B, 4]
        # input_ids: [B, N] (Language tokens)
        
        batch_size = vision_embeddings.shape[0]
        
        # Align Vision to Language Embedding Space
        v_tokens = self.modality_aligner(vision_embeddings).view(batch_size, 8, self.hidden_size)
        
        # Project State
        s_tokens = self.state_proj(state).view(batch_size, 2, self.hidden_size)
        
        # Get Word Embeddings
        w_tokens = self.backbone.base_model.model.embed_tokens(input_ids)
        
        # Sequence: [BOS] [VISION] [STATE] [LANGUAGE]
        # BOS token is usually included in input_ids or handled manually.
        # If input_ids starts with BOS, we just cat.
        
        # Concatenate tokens
        # Note: We assume BOS is the first token of w_tokens if provided.
        # User sequence: [VISION] [STATE] [LANGUAGE]
        # Let's follow: [VISION (8)] [STATE (2)] [LANGUAGE (N)]
        tokens = torch.cat([v_tokens, s_tokens, w_tokens], dim=1)
        
        # Backbone Forward
        outputs = self.backbone(inputs_embeds=tokens)
        last_hidden_state = outputs.last_hidden_state
        
        # Predict trajectory from the last token's hidden state
        # Or pool? Let's take the last one.
        traj_hidden = last_hidden_state[:, -1, :]
        trajectory = self.trajectory_head(traj_hidden)
        
        return trajectory

if __name__ == "__main__":
    # Test model initialization
    model = SmolVLA()
    print("Model initialized.")
    # Check trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
