import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
device = "cpu"
vlm_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct").to(device).eval()
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
inputs = processor(images=[np.zeros((512, 512, 3), dtype=np.uint8)], size={"longest_edge": 512}, return_tensors="pt")
with torch.no_grad():
    px = inputs.pixel_values.view(-1, 3, 512, 512)
    vision_outputs = vlm_model.model.vision_model(pixel_values=px)
    print("Vision Out:", vision_outputs.last_hidden_state.shape)
    v_tokens = vlm_model.model.connector(vision_outputs.last_hidden_state)
    print("Connector Out:", v_tokens.shape)
    print("Flattened length:", v_tokens.numpy().flatten().shape)
