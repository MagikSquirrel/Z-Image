import torch
import os
from pathlib import Path
from diffusers import ZImagePipeline
import os
from pathlib import Path
import time

import torch
from utils.helpers import ensure_model_weights
from utils.selector import select_device
from utils import AttentionBackend, load_from_local_dir, set_attention_backend
from zimage import generate

# File extension constant
FILE_EXT = ".png"
# Output directory for generated images
IMG_DIR = "outputs"

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
#pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
#pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

# Create img directory if it doesn't exist
img_dir = Path(IMG_DIR)
img_dir.mkdir(exist_ok=True)

# Load prompts from CSV file (split on first two commas)
prompts_to_generate = []
with open("prompts.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # Split on first two commas
        comma_pos1 = line.find(",")
        if comma_pos1 == -1:
            continue
        comma_pos2 = line.find(",", comma_pos1 + 1)
        if comma_pos2 == -1:
            continue
        
        filename = line[:comma_pos1].strip()
        force = line[comma_pos1 + 1:comma_pos2].strip()
        prompt = line[comma_pos2 + 1:].strip()
        
        # Skip commented out rows
        if filename.startswith("#"):
            continue
        
        # Check if output file already exists
        filepath = os.path.join(IMG_DIR, filename + FILE_EXT)
        if os.path.exists(filepath) and force != "t":
            continue

        prompts_to_generate.append((filename, filepath, prompt, force))

# Setup model
model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)  # True to verify with md5
dtype = torch.bfloat16
compile = False
height = 512
width = 512
num_inference_steps = 3
guidance_scale = 0.0
attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
output_dir = Path(IMG_DIR)
output_dir.mkdir(exist_ok=True)
device = select_device()

components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
AttentionBackend.print_available_backends()
set_attention_backend(attn_backend)
print(f"Chosen attention backend: {attn_backend}")

# Generate images sequentially
idx=1
for filename, filepath, prompt, force in prompts_to_generate:
    generator = torch.Generator(device).manual_seed(42)
    print(f"Generating {filepath}: {prompt}")
    
    start_time = time.time()
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - start_time    
    images[0].save(filepath)

    print(f"[{idx}] Saved {filepath} in {elapsed:.2f} seconds")
    idx += 1