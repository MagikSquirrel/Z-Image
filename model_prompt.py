import torch
import os
import csv
from pathlib import Path
from diffusers import ZImagePipeline

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

# Output directory for generated images
IMG_DIR = "img"

# Create img directory if it doesn't exist
img_dir = Path(IMG_DIR)
img_dir.mkdir(exist_ok=True)

# Load prompts from CSV file
prompts_to_generate = []
with open("prompts.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            prompts_to_generate.append((row[0], row[1]))

# Generate images sequentially if they don't exist
for filename, prompt in prompts_to_generate:
    filepath = os.path.join(IMG_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Generating {filepath}...")
        
        # 2. Generate Image
        image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

        image.save(filepath)
        print(f"Saved {filepath}")
    else:
        print(f"{filepath} already exists, skipping...")