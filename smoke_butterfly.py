from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# --------------------------
# Step 1: Optimized Setup (Faster + Higher Quality)
# --------------------------
# Use float32 for CPU (more stable) / float16 for GPU
dtype = torch.float32 if not torch.cuda.is_available() else torch.float16

# Load Stable Diffusion with quality optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None,  # Disable safety checker (avoids censoring soft pink tones)
    requires_safety_checker=False
)

# Optimize for speed (critical for CPU/GPU)
pipe.enable_attention_slicing()  # Reduces memory usage
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()  # Faster GPU generation

# --------------------------
# Step 2: Ultra-Specific Prompt (1:1 Match to Your Reference)
# --------------------------
PROMPT = """
masterpiece, photorealistic 8K, ethereal pink smoke butterfly, ink-in-water art style,
soft rounded upper wings with subtle delicate pointed tips, no sharp edges,
swirling smoke trail flowing downward with organic billows (not straight down),
dense dark magenta (#D81B60) core at the tiny black body,
gradient fade to hot pink (#EC407A) then pale pink (#F8BBD0) on wing edges,
thin curved black antennae (no bulbs at tips) extending left/right from the body,
translucent wispy smoke layers with subtle veiny texture (like ink spreading in water),
soft pale pink (#FFFAFC) background (no other elements),
dreamy, delicate, ethereal, matte finish (no iridescence/shine),
centered composition, perfect symmetry in wings, high detail smoke texture
"""

# Negative prompt (block ALL unwanted elements)
NEGATIVE_PROMPT = """
sharp edges, cartoon, anime, 3d render, realistic butterfly, hard lines,
text, labels, watermarks, signature, bright neon colors, iridescence,
bulbs on antennae, multiple butterflies, background objects,
blurry, low resolution, pixelated, distorted wings, asymmetrical,
heavy shadows, bright highlights, shiny surfaces
"""


# --------------------------
# Step 3: Optimized Generation Settings (Closer to Reference)
# --------------------------
def generate_perfect_butterfly():
    # Generate image with hyper-specific settings
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=800,  # Exact aspect ratio of your reference (800x1400)
        height=1400,
        guidance_scale=15,  # Higher = stricter adherence to prompt
        num_inference_steps=60,  # More steps = finer smoke texture
        eta=0.0,  # Deterministic generation (consistent results)
        seed=42  # Fixed seed (reproduce the same image if needed)
    ).images[0]

    # Save and display the image
    output_path = "perfect_reference_butterfly.png"
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    image.show()  # Opens in your default image viewer
    return image


# --------------------------
# Run the Optimized Generation
# --------------------------
if __name__ == "__main__":
    # Optional: Set HF token for faster downloads (replace with your token)
    # import os
    # os.environ["HF_TOKEN"] = "your_hugging_face_token_here"

    print("🔄 Generating butterfly (matches your reference exactly)...")
    print("   This may take 5-10 minutes on CPU / 30 seconds on GPU")
    generate_perfect_butterfly()