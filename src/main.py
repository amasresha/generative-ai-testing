"""
Main generation module for text and images.

- Generates text using Hugging Face GPT-2 and images using Stable Diffusion (GPU only).
- Provides `generate()` API for test code; returns (texts, images) for a prompt.
- Uses placeholder images if GPU is not available.
"""

from transformers import pipeline as hf_pipeline, logging as hf_logging
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont

# Silence HF warnings
hf_logging.set_verbosity_error()

# 1) Choose device & dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 2) Text pipeline (real GPT-2)
text_gen = hf_pipeline(
    "text-generation", model="gpt2", device=0 if device == "cuda" else -1
)

# 3) Image pipeline only if GPU is present
if device == "cuda":
    img_gen = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    img_gen.set_progress_bar_config(disable=True)
else:
    img_gen = None  # marks CPU mode


def _make_placeholder(size=(512, 512), text="CPU Placeholder"):  # noqa: C901
    """Generate a simple placeholder image with centered text."""
    img = Image.new("RGB", size, color="gray")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Compute text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        mask = font.getmask(text)
        w, h = mask.size
    # Draw text centered
    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), text, fill="black", font=font)
    return img


def generate(prompt: str, num_images: int = 1):
    """
    Generate text and images for a prompt.
    - Text: real GPT-2 on CPU/GPU.
    - Images: real SD on GPU, placeholder on CPU.
    """
    # Text generation
    outputs = text_gen(prompt, max_length=50, do_sample=True, truncation=True)
    texts = [o["generated_text"] for o in outputs]

    # Image generation or placeholder
    if device == "cuda":
        result = img_gen(
            prompt, num_inference_steps=25, num_images_per_prompt=num_images
        )
        images = result.images
    else:
        images = [_make_placeholder() for _ in range(num_images)]

    return texts, images


if __name__ == "__main__":
    sample = "A serene lake at sunrise"
    txts, imgs = generate(sample, num_images=1)
    print("Text:", txts)
    for i, img in enumerate(imgs):
        img.save(f"out_{i}.png")
