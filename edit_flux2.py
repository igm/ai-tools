"""
FLUX.2-klein-9B - Image-to-Image Editing on Mac

Supports both 4-bit SDNQ quantized model and full precision model.
"""

import os
import argparse

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


def calculate_dimensions(img_width: int, img_height: int, target_longest_side: int) -> tuple:
    """Calculate output dimensions maintaining aspect ratio."""
    aspect_ratio = img_width / img_height

    if aspect_ratio >= 1:
        # Landscape or square
        new_width = target_longest_side
        new_height = int(target_longest_side / aspect_ratio)
    else:
        # Portrait
        new_height = target_longest_side
        new_width = int(target_longest_side * aspect_ratio)

    # Round to multiples of 64 (required by model)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    # Clamp to valid range
    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def load_pipeline(device="mps", model="sdnq"):
    """Load the FLUX.2-klein-9B pipeline.

    Args:
        device: Device to load model on (mps, cuda, cpu)
        model: Model variant - "sdnq" for 4-bit quantized, "full" for full precision
    """
    # Use bfloat16 for better quality
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    if model == "full":
        print("Loading FLUX.2-klein-9B (full precision)...")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"  [MEM] Before loading: {get_memory_usage():.2f} GB")

        print("  Loading pipeline...")
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=dtype,
        )
    else:  # sdnq
        print("Loading FLUX.2-klein-9B (4-bit SDNQ)...")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"  [MEM] Before loading: {get_memory_usage():.2f} GB")

        from transformers import AutoTokenizer
        import sdnq  # Import to register SDNQ quantization type

        print("  Loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            subfolder="tokenizer",
            use_fast=False,
        )

        print("  Loading pipeline...")
        pipe = Flux2KleinPipeline.from_pretrained(
            "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
            tokenizer=tokenizer,
            torch_dtype=dtype,
        )

    print(f"  [MEM] After loading: {get_memory_usage():.2f} GB")

    pipe.to(device)
    print(f"  [MEM] After pipe.to(device): {get_memory_usage():.2f} GB")

    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    print(f"  [MEM] After optimizations: {get_memory_usage():.2f} GB")
    print("Pipeline loaded!")
    return pipe


def edit_image(
    pipe,
    prompt: str,
    input_images: list[Image.Image],
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    guidance: float = 3.5,
    seed: int = None,
    device: str = "mps",
):
    """Edit an image using text prompt."""
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Editing with seed {seed}...")

    # Resize all input images to target dimensions
    images_to_process = []
    for img in input_images[:6]:  # Max 6 images
        resized = img.copy().resize((width, height), Image.LANCZOS)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        images_to_process.append(resized)
    print(f"  Processing {len(images_to_process)} image(s)...")

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    print(f"  [MEM] Before generation: {get_memory_usage():.2f} GB")

    # Disable VAE tiling for img2img (better quality)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
        pipe.vae.disable_tiling()

    with torch.inference_mode():
        # Pass single image or list based on count
        image = pipe(
            prompt=prompt,
            image=images_to_process[0] if len(images_to_process) == 1 else images_to_process,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

    # Re-enable VAE tiling
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    print(f"  [MEM] After generation: {get_memory_usage():.2f} GB")

    # Force memory cleanup
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="Edit images with FLUX.2-klein-9B")
    parser.add_argument("prompt", type=str, help="Text prompt for image editing")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input image path(s) - up to 6 images")
    parser.add_argument("--model", type=str, default="sdnq", choices=["sdnq", "full"],
                        help="Model: sdnq (4-bit, low VRAM) or full (higher quality)")
    parser.add_argument("--resolution", type=int, default=1024, help="Target longest side (default: 1024)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load input images
    input_images = []
    for path in args.input:
        if not os.path.exists(path):
            print(f"Warning: Input image not found: {path}")
            continue
        img = Image.open(path)
        img_width, img_height = img.size
        print(f"Input image: {path} ({img_width}x{img_height})")
        input_images.append(img)

    if not input_images:
        print("Error: No valid input images found")
        return

    # Calculate output dimensions from first image's aspect ratio
    first_img_width, first_img_height = input_images[0].size
    width, height = calculate_dimensions(first_img_width, first_img_height, args.resolution)
    print(f"Output size: {width}x{height} from {len(input_images)} input image(s)")

    pipe = load_pipeline(device, args.model)

    image, seed = edit_image(
        pipe,
        args.prompt,
        input_images,
        height,
        width,
        args.steps,
        args.guidance,
        args.seed,
        device,
    )

    image.save(args.output)
    print(f"Saved to {args.output} (seed: {seed})")


if __name__ == "__main__":
    main()
