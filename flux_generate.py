#!/usr/bin/env python3
"""Flux2 text-to-image generation CLI for Apple Silicon (MPS)."""

import argparse
import gc
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import warnings

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)


def get_memory_gb() -> float:
    """Get current MPS memory usage in GB."""
    return torch.mps.current_allocated_memory() / 1024**3


def load_pipeline(verbose: bool = True) -> Flux2KleinPipeline:
    """Load the FLUX.2-klein-9B pipeline."""
    device = "mps"
    dtype = torch.bfloat16

    if verbose:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"  [MEM] Before loading: {get_memory_gb():.2f} GB")
        print("Loading FLUX.2-klein-9B...")

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=dtype,
    )

    if verbose:
        print(f"  [MEM] After loading: {get_memory_gb():.2f} GB")

    pipe.to(device)

    if verbose:
        print(f"  [MEM] After pipe.to(device): {get_memory_gb():.2f} GB")

    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    if verbose:
        print(f"  [MEM] After optimizations: {get_memory_gb():.2f} GB")
        print("Pipeline loaded!")

    return pipe


def cleanup_memory():
    """Force MPS memory cleanup."""
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text using Flux2 Klein (MPS only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Examples:
  %(prog)s "a cat in a hat"
  %(prog)s "a sunset over mountains" -o sunset.png
  %(prog)s "portrait of a warrior" -w 768 -h 1024 -s 8
  %(prog)s "cyberpunk city" --seed 42 -g 2.5
        """
    )

    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("-o", "--output", default="flux-output.png", help="Output filename")
    parser.add_argument("-w", "--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("-h", "--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-g", "--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--no-memory-opts", action="store_true", help="Disable memory optimizations")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Set seed
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()

    if not args.quiet:
        print(f"Seed: {seed}")
        print(f"Prompt: {args.prompt}")
        print("Loading model...")

    pipe = load_pipeline(verbose=not args.quiet)

    if args.no_memory_opts:
        pipe.disable_attention_slicing()
        if hasattr(pipe, "disable_vae_slicing"):
            pipe.disable_vae_slicing()

    if not args.quiet:
        print(f"Generating: {args.width}x{args.height}")

    generator = torch.Generator("mps").manual_seed(seed)

    generate_kwargs = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
    }

    if not args.quiet:
        print(f"  [MEM] Before generation: {get_memory_gb():.2f} GB")

    with torch.inference_mode():
        result = pipe(**generate_kwargs)

    if not args.quiet:
        print(f"  [MEM] After generation: {get_memory_gb():.2f} GB")

    # Save output (strip metadata, auto-increment if exists)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_path = output_path
    counter = 1
    while final_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        final_path = output_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1

    img = result.images[0]
    img_data = list(img.getdata())
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(img_data)
    clean_img.save(final_path, optimize=False)
    print(f"Image saved to: {final_path}")

    cleanup_memory()


if __name__ == "__main__":
    main()
