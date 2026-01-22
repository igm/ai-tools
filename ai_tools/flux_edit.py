"""Flux2 image-to-image editing CLI for Apple Silicon (MPS)."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import warnings

from ai_tools.mps_common import (
    get_memory_gb,
    cleanup_memory,
    enable_memory_opts,
    disable_memory_opts,
    save_image_clean,
    get_generator,
)

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_image_list(images_str: str) -> List[Path]:
    """Parse comma-separated image paths."""
    if not images_str:
        return []
    paths = []
    for p in images_str.split(","):
        p = p.strip()
        if p and Path(p).exists():
            paths.append(Path(p))
        elif p:
            print(f"Warning: Image file not found: {p}", file=sys.stderr)
    return paths


def load_images(image_paths: List[Path]) -> List[Image.Image]:
    """Load PIL Images from paths."""
    return [Image.open(p).convert("RGB") for p in image_paths]


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

    enable_memory_opts(pipe, verbose=verbose)

    if verbose:
        print("Pipeline loaded!")

    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="Edit images using Flux2 Klein img2img (MPS only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Examples:
  %(prog)s "turn into oil painting" -i photo.jpg
  %(prog)s "in anime style" -i portrait.jpg -o anime_portrait.png
  %(prog)s "add sunglasses" -i photo.jpg --match-size 0
  %(prog)s "make it cyberpunk" -i input.jpg -s 8 --seed 42
        """
    )

    parser.add_argument("prompt", help="Text prompt for editing")
    parser.add_argument("-i", "--input-images", required=True, help="Input image path(s), comma-separated")
    parser.add_argument("-o", "--output", default="flux-edit.png", help="Output filename")
    parser.add_argument("-r", "--resolution", type=int, default=None, help="Max dimension (longest side). Default: match input image")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-g", "--guidance", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument("--no-memory-opts", action="store_true", help="Disable memory optimizations")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Parse input images
    input_images = parse_image_list(args.input_images)
    if not input_images:
        sys.exit(f"Error: No valid input images found in: {args.input_images}")

    generator = get_generator(args.seed)
    seed = generator.initial_seed()

    if not args.quiet:
        print(f"Seed: {seed}")
        print(f"Prompt: {args.prompt}")
        print("Loading model...")

    pipe = load_pipeline(verbose=not args.quiet)

    if args.no_memory_opts:
        disable_memory_opts(pipe)

    # Load input images
    if not args.quiet:
        print(f"Loading {len(input_images)} input image(s)...")

    pil_images = load_images(input_images)

    # Determine dimensions from input image
    img_w, img_h = pil_images[0].size

    if args.resolution is None:
        # Match input image dimensions
        width, height = img_w, img_h
    else:
        # Scale to fit within resolution (longest side)
        if img_w >= img_h:
            # Landscape: width is the longer side
            width = args.resolution
            height = int(img_h * (args.resolution / img_w))
        else:
            # Portrait: height is the longer side
            height = args.resolution
            width = int(img_w * (args.resolution / img_h))

    if not args.quiet:
        print(f"Editing: {width}x{height} (input: {img_w}x{img_h})")

    generate_kwargs = {
        "prompt": args.prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
        "image": pil_images[0],  # Use first image as primary input
    }

    if not args.quiet:
        print(f"  [MEM] Before generation: {get_memory_gb():.2f} GB")

    with torch.inference_mode():
        result = pipe(**generate_kwargs)

    if not args.quiet:
        print(f"  [MEM] After generation: {get_memory_gb():.2f} GB")

    final_path = save_image_clean(result.images[0], Path(args.output))
    print(f"Image saved to: {final_path}")

    cleanup_memory()


if __name__ == "__main__":
    main()
