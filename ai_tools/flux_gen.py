"""Flux2 text-to-image generation CLI for Apple Silicon (MPS)."""

import argparse
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
import warnings

from mps_common import (
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

    generator = get_generator(args.seed)
    seed = generator.initial_seed()

    if not args.quiet:
        print(f"Seed: {seed}")
        print(f"Prompt: {args.prompt}")
        print("Loading model...")

    pipe = load_pipeline(verbose=not args.quiet)

    if args.no_memory_opts:
        disable_memory_opts(pipe)

    if not args.quiet:
        print(f"Generating: {args.width}x{args.height}")

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

    final_path = save_image_clean(result.images[0], Path(args.output))
    print(f"Image saved to: {final_path}")

    cleanup_memory()


if __name__ == "__main__":
    main()
