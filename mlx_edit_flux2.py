"""
FLUX.2-klein - MLX Native Image-to-Image Editing on Mac

Uses mflux (MLX port) for Apple Silicon acceleration.
Supports FLUX.2-klein-4B and FLUX.2-klein-9B with quantization.
"""

import os
import argparse
import time
from pathlib import Path

try:
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    from mflux.models.common.config.model_config import ModelConfig
except ImportError:
    print("Error: mflux not installed. Run: pip install mflux --pre")
    raise

from PIL import Image


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
    """Get current MLX memory usage in GB (approximate)."""
    try:
        import mlx.core as mx
        return mx.get_active_memory() / 1024**3
    except:
        return 0


def load_model(model="flux2-klein-4b", quantize: int = 8):
    """Load the FLUX.2-klein edit model.

    Args:
        model: Model variant - "flux2-klein-4b" or "flux2-klein-9b"
        quantize: Quantization level (3, 4, 5, 6, 8, or None for no quantization)
    """
    print(f"Loading {model} (MLX)...")
    print(f"  Mode: Image-to-image editing")
    print(f"  Quantization: {quantize}-bit" if quantize else "  Quantization: None (full precision)")
    print(f"  [MEM] Before loading: {get_memory_usage():.2f} GB")

    # Get model config
    if model == "flux2-klein-4b":
        config = ModelConfig.flux2_klein_4b()
    elif model == "flux2-klein-9b":
        config = ModelConfig.flux2_klein_9b()
    else:
        raise ValueError(f"Unknown model: {model}")

    # Load Flux2KleinEdit for all image editing (single or multi)
    flux_model = Flux2KleinEdit(
        quantize=quantize,
        model_config=config,
    )

    print(f"  [MEM] After loading: {get_memory_usage():.2f} GB")
    print("Model loaded!")
    return flux_model


def edit_image(
    model,
    prompt: str,
    input_images: list[Path | str],
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    seed: int = None,
):
    """Edit an image using text prompt (image-to-image).

    Args:
        model: Loaded Flux2KleinEdit model
        prompt: Text prompt for editing
        input_images: List of paths to input images
        height: Output height
        width: Output width
        steps: Number of inference steps
        seed: Random seed

    Note: FLUX.2 does not support CFG guidance, so guidance is fixed at 1.0
    """
    if seed is None:
        seed = int(time.time() * 1000000) % 2**32

    print(f"Editing with seed {seed}...")
    print(f"  Input images: {len(input_images)}")
    print(f"  [MEM] Before generation: {get_memory_usage():.2f} GB")

    start_time = time.time()

    # Convert input paths to strings
    image_paths = [str(p) for p in input_images]

    # Generate using Flux2KleinEdit with image_paths (list)
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,  # FLUX.2 does not support CFG
        image_paths=image_paths,
    )

    elapsed = time.time() - start_time

    print(f"  [MEM] After generation: {get_memory_usage():.2f} GB")
    print(f"  Generation time: {elapsed:.2f}s")

    # Force memory cleanup
    import gc
    gc.collect()

    return image, seed


def main():
    parser = argparse.ArgumentParser(
        description="Edit images with FLUX.2-klein using MLX (mflux). "
                    "Supports both single-image editing and multi-image compositing."
    )
    parser.add_argument("prompt", type=str, help="Text prompt for image editing")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="Input image path(s). Use multiple times for multi-image editing (e.g., --input img1.jpg --input img2.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux2-klein-9b",
        choices=["flux2-klein-4b", "flux2-klein-9b"],
        help="Model variant (default: flux2-klein-9b)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target longest side (default: 1024)",
    )
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[3, 4, 5, 6, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization (use full precision)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Strip metadata from output image",
    )

    args = parser.parse_args()

    # Handle quantization flag
    quantize = None if args.no_quantize else args.quantize

    # Validate all input images
    input_paths = []
    for path in args.input:
        if not os.path.exists(path):
            print(f"Error: Input image not found: {path}")
            return
        input_paths.append(Path(path))

    # Show mode info
    if len(input_paths) > 1:
        print(f"Multi-image mode: {len(input_paths)} images")
        for p in input_paths:
            print(f"  - {p}")
    else:
        print(f"Single-image mode: {input_paths[0]}")

    # Load model (Flux2KleinEdit for all image editing)
    model = load_model(args.model, quantize)

    # Load primary image for dimension calculation
    primary_img = Image.open(input_paths[0])
    img_width, img_height = primary_img.size
    print(f"Primary image: {input_paths[0]} ({img_width}x{img_height})")

    # Calculate output dimensions maintaining aspect ratio
    width, height = calculate_dimensions(img_width, img_height, args.resolution)
    print(f"Output size: {width}x{height}")

    # Edit image(s)
    image, seed = edit_image(
        model,
        args.prompt,
        input_paths,
        height,
        width,
        args.steps,
        args.seed,
    )

    # Save image
    output_path = Path(args.output)
    image.save(str(output_path))

    # Strip metadata if requested
    if args.no_metadata:
        img = Image.open(output_path)
        img.save(output_path)

    print(f"Saved to {output_path} (seed: {seed})")


if __name__ == "__main__":
    main()
