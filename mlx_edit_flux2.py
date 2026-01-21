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
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
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
        return mx.metal.get_active_memory() / 1024**3 if hasattr(mx.metal, 'get_active_memory') else 0
    except:
        return 0


def load_model(model="flux2-klein-4b", quantize: int = 8):
    """Load the FLUX.2-klein model.

    Args:
        model: Model variant - "flux2-klein-4b" or "flux2-klein-9b"
        quantize: Quantization level (3, 4, 5, 6, 8, or None for no quantization)
    """
    print(f"Loading {model} (MLX)...")
    print(f"  Quantization: {quantize}-bit" if quantize else "  Quantization: None (full precision)")
    print(f"  [MEM] Before loading: {get_memory_usage():.2f} GB")

    # Get model config
    if model == "flux2-klein-4b":
        config = ModelConfig.flux2_klein_4b()
    elif model == "flux2-klein-9b":
        config = ModelConfig.flux2_klein_9b()
    else:
        raise ValueError(f"Unknown model: {model}")

    # Load model
    flux_model = Flux2Klein(
        quantize=quantize,
        model_config=config,
    )

    print(f"  [MEM] After loading: {get_memory_usage():.2f} GB")
    print("Model loaded!")
    return flux_model


def edit_image(
    model,
    prompt: str,
    input_image: Path | str,
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    image_strength: float = 0.4,
    seed: int = None,
):
    """Edit an image using text prompt (image-to-image).

    Args:
        model: Loaded Flux2Klein model
        prompt: Text prompt for editing
        input_image: Path to input image
        height: Output height
        width: Output width
        steps: Number of inference steps
        image_strength: How strongly the input image influences output (0.0-1.0)
        seed: Random seed

    Note: FLUX.2 does not support CFG guidance, so guidance is fixed at 1.0
    """
    import mlx.core as mx

    if seed is None:
        seed = int(time.time() * 1000000) % 2**32

    print(f"Editing with seed {seed}...")
    print(f"  Image strength: {image_strength}")
    print(f"  [MEM] Before generation: {get_memory_usage():.2f} GB")

    start_time = time.time()

    # Generate image with image-to-image
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,  # FLUX.2 does not support CFG
        image_path=str(input_image),
        image_strength=image_strength,
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
        description="Edit images with FLUX.2-klein using MLX (mflux)"
    )
    parser.add_argument("prompt", type=str, help="Text prompt for image editing")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--model",
        type=str,
        default="flux2-klein-4b",
        choices=["flux2-klein-4b", "flux2-klein-9b"],
        help="Model variant (default: flux2-klein-4b)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target longest side (default: 1024)",
    )
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="Image strength: how much input influences output (0.0-1.0, default: 0.4)",
    )
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

    args = parser.parse_args()

    # Handle quantization flag
    quantize = None if args.no_quantize else args.quantize

    # Validate input image
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return

    # Load input image
    img = Image.open(args.input)
    img_width, img_height = img.size
    print(f"Input image: {args.input} ({img_width}x{img_height})")

    # Calculate output dimensions maintaining aspect ratio
    width, height = calculate_dimensions(img_width, img_height, args.resolution)
    print(f"Output size: {width}x{height}")

    # Load model
    model = load_model(args.model, quantize)

    # Edit image
    image, seed = edit_image(
        model,
        args.prompt,
        args.input,
        height,
        width,
        args.steps,
        args.strength,
        args.seed,
    )

    # Save image
    output_path = Path(args.output)
    image.save(str(output_path))
    print(f"Saved to {output_path} (seed: {seed})")


if __name__ == "__main__":
    main()
