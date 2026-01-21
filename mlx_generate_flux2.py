"""
FLUX.2-klein - MLX Native Image Generation on Mac

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


def get_memory_usage():
    """Get current MLX memory usage in GB (approximate)."""
    try:
        import mlx.core as mx
        # MLX doesn't have a direct memory API like PyTorch
        # This is a placeholder - actual MLX memory tracking is different
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


def generate(
    model,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    seed: int = None,
):
    """Generate an image from a prompt.

    Note: FLUX.2 does not support CFG guidance, so guidance is fixed at 1.0
    """
    import mlx.core as mx

    if seed is None:
        # Generate random seed
        seed = int(time.time() * 1000000) % 2**32

    print(f"Generating with seed {seed}...")
    print(f"  [MEM] Before generation: {get_memory_usage():.2f} GB")

    start_time = time.time()

    # Generate image
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,  # FLUX.2 does not support CFG
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
        description="Generate images with FLUX.2-klein using MLX (mflux)"
    )
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--model",
        type=str,
        default="flux2-klein-4b",
        choices=["flux2-klein-4b", "flux2-klein-9b"],
        help="Model variant (default: flux2-klein-4b)",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[3, 4, 5, 6, 8],
        help="Quantization bits (default: 8, use None for no quantization)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization (use full precision)",
    )

    args = parser.parse_args()

    # Handle quantization flag
    quantize = None if args.no_quantize else args.quantize

    # Load model
    model = load_model(args.model, quantize)

    # Generate image
    image, seed = generate(
        model,
        args.prompt,
        args.height,
        args.width,
        args.steps,
        args.seed,
    )

    # Save image
    output_path = Path(args.output)
    image.save(str(output_path))
    print(f"Saved to {output_path} (seed: {seed})")


if __name__ == "__main__":
    main()
