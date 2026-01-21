"""
Image-to-Text Description using Qwen3-VL with MLX
Generates detailed image descriptions locally on Apple Silicon.
"""

import os
import argparse

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template


def describe_images(
    image_paths: list[str],
    prompt: str = "Describe this image in detail.",
    model_id: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    """Generate description for image(s) using MLX."""
    print(f"Loading {model_id}...")

    # Load model and processor
    model, processor = load(model_id)
    config = model.config

    print(f"\nProcessing {len(image_paths)} image(s)...")
    print(f"Prompt: {prompt}\n")

    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(image_paths)
    )

    # Generation
    output_text = generate(
        model,
        processor,
        formatted_prompt,
        image_paths,
        max_tokens=max_new_tokens,
        temperature=temperature,
        verbose=False
    )

    return output_text


def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions with Qwen3-VL using MLX")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input image path(s)")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Custom prompt (default: 'Describe this image in detail.')")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-VL-8B-Instruct-4bit",
                        help="Model ID on Hugging Face")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens in output (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")

    args = parser.parse_args()

    # Validate images exist
    valid_paths = []
    for path in args.input:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Image not found: {path}")

    if not valid_paths:
        print("Error: No valid input images found")
        return

    # Generate description
    result = describe_images(
        image_paths=valid_paths,
        prompt=args.prompt,
        model_id=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print("DESCRIPTION:")
    print("=" * 60)
    print(result.text)
    print("=" * 60)


if __name__ == "__main__":
    main()
