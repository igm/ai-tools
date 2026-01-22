"""Qwen3-VL-8B-Instruct image-to-text description CLI for Apple Silicon (MPS)."""

import argparse
import os
import sys
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from igm.ai_tools.mps_common import cleanup_memory, get_memory_gb

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"


def validate_and_load_image(image_input: str) -> Image.Image:
    """Validate and load image from file path or URL.

    Args:
        image_input: File path or URL to image

    Returns:
        PIL Image object

    Raises:
        ValueError: If input is invalid or image cannot be loaded
    """
    path = Path(image_input)

    # Check if it's a URL
    if image_input.startswith(("http://", "https://")):
        try:
            response = requests.get(image_input, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    # Check if it's a local file
    if not path.exists():
        raise ValueError(f"File not found: {image_input}")

    if not path.is_file():
        raise ValueError(f"Not a file: {image_input}")

    try:
        return Image.open(image_input).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def build_messages(image: Image.Image, prompt: str) -> list[dict]:
    """Build Qwen3VL message format.

    Args:
        image: PIL Image
        prompt: Description prompt

    Returns:
        List of message dicts in Qwen3VL format
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def load_model(verbose: bool = True) -> tuple:
    """Load Qwen3VL model and processor with explicit MPS device assignment.

    Args:
        verbose: Print memory info

    Returns:
        Tuple of (model, processor)
    """
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    device = "mps"
    dtype = torch.bfloat16

    if verbose:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"  [MEM] Before loading: {get_memory_gb():.2f} GB")
        print(f"Loading {MODEL_NAME}...")

    # Load processor first
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if verbose:
        print(f"  [MEM] After processor: {get_memory_gb():.2f} GB")

    # Load model to CPU first, then explicitly move to MPS
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="cpu",  # Load to CPU first
        trust_remote_code=True,
    )

    if verbose:
        print(f"  [MEM] After model load (CPU): {get_memory_gb():.2f} GB")

    # Explicitly move to MPS
    model = model.to(device)

    if verbose:
        print(f"  [MEM] After model.to(mps): {get_memory_gb():.2f} GB")

    if verbose:
        print("Model loaded!")

    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Describe images using Qwen3-VL-8B-Instruct (MPS only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Examples:
  %(prog)s photo.jpg
  %(prog)s portrait.jpg -p "Describe the person's expression"
  %(prog)s https://example.com/image.jpg -q
        """
    )

    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument("-p", "--prompt", default="Describe this image.", help="Description prompt (default: 'Describe this image.')")
    parser.add_argument("-t", "--max-tokens", type=int, default=512, help="Max output tokens (default: 512)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    parser.add_argument("--help", action="help", help="Show this help message and exit")

    args = parser.parse_args()

    # Load and validate image
    try:
        image = validate_and_load_image(args.image)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Image: {args.image}")
        print(f"Prompt: {args.prompt}")
        print("Loading model...")

    # Load model
    model, processor = load_model(verbose=not args.quiet)

    # Build messages
    messages = build_messages(image, args.prompt)

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to("mps")

    if not args.quiet:
        print(f"  [MEM] Before generation: {get_memory_gb():.2f} GB")
        print("Generating description...")

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)

    if not args.quiet:
        print(f"  [MEM] After generation: {get_memory_gb():.2f} GB")

    # Decode output
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Print result
    print(output_text)

    cleanup_memory()


if __name__ == "__main__":
    main()
