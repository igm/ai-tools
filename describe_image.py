"""
Image-to-Text Description using Qwen3-VL
Generates detailed image descriptions locally.
"""

import os
import argparse

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def describe_images(
    image_paths: list[str],
    prompt: str = "Describe this image in detail.",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "mps",
    max_new_tokens: int = 512,
):
    """Generate description for image(s)."""
    # Determine device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading {model_id} on {device}...")

    # Load model with explicit dtype for MPS
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Build messages
    content = [{"type": "image", "image": path} for path in image_paths]
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Prepare for inference (Qwen3 handles vision processing internally)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    print(f"\nProcessing {len(image_paths)} image(s)...")
    print(f"Prompt: {prompt}\n")

    # Generation
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions with Qwen3-VL")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Input image path(s)")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Custom prompt (default: 'Describe this image in detail.')")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model ID on Hugging Face")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens in output (default: 512)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps, cuda, cpu)")

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
        device=args.device,
        max_new_tokens=args.max_tokens,
    )

    print("\n" + "=" * 60)
    print("DESCRIPTION:")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
