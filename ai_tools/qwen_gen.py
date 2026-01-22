"""Qwen-Image-2512 text-to-image generation CLI for Apple Silicon (MPS)."""

import argparse
import math
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
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

# Native aspect ratios for Qwen-Image-2512
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

# Default negative prompt (Chinese, as recommended for Qwen)
DEFAULT_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

# LoRA configurations
LORA_CONFIGS = {
    "turbo": {
        "path": "Wuli-Art/Qwen-Image-2512-Turbo-LoRA",
        "weight_name": "Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V1.0-bf16.safetensors",
        "adapter_name": "turbo",
        "steps": 4,
        "cfg": 1.0,
        "shift": 3.0,
        "name": "Turbo",
    },
    "lightning": {
        "path": "lightx2v/Qwen-Image-2512-Lightning",
        "weight_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        "adapter_name": "lightning",
        "steps": 4,
        "cfg": 1.0,
        "shift": 3.0,
        "name": "Lightning",
    },
}


def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    """Find closest native resolution maintaining aspect ratio.

    Args:
        width: Desired width
        height: Desired height

    Returns:
        Closest native resolution (width, height)
    """
    target_aspect = width / height

    best_match = None
    best_diff = float("inf")

    for name, (w, h) in ASPECT_RATIOS.items():
        aspect = w / h
        diff = abs(aspect - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_match = (w, h)

    return best_match if best_match else (1328, 1328)


def load_pipeline(lora_mode: str | None = None, verbose: bool = True) -> DiffusionPipeline:
    """Load the Qwen-Image-2512 pipeline.

    Args:
        lora_mode: LoRA mode - None, "turbo", or "lightning"
        verbose: Print memory info
    """
    device = "mps"
    dtype = torch.bfloat16

    if lora_mode:
        config = LORA_CONFIGS[lora_mode]
        mode_str = config["name"]
    else:
        mode_str = "Standard"

    if verbose:
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"  [MEM] Before loading: {get_memory_gb():.2f} GB")
        print(f"Loading Qwen-Image-2512 ({mode_str} mode)...")

    if lora_mode:
        # Load transformer separately for LoRA
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image-2512",
            subfolder="transformer",
            torch_dtype=dtype,
        )

        if verbose:
            print(f"  [MEM] After transformer: {get_memory_gb():.2f} GB")

        # Configure scheduler for LoRA (shift=3 for distillation)
        shift = config["shift"]
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(shift),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(shift),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-2512",
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=dtype,
        )

        if verbose:
            print(f"Loading {mode_str} LoRA from {config['path']}...")

        lora_kwargs = {
            "pretrained_model_name_or_path_or_dict": config["path"],
            "adapter_name": config["adapter_name"],
        }
        if config["weight_name"]:
            lora_kwargs["weight_name"] = config["weight_name"]

        # Load LoRA and handle the adapter setting
        pipe.load_lora_weights(**lora_kwargs)

        # Check if adapter was actually loaded before trying to set it
        active_adapters = pipe.get_active_adapters()
        if active_adapters and config["adapter_name"] in active_adapters:
            pipe.set_adapters([config["adapter_name"]])
        elif active_adapters:
            # Adapter was loaded with a different name, use the first available
            if verbose:
                print(f"  Using adapter: {active_adapters[0]}")
            pipe.set_adapters([active_adapters[0]])

        if verbose:
            print(f"  [MEM] After LoRA: {get_memory_gb():.2f} GB")
    else:
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-2512",
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
        description="Generate images from text using Qwen-Image-2512 (MPS only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Examples:
  %(prog)s "a cat in a hat"
  %(prog)s "a sunset over mountains" -o sunset.png
  %(prog)s "portrait" -w 1472 -h 1104
  %(prog)s "landscape" -w 1664 -h 928
  %(prog)s "fast" --turbo
  %(prog)s "fastest" --lightning
        """
    )

    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("-o", "--output", default="qwen-output.png", help="Output filename")
    parser.add_argument("-w", "--width", type=int, default=1328, help="Image width (default: 1328)")
    parser.add_argument("-h", "--height", type=int, default=1328, help="Image height (default: 1328)")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    parser.add_argument("-s", "--steps", type=int, default=50, help="Inference steps (default: 50, LoRA modes: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-g", "--cfg-scale", type=float, default=4.0, help="CFG scale (default: 4.0, LoRA modes: 1.0)")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--turbo", action="store_true", help="Enable turbo LoRA (4 steps, CFG=1.0)")
    parser.add_argument("--lightning", action="store_true", help="Enable lightning LoRA (4 steps, CFG=1.0)")
    parser.add_argument("--no-memory-opts", action="store_true", help="Disable memory optimizations")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Determine LoRA mode
    lora_mode = None
    if args.turbo and args.lightning:
        print("Error: Cannot use both --turbo and --lightning at the same time")
        return
    elif args.turbo:
        lora_mode = "turbo"
    elif args.lightning:
        lora_mode = "lightning"

    # Adjust defaults for LoRA modes
    if lora_mode:
        config = LORA_CONFIGS[lora_mode]
        if args.steps == 50:  # User didn't override
            args.steps = config["steps"]
        if args.cfg_scale == 4.0:  # User didn't override
            args.cfg_scale = config["cfg"]

    generator = get_generator(args.seed)
    seed = generator.initial_seed()

    # Find closest native resolution
    width, height = find_closest_resolution(args.width, args.height)

    if not args.quiet:
        print(f"Seed: {seed}")
        print(f"Prompt: {args.prompt}")
        mode_str = LORA_CONFIGS[lora_mode]["name"] if lora_mode else "Standard"
        print(f"Mode: {mode_str} ({args.steps} steps, CFG={args.cfg_scale})")
        if (width, height) != (args.width, args.height):
            print(f"Resolution adjusted: {args.width}x{args.height} -> {width}x{height} (native)")
        else:
            print(f"Resolution: {width}x{height}")
        print("Loading model...")

    pipe = load_pipeline(lora_mode=lora_mode, verbose=not args.quiet)

    if args.no_memory_opts:
        disable_memory_opts(pipe)

    if not args.quiet:
        print("Generating...")

    generate_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.cfg_scale,
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
