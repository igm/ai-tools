"""Shared utilities for Flux CLI tools."""

import gc
from pathlib import Path

import torch
from PIL import Image


def get_memory_gb() -> float:
    """Get current MPS memory usage in GB."""
    return torch.mps.current_allocated_memory() / 1024**3


def cleanup_memory():
    """Force MPS memory cleanup."""
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()


def enable_memory_opts(pipe, verbose: bool = True) -> None:
    """Enable memory optimizations on pipeline."""
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    if verbose:
        print(f"  [MEM] After optimizations: {get_memory_gb():.2f} GB")


def disable_memory_opts(pipe) -> None:
    """Disable memory optimizations on pipeline."""
    pipe.disable_attention_slicing()
    if hasattr(pipe, "disable_vae_slicing"):
        pipe.disable_vae_slicing()


def save_image_clean(img: Image.Image, output_path: Path) -> Path:
    """Save image with metadata stripped, auto-increment if exists.

    Args:
        img: PIL Image to save
        output_path: Desired output path

    Returns:
        Actual path where image was saved (may have _N suffix)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_path = output_path
    counter = 1
    while final_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        final_path = output_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1


    # Strip metadata by rebuilding from raw pixel bytes (fresh image has no info/exif)
    clean_img = Image.frombytes(img.mode, img.size, img.tobytes())
    if img.mode == "P" and img.getpalette():
        clean_img.putpalette(img.getpalette())
    clean_img.save(final_path, optimize=False)
    return final_path


def get_generator(seed: int | None) -> torch.Generator:
    """Get MPS generator with seed.

    Args:
        seed: Random seed, or None for random seed

    Returns:
        Torch Generator for MPS
    """
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    return torch.Generator("mps").manual_seed(seed)
