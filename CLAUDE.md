# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flux2 CLI Tools - opinionated command-line tools for fast AI image generation on Apple Silicon (MPS). Supports FLUX.2-klein-9B, Z-Image-Turbo, and Qwen-Image-2512 models.

**MPS-only.** All tools require Apple Silicon GPU acceleration.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tools
uv run flux_generate.py "prompt"
uv run flux_edit.py "prompt" -i input.jpg
uv run z_generate.py "prompt"
uv run qwen_generate.py "prompt" [--turbo|--lightning]
```

## Architecture

**4 tool scripts** - each loads its specific model and provides a CLI interface:
- `flux_generate.py` - Flux2 text-to-image (Flux2KleinPipeline)
- `flux_edit.py` - Flux2 image-to-image (Flux2Img2ImgPipeline)
- `z_generate.py` - Z-Image-Turbo text-to-image (Flux2KleinPipeline + LoRA)
- `qwen_generate.py` - Qwen-Image-2512 text-to-image (DiffusionPipeline, supports turbo/lightning LoRA modes)

**Shared utilities** (`mps_common.py`):
- `get_memory_gb()` - Track MPS VRAM usage
- `cleanup_memory()` - Force MPS memory cleanup
- `enable_memory_opts()` / `disable_memory_opts()` - VAE slicing, VAE tiling, attention slicing
- `save_image_clean()` - Save PNG with metadata stripped, auto-increment `_1`, `_2` suffixes
- `get_generator()` - MPS torch.Generator with seed

## Code Patterns

All tools follow same structure:
1. Enable `PYTORCH_MPS_FAST_MATH=1`
2. Load pipeline with `torch_dtype=torch.bfloat16`
3. Move to `"mps"` device
4. Enable memory opts (unless `--no-memory-opts`)
5. Generate with `torch.inference_mode()`
6. Clean output image (strip metadata), save with auto-increment
7. Cleanup MPS memory

Default steps: Flux=4, Z-Image-Turbo=9, Qwen=50 (4 with LoRA).

## Qwen-Image-2512 Specifics

Native resolutions only - `find_closest_resolution()` auto-adjusts to nearest supported aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3).

LoRA modes (Turbo/Lightning) require custom scheduler with `shift=3.0` for distillation. See `LORA_CONFIGS` dict.

Uses `true_cfg_scale` parameter (not `guidance_scale`).
