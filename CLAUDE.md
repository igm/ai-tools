# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI CLI Tools - opinionated command-line tools for fast AI image generation on Apple Silicon (MPS). Supports FLUX.2-klein-9B, Z-Image-Turbo, and Qwen-Image-2512 models.

**MPS-only.** All tools require Apple Silicon GPU acceleration.

Package name: `ai-tools` (installed as `flux-gen`, `flux-edit`, `z-gen`, `qwen-gen`, `qwen-desc`)

## Development Commands

```bash
# Install in development mode
uv pip install -e .

# Run tools
flux-gen "prompt"
flux-edit "prompt" -i input.jpg
z-gen "prompt"
qwen-gen "prompt" [--turbo|--lightning]
qwen-desc image.jpg
```

## Architecture

**5 tool modules** in `ai_tools/` package - each loads its specific model and provides a CLI interface:
- `flux_gen.py` - Flux2 text-to-image (Flux2KleinPipeline)
- `flux_edit.py` - Flux2 image-to-image (Flux2Img2ImgPipeline)
- `z_gen.py` - Z-Image-Turbo text-to-image (ZImagePipeline)
- `qwen_gen.py` - Qwen-Image-2512 text-to-image (DiffusionPipeline, supports turbo/lightning LoRA modes)
- `qwen3vl_desc.py` - Qwen3-VL-8B-Instruct image-to-text description (vision language model)

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

## Qwen3-VL-8B-Instruct Specifics

Image-to-text description tool. Input: image path/URL, output: text description to stdout.

Forces explicit MPS device assignment (not auto). Uses `dtype` param (not deprecated `torch_dtype`).
