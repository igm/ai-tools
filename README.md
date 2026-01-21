# AI Tools

AI image generation and vision tools optimized for Apple Silicon. Features FLUX.2-klein for text-to-image generation, image-to-image editing, and Qwen3-VL for image description.

> **Note:** MLX versions are ~2-3.5x faster on Apple Silicon. Use the `mlx_*.py` scripts for best performance.

## Installation

### 1. Create virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

## Scripts Overview

| Script | Purpose | Backend | Speed |
|--------|---------|---------|-------|
| `mlx_generate_flux2.py` | Text-to-image | MLX | ⚡ Fast |
| `mlx_edit_flux2.py` | Image editing | MLX | ⚡ Fast |
| `mlx_describe_image.py` | Image description | MLX | ⚡ Fast |
| `generate_flux2.py` | Text-to-image | PyTorch | Slow |
| `edit_flux2.py` | Image editing | PyTorch | Slow |
| `describe_image.py` | Image description | PyTorch | Slow |

## Text-to-Image Generation

### MLX Version (Recommended)

```bash
python mlx_generate_flux2.py \
  "a photorealistic cat sitting on a windowsill" \
  --model flux2-klein-4b \
  --steps 4 \
  --quantize 8 \
  --output cat.png
```

**Parameters:**
- `--model`: `flux2-klein-4b` (faster) or `flux2-klein-9b` (better quality)
- `--steps`: Inference steps (4-10, default: 4)
- `--quantize`: Quantization bits (3, 4, 5, 6, 8; default: 8)
- `--no-quantize`: Use full precision (slower, best quality)
- `--width` / `--height`: Output dimensions (default: 1024x1024)
- `--seed`: Random seed for reproducibility

### PyTorch Version

```bash
python generate_flux2.py \
  "a photorealistic cat sitting on a windowsill" \
  --model sdnq \
  --steps 4 \
  --output cat.png
```

**Parameters:**
- `--model`: `sdnq` (4-bit quantized) or `full` (full precision)
- `--steps`: Inference steps (default: 4)
- `--guidance`: Guidance scale (default: 3.5)
- `--width` / `--height`: Output dimensions (default: 1024x1024)

## Image-to-Image Editing

### MLX Version (Recommended)

Supports both **single-image editing** and **multi-image compositing**.

```bash
# Single-image editing
python mlx_edit_flux2.py \
  "transform into a watercolor painting" \
  --input photo.jpg \
  --steps 4 \
  --output edited.png

# Multi-image compositing (e.g., add glasses to a person)
python mlx_edit_flux2.py \
  "Make the woman wear the eyeglasses" \
  --input person.jpg glasses.jpg \
  --steps 4 \
  --seed 42 \
  --output edited.png
```

**Parameters:**
- `--input`: Input image path(s) - accepts 1+ images for single/multi-image editing
- `--resolution`: Target longest side (default: 1024)
- `--steps`: Inference steps (4-10, default: 4)
- `--quantize`: Quantization bits (3, 4, 5, 6, 8; default: 8)

### PyTorch Version

```bash
python edit_flux2.py \
  "transform into a watercolor painting" \
  --input photo.jpg \
  --steps 4 \
  --output edited.png
```

**Parameters:**
- `--input`: Input image path(s) - supports up to 6 images for combining
- `--resolution`: Target longest side (default: 1024)
- `--guidance`: Guidance scale (default: 3.5)
- `--steps`: Inference steps (default: 4)

## Image Description

### MLX Version (Recommended)

```bash
python mlx_describe_image.py \
  --input photo.jpg \
  --prompt "describe this image in detail" \
  --max-tokens 512
```

**Parameters:**
- `--input`: Image path(s) - multiple images supported
- `--prompt`: Custom prompt (default: "Describe this image in detail.")
- `--max-tokens`: Maximum output tokens (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)
- `--model`: Hugging Face model ID (default: mlx-community/Qwen3-VL-8B-Instruct-4bit)

### PyTorch Version

```bash
python describe_image.py \
  --input photo.jpg \
  --prompt "describe this image in detail" \
  --max-tokens 512
```

**Parameters:**
- `--input`: Image path(s) - multiple images supported
- `--prompt`: Custom prompt (default: "Describe this image in detail.")
- `--max-tokens`: Maximum output tokens (default: 512)
- `--device`: Device (mps, cuda, cpu; default: mps)

## Quick Examples

### Generate a portrait

```bash
python mlx_generate_flux2.py \
  "Photorealistic close-up of a hummingbird hovering near red flowers, frozen wings, detailed feathers, soft green background bokeh, high shutter speed look" \
  --width 1024 --height 560 --steps 4 --seed 42
```

### Edit a photo

```bash
python mlx_edit_flux2.py \
  "Add dramatic sunset lighting with orange and purple sky" \
  --input landscape.jpg \
  --steps 6
```

### Describe an image

```bash
python mlx_describe_image.py \
  --input screenshot.png \
  --prompt "What's shown in this screenshot? List all UI elements."
```

## Model Cache

Models are cached in:
- `~/.cache/huggingface/hub/` - Hugging Face models
- `~/Library/Caches/mflux/` - MLX-specific cache

To check cached models:
```bash
ls ~/.cache/huggingface/hub/ | grep -E "(FLUX|Qwen)"
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- ~16GB RAM recommended for 9B models
- ~8GB RAM sufficient for 4B models with quantization

## License

This repository uses third-party models. Check respective model licenses for usage terms.
