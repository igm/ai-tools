---
name: ai-image-tools
description: Generate and edit images using FLUX.2-klein, Z-Image-Turbo, and Qwen-Image models on Apple Silicon. Use when user wants to create, modify, or generate images.
---

# AI Image Tools

This skill provides CLI tools for AI image generation and editing on Apple Silicon (MPS).

## Available Tools

- **`igm.flux-gen`** - Text-to-image using FLUX.2-klein-9B (4 steps, fast)
- **`igm.flux-edit`** - Image-to-image editing using FLUX.2-klein-9B
- **`igm.z-gen`** - Text-to-image using Z-Image-Turbo (9 steps)
- **`igm.qwen-gen`** - Text-to-image using Qwen-Image-2512 (supports turbo/lightning modes)

## When to Use

Use these tools when the user wants to:
- Generate an image from a text description
- Edit or transform an existing image
- Create variations of an image
- Apply specific styles to images

## Tool Selection

| Use case | Recommended tool |
|----------|------------------|
| Fast text-to-image | `igm.flux-gen` (4 steps) |
| Image editing/transforms | `igm.flux-edit` |
| Alternative text-to-image | `igm.z-gen` |
| High quality / aspect ratios | `igm.qwen-gen` |
| Very fast generation | `igm.qwen-gen --turbo` or `--lightning` |

## Usage Patterns

### Basic Generation

```bash
igm.flux-gen "a serene mountain landscape at sunset"
igm.flux-gen "portrait of a warrior" -w 768 -h 1024
```

### Image Editing

```bash
igm.flux-edit "turn into oil painting" -i photo.jpg
igm.flux-edit "anime style" -i portrait.jpg -o anime.png
```

### Fast Generation (Qwen)

```bash
igm.qwen-gen "a cat sitting on a fence" --turbo
igm.qwen-gen "cyberpunk city" --lightning
```

### Custom Resolution

```bash
igm.flux-gen "landscape" -w 1920 -h 1080
igm.qwen-gen "portrait" -w 1472 -h 1104  # 4:3 aspect ratio
```

### Reproducible Results

```bash
igm.flux-gen "a cat" --seed 42
```

## Common Options

All tools support:
- `-o, --output` - Output filename
- `-w, --width` - Image width
- `-h, --height` - Image height
- `-s, --steps` - Inference steps
- `--seed` - Random seed
- `-q, --quiet` - Less verbose output
- `--help` - Show help

## Notes

- All tools require Apple Silicon (MPS)
- Images are saved with metadata stripped
- Output files auto-increment (_1, _2) if file exists
- Memory optimizations enabled by default
