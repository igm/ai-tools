# Flux2 CLI Tools

Opinionated tools for fast image generation and editing with FLUX.2-klein-9B, Z-Image-Turbo, and Qwen-Image-2512 on Apple Silicon (MPS).

**Focused tools for specific models.** Each tool is designed for simplicity and speed with its specific model.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS with MPS support
- `uv` package manager

## Installation

```bash
# Install dependencies
uv sync

# Or install as global tools (optional)
uv tool install .
```

## Tools

### `flux_generate.py` - Text to Image

Generate images from text prompts.

```bash
uv run flux_generate.py "a cat in a hat"
uv run flux_generate.py "a sunset" -o sunset.png
uv run flux_generate.py "portrait" -w 768 -h 1024 -s 8 --seed 42
```

**Arguments:**
- `prompt` - Text description (required)
- `-o, --output` - Output filename (default: `flux-output.png`)
- `-w, --width` - Image width (default: 1024)
- `-h, --height` - Image height (default: 1024)
- `-s, --steps` - Inference steps (default: 4, Klein is fast!)
- `--seed` - Random seed for reproducibility
- `-g, --guidance` - Guidance scale (default: 1.0)
- `--no-memory-opts` - Disable memory optimizations
- `-q, --quiet` - Less verbose output
- `--help` - Show help

### `z_generate.py` - Text to Image (Z-Image-Turbo)

Generate images using Z-Image-Turbo model.

```bash
uv run z_generate.py "a cat in a hat"
uv run z_generate.py "a sunset" -o sunset.png
uv run z_generate.py "portrait" -w 1024 -h 1024 -s 9 --seed 42
```

**Arguments:**
- `prompt` - Text description (required)
- `-o, --output` - Output filename (default: `z-output.png`)
- `-w, --width` - Image width (default: 1024)
- `-h, --height` - Image height (default: 1024)
- `-s, --steps` - Inference steps (default: 9)
- `--seed` - Random seed
- `-g, --guidance` - Guidance scale (default: 0.0, Turbo requires 0)
- `--no-memory-opts` - Disable memory optimizations
- `-q, --quiet` - Less verbose output

### `qwen_generate.py` - Text to Image (Qwen-Image-2512)

Generate images using Qwen-Image-2512 with native aspect ratio support. Includes turbo and lightning LoRA modes.

```bash
# Standard mode (50 steps)
uv run qwen_generate.py "a cat in a hat"

# Turbo mode (4 steps, ~12x faster)
uv run qwen_generate.py "a cat" --turbo

# Lightning mode (4 steps, ~12x faster)
uv run qwen_generate.py "a cat" --lightning

# Custom output with turbo
uv run qwen_generate.py "sunset" --turbo -o sunset.png

# 16:9 aspect ratio (auto-adjusts to native 1664x928)
uv run qwen_generate.py "landscape" -w 1664 -h 928
```

**Arguments:**
- `prompt` - Text description (required)
- `-o, --output` - Output filename (default: `qwen-output.png`)
- `-w, --width` - Image width (default: 1328, auto-adjusts to native resolutions)
- `-h, --height` - Image height (default: 1328, auto-adjusts to native resolutions)
- `-s, --steps` - Inference steps (default: 50, LoRA modes: 4)
- `--seed` - Random seed
- `-g, --cfg-scale` - CFG scale (default: 4.0, LoRA modes: 1.0)
- `--negative-prompt` - Negative prompt (default: Chinese quality prompt)
- `--turbo` - Enable turbo LoRA (4 steps, CFG=1.0)
- `--lightning` - Enable lightning LoRA (4 steps, CFG=1.0)
- `--no-memory-opts` - Disable memory optimizations
- `-q, --quiet` - Less verbose output

**LoRA Modes:**
| Mode | Steps | CFG | LoRA Path |
|------|-------|-----|-----------|
| Standard | 50 | 4.0 | (none) |
| Turbo | 4 | 1.0 | Wuli-Art/Qwen-Image-2512-Turbo-LoRA |
| Lightning | 4 | 1.0 | lightx2v/Qwen-Image-2512-Lightning |

**Native Aspect Ratios:**
| Ratio | Resolution |
|-------|------------|
| 1:1 | 1328x1328 |
| 16:9 | 1664x928 |
| 9:16 | 928x1664 |
| 4:3 | 1472x1104 |
| 3:4 | 1104x1472 |
| 3:2 | 1584x1056 |
| 2:3 | 1056x1584 |

### `flux_edit.py` - Image to Image

Edit existing images with text prompts.

```bash
uv run flux_edit.py "turn into oil painting" -i photo.jpg
uv run flux_edit.py "anime style" -i portrait.jpg -o anime.png
uv run flux_edit.py "cyberpunk" -i input.jpg -r 1024 -s 8
```

**Arguments:**
- `prompt` - Edit description (required)
- `-i, --input-images` - Input image path (required)
- `-o, --output` - Output filename (default: `flux-edit.png`)
- `-r, --resolution` - Max dimension (default: match input)
- `-s, --steps` - Inference steps (default: 4)
- `--seed` - Random seed
- `-g, --guidance` - Guidance scale (default: 1.0)
- `--no-memory-opts` - Disable memory optimizations
- `-q, --quiet` - Less verbose output
- `--help` - Show help

## Features

- **MPS Only** - Optimized for Apple Silicon GPU acceleration
- **Auto-increment** - Adds `_1`, `_2` suffix if output exists
- **Clean Output** - Strips all metadata from generated images
- **Memory Tracking** - Shows VRAM usage with `[MEM]` tags
- **Fast** - 4 steps for Flux, 9 steps for Z-Image-Turbo, 50 steps for Qwen

## Memory Optimization

By default, all tools enable:
- Attention slicing
- VAE slicing
- VAE tiling (if available)

Disable with `--no-memory-opts` for faster generation (uses more VRAM).

## Output

Images are saved as PNG with:
- No EXIF metadata
- No software tags
- Clean pixel data only
