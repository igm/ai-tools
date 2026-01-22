# Flux2 CLI Tools

Fast image generation and editing with FLUX.2-klein-9B on Apple Silicon (MPS).

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
- **Fast** - 4 steps by default (Klein is distilled for speed)

## Memory Optimization

By default, both tools enable:
- Attention slicing
- VAE slicing
- VAE tiling (if available)

Disable with `--no-memory-opts` for faster generation (uses more VRAM).

## Output

Images are saved as PNG with:
- No EXIF metadata
- No software tags
- Clean pixel data only
