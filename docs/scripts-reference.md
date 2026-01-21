# Scripts Reference

AI image generation and vision tools with dual backend implementations (MLX and PyTorch).

## Core Scripts

| Script | Purpose | Backend |
|--------|---------|---------|
| `mlx_generate_flux2.py` | Text-to-image | MLX |
| `mlx_edit_flux2.py` | Image-to-image editing | MLX |
| `mlx_describe_image.py` | Image captioning | MLX |
| `generate_flux2.py` | Text-to-image | PyTorch |
| `edit_flux2.py` | Image-to-image editing | PyTorch |
| `describe_image.py` | Image captioning | PyTorch |

**Performance:** MLX versions are 2-3.5x faster on Apple Silicon. Prefer MLX for this platform.

## Usage

Run any script with `--help` for full command-line options.

### Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install mflux --pre
```
