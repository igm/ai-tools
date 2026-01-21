# CLAUDE.md

AI image generation and vision tools for Apple Silicon. Dual backend implementations (MLX/PyTorch) for text-to-image, image editing, and captioning.

## Environment

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

- **Text-to-image:** `python mlx_generate_flux2.py --help`
- **Image editing:** `python mlx_edit_flux2.py --help`
- **Captioning:** `python mlx_describe_image.py --help`

Prefer `mlx_*.py` scripts on Apple Silicon (2-3.5x faster).

## Docs

- [Scripts Reference](docs/scripts-reference.md)
- [Architecture](docs/architecture.md)
- [MLX Implementation](docs/mlx-implementation.md)
- [PyTorch Implementation](docs/pytorch-implementation.md)
- [FLUX.2 Constraints](docs/flux2-constraints.md)
