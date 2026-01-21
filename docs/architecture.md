# Architecture

## Dual Backend Pattern

Each tool has two implementations:

```
PyTorch              MLX
---------            ---
diffusers + torch  ↔  mflux
transformers + torch ↔ mlx-vlm
```

## Key Differences

- **Quantization:** MLX uses `--quantize` flag (3-8 bits); PyTorch uses model variants
- **Guidance:** MLX FLUX.2 has no `--guidance` parameter (not supported)
- **Multi-image:** MLX edit supports single images only; PyTorch supports up to 6

## Model Sources

- **Hugging Face cache:** `~/.cache/huggingface/hub/`
- **MLX models:** `mlx-community/*` and `lmstudio-community/*`
- **Base models:** `black-forest-labs/FLUX.2-klein-*`
