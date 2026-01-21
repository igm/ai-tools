# FLUX.2 Constraints

## Guidance

- **MLX:** `guidance` must be `1.0` (no CFG support)
- **PyTorch:** `guidance_scale` parameter supported

## Resolution

- Dimensions must be multiples of 64
- Use `calculate_dimensions()` helper to maintain aspect ratio when resizing
