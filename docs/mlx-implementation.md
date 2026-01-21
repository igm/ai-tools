# MLX Implementation

## Memory Management

Scripts use `gc.collect()` after generation.

## Low-RAM Systems

Use `--quantize 4` for reduced memory usage.

## Model Variants

- `flux2-klein-4b`: Smaller, faster (~4B parameters)
- `flux2-klein-9b`: Higher quality (~9B parameters)

## Quantization Levels

| Bits | Memory | Quality |
|------|--------|---------|
| 3-4  | Low    | Some loss |
| 5-6  | Medium | Balanced |
| 8    | Good   | Good quality |
| None | High   | Full precision, slowest |
