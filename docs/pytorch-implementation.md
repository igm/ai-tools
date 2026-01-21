# PyTorch Implementation

## MPS Optimization

```python
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
```

## Memory Management

Scripts use `torch.mps.empty_cache()` and `torch.mps.synchronize()`.

## Low-RAM Systems

Use SDNQ quantized models (e.g., `Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32`).
