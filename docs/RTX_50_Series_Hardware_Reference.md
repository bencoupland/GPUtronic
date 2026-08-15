# GPUtronic Stage 13 — Blackwell PoC Test Suite v2

## Hardware Reference Table

| Model | Die | SMs | Threads/SM | Max Threads | CUDA Cores | Memory Interface |
|-------|-----|-----|------------|-------------|------------|------------------|
| RTX 5090 | GB202 | 170 | 1536 | 261,120 | 21,760+ | GDDR7 (512-bit) |
| RTX 5080 | GB203 | 84 | 1536 | 128,960 | 10,752 | GDDR7 (256-bit) |
| RTX 5070 Ti | GB203 | 70 | 1536 | 107,520 | 8,960 | GDDR7 (256-bit) |
| RTX 5070 | GB205 | 48 | 1536 | 73,728 | 6,144 | GDDR7 (192-bit) |
| RTX 5060 Ti | GB206 | 36 | 1536 | 55,296 | 4,608 | GDDR7 (128-bit) |
| RTX 5060 | GB206 | 30 | 1536 | 46,080 | 3,840 | GDDR7 (128-bit) |

## Key Architectural Notes

- **Thread-per-SM limit**: Constant at 1536 across all Blackwell cards
- **SM count varies by model**: Different dies have different SM counts
- **Memory interface**: Varies between models (GDDR7, bus width (512-bit vs 256-bit))
- **CUDA cores per SM**: Consistent at 128 CUDA cores per SM across all models

## References

- NVIDIA RTX Blackwell GPU Architecture Documentation
- TechPowerUp GPU Database
- Wikipedia GeForce RTX 50 series article
