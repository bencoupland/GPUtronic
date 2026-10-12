# Seventh pass: read this when you resume

You started the seventh pass before commit `6a7171f` landed. That commit added the rules below to `docs/directives/seventh-pass-2026-10-13.md`. Apply them to everything you run from now on. Note in the report which holds, if any, ran before you read this.

1. Keep two SMs free for the operating system at all times. The 5080 also drives the desktop, so those two SMs keep the display refreshing and the machine responsive. Put no test work on them.
2. The ceiling is 82 SMs. Launch only 82 persistent blocks, so the two reserved SMs carry no test block at all. Use 82 wherever the directive says 84, including the reference rate Z is measured against.
3. The floor is two active SMs. Never command fewer.
4. Do not try to measure 84 against 82 or 80 SMs. With the desktop load in the background, that difference won't give useful data.
5. Testing runs on the host against the real pinned, mapped control page, GPUtronic's own `struct GPUTronicControl`. There is no emulation layer.
6. Log the smid of each launched block, so we know which two SMs CUDA left free. Run plain `nvidia-smi` at the start and the end, and record whether the display is attached and which processes are on the GPU.

Everything else in the seventh-pass directive is unchanged.
