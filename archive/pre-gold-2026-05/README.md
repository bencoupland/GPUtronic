# Pre-Gold archive (frozen May 2026 experiments)

Everything here predates **GPUTronic Gold 1.0.0**.

Do **not** build or run these as the project reference. Kept for history only:
version ladder v14→v26, Cyberpunk campaigns, launcher logs, PoC notes.

## Active project

```bash
cd "../.."   # repo root
make gold && ./build/gputronic_gold check
```

## Layout

| Dir | Contents |
|-----|----------|
| sources/ | `.cu` / `.cpp` / headers from experimental ladder |
| binaries/ | Prebuilt governors, launchers, harnesses |
| logs/ | Run logs (may be `.xz` compressed) |
| docs/ | Old READMEs, test plans, stage reports |
| scripts/ | Cyberpunk launch / sweep / capture scripts |
| results/ | Dated cyberpunk_* and early analysis folders |
| misc/ | One-offs |

Large logs may be xz-compressed (`*.log.xz`) to save disk. Decompress with `xz -d file.log.xz`.
