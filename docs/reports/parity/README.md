# Parity and inference reports (M2)

The M2 family's reports (ForwardOpsAndQwenModel), rendered by `lrm report` from real run logs. The Markdown and SVG files are never edited by hand; this index is.

| Report | Run | What produced it |
| --- | --- | --- |
| [tiny/parity.md](tiny/parity.md) | `20261018-051350-1c9f3dd8` | M2a's tiny-model acceptance (`tests/Lrm.Models.Qwen.Tests`, cpu tier): the Qwen forward on CpuOps against O1's goldens, greedy decoding, and the bit-identity proofs. Node n0 is cpu-f64, n1 cpu-f32 |
| [tiny/inference.md](tiny/inference.md) | `20261018-051353-a6ef3348` | `lrm generate` on the tiny model: greedy decoding of p0 for 64 steps on cpu-f32 (n0) and cpu-f64 (n1), after 2 warm-up steps |

Run logs live under `LRM_DATA/runs` and are never committed. To make the same reports again from a Release build:

```bash
LRM_TEST_TIER=cpu dotnet artifacts/bin/Lrm.Models.Qwen.Tests/release/Lrm.Models.Qwen.Tests.dll   # prints "parity run log: <dir>"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <dir> --out docs/reports/parity/tiny

dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll generate --model tiny --backend cpu-f32,cpu-f64 \
  --ids-from goldens/tiny/g1/fwd_f32.p0.safetensors --steps 64 --warmup 2 \
  --expect goldens/tiny/g1/greedy.safetensors:p0.ids.torch32 --run-log \
  --title "Tiny model: greedy decoding of p0, 64 steps, on cpu-f32 and cpu-f64 (M2a)"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <runId> --out docs/reports/parity/tiny
```

The parity numbers are deterministic. The timings in the inference report are host times on the shared development machine (32 threads, 8 used), so they are informative only. The memory chart of the inference report is empty on purpose: `mem.arena` belongs to the GPU heap and executor (TelemetryFacade), and CpuOps has no heap.

M2b adds the full-model parity (`qwen3-0.6b/`) for gate1-cpu.
