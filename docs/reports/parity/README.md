# Parity and inference reports (M2)

The M2 family's reports (ForwardOpsAndQwenModel), rendered by `lrm report` from real run logs. The Markdown and SVG files are never edited by hand; this index is.

| Report | Run | What produced it |
| --- | --- | --- |
| [tiny/parity.md](tiny/parity.md) | `20261018-060450-cfb11171` | M2a's tiny-model acceptance (`tests/Lrm.Models.Qwen.Tests`, cpu tier): the Qwen forward on CpuOps against O1's goldens, greedy decoding, and the bit-identity proofs. Node n0 is cpu-f64, n1 cpu-f32 |
| [tiny/inference-cpu-f32/inference.md](tiny/inference-cpu-f32/inference.md) | `20261018-060459-1cc52a2e` | `lrm generate` on the tiny model: greedy decoding of p0 for 64 steps on cpu-f32, after 2 warm-up steps |
| [tiny/inference-cpu-f64/inference.md](tiny/inference-cpu-f64/inference.md) | `20261018-060459-d3304488` | The same command's run on cpu-f64 |

`lrm generate --run-log` writes one run per backend, so each inference report holds one ExecutionClass and its headline rates are that class's alone.

Run logs live under `LRM_DATA/runs` and are never committed. The test exe keeps its parity run there only when `LRM_WRITE_PARITY_RUN=1` is set; otherwise the run goes to a temporary folder that the exe deletes. To make the same reports again from a Release build:

```bash
LRM_WRITE_PARITY_RUN=1 LRM_TEST_TIER=cpu dotnet artifacts/bin/Lrm.Models.Qwen.Tests/release/Lrm.Models.Qwen.Tests.dll   # prints "parity run log: <dir>"
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <dir> --out docs/reports/parity/tiny

dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll generate --model tiny --backend cpu-f32,cpu-f64 \
  --ids-from goldens/tiny/g1/fwd_f32.p0.safetensors --steps 64 --warmup 2 \
  --expect goldens/tiny/g1/greedy.safetensors:p0.ids.torch32 --run-log \
  --title "Tiny model: greedy decoding of p0, 64 steps (M2a)"          # prints one "run log: <dir>" per backend
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <cpu-f32 runId> --out docs/reports/parity/tiny/inference-cpu-f32
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <cpu-f64 runId> --out docs/reports/parity/tiny/inference-cpu-f64
```

The parity numbers are deterministic. The timings in the inference reports are host times on the shared development machine (32 threads, 8 used), so they are informative only. The memory chart of the inference reports is empty on purpose: `mem.arena` belongs to the GPU heap and executor (TelemetryFacade), and CpuOps has no heap.

M2b adds the full-model parity (`qwen3-0.6b/`) for gate1-cpu.
