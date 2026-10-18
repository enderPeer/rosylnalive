# matmul_bf16w reports (K2)

Rendered from real runs at commit `558815b`. The device is an NVIDIA GeForce RTX 4080 with driver 616.92 (UMD 32.0.16.1692). The DXIL ShaderSetHash is `2d513d81…`. The design and the numbers are in [ADR K2-0001](../../../../../docs/decisions/K2-0001-matmul-bf16w.md).

| Report | Run | What it shows |
| --- | --- | --- |
| [d3d12-conformance/kernels.md](d3d12-conformance/kernels.md) | `20261018-132406-e1c048e8` | The gpu tier's conformance run on D3D12: every case, the matrix (matmul_bf16w 41/41), the denormal mode, the timeline and the slowest dispatches. Each case runs after seconds of host work, so these are cold-GPU times |
| [d3d12-bench/matmul.md](d3d12-bench/matmul.md) | `20261018-132542-33830713` | MatmulBench: warm throughput, traffic and dispatch time by m for every Qwen3-0.6B shape, the cross-call batch-invariance and CpuRef checks, and the arithmetic probes |
| [d3d12-bench/kernels.md](d3d12-bench/kernels.md) | `20261018-132542-33830713` | The same bench run through T1's kernels template: the 438-dispatch timeline and the slowest dispatches |
| [cpu-device/kernels.md](cpu-device/kernels.md) | `20261018-131829-8cc803cc` | The cpu tier's conformance run on CpuDevice (CI's cases), bit-exact to CpuRef F32; its times are emulation, not GPU |

To reproduce, build the solution first with `dotnet build Lrm.slnx -c Release`. Then:

```bash
pwsh tools/test.ps1 -Tier gpu -NoBuild -Project Lrm.Kernels.Conformance   # prints the conformance run log
dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/bench.fsx bench     # prints the bench run log
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <run id> --out <dir>
dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/report.fsx <bench run id> <dir>
```
