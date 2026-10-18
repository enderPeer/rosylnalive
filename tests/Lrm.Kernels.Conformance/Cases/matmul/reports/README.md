# matmul_bf16w reports (K2)

Rendered from real runs: the conformance runs at commit `e1f776a`, the bench at `3f84e15` (the kernel is the same in both). The device is an NVIDIA GeForce RTX 4080 with driver 616.92 (UMD 32.0.16.1692). The DXIL ShaderSetHash is `2d513d81…`. The design and the numbers are in [ADR K2-0001](../../../../../docs/decisions/K2-0001-matmul-bf16w.md).

| Report | Run | What it shows |
| --- | --- | --- |
| [d3d12-conformance/kernels.md](d3d12-conformance/kernels.md) | `20261018-140156-4152f402` | The gpu tier's conformance run on D3D12: every case, the matrix (matmul_bf16w 45/45), the denormal mode, the timeline and the slowest dispatches. Each case runs after seconds of host work, so these are cold-GPU times |
| [d3d12-bench/matmul.md](d3d12-bench/matmul.md) | `20261018-140617-1ce83fea` | MatmulBench: warm throughput, traffic and dispatch time by m for every Qwen3-0.6B shape; the cold dispatches (lm_head at every m, the projections at m = 4,000) against the budget and X1's 64-row lm_head cap; the cross-call batch-invariance and CpuRef checks; and the arithmetic probes |
| [d3d12-bench/kernels.md](d3d12-bench/kernels.md) | `20261018-140617-1ce83fea` | The same bench run through T1's kernels template: the 1,353-dispatch timeline and the slowest dispatches. Its "4 dispatches exceed the 100 ms budget" are the cold lm_head dispatches at m = 512 and 1,024, past X1's cap, measured on purpose |
| [cpu-device/kernels.md](cpu-device/kernels.md) | `20261018-140141-3dc48e1e` | The cpu tier's conformance run on CpuDevice (CI's cases), bit-exact to CpuRef F32; its times are emulation, not GPU |

To reproduce, build the solution first with `dotnet build Lrm.slnx -c Release`. Then:

```bash
pwsh tools/test.ps1 -Tier gpu -NoBuild -Project Lrm.Kernels.Conformance   # prints the conformance run log
dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/bench.fsx bench     # prints the bench run log (about 85 s, 78 s of it idle before the cold dispatches)
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll report <run id> --out <dir>
dotnet fsi tests/Lrm.Kernels.Conformance/Cases/matmul/report.fsx <bench run id> <dir>
```

The bench is not in a test tier yet (ADR K2-0001, the request to I1), so `tools/test.ps1 -Tier bench` does not run it.
