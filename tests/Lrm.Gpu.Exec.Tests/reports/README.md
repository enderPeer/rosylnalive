# X1a reports

Rendered with `lrm report` (ReportTemplates) from real runs on the development machine, class `cpu-device cpu, .NET 10.0.13, win-x64+avx2+fma` (the CpuDevice of E1: preserve, CpuRef's NaN, barrier checks on, 8 threads). The run logs stay under `LRM_DATA/runs`; nothing here is written by hand. Commit `58c7f26` for all three.

| Folder | Template | Run | How it was made |
| --- | --- | --- | --- |
| `acceptance/` | parity | `20261018-153024-802629be` | `LRM_WRITE_EXEC_RUN=1 LRM_TEST_TIER=cpu dotnet artifacts/bin/Lrm.Gpu.Exec.Tests/release/Lrm.Gpu.Exec.Tests.dll`: the acceptance tests' records, cpu-device against the torch32 goldens (pair gpu-vs-torch32, every tap of p0 and p1, 16-step greedy match) and the ten bit-identity self-tests (cpu-device = cpu-f32, prefill = decode = chunked prefill, last-row head, two stages, split dispatches, thread counts) |
| `inference-cpu-device/` | inference | `20261018-153042-bac6acca` | `lrm generate --model tiny --backend cpu-device --ids-from goldens/tiny/g1/fwd_f32.p0.safetensors --steps 64 --warmup 2 --expect goldens/tiny/g1/greedy.safetensors:p0.ids.torch32 --case p0 --run-log`: 64 of 64 ids equal the torch32 golden |
| `kernels-cpu-device/` | kernels | the same run | `lrm report 20261018-153042-bac6acca --template kernels`: the executor's `gpu.dispatch` records per kernel, layer and step |

Reading them:
- **CpuDevice times are emulation times.** The `gpu.dispatch` durations and the GPU column of the metering table are the CpuDevice's host time per dispatch (ADR E1-0001), not a GPU's; the budget line means nothing here. X1b renders the same reports on D3D12 and Vulkan.
- **Layer time** in the inference report is host time of `model.layer` spans, which on the executor is the time to record a layer's dispatches; the device's time per layer is the kernels report's `layer_stack` (GpuExecutor contract, "Telemetry").
- **Memory** is the heap layout's arenas: no `taps` arena, because `lrm generate` plans no taps.
