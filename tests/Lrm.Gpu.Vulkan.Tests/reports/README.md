# V1 gpu-tier reports

`lrm report` rendered these files from real Vulkan runs on the NVIDIA GeForce RTX 4080 (`pci:10de:2704`), Vulkan 1.4.351, driver 616.92. Each report names its run and commit in its Run table.

| Folder | Run | What ran |
| --- | --- | --- |
| [`selftests/`](selftests/kernels.md) | `20261018-154459-f8574c3a`, commit `42a93ba` | The gpu tier of `tests/Lrm.Gpu.Vulkan.Tests`: the probe, the five selftests, the copied-heap fallback, copies, a whole 1 GiB page, timers, the 1.5 GB round trip and three 10,000-dispatch soaks |
| [`conformance/`](conformance/kernels.md) | `20261018-154101-9eb7a6c2`, commit `42a93ba` merged locally with `origin/main` at `707427f` | E1's conformance harness on Vulkan over every kernel that exists: the selftests and K1 to K3 (14 kernels, 214 cases, 213 pass) |

- [`selftests/kernels.md`](selftests/kernels.md) is the default template: the fp32 denormal probe's bit table (flush), GPU time by kernel, the timeline of one selftest submission and the dispatch histogram.
- [`selftests/inference.md`](selftests/inference.md) was rendered with `--template inference`: peak memory per arena (`mem.arena`) and the metering.
- [`conformance/kernels.md`](conformance/kernels.md) shows the conformance matrix per kernel and the one failing case, `scale_f32/denormals`. Its element 11 is `0x80FFFFFF * 0.5`, whose exact product is just below 2^-126: the Vulkan class flushes it before rounding, while the harness's flush-mode reference rounds first (ADR V1-0001, finding 1; the int-0001 addendum). The conformance run used a local, uncommitted merge with `main`, because K1 to K3 are on `main` and this branch is stacked on `ws/R2-vkgen`; the integrator's verify run on the future tree repeats it.

## What the reports do not show yet

No T1 template reads the records below, so they exist only in the run log (`events.n0.jsonl`). Request 3 of the V1 pull request asks the integrator for template slots.

| Record | Kind | What it holds |
| --- | --- | --- |
| `gpu.selftest` | check | One per selftest case and run. The negative control's cases fail by design and carry `negative_control = true` |
| `gpu.selftest.across_backends` | check | One per selftest case: the words compared with D3D12's, the elements excluded (a denormal input or a tiny product), the two modes |
| `gpu.fp32_tininess` | event | The eight inputs at the bottom of the normal range, the measured words, IEEE, flush after rounding and flush before rounding |
| `gpu.soak` | check | One per soak run (`mode` = `timed_dispatches`, `submission_timer`, `no_timers`): host, GPU, frame, Vulkan-object and private-memory flatness, and managed memory per dispatch against its bound |
| `gpu.heap.descriptor_writes` | check | The resource heap's 64 and the probe heap's 64 descriptor writes |
| `gpu.heap.reserved_range` | check | No descriptor write into the reserved range; the slot area unchanged after a dispatch, a copy and transfers |
| `gpu.memory_type` | event | The memory type chosen by flags for each kind of buffer (`page`, `resource_heap`, `upload_staging`, `readback_staging`; `resource_heap_copied` for the fallback) |
| `gpu.shader.spirv_check`, `gpu.shader.push_bytes`, `gpu.shader.workgroup_bytes` | check, gauge | R4b's SPIR-V inspector on every blob at `LoadKernels` |
| `gpu.device.check`, `gpu.device.memory`, `gpu.device.discover`, `gpu.device.create` | check, gauge, span | Device discovery, verdicts and device creation (R2's DeviceCheck) |
| `gpu.transfer`, `gpu.transfer.rate` | span, gauge | Every staged transfer, and the 1.5 GB round trip's rates |
| `gpu.buffer_memory` | gauge | Staging, resource-heap, dummy-page and probe-heap memory outside the arenas |

The `gpu.probe`, `gpu.load_kernels` and every other span, GPU and meter record are counted in the metering table of `inference.md`.
