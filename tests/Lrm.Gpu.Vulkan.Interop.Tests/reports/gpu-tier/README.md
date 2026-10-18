# R2 gpu-tier reports

`lrm report` rendered these files from the gpu tier's run log (kind `kernels`, run `20261018-144112-f93fa5d9`, commit `2b77c60`) of `tests/Lrm.Gpu.Vulkan.Interop.Tests` on the RTX 4080 with driver 616.92. Both name the run and the commit in their Run table.

- [`kernels.md`](kernels.md) is the default template: GPU time by kernel, the timeline and the dispatch histogram of hello-compute's two `args_echo` runs (VulkanHeapRecipe KAT-4).
- [`inference.md`](inference.md) was rendered with `--template inference`: the metering (host time, GPU time and bytes moved).

## What the reports do not show yet

No T1 template reads the records below, so they exist only in the run log (`events.n0.jsonl`). The R2 pull request asks for a device and interop template slot. The conformance matrix and the denormal table in `kernels.md` stay empty by design: the interop runs no conformance cases and no denormal probe (E1 and V1 do).

| Record | Kind | What it holds |
| --- | --- | --- |
| `gpu.device.check`, `gpu.device.memory`, `gpu.device.discover` | check, gauge, span | The device verdicts (with reasons and `max_push_data_size`), heap sizes, budgets and usage |
| `gpu.device.create` | span | Each logical device: device, queue family, extensions, commands missing |
| `gpu.device.dispatch_table` | check | Loaded against expected device commands, per round |
| `gpu.interop.hello` | check | Hello-compute's output digest against KAT-4 |
| `gpu.interop.allocations` | check | Managed bytes allocated by 3,000 recorded wrapper calls |
| `gpu.device.fault_query`, `gpu.debug.utils`, `gpu.device.cooperative_matrix`, `gpu.device.calibration_deviation` | check, gauge | The diagnostics outcomes: fault query result, messages, shapes and time domains, calibration deviation |
| `gpu.debug.messenger_rooted` | check | A messenger dropped without Dispose: messages received after three full collections |

The cpu tier's records (`gpu.codegen.vulkan`, `gpu.abi.oracle`, `gpu.abi.mutants`, `gpu.abi.parameter_names`, `gpu.abi.spike_crosscheck`, `gpu.interop.tailcalls`) go to whatever facade the host installs; the cpu tier writes no run log.
