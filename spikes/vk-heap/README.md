# R1 spike: Vulkan descriptor heap

Throwaway proof that the planned binding model works on the RTX 4080 with `VK_EXT_descriptor_heap`, and that the same HLSL gives the same bits on D3D12. The deliverable is the recipe, [`docs/decisions/R1-heap-recipe.md`](../../docs/decisions/R1-heap-recipe.md), and its contract text, [`docs/contracts/vulkan-heap-recipe.md`](../../docs/contracts/vulkan-heap-recipe.md). V1 implements the recipe in `src/Lrm.Gpu.Vulkan`; nothing here is a dependency of the system.

`spikes/**` is outside the project graph, the solution, CI and fantomas (ProjectGraph contract). `Directory.Build.props` marks the project as a spike, so it may reference projects directly.

## What it runs

- **Vulkan**, per configuration:
  - a logical device on the compute-only queue family, with `descriptorHeap`, `shaderStorageBufferArrayDynamicIndexing`, `bufferDeviceAddress`, `timelineSemaphore`, `runtimeDescriptorArray`, `synchronization2` and `maintenance5`;
  - arena pages as device-local buffers with device addresses;
  - one resource heap, written once with `vkWriteResourceDescriptorsEXT`;
  - pipelines created with `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`, a `NULL` layout and a set 0 binding 0 → heap mapping (`HEAP_WITH_CONSTANT_OFFSET`, heapOffset 0, stride `bufferDescriptorSize`);
  - per dispatch, `vkCmdBindResourceHeapEXT`, `vkCmdPushDataEXT` with 128 bytes at offset 0 and timestamps, then a timeline-semaphore wait.

  The configurations cover two binding variants and three heap placements:
  - binding variants: explicit `[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[]`, and emulated `-fvk-bind-resource-heap 0 0`;
  - heap placements: BAR memory written in place, host memory written in place, and device-local memory filled by a copy.

  Two mapping controls (heapOffset 512, and stride 32) show that the driver honours the mapping values.
- **D3D12:** the same HLSL as DXIL `cs_6_6`, using `ResourceDescriptorHeap` with the global root signature (32 root constants at b0), one shader-visible heap of 64 raw UAVs, and timestamp queries.
- **Kernels** (`hlsl/`):
  - `scale_f32`;
  - `args_echo`, which echoes all 32 ArgBlock dwords;
  - `table_echo`, which echoes the whole 16,384-row tensor table;
  - `bf16_unpack`, run on all 65,536 bf16 patterns and on 2 Mi values of the real Qwen3-0.6B `embed_tokens`.

  There are also two dependent dispatches in one command buffer, a scale at the end of a 1 GiB page, and an informational edge-value case (denormals, infinities and NaN).
- **Checks:**
  - before each run, every region the job writes is filled with poison: signalling NaNs that differ per configuration, case and run, and never equal the expected word;
  - every case runs twice, must leave no poison word and must give identical bytes;
  - gating cases must match the F# CPU reference word for word, and every configuration must give the same SHA-256 as D3D12;
  - a negative control reruns every case on every configuration with its dispatches removed, and each run must fail.
- **CPU tier:**
  - struct layouts are checked against the C declarations, and cl.exe checks the layout calculator;
  - the pinned DXC version;
  - SPIR-V inspection: resource variables, the push block, LocalSize and capabilities;
  - known answers for the recipe's mapping bytes and heap geometry;
  - the case regions (no write overlaps an input, every readback is poisoned) and the poison itself.

## Running

The spike uses only local tools: the pinned DXC under `LRM_DATA`, and the Visual Studio 2022 Build Tools for the cl.exe check. The GPU tier takes the per-backend mutexes through `Lrm.Testing`. The whole run takes about 5 seconds, most of it host work. The timed cases run first and the negative controls last, because a few seconds into a mostly idle run the driver drops the GPU from P0 to P5, which roughly doubles dispatch times.

```bash
dotnet build spikes/vk-heap/VkHeapSpike.fsproj -c Release
LRM_TEST_TIER=cpu dotnet artifacts/bin/VkHeapSpike/release/VkHeapSpike.dll --sequenced
LRM_TEST_TIER=gpu dotnet artifacts/bin/VkHeapSpike/release/VkHeapSpike.dll --sequenced --results artifacts/spike-results/gpu
```

A gpu-tier run does three things:
- writes `results.md` and `dispatch-gpu-us.svg` to the `--results` directory, or to `artifacts/spike-results/gpu` without one. Only `--update-committed-results` also writes them over the committed [`results/results.md`](results/results.md) and [`results/dispatch-gpu-us.svg`](results/dispatch-gpu-us.svg), so a rerun leaves the tree clean;
- writes a run log, `LRM_DATA/runs/<runId>/{manifest.json,events.n0.jsonl}`, in the RunLog key layout (`v t k n d x u node seq a`). The records are `gpu.device` and `gpu.spike.config` events, `gpu.dispatch` GPU intervals, `gpu.bytes_moved` meters, and `gpu.spike.case` and `gpu.spike.negative_control` checks;
- writes the Lrm.Testing summary, with device, driver and ShaderSetHash.

## Layout

| Path | Contents |
| --- | --- |
| `hlsl/lrm_spike.hlsli` | The binding convention under test: `LRM_ARG`, `LRM_BUF`, `lrm_tensor` |
| `hlsl/*.hlsl` | The four kernels, the root signature and the `uint d[32]` layout probe |
| `src/Layout.fs` | C declarations transcribed from the reference pages, an x64 layout calculator, the CLR-layout check and the cl.exe source |
| `src/Vk.fs` | Hand-written structs, constants and PFNs beyond `Lrm.Gpu.Vulkan.Interop` |
| `src/VkRunner.fs` | Device, pages, heap, pipelines, jobs and the recipe's mapping |
| `src/D3D12Runner.fs` | The D3D12 comparison path, through vtable slots from the SDK 10.0.26100 `d3d12.h` |
| `src/Heap.fs`, `src/Cases.fs` | Slot plan, tensor table, cases and CPU references |
| `src/Shaders.fs` | DXC invocation and SPIR-V inspection |
| `src/RunLog.fs`, `src/Report.fs` | Telemetry sink and the results page |
