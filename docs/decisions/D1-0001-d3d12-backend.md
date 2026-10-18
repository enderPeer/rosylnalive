# ADR D1-0001: The D3D12 backend: per-list state tracking, copies through a scratch buffer, and what it measured

- **Status:** proposed (D1, 2026-10-18), with the D1 pull request.
- **Owner:** D1.
- **Scope:** `src/Lrm.Gpu.D3D12` (the `d3d12` backend of GpuBackendInterface) and the facts the executor (X1), the conformance harness (E1), the Vulkan backend (V1) and the WARP spike (R5) build on.

## Context

GpuBackendInterface fixes what a backend does: one heap of 64 slots written once, one kernel set, command lists of dispatches, barriers, copies and timers, a fence per submission, the shared checks (`Dispatch.prepare`, `checkCopy`, `checkTransfer`) so every backend raises the same `GpuContractException`, and `Copy` that is state-correct on every backend. It leaves open how D3D12's resource states, staging, timestamps and device-loss diagnostics are handled. Those choices bind X1, which records many copies (taps) and depends on their barrier semantics, and they are measured facts later gates rely on.

## Decision

1. **A pure encoder.** A command list records commands, each checked when it is recorded, with RecordingBackend's checks in the same order and with the same texts. At `Submit`, `Plan.encode` turns them into native calls (`NativeOp`), and the backend replays them on an `ID3D12GraphicsCommandList`. `Plan.verify` checks an encoding against the state rules. The cpu tier runs it on known answers and on random command lists. With `DeviceOptions.Debug`, the backend also runs it on every submission.
2. **Pages are COMMON between command lists.** Each list starts with every page and the table in COMMON and ends with every page it touched back in COMMON, through explicit transitions. The invariant does not depend on D3D12's implicit promotion and decay.
   - A dispatch first moves the table (slot 0) and the pages of its tokens to UNORDERED_ACCESS.
   - `Barrier ()` is a UAV barrier with a null resource (all UAV accesses).
   - Pipelines are set when the kernel changes, and the argument block is the 32 root constants at b0.
3. **`Copy` is a full barrier on both sides:** a null UAV barrier and UAV→COPY_SOURCE and UAV→COPY_DEST transitions before the copy, and both pages back to UNORDERED_ACCESS after it.
   - A copy within one page goes through a 64 MiB DEFAULT scratch buffer, in chunks of at most 64 MiB, because one buffer cannot be in COPY_SOURCE and COPY_DEST at once.
   - The scratch buffer is created on the first such copy.
4. **Transfers are staged.** `Upload`, `UploadFrom` and `Download` wait for all submitted work, then go through two staging buffers per direction, so one chunk's copy overlaps the next chunk's memcpy.
   - Each staging buffer is at most 64 MiB (the chunk size), or 64 KiB steps for smaller transfers. Uploads use persistently mapped UPLOAD buffers and downloads READBACK buffers mapped per chunk.
   - `UploadFrom` reads the stream straight into the mapped upload memory.
   - Every transfer is a `gpu.transfer` span (direction, slot, bytes) and a `gpu.bytes_moved` meter.
5. **Submissions and frames.**
   - `Submission` values count the caller's submissions (1, 2, 3, ...). The fence also counts the backend's own transfer and probe lists.
   - A frame (allocator, list, timestamp query heap and readback buffer) is reused only after its fence value has completed. The pool stays at its in-flight depth: 1 in the soak.
   - **The backend keeps only what `Wait` needs.** The native ops are recorded into the frame's list at `Submit` and not kept. Until its frame retires, a submission keeps its fence value and timer slots; from then on, its dispatch count (4 bytes) and, if it has timers, their results. A timed dispatch's `gpu.dispatch` timing is kept as three columns (the kernel's shared attribute list, start and duration: 24 bytes), from which `Wait` rebuilds an equal `Completion` every time. GpuBackendInterface keeps every completion waitable, so this much stays for the backend's lifetime (request to the integrator).
   - `GetDeviceRemovedReason` runs after every `ExecuteCommandLists` and every fence wait, and waits are bounded at 30 s. A removed device or a timed-out wait raises `D3D12DeviceLostException`, which carries the removed reason and DRED's report. It is not a contract violation.
6. **Timers are emitted once.** A submission's timers and `gpu.dispatch` timings go to telemetry at its first `Wait`. A later `Wait` returns the same `Completion` and emits nothing, because emitting again would double-count GPU time in reports. RecordingBackend emits on every `Wait`. Timestamps are converted to ns exactly, in integers.
   - **With `TimeDispatches`, every dispatch runs alone:** a UAV barrier before each of its two timestamps. Without them the RTX 4080 overlaps independent dispatches and their timestamps do not bracket them; for example, a folded groups_echo of 131,077 groups read 1 µs and the args_echo before it 64 µs. With them, the same groups_echo reads 65.5 µs and args_echo 1 µs. Timing mode therefore serializes dispatches; without it nothing is added.
7. **DRED and the debug layer.**
   - DRED auto-breadcrumbs, breadcrumb contexts and page-fault reporting are forced on around the backend's `D3D12CreateDevice`, then restored to system-controlled. The settings are process-wide and apply to devices created while they are set.
   - `DeviceOptions.Debug` enables the debug layer only before the process's first device, because enabling it while a device exists removes that device.
   - `Dred.report` and `Dred.format` read the breadcrumb and allocation lists, and mark the first unfinished operation.
8. **The fp32 denormal probe runs inside `LoadKernels`.** It uses a private heap (a table and one 64 KiB page, its own descriptor heap), so it needs no caller heap and does not count as `CreateHeap`.
   - The outputs are poisoned first, and a word that keeps its poison fails the probe.
   - It records with `Fp32Denormals.recordProbe`, in case `fp32_denormal_probe`.
   - The mode is `D3D12Backend.Fp32Denormals`.
   - **The probe's heap is a second, transient descriptor heap.** It writes its own 64 descriptors and is released when the probe ends. `D3D12Backend.DescriptorWrites` counts both heaps: `HeapDescriptorWrites` (64, at `CreateHeap`) plus `ProbeDescriptorWrites` (64, at `LoadKernels`). The fixed decision "one resource heap, written once at startup" holds for the caller's heap; the transient probe heap needs a line in HeapLayoutAndTensorTable or GpuBackendInterface (request to the integrator).
   - **Its telemetry follows the GpuBackendInterface table.** The dispatch is in `gpu.dispatches`, and it is a `gpu.dispatch` interval only with `TimeDispatches`. Its private transfers are not `IGpuHeap` transfers, so they are not `gpu.bytes_moved`. The `gpu.probe` span carries them (`upload_bytes`, `download_bytes`) with the heap's `descriptor_writes`.
9. **The selftests live in the backend library** (`SelfTests`, over `IGpuBackend` only), so `hello-d3d12` can run them and V1 can call the same module.
   - Every output is poisoned per run and per case, with a pattern that never equals, or NaN-matches, the expected word. The words after each written region are poisoned and must keep their poison.
   - A negative control without dispatches must fail every case.
   - scale_f32 is compared in the class's measured mode, with any NaN matching any NaN; the bit-moving selftests compare every bit.
10. **`hello-d3d12`** is `Hello.run shaders args` in the library. The test exe's `hello-d3d12` entry point runs it under the d3d12 GPU mutex. The CLI command is one line for the integrator (the CLI files are not D1's).
11. **Memory outside the arenas is a gauge.** `gpu.buffer_memory` (bytes; `buffer` = `upload_staging`, `readback_staging`, `copy_scratch` or `probe_heap`) is emitted whenever that memory changes. `mem.arena` stays the heap's arenas, as GpuBackendInterface defines it.
12. **After `Dispose`, every call raises `ObjectDisposedException`,** including calls on a command list begun before it. RecordingBackend accepts those calls (request to the integrator). A shader set whose `rootsig.bin` is not canonical raises `ArgumentException` from `LoadKernels`, not `GpuContractException`, because no contract rule covers it and RecordingBackend accepts such a set. `ShaderSet.embedded` already refuses one.

## Evidence

Measured on the development machine with the gpu tier of `tests/Lrm.Gpu.D3D12.Tests` (17 tests, d3d12 mutex): NVIDIA GeForce RTX 4080 (`pci:10de:2704`), UMD 32.0.16.1692 (driver 616.92), Windows 11 10.0.26200, DXIL ShaderSetHash `b12cabbf300be63253a5e5bf1dbba6aa6fd34d0ef3a6f4303e712ae6f148f4be`, ExecutionClass id `9469669b5935e96a16218c5d67adc1b8b479989cca03ef3370eaa556df640569`. Run `20261018-055325-c51ed242` (commit `0e5e4b9`) renders to [`kernels.md`](../../tests/Lrm.Gpu.D3D12.Tests/reports/selftests/kernels.md) (the denormal bit table, GPU time by kernel, the timeline of one selftest submission, the dispatch histogram) and [`inference.md`](../../tests/Lrm.Gpu.D3D12.Tests/reports/selftests/inference.md) (peak memory per arena and metering) in `tests/Lrm.Gpu.D3D12.Tests/reports/selftests/`. That folder's README lists the checks no template reads yet (`gpu.selftest`, `gpu.soak`, `gpu.heap.descriptor_writes`).

- **Probe:** `preserve`, `3f000000 00800000 00400000 80400000 00000001 803fffff 00200000`, the same in every run (int-0001 agrees).
- **Selftests:** nine cases of the five selftests are bit-exact against the CPU and against `KernelBounds.checkExact`, and bit-identical across two runs on one backend and across two backends.
  - The cases: scale_f32 x→y (4,099 elements) and in place; table_echo on a strided view, on the null token and on a bf16 row; args_echo over all 32 words; groups_echo at 300 groups and at 131,077 groups (folded to 43,693 × 3, with the 2 extra groups writing nothing); bf16_unpack over all 65,536 patterns.
  - Every NaN result of scale_f32 is `0x7FFFFFFF`, whatever the input payload (R1's pitfall 10).
- **Contract errors:** a 48-step script of valid and invalid calls (39 of them errors) gives exactly RecordingBackend's outcomes, exception type and text, step by step.
- **After `Dispose`:** `Dispatch`, `Barrier`, `Copy`, `BeginTimer` and `EndTimer` on a list begun before it, and `Submit`, `Wait`, `Begin`, `ExecutionClass`, `CreateHeap`, `LoadKernels`, `WriteTable`, `Upload` and `Download`, all raise `ObjectDisposedException`. A `rootsig.bin` with 4 extra bytes raises `ArgumentException`, and `LoadKernels` then succeeds with the embedded set.
- **Probe telemetry,** in an `InMemorySink`, untimed and timed: 0 and 1 `gpu.dispatch` intervals, 1 `gpu.dispatches` either way, no `gpu.bytes_moved`, and a `gpu.probe` span with `upload_bytes` 1,048,860, `download_bytes` 28 and `descriptor_writes` 64. `gpu.buffer_memory` for `probe_heap` is 1,114,112 bytes, then 0.
- **Heap:** 64 descriptor writes on the resource heap at `CreateHeap`, and 64 on the probe's transient heap at `LoadKernels`, released after the probe. There are none afterwards, through `WriteTable`, a dispatch, a copy, an upload and a download (`DescriptorWrites` = 128). There is one committed DEFAULT buffer per page, of the page's size.
- **1 GiB pages:** a raw UAV with `NumElements` = 2^28 over a 1 GiB page reads and writes correctly at bytes 0, 512 MiB + 64 KiB and 1 GiB − 64 KiB. No debug layer is installed, so whether the runtime's validation accepts a raw view of more than 2^27 elements is unverified.
- **Copies:** dispatch → copy across pages → dispatch → copy within a page → dispatch → copy within a page, in one list, gives the CPU's values for every intermediate.
- **Round trip:** `model.safetensors` of Qwen3-0.6B (1,503,300,328 bytes, two Weights pages) goes up with `UploadFrom` from the file and comes back with its SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`, twice in every gpu-tier run. `UploadFrom` takes one call per page. `Download` takes nine calls of at most 200 MiB + 12 bytes; seven of them are four chunks long, so the pipelined path runs, and every call after a page's first starts at an offset that is not chunk-aligned. Upload took 0.41 to 0.53 s (2.8 to 3.7 GB/s, the file in the page cache) and download 0.87 to 0.94 s (1.6 to 1.7 GB/s, with SHA-256 on the host).
- **Soak:** 10,000 dependent scale_f32 dispatches (a = −1, a barrier between each), twice in each of three modes. With timed dispatches and with one timer per submission, they run in 100 submissions of 100; without timers, in 1,000 submissions of 10. The result is exact. From the first tenth of the submissions to the end, the frame pool stays at 1 and live COM references at 20 (18 without timers: no query heap). No submission is pending after the last `Wait`.
  - Median GPU time is flat: 2.0 µs per timed dispatch (1.0 µs before the timing barriers), and 150.3 to 150.5 µs per timed submission of 100.
  - Median host time per submission does not grow: from 1.11 to 1.00 ms timed, 0.53 to 0.49 ms with a submission timer, and 0.17 ms without timers.
  - Private memory changes by −207 to +1.7 MB.
  - **Managed memory** is sampled after a full GC at every tenth, and the Theil–Sen slope through the ten samples gives the growth per dispatch. It is 14.7 to 26.0 bytes per timed dispatch (bound 48), mostly the 24-byte timing that a completion keeps. With a submission timer it is 2.5 to 2.8 bytes (248 to 277 per submission, bound 16), and without timers 0.54 bytes (5.4 per submission, bound 16), which is the 4-byte dispatch count. Before the fix the backend kept every submission's encoding: the review measured 281 bytes per untimed dispatch and 809 per timed one.
- **References:** `Com.liveReferences ()` is the same before a backend is created and after it is disposed.
- **DRED:** forced on; on a healthy device both reports return `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE`, which the report names. The parser is checked in the cpu tier on breadcrumb and allocation lists laid out in unmanaged memory, as d3d12.h declares them. No device removal was provoked: the GPU is shared.

## Consequences

- **X1:** every `Copy` costs two `ResourceBarrier` calls (four transitions and a UAV barrier). If taps make that measurable, coalescing consecutive copies is a backend change with no interface change. Copies within one page cost twice the bytes, through the scratch buffer.
- **Long runs (X1, C2, training):** a backend keeps 4 bytes per submission, a few hundred per submission timer, and 24 bytes per timed dispatch, for its lifetime, because GpuBackendInterface keeps every completion waitable. A million untimed submissions cost 4 MB. A profiling run that times every dispatch of a long training run grows by 24 bytes per dispatch. If the integrator bounds how long a completion stays waitable, the backend can drop these too.
- **E1 and V1:** `SelfTests` takes any `IGpuBackend` and the measured mode. V1 can run it as it is, or the integrator can move it to `Lrm.Gpu.Core`.
- **R5:** `D3D12Backend.Create` accepts only `d3d12` devices. A `d3d12-warp` backend needs a provider that enumerates the software adapter and a class with `Backend = D3D12Warp`. The encoder, heap and probe would be reused unchanged.
- **Reopen** this decision if a debug-layer run (AP-3, or the Graphics Tools feature) reports a state or view error, if a driver update changes the probe, or if enhanced barriers are adopted. The adapter reports enhanced barriers, but the interop does not bind `ID3D12GraphicsCommandList7`.

## Alternatives

- **Rely on implicit promotion and decay.** Buffers decay to COMMON after `ExecuteCommandLists`, so a tracker could start every list from COMMON and transition only on demand. Rejected: explicit transitions back to COMMON make the invariant independent of decay rules, cost one barrier call per list, and `Plan.verify` can check them.
- **Lazy copy states,** leaving a page in COPY_DEST until something else needs it. Rejected: two copies into the same bytes would then have no barrier between them, so `Copy` would not be a full barrier.
- **One placed-resource heap per arena** instead of committed buffers. Rejected for now: pages are at most 1 GiB and few, and committed buffers need no heap sizing, offsets or aliasing rules.
- **Recording into native lists at record time.** Rejected: the timestamp query count and the state transitions are only known for the whole list, and a pure encoder can be tested without a GPU.
