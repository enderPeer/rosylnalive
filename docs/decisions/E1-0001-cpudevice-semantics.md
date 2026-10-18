# ADR E1-0001: What CpuDevice enforces beyond the shared checks, and how it models GPU arithmetic

- **Status:** proposed (E1, 2026-10-18), with the ConformanceHarness contract.
- **Owner:** E1.
- **Scope:** `src/Lrm.Gpu.CpuDevice`, `tests/Lrm.Kernels.Conformance`; it binds X1 (which runs on CpuDevice in CI), K1 to K3 (whose cases run on it) and the integrator (the CPU fill rule of ExecutionClass).

## Context

CpuDevice is the executing counterpart of RecordingBackend: the only backend in CI, and the device X1's executor is proven on before any GPU. Whatever it lets through reaches the GPUs untested, where the failure is a wrong number, a TDR or nothing at all. The shared checks (`Dispatch.prepare`, `checkCopy`, `checkTransfer`) leave three classes of mistakes open: kernel-specific preconditions (kernel-abi.md, "Tensor access", which `Dispatch.prepare` deliberately does not check), missing barriers between dependent dispatches (GpuBackendInterface allows consecutive dispatches to overlap), and code that relies on zero-initialized memory. The class-dependent rules of fp32 denormals and NaN (ADR int-0001, R1 pitfall 10) also need a CI test before V1, the likely flush class, exists.

## Decision

1. **Kernel-specific rules raise when a dispatch is recorded.** `KernelRules` computes each data token's footprint from the u32 fields (in int64, overflow raising) after the other argument rules, and raises `GpuContractException` when a token holds fewer elements than the footprint's end. The message names the kernel, field, tensor, both counts and the rule as kernel-abi.md writes it.
2. **Barriers are enforced** (`CheckBarriers`, on by default). Within one command list, a dispatch that reads bytes an earlier dispatch wrote, or writes bytes an earlier one read or wrote, with no `Barrier` or `Copy` in between, raises. The bytes are the footprints in whole words, not the tokens' extents, so disjoint writes into one tensor (chunked attention, interleaved `copy_rows`) need no barrier. A command-list boundary counts as a barrier, because every backend serializes submissions.
3. **Memory starts as `7fbadbad` words**, not zeros (`UninitializedWord`), because Vulkan memory and a reused page are undefined.
4. **GPU arithmetic models are options, never the default.** `Fp32Denormals = Flush` models a flushing class (denormal fp32 and bf16 inputs and f32 arguments of the kernels that compute on fp32 values read as zero of the same sign, and their fp32 results are flushed); `CanonicalNaN` returns `7fffffff` for every NaN result of an fp32 arithmetic kernel. A model appends `+ftz` or `+nan-canonical` to the class's `Arch`, so its results never share a class, and no bit-identity claim, with the default CpuDevice, whose class is `ExecutionClass.forCpu cpu-device` exactly.
5. **Every catalog kernel has an emulator**, sources or not, so X1 and the harness run the whole catalog in CI before K1 to K3 land; `MirrorShaderSet` gives a shader backend's view instead.
6. **The dispatch budget gates only shader backends.** CpuDevice reports emulation time as `gpu.dispatch` (host time) and in `conformance.case`, but 100 ms of emulation says nothing about a GPU.

## Evidence

- Every rule above has a cpu-tier test in `tests/Lrm.Gpu.CpuDevice.Tests` (`DispatchRuleTests.fs`, `InterfaceTests.fs`, `EmulatorTests.fs`).
- Mutation check on this branch: making the minimum-count check one element lax failed all 34 "one element short" tests; corrupting one element of `scale_f32` and one of `matmul_bf16w` (only when m > 1) failed the emulator tests, A1, the harness's fault-free control and three conformance cases, including the m = 37 matmul through its M = 1 batch split.
- The harness's model tests run every case on a flushing, a NaN-canonical and a combined CpuDevice: the probe measures flush where it should, and every case passes against references built in that mode.

## Consequences

- **X1** inserts a barrier between dependent dispatches, or its CpuDevice runs raise; it never reads memory its plan did not write.
- **K1 to K3** get the minimum-count and argument checks on CpuDevice for free; GPU kernels keep assuming them.
- **The integrator** may want ExecutionClass's CPU fill rule to name the `+ftz` and `+nan-canonical` Arch suffixes of CpuDevice models (request in the E1 PR); nothing outside the harness's tests creates a model.
- Reopen point 2 if a backend is found that does not serialize submissions on its queue.

## Alternatives

- **Check against token extents instead of footprints.** Rejected: it would refuse legitimate disjoint writes into one max-size token, which the sealed table makes the normal case (kernel-abi.md, "A token's count is a capacity").
- **Model flush with FTZ/DAZ in MXCSR.** Not available from .NET without native code, and it would change every intermediate rounding, not the contract's model of inputs and results.
- **Zero-initialized pages,** as RecordingBackend has. Rejected for CpuDevice: it would hide a read of unwritten memory that a GPU returns as garbage.
