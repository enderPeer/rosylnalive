# R5 spike: Direct3D 12 on in-box WARP

Throwaway proof that in-box WARP, the software rasterizer that ships with Windows, runs LRM's D3D12 path: D1's backend unchanged, the embedded DXIL shader set at SM 6.6 with a directly indexed resource heap, D1's five selftests and E1's conformance harness, as a `d3d12-warp` ExecutionClass. The deliverable is the decision record, [`docs/decisions/R5-0001-warp.md`](../../docs/decisions/R5-0001-warp.md), with the measurements and the proposed `d3d12-warp` CI job. Nothing here is a dependency of the system.

`spikes/**` is outside the project graph, the solution, CI and fantomas (ProjectGraph contract). `Directory.Build.props` marks the project as a spike, so it references projects directly.

The Agility SDK half of R5 is skipped: AP-4 (the Agility SDK NuGet) is not approved, and nothing here downloads anything.

## What it does

- **The adapter.** `IDXGIFactory4::EnumWarpAdapter` gives the Microsoft Basic Render Driver. A device on it at feature level 11_0 reports its capabilities through R3's `DeviceCheck.capabilities`. The spike also reads the in-box DLL versions (`d3d10warp.dll`, `d3d12.dll`, `D3D12Core.dll`, `dxgi.dll`) and the Windows build. It looks the adapter's LUID up in D1's enumeration (`IDXGIFactory6::EnumAdapterByGpuPreference`, high performance), where the d3d12 device check rejects it on purpose.
- **The d3d12-warp verdict** (`WarpAdapter.rejections`) is the d3d12 one with the hardware rule inverted: a software adapter with Microsoft's vendor id, a device, shader model 6.6 and resource binding tier 3.
- **The backend** (`WarpBackend`) is D1's `D3D12Backend`, created with the WARP adapter's d3d12 DeviceInfo. D1's `Create` accepts any adapter of its enumeration (the device check, not `Create`, rejects WARP for d3d12), so no D1 code changes. The wrapper changes only what names the class:
  - `Info` has `Backend = D3D12Warp`, index 0 (WARP is the class's only device), and the DXGI index, the source and the DLL versions in `Details`;
  - the ExecutionClass is `GpuClass.fill` over that Info, so `Arch` keeps the CPU feature flags (WARP runs on the CPU);
  - D1 writes `backend = d3d12` on every record, so the wrapper makes a relabelling facade ambient around every call into D1's code (`Relabel`): a `backend` attribute naming `d3d12` names `d3d12-warp`, and nothing else changes.
  - `WarpProvider` is the d3d12-warp `IGpuBackendProvider` that D1's ADR asks for, built outside D1's paths.
- **The CPU limit.** WARP has a worker per logical processor. The spike pins its own process (never the system) to `LRM_WARP_CPUS` logical processors before it creates a device: a count, or `all`. The default, 4, is the vCPU count of a windows-latest runner for a public repository, so the timings model the proposed CI job.
- **Tests** (Lrm.Testing tiers; the gpu tier holds the d3d12 mutex, because WARP is Direct3D 12 and Lrm.Testing has no d3d12-warp backend):
  - cpu: the CPU limit, the relabelling facade, the verdict, the d3d12-warp DeviceInfo and class, the wrapper's refusal of a d3d12 Info, and the control shader set;
  - gpu, on WARP:
    - the adapter and its capabilities, and D1's backend opening it (with the debug layer asked for: on only when `D3D12SDKLayers.dll` is installed, which it is not here);
    - D1's fp32 denormal probe inside `LoadKernels`, with the embedded set (`-denorm preserve`), and the int-0001 control: `scale_f32` compiled now with the pinned DXC without `-denorm`;
    - D1's five selftests (nine cases) twice, bit-exact against the CPU and `KernelBounds.checkExact`, every dispatch under 100 ms, then their negative control;
    - the same selftests on the hardware adapter (the RTX 4080), compared word for word with WARP's, with the hardware run's records on the run log's `hw` node (skipped without hardware);
    - D1's copy test, the descriptor-write count, `scale_f32` over 64 Ki to 16 Mi elements for the 100 ms budget, and COM references;
    - E1's conformance suite (every discovered case, then its negative control) on d3d12-warp, through `Suites.BackendRun`.

## Running

The spike uses only local tools: the pinned DXC under `LRM_DATA` (for the control blob) and the in-box D3D12 runtime and WARP.

```bash
dotnet build spikes/d3d12/WarpSpike.fsproj -c Release
dotnet artifacts/bin/WarpSpike/release/WarpSpike.dll warp-info            # the adapter report; no mutex, no run log
LRM_TEST_TIER=cpu dotnet artifacts/bin/WarpSpike/release/WarpSpike.dll --sequenced
LRM_TEST_TIER=gpu dotnet artifacts/bin/WarpSpike/release/WarpSpike.dll --sequenced
LRM_WARP_CPUS=8 LRM_TEST_TIER=gpu dotnet artifacts/bin/WarpSpike/release/WarpSpike.dll --sequenced --filter-test-case throughput
pwsh spikes/d3d12/ci-probe.ps1 -Results artifacts/warp-ci                # what the proposed CI job runs
```

A gpu-tier run takes about 4 s at 4 processors. It writes two run logs under `LRM_DATA/runs`, both of kind `kernels`, which `lrm report <runId>` renders:

- **The spike's own run log** has two nodes, because a node has one class (RunLog):
  - **`n0`**, the creator, with the d3d12-warp class. Its records are:
    - from D1's backend, relabelled: `gpu.dispatch`, `gpu.dispatches`, `gpu.bytes_moved`, `gpu.transfer`, `gpu.wait`, `gpu.probe`, `gpu.load_kernels`, `gpu.buffer_memory`, `mem.arena`, `gpu.heap.slots`, `gpu.fp32_denormals` and `gpu.selftest`;
    - from the spike: the WARP `gpu.device.check` and `gpu.device.memory` in a `gpu.device.discover` span, `gpu.warp.backend` (timestamp frequency, DRED and whether the debug layer is on), `gpu.heap.descriptor_writes`, and the `gpu.warp.versus_hardware` checks;
    - D1's d3d12 enumeration, once: its `gpu.device.check` records keep `backend = d3d12`, since they are the d3d12 class's verdicts on each adapter (WARP rejected).
  - **`hw`**, joined with the hardware adapter's own d3d12 class (the RTX 4080, driver 616.92 here): every record of the hardware comparison's run, with `backend = d3d12`. There is no `hw` node without hardware, as on a runner.
- **E1's conformance run log** has `conformance.case` checks and the backend's records, all `d3d12-warp`.

The kernels report's headline totals ("GPU dispatches" and "total GPU time") cover every node, so the selftests report counts the hardware comparison's 10 dispatches with WARP's. The GPU-time chart, the slowest dispatches and the denormal tables name each record's backend and node, and the Nodes table names both classes.

The rendered reports are committed under [`reports/`](reports/):

- [`warp-selftests`](reports/warp-selftests/kernels.md): the spike's run at 4 processors (the probe and control bit table next to the RTX 4080's, GPU time per kernel, the selftest timeline, the slowest dispatches);
- [`warp-conformance`](reports/warp-conformance/kernels.md): E1's harness on d3d12-warp (the matrix, the denormal mode, GPU time);
- [`warp-throughput-2cpu`](reports/warp-throughput-2cpu/kernels.md) and [`warp-throughput-8cpu`](reports/warp-throughput-8cpu/kernels.md): the throughput series alone at 2 and 8 processors (`--filter-test-case throughput`).

The kernels template does not read `gpu.device.check`, `gpu.selftest`, `gpu.warp.backend` or `gpu.warp.versus_hardware`, so the ADR tabulates them (a request to the integrator, like D1's).

`ci-probe.ps1` is what a one-off run on a windows-latest runner executes. It prints the OS (`Win32_OperatingSystem`'s caption, the release and the build with its update revision), the in-box versions and whether `D3D12SDKLayers.dll` is present. Then it builds the spike and the CLI, runs `warp-info` and both tiers (`LRM_WARP_CPUS=all` by default there), and renders every new run log. It writes `summary.md` (and the GitHub step summary) on every path, failures included, with whether D1's backend ran WARP under the debug layer. It changes no setting and downloads nothing; the workflow's bootstrap step provides DXC. Its exit code:

- 0: WARP accepted and both tiers passed;
- 1: a build, `warp-info` or a tier failed, or the script did: no verdict on WARP;
- 3: `warp-info` rejected WARP for d3d12-warp (its own exit 3), the only code that is a verdict.

## Layout

| Path | Contents |
| --- | --- |
| `src/Affinity.fs` | `LRM_WARP_CPUS`: the process's own processor affinity |
| `src/Relabel.fs` | The facade that relabels D1's `backend = d3d12` records as `d3d12-warp` |
| `src/WarpAdapter.fs` | `EnumWarpAdapter`, capabilities, DLL versions, the d3d12-warp verdict, the DeviceInfos |
| `src/WarpBackend.fs` | The d3d12-warp backend over D1's `D3D12Backend`, and its provider |
| `src/Control.fs` | The int-0001 control: `scale_f32` without `-denorm`, in a copy of the shader set |
| `src/Common.fs`, `src/CpuTests.fs`, `src/GpuTests.fs`, `src/Main.fs` | The run log, the tests and the entry point |
| `ci-probe.ps1` | The runner probe for the proposed `d3d12-warp` job |
| `reports/` | Rendered kernels reports of the committed runs |
