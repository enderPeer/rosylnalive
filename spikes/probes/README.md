# Planning probes

The small probes behind the facts the dev plan calls "verified while planning" (`docs/dev-plan.md`, section 1).

- **Where they come from.** They were written in the planning session of 2026-10-17 and staged by the integrator in `LRM_DATA/spikes-staging` (plan, step 0). S0b copied them here after reading each file.
- **Status.** They are throwaway evidence:
  - never built, and not in the solution or CI (ProjectGraph);
  - not formatted (`.fantomasignore` excludes `spikes/**`).
- **Kept as they were run.** The contents are unchanged, except that `redteam/inv_torch.txt` was converted from CRLF to LF. Paths inside the probes are the ones they ran with, and some predate S0a's layout (see "Running them today").
- **Not copied:**
  - `plan.json` and `critique.json`, which are planning outputs, not probes;
  - `bindtest/explicit.dxil` and `bindtest/rs.bin`, which are compiled outputs. The pinned DXC reproduces them; `rs.bin` is byte-identical to today's `tools/bootstrap/dxc-regression` root signature (92 bytes, SHA-256 `a54491c9…`).

  The plan's step 0 never staged `dl/` (Hugging Face files), third-party docs or `extract-roslyn/`.

## What each probe checks

| Probe | What it checks | Where the finding lives |
| --- | --- | --- |
| `bindtest/explicit.hlsl` | An explicit `[[vk::binding(0,0)]] RWByteAddressBuffer g_buf[]` with `NonUniformResourceIndex` gives one SPIR-V variable at set 0 binding 0 | ADR 0001 (the convention drops `NonUniformResourceIndex`; the layout is the same) |
| `bindtest/emul.hlsl` | `ResourceDescriptorHeap[…]` with `-fvk-bind-resource-heap 0 0` gives one aliased SPIR-V variable per access | ADR 0001 (this form is not used) |
| `bindtest/rs.hlsl` | The global root signature compiles with `-T rootsig_1_1 -E LRM_ROOTSIG` to 92 bytes | ADR 0001; `tools/bootstrap/dxc-regression/rootsig.hlsl` |
| `dxc-probe/probe.hlsl` | The first DXC probe: `ResourceDescriptorHeap` loads with a bf16 widen through `uint16_t`. It needs `-enable-16bit-types`, and kernels may not use 16-bit types (KernelAbiAndCatalog) | STATUS (DXC verified) |
| `shaders/add.hlsl` | A `RWStructuredBuffer<float> bufs[]` descriptor array with push constants | planning comparison of binding forms |
| `shaders/add_heap.hlsl`, `shaders/add_heap_rs.hlsl` | An SM 6.6 `ResourceDescriptorHeap` add; the `_rs` variant embeds the root signature for `probe/d3d12probe.fsx` | planning; R3 and D1 |
| `shaders/half.hlsl` | A `half` load from a byte-address buffer | planning (16-bit types are banned in kernels) |
| `nfc.fsx`, `nfc/nfc.fsx`, `bindtest/nfc.fsx`, `planner-contract/nfc_inv.fsx` | `string.Normalize(FormC)` is silently a no-op under `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1`. `nfc.fsx` also prints Rune categories of astral code points (U+1D49C, U+1FAE9, U+31350). The strings hold decomposed sequences on purpose (`e` U+0301, U+212B, U+1E9B U+0323, conjoining Hangul jamo), so never let an editor normalize these files | LrmCore `Process.init` and its NFC canary; M1 |
| `golden-smoke/smoke.py` | The golden venv runs Qwen3-0.6B in fp32 (eager). It checks that `lm_head` is bitwise equal to `embed_tokens`, prints the `q_norm` weight shape and the ids of a tokenized C# line, then prints the top 3 next tokens and a logits hash | STATUS (golden-tool venv) |
| `redteam/fp64check.py` | HF Qwen3 keeps RMSNorm and RoPE in fp32 inside a `.double()` model, so plain fp64 is not an fp64 truth. It also counts the `inv_freq` entries where torch's fp32 differs from `f32(f64 formula)` | Dev plan, section 6 (`torch64p` patches, `T_spec`) |
| `redteam/inv.py` | Dumps torch's fp32 `inv_freq` (head_dim 128, theta 1e6) as u32 bits into `inv_torch.txt` | |
| `redteam/inv_torch.txt` | That dump: 64 values, the input of `inv.fsx`, kept so `inv.fsx` runs without torch | |
| `redteam/inv.fsx` | Compares three closed-form F# `inv_freq` formulas with the torch bits; none matches in every entry | Dev plan, section 1: `inv_freq` is pinned data (O1, O2, G2) |
| `probe/d3d12probe.fsx` | A read-only D3D12 probe through raw vtable slots (indices from Windows SDK 10.0.26100 `d3d12.h`). It reports the adapter and feature support (shader model, binding tier, wave ops, 16-bit ops, 64-bit atomics, root signature version), then runs one SM 6.6 `ResourceDescriptorHeap` compute dispatch and checks the readback | R3 ("slots match the probe"); D1 |
| `probe/vkdevice.fsx` | Vulkan queue families, memory heaps and types, and descriptor sizes. It creates a logical device with `descriptorHeap`, untyped pointers and cooperative matrix enabled, and checks that the `VK_EXT_descriptor_heap` commands resolve. It creates and destroys the device only | R1, V1 |
| `probe/vklimits.fsx` | Vulkan device limits: storage-buffer range, push-constant size, shared memory, workgroup counts, timestamp period | R1, R4a |
| `probe/vkcoop.fsx` | The device extensions of interest and the `VK_KHR_cooperative_matrix` configurations | Phase 2 performance track |
| `probe/callbench.fsx` | The cost per native call from F#: `DllImport`, a delegate from a function pointer, and `calli` through a DynamicMethod | R2, R3 (binding style) |
| `tools/vtbl.py` | Counts `DECLSPEC_XFGVIRT` entries per interface vtable in `d3d12.h` to get COM slot indices | R3's vtable generator uses the same method |

## Running them today

- **Shaders.** Use the pinned DXC only (`$(LrmDxcExe)`, which is `LRM_DATA/toolchain/dxc/1.9.2609.5/build/native/bin/x64/dxc.exe`).
  - DXIL: `dxc -T cs_6_6 -E main -HV 2021 <file> -Fo <out>`.
  - SPIR-V: add `-spirv -fspv-target-env=vulkan1.3`, plus `-fvk-bind-resource-heap 0 0` for `emul.hlsl` and `-enable-16bit-types` for `dxc-probe/probe.hlsl`.
  - Compile `shaders/add_heap_rs.hlsl` to `shaders/add_heap_rs.dxil` before running `d3d12probe.fsx`.
- **F# scripts:** `dotnet fsi <file>.fsx`.
  - The three `vk*.fsx` probes reference the pre-S0a build output `C:\Users\end\dev\rosylnalive\gpu\vulkan\app\VkDeviceCheck\bin\Debug\net10.0\Lrm.Gpu.Vulkan.dll`. Today, build `src/Lrm.Gpu.Vulkan.Interop` and point the `#r` at `artifacts/bin/Lrm.Gpu.Vulkan.Interop/debug/Lrm.Gpu.Vulkan.Interop.dll`. The namespace is still `Lrm.Gpu.Vulkan`, but the binding has moved on since, so expect small API differences.
  - The probes that touch the GPU (`d3d12probe.fsx`, `vkdevice.fsx`, `vklimits.fsx`, `vkcoop.fsx`) run outside the test mutexes. Other agents share the RTX 4080, so keep runs short, and prefer the GPU tier of the owning workstream's tests.
- **Python:** use the golden venv only, with `python -I`. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `USE_HUB_KERNELS=0` (dev plan, section 6). `smoke.py` takes the model folder, `LRM_DATA/models/Qwen3-0.6B`. `inv.py` writes `inv_torch.txt` into the current directory.
