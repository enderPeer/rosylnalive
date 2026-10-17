# Lrm.Gpu.D3D12.Interop

D3D12 and DXGI COM interop for the D3D12 backend, plus the device check. Owner: R3. Contract: [D3D12ComIdiom](../../docs/contracts/d3d12-com-idiom.md). No ComImport, no RCWs, no third-party bindings.

| File | Contents |
| --- | --- |
| `AbiAttributes.fs` | `CStruct`, `CEnum` and `CArray`: the C names the ABI oracle checks |
| `D3D12.g.fs` | Generated: enums, structs, marker interfaces, `Iids`, `Slots`, `Fn` delegates |
| `Com.fs` | `ComPtr<'I>`, `Vtbl.get` (cached delegates), `Com` (IUnknown, live references, `check`), `D3D12Exception`, `HResult` |
| `D3D12Methods.g.fs` | Generated: typed extension methods per interface, with `…Raw` HRESULT forms |
| `Native.fs` | `D3D12CreateDevice`, `D3D12GetDebugInterface`, `CreateDXGIFactory2`, and their typed forms |
| `Desc.fs` | Description builders: buffers, heaps, raw UAVs, barriers, pipelines, query heaps |
| `DeviceCheck.fs` | Adapter discovery and the SM 6.6 / binding tier 3 verdict |

```fsharp
open Lrm.Gpu.D3D12.Interop

use factory = DxgiApi.createFactory<IDXGIFactory6> 0u
use adapter: ComPtr<IDXGIAdapter1> = factory.EnumAdapterByGpuPreference(0u, DXGI_GPU_PREFERENCE.HIGH_PERFORMANCE)
use dev = D3D12Api.createDevice<IDXGIAdapter1, ID3D12Device> adapter D3D_FEATURE_LEVEL.LEVEL_11_0
let qdesc = Desc.commandQueue D3D12_COMMAND_LIST_TYPE.COMPUTE
use queue: ComPtr<ID3D12CommandQueue> = dev.CreateCommandQueue(&qdesc)
```

Callers that pass `NativePtr.toNativeInt &&local` must build with `<Tailcalls>false</Tailcalls>` in their `.lrm.targets` (ADR R3-0001).
