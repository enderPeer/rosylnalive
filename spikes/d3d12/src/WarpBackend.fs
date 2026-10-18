/// A d3d12-warp backend made of D1's D3D12Backend, unchanged, on the WARP adapter. D1's Create accepts any adapter
/// of its enumeration (the device check, not Create, rejects WARP for d3d12), so the spike passes it the WARP
/// adapter's d3d12 DeviceInfo. The wrapper changes only what names the class: Info has Backend = D3D12Warp, the
/// ExecutionClass is GpuClass.fill over that Info (Arch keeps the CPU feature flags), and D1's records are relabelled
/// d3d12-warp (Relabel). Every call goes to D1's code with D1's checks, errors and texts.
module Lrm.Spikes.D3D12.WarpBackend

open System
open System.IO
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12

/// D1's heap view, with its telemetry relabelled.
[<Sealed>]
type WarpHeap(inner: IGpuHeap) =
    interface IGpuHeap with
        member _.Layout = inner.Layout
        member _.Table = inner.Table

        member _.WriteTable(t: SealedTable) =
            use _ = Relabel.install ()
            inner.WriteTable t

        member _.Upload(slot: int, offset: int64, data: ReadOnlySpan<byte>) =
            use _ = Relabel.install ()
            inner.Upload(slot, offset, data)

        member _.UploadFrom(slot: int, offset: int64, source: Stream, length: int64) =
            use _ = Relabel.install ()
            inner.UploadFrom(slot, offset, source, length)

        member _.Download(slot: int, offset: int64, destination: Span<byte>) =
            use _ = Relabel.install ()
            inner.Download(slot, offset, destination)

/// The d3d12-warp backend over D1's backend.
[<Sealed>]
type WarpBackend(inner: D3D12Backend, info: DeviceInfo) =
    let ib = inner :> IGpuBackend
    let mutable klass: ExecutionClass option = None

    do
        if info.Backend <> ExecutionBackend.D3D12Warp then
            invalidArg "info" "the WARP backend's DeviceInfo names d3d12-warp"

    /// D1's backend: its probe results, descriptor counts, removed reason and DRED report.
    member _.Inner = inner

    interface IGpuBackend with
        member _.Info = info

        member _.ExecutionClass =
            // D1's own checks first (after Dispose, before LoadKernels), then the d3d12-warp class.
            let d3d12 = ib.ExecutionClass

            match klass with
            | Some c -> c
            | None -> d3d12

        member _.CreateHeap(layout: HeapLayout) =
            use _ = Relabel.install ()
            WarpHeap(ib.CreateHeap layout) :> IGpuHeap

        member _.LoadKernels(shaders: ShaderSet) =
            use _ = Relabel.install ()
            let set = ib.LoadKernels shaders
            klass <- Some(GpuClass.fill info shaders)
            set

        member _.Begin() = ib.Begin()

        member _.Submit(commands: ICommandList) =
            use _ = Relabel.install ()
            ib.Submit commands

        member _.Wait(s: Submission) =
            use _ = Relabel.install ()
            ib.Wait s

        member _.Dispose() =
            use _ = Relabel.install ()
            ib.Dispose()

/// The d3d12-warp provider: one device, WARP from EnumWarpAdapter, accepted under the d3d12-warp verdict. It is what
/// D1's ADR asks for ("a provider that enumerates the software adapter and a class with Backend = D3D12Warp"), built
/// outside D1's paths.
[<Sealed>]
type WarpProvider() =
    let report = lazy (WarpAdapter.probe ())

    member _.Report = report.Value

    interface IGpuBackendProvider with
        member _.Backend = ExecutionBackend.D3D12Warp

        member _.Enumerate() =
            let r = report.Value

            [
                {
                    Info = WarpAdapter.warpInfo r
                    Rejections = r.Rejections
                }
            ]

        member _.Create(device: DeviceInfo, options: DeviceOptions) =
            let r = report.Value
            let expected = WarpAdapter.warpInfo r

            if device <> expected then
                invalidArg "device" (sprintf "%s is not this provider's WARP device" device.Name)

            if r.Adapter.Index < 0 then
                invalidOp "EnumAdapterByGpuPreference does not list the WARP adapter, so D3D12Backend.Create cannot open it"

            let d3d12 = WarpAdapter.d3d12Info r

            let inner =
                use _ = Relabel.install ()

                D3D12Backend.Create(
                    d3d12,
                    { options with
                        Backend = ExecutionBackend.D3D12
                    }
                )

            new WarpBackend(inner, expected) :> IGpuBackend

/// A d3d12-warp backend, through a private registry holding only the WARP provider (as D3D12.tryCreate does for
/// d3d12).
let tryCreate (provider: WarpProvider) (options: DeviceOptions) : Result<IGpuBackend, string> =
    let registry = BackendRegistry()
    registry.Register provider

    registry.TryCreate
        { options with
            Backend = ExecutionBackend.D3D12Warp
        }
