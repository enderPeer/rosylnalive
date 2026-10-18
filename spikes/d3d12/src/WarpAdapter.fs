/// The in-box WARP adapter, found the way the plan names it (IDXGIFactory4::EnumWarpAdapter), with its capabilities
/// from R3's DeviceCheck.capabilities, the in-box DLL versions, the d3d12-warp verdict and its position in D1's
/// enumeration (IDXGIFactory6::EnumAdapterByGpuPreference), which D3D12Backend.Create needs.
module Lrm.Spikes.D3D12.WarpAdapter

#nowarn "9"
#nowarn "51"

open System
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12
open Lrm.Gpu.D3D12.Interop
open Microsoft.FSharp.NativeInterop

/// The PCI vendor id DXGI reports for WARP (Microsoft).
[<Literal>]
let VendorMicrosoft = 0x1414u

/// The device id of the Microsoft Basic Render Driver.
[<Literal>]
let DeviceBasicRender = 0x008Cu

/// The in-box files whose versions name the WARP build and the D3D12 runtime.
let inboxFiles = [ "d3d10warp.dll"; "d3d12.dll"; "D3D12Core.dll"; "dxgi.dll" ]

/// The debug layer's file, which the Graphics Tools optional feature installs (not in-box): when it is present, D1's
/// DeviceOptions.Debug turns the debug layer on for the process's first device.
let layerFiles = [ "D3D12SDKLayers.dll" ]

/// The file version of a System32 DLL, or "missing".
let fileVersion (name: string) =
    let path = Path.Combine(Environment.SystemDirectory, name)

    if File.Exists path then
        let v = FileVersionInfo.GetVersionInfo path

        sprintf "%d.%d.%d.%d" v.FileMajorPart v.FileMinorPart v.FileBuildPart v.FilePrivatePart
    else
        "missing"

/// Windows' build with its update revision, and its edition, from the registry (read only): "10.0.26200.9711" and
/// "Professional 25H2" here, "ServerDatacenter ..." on a runner. Environment.OSVersion has no revision.
let osBuild () =
    let v = Environment.OSVersion.Version

    try
        use key =
            Microsoft.Win32.Registry.LocalMachine.OpenSubKey @"SOFTWARE\Microsoft\Windows NT\CurrentVersion"

        let text (name: string) =
            match key.GetValue name with
            | null -> ""
            | x -> string x

        let ubr =
            match key.GetValue "UBR" with
            | :? int as u -> u
            | _ -> v.Revision

        sprintf "%d.%d.%d.%d" v.Major v.Minor v.Build ubr, (text "EditionID" + " " + text "DisplayVersion").Trim()
    with _ ->
        v.ToString(), ""

/// What EnumWarpAdapter returned and what a device on it supports.
type WarpReport =
    {
        /// The adapter as DeviceCheck describes adapters (Index: its position in EnumAdapterByGpuPreference order,
        /// matched by LUID; -1 when that enumeration does not list it).
        Adapter: AdapterIdentity
        Capabilities: Result<Capabilities, string>
        /// DLL name and file version ("missing" when absent), in `inboxFiles` then `layerFiles` order.
        Files: (string * string) list
        /// Windows' build with its update revision (osBuild).
        Os: string
        /// Windows' edition and release (osBuild).
        OsEdition: string
        /// The reasons the d3d12-warp class rejects the adapter (empty: accepted).
        Rejections: string list
    }

    member r.Accepted = r.Rejections.IsEmpty

/// The d3d12-warp verdict: WARP (a software adapter with Microsoft's vendor id), a device, shader model 6.6 and
/// resource binding tier 3 (the d3d12 requirements of GpuBackendInterface, with the hardware rule inverted). Pure, so
/// the cpu tier checks every rule.
let rejections (a: AdapterIdentity) (caps: Result<Capabilities, string>) : string list =
    [
        if not a.Software then
            "not a software adapter: d3d12-warp runs only on WARP"
        if a.VendorId <> VendorMicrosoft then
            sprintf "vendor %04x is not Microsoft (1414)" a.VendorId
        match caps with
        | Error e -> "no D3D12 device: " + e
        | Ok c ->
            if c.HighestShaderModel < DeviceCheck.RequiredShaderModel then
                Requirements.D3D12ShaderModel66

            if c.ResourceBindingTier < DeviceCheck.RequiredBindingTier then
                Requirements.D3D12BindingTier3
    ]

let private identity (index: int) (adapter: inref<ComPtr<IDXGIAdapter1>>) : AdapterIdentity =
    let mutable d = adapter.GetDesc1()
    let name = MemoryMarshal.CreateReadOnlySpan(&d.Description.e0, 128)

    let len =
        let k = name.IndexOf 0us
        if k < 0 then name.Length else k

    let chars = Array.zeroCreate<char> len

    for i in 0 .. len - 1 do
        chars[i] <- char name[i]

    let umd =
        let mutable iid = Iids.IDXGIDevice
        let mutable v = 0L

        let hr =
            adapter.CheckInterfaceSupportRaw(NativePtr.toNativeInt &&iid, NativePtr.toNativeInt &&v)

        if hr >= 0 then Some(DeviceCheck.umdParts v) else None

    {
        Index = index
        Description = String(chars)
        VendorId = d.VendorId
        DeviceId = d.DeviceId
        SubSysId = d.SubSysId
        Revision = d.Revision
        DedicatedVideoMemory = uint64 d.DedicatedVideoMemory
        DedicatedSystemMemory = uint64 d.DedicatedSystemMemory
        SharedSystemMemory = uint64 d.SharedSystemMemory
        Luid = (int64 d.AdapterLuid.HighPart <<< 32) ||| int64 d.AdapterLuid.LowPart
        Software = (d.Flags &&& uint32 DXGI_ADAPTER_FLAG.SOFTWARE) <> 0u
        UmdVersion = umd |> Option.map (Array.map string >> String.concat ".")
        DriverVersion =
            match umd with
            | Some parts -> DeviceCheck.formatDriverVersion d.VendorId parts
            | None -> "unknown"
    }

/// The position of the adapter with `luid` in EnumAdapterByGpuPreference(HIGH_PERFORMANCE) order, or -1.
let enumerationIndex (luid: int64) : int =
    use factory = DxgiApi.createFactory<IDXGIFactory6> 0u
    let mutable index = 0
    let mutable found = -1
    let mutable go = true

    while go do
        let mutable iid = Iids.IDXGIAdapter1
        let mutable out = 0n

        let hr =
            factory.EnumAdapterByGpuPreferenceRaw(
                uint32 index,
                DXGI_GPU_PREFERENCE.HIGH_PERFORMANCE,
                NativePtr.toNativeInt &&iid,
                NativePtr.toNativeInt &&out
            )

        if hr = HResult.DXGI_ERROR_NOT_FOUND then
            go <- false
        else
            Com.check "IDXGIFactory6::EnumAdapterByGpuPreference" hr factory.Ptr
            use adapter = ComPtr<IDXGIAdapter1>.Attach out
            let d = adapter.GetDesc1()
            let l = (int64 d.AdapterLuid.HighPart <<< 32) ||| int64 d.AdapterLuid.LowPart

            if l = luid && found < 0 then
                found <- index

            index <- index + 1

    found

/// EnumWarpAdapter, a device at feature level 11_0 and its capabilities, the in-box versions and the verdict. Records
/// gpu.device.check (backend d3d12-warp, with the facts the device check reports) and gpu.device.memory, inside a
/// gpu.device.discover span, as R3's DeviceCheck does for d3d12.
let probe () : WarpReport =
    let backend = Attr.str TelemetryKeys.Backend Relabel.To
    use span = Telemetry.span "gpu.device.discover" [ backend; Attr.str "via" "EnumWarpAdapter" ]
    use factory = DxgiApi.createFactory<IDXGIFactory6> 0u
    use adapter: ComPtr<IDXGIAdapter1> = factory.EnumWarpAdapter()
    let a0 = identity -1 &adapter
    let a = { a0 with Index = enumerationIndex a0.Luid }

    let caps =
        try
            use dev =
                D3D12Api.createDevice<IDXGIAdapter1, ID3D12Device> adapter DeviceCheck.CreationFeatureLevel

            Ok(DeviceCheck.capabilities dev)
        with
        | :? D3D12Exception as e -> Error e.Message
        | :? DllNotFoundException as e -> Error e.Message

    let os, edition = osBuild ()

    let report =
        {
            Adapter = a
            Capabilities = caps
            Files = inboxFiles @ layerFiles |> List.map (fun f -> f, fileVersion f)
            Os = os
            OsEdition = edition
            Rejections = rejections a caps
        }

    Telemetry.check
        "gpu.device.check"
        report.Accepted
        [
            backend
            Attr.str "device" a.Description
            Attr.int64 "vendor_id" (int64 a.VendorId)
            Attr.int64 "device_id" (int64 a.DeviceId)
            Attr.str "driver" a.DriverVersion
            Attr.str "luid" (Adapters.luidText a.Luid)
            Attr.int "dxgi_index" a.Index
            match caps with
            | Ok c ->
                Attr.str "shader_model" (DeviceCheck.formatShaderModel c.HighestShaderModel)
                Attr.int "binding_tier" (int c.ResourceBindingTier)
                Attr.str "feature_level" (DeviceCheck.formatFeatureLevel c.MaxFeatureLevel)
                Attr.str "root_signature" (sprintf "1.%d" (int c.HighestRootSignature - 1))
                Attr.str "wave_lanes" (sprintf "%d..%d" c.WaveLaneCountMin c.WaveLaneCountMax)

                match c.RuntimeSdkVersion with
                | Some v -> Attr.int64 "runtime_sdk" (int64 v)
                | None -> ()
            | Error e -> Attr.str "error" e
            for f, v in report.Files do
                Attr.str ("file_" + f.ToLowerInvariant().Replace(".dll", "").Replace('.', '_')) v
            Attr.str "os" report.Os
            Attr.str "os_edition" report.OsEdition
            if not report.Accepted then
                Attr.str "reason" (String.concat "; " report.Rejections)
        ]

    Telemetry.gauge
        "gpu.device.memory"
        (float a.SharedSystemMemory)
        TelemetryUnits.Bytes
        [ backend; Attr.str "device" a.Description; Attr.str "pool" "shared_system" ]

    span.Add(Attr.int "adapters" 1)
    span.Add(Attr.int "accepted" (if report.Accepted then 1 else 0))
    report

/// The d3d12 DeviceInfo D1's D3D12Backend.Create takes for the WARP adapter: Adapters.info over the report, so the
/// index, the LUID that Create checks and the details are D1's own.
let d3d12Info (r: WarpReport) : DeviceInfo =
    Adapters.info
        {
            Adapter = r.Adapter
            Capabilities = r.Capabilities
            Verdict = DeviceCheck.verdict r.Adapter r.Capabilities
        }

/// The d3d12-warp DeviceInfo: the d3d12 one with Backend = D3D12Warp, Index 0 (WARP is the class's only device),
/// and the DXGI index, the source and the in-box file versions added to Details.
let warpInfo (r: WarpReport) : DeviceInfo =
    let d = d3d12Info r

    { d with
        Backend = ExecutionBackend.D3D12Warp
        Index = 0
        Details =
            d.Details
            @ [
                "dxgi_index", Invariant.i32 r.Adapter.Index
                "source", "IDXGIFactory4::EnumWarpAdapter"
                "os", r.Os
                "os_edition", r.OsEdition
            ]
            @ (r.Files
               |> List.map (fun (f, v) -> "file_" + f.ToLowerInvariant().Replace(".dll", "").Replace('.', '_'), v))
    }
