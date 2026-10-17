/// List D3D12 adapters and the SM 6.6 / binding tier 3 verdict.
/// Owner: R3.
module Lrm.Cli.Commands.DevicesD3D12

open System
open System.Text.Json
open System.Text.Json.Nodes
open Lrm.Gpu.D3D12.Interop

let private mib (bytes: uint64) = bytes / 1048576UL

let private yesNo (b: bool) = if b then "yes" else "no"

let private printText (reports: AdapterReport list) =
    printfn "D3D12 adapters (IDXGIFactory6, high-performance order):"

    for r in reports do
        let a = r.Adapter

        printfn
            "  [%d] %s  vendor 0x%04X device 0x%04X  VRAM %d MiB  driver %s%s"
            a.Index
            a.Description
            a.VendorId
            a.DeviceId
            (mib a.DedicatedVideoMemory)
            a.DriverVersion
            (match a.UmdVersion with
             | Some u when u <> a.DriverVersion -> sprintf " (UMD %s)" u
             | _ -> "")

        match r.Capabilities with
        | Ok c ->
            printfn
                "      feature level %s, shader model %s, resource binding tier %d, heap tier %d, root signature 1.%d"
                (DeviceCheck.formatFeatureLevel c.MaxFeatureLevel)
                (DeviceCheck.formatShaderModel c.HighestShaderModel)
                (int c.ResourceBindingTier)
                (int c.ResourceHeapTier)
                (int c.HighestRootSignature - 1)

            printfn
                "      waves %d..%d, int64 %s, fp64 %s, native 16-bit %s, int64 atomics typed/groupshared/heap %s/%s/%s"
                c.WaveLaneCountMin
                c.WaveLaneCountMax
                (yesNo c.Int64ShaderOps)
                (yesNo c.DoublePrecisionShaderOps)
                (yesNo c.Native16BitShaderOps)
                (yesNo c.AtomicInt64OnTypedResource)
                (yesNo c.AtomicInt64OnGroupShared)
                (yesNo c.AtomicInt64OnDescriptorHeapResource)

            printfn
                "      enhanced barriers %s, UMA %s, GPU VA bits %d per resource / %d per process, nodes %d, runtime SDK %s"
                (yesNo c.EnhancedBarriers)
                (yesNo c.Uma)
                c.MaxGpuVirtualAddressBitsPerResource
                c.MaxGpuVirtualAddressBitsPerProcess
                c.NodeCount
                (match c.RuntimeSdkVersion with
                 | Some v -> string v
                 | None -> "unknown")
        | Error e -> printfn "      no D3D12 device: %s" e

        match r.Verdict with
        | Accepted -> printfn "      ACCEPTED"
        | Rejected reasons -> printfn "      REJECTED: %s" (String.concat "; " reasons)

let private opt (v: 'a option) (f: 'a -> JsonNode) : JsonNode = v |> Option.map f |> Option.toObj

let private toJson (reports: AdapterReport list) =
    let arr = JsonArray()

    for r in reports do
        let a = r.Adapter
        let o = JsonObject()
        o["index"] <- JsonValue.Create a.Index
        o["description"] <- JsonValue.Create a.Description
        o["vendorId"] <- JsonValue.Create a.VendorId
        o["deviceId"] <- JsonValue.Create a.DeviceId
        o["subSysId"] <- JsonValue.Create a.SubSysId
        o["revision"] <- JsonValue.Create a.Revision
        o["dedicatedVideoMemory"] <- JsonValue.Create a.DedicatedVideoMemory
        o["software"] <- JsonValue.Create a.Software
        o["umdVersion"] <- opt a.UmdVersion (fun v -> JsonValue.Create v)
        o["driverVersion"] <- JsonValue.Create a.DriverVersion

        match r.Capabilities with
        | Ok c ->
            let k = JsonObject()
            k["featureLevel"] <- JsonValue.Create(DeviceCheck.formatFeatureLevel c.MaxFeatureLevel)
            k["shaderModel"] <- JsonValue.Create(DeviceCheck.formatShaderModel c.HighestShaderModel)
            k["resourceBindingTier"] <- JsonValue.Create(int c.ResourceBindingTier)
            k["resourceHeapTier"] <- JsonValue.Create(int c.ResourceHeapTier)
            k["waveLaneCountMin"] <- JsonValue.Create c.WaveLaneCountMin
            k["waveLaneCountMax"] <- JsonValue.Create c.WaveLaneCountMax
            k["int64ShaderOps"] <- JsonValue.Create c.Int64ShaderOps
            k["native16BitShaderOps"] <- JsonValue.Create c.Native16BitShaderOps
            k["enhancedBarriers"] <- JsonValue.Create c.EnhancedBarriers
            k["runtimeSdkVersion"] <- opt c.RuntimeSdkVersion (fun v -> JsonValue.Create v)
            o["capabilities"] <- k
        | Error e -> o["error"] <- JsonValue.Create e

        match r.Verdict with
        | Accepted -> o["verdict"] <- JsonValue.Create "ACCEPTED"
        | Rejected reasons ->
            o["verdict"] <- JsonValue.Create "REJECTED"
            let rs = JsonArray()

            for x in reasons do
                rs.Add(JsonValue.Create x)

            o["reasons"] <- rs

        arr.Add o

    arr.ToJsonString(JsonSerializerOptions(WriteIndented = true))

/// Exit codes: 0 at least one adapter is accepted; 3 none is; 1 D3D12 or DXGI is unavailable; 2 usage error.
let private run (args: string[]) : int =
    let json =
        match args with
        | [||] -> Some false
        | [| "--json" |] -> Some true
        | _ -> None

    match json with
    | None ->
        eprintfn "usage: lrm devices-d3d12 [--json]"
        2
    | Some json ->
        try
            let reports = DeviceCheck.enumerate ()

            if json then
                printfn "%s" (toJson reports)
            else
                printText reports

            if reports |> List.exists (fun r -> r.Verdict = Accepted) then
                0
            else
                3
        with
        | :? DllNotFoundException as e ->
            eprintfn "lrm devices-d3d12: %s" e.Message
            1
        | :? D3D12Exception as e ->
            eprintfn "lrm devices-d3d12: %s" e.Message
            1

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "devices-d3d12"
            Summary = "List D3D12 adapters and the SM 6.6 / binding tier 3 verdict."
            Run = run
        |}
    ]
