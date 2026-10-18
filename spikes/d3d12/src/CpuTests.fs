/// cpu tier: what the spike decides without a device. The affinity setting; the relabelling facade (only `backend =
/// d3d12` changes, through spans, children and every record kind); the d3d12-warp verdict; the d3d12-warp DeviceInfo
/// and class (Arch keeps the CPU feature flags, so the id differs from a d3d12 class of the same device); the wrapper's
/// refusal of a d3d12 Info; and the control shader set (only scale_f32 and the DXIL flags change).
module Lrm.Spikes.D3D12.CpuTests

open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop

let private adapter (software: bool) (vendor: uint32) : AdapterIdentity =
    {
        Index = 1
        Description = "Microsoft Basic Render Driver"
        VendorId = vendor
        DeviceId = WarpAdapter.DeviceBasicRender
        SubSysId = 0u
        Revision = 0u
        DedicatedVideoMemory = 0UL
        DedicatedSystemMemory = 0UL
        SharedSystemMemory = 17179869184UL
        Luid = 0x0000000000012345L
        Software = software
        UmdVersion = Some "10.0.26100.9711"
        DriverVersion = "10.0.26100.9711"
    }

let private caps (sm: D3D_SHADER_MODEL) (tier: D3D12_RESOURCE_BINDING_TIER) : Capabilities =
    {
        MaxFeatureLevel = D3D_FEATURE_LEVEL.LEVEL_12_1
        HighestShaderModel = sm
        ResourceBindingTier = tier
        ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER.TIER_2
        HighestRootSignature = D3D_ROOT_SIGNATURE_VERSION.VERSION_1_1
        DoublePrecisionShaderOps = true
        WaveOps = true
        WaveLaneCountMin = 4u
        WaveLaneCountMax = 4u
        Int64ShaderOps = true
        Native16BitShaderOps = true
        AtomicInt64OnTypedResource = true
        AtomicInt64OnGroupShared = true
        AtomicInt64OnDescriptorHeapResource = true
        EnhancedBarriers = true
        Uma = true
        CacheCoherentUma = true
        MaxGpuVirtualAddressBitsPerResource = 40u
        MaxGpuVirtualAddressBitsPerProcess = 40u
        NodeCount = 1u
        RuntimeSdkVersion = Some 616u
    }

let private good =
    caps D3D_SHADER_MODEL.SHADER_MODEL_6_8 D3D12_RESOURCE_BINDING_TIER.TIER_3

let private report: WarpAdapter.WarpReport =
    {
        Adapter = adapter true WarpAdapter.VendorMicrosoft
        Capabilities = Ok good
        Files = [ "d3d10warp.dll", "10.0.26100.9711"; "D3D12Core.dll", "10.0.26100.9549" ]
        Os = "10.0.26200.9711"
        OsEdition = "Professional 25H2"
        Rejections = []
    }

let private affinity =
    testList
        "affinity"
        [
            testCase "LRM_WARP_CPUS: unset is the runner's 4, 'all' keeps the process, counts 1 to 64"
            <| fun () ->
                Expect.equal (Affinity.parse None) (Ok(Some 4)) "default"
                Expect.equal (Affinity.parse (Some "all")) (Ok None) "all"
                Expect.equal (Affinity.parse (Some " ALL ")) (Ok None) "case and blanks"
                Expect.equal (Affinity.parse (Some "8")) (Ok(Some 8)) "8"
                Expect.isError (Affinity.parse (Some "0")) "0"
                Expect.isError (Affinity.parse (Some "65")) "65"
                Expect.isError (Affinity.parse (Some "four")) "text"

            testCase "the mask keeps the first n processors the process already has"
            <| fun () ->
                Expect.equal (Affinity.mask 4 0xFFFFFFFFL) 0xFL "4 of 32"
                Expect.equal (Affinity.mask 2 0b10110L) 0b00110L "the first two set bits"
                Expect.equal (Affinity.mask 40 0xFFL) 0xFFL "more than there are"
                Expect.equal (Affinity.mask 64 -1L) -1L "all 64"
        ]

let private relabel =
    testList
        "relabel"
        [
            testCase "only a backend attribute naming d3d12 changes"
            <| fun () ->
                let a = Relabel.attr (Attr.str TelemetryKeys.Backend "d3d12")
                Expect.equal a (Attr.str TelemetryKeys.Backend "d3d12-warp") "d3d12 becomes d3d12-warp"

                for x in
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str TelemetryKeys.Backend "cpu-device"
                        Attr.str "device" "d3d12"
                        Attr.int TelemetryKeys.Backend 12
                    ] do
                    Expect.equal (Relabel.attr x) x (sprintf "%A is unchanged" x)

            testCase "every record kind, spans, span attributes and children are relabelled; values are not"
            <| fun () ->
                let sink = InMemorySink()
                let b = Attr.str TelemetryKeys.Backend "d3d12"

                do
                    use _ = Telemetry.install (Telemetry.ofSink sink)
                    use _ = Relabel.install ()
                    // Installing twice wraps once.
                    use _ = Relabel.install ()
                    Expect.isTrue ((Telemetry.current ()) :? Relabel.RelabelTelemetry) "the relabelling facade"

                    Expect.isFalse
                        ((Telemetry.current () :?> Relabel.RelabelTelemetry).Inner :? Relabel.RelabelTelemetry)
                        "not wrapped twice"

                    use span = Telemetry.span "gpu.wait" [ b ]
                    span.Add b
                    Telemetry.counter "gpu.dispatches" 9.0 TelemetryUnits.Count [ b ]
                    Telemetry.gauge "mem.arena" 64.0 TelemetryUnits.Bytes [ b; Attr.str TelemetryKeys.Arena "table" ]
                    Telemetry.histogram "gpu.lat" 1.5 TelemetryUnits.Nanoseconds [ b ]
                    Telemetry.event "gpu.fp32_denormals" [ b; Attr.str "mode" "preserve" ]
                    Telemetry.gpuInterval "gpu.dispatch" 10L 20L [ b; Attr.str TelemetryKeys.Kernel "scale_f32" ]
                    Telemetry.meter "gpu.bytes_moved" 128.0 TelemetryUnits.Bytes [ b ]
                    Telemetry.check "gpu.selftest" true [ b ]

                    use _ = Telemetry.scope [ b; Attr.str TelemetryKeys.Case "probe" ]
                    Telemetry.check "gpu.selftest" false []

                let records = sink.Records
                Expect.equal records.Length 9 "eight point records and one span"

                for r in records do
                    let backends =
                        r.Attrs
                        |> List.filter (fun a -> a.Key = TelemetryKeys.Backend)
                        |> List.map _.Value

                    Expect.isNonEmpty backends (r.Name + " has a backend")

                    Expect.all
                        backends
                        (fun v -> v = AttrValue.Str "d3d12-warp")
                        (r.Name + ": every backend attribute is d3d12-warp")

                let value name =
                    records |> List.find (fun r -> r.Name = name) |> _.Value

                Expect.equal (value "gpu.dispatches") 9.0 "counter value"
                Expect.equal (value "mem.arena") 64.0 "gauge value"
                Expect.equal (value "gpu.dispatch") 20.0 "gpu duration"

                let interval = records |> List.find (fun r -> r.Name = "gpu.dispatch")
                Expect.equal interval.GpuStartNs 10L "gpu start"

                Expect.contains
                    interval.Attrs
                    (Attr.str TelemetryKeys.Kernel "scale_f32")
                    "other attributes are kept"

            testCase "a no-op facade stays a no-op"
            <| fun () ->
                use _ = Relabel.install ()
                Expect.isFalse (Telemetry.current ()).Enabled "no-op"
        ]

let private verdict =
    testList
        "d3d12-warp verdict"
        [
            testCase "WARP with SM 6.6+ and binding tier 3 is accepted"
            <| fun () ->
                Expect.isEmpty (WarpAdapter.rejections (adapter true WarpAdapter.VendorMicrosoft) (Ok good)) "SM 6.8"

                let sm66 =
                    caps D3D_SHADER_MODEL.SHADER_MODEL_6_6 D3D12_RESOURCE_BINDING_TIER.TIER_3

                Expect.isEmpty (WarpAdapter.rejections (adapter true WarpAdapter.VendorMicrosoft) (Ok sm66)) "SM 6.6"

            testCase "hardware, another vendor, no device, SM 6.5 and tier 2 are rejected, each with its reason"
            <| fun () ->
                let reasons a c = WarpAdapter.rejections a c

                Expect.equal
                    (reasons (adapter false 0x10DEu) (Ok good))
                    [
                        "not a software adapter: d3d12-warp runs only on WARP"
                        "vendor 10de is not Microsoft (1414)"
                    ]
                    "a GPU"

                Expect.equal
                    (reasons (adapter true WarpAdapter.VendorMicrosoft) (Error "E_FAIL"))
                    [ "no D3D12 device: E_FAIL" ]
                    "no device"

                Expect.equal
                    (reasons
                        (adapter true WarpAdapter.VendorMicrosoft)
                        (Ok(caps D3D_SHADER_MODEL.SHADER_MODEL_6_5 D3D12_RESOURCE_BINDING_TIER.TIER_2)))
                    [ Requirements.D3D12ShaderModel66; Requirements.D3D12BindingTier3 ]
                    "an old WARP"
        ]

let private klass =
    testList
        "d3d12-warp class"
        [
            testCase "the Info is D1's with Backend d3d12-warp, index 0 and the WARP facts; the class keeps the ISA flags"
            <| fun () ->
                let d3d12 = WarpAdapter.d3d12Info report
                let warp = WarpAdapter.warpInfo report
                Expect.equal d3d12.Backend ExecutionBackend.D3D12 "D1's Info names d3d12"
                Expect.equal d3d12.Index 1 "at the adapter's DXGI position"
                Expect.equal warp.Backend ExecutionBackend.D3D12Warp "d3d12-warp"
                Expect.equal warp.Index 0 "WARP is the class's only device"
                Expect.equal warp.Driver "10.0.26100.9711" "the Driver is WARP's UMD version"
                Expect.contains warp.Details ("dxgi_index", "1") "the DXGI index"
                Expect.contains warp.Details ("source", "IDXGIFactory4::EnumWarpAdapter") "the source"
                Expect.contains warp.Details ("file_d3d10warp", "10.0.26100.9711") "the WARP DLL"
                Expect.contains warp.Details ("luid", "0000000000012345") "the LUID D1's Create checks"
                let set = Lrm.Shaders.Shaders.embedded ()
                let w = GpuClass.fill warp set
                let d = GpuClass.fill d3d12 set
                Expect.equal w.Backend ExecutionBackend.D3D12Warp "class backend"
                Expect.equal w.DeviceId "pci:1414:008c" "DeviceId"
                Expect.equal w.Arch (ExecutionClass.hostArch ()) "Arch keeps the CPU feature flags"
                Expect.equal w.ShaderSetHash d.ShaderSetHash "the same DXIL set"
                Expect.notEqual (ExecutionClass.id w) (ExecutionClass.id d) "a different class from d3d12"

            testCase "the wrapper refuses an Info that does not name d3d12-warp"
            <| fun () ->
                Expect.throwsT<System.ArgumentException>
                    (fun () ->
                        new WarpBackend.WarpBackend(Unchecked.defaultof<_>, WarpAdapter.d3d12Info report)
                        |> ignore)
                    "a d3d12 Info"
        ]

let private control =
    testList
        "denormal control"
        [
            testCase "the control set changes scale_f32's DXIL blob and the DXIL flags only"
            <| fun () ->
                let set = Lrm.Shaders.Shaders.embedded ()
                let flags = Control.stripDenorm (ShaderSet.flagsFor ShaderTarget.Dxil set)
                Expect.isFalse (List.contains "-denorm" flags) "no -denorm"

                Expect.equal
                    (Control.stripDenorm [ "-T"; "cs_6_6"; "-denorm"; "preserve"; "-WX" ])
                    [ "-T"; "cs_6_6"; "-WX" ]
                    "strip"

                let blob = [| 1uy; 2uy; 3uy |]
                let c = Control.withScale set flags blob
                Expect.equal c.CatalogHash set.CatalogHash "the catalog"
                Expect.equal c.RootSignature set.RootSignature "the root signature"
                Expect.equal (ShaderSet.flagsFor ShaderTarget.Spirv c) (ShaderSet.flagsFor ShaderTarget.Spirv set) "SPIR-V flags"

                for a, b in List.zip set.Blobs c.Blobs do
                    if a.Kernel = "scale_f32" && a.Target = ShaderTarget.Dxil then
                        Expect.equal b.Bytes blob "the control blob"
                        Expect.equal b.Sha256 (Hash.sha256 blob) "its SHA-256"
                    else
                        Expect.equal b a (sprintf "%s %A is unchanged" a.Kernel a.Target)

                Expect.notEqual
                    (ShaderSet.setHash ShaderTarget.Dxil c)
                    (ShaderSet.setHash ShaderTarget.Dxil set)
                    "the DXIL ShaderSetHash differs, so the class differs"
        ]

/// The spike passes stack addresses to the interop, so it builds with Tailcalls=false like every GPU project (ADR
/// R3-0001); the scan proves the setting took effect.
let private tailCalls =
    testCase "the spike's assembly has no tail. call sites"
    <| fun () ->
        let asm = typeof<Relabel.RelabelTelemetry>.Assembly

        if Lrm.Testing.TailCalls.builtWithoutOptimization asm then
            skiptest Lrm.Testing.TailCalls.UnoptimizedBuild

        Expect.isEmpty (Lrm.Testing.TailCalls.scan asm.Location) "no tail. prefixes"

let tests =
    testList "warp.cpu" [ affinity; relabel; verdict; klass; control; tailCalls ]
