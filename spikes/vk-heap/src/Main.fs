/// R1 spike entry point: an Expecto exe on Lrm.Testing's runner, so GPU groups hold the per-backend mutex.
///   LRM_TEST_TIER=cpu dotnet artifacts/bin/VkHeapSpike/release/VkHeapSpike.dll   (layouts, DXC, SPIR-V, CPU refs)
///   LRM_TEST_TIER=gpu dotnet artifacts/bin/VkHeapSpike/release/VkHeapSpike.dll   (d3d12 group, then vulkan group)
/// The gpu tier writes spikes/vk-heap/results/{results.md,dispatch-gpu-us.svg} and a run log under LRM_DATA/runs.
module VkHeapSpike.Main

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.IO
open Expecto
open Lrm.Core
open Lrm.Testing
open VkHeapSpike.Cases
open VkHeapSpike.Shaders
open VkHeapSpike.VkRunner

let shaderDir = Path.Combine(AppContext.BaseDirectory, "shaders")
let shaders = lazy (compileAll shaderDir)

/// Outcomes per configuration label, in run order.
let results = ConcurrentDictionary<string, CaseOutcome list>()
let order = ConcurrentQueue<string>()
let notes = ConcurrentQueue<string>()

let record (label: string) (o: CaseOutcome) =
    if not (results.ContainsKey label) then
        order.Enqueue label

    results.AddOrUpdate(label, [ o ], (fun _ xs -> xs @ [ o ])) |> ignore

let cases = lazy (Cases.all ())

let attrsOf (label: string) =
    let parts = label.Split('/')

    [
        Attr.str TelemetryKeys.Backend parts[0]
        if parts.Length > 1 then
            Attr.str "variant" parts[1]
        if parts.Length > 2 then
            Attr.str "placement" parts[2]
    ]

/// Runs one case on `gpu`, emits telemetry, records the outcome and fails on a gating mismatch.
let runAndCheck (gpu: Heap.IGpu) (c: Case) =
    let o = runCase gpu c
    let a = attrsOf gpu.Label @ [ Attr.str TelemetryKeys.Kernel c.Kernel; Attr.str TelemetryKeys.Case c.Name ]

    for ns in o.GpuNs do
        Telemetry.gpuInterval "gpu.dispatch" 0L (int64 ns) a

    let bytes =
        (c.Job.Uploads |> List.sumBy (fun u -> int64 u.Data.Length))
        + (c.Job.Readbacks |> List.sumBy (fun r -> int64 r.Length))

    Telemetry.meter "gpu.bytes_moved" (float (2L * bytes)) TelemetryUnits.Bytes a

    Telemetry.check
        "gpu.spike.case"
        (o.Mismatches = 0 && o.RunTwiceIdentical)
        (a
         @ [
             Attr.bool "gating" c.Gating
             Attr.int "mismatches" o.Mismatches
             Attr.bool "run_twice_identical" o.RunTwiceIdentical
             Attr.str "digest" o.Digest
         ])

    record gpu.Label o

    if not o.RunTwiceIdentical then
        failwithf "%s %s: run 1 and run 2 differ" gpu.Label c.Name

    if c.Gating && o.Mismatches <> 0 then
        failwithf
            "%s %s: %d words differ from the CPU reference; first (got/expected) %s"
            gpu.Label
            c.Name
            o.Mismatches
            (describeMismatches o.FirstMismatches)

    if not c.Gating then
        notes.Enqueue(
            sprintf
                "%s %s (informational): %d of %d words differ from the CPU; first (got/expected) %s"
                gpu.Label
                c.Name
                o.Mismatches
                (c.Expected |> List.sumBy (fun e -> e.Length / 4))
                (describeMismatches o.FirstMismatches)
        )

let caseTestsWhere (keep: Case -> bool) (label: string) (gpu: Lazy<#Heap.IGpu>) =
    [
        for c in cases.Force() do
            if keep c then
                yield testCase (sprintf "%s %s" label c.Name) (fun () -> runAndCheck (gpu.Force()) c)
    ]

let caseTests label gpu = caseTestsWhere (fun _ -> true) label gpu

// ---------------- cpu tier ----------------

let findVcvars () =
    [
        @"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        @"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        @"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ]
    |> List.tryFind File.Exists

/// Shader observations that also feed the gpu-tier report (they use DXC, not the GPU).
let shaderFactTests =
    [
        testCase "spirv: the emulated variant aliases one variable per heap use, all at set 0 binding 0" (fun () ->
            for k in kernels do
                let f = inspect (shaders.Force().Spirv[(Emulated, k)])
                Expect.isGreaterThanOrEqual f.Resources.Length 1 (k + ": at least one variable")

                for r in f.Resources do
                    Expect.equal (r.Set, r.Binding) (Some 0u, Some 0u) (sprintf "%s %s: set 0 binding 0" k r.Name)
                    Expect.equal r.StorageClass 12u (k + ": StorageBuffer")

                notes.Enqueue(sprintf "spirv %s emulated: %d aliased variables at set 0 binding 0" k f.Resources.Length))

        testCase "dxc: -fspv-use-descriptor-heap (direct heap indexing) is recorded, not used" (fun () ->
            // The fixed decision is the set/binding mapping path; this only records what the pinned DXC does.
            let dir = Path.Combine(shaderDir, "native")

            let outcome =
                try
                    let args =
                        [ "-E"; "main" ]
                        @ common
                        @ [
                            "-spirv"
                            "-fspv-target-env=vulkan1.3"
                            "-D"
                            "LRM_EMULATED_HEAP"
                            "-fspv-use-descriptor-heap"
                        ]

                    let f = inspect (compile dir "cs_6_6" args "scale_f32.hlsl" "scale_f32.native.spv")
                    sprintf "compiles (capabilities %A, extensions %A)" f.Capabilities f.Extensions
                with e ->
                    let first =
                        e.Message.Split('\n')
                        |> Array.tryFind (fun l -> l.Contains "error:")
                        |> Option.defaultValue e.Message

                    // Keep the message, not the local path of the copied source.
                    let msg = first.Trim()

                    let msg =
                        match msg.IndexOf "error:" with
                        | i when i >= 0 -> msg.Substring i
                        | _ -> msg

                    "does not compile: " + msg

            notes.Enqueue("dxc -fspv-use-descriptor-heap on scale_f32 (RWByteAddressBuffer): " + outcome))

        testCase "dxc: a flat uint d[32] ArgBlock would not match across backends (why the block is uint4 d[8])" (fun () ->
            let dir = Path.Combine(shaderDir, "probe")
            compile dir "cs_6_6" dxilArgs "flat_args_probe.hlsl" "flat_args_probe.dxil" |> ignore
            let listing = File.ReadAllText(Path.Combine(dir, "flat_args_probe.dxil.txt"))
            Expect.stringContains listing "Size:   500" "DXIL legacy cbuffer packing: 31 * 16 + 4 = 500 bytes"
            let f = inspect (compile dir "cs_6_6" (spirvArgs Explicit) "flat_args_probe.hlsl" "flat_args_probe.spv")
            Expect.equal f.PushArrayStride (Some 4u) "SPIR-V push constant: ArrayStride 4"
            Expect.equal f.PushBlockBytes (Some 128u) "SPIR-V push constant: 128 bytes"
            notes.Enqueue "dxc: uint d[32] in the ArgBlock is a 500-byte legacy cbuffer on DXIL but 128 bytes (stride 4) on SPIR-V; uint4 d[8] is 128 bytes, stride 16, on both")
    ]

let cpuTests =
    shaderFactTests
    @ [
        testCase "layout: every hand-written Vulkan struct matches its C declaration" (fun () ->
            match Vk.layoutMismatches () with
            | [] -> ()
            | errors -> failwithf "layout mismatches:\n%s" (String.concat "\n" errors))

        testCase "layout: cl.exe agrees with the C layout calculator" (fun () ->
            match findVcvars () with
            | None -> skiptest "Visual Studio 2022 Build Tools (vcvars64.bat) not found"
            | Some vcvars ->
                let dir = Path.Combine(AppContext.BaseDirectory, "abi")
                Directory.CreateDirectory dir |> ignore
                let c = Path.Combine(dir, "layout_check.c")
                File.WriteAllText(c, Layout.cSource ())
                // A .cmd file sidesteps cmd.exe's quoting rules for the vcvars path.
                let script = Path.Combine(dir, "layout_check.cmd")

                File.WriteAllText(
                    script,
                    sprintf
                        "@echo off
call \"%s\" >nul || exit /b 9
cl /nologo /c /W4 /WX /std:c11 /Fo\"%s\" \"%s\"
"
                        vcvars
                        (Path.Combine(dir, "layout_check.obj"))
                        c
                )

                let psi = ProcessStartInfo("cmd.exe")
                psi.ArgumentList.Add "/c"
                psi.ArgumentList.Add script
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                use p = Process.Start psi
                let out = p.StandardOutput.ReadToEnd() + p.StandardError.ReadToEnd()
                p.WaitForExit()

                if p.ExitCode <> 0 then
                    failwithf "cl.exe rejected the generated static_asserts (exit %d):\n%s" p.ExitCode out)

        testCase "dxc: the pinned build compiles every kernel, both SPIR-V variants and the root signature" (fun () ->
            let s = shaders.Force()
            Expect.equal s.Dxil.Count 4 "4 DXIL kernels"
            Expect.equal s.Spirv.Count 8 "4 kernels x 2 SPIR-V variants"
            Expect.equal s.RootSignature.Length 92 "rootsig_1_1 blob is 92 bytes"

            for KeyValue(k, b) in s.Dxil do
                Expect.equal (Text.Encoding.ASCII.GetString(b, 0, 4)) "DXBC" (k + " is a DXBC container"))

        testCase "spirv: the explicit variant has one resource variable, at set 0 binding 0" (fun () ->
            for k in kernels do
                let f = inspect (shaders.Force().Spirv[(Explicit, k)])

                match f.Resources with
                | [ r ] ->
                    Expect.equal (r.Set, r.Binding) (Some 0u, Some 0u) (k + ": set 0 binding 0")
                    Expect.equal r.StorageClass 12u (k + ": StorageBuffer")
                    Expect.isFalse r.NonWritable (k + ": not NonWritable")
                | rs -> failwithf "%s: expected one resource variable, got %A" k rs)

        testCase "spirv: the push block is 128 bytes at offset 0 with a 16-byte stride, in both variants" (fun () ->
            for v in [ Explicit; Emulated ] do
                for k in kernels do
                    let f = inspect (shaders.Force().Spirv[(v, k)])
                    let name = sprintf "%s %s" k (variantName v)
                    Expect.equal f.PushMemberOffsets [ 0u ] (name + ": one member at offset 0")
                    Expect.equal f.PushArrayStride (Some 16u) (name + ": ArrayStride 16")
                    Expect.equal f.PushBlockBytes (Some 128u) (name + ": 128 bytes")
                    let expectedX = if k = "args_echo" then 32u else 256u
                    Expect.equal f.LocalSize (expectedX, 1u, 1u) (name + ": LocalSize")
                    Expect.contains f.Capabilities 5302u (name + ": RuntimeDescriptorArray capability"))

        testCase "recipe KAT: the set 0 binding 0 mapping, byte for byte" (fun () ->
            // docs/contracts/vulkan-heap-recipe.md, KAT-1. pNext and pEmbeddedSampler are NULL.
            let expected =
                "5dd99c3b000000000000000000000000000000000000000001000000c0000000"
                + "0000000000000000000000001000000000000000000000000000000000000000"
                + "0000000000000000000000000000000000000000000000000000000000000000"

            Expect.equal (Convert.ToHexStringLower(structBytes (recipeMapping 0UL 16UL))) expected "mapping bytes")

        testCase "recipe KAT: heap geometry for the RTX 4080 limits" (fun () ->
            // docs/contracts/vulkan-heap-recipe.md, KAT-2.
            let limits: Lrm.Gpu.Vulkan.DeviceCheck.DescriptorHeapLimits =
                {
                    SamplerHeapAlignment = 0UL
                    ResourceHeapAlignment = 64UL
                    MaxSamplerHeapSize = 131072UL
                    MaxResourceHeapSize = 33554432UL
                    MinSamplerHeapReservedRange = 0UL
                    MinSamplerHeapReservedRangeWithEmbedded = 0UL
                    MinResourceHeapReservedRange = 96768UL
                    SamplerDescriptorSize = 32UL
                    ImageDescriptorSize = 32UL
                    BufferDescriptorSize = 16UL
                    SamplerDescriptorAlignment = 32UL
                    ImageDescriptorAlignment = 32UL
                    BufferDescriptorAlignment = 8UL
                    MaxPushDataSize = 256UL
                    ImageCaptureReplayOpaqueDataSize = 0UL
                    MaxDescriptorHeapEmbeddedSamplers = 0u
                    SamplerYcbcrConversionCount = 0u
                    SparseDescriptorHeaps = false
                    ProtectedDescriptorHeaps = false
                }

            let g = geometry limits
            Expect.equal (g.HeapOffset, g.Stride, g.SlotBytes) (0UL, 16UL, 1024UL) "slots"
            Expect.equal (g.ReservedOffset, g.ReservedSize, g.HeapBytes) (1024UL, 96768UL, 97792UL) "reserved range")

        testCase "cpu refs: known answers" (fun () ->
            Expect.equal (scaleRef [| 2.0f; -4.0f |] 0.5f) [| 1.0f; -2.0f |] "scale"
            Expect.equal (Bf16.widenBits 0x3F80us) 0x3F800000u "bf16 1.0"
            Expect.equal (Bf16.widenBits 0xFFC1us) 0xFFC10000u "bf16 NaN payload kept"
            let t = Native.words Heap.table
            Expect.equal t[16 * Heap.Tok.ScaleY] 2u "ScaleY row: slot 2"
            Expect.equal t[16 * Heap.Tok.ScaleY + 1] (uint32 (4L * Heap.MiB + 256L)) "ScaleY row: byteOffset"
            Expect.equal t[16 * Heap.Tok.Bf16In + 2] 2u "Bf16In row: dtype BF16 = 2"
            Expect.equal (Array.sub t 0 16) (Array.zeroCreate 16) "token 0 is the null row"

            for x in normalInputs 1UL 4096 do
                let y = x * ScaleFactor
                let e = (BitConverter.SingleToUInt32Bits y >>> 23) &&& 0xFFu
                Expect.isTrue (e > 0u && e < 255u) "scale inputs give normal products")
    ]

// ---------------- gpu tier ----------------

let d3d12 = lazy (new D3D12Runner.D3D12Gpu(shaders.Force()))

let d3d12Tests =
    testSequenced
    <| testList
        "d3d12"
        [
            testCase "d3d12: device, root signature, PSOs and the 64-slot heap" (fun () ->
                let f = d3d12.Force().Facts
                Devices.record Backend.D3D12 f.Adapter f.Driver (Some(shaders.Force().Hash.Hex))

                Telemetry.event
                    "gpu.device"
                    [
                        Attr.str TelemetryKeys.Backend "d3d12"
                        Attr.str "device" f.Adapter
                        Attr.str "driver" f.Driver
                        Attr.int "shader_model" f.HighestShaderModel
                        Attr.int "binding_tier" f.ResourceBindingTier
                        Attr.int "descriptor_increment" (int f.DescriptorIncrement)
                    ]

                notes.Enqueue(
                    sprintf
                        "d3d12: %s, UMD %s, highest SM 0x%x, binding tier %d, CBV_SRV_UAV increment %d B, timestamp %d Hz"
                        f.Adapter
                        f.Driver
                        f.HighestShaderModel
                        f.ResourceBindingTier
                        f.DescriptorIncrement
                        f.TimestampFrequency
                ))
            yield! caseTests "d3d12" d3d12
            testCase "d3d12: teardown" (fun () ->
                if d3d12.IsValueCreated then
                    (d3d12.Value :> IDisposable).Dispose())
        ]

let vkContext = lazy (new VulkanContext())

let vkConfigs =
    [
        for v in [ Explicit; Emulated ] do
            for p in [ Bar; HostMemory; CopyToDeviceLocal ] -> v, p
    ]

let vkConfigLazies =
    vkConfigs
    |> List.map (fun (v, p) -> (v, p), lazy (new VulkanConfig(vkContext.Force(), shaders.Force(), v, p, None)))
    |> Map.ofList

/// Mapping controls: a non-zero heapOffset, and a stride twice the descriptor size (heap written to match). If the
/// driver ignored either value, slot i would resolve to another descriptor and the cases would fail.
let vkControlLazies =
    [
        { HeapOffset = 512UL; Stride = 16UL; Suffix = "offset512" }
        { HeapOffset = 0UL; Stride = 32UL; Suffix = "stride32" }
    ]
    |> List.map (fun c ->
        c, lazy (new VulkanConfig(vkContext.Force(), shaders.Force(), Explicit, Bar, Some c)))

let observations = ConcurrentDictionary<string, HeapObservation>()

let vulkanTests =
    testSequenced
    <| testList
        "vulkan"
        [
            yield! shaderFactTests

            testCase "vulkan: struct layouts match their C declarations" (fun () ->
                Expect.isEmpty (Vk.layoutMismatches ()) "layouts")

            testCase "vulkan: device, features, queue, memory types and heap properties" (fun () ->
                let f = vkContext.Force().Facts
                Devices.record Backend.Vulkan f.Name f.Driver (Some(shaders.Force().Hash.Hex))
                let g = geometry f.Heap
                Expect.isGreaterThanOrEqual f.Heap.MaxPushDataSize 128UL "maxPushDataSize >= 128"
                Expect.equal f.StorageBufferDescriptorSize f.Heap.BufferDescriptorSize "descriptor size query agrees"
                Expect.equal (g.ReservedOffset % f.Heap.BufferDescriptorAlignment) 0UL "reserved offset alignment (buffer)"
                Expect.equal (g.ReservedOffset % f.Heap.ImageDescriptorAlignment) 0UL "reserved offset alignment (image)"
                Expect.isLessThanOrEqual g.HeapBytes f.Heap.MaxResourceHeapSize "heap fits maxResourceHeapSize"
                Expect.isGreaterThanOrEqual (uint64 f.MaxStorageBufferRange) (uint64 Heap.GiB) "1 GiB page fits maxStorageBufferRange"

                Telemetry.event
                    "gpu.device"
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str "device" f.Name
                        Attr.str "driver" f.Driver
                        Attr.str "api" f.ApiVersion
                        Attr.int "queue_family" f.QueueFamily
                        Attr.int64 "buffer_descriptor_size" (int64 f.Heap.BufferDescriptorSize)
                        Attr.int64 "min_resource_heap_reserved_range" (int64 f.Heap.MinResourceHeapReservedRange)
                        Attr.int64 "max_push_data_size" (int64 f.Heap.MaxPushDataSize)
                    ]

                for m in f.MemoryTypes do
                    if m.Flags <> 0u then
                        notes.Enqueue(
                            sprintf
                                "vulkan memory type %d: %s, heap %d (%d MiB%s)"
                                m.Index
                                (memFlagNames m.Flags)
                                m.Heap
                                (m.HeapBytes / 1048576UL)
                                (if m.HeapFlags &&& Vk.Flags.HeapDeviceLocal <> 0u then ", DEVICE_LOCAL" else "")
                        )

                notes.Enqueue(
                    sprintf
                        "vulkan: %s, %s, API %s, queue family %d (flags 0x%x, %d timestamp bits, %g ns/tick)"
                        f.Name
                        f.Driver
                        f.ApiVersion
                        f.QueueFamily
                        f.QueueFamilyFlags
                        f.TimestampValidBits
                        f.TimestampPeriodNs
                )

                notes.Enqueue(
                    sprintf
                        "vulkan heap: bufferDescriptorSize %d (align %d), imageDescriptorAlignment %d, resourceHeapAlignment %d, minResourceHeapReservedRange %d, maxResourceHeapSize %d, maxPushDataSize %d; geometry: stride %d, slots %d B, reserved [%d, +%d), heap %d B"
                        f.Heap.BufferDescriptorSize
                        f.Heap.BufferDescriptorAlignment
                        f.Heap.ImageDescriptorAlignment
                        f.Heap.ResourceHeapAlignment
                        f.Heap.MinResourceHeapReservedRange
                        f.Heap.MaxResourceHeapSize
                        f.Heap.MaxPushDataSize
                        g.Stride
                        g.SlotBytes
                        g.ReservedOffset
                        g.ReservedSize
                        g.HeapBytes
                )

                notes.Enqueue(
                    sprintf
                        "vulkan memory types used: pages %d, upload %d, readback %d, heap %A; minStorageBufferOffsetAlignment %d, maxStorageBufferRange %d"
                        f.PageMemoryType
                        f.UploadMemoryType
                        f.ReadbackMemoryType
                        f.HeapMemoryTypes
                        f.MinStorageBufferOffsetAlignment
                        f.MaxStorageBufferRange
                ))

            for ((v, p), cfg) in Map.toList vkConfigLazies |> List.sortBy (fun ((v, p), _) -> vkConfigs |> List.findIndex ((=) (v, p))) do
                let label = sprintf "vulkan/%s/%s" (variantName v) (placementName p)

                testCase (label + ": pipelines with a NULL layout and the set 0 binding 0 mapping") (fun () ->
                    let c = cfg.Force()
                    let o = c.Observe()
                    observations[label] <- o

                    Telemetry.event
                        "gpu.spike.config"
                        (attrsOf label
                         @ [
                             Attr.str "heap_address" (sprintf "0x%x" o.HeapAddress)
                             Attr.int "heap_memory_type" o.HeapMemoryType
                             Attr.int64 "heap_bytes" (int64 c.Geometry.HeapBytes)
                         ]))

                yield! caseTests label cfg

                testCase (label + ": teardown") (fun () ->
                    if cfg.IsValueCreated then
                        let o = cfg.Value.Observe()
                        observations[label] <- o

                        notes.Enqueue(
                            sprintf
                                "%s heap at 0x%x (buffer alignment %d, memory type %d); reserved range bytes changed by the driver: %s; slot 1 descriptor bytes %s"
                                label
                                o.HeapAddress
                                o.BufferAlignment
                                o.HeapMemoryType
                                (match o.ReservedBytesChanged with
                                 | Some n -> string n
                                 | None -> "not observable (device-local)")
                                (Convert.ToHexStringLower o.Slot1Descriptor)
                        )

                        (cfg.Value :> IDisposable).Dispose())

            for (ctl, cfg) in vkControlLazies do
                let label = sprintf "vulkan/explicit/bar+%s" ctl.Suffix

                yield! caseTestsWhere (fun c -> controlSubset.Contains c.Name) label cfg

                testCase (label + ": teardown") (fun () ->
                    if cfg.IsValueCreated then
                        (cfg.Value :> IDisposable).Dispose())

            testCase "cross-backend: every vulkan configuration equals d3d12 bit for bit" (fun () ->
                match results.TryGetValue "d3d12" with
                | false, _ -> skiptest "the d3d12 group did not run in this process"
                | true, d ->
                    for KeyValue(label, outs) in results do
                        if label.StartsWith "vulkan/" then
                            for o in outs do
                                match d |> List.tryFind (fun x -> x.Case = o.Case) with
                                | Some x when o.Gating ->
                                    Expect.equal o.Digest x.Digest (sprintf "%s %s equals d3d12" label o.Case)
                                | Some x ->
                                    notes.Enqueue(
                                        sprintf
                                            "%s %s (informational) vs d3d12: %s"
                                            label
                                            o.Case
                                            (if o.Digest = x.Digest then "identical" else "different")
                                    )
                                | None -> ())

            testCase "vulkan: teardown" (fun () ->
                if vkContext.IsValueCreated then
                    (vkContext.Value :> IDisposable).Dispose())
        ]

let tiers: TestTiers =
    { TestTiers.empty with
        Cpu = cpuTests
        Gpu = [ Backend.D3D12, [ d3d12Tests ]; Backend.Vulkan, [ vulkanTests ] ] }

let writeReport (runId: RunId) =
    let configs =
        [ for label in order -> { Report.Label = label; Report.Outcomes = results[label] } ]

    if not configs.IsEmpty then
        let dir = Path.Combine(Env.repoRoot (), "spikes", "vk-heap", "results")

        let header =
            [
                sprintf "Run `%s` (run log under `LRM_DATA/runs/%s`)" runId.Value runId.Value
                sprintf "Shaders: DXC %s, ShaderSetHash `%s`" PinnedDxcVersion (shaders.Force().Hash.Hex)
                yield! Report.groupNotes (notes |> Seq.distinct |> List.ofSeq)
            ]

        Report.write dir header configs
        printfn "r1-spike: wrote %s" dir

[<EntryPoint>]
let main argv =
    Process.init ()
    let runId, sink = RunLog.start "spikes/vk-heap"
    use _sink = sink
    use _t = Telemetry.install (Telemetry.ofSink sink)
    use _s = Telemetry.scope [ Attr.str TelemetryKeys.Component "r1.vk_heap_spike" ]
    printfn "r1-spike: run %s, events %s" runId.Value sink.Path
    let code = Runner.main argv tiers

    if Env.tryGet "LRM_TEST_TIER" = Some "gpu" || (argv |> Array.contains "gpu") then
        let classes =
            [
                if d3d12.IsValueCreated then
                    let f = d3d12.Value.Facts

                    yield
                        "d3d12",
                        {
                            Backend = ExecutionBackend.D3D12
                            DeviceId = f.Adapter
                            Driver = f.Driver
                            ShaderSetHash = Some(shaders.Force().Hash)
                            Arch = "dxil-cs_6_6"
                        }
                if vkContext.IsValueCreated then
                    let f = vkContext.Value.Facts

                    for v in [ Explicit; Emulated ] ->
                        "vulkan/" + variantName v,
                        {
                            Backend = ExecutionBackend.Vulkan
                            DeviceId = f.Name
                            Driver = f.Driver
                            ShaderSetHash = Some(shaders.Force().Hash)
                            Arch = "spirv-vulkan1.3-" + variantName v
                        }
            ]

        if not classes.IsEmpty then
            RunLog.addClasses runId classes

        writeReport runId

    code
