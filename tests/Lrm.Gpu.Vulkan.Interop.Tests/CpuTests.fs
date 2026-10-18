/// The cpu tier (CI): codegen diff and idempotence, layouts against the generator and the recipe's known answers,
/// the ABI oracle with cl.exe and its mutants, the cross-check against R1's spike layouts, the generated wrappers on
/// fake entry points, pNext chains, the device-check rules, tail calls and the contract's idiom rules.
module Lrm.Gpu.Vulkan.Interop.Tests.CpuTests

#nowarn "9"
#nowarn "51"

open System
open System.Diagnostics
open System.IO
open System.Reflection
open System.Reflection.Metadata
open System.Reflection.PortableExecutable
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Expecto
open Microsoft.FSharp.NativeInterop
open Lrm.Core
open Lrm.Gpu.Vulkan.Interop
open Lrm.Gpu.Vulkan.Interop.Tests.Fakes
open Lrm.Testing

let private interop = typeof<VkInstance>.Assembly

let private config () =
    Path.GetFileName(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location))

/// Runs a process; returns (exit code, stdout, stderr).
let private exec (file: string) (args: string list) =
    let psi = ProcessStartInfo(file, UseShellExecute = false)

    for a in args do
        psi.ArgumentList.Add a

    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    use p = Process.Start psi
    let out = p.StandardOutput.ReadToEndAsync()
    let err = p.StandardError.ReadToEndAsync()
    p.WaitForExit()
    p.ExitCode, out.Result, err.Result

/// The directory of the dotnet host running this exe (tools/test.ps1 runs `dotnet <exe>.dll`).
let private dotnetRoot () =
    let host = Environment.ProcessPath

    if not (isNull host) && Path.GetFileNameWithoutExtension host = "dotnet" then
        Path.GetDirectoryName host
    else
        match Env.tryGet "DOTNET_ROOT" with
        | Some r -> r
        | None -> failtestf "cannot find the dotnet root (host %s, no DOTNET_ROOT)" host

let private dotnet () =
    Path.Combine(
        dotnetRoot (),
        (if OperatingSystem.IsWindows() then
             "dotnet.exe"
         else
             "dotnet")
    )

let private generatorDll () =
    let tool =
        Path.Combine(Env.repoRoot (), "artifacts", "bin", "Lrm.Tools.VkGen", config (), "Lrm.Tools.VkGen.dll")

    if not (File.Exists tool) then
        skiptestf "%s is not built" tool

    tool

let private codegenAttrs =
    [ Attr.str TelemetryKeys.Backend "vulkan"; Attr.str "check" "codegen_diff" ]

let private codegen =
    testList
        "codegen"
        [
            testCase "vkgen regenerates the committed files byte for byte"
            <| fun () ->
                let repo = Env.repoRoot ()
                let code, out, err = exec (dotnet ()) [ generatorDll (); "--check"; "--repo"; repo ]

                match code with
                | 0 -> Telemetry.check "gpu.codegen.vulkan" true codegenAttrs
                | 4 ->
                    // Not a pass: without the pinned vk.xml nothing can be regenerated. Say so, then skip.
                    let reason = err.Trim()
                    Telemetry.event "gpu.codegen.vulkan.skipped" (Attr.str "reason" reason :: codegenAttrs)
                    eprintfn "vkgen codegen diff SKIPPED (not passed): %s" reason
                    skiptestf "the pinned vk.xml is not available: %s" reason
                | c ->
                    Telemetry.check "gpu.codegen.vulkan" false codegenAttrs
                    failtestf "vkgen --check exited %d: %s%s" c out err

            testCase "regeneration is idempotent: two runs into a fresh tree give the committed bytes"
            <| fun () ->
                let repo = Env.repoRoot ()
                let tool = generatorDll ()

                let root =
                    Path.Combine(repo, "artifacts", "vulkan-interop-tests", "regen", string (Guid.NewGuid()))

                for rel in [ "tools/vkgen/bindings.json"; "tools/bootstrap/toolchain.json" ] do
                    let dst = Path.Combine(root, rel)
                    Directory.CreateDirectory(Path.GetDirectoryName dst) |> ignore
                    File.Copy(Path.Combine(repo, rel), dst)

                let outDir = Path.Combine(root, "src", "Lrm.Gpu.Vulkan.Interop")
                Directory.CreateDirectory outDir |> ignore

                let snapshot () =
                    [
                        for f in [ "Vk.g.fs"; "VkCommands.g.fs"; "VkMeta.g.fs" ] ->
                            f, File.ReadAllBytes(Path.Combine(outDir, f))
                    ]

                let first = exec (dotnet ()) [ tool; "--repo"; root ]

                match first with
                | 4, _, err -> skiptestf "the pinned vk.xml is not available: %s" (err.Trim())
                | 0, _, _ -> ()
                | c, o, e -> failtestf "vkgen exited %d: %s%s" c o e

                let a = snapshot ()
                let code, o, e = exec (dotnet ()) [ tool; "--repo"; root ]
                Expect.equal code 0 (sprintf "second run: %s%s" o e)
                let b = snapshot ()

                for ((name, x), (_, y)) in List.zip a b do
                    Expect.isTrue (x = y) (sprintf "%s is the same after a second run" name)

                    let committed =
                        File.ReadAllBytes(Path.Combine(repo, "src", "Lrm.Gpu.Vulkan.Interop", name))

                    Expect.isTrue (x = committed) (sprintf "%s equals the committed file" name)
                    Expect.isFalse (Array.contains 13uy x) (sprintf "%s has LF endings only" name)

                Directory.Delete(root, true)

            testCase "bindings.json is the single list: every listed command, struct and enum is generated"
            <| fun () ->
                use doc =
                    Text.Json.JsonDocument.Parse(
                        File.ReadAllText(Path.Combine(Env.repoRoot (), "tools", "vkgen", "bindings.json"))
                    )

                let root = doc.RootElement

                let strings (e: Text.Json.JsonElement) =
                    [ for x in e.EnumerateArray() -> x.GetString() ]

                let commands = root.GetProperty "commands"

                let listed =
                    strings (commands.GetProperty "global")
                    @ strings (commands.GetProperty "instance")
                    @ strings (commands.GetProperty "device")

                Expect.equal
                    (VkMeta.Commands |> List.map _.Name)
                    listed
                    "VkMeta.Commands is bindings.json's list, in order"

                Expect.equal
                    (VkGlobalCommands.Names @ VkInstanceCommands.Names @ VkDeviceCommands.Names)
                    listed
                    "the three tables bind exactly the listed commands"

                let structNames = VkMeta.Structs |> List.map _.Name |> set
                let aliasNames = VkMeta.Aliases |> List.map fst |> set

                for s in strings (root.GetProperty "structs") do
                    Expect.isTrue (structNames.Contains s || aliasNames.Contains s) (sprintf "%s is bound" s)

                let enumNames =
                    interop.GetTypes()
                    |> Array.choose (fun t ->
                        match t.GetCustomAttribute<VkEnumAttribute>() with
                        | null -> None
                        | a -> Some a.Name)
                    |> set

                for e in strings (root.GetProperty "enums") do
                    Expect.isTrue (enumNames.Contains e || aliasNames.Contains e) (sprintf "%s is bound" e)

                Expect.equal VkMeta.HeaderVersion 365 "vk.xml 1.4.365"
                Expect.equal VkMeta.CoreVersion (1, 4) "core 1.4"
        ]

// ---- layouts and known answers -------------------------------------------------------------------------------------

let private hexOf (bytes: byte[]) =
    Convert.ToHexString(bytes).ToLowerInvariant()

let private bytesOfStruct<'T when 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType> (v: 'T) =
    let mutable copy = v
    MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(&copy, 1)).ToArray()

let private offset<'T> (field: string) =
    int (Marshal.OffsetOf(typeof<'T>, field))

let private layouts =
    testList
        "layouts"
        [
            testCase "every struct and union matches the generator's own vk.xml layout"
            <| fun () ->
                let errors = LayoutCheck.run ()
                Expect.isEmpty errors "LayoutCheck.run"
                Expect.isGreaterThan VkMeta.Structs.Length 100 "the bound structs"

            testCase "VulkanHeapRecipe KAT-3: struct layouts on x64"
            <| fun () ->
                Expect.equal (Marshal.SizeOf<VkDescriptorSetAndBindingMappingEXT>()) 96 "mapping size"

                Expect.equal
                    [
                        offset<VkDescriptorSetAndBindingMappingEXT> "descriptorSet"
                        offset<VkDescriptorSetAndBindingMappingEXT> "firstBinding"
                        offset<VkDescriptorSetAndBindingMappingEXT> "bindingCount"
                        offset<VkDescriptorSetAndBindingMappingEXT> "resourceMask"
                        offset<VkDescriptorSetAndBindingMappingEXT> "source"
                        offset<VkDescriptorSetAndBindingMappingEXT> "sourceData"
                    ]
                    [ 16; 20; 24; 28; 32; 40 ]
                    "mapping offsets"

                Expect.equal (Marshal.SizeOf<VkDescriptorMappingSourceDataEXT>()) 56 "the union is its largest member"

                for f in
                    typeof<VkDescriptorMappingSourceDataEXT>.GetFields(BindingFlags.Instance ||| BindingFlags.Public) do
                    Expect.equal (offset<VkDescriptorMappingSourceDataEXT> f.Name) 0 (f.Name + " at 0")

                Expect.equal (Marshal.SizeOf<VkDescriptorMappingSourceConstantOffsetEXT>()) 24 "constant offset"

                Expect.equal
                    [
                        offset<VkDescriptorMappingSourceConstantOffsetEXT> "heapArrayStride"
                        offset<VkDescriptorMappingSourceConstantOffsetEXT> "pEmbeddedSampler"
                        offset<VkDescriptorMappingSourceConstantOffsetEXT> "samplerHeapOffset"
                        offset<VkDescriptorMappingSourceConstantOffsetEXT> "samplerHeapArrayStride"
                    ]
                    [ 4; 8; 16; 20 ]
                    "constant offset members"

                Expect.equal (Marshal.SizeOf<VkShaderDescriptorSetAndBindingMappingInfoEXT>()) 32 "mapping info"
                Expect.equal (offset<VkShaderDescriptorSetAndBindingMappingInfoEXT> "pMappings") 24 "pMappings"
                Expect.equal (Marshal.SizeOf<VkBindHeapInfoEXT>()) 48 "bind heap info"
                Expect.equal (offset<VkBindHeapInfoEXT> "reservedRangeOffset") 32 "reservedRangeOffset"
                Expect.equal (Marshal.SizeOf<VkPushDataInfoEXT>()) 40 "push data info"
                Expect.equal (offset<VkPushDataInfoEXT> "data") 24 "data"
                Expect.equal (Marshal.SizeOf<VkResourceDescriptorInfoEXT>()) 32 "resource descriptor info"
                Expect.equal (offset<VkResourceDescriptorInfoEXT> "data") 24 "data"
                Expect.equal (Marshal.SizeOf<VkHostAddressRangeEXT>()) 16 "host range"
                Expect.equal (Marshal.SizeOf<VkDeviceAddressRangeEXT>()) 16 "device range"
                Expect.equal (Marshal.SizeOf<VkPipelineCreateFlags2CreateInfo>()) 24 "flags 2"
                Expect.equal (offset<VkPipelineCreateFlags2CreateInfo> "flags") 16 "flags at 16"

            testCase "VulkanHeapRecipe KAT-1: the mapping struct's 96 bytes"
            <| fun () ->
                let mutable m = VkDescriptorSetAndBindingMappingEXT.Create()
                m.bindingCount <- 1u

                m.resourceMask <-
                    VkSpirvResourceTypeFlagBitsEXT.VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT
                    ||| VkSpirvResourceTypeFlagBitsEXT.VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT

                m.source <- VkDescriptorMappingSourceEXT.VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT
                m.sourceData.constantOffset.heapArrayStride <- 16u

                Expect.equal
                    (hexOf (bytesOfStruct m))
                    ("5dd99c3b000000000000000000000000000000000000000001000000c0000000"
                     + "0000000000000000000000001000000000000000000000000000000000000000"
                     + "0000000000000000000000000000000000000000000000000000000000000000")
                    "KAT-1"

            testCase "VulkanHeapRecipe KAT-2: the heap geometry for the RTX 4080 limits"
            <| fun () ->
                let limits: DeviceCheck.DescriptorHeapLimits =
                    {
                        SamplerHeapAlignment = 64UL
                        ResourceHeapAlignment = 64UL
                        MaxSamplerHeapSize = 0UL
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

                let g = DeviceCheck.geometry limits
                Expect.equal g.SlotBytes 16UL "slot i at 16 i"
                Expect.equal g.SlotArea 1024UL "slot area"
                Expect.equal g.ReservedRangeOffset 1024UL "reserved range offset"
                Expect.equal g.ReservedRangeSize 96768UL "reserved range size"
                Expect.equal g.HeapRangeSize 97792UL "heapRange.size"

            testCase
                "DeviceCheck's heap rules and geometry agree with Heap.vulkanGeometry (Lrm.Gpu.Core) on KAT-2 and 14,400 limits"
            <| fun () ->
                let kat2: DeviceCheck.DescriptorHeapLimits =
                    {
                        SamplerHeapAlignment = 64UL
                        ResourceHeapAlignment = 64UL
                        MaxSamplerHeapSize = 0UL
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

                let core (h: DeviceCheck.DescriptorHeapLimits) : Lrm.Gpu.Core.VulkanHeapLimits =
                    {
                        BufferDescriptorSize = int64 h.BufferDescriptorSize
                        BufferDescriptorAlignment = int64 h.BufferDescriptorAlignment
                        ImageDescriptorAlignment = int64 h.ImageDescriptorAlignment
                        MinResourceHeapReservedRange = int64 h.MinResourceHeapReservedRange
                        MaxResourceHeapSize = int64 h.MaxResourceHeapSize
                    }

                Expect.equal (core kat2) Lrm.Gpu.Core.Heap.rtx4080VulkanLimits "KAT-2 is the RTX 4080's limits in both"

                let accepted = ref 0
                let rejected = ref 0

                let agree (h: DeviceCheck.DescriptorHeapLimits) =
                    let mine = DeviceCheck.heapReasons h

                    let theirs =
                        try
                            Ok(Lrm.Gpu.Core.Heap.vulkanGeometry (core h))
                        with :? Lrm.Gpu.Core.GpuContractException as e ->
                            Error e.Message

                    match mine, theirs with
                    | [], Ok c ->
                        accepted.Value <- accepted.Value + 1
                        let g = DeviceCheck.geometry h

                        Expect.equal
                            (int64 g.SlotBytes,
                             int64 g.SlotArea,
                             int64 g.ReservedRangeOffset,
                             int64 g.ReservedRangeSize,
                             int64 g.HeapRangeSize)
                            (c.SlotStride, c.SlotAreaBytes, c.ReservedRangeOffset, c.ReservedRangeBytes, c.HeapBytes)
                            (sprintf "the same geometry for %A" (core h))
                    | [], Error e ->
                        failtestf "the device check accepts %A, which Heap.vulkanGeometry refuses: %s" (core h) e
                    | rs, Ok _ ->
                        failtestf "the device check refuses %A (%A), which Heap.vulkanGeometry lays out" (core h) rs
                    | _, Error _ -> rejected.Value <- rejected.Value + 1

                agree kat2

                for size in [ 0UL; 8UL; 12UL; 16UL; 20UL; 24UL; 32UL; 64UL; 256UL ] do
                    for bufferAlignment in [ 0UL; 1UL; 3UL; 4UL; 8UL; 12UL; 16UL; 32UL; 64UL; 128UL ] do
                        for imageAlignment in [ 0UL; 1UL; 2UL; 6UL; 32UL; 64UL; 256UL; 2048UL ] do
                            for reserved in [ 0UL; 1UL; 96768UL; 1048576UL ] do
                                for maxHeap in [ 0UL; 97791UL; 97792UL; 33554432UL; 1UL <<< 40 ] do
                                    agree
                                        { kat2 with
                                            BufferDescriptorSize = size
                                            BufferDescriptorAlignment = bufferAlignment
                                            ImageDescriptorAlignment = imageAlignment
                                            MinResourceHeapReservedRange = reserved
                                            MaxResourceHeapSize = maxHeap
                                        }

                Expect.equal (accepted.Value + rejected.Value) 14401 "KAT-2 and the grid"
                Expect.isGreaterThan accepted.Value 1000 "many limits are laid out"
                Expect.isGreaterThan rejected.Value 1000 "many are refused"

            testCase "the sType values and the recipe's flag values come from the registry"
            <| fun () ->
                Expect.equal (int VkDescriptorSetAndBindingMappingEXT.StructureType) 1000135005 "mapping sType"
                Expect.equal (int VkPhysicalDeviceMaintenance5Features.StructureType) 1000470000 "maintenance5"
                Expect.equal (int VkPushDataInfoEXT.StructureType) 1000135004 "push data info"

                Expect.equal
                    (uint64 VkPipelineCreateFlagBits2.VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT)
                    0x1000000000UL
                    "the heap pipeline flag"

                Expect.equal
                    (uint32 VkBufferUsageFlagBits.VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT)
                    0x10000000u
                    "not 0x08000000 (VK_BUFFER_USAGE_TILE_MEMORY_BIT_QCOM)"

                Expect.equal
                    (uint64 VkAccessFlagBits2.VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT)
                    0x0400000000000000UL
                    "the heap read access bit"

                Expect.equal (Unsafe.SizeOf<VkPipelineCreateFlags2>()) 8 "VkFlags64 is 8 bytes"

            testCase "handles: dispatchable ones are nativeint, non-dispatchable ones uint64, both 8 bytes"
            <| fun () ->
                for t in interop.GetTypes() do
                    match t.GetCustomAttribute<VkHandleAttribute>() with
                    | null -> ()
                    | a ->
                        let f =
                            t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
                            |> Array.exactlyOne

                        Expect.equal (Marshal.SizeOf t) 8 (a.Name + " is 8 bytes")

                        Expect.equal
                            f.FieldType
                            (if a.Dispatchable then typeof<nativeint> else typeof<uint64>)
                            (a.Name + " field type")

                Expect.isTrue (VkBuffer.Null.IsNull && not (VkBuffer 1UL).IsNull) "null handles"
                Expect.equal (VkBuffer 5UL) (VkBuffer 5UL) "handles compare by value"
        ]

// ---- the ABI oracle --------------------------------------------------------------------------------------------------

let private oracleRun (srcDir: string option) (outName: string) =
    match AbiOracle.findToolchain () with
    | None -> skiptest "no Visual Studio C++ toolchain (vswhere) on this machine"
    | Some tc ->
        let repo = Env.repoRoot ()

        match AbiOracle.headers repo with
        | Error e -> failtest e
        | Ok include' ->
            let src = defaultArg srcDir (AbiOracle.oracleDir repo)
            let out = Path.Combine(repo, "artifacts", "abi-oracle", "vulkan", outName)
            AbiOracle.compileAndRunIn tc include' src out

let private abi =
    testList
        "abi oracle"
        [
            testCase "the committed oracle sources equal the runtime dump"
            <| fun () ->
                let stale = AbiOracle.stale (Env.repoRoot ())

                Expect.isEmpty
                    stale
                    "tools/abi-oracle/vulkan is stale; regenerate with: pwsh tools/abi-oracle/vulkan/run.ps1 -Update"

            testCase "the dump covers every struct, enum, handle, constant, sType, extension and delegate"
            <| fun () ->
                let c, n = AbiOracle.cSource ()
                let cpp, m = AbiOracle.cppSource ()
                Expect.stringContains c "LRM_SIZE(VkDescriptorSetAndBindingMappingEXT, 96);" "KAT-3 size"
                Expect.stringContains c "LRM_OFFSET(VkBindHeapInfoEXT, reservedRangeOffset, 32);" "KAT-3 offset"
                Expect.stringContains c "LRM_ALIGN(VkDescriptorMappingSourceDataEXT, 8);" "a union's alignment"
                Expect.stringContains c "LRM_HANDLE(VkDevice, 8);" "a handle"
                Expect.stringContains c "LRM_ENUM(VK_ERROR_DEVICE_LOST, -4);" "a negative VkResult"
                Expect.stringContains c "LRM_HEADER_VERSION(365);" "the header version"
                Expect.stringContains cpp "LRM_STYPE(VkPushDataInfoEXT, PushDataInfoEXT, 1000135004);" "an sType"

                Expect.stringContains
                    cpp
                    "LRM_EXTENDS(PhysicalDeviceDescriptorHeapFeaturesEXT, DeviceCreateInfo);"
                    "pNext"

                Expect.stringContains
                    cpp
                    "LRM_FLAG64(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, 68719476736);"
                    "flags 64"

                Expect.stringContains cpp "LRM_SIG_SHAPE(PFN_vkCmdPushDataEXT, 'v', 0, 2);" "a signature"

                Expect.stringContains
                    cpp
                    "LRM_FIELD_ARRAY(VkPhysicalDeviceProperties, deviceName, 'i', 1, 256);"
                    "an array"

                Expect.stringContains
                    cpp
                    "LRM_STRING(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, \"VK_EXT_descriptor_heap\");"
                    "a name"

                Expect.stringContains cpp "LRM_IS_HANDLE(VkDevice);" "a handle in C"

                Expect.stringContains
                    cpp
                    "LRM_FIELD_TYPE(VkBindHeapInfoEXT, heapRange, lrm::is<VkDeviceAddressRangeKHR>);"
                    "a struct field's exact type"

                Expect.stringContains
                    cpp
                    "LRM_FIELD_ELEMENT_TYPE(VkPhysicalDeviceProperties, deviceName, lrm::is<uint8_t, char>);"
                    "an array's element type"

                Expect.stringContains
                    cpp
                    "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 0, lrm::is<VkPhysicalDevice>);"
                    "a handle parameter"

                Expect.stringContains cpp "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 3, lrm::pointer);" "a pointer"

                Expect.stringContains
                    cpp
                    "LRM_WRAP_CONST(PFN_vkCreateDevice, 1, pCreateInfo, lrm::is<VkDeviceCreateInfo>);"
                    "an inref"

                Expect.stringContains cpp "LRM_WRAP_MUT(PFN_vkCreateDevice, 3, pDevice, lrm::is<VkDevice>);" "a byref"

                Expect.stringContains
                    cpp
                    "LRM_WRAP_CONST(PFN_vkCmdUpdateBuffer, 4, pData, lrm::is<uint8_t, char, void>);"
                    "a span over void data"

                // Every bound field, delegate parameter and wrapper pointer parameter has its exact-type check.
                let ss = AbiOracle.structs ()
                let fields = ss |> List.sumBy _.Fields.Length

                let parameters =
                    AbiOracle.delegateMethods ()
                    |> List.sumBy (fun (_, i) -> i.GetParameters().Length + 1)

                let pointers = AbiOracle.wrappers () |> List.sumBy (snd >> List.length)
                let lines = cpp.Split('\n')

                let count (prefix: string) =
                    lines |> Array.filter _.StartsWith(prefix) |> Array.length

                Expect.equal
                    (count "LRM_FIELD_TYPE(" + count "LRM_FIELD_ELEMENT_TYPE(")
                    fields
                    "one exact type per field"

                Expect.equal
                    (count "LRM_SIG_ARG_TYPE(" + count "LRM_SIG_RET_TYPE(")
                    parameters
                    "one per parameter and return"

                Expect.equal (count "LRM_WRAP_CONST(" + count "LRM_WRAP_MUT(") pointers "one per pointer parameter"
                Expect.equal (AbiOracle.wrappers ()).Length VkMeta.Commands.Length "every command has a wrapper"
                Expect.isGreaterThan pointers 50 "the wrappers pass pointers"
                Expect.isGreaterThan n 2500 "C checks"
                Expect.isGreaterThan m 2900 "C++ checks"

            testCase
                "every delegate's parameter names are vulkan_core.h's, in order, so wrappers match C parameters by name"
            <| fun () ->
                match AbiOracle.headers (Env.repoRoot ()) with
                | Error e -> failtest e
                | Ok include' ->
                    let header = File.ReadAllText(Path.Combine(include', "vulkan", "vulkan_core.h"))
                    let problems = AbiOracle.parameterNameProblems header

                    Telemetry.check
                        "gpu.abi.parameter_names"
                        problems.IsEmpty
                        [
                            Attr.str TelemetryKeys.Backend "vulkan"
                            Attr.int "delegates" (AbiOracle.delegateMethods ()).Length
                            Attr.int "mismatches" problems.Length
                        ]

                    Expect.isEmpty problems "every raw delegate names its parameters as its PFN_* typedef"

                    let names = AbiOracle.headerParameterNames header

                    Expect.equal
                        names["PFN_vkCreateDevice"]
                        [ "physicalDevice"; "pCreateInfo"; "pAllocator"; "pDevice" ]
                        "a command"

                    Expect.equal
                        names["PFN_vkDebugUtilsMessengerCallbackEXT"]
                        [ "messageSeverity"; "messageTypes"; "pCallbackData"; "pUserData" ]
                        "a multi-line funcpointer"

                    Expect.equal names["PFN_vkCmdSetBlendConstants"] [ "commandBuffer"; "blendConstants" ] "an array"

                    // A header whose prototype swaps two parameters is reported.
                    let swapped =
                        header.Replace(
                            "(VKAPI_PTR *PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);",
                            "(VKAPI_PTR *PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkAllocationCallbacks* pAllocator, const VkDeviceCreateInfo* pCreateInfo, VkDevice* pDevice);"
                        )

                    Expect.notEqual swapped header "the mutant applies"

                    Expect.equal
                        (AbiOracle.parameterNameProblems swapped)
                        [
                            "PFN_vkCreateDevice: parameters (physicalDevice, pCreateInfo, pAllocator, pDevice), vulkan_core.h has (physicalDevice, pAllocator, pCreateInfo, pDevice)"
                        ]
                        "the swap is reported"

            testCase "cl.exe compiles the oracle against Vulkan-Headers 1.4.365 with 0 failures"
            <| fun () ->
                let r = oracleRun None "cpu-tier"

                Telemetry.check
                    "gpu.abi.oracle"
                    (r.ExitCode = 0 && r.Failures.IsEmpty)
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.int "static_checks" r.StaticChecks
                        Attr.int "failures" r.Failures.Length
                    ]

                if r.ExitCode <> 0 || not r.Failures.IsEmpty then
                    failtestf "ABI oracle failed (exit %d):\n%s" r.ExitCode r.Output

                Expect.isSome r.Summary "the oracle printed its summary"
                Expect.stringContains r.Summary.Value "; 0 failures" "0 failures"
                Expect.stringContains r.Summary.Value "Vulkan-Headers 1.4.365" "the pinned headers"

            testCase
                "the oracle rejects a wrong size, alignment, offset, field, enum, flag, constant, sType, pNext, signature or exact type"
            <| fun () ->
                let repo = Env.repoRoot ()
                let root = Path.Combine(repo, "artifacts", "abi-oracle", "vulkan", "mutants")
                let c = AbiOracle.cFile
                let cpp = AbiOracle.cppFile

                let edits =
                    [
                        c,
                        "LRM_SIZE(VkDescriptorSetAndBindingMappingEXT, 96);",
                        "LRM_SIZE(VkDescriptorSetAndBindingMappingEXT, 88);",
                        "sizeof(VkDescriptorSetAndBindingMappingEXT) == 88"
                        c,
                        "LRM_ALIGN(VkDescriptorMappingSourceDataEXT, 8);",
                        "LRM_ALIGN(VkDescriptorMappingSourceDataEXT, 4);",
                        "_Alignof(VkDescriptorMappingSourceDataEXT) == 4"
                        c,
                        "LRM_OFFSET(VkBindHeapInfoEXT, reservedRangeOffset, 32);",
                        "LRM_OFFSET(VkBindHeapInfoEXT, reservedRangeOffset, 24);",
                        "offsetof(VkBindHeapInfoEXT, reservedRangeOffset) == 24"
                        c,
                        "LRM_FIELD_SIZE(VkPushDataInfoEXT, offset, 4);",
                        "LRM_FIELD_SIZE(VkPushDataInfoEXT, offset, 2);",
                        "sizeof(VkPushDataInfoEXT.offset) == 2"
                        c,
                        "LRM_ENUM(VK_ERROR_DEVICE_LOST, -4);",
                        "LRM_ENUM(VK_ERROR_DEVICE_LOST, -5);",
                        "VK_ERROR_DEVICE_LOST == -5"
                        c,
                        "LRM_FLAG(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, 268435456);",
                        "LRM_FLAG(VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, 134217728);",
                        "VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT == 134217728"
                        c, "LRM_CONST(VK_UUID_SIZE, 16);", "LRM_CONST(VK_UUID_SIZE, 8);", "VK_UUID_SIZE == 8"
                        cpp,
                        "LRM_FLAG64(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, 68719476736);",
                        "LRM_FLAG64(VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT, 34359738368);",
                        "VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT == 34359738368"
                        cpp,
                        "LRM_STYPE(VkPushDataInfoEXT, PushDataInfoEXT, 1000135004);",
                        "LRM_STYPE(VkPushDataInfoEXT, PushDataInfoEXT, 1000135003);",
                        "VkPushDataInfoEXT has sType 1000135003"
                        cpp,
                        "LRM_EXTENDS(PhysicalDeviceDescriptorHeapFeaturesEXT, DeviceCreateInfo);",
                        "LRM_EXTENDS(PhysicalDeviceDescriptorHeapFeaturesEXT, BufferCreateInfo);",
                        "PhysicalDeviceDescriptorHeapFeaturesEXT extends BufferCreateInfo"
                        // A uint64 bound as a double of the same size and offset: only the field kind catches it.
                        cpp,
                        "LRM_FIELD_KIND(VkBindHeapInfoEXT, reservedRangeSize, 'i', 8);",
                        "LRM_FIELD_KIND(VkBindHeapInfoEXT, reservedRangeSize, 'f', 8);",
                        "VkBindHeapInfoEXT.reservedRangeSize is 'f'8"
                        cpp,
                        "LRM_FIELD_ARRAY(VkPhysicalDeviceProperties, deviceName, 'i', 1, 256);",
                        "LRM_FIELD_ARRAY(VkPhysicalDeviceProperties, deviceName, 'i', 1, 255);",
                        "VkPhysicalDeviceProperties.deviceName is 'i'1[255]"
                        cpp,
                        "LRM_SIG_ARG(PFN_vkCmdDispatch, 1, 'i', 4);",
                        "LRM_SIG_ARG(PFN_vkCmdDispatch, 1, 'i', 8);",
                        "PFN_vkCmdDispatch: parameter 1 is 'i'8"
                        cpp,
                        "LRM_SIG_SHAPE(PFN_vkCmdPushDataEXT, 'v', 0, 2);",
                        "LRM_SIG_SHAPE(PFN_vkCmdPushDataEXT, 'v', 0, 3);",
                        "PFN_vkCmdPushDataEXT: 3 parameters"
                        cpp,
                        "LRM_STRING(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, \"VK_EXT_descriptor_heap\");",
                        "LRM_STRING(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, \"VK_EXT_descriptor_heaps\");",
                        "VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME == \"VK_EXT_descriptor_heaps\""
                    ]
                    |> List.map (fun (f, o, n, needle) -> f, o, n, Some needle)

                // Exact-type mutants: mix-ups of the same kind and size, which only the exact types catch.
                let exact =
                    [
                        // A handle parameter bound as a nativeint.
                        cpp,
                        "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 0, lrm::is<VkPhysicalDevice>);",
                        "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 0, lrm::pointer);",
                        Some "PFN_vkCreateDevice: parameter 0 is lrm::pointer"
                        // A pointer to a handle (VkDevice* pDevice) bound as the handle itself.
                        cpp,
                        "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 3, lrm::pointer);",
                        "LRM_SIG_ARG_TYPE(PFN_vkCreateDevice, 3, lrm::is<VkDevice>);",
                        Some "PFN_vkCreateDevice: parameter 3 is lrm::is<VkDevice>"
                        // Two handles of one kind swapped.
                        cpp,
                        "LRM_SIG_ARG_TYPE(PFN_vkFreeCommandBuffers, 1, lrm::is<VkCommandPool>);",
                        "LRM_SIG_ARG_TYPE(PFN_vkFreeCommandBuffers, 1, lrm::is<VkFence>);",
                        Some "PFN_vkFreeCommandBuffers: parameter 1 is lrm::is<VkFence>"
                        // An unsigned return bound as signed.
                        cpp,
                        "LRM_SIG_RET_TYPE(PFN_vkGetBufferDeviceAddress, lrm::is<uint64_t>);",
                        "LRM_SIG_RET_TYPE(PFN_vkGetBufferDeviceAddress, lrm::is<int64_t>);",
                        Some "PFN_vkGetBufferDeviceAddress: returns lrm::is<int64_t>"
                        // A wrapper whose inref has the wrong pointee.
                        cpp,
                        "LRM_WRAP_CONST(PFN_vkCreateDevice, 1, pCreateInfo, lrm::is<VkDeviceCreateInfo>);",
                        "LRM_WRAP_CONST(PFN_vkCreateDevice, 1, pCreateInfo, lrm::is<VkInstanceCreateInfo>);",
                        Some
                            "PFN_vkCreateDevice: wrapper parameter pCreateInfo (1) points to const lrm::is<VkInstanceCreateInfo>"
                        // A wrapper's span over a struct of the same size but another type.
                        cpp,
                        "LRM_WRAP_CONST(PFN_vkWriteResourceDescriptorsEXT, 3, pDescriptors, lrm::is<VkHostAddressRangeEXT>);",
                        "LRM_WRAP_CONST(PFN_vkWriteResourceDescriptorsEXT, 3, pDescriptors, lrm::is<VkDeviceAddressRangeKHR>);",
                        Some
                            "PFN_vkWriteResourceDescriptorsEXT: wrapper parameter pDescriptors (3) points to const lrm::is<VkDeviceAddressRangeKHR>"
                        // A wrapper output (T*) declared as an inref (const T*).
                        cpp,
                        "LRM_WRAP_MUT(PFN_vkCreateDevice, 3, pDevice, lrm::is<VkDevice>);",
                        "LRM_WRAP_CONST(PFN_vkCreateDevice, 3, pDevice, lrm::is<VkDevice>);",
                        Some "PFN_vkCreateDevice: wrapper parameter pDevice (3) points to const lrm::is<VkDevice>"
                        // A uint32_t field bound as int32.
                        cpp,
                        "LRM_FIELD_TYPE(VkPushDataInfoEXT, offset, lrm::is<uint32_t>);",
                        "LRM_FIELD_TYPE(VkPushDataInfoEXT, offset, lrm::is<int32_t>);",
                        Some "VkPushDataInfoEXT.offset is lrm::is<int32_t>"
                        // Two 32-bit enums swapped.
                        cpp,
                        "LRM_FIELD_TYPE(VkDescriptorSetAndBindingMappingEXT, source, lrm::is<VkDescriptorMappingSourceEXT>);",
                        "LRM_FIELD_TYPE(VkDescriptorSetAndBindingMappingEXT, source, lrm::is<VkResult>);",
                        Some "VkDescriptorSetAndBindingMappingEXT.source is lrm::is<VkResult>"
                        // Two 16-byte structs swapped.
                        cpp,
                        "LRM_FIELD_TYPE(VkBindHeapInfoEXT, heapRange, lrm::is<VkDeviceAddressRangeKHR>);",
                        "LRM_FIELD_TYPE(VkBindHeapInfoEXT, heapRange, lrm::is<VkHostAddressRangeEXT>);",
                        Some "VkBindHeapInfoEXT.heapRange is lrm::is<VkHostAddressRangeEXT>"
                        // Two non-dispatchable handles swapped in a field.
                        cpp,
                        "LRM_FIELD_TYPE(VkBufferMemoryBarrier2, buffer, lrm::is<VkBuffer>);",
                        "LRM_FIELD_TYPE(VkBufferMemoryBarrier2, buffer, lrm::is<VkDeviceMemory>);",
                        Some "VkBufferMemoryBarrier2.buffer is lrm::is<VkDeviceMemory>"
                        // A handle field bound as a nativeint.
                        cpp,
                        "LRM_FIELD_TYPE(VkCommandBufferSubmitInfo, commandBuffer, lrm::is<VkCommandBuffer>);",
                        "LRM_FIELD_TYPE(VkCommandBufferSubmitInfo, commandBuffer, lrm::pointer);",
                        Some "VkCommandBufferSubmitInfo.commandBuffer is lrm::pointer"
                    ]

                // Where such a generator bug would also change a kind line, that edit comes along; the kind check
                // accepts a handle as any 8-byte pointer, so it must pass unreported.
                let companions =
                    [
                        cpp,
                        "LRM_SIG_ARG(PFN_vkCreateDevice, 0, 'h', 8);",
                        "LRM_SIG_ARG(PFN_vkCreateDevice, 0, 'p', 8);",
                        "PFN_vkCreateDevice: parameter 0 is 'p'8"
                        cpp,
                        "LRM_SIG_ARG(PFN_vkCreateDevice, 3, 'p', 8);",
                        "LRM_SIG_ARG(PFN_vkCreateDevice, 3, 'h', 8);",
                        "PFN_vkCreateDevice: parameter 3 is 'h'8"
                        cpp,
                        "LRM_FIELD_KIND(VkCommandBufferSubmitInfo, commandBuffer, 'h', 8);",
                        "LRM_FIELD_KIND(VkCommandBufferSubmitInfo, commandBuffer, 'p', 8);",
                        "VkCommandBufferSubmitInfo.commandBuffer is 'p'8"
                    ]

                let all =
                    edits @ exact @ (companions |> List.map (fun (f, o, n, _) -> f, o, n, None))

                AbiOracle.mutant repo (Path.Combine(root, "src")) (all |> List.map (fun (f, o, n, _) -> f, o, n))
                let r = oracleRun (Some(Path.Combine(root, "src"))) "mutants-out"
                Expect.notEqual r.ExitCode 0 "the mutants fail"
                let needles = all |> List.choose (fun (_, _, _, n) -> n)
                Expect.equal needles.Length 27 "27 mutants"

                for needle in needles do
                    Expect.isTrue
                        (r.Failures |> List.exists (fun l -> l.Contains needle))
                        (sprintf "the oracle reports '%s'\n%s" needle (String.concat "\n" r.Failures))

                // The kind lines that came along pass: only the exact types tell these mix-ups apart.
                for (_, _, _, unreported) in companions do
                    Expect.isFalse
                        (r.Failures |> List.exists (fun l -> l.Contains unreported))
                        (sprintf "the kind check alone accepts '%s'" unreported)

                Telemetry.check
                    "gpu.abi.mutants"
                    true
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.int "mutants" needles.Length
                        Attr.int "reported" needles.Length
                    ]

            testCase "R1's spike layouts (spikes/vk-heap/src/Layout.fs) agree with the binding"
            <| fun () ->
                let repo = Env.repoRoot ()
                let layout = Path.Combine(repo, "spikes", "vk-heap", "src", "Layout.fs")

                if not (File.Exists layout) then
                    skiptest "spikes/vk-heap/src/Layout.fs is gone; nothing to cross-check"

                let script = Path.Combine(AbiOracle.oracleDir repo, "spike-crosscheck.fsx")

                let code, out, err =
                    exec (dotnet ()) [ "fsi"; "--quiet"; "--nowarn:3886"; script; interop.Location ]

                let summary =
                    out.Split('\n') |> Array.tryFind (fun l -> l.StartsWith "spike cross-check:")

                Telemetry.check
                    "gpu.abi.spike_crosscheck"
                    (code = 0)
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.str "summary" (defaultArg summary "")
                    ]

                Expect.equal code 0 (sprintf "the cross-check passes:\n%s%s" out err)
                Expect.isSome summary "the script printed its summary"
                Expect.stringContains summary.Value "; 0 mismatches" "0 mismatches"
        ]

// ---- wrappers on fake entry points ---------------------------------------------------------------------------------

let private wrappers =
    testList
        "wrappers"
        [
            testCase "inref: the wrapper passes the address of the caller's struct, pinned for the call"
            <| fun () ->
                let fake = FakeLoader()
                let seen = ref 0n
                let seenOffset = ref 0u

                fake.Add(
                    "vkCmdPushDataEXT",
                    PFN_vkCmdPushDataEXT(fun cmd p ->
                        seen.Value <- p
                        seenOffset.Value <- (NativePtr.read (NativePtr.ofNativeInt<VkPushDataInfoEXT> p)).offset)
                )

                let d = VkDeviceCommands(fake.Load)
                let mutable info = VkPushDataInfoEXT.Create()
                info.offset <- 12u
                d.CmdPushDataEXT(VkCommandBuffer 1n, &info)
                Expect.equal seen.Value (NativePtr.toNativeInt &&info) "the struct's own address"
                Expect.equal seenOffset.Value 12u "the callee reads the caller's values"

            testCase "spans: the length is the count, an empty span passes NULL, and two spans must agree"
            <| fun () ->
                let fake = FakeLoader()
                let calls = ResizeArray<uint32 * nativeint>()

                fake.Add("vkCmdCopyBuffer", PFN_vkCmdCopyBuffer(fun _ _ _ n p -> calls.Add((n, p))))

                fake.Add(
                    "vkWriteResourceDescriptorsEXT",
                    PFN_vkWriteResourceDescriptorsEXT(fun _ _ _ _ -> VkResult.VK_SUCCESS)
                )

                let d = VkDeviceCommands(fake.Load)
                let regions = [| VkBufferCopy(); VkBufferCopy(); VkBufferCopy() |]
                d.CmdCopyBuffer(VkCommandBuffer 1n, VkBuffer 2UL, VkBuffer 3UL, ReadOnlySpan regions)
                d.CmdCopyBuffer(VkCommandBuffer 1n, VkBuffer 2UL, VkBuffer 3UL, ReadOnlySpan.Empty)
                let n0, p0 = calls[0]
                Expect.equal n0 3u "regionCount is the span's length"
                Expect.notEqual p0 0n "a pointer to the first region"
                Expect.equal calls[1] (0u, 0n) "an empty span is NULL and 0"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        d.WriteResourceDescriptorsEXT(
                            VkDevice 1n,
                            ReadOnlySpan [| VkResourceDescriptorInfoEXT.Create() |],
                            ReadOnlySpan [| VkHostAddressRangeEXT(); VkHostAddressRangeEXT() |]
                        ))
                    "pResources and pDescriptors share resourceCount"

            testCase "byref outputs, NULL pAllocator, VkResult checks and success codes"
            <| fun () ->
                let fake = FakeLoader()
                let allocator = ref 1n

                fake.Add(
                    "vkCreateBuffer",
                    PFN_vkCreateBuffer(fun _ _ pAllocator pBuffer ->
                        allocator.Value <- pAllocator
                        NativePtr.write (NativePtr.ofNativeInt<VkBuffer> pBuffer) (VkBuffer 0xABCDUL)
                        VkResult.VK_SUCCESS)
                )

                fake.Add(
                    "vkAllocateMemory",
                    PFN_vkAllocateMemory(fun _ _ _ _ -> VkResult.VK_ERROR_OUT_OF_DEVICE_MEMORY)
                )

                fake.Add("vkWaitSemaphores", PFN_vkWaitSemaphores(fun _ _ _ -> VkResult.VK_TIMEOUT))
                let d = VkDeviceCommands(fake.Load)
                let mutable ci = VkBufferCreateInfo.Create()
                let mutable b = VkBuffer.Null
                d.CreateBuffer(VkDevice 1n, &ci, &b)
                Expect.equal b (VkBuffer 0xABCDUL) "the handle arrives through the byref"
                Expect.equal allocator.Value 0n "pAllocator is NULL"
                let mutable ai = VkMemoryAllocateInfo.Create()
                let mutable mem = VkDeviceMemory.Null

                let e =
                    Expect.throwsC (fun () -> d.AllocateMemory(VkDevice 1n, &ai, &mem)) (fun (e: exn) ->
                        match e with
                        | VulkanException(command, result) -> command, result
                        | e -> raise e)

                Expect.equal e ("vkAllocateMemory", VkResult.VK_ERROR_OUT_OF_DEVICE_MEMORY) "an error code raises"
                let mutable wi = VkSemaphoreWaitInfo.Create()

                Expect.equal
                    (d.WaitSemaphores(VkDevice 1n, &wi, 0UL))
                    VkResult.VK_TIMEOUT
                    "VK_TIMEOUT is a success code"

            testCase "a NULL entry point raises VulkanMissingCommandException and is listed as missing"
            <| fun () ->
                let d = VkDeviceCommands(fun _ -> 0n)
                Expect.equal d.Missing VkDeviceCommands.Names "everything is missing"
                Expect.isNull d.vkCmdDispatch "the raw delegate is null"

                Expect.throwsC (fun () -> d.CmdDispatch(VkCommandBuffer 1n, 1u, 1u, 1u)) (fun e ->
                    match e with
                    | VulkanMissingCommandException "vkCmdDispatch" -> ()
                    | e -> raise e)

            // On a fake, GetDelegateForFunctionPointer hands back the managed delegate itself, so this checks the
            // wrapper's own code (null check, pins, spans, result check); the gpu tier repeats it into the driver.
            testCase "a wrapper call allocates nothing"
            <| fun () ->
                let fake = FakeLoader()
                fake.Add("vkCmdPushDataEXT", PFN_vkCmdPushDataEXT(fun _ _ -> ()))
                fake.Add("vkCmdDispatch", PFN_vkCmdDispatch(fun _ _ _ _ -> ()))
                fake.Add("vkCmdCopyBuffer", PFN_vkCmdCopyBuffer(fun _ _ _ _ _ -> ()))
                let d = VkDeviceCommands(fake.Load)
                let mutable info = VkPushDataInfoEXT.Create()
                let regions = [| VkBufferCopy() |]
                let cmd = VkCommandBuffer 1n

                let calls () =
                    for _ in 1..1000 do
                        d.CmdPushDataEXT(cmd, &info)
                        d.CmdDispatch(cmd, 1u, 1u, 1u)
                        d.CmdCopyBuffer(cmd, VkBuffer.Null, VkBuffer.Null, ReadOnlySpan regions)

                calls () // warm up: stubs are generated on first use
                let before = GC.GetAllocatedBytesForCurrentThread()
                calls ()
                let after = GC.GetAllocatedBytesForCurrentThread()
                Expect.equal (after - before) 0L "3,000 wrapper calls allocate 0 bytes"

            testCase "every command delegate is an unmanaged, blittable delegate with its attribute"
            <| fun () ->
                for m in VkMeta.Commands do
                    let t = m.Delegate
                    let a = t.GetCustomAttribute<VkCommandAttribute>()
                    Expect.isNotNull a (m.Name + " has VkCommand")
                    Expect.equal a.Name m.Name "the attribute names the command"
                    let u = t.GetCustomAttribute<UnmanagedFunctionPointerAttribute>()
                    Expect.isNotNull u (m.Name + " is unmanaged")
                    Expect.equal u.CallingConvention CallingConvention.Winapi "VKAPI_CALL"
                    let invoke = t.GetMethod "Invoke"

                    for p in invoke.GetParameters() do
                        Expect.isTrue
                            (p.ParameterType.IsValueType && not p.ParameterType.IsByRef)
                            (sprintf "%s.%s is a value type" m.Name p.Name)

                // Each table binds its level's list in bindings.json (the one list), however long it grows.
                use doc =
                    Text.Json.JsonDocument.Parse(
                        File.ReadAllText(Path.Combine(Env.repoRoot (), "tools", "vkgen", "bindings.json"))
                    )

                let listed (level: string) =
                    [
                        for x in doc.RootElement.GetProperty("commands").GetProperty(level).EnumerateArray() ->
                            x.GetString()
                    ]

                Expect.equal VkGlobalCommands.Names (listed "global") "global commands"
                Expect.equal VkInstanceCommands.Names (listed "instance") "instance commands"
                Expect.equal VkDeviceCommands.Names (listed "device") "device commands"

            testCase "runtime marshalling is disabled for the interop assembly"
            <| fun () ->
                Expect.isTrue
                    (interop.IsDefined(typeof<DisableRuntimeMarshallingAttribute>))
                    "DisableRuntimeMarshalling"
        ]

// ---- pNext chains ---------------------------------------------------------------------------------------------------

let private chains =
    testList
        "pNext chains"
        [
            testCase "the recipe's feature chain is valid for VkDeviceCreateInfo and in recipe order"
            <| fun () ->
                let mutable chain = FeatureChain()
                let head = FeatureChain.link &chain (FeatureChain.forCreate false)
                FeatureChain.enableRecipe &chain false
                Expect.isEmpty (VkChain.problems "VkDeviceCreateInfo" head) "valid"

                Expect.equal
                    (VkChain.walk head)
                    [
                        VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
                        VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
                        VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
                        VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES
                        VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT
                    ]
                    "Features2 -> Vulkan12 -> Vulkan13 -> Maintenance5 -> DescriptorHeap"

                let trues (bytes: byte[]) =
                    MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan bytes).ToArray()
                    |> Array.skip 4 // sType, padding and pNext
                    |> Array.filter ((=) 1u)
                    |> Array.length

                Expect.equal
                    (trues (bytesOfStruct chain.Features2)
                     + trues (bytesOfStruct chain.Vulkan12)
                     + trues (bytesOfStruct chain.Vulkan13)
                     + trues (bytesOfStruct chain.Maintenance5)
                     + trues (bytesOfStruct chain.DescriptorHeap))
                    7
                    "exactly the recipe's seven feature bits (five features, one with three bits)"

                Expect.isTrue (chain.Vulkan12.runtimeDescriptorArray = VkConst.VK_TRUE) "runtimeDescriptorArray"

            testCase "the fault struct joins the creation chain only when asked for"
            <| fun () ->
                let mutable chain = FeatureChain()
                let head = FeatureChain.link &chain (FeatureChain.forCreate true)
                FeatureChain.enableRecipe &chain true
                Expect.isEmpty (VkChain.problems "VkDeviceCreateInfo" head) "valid"

                Expect.equal
                    (List.last (VkChain.walk head))
                    VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT
                    "last"

                Expect.equal chain.Fault.deviceFault VkConst.VK_TRUE "deviceFault"

            testCase "the properties chain is valid for VkPhysicalDeviceProperties2"
            <| fun () ->
                let mutable chain = PropertiesChain()

                let parts =
                    PropertiesChain.forQuery
                        VkVersion.VK_API_VERSION_1_4
                        (Map.ofList
                            [
                                VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, 1u
                                VkConst.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, 2u
                            ])

                let head = PropertiesChain.link &chain parts

                Expect.equal
                    (VkChain.walk head).Head
                    VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
                    "head"
                // The structs after the head extend it.
                Expect.isEmpty (VkChain.problems "VkPhysicalDeviceProperties2" chain.Properties2.pNext) "valid"
                Expect.equal (VkChain.walk chain.Properties2.pNext).Length 6 "six structs after Properties2"

            testCase "VkChain.problems finds a wrong parent, a duplicate and a cycle"
            <| fun () ->
                let mutable a = VkBufferUsageFlags2CreateInfo.Create()

                Expect.equal
                    (VkChain.problems "VkBufferCreateInfo" (NativePtr.toNativeInt &&a))
                    []
                    "a valid one-node chain"

                Expect.equal
                    (VkChain.problems "VkDeviceCreateInfo" (NativePtr.toNativeInt &&a))
                    [ "VkBufferUsageFlags2CreateInfo does not extend VkDeviceCreateInfo" ]
                    "wrong parent"

                let mutable f1 = VkPhysicalDeviceVulkan12Features.Create()
                let mutable f2 = VkPhysicalDeviceVulkan12Features.Create()
                f1.pNext <- NativePtr.toNativeInt &&f2

                Expect.equal
                    (VkChain.problems "VkDeviceCreateInfo" (NativePtr.toNativeInt &&f1))
                    [
                        "VkPhysicalDeviceVulkan12Features appears twice in the chain of VkDeviceCreateInfo"
                    ]
                    "duplicate"

                f2.pNext <- NativePtr.toNativeInt &&f1
                let ps = VkChain.problems "VkDeviceCreateInfo" (NativePtr.toNativeInt &&f1)
                Expect.isTrue (ps |> List.exists (fun p -> p.Contains "more than 64 nodes")) "cycle"

            testCase "every struct's sType is unique, and every bound parent it names is a bound struct with an sType"
            <| fun () ->
                let withSType = VkMeta.Structs |> List.filter (fun s -> s.SType.IsSome)

                Expect.equal
                    (withSType |> List.map (fun s -> s.SType.Value) |> List.distinct |> List.length)
                    withSType.Length
                    "unique sTypes"

                for s in withSType do
                    let p =
                        s.Type.GetProperty("StructureType", BindingFlags.Static ||| BindingFlags.Public)

                    Expect.equal (p.GetValue null :?> VkStructureType) s.SType.Value (s.Name + ".StructureType")
                    let created = s.Type.GetMethod("Create").Invoke(null, [||])
                    let stype = s.Type.GetField("sType").GetValue created :?> VkStructureType
                    Expect.equal stype s.SType.Value (s.Name + ".Create() sets sType")
        ]

// ---- device-check rules ---------------------------------------------------------------------------------------------

/// What the RTX 4080 reports (driver 616.92), as the baseline of the rule tests.
let private rtx4080 () : DeviceCheck.Capabilities =
    {
        Identity =
            {
                Name = "NVIDIA GeForce RTX 4080"
                Type = VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                ApiVersion = VkVersion.make 0u 1u 4u 351u
                DriverVersion = (616u <<< 22) ||| (92u <<< 14)
                Driver = "616.92"
                VendorId = 0x10DEu
                DeviceId = 0x2704u
                DriverId = enum 4
                DriverName = "NVIDIA"
                DriverInfo = "616.92"
                DeviceUuid = ""
                DriverUuid = ""
            }
        Extensions =
            Map.ofList
                [
                    VkConst.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, 1u
                    VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, 1u
                ]
        DescriptorHeap =
            Some
                {
                    SamplerHeapAlignment = 64UL
                    ResourceHeapAlignment = 64UL
                    MaxSamplerHeapSize = 0UL
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
        StorageBufferDescriptorSize = Some 16UL
        Features =
            {
                ShaderStorageBufferArrayDynamicIndexing = true
                BufferDeviceAddress = true
                TimelineSemaphore = true
                RuntimeDescriptorArray = true
                Synchronization2 = true
                Maintenance5 = true
                DescriptorHeap = true
            }
        Optional =
            {
                DescriptorHeapCaptureReplay = true
                ShaderUntypedPointers = true
                CooperativeMatrix = true
                ShaderFloat16 = true
                ShaderInt8 = true
                DeviceFault = true
                DeviceFaultVendorBinary = true
                MemoryBudget = true
                CalibratedTimestamps = true
                ShaderDenormPreserveFloat32 = false
                ShaderDenormFlushToZeroFloat32 = false
            }
        QueueFamily = Some 2u
        TimestampValidBits = 64u
        TimestampPeriod = 1.0f
        MaxStorageBufferRange = UInt32.MaxValue
        MinStorageBufferOffsetAlignment = 16UL
        MaxComputeSharedMemorySize = 49152u
        MaxComputeWorkGroupInvocations = 1024u
        MemoryTypes = []
        MemoryHeaps = []
        CooperativeMatrixShapes = []
        TimeDomains = []
    }

let private rules =
    let heap (c: DeviceCheck.Capabilities) = c.DescriptorHeap.Value

    let withHeap
        (c: DeviceCheck.Capabilities)
        (f: DeviceCheck.DescriptorHeapLimits -> DeviceCheck.DescriptorHeapLimits)
        =
        { c with
            DescriptorHeap = Some(f (heap c))
        }

    let features (c: DeviceCheck.Capabilities) f = { c with Features = f c.Features }

    testList
        "device-check rules"
        [
            testCase "the RTX 4080's capabilities are accepted"
            <| fun () -> Expect.isEmpty (DeviceCheck.reasons (rtx4080 ())) "accepted"

            testCase "each missing requirement gives exactly its reason"
            <| fun () ->
                let b = rtx4080 ()

                let cases =
                    [
                        { b with
                            Identity =
                                { b.Identity with
                                    ApiVersion = VkVersion.make 0u 1u 3u 0u
                                }
                        },
                        "device API version 1.3.0 < 1.4"
                        { b with DescriptorHeap = None }, "VK_EXT_descriptor_heap not supported"
                        features b (fun f -> { f with DescriptorHeap = false }),
                        "VK_EXT_descriptor_heap present but descriptorHeap = VK_FALSE"
                        withHeap b (fun h -> { h with MaxPushDataSize = 128UL }), "maxPushDataSize 128 < 256"
                        { b with
                            StorageBufferDescriptorSize = Some 32UL
                        },
                        "vkGetPhysicalDeviceDescriptorSizeEXT(STORAGE_BUFFER) = 32 differs from bufferDescriptorSize 16"
                        withHeap b (fun h -> { h with MaxResourceHeapSize = 65536UL }),
                        "the 64-slot heap needs 97792 bytes > maxResourceHeapSize 65536"
                        // The limits Heap.vulkanGeometry (Lrm.Gpu.Core) refuses to lay a heap out with.
                        { withHeap b (fun h -> { h with BufferDescriptorSize = 0UL }) with
                            StorageBufferDescriptorSize = Some 0UL
                        },
                        "bufferDescriptorSize is 0"
                        withHeap b (fun h ->
                            { h with
                                BufferDescriptorAlignment = 12UL
                            }),
                        "descriptor alignments 12 (buffer) and 32 (image) are not both powers of two"
                        withHeap b (fun h ->
                            { h with
                                ImageDescriptorAlignment = 0UL
                            }),
                        "descriptor alignments 8 (buffer) and 0 (image) are not both powers of two"
                        { withHeap b (fun h -> { h with BufferDescriptorSize = 20UL }) with
                            StorageBufferDescriptorSize = Some 20UL
                        },
                        "bufferDescriptorSize 20 is not a multiple of bufferDescriptorAlignment 8"
                        { b with
                            Extensions = b.Extensions.Remove VkConst.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME
                        },
                        "VK_KHR_shader_untyped_pointers not supported (VulkanHeapRecipe section 1)"
                        features b (fun f ->
                            { f with
                                ShaderStorageBufferArrayDynamicIndexing = false
                            }),
                        "shaderStorageBufferArrayDynamicIndexing not supported"
                        features b (fun f -> { f with BufferDeviceAddress = false }),
                        "bufferDeviceAddress not supported"
                        features b (fun f -> { f with TimelineSemaphore = false }), "timelineSemaphore not supported"
                        features b (fun f ->
                            { f with
                                RuntimeDescriptorArray = false
                            }),
                        "runtimeDescriptorArray not supported"
                        features b (fun f -> { f with Synchronization2 = false }), "synchronization2 not supported"
                        features b (fun f -> { f with Maintenance5 = false }), "maintenance5 not supported"
                        { b with
                            MaxStorageBufferRange = 134217728u
                        },
                        "maxStorageBufferRange 134217728 is below a 1 GiB page"
                        { b with QueueFamily = None },
                        "no queue family with COMPUTE, without GRAPHICS and with timestamps"
                    ]

                for (c, reason) in cases do
                    Expect.equal (DeviceCheck.reasons c) [ reason ] reason

            testCase "maxStorageBufferRange: a 1 GiB page needs 2^30 bytes, not one less"
            <| fun () ->
                let b = rtx4080 ()

                Expect.isEmpty
                    (DeviceCheck.reasons
                        { b with
                            MaxStorageBufferRange = 1073741824u
                        })
                    "2^30"

                Expect.isNonEmpty
                    (DeviceCheck.reasons
                        { b with
                            MaxStorageBufferRange = 1073741823u
                        })
                    "2^30 - 1"

            testCase "maxPushDataSize: 256 passes the floor, 255 does not; the recipe itself needs 128"
            <| fun () ->
                let b = rtx4080 ()
                Expect.equal DeviceCheck.RecipePushDataSize 128UL "the recipe's ArgBlock"
                Expect.equal DeviceCheck.RequiredPushDataSize 256UL "the acceptance floor (ADR R2-0001)"
                Expect.isEmpty (DeviceCheck.reasons (withHeap b (fun h -> { h with MaxPushDataSize = 256UL }))) "256"
                Expect.isNonEmpty (DeviceCheck.reasons (withHeap b (fun h -> { h with MaxPushDataSize = 255UL }))) "255"

            testCase "driver versions: NVIDIA 616.92, Intel on Windows, and the rest"
            <| fun () ->
                Expect.equal
                    (DeviceCheck.formatDriverVersion 0x10DEu ((616u <<< 22) ||| (92u <<< 14)))
                    "616.92"
                    "NVIDIA"

                Expect.equal
                    (DeviceCheck.formatDriverVersion 0x10DEu ((552u <<< 22) ||| (22u <<< 14)))
                    "552.22"
                    "NVIDIA"

                Expect.equal (DeviceCheck.formatDriverVersion 0x1002u (VkVersion.make 0u 2u 0u 302u)) "2.0.302" "AMD"

                if OperatingSystem.IsWindows() then
                    Expect.equal (DeviceCheck.formatDriverVersion 0x8086u ((101u <<< 14) ||| 5186u)) "101.5186" "Intel"

            testCase "the recipe's queue family: COMPUTE, no GRAPHICS, timestamps"
            <| fun () ->
                let family (flags: VkQueueFlagBits) bits =
                    let mutable f = VkQueueFamilyProperties()
                    f.queueFlags <- flags
                    f.queueCount <- 1u
                    f.timestampValidBits <- bits
                    f

                let g = VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT
                let c = VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT
                let t = VkQueueFlagBits.VK_QUEUE_TRANSFER_BIT
                // The RTX 4080 with driver 616.92: graphics+compute, transfer, compute, then video families.
                let families =
                    [|
                        family (g ||| c ||| t) 64u
                        family t 64u
                        family (c ||| t) 64u
                        family c 0u
                    |]

                Expect.equal (QueueFamily.recipe families) (Some 2u) "family 2"
                Expect.equal (QueueFamily.recipe [| family (g ||| c) 64u; family c 0u |]) None "none without timestamps"
        ]

// ---- idiom ------------------------------------------------------------------------------------------------------------

/// Compiles TailCallConsumer.fs (an embedded resource) with the pinned SDK's F# compiler, optimized and with tail
/// calls on (the Release defaults), against the interop next to this exe. Returns the assembly's path.
let private compileConsumer () : string =
    let repo = Env.repoRoot ()

    let sdk =
        use doc =
            Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(repo, "global.json")))

        doc.RootElement.GetProperty("sdk").GetProperty("version").GetString()

    let fsc = Path.Combine(dotnetRoot (), "sdk", sdk, "FSharp", "fsc.dll")

    if not (File.Exists fsc) then
        failtestf "the F# compiler of SDK %s is not at %s" sdk fsc

    let dir = Path.Combine(repo, "artifacts", "vulkan-interop-tests", "tailcalls")
    Directory.CreateDirectory dir |> ignore
    let src = Path.Combine(dir, "TailCallConsumer.fs")
    let dll = Path.Combine(dir, "TailCallConsumer.dll")

    do
        use r =
            Assembly.GetExecutingAssembly().GetManifestResourceStream "TailCallConsumer.fs"

        use w = File.Create src
        r.CopyTo w

    let isManaged (path: string) =
        try
            use fs = File.OpenRead path
            use pe = new PEReader(fs)
            pe.HasMetadata
        with _ ->
            false

    let framework =
        Directory.GetFiles(RuntimeEnvironment.GetRuntimeDirectory(), "*.dll")
        |> Array.filter isManaged
        |> Array.sort

    let bin = AppContext.BaseDirectory

    let args =
        [
            "--nologo"
            "--noframework"
            "--targetprofile:netcore"
            "--target:library"
            "--optimize+"
            "--tailcalls+"
            "--deterministic+"
            "--nowarn:9,51"
            "-o:" + dll
            "-r:" + Path.Combine(bin, "FSharp.Core.dll")
            "-r:" + interop.Location
            "-r:" + Path.Combine(bin, "Lrm.Core.dll")
            yield! framework |> Array.map (fun f -> "-r:" + f)
            src
        ]

    // One argument per line: fsc does not split response-file lines at spaces (C:\Program Files\dotnet on CI).
    let rsp = Path.Combine(dir, "fsc.rsp")
    File.WriteAllLines(rsp, args)
    let code, out, err = exec (dotnet ()) [ fsc; "@" + rsp ]

    if code <> 0 then
        failtestf "fsc exited %d:\n%s%s" code out err

    dll

/// The interop's own sources (the hand-written files and the generated ones).
let private interopSources () =
    Directory.GetFiles(Path.Combine(Env.repoRoot (), "src", "Lrm.Gpu.Vulkan.Interop"), "*.fs")
    |> Array.sort
    |> List.ofArray

let private idiom =
    testList
        "idiom"
        [
            testCase "no tail calls in the interop or its tests (Tailcalls=false)"
            <| fun () ->
                // In a Debug build every project has Tailcalls=false, so an empty scan would pass trivially.
                if TailCalls.builtWithoutOptimization (Assembly.GetExecutingAssembly()) then
                    skiptest TailCalls.UnoptimizedBuild

                for asm in [ interop; Assembly.GetExecutingAssembly() ] do
                    Expect.isEmpty (TailCalls.scan asm.Location) (asm.GetName().Name + " has no tail. prefixes")

            testCase "a consumer built with tail calls never tail-calls a wrapper that takes inref or byref"
            <| fun () ->
                let dll = compileConsumer ()
                let sites = TailCalls.scan dll

                let byref =
                    sites
                    |> List.filter (fun s ->
                        [ "pushData"; "barrier"; "createBuffer"; "queue" ]
                        |> List.exists (fun f -> s.Caller.EndsWith("." + f)))

                Telemetry.check
                    "gpu.interop.tailcalls"
                    byref.IsEmpty
                    [
                        Attr.str TelemetryKeys.Backend "vulkan"
                        Attr.int "tail_calls" sites.Length
                        Attr.int "tail_calls_at_byref_wrappers" byref.Length
                    ]

                Expect.isEmpty byref "no inref or byref wrapper call is a tail call"

                // The control: the raw delegate with a stack address in tail position is a tail call, so tail calls
                // were on; this is what Tailcalls=false (Directory.Build.props) covers for the interop's consumers.
                Expect.isTrue
                    (sites |> List.exists (fun s -> s.Caller.EndsWith ".controlRawDelegate"))
                    (sprintf "the control has a tail call: %A" sites)

            testCase "no source silences FS0104, so every enum match has a wildcard arm"
            <| fun () ->
                // FS0104 ("enums may take values outside known cases") is an error under TreatWarningsAsErrors: a
                // match on a Vulkan enum compiles only with a wildcard arm (VulkanInteropApi contract).
                let repo = Env.repoRoot ()

                let files =
                    interopSources ()
                    @ [
                        Path.Combine(repo, "apps", "VkDeviceCheck", "Program.fs")
                        Path.Combine(repo, "apps", "Lrm.Cli", "Commands", "DevicesVulkan.fs")
                    ]
                    @ (Directory.GetFiles(Path.Combine(repo, "tests", "Lrm.Gpu.Vulkan.Interop.Tests"), "*.fs")
                       |> List.ofArray)

                for f in files do
                    let text = File.ReadAllText f
                    // The needle is built so that this file does not contain it.
                    let directive = "nowarn \"" + "104\""
                    Expect.isFalse (text.Contains directive || text.Contains("nowarn:" + "104")) (f + " keeps FS0104")

            testCase "no hand-written source keeps a pNext chain beyond its frame: chain structs are byref-like"
            <| fun () ->
                for name in [ "FeatureChain"; "PropertiesChain" ] do
                    let t = interop.GetType("Lrm.Gpu.Vulkan.Interop." + name)
                    Expect.isNotNull t name
                    Expect.isTrue t.IsByRefLike (name + " is a byref-like struct")
        ]

let tests =
    testList "vulkan interop (cpu)" [ codegen; layouts; abi; wrappers; chains; rules; idiom ]
