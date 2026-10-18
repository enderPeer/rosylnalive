namespace Lrm.Gpu.Vulkan

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Vulkan.Interop

/// The kernels of a loaded shader set: one heap pipeline per SPIR-V blob (VulkanHeapRecipe section 5), each blob
/// first checked by R4b's SPIR-V inspector (one resource variable at set 0 binding 0, the push block of its catalog
/// entry, LocalSize, no float atomics and no 16-bit types), with its gpu.shader.spirv_check, gpu.shader.push_bytes and
/// gpu.shader.workgroup_bytes records.
[<RequireQualifiedAccess>]
module SpirvGuard =
    /// OpFunctionEnd with word count 1: the last instruction of every valid module (functions are its last section).
    [<Literal>]
    let OpFunctionEndWord = 0x00010038u

    /// Why a blob is not handed to the driver, beyond the inspector's rules: a length that is not whole words, or a
    /// last word that is not OpFunctionEnd. The inspector accepts a module truncated by its final OpFunctionEnd (or by
    /// OpReturn and OpFunctionEnd), and NVIDIA 616.92 hung for more than ten minutes creating a pipeline from one, so V1
    /// refuses it before vkCreateShaderModule. None when the blob passes.
    let problem (bytes: byte[]) : string option =
        if isNull bytes || bytes.Length < 20 || bytes.Length % 4 <> 0 then
            Some "its length is not a whole number of SPIR-V words"
        elif BitConverter.ToUInt32(bytes, bytes.Length - 4) <> OpFunctionEndWord then
            Some "its last instruction is not OpFunctionEnd (the module is truncated)"
        else
            None

[<Sealed>]
type internal VulkanKernels(ctx: DeviceContext, shaders: ShaderSet, stride: uint32) =
    let pipelines: VkPipeline[] = Array.create Kernels.all.Length VkPipeline.Null
    let hash = ShaderSet.setHash ShaderTarget.Spirv shaders
    let blobs = ShaderSet.blobsFor ShaderTarget.Spirv shaders
    let reports = ResizeArray<SpirvReport>()

    do
        try
            for b in blobs do
                match Kernels.tryFind b.Kernel with
                | None -> Contract.fail "the shader set has a blob for %s, which is not in the catalog" b.Kernel
                | Some k ->
                    let r = SpirvInspector.inspect Kernels.all[k.Value] b.Bytes
                    SpirvInspector.record r
                    reports.Add r

                    // Not a GpuBackendInterface rule (RecordingBackend accepts such a set; the shader build already
                    // refuses one), so not a GpuContractException: Vulkan alone builds pipelines from these bytes.
                    if not r.Passed then
                        invalidArg
                            "shaders"
                            (sprintf
                                "vulkan: the SPIR-V blob of %s breaks the rules %s"
                                b.Kernel
                                (SpirvInspector.rules r |> List.map SpirvRule.name |> String.concat ", "))

                    match SpirvGuard.problem b.Bytes with
                    | Some p -> invalidArg "shaders" (sprintf "vulkan: the SPIR-V blob of %s is refused: %s" b.Kernel p)
                    | None -> ()

                    pipelines[k.Value] <- ctx.CreatePipeline(b.Kernel, b.Bytes, stride)
        with _ ->
            for i in 0 .. pipelines.Length - 1 do
                ctx.DestroyPipeline pipelines[i]
                pipelines[i] <- VkPipeline.Null

            reraise ()

    member _.Shaders = shaders
    member _.Count = blobs.Length

    /// The inspector's report of every blob, in blob order.
    member _.Reports = List.ofSeq reports

    /// The pipeline of a kernel the set has.
    member _.Pipeline(k: KernelId) =
        if k.Value < 0 || k.Value >= pipelines.Length || pipelines[k.Value].IsNull then
            invalidOp (sprintf "%O has no pipeline" k)

        pipelines[k.Value]

    member _.Contains(k: KernelId) =
        k.Value >= 0 && k.Value < pipelines.Length && not pipelines[k.Value].IsNull

    interface IKernelSet with
        member _.ShaderSetHash = hash
        member x.Contains(k: KernelId) = x.Contains k

    member _.Dispose() =
        for i in 0 .. pipelines.Length - 1 do
            ctx.DestroyPipeline pipelines[i]
            pipelines[i] <- VkPipeline.Null
