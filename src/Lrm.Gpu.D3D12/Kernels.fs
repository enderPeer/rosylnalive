namespace Lrm.Gpu.D3D12

#nowarn "9"

open System
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop
open Microsoft.FSharp.NativeInterop

/// The kernels of a loaded shader set: the root signature from the set's rootsig.bin and one compute pipeline per DXIL
/// blob (GpuBackendInterface: D3D12 builds every PSO with the set's rootsig.bin).
[<Sealed>]
type internal D3D12Kernels(ctx: DeviceContext, shaders: ShaderSet) =
    let device = ctx.Device
    let mutable rootSignature = ComPtr<ID3D12RootSignature>.Null

    let pipelines: ComPtr<ID3D12PipelineState>[] =
        Array.create Kernels.all.Length ComPtr<ID3D12PipelineState>.Null

    let hash = ShaderSet.setHash ShaderTarget.Dxil shaders
    let blobs = ShaderSet.blobsFor ShaderTarget.Dxil shaders

    do
        try
            if not (RootSignature.isCanonical shaders.RootSignature) then
                Contract.fail "the shader set's rootsig.bin is not the canonical root signature"

            rootSignature <- device.CreateRootSignature(0u, ReadOnlySpan shaders.RootSignature)
            rootSignature.SetName "lrm.d3d12.rootsig"

            for b in blobs do
                match Kernels.tryFind b.Kernel with
                | None -> Contract.fail "the shader set has a blob for %s, which is not in the catalog" b.Kernel
                | Some k ->
                    use pin = fixed b.Bytes

                    let desc =
                        Desc.computePipeline rootSignature (NativePtr.toNativeInt pin) (unativeint b.Bytes.Length)

                    let pso: ComPtr<ID3D12PipelineState> = device.CreateComputePipelineState(&desc)
                    pso.SetName("lrm.d3d12.pso." + b.Kernel)
                    pipelines[k.Value] <- pso
        with _ ->
            for i in 0 .. pipelines.Length - 1 do
                ComPtr.release &pipelines[i]

            ComPtr.release &rootSignature
            reraise ()

    member _.Shaders = shaders
    member _.RootSignature = rootSignature
    member _.Count = blobs.Length

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
            ComPtr.release &pipelines[i]

        ComPtr.release &rootSignature
