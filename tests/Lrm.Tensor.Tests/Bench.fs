/// Bench tier (local, informational): CpuRef throughput at Qwen3-0.6B sizes.
module Lrm.Tensor.Tests.Bench

open System
open System.Diagnostics
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Tensor.Tests.Common

let private time (label: string) (flops: float) (f: unit -> unit) =
    f ()
    let sw = Stopwatch.StartNew()
    let mutable runs = 0

    while runs < 3 || sw.Elapsed.TotalSeconds < 1.0 do
        f ()
        runs <- runs + 1

    let s = sw.Elapsed.TotalSeconds / float runs

    printfn
        "bench: %-44s %10s ms  %8s GFLOP/s"
        label
        (Invariant.format $"{s * 1e3:F3}")
        (if flops > 0.0 then
             Invariant.format $"{flops / s / 1e9:F2}"
         else
             "-")

let tests =
    testList
        "bench"
        [
            testCase "matmul, attention and the RoPE table at Qwen3-0.6B sizes"
            <| fun () ->
                let exec = CpuExec.Machine

                let w =
                    HostTensor.bf16 (Shape.matrix 3072 1024) (randomBf16 1UL "bench/w" (3072 * 1024) 0.05f)

                let head =
                    HostTensor.bf16 (Shape.matrix 151_936 1024) (randomBf16 1UL "bench/head" (151_936 * 1024) 0.05f)

                for rows in [ 1; 128 ] do
                    let x32 =
                        HostTensor.f32 (Shape.matrix rows 1024) (randomF32 1UL "bench/x" (rows * 1024) 1.0f)

                    let x64 =
                        HostTensor.f64 (Shape.matrix rows 1024) (randomF64 1UL "bench/x" (rows * 1024) 1.0)

                    let flops = 2.0 * float rows * 3072.0 * 1024.0

                    time (Invariant.format $"matmul F32 [{rows},1024] x [3072,1024]") flops (fun () ->
                        CpuRef.matmulBf16W exec x32 w |> ignore)

                    time (Invariant.format $"matmul F64 [{rows},1024] x [3072,1024]") flops (fun () ->
                        CpuRef.matmulBf16W exec x64 w |> ignore)

                let x1 = HostTensor.f32 (Shape.matrix 1 1024) (randomF32 1UL "bench/x1" 1024 1.0f)

                time "lm_head F32 [1,1024] x [151936,1024]" (2.0 * 151_936.0 * 1024.0) (fun () ->
                    CpuRef.matmulBf16W exec x1 head |> ignore)

                let shape =
                    {
                        QHeads = 16
                        KvHeads = 8
                        HeadDim = 128
                    }

                let keys = 4096
                let q = HostTensor.f32 (Shape.matrix 1 2048) (randomF32 1UL "bench/q" 2048 1.0f)

                let k =
                    HostTensor.f32 (Shape.matrix keys 1024) (randomF32 1UL "bench/k" (keys * 1024) 1.0f)

                let v =
                    HostTensor.f32 (Shape.matrix keys 1024) (randomF32 1UL "bench/v" (keys * 1024) 1.0f)

                let scale = CpuRef.attentionScale<float32> 128

                time "attention F32 decode, 4096 keys, 16 heads" (4.0 * 16.0 * float keys * 128.0) (fun () ->
                    CpuRef.attention exec q k v shape (keys - 1) scale |> ignore)

                time "ropeTable Qwen3 (8192 x 64, exact)" 0.0 (fun () ->
                    RopeTable.create exec RopeTests.qwenInvFreq 8192 |> ignore)
        ]
