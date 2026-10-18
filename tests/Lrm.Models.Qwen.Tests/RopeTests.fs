/// RopeConstants (ForwardOpsAndQwenModel, "RoPE"): the pinned inv_freq bits equal the golden and probe bits, and the
/// tables they give are T_spec bit for bit (ReferenceOracles vectors).
module Lrm.Models.Qwen.Tests.RopeTests

open System
open System.Collections.Generic
open System.IO
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Models.Qwen
open Lrm.Models.Qwen.Tests.Common

let private bitsOf (x: float32) = BitConverter.SingleToUInt32Bits x

let tests =
    testList
        "rope"
        [
            testCase "the pinned bits hash to the ReferenceOracles digests"
            <| fun () ->
                Expect.equal
                    (RopeConstants.digest (RopeConstants.invFreqBits 1e6 32)).Hex
                    "7f7aa6f39d2fd0a08c069fbf6d7cf6679288240227d78c053382b884db09abee"
                    "theta 1e6, head_dim 32"

                Expect.equal
                    (RopeConstants.digest (RopeConstants.invFreqBits 1e6 128)).Hex
                    "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e"
                    "theta 1e6, head_dim 128"

            testCase "the tiny bits equal goldens/tiny/g1/rope.safetensors inv_freq.bits"
            <| fun () ->
                use f = tiny.Force().OpenSafetensors "rope.safetensors"

                Expect.sequenceEqual
                    (RopeConstants.forConfig tinyWeights.Value.Config)
                    (f.ReadU32 "inv_freq.bits").Data
                    "bits"

            testCase "the Qwen3-0.6B bits equal the planning probe redteam/inv_torch.txt"
            <| fun () ->
                let probe =
                    File.ReadAllLines(Path.Combine(repoRoot (), "spikes", "probes", "redteam", "inv_torch.txt"))
                    |> Array.filter (fun l -> l.Trim().Length > 0)
                    |> Array.map (fun l -> UInt32.Parse(l.Trim(), Globalization.CultureInfo.InvariantCulture))

                Expect.sequenceEqual (RopeConstants.invFreqBits 1e6 128) probe "bits"
                Expect.equal (RopeConstants.forConfig (Qwen3Config.parse ConfigTests.qwenConfigText)) probe "by config"

            testCase "the tiny table equals the dumped T_spec bit for bit, with its digest"
            <| fun () ->
                use f = tiny.Force().OpenSafetensors "rope.safetensors"

                let t =
                    RopeConstants.table (CpuExec.WithThreads Threads) tinyWeights.Value.Config 512

                expectSameBits "cos" (HostTensor.f32 (Shape.matrix 512 16) t.Cos) (f.ReadF32 "t_spec.cos")
                expectSameBits "sin" (HostTensor.f32 (Shape.matrix 512 16) t.Sin) (f.ReadF32 "t_spec.sin")

                Expect.equal
                    (t.Digest()).Hex
                    "615b420404dd336bbf4509e20dd121daefdcc0cd8039048ca71097b1b6d60801"
                    "digest"

                // CpuOps builds the same table for its MaxContext.
                use ops = new CpuOps<float32>(tinyWeights.Value, tinyOptions ())
                Expect.equal (ops.RopeTable.Digest()) (t.Digest()) "CpuOps table"

            testCase "the Qwen3-0.6B table for positions 0..8191 has the ReferenceOracles digest and entries"
            <| fun () ->
                let c = Qwen3Config.parse ConfigTests.qwenConfigText
                let t = RopeConstants.table (CpuExec.WithThreads Threads) c 8192

                Expect.equal
                    (t.Digest()).Hex
                    "da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf"
                    "digest"

                for pos, j, cos, sin in
                    [
                        4095, 0, 0xbd871e6cu, 0xbf7f7136u
                        4095, 63, 0x3f7fff27u, 0x3ba683b4u
                        8191, 1, 0xbf7c693cu, 0xbe2add27u
                    ] do
                    Expect.equal
                        (bitsOf (t.CosRow(pos)[j]), bitsOf (t.SinRow(pos)[j]))
                        (cos, sin)
                        (sprintf "(%d, %d)" pos j)

            testCase "the tiny table matches the ReferenceOracles entry vectors"
            <| fun () ->
                let t = RopeConstants.table CpuExec.Serial tinyWeights.Value.Config 512

                for pos, j, cos, sin in
                    [
                        0, 0, 0x3f800000u, 0x00000000u
                        1, 0, 0x3f0a5140u, 0x3f576aa4u
                        1, 15, 0x3f800000u, 0x361f23e4u
                        13, 3, 0x3f0fb02cu, 0x3f53df36u
                        100, 7, 0x3f78d5f0u, 0x3e708f27u
                        511, 0, 0xbef17fe5u, 0x3f61bbb4u
                        511, 15, 0x3f7ffff4u, 0x3a9ed44fu
                    ] do
                    Expect.equal
                        (bitsOf (t.CosRow(pos)[j]), bitsOf (t.SinRow(pos)[j]))
                        (cos, sin)
                        (sprintf "(%d, %d)" pos j)

            testCase "a (theta, head_dim) without pinned bits raises, naming the pinned pairs"
            <| fun () ->
                Expect.isNone (RopeConstants.tryFind 1e4 128) "theta 1e4"
                Expect.isNone (RopeConstants.tryFind 1e6 64) "head_dim 64"

                let e =
                    throws<KeyNotFoundException> "raises" (fun () -> RopeConstants.invFreqBits 1e4 64 |> ignore)

                Expect.stringContains e.Message "head_dim 128" "names the pinned pairs"

            testCase "tryFind returns a copy"
            <| fun () ->
                let a = (RopeConstants.tryFind 1e6 32).Value
                a[0] <- 0u
                Expect.equal (RopeConstants.invFreqBits 1e6 32).[0] 0x3f800000u "unchanged"
        ]
