/// CpuReferenceSemantics: ropeTable(invFreqBits, maxPos) is T_spec, bit for bit, with no libm involved.
module Lrm.Tensor.Tests.RopeTests

open System
open Expecto
open FsCheck
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Testing
open Lrm.Tensor.Tests.Common

/// torch's float32 inv_freq for theta 1e6 and head_dim 32 (ReferenceOracles vectors).
let tinyInvFreq =
    [|
        0x3f800000u
        0x3ed7e89bu
        0x3e361887u
        0x3d99940du
        0x3d0186e3u
        0x3c5a7bf2u
        0x3bb8449cu
        0x3b1b690du
        0x3a83126fu
        0x39dd1725u
        0x393a7753u
        0x389d43a4u
        0x3804a2b2u
        0x375fba4fu
        0x36bcb0c1u
        0x361f23e4u
    |]

/// torch's float32 inv_freq for theta 1e6 and head_dim 128 (Qwen3-0.6B), from the planning probe
/// redteam/inv_torch.txt; the ReferenceOracles contract pins their SHA-256.
let qwenInvFreq =
    [|
        0x3f800000u
        0x3f4e4badu
        0x3f263de0u
        0x3f05f6efu
        0x3ed7e89bu
        0x3eadfcffu
        0x3e8c3504u
        0x3e61f835u
        0x3e361887u
        0x3e12bd91u
        0x3dec7fd6u
        0x3dbe94c6u
        0x3d99940du
        0x3d778513u
        0x3d47763fu
        0x3d20bc1du
        0x3d0186e3u
        0x3cd0c1a8u
        0x3ca8398bu
        0x3c879008u
        0x3c5a7bf2u
        0x3c301052u
        0x3c0de12du
        0x3be4aa46u
        0x3bb8449cu
        0x3b947daeu
        0x3b6f520eu
        0x3b40dac5u
        0x3b1b690du
        0x3afa78f0u
        0x3ac9d75cu
        0x3aa2a6f7u
        0x3a83126fu
        0x3a533f28u
        0x3a2a3b44u
        0x3a092e02u
        0x39dd1725u
        0x39b229fbu
        0x398f9272u
        0x39676492u
        0x393a7753u
        0x39164324u
        0x38f22ce2u
        0x38c327b5u
        0x389d43a4u
        0x387d75d4u
        0x384c3fbeu
        0x382497abu
        0x3804a2b2u
        0x37d5c441u
        0x37ac431du
        0x378ad0edu
        0x375fba4fu
        0x37344a0eu
        0x371148e3u
        0x36ea2732u
        0x36bcb0c1u
        0x36980e02u
        0x36751070u
        0x36457babu
        0x361f23e4u
        0x36003decu
        0x35ceaf79u
        0x35a68e4cu
    |]

let private bitsHash (bits: uint32[]) =
    Hash.sha256 (Runtime.InteropServices.MemoryMarshal.AsBytes(ReadOnlySpan bits).ToArray())

let private bitsOf (x: float32) = BitConverter.SingleToUInt32Bits x

/// The Qwen table for positions 0..8191, built once (it is checked by two tests).
let private qwenTable = lazy (RopeTable.create CpuExec.Machine qwenInvFreq 8192)

let tests =
    testList
        "rope"
        [
            testCase "the pinned inverse frequencies match the contract hashes and the tiny golden"
            <| fun () ->
                Expect.equal
                    (bitsHash tinyInvFreq).Hex
                    "7f7aa6f39d2fd0a08c069fbf6d7cf6679288240227d78c053382b884db09abee"
                    "tiny"

                Expect.equal
                    (bitsHash qwenInvFreq).Hex
                    "138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e"
                    "qwen"

                Expect.sequenceEqual TinyModel.model.Value.InvFreqBits tinyInvFreq "rope.safetensors inv_freq.bits"

            testCase "ropeTable equals the dumped tiny T_spec bit for bit"
            <| fun () ->
                use f = tiny.Force().OpenSafetensors "rope.safetensors"
                let cos = f.ReadF32 "t_spec.cos"
                let sin = f.ReadF32 "t_spec.sin"
                let t = TinyModel.model.Value.Rope
                Expect.equal (t.MaxPos, t.Half) (512, 16) "shape"
                expectSameBits "cos" (HostTensor.f32 cos.Shape t.Cos) cos
                expectSameBits "sin" (HostTensor.f32 sin.Shape t.Sin) sin

                Expect.equal
                    (t.Digest()).Hex
                    "615b420404dd336bbf4509e20dd121daefdcc0cd8039048ca71097b1b6d60801"
                    "digest"

            testCase "the Qwen3-0.6B table (positions 0..8191) has the contract digest"
            <| fun () ->
                Expect.equal
                    (qwenTable.Force().Digest()).Hex
                    "da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf"
                    "digest"

            testCase "contract T_spec entries"
            <| fun () ->
                let tinyT = TinyModel.model.Value.Rope
                let q = qwenTable.Force()

                for t, pos, j, c, s in
                    [
                        tinyT, 0, 0, 0x3f800000u, 0x00000000u
                        tinyT, 1, 0, 0x3f0a5140u, 0x3f576aa4u
                        tinyT, 1, 15, 0x3f800000u, 0x361f23e4u
                        tinyT, 13, 3, 0x3f0fb02cu, 0x3f53df36u
                        tinyT, 100, 7, 0x3f78d5f0u, 0x3e708f27u
                        tinyT, 511, 0, 0xbef17fe5u, 0x3f61bbb4u
                        tinyT, 511, 15, 0x3f7ffff4u, 0x3a9ed44fu
                        q, 4095, 0, 0xbd871e6cu, 0xbf7f7136u
                        q, 4095, 63, 0x3f7fff27u, 0x3ba683b4u
                        q, 8191, 1, 0xbf7c693cu, 0xbe2add27u
                    ] do
                    let cosBits = bitsOf (t.CosRow pos).[j]
                    let sinBits = bitsOf (t.SinRow pos).[j]
                    Expect.equal (cosBits, sinBits) (c, s) (Invariant.format $"pos {pos} j {j}")

            testCase "the table is identical at 1 and 32 threads"
            <| fun () ->
                let a = RopeTable.create CpuExec.Serial tinyInvFreq 512
                let b = RopeTable.create (CpuExec.WithThreads 32) tinyInvFreq 512
                Expect.equal (a.Digest()) (b.Digest()) "digest"

            testCase "exact cos and sin of special arguments"
            <| fun () ->
                let struct (c0, s0) = ExactTrig.cosSin 0.0f
                Expect.equal (c0, s0) (1.0, 0.0) "zero"
                let struct (_, sn) = ExactTrig.cosSin -0.0f
                Expect.equal (BitConverter.DoubleToInt64Bits sn) (BitConverter.DoubleToInt64Bits -0.0) "sin(-0) = -0"
                let struct (c1, s1) = ExactTrig.cosSin 1.0f
                // The correctly rounded values, from mpmath at 200 bits.
                Expect.equal
                    (BitConverter.DoubleToInt64Bits c1, BitConverter.DoubleToInt64Bits s1)
                    (0x3fe14a280fb5068cL, 0x3feaed548f090ceeL)
                    "1 rad"

                let tiny = MathF.ScaleB(1.0f, -30)
                let struct (ct, st) = ExactTrig.cosSin tiny
                Expect.equal (ct, st) (1.0, float tiny) "below 2^-27"
                let struct (cn, sn) = ExactTrig.cosSin nanf
                Expect.isTrue (Double.IsNaN cn && Double.IsNaN sn) "NaN"
                let struct (cb, sb) = ExactTrig.cosSin 1e30f
                Expect.isTrue (abs cb <= 1.0 && abs sb <= 1.0 && abs (cb * cb + sb * sb - 1.0) < 1e-15) "huge argument"

            Props.testProperty "cosSin is within one float64 ulp of the C runtime and odd/even in x"
            <| fun (bits: uint32) ->
                // Arguments in (-2^14, 2^14), the range RoPE uses.
                let x =
                    BitConverter.UInt32BitsToSingle((bits &&& 0x7FFFFFFFu) % 0x46800000u)
                    * (if bits &&& 1u = 0u then 1.0f else -1.0f)

                let struct (c, s) = ExactTrig.cosSin x
                let struct (cm, sm) = ExactTrig.cosSin -x
                let near (a: float) (b: float) = Metrics.ulpDistance64 a b <= 1.0
                near c (Math.Cos(float x)) && near s (Math.Sin(float x)) && c = cm && s = -sm

            testCase "ropeTable rejects bad inputs"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () -> RopeTable.create CpuExec.Serial [||] 4 |> ignore)
                    "no frequencies"

                Expect.throwsT<ArgumentException>
                    (fun () -> RopeTable.create CpuExec.Serial tinyInvFreq 0 |> ignore)
                    "max_pos"

                Expect.throwsT<ArgumentException>
                    (fun () -> RopeTable.create CpuExec.Serial [| 0x7F800000u |] 4 |> ignore)
                    "inf"

                let t = RopeTable.create CpuExec.Serial tinyInvFreq 4
                Expect.throwsT<ArgumentOutOfRangeException> (fun () -> t.CosRow(4).Length |> ignore) "position"

            testCase "informational: entries where the C runtime's cos/sin would round differently (Qwen table)"
            <| fun () ->
                let t = qwenTable.Force()
                let mutable differ = 0

                for pos in 0 .. t.MaxPos - 1 do
                    let c = t.CosRow pos
                    let s = t.SinRow pos

                    for j in 0 .. t.Half - 1 do
                        let a = float (RopeTable.angle pos qwenInvFreq[j])

                        if float32 (Math.Cos a) <> c[j] || float32 (Math.Sin a) <> s[j] then
                            differ <- differ + 1

                // Not a requirement: the exact table never depends on the C runtime. O1 found 0 on the generation machine.
                printfn
                    "rope: %d of %d Qwen entries differ from f32(Math.Cos/Sin) on this machine"
                    differ
                    (t.MaxPos * t.Half)
        ]
