namespace Lrm.Models.Qwen

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Tensor.CpuReference

/// The pinned RoPE inverse frequencies (ForwardOpsAndQwenModel contract; ReferenceOracles, T_spec). torch's float32
/// inv_freq differs from every closed-form F# formula in some entries, so the bits are data, keyed by (rope_theta,
/// head_dim). The table itself is T_spec, built by CpuRef in exact arithmetic (ADR O2-0001), so it is the same on
/// every machine, runtime and thread count.
[<RequireQualifiedAccess>]
module RopeConstants =
    /// theta 1e6, head_dim 32: the tiny model (goldens/tiny/g1/rope.safetensors, inv_freq.bits).
    let private theta1e6HeadDim32 =
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

    /// theta 1e6, head_dim 128: Qwen3-0.6B (the planning probe redteam/inv_torch.txt; ReferenceOracles pins the
    /// SHA-256 of the 64 words).
    let private theta1e6HeadDim128 =
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

    /// Every pinned entry: (rope_theta, head_dim, inv_freq bits). head_dim / 2 words each.
    let entries: (float * int * uint32[]) list =
        [ 1e6, 32, theta1e6HeadDim32; 1e6, 128, theta1e6HeadDim128 ]

    /// The pinned bits for (theta, headDim) (a copy), or None. Theta matches exactly, bit for bit.
    let tryFind (theta: float) (headDim: int) : uint32[] option =
        entries
        |> List.tryFind (fun (t, h, _) ->
            BitConverter.DoubleToInt64Bits t = BitConverter.DoubleToInt64Bits theta
            && h = headDim)
        |> Option.map (fun (_, _, bits) -> Array.copy bits)

    /// The pinned bits for (theta, headDim); raises KeyNotFoundException for a pair with no pinned bits.
    let invFreqBits (theta: float) (headDim: int) : uint32[] =
        match tryFind theta headDim with
        | Some bits -> bits
        | None ->
            let known =
                entries
                |> List.map (fun (t, h, _) -> Invariant.format $"(theta {Invariant.f64 t}, head_dim {h})")
                |> String.concat ", "

            raise (
                KeyNotFoundException(
                    Invariant.format
                        $"no pinned RoPE inv_freq for theta {Invariant.f64 theta} and head_dim {headDim}; pinned: {known}. Dump torch's bits with the golden tools (ReferenceOracles, T_spec) and add them to RopeConstants"
                )
            )

    /// The pinned bits of a config.
    let forConfig (c: Qwen3Config) = invFreqBits c.RopeTheta c.HeadDim

    /// SHA-256 of bits as u32 little-endian: the ReferenceOracles digest of an inv_freq vector.
    let digest (bits: uint32[]) =
        Hash.sha256 (Runtime.InteropServices.MemoryMarshal.AsBytes(ReadOnlySpan bits).ToArray())

    /// T_spec of a config for positions 0 .. maxPos - 1 (RopeTable.create on `exec`).
    let table (exec: CpuExec) (c: Qwen3Config) (maxPos: int) : RopeTable =
        RopeTable.create exec (forConfig c) maxPos
