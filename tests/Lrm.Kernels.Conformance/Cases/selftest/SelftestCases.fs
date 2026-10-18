/// Conformance cases of the five selftests (KernelAbiAndCatalog, "Notes for the kernel owners"): scale_f32 (also the
/// fp32 denormal probe), table_echo, args_echo, groups_echo (grid folding above 65,535 groups) and bf16_unpack. The
/// backends run them first; they are bit-exact on every backend, scale_f32 in the class's denormal mode.
module Lrm.Kernels.Conformance.Cases.Selftest

open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Kernels.Conformance

let private scale (name: string) (x: uint32[]) (capacity: int64) (a: uint32) (doc: string) =
    Case.create "scale_f32" name
    |> Case.f32Bits "x" x
    |> Case.output "y" capacity
    |> Case.u32 "n" (uint32 x.Length)
    |> Case.f32ArgBits "a" a
    |> Case.doc doc

let private bits (xs: float32[]) =
    xs |> Array.map System.BitConverter.SingleToUInt32Bits

let private f32 (x: float32) =
    System.BitConverter.SingleToUInt32Bits x

let private scaleCases =
    let r = CaseGen.rng "scale_f32"
    let uniform n = CaseGen.uniform r n -4.0f 4.0f |> bits

    [
        scale "n1" [| f32 3.0f |] 1L (f32 2.0f) "One element, one group"
        scale "n1000_a0.5" (uniform 1000) 1024L (f32 0.5f) "Vector A1's sizes; the output's capacity exceeds n"
        scale "n257" (uniform 257) 257L (f32 -3.0f) "Two groups, the second with one element"
        scale
            "denormals"
            (Array.append
                Fp32Denormals.probeInputs
                [|
                    0x00000001u
                    0x80000001u
                    0x007FFFFFu
                    0x00800001u
                    0x80FFFFFFu
                    0x3F800000u
                |])
            13L
            (f32 0.5f)
            "Denormal inputs and results: the probe's words and more; follows the class's mode"
        scale
            "specials"
            [|
                0x00000000u
                0x80000000u
                0x7F800000u
                0xFF800000u
                0x7FC00001u
                0x7F800001u
                0xFFC00000u
                0x7F7FFFFFu
                0xFF7FFFFFu
                0x3F800000u
            |]
            10L
            (f32 2.0f)
            "Zeros, infinities, NaN payloads and overflow to infinity; any NaN matches any NaN"
        scale "a_denormal" (uniform 64) 64L 0x00000003u "A denormal scale factor"
        scale "a_negative_zero" (uniform 33) 33L 0x80000000u "a = -0: every result is a signed zero"
        scale "a_nan" (uniform 5) 5L 0x7FC00042u "a = NaN with a payload: every result is NaN"
        scale
            "both_nan"
            (Array.init 600 (fun i ->
                match i % 6 with
                | 0 -> 0xFFC00005u
                | 1 -> 0x7F800001u
                | 2 -> 0x3F800000u
                | 3 -> 0x7F800000u
                | _ -> 0x7FC00001u))
            600L
            0x7FC00003u
            "NaN x and NaN a with other payloads (and a few other x): the first NaN operand, quieted (the NaN rule)"
        (scale "in_place" (uniform 300) 300L (f32 -1.5f) "y aliases x (declared alias): in place"
         |> fun c ->
             { c with
                 Io =
                     { c.Io with
                         Tensors = c.Io.Tensors |> List.filter (fun t -> t.Field <> "y")
                     }
             }
             |> Case.inPlace "y" "x")
        (scale "dims_ignored" (uniform 300) 300L (f32 0.25f) "Rank-3 dims on x: kernels read flat tokens"
         |> Case.dims "x" [ 2L; 3L; 50L ]
         |> Case.dims "y" [ 300L ])
    ]

let private tableEchoCases =
    let echo (name: string) (doc: string) (withSrc: KernelCase -> KernelCase) (capacity: int64) =
        Case.create "table_echo" name
        |> withSrc
        |> Case.outputU32 "y" capacity
        |> Case.doc doc

    [
        echo "null" "The null token's row is 16 zero words" (Case.nullMeta "src") 16L
        echo "f32_rank1" "A rank-1 f32 row" (Case.meta "src" DType.F32 [ 5L ]) 16L
        echo
            "bf16_rank4"
            "A rank-4 bf16 row: dims, strides and count"
            (Case.meta "src" DType.BF16 [ 2L; 3L; 4L; 5L ])
            16L
        echo
            "u32_rank2_capacity"
            "A u32 row; the output's capacity exceeds 16"
            (Case.meta "src" DType.U32 [ 7L; 9L ])
            20L
    ]

let private argsEchoCases =
    let echo (name: string) (values: uint32[]) (doc: string) =
        values
        |> Array.indexed
        |> Array.fold (fun c (i, v) -> c |> Case.u32 (sprintf "v%d" (i + 2)) v) (Case.create "args_echo" name)
        |> Case.outputU32 "y" 32L
        |> Case.doc doc

    let r = CaseGen.rng "args_echo"

    [
        echo "zeros" (Array.zeroCreate 30) "Every field 0: y[0] is lrm_groups_x (1), y[1] the token"
        echo
            "patterns"
            (Array.append
                [|
                    0xFFFFFFFFu
                    0x80000000u
                    0x7F800001u
                    0x7FC00000u
                    0x00000001u
                    0xDEADBEEFu
                |]
                (CaseGen.words r 24))
            "All 32 words arrive, NaN patterns and the top bit included"
    ]

let private groupsEchoCases =
    let echo (groups: uint32) (capacity: int64) (doc: string) =
        Case.create "groups_echo" (sprintf "groups%d" groups)
        |> Case.outputU32 "y" capacity
        |> Case.u32 "groups" groups
        |> Case.doc doc

    [
        echo 1u 1L "One group"
        echo 255u 256L "No folding; one extra element of capacity"
        echo 65535u 65535L "The largest unfolded grid"
        echo 65536u 65536L "Folded to 32768 x 2, no extra group"
        echo 65537u 65540L "Folded to 32769 x 2: one extra group, which writes nothing"
        echo 131071u 131072L "Folded to 43691 x 3: two extra groups"
    ]

let private bf16UnpackCases =
    let unpack (name: string) (x: uint16[]) (doc: string) =
        Case.create "bf16_unpack" name
        |> Case.bf16 "x" x
        |> Case.output "y" (int64 x.Length)
        |> Case.u32 "n" (uint32 x.Length)
        |> Case.doc doc

    let r = CaseGen.rng "bf16_unpack"

    [
        unpack "n1" [| 0x3F80us |] "One element (1.0)"
        unpack "n7_odd" (CaseGen.uniformBf16 r 7 -2.0f 2.0f) "An odd count: the last word holds one element"
        unpack
            "specials"
            [|
                0x7F81us
                0x7FC1us
                0xFFFFus
                0x0001us
                0x8001us
                0x8000us
                0x0000us
                0x7F80us
                0xFF80us
                0x007Fus
            |]
            "Signalling and quiet NaN payloads, denormals, zeros and infinities move bit for bit"
        unpack "n513" (CaseGen.uniformBf16 r 513 -100.0f 100.0f) "Three groups"
        (unpack "n1000_dims" (CaseGen.uniformBf16 r 1000 -1.0f 1.0f) "Rank-2 dims on x"
         |> Case.dims "x" [ 40L; 25L ])
    ]

/// Every selftest case.
let cases: KernelCase list =
    scaleCases @ tableEchoCases @ argsEchoCases @ groupsEchoCases @ bf16UnpackCases
