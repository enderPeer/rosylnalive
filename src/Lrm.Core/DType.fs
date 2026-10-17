namespace Lrm.Core

/// Element types (LrmCore contract). The numeric codes are frozen: they appear in the GPU tensor table, in canonical
/// encodings and in hashes. Code 0 is invalid on purpose, so zeroed memory never reads as a valid type.
[<RequireQualifiedAccess>]
type DType =
    | F32
    | BF16
    | F16
    | I32
    | U32
    | U8
    | F64
    | I64

[<RequireQualifiedAccess>]
module DType =
    /// Every dtype, in code order.
    let all =
        [
            DType.F32
            DType.BF16
            DType.F16
            DType.I32
            DType.U32
            DType.U8
            DType.F64
            DType.I64
        ]

    /// The frozen code: F32 1, BF16 2, F16 3, I32 4, U32 5, U8 6, F64 7, I64 8. Canonical encodings write it as one byte.
    let code (t: DType) : byte =
        match t with
        | DType.F32 -> 1uy
        | DType.BF16 -> 2uy
        | DType.F16 -> 3uy
        | DType.I32 -> 4uy
        | DType.U32 -> 5uy
        | DType.U8 -> 6uy
        | DType.F64 -> 7uy
        | DType.I64 -> 8uy

    /// The dtype with this code; None for 0 (invalid) and every unassigned code.
    let tryOfCode (c: int) : DType option =
        match c with
        | 1 -> Some DType.F32
        | 2 -> Some DType.BF16
        | 3 -> Some DType.F16
        | 4 -> Some DType.I32
        | 5 -> Some DType.U32
        | 6 -> Some DType.U8
        | 7 -> Some DType.F64
        | 8 -> Some DType.I64
        | _ -> None

    /// The dtype with this code; raises ArgumentException for 0 and every unassigned code.
    let ofCode (c: int) =
        match tryOfCode c with
        | Some t -> t
        | None -> invalidArg "c" (sprintf "invalid dtype code %d" c)

    /// Bytes per element.
    let size (t: DType) =
        match t with
        | DType.U8 -> 1
        | DType.BF16
        | DType.F16 -> 2
        | DType.F32
        | DType.I32
        | DType.U32 -> 4
        | DType.F64
        | DType.I64 -> 8

    /// True for the floating-point types.
    let isFloat (t: DType) =
        match t with
        | DType.F32
        | DType.BF16
        | DType.F16
        | DType.F64 -> true
        | DType.I32
        | DType.U32
        | DType.U8
        | DType.I64 -> false

    /// Lowercase name used in logs, reports and manifests: "f32", "bf16", "f16", "i32", "u32", "u8", "f64", "i64".
    let name (t: DType) =
        match t with
        | DType.F32 -> "f32"
        | DType.BF16 -> "bf16"
        | DType.F16 -> "f16"
        | DType.I32 -> "i32"
        | DType.U32 -> "u32"
        | DType.U8 -> "u8"
        | DType.F64 -> "f64"
        | DType.I64 -> "i64"

    /// Parses a lowercase name (ordinal match).
    let tryParse (text: string) =
        all |> List.tryFind (fun t -> Invariant.equals (name t) text)

    /// The dtype string in a safetensors header: "F32", "BF16", "F16", "I32", "U32", "U8", "F64", "I64".
    let safetensorsName (t: DType) = (name t).ToUpperInvariant()

    /// Parses a safetensors dtype string (ordinal match). Types LRM does not use, such as "BOOL", give None.
    let tryOfSafetensorsName (text: string) =
        all |> List.tryFind (fun t -> Invariant.equals (safetensorsName t) text)
