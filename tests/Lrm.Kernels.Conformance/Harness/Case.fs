namespace Lrm.Kernels.Conformance

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core

/// What a case tensor holds (ConformanceHarness contract, CaseIo).
[<RequireQualifiedAccess>]
type CaseData =
    /// An input: its little-endian element bytes. The element count is bytes / element size.
    | Input of bytes: byte[]
    /// An output of `elements` elements (its capacity). The harness poisons it before every run.
    | Output of elements: int64
    /// A tensor only a meta field names: the kernel reads its table row, never its bytes. It has `elements` elements;
    /// its dims are the tensor's Dims.
    | Meta of elements: int64

/// One tensor of a case, bound to a token field of the kernel.
type CaseTensor =
    {
        /// The kernel field (catalog name) the tensor's token is passed in.
        Field: string
        DType: DType
        Data: CaseData
        /// The row's dims; None gives [elements]. Kernels never read the dims of a data token (kernel-abi.md, tensor
        /// access), so a case may give any dims whose product is the element count.
        Dims: int64 list option
    }

    member t.Elements =
        match t.Data with
        | CaseData.Input b -> int64 b.Length / int64 (DType.size t.DType)
        | CaseData.Output n
        | CaseData.Meta n -> n

/// The inputs and outputs of one case (ConformanceHarness contract, CaseIo): one tensor per token field, except that a
/// written field may share a read field's tensor (in place, when the catalog lets it alias), and a meta field may be
/// the null token.
type CaseIo =
    {
        Tensors: CaseTensor list
        /// (written field, read field): the written field is passed the read field's token.
        InPlace: (string * string) list
        /// Meta fields passed the null token.
        NullMeta: string list
    }

/// One conformance case (ConformanceHarness contract, KernelCase): a catalog kernel, a name unique for that kernel, its
/// tensors and its scalar argument words. Everything else (sizes of the work, the reference, the outputs to poison and
/// compare, the batch-invariance splits, the bound) the harness derives from the kernel and these.
type KernelCase =
    {
        Kernel: string
        Name: string
        Io: CaseIo
        /// The u32, i32 and f32 fields, by catalog name, as argument words.
        Scalars: (string * uint32) list
        /// Free text for reports: why the case exists.
        Doc: string
    }

    /// "<kernel>/<name>".
    member c.Id = c.Kernel + "/" + c.Name

/// Builders for cases, so case files read as data:
///
///     Case.create "scale_f32" "n1000"
///     |> Case.f32 "x" xs
///     |> Case.output "y" 1000L
///     |> Case.u32 "n" 1000u
///     |> Case.f32Arg "a" 0.5f
[<RequireQualifiedAccess>]
module Case =
    let create (kernel: string) (name: string) =
        {
            Kernel = kernel
            Name = name
            Io =
                {
                    Tensors = []
                    InPlace = []
                    NullMeta = []
                }
            Scalars = []
            Doc = ""
        }

    let doc (text: string) (c: KernelCase) = { c with Doc = text }

    let tensor (t: CaseTensor) (c: KernelCase) =
        { c with
            Io =
                { c.Io with
                    Tensors = c.Io.Tensors @ [ t ]
                }
        }

    /// An input of raw element bytes.
    let inputBytes (field: string) (dtype: DType) (bytes: byte[]) (c: KernelCase) =
        tensor
            {
                Field = field
                DType = dtype
                Data = CaseData.Input bytes
                Dims = None
            }
            c

    let private bytesOf<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (values: 'T[])
        =
        MemoryMarshal.AsBytes(ReadOnlySpan<'T>(values)).ToArray()

    /// A float32 input.
    let f32 (field: string) (values: float32[]) (c: KernelCase) =
        inputBytes field DType.F32 (bytesOf values) c

    /// A float32 input given as raw bits (for NaN payloads and signalling NaNs, which a float32 value may lose).
    let f32Bits (field: string) (bits: uint32[]) (c: KernelCase) =
        inputBytes field DType.F32 (bytesOf bits) c

    /// A bf16 input (raw bits).
    let bf16 (field: string) (bits: uint16[]) (c: KernelCase) =
        inputBytes field DType.BF16 (bytesOf bits) c

    /// A u32 input.
    let u32s (field: string) (values: uint32[]) (c: KernelCase) =
        inputBytes field DType.U32 (bytesOf values) c

    /// An i32 input.
    let i32s (field: string) (values: int[]) (c: KernelCase) =
        inputBytes field DType.I32 (bytesOf values) c

    /// An output of `elements` elements of the field's first catalog dtype unless `dtype` says otherwise.
    let outputOf (field: string) (dtype: DType) (elements: int64) (c: KernelCase) =
        tensor
            {
                Field = field
                DType = dtype
                Data = CaseData.Output elements
                Dims = None
            }
            c

    /// An f32 output.
    let output (field: string) (elements: int64) (c: KernelCase) = outputOf field DType.F32 elements c

    /// A u32 output.
    let outputU32 (field: string) (elements: int64) (c: KernelCase) = outputOf field DType.U32 elements c

    /// A tensor named only by a meta field.
    let meta (field: string) (dtype: DType) (dims: int64 list) (c: KernelCase) =
        tensor
            {
                Field = field
                DType = dtype
                Data = CaseData.Meta(List.fold (*) 1L dims)
                Dims = Some dims
            }
            c

    /// The null token for a meta field.
    let nullMeta (field: string) (c: KernelCase) =
        { c with
            Io =
                { c.Io with
                    NullMeta = c.Io.NullMeta @ [ field ]
                }
        }

    /// Gives the last tensor added for `field` these dims.
    let dims (field: string) (dims: int64 list) (c: KernelCase) =
        { c with
            Io =
                { c.Io with
                    Tensors =
                        c.Io.Tensors
                        |> List.map (fun t -> if t.Field = field then { t with Dims = Some dims } else t)
                }
        }

    /// The written field is passed the read field's token (in place).
    let inPlace (written: string) (read: string) (c: KernelCase) =
        { c with
            Io =
                { c.Io with
                    InPlace = c.Io.InPlace @ [ written, read ]
                }
        }

    /// A u32 argument.
    let u32 (field: string) (value: uint32) (c: KernelCase) =
        { c with
            Scalars = c.Scalars @ [ field, value ]
        }

    /// An i32 argument.
    let i32 (field: string) (value: int) (c: KernelCase) =
        { c with
            Scalars = c.Scalars @ [ field, uint32 value ]
        }

    /// An f32 argument, as its raw bits.
    let f32Arg (field: string) (value: float32) (c: KernelCase) =
        { c with
            Scalars = c.Scalars @ [ field, BitConverter.SingleToUInt32Bits value ]
        }

    /// An f32 argument given as raw bits.
    let f32ArgBits (field: string) (bits: uint32) (c: KernelCase) =
        { c with
            Scalars = c.Scalars @ [ field, bits ]
        }

/// Deterministic test data for cases (DetRng, so a case's inputs are the same on every machine).
[<RequireQualifiedAccess>]
module CaseGen =
    /// A generator for a named case.
    let rng (name: string) =
        DetRng(0x4C524D2D45314341UL, DetRng.streamOf ("conformance." + name))

    /// Uniform float32 values in [lo, hi), each a multiple of 2^-24 of the span (exact in float32).
    let uniform (r: DetRng) (n: int) (lo: float32) (hi: float32) =
        Array.init n (fun _ -> lo + (hi - lo) * r.NextF32())

    /// bf16 bits of uniform values in [lo, hi), rounded to nearest even.
    let uniformBf16 (r: DetRng) (n: int) (lo: float32) (hi: float32) =
        uniform r n lo hi |> Array.map Bf16.narrowRne

    /// Random u32 words.
    let words (r: DetRng) (n: int) = Array.init n (fun _ -> r.NextU32())
