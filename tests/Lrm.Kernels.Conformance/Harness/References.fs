namespace Lrm.Kernels.Conformance

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Tensor
open Lrm.Tensor.CpuReference

/// What a reference is computed from: the kernel, the block as dispatched (word 0 is the physical x), the case's input
/// bytes by read field, the planned rows of the token fields, and the class's measured fp32 denormal mode.
type RefInputs =
    {
        Spec: KernelSpec
        Args: ArgBlock
        Input: string -> byte[]
        Row: string -> TensorRow
        Mode: Fp32DenormalMode
        Exec: CpuExec
    }

    member r.U(name: string) = int64 r.Args[r.Spec.Word name]
    member r.F(name: string) = r.Args.F32(r.Spec.Word name)

/// The reference of one output field: its touched elements (the dispatch's write footprint, in the tensor's element
/// space), CpuRef F32 in the class's mode as one u32 word per touched element in increasing order, CpuRef F64 for the
/// relative kernels, and the elements a mixed class compares within the class only.
type RefOutput =
    {
        Field: string
        Touched: ElementSet
        Bits: uint32[]
        F64: float[] option
        /// Some, in mixed mode, when some elements are left out of the comparison with CpuRef.
        WithinClassOnly: bool[] option
    }

/// One batch-invariance split: the case restricted to some of its rows. `Scalars` override the case's u32 fields;
/// `Views` give a field a view of its tensor (field, element offset, elements), so the restricted dispatch writes the
/// same elements of the same buffers as the whole case does for those rows.
type Split =
    {
        Label: string
        Scalars: (string * uint32) list
        Views: (string * int64 * int64) list
    }

/// The CpuRef reference of every catalog kernel (KernelBounds, "Measuring a case"), its splits for batch invariance,
/// and the class-dependent rules of fp32 denormals (KernelAbiAndCatalog, kernel rules; ADR int-0001).
[<RequireQualifiedAccess>]
module References =
    let isDenormal32 (bits: uint32) =
        bits &&& 0x7F800000u = 0u && bits &&& 0x007FFFFFu <> 0u

    let isDenormal16 (bits: uint16) =
        bits &&& 0x7F80us = 0us && bits &&& 0x007Fus <> 0us

    let flush16 (bits: uint16) =
        if isDenormal16 bits then bits &&& 0x8000us else bits

    let private cast<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
        (bytes: byte[])
        (count: int64)
        : 'T[] =
        let size = Marshal.SizeOf<'T>()

        if int64 bytes.Length < count * int64 size then
            invalidArg "bytes" (sprintf "%d bytes hold fewer than %d elements" bytes.Length count)

        MemoryMarshal.Cast<byte, 'T>(ReadOnlySpan(bytes, 0, int count * size)).ToArray()

    let private bitsOf (xs: float32[]) =
        MemoryMarshal.Cast<float32, uint32>(ReadOnlySpan xs).ToArray()

    let private ofBits (ws: uint32[]) =
        MemoryMarshal.Cast<uint32, float32>(ReadOnlySpan ws).ToArray()

    let private widen (xs: float32[]) = xs |> Array.map float

    /// The kernels whose reference follows the class's denormal mode (KernelBounds): the fp32 arithmetic kernels,
    /// argmax_rows and every relative kernel.
    let followsMode (kernel: string) =
        kernel = "scale_f32"
        || kernel = "add"
        || kernel = "argmax_rows"
        || (KernelBounds.find kernel).Kind = KernelBoundKind.Relative

    type private Inputs(r: RefInputs) =
        let flush = r.Mode = Fp32DenormalMode.Flush

        /// The first `n` float32 bits of an input, flushed in a flush class when `mode`.
        member _.F32Bits(field: string, n: int64, mode: bool) =
            let w = cast<uint32> (r.Input field) n

            if mode && flush then
                w |> Array.map Fp32Denormals.flushBits
            else
                w

        member x.F32(field: string, n: int64, mode: bool) = ofBits (x.F32Bits(field, n, mode))

        member _.Bf16(field: string, n: int64, mode: bool) =
            let h = cast<uint16> (r.Input field) n
            if mode && flush then h |> Array.map flush16 else h

        member _.U32(field: string, n: int64) = cast<uint32> (r.Input field) n

        /// An f32 argument, flushed in a flush class when `mode`.
        member _.Arg(field: string, mode: bool) =
            let b = r.Args[r.Spec.Word field]

            BitConverter.UInt32BitsToSingle(if mode && flush then Fp32Denormals.flushBits b else b)

        /// True when any of the first n float32 (or bf16) elements of an input is a denormal.
        member _.AnyDenormal32(field: string, n: int64) =
            cast<uint32> (r.Input field) n |> Array.exists isDenormal32

        member _.AnyDenormal16(field: string, n: int64) =
            cast<uint16> (r.Input field) n |> Array.exists isDenormal16

    let private output field (touched: ElementSet) bits f64 within =
        {
            Field = field
            Touched = touched
            Bits = bits
            F64 = f64
            WithinClassOnly = within
        }

    let private range (n: int64) = ElementSet.Range(0L, n)

    /// An element-wise fp32 arithmetic kernel (scale_f32, add) over the bits of its one or two inputs and its scalar,
    /// in the class's mode: flush flushes the inputs and the result; mixed leaves out every element whose input or
    /// result is a denormal.
    let private elementwise
        (r: RefInputs)
        (a: uint32[])
        (b: uint32 voption)
        (bs: uint32[])
        (op: float32 -> float32 -> float32)
        =
        let n = a.Length
        let flush = r.Mode = Fp32DenormalMode.Flush

        let mapIn (w: uint32) =
            if flush then Fp32Denormals.flushBits w else w

        let result = Array.zeroCreate<uint32> n

        for i in 0 .. n - 1 do
            let rhs =
                match b with
                | ValueSome s -> s
                | ValueNone -> bs[i]

            let v =
                op (BitConverter.UInt32BitsToSingle(mapIn a[i])) (BitConverter.UInt32BitsToSingle(mapIn rhs))

            let w = BitConverter.SingleToUInt32Bits v
            result[i] <- if flush then Fp32Denormals.flushBits w else w

        let within =
            if r.Mode = Fp32DenormalMode.Mixed then
                Some(
                    Array.init n (fun i ->
                        let rhs =
                            match b with
                            | ValueSome s -> s
                            | ValueNone -> bs[i]

                        isDenormal32 result[i] || isDenormal32 a[i] || isDenormal32 rhs)
                )
            else
                None

        output "y" (range (int64 n)) result None within

    /// The reference of one dispatch's outputs.
    let compute (r: RefInputs) : RefOutput list =
        let k = r.Spec
        let inp = Inputs r
        let mixed = r.Mode = Fp32DenormalMode.Mixed
        let exec = r.Exec

        let relative (field: string) (touched: ElementSet) (f32: float32[]) (f64: float[]) (denormalInputs: bool) =
            output
                field
                touched
                (bitsOf f32)
                (Some f64)
                (if mixed && denormalInputs then
                     Some(Array.create f32.Length true)
                 else
                     None)

        match k.Name with
        | "scale_f32" ->
            let n = r.U "n"

            [
                elementwise r (inp.U32("x", n)) (ValueSome r.Args[k.Word "a"]) [||] (fun x a -> x * a)
            ]
        | "add" ->
            let n = r.U "n"

            [
                elementwise r (inp.U32("a", n)) ValueNone (inp.U32("b", n)) (fun a b -> a + b)
            ]
        | "table_echo" ->
            let row = r.Row "src"
            [ output "y" (range 16L) (row.ToWords()) None None ]
        | "args_echo" -> [ output "y" (range 32L) (r.Args.ToWords()) None None ]
        | "groups_echo" ->
            let g = r.U "groups"
            [ output "y" (range g) (Array.init (int g) uint32) None None ]
        | "bf16_unpack" ->
            let n = r.U "n"
            let x = inp.Bf16("x", n, false)
            [ output "y" (range n) (x |> Array.map Bf16.widenBits) None None ]
        | "embed_gather_bf16" ->
            let rows = r.U "rows"
            let cols = r.U "cols"
            let vocab = r.U "vocab"

            let table =
                HostTensor.bf16 (Shape.matrix (int vocab) (int cols)) (inp.Bf16("table", vocab * cols, false))

            let y = CpuRef.embedGather<float32> exec table (inp.U32("ids", rows))
            [ output "y" (range (rows * cols)) (bitsOf y.Data) None None ]
        | "copy_rows" ->
            let rows = r.U "rows"
            let cols = r.U "cols"
            let srcRow0 = r.U "src_row0"
            let srcStride = r.U "src_stride"
            let need = (srcRow0 + rows - 1L) * srcStride + cols
            let src = inp.U32("src", need)

            let bits =
                Array.init (int (rows * cols)) (fun e ->
                    let row = int64 e / cols
                    let c = int64 e % cols
                    src[int ((srcRow0 + row) * srcStride + c)])

            let touched =
                let dst0 = r.U "dst_row0"
                let stride = r.U "dst_stride"

                if rows = 1L || stride = cols then
                    ElementSet.Range(dst0 * stride, (dst0 + rows - 1L) * stride + cols)
                else
                    {
                        Start = dst0 * stride
                        Stride = stride
                        Length = cols
                        Count = rows
                    }

            [ output "dst" touched bits None None ]
        | "argmax_rows" ->
            let rows = r.U "rows"
            let cols = r.U "cols"
            let x = inp.F32Bits("x", rows * cols, true)

            let y =
                CpuRef.argmaxRows<float32>(HostTensor.f32 (Shape.matrix (int rows) (int cols)) (ofBits x))

            let within =
                if mixed then
                    let raw = inp.U32("x", rows * cols)

                    Some(
                        Array.init (int rows) (fun row ->
                            Array.sub raw (row * int cols) (int cols) |> Array.exists isDenormal32)
                    )
                else
                    None

            [ output "y" (range rows) y None within ]
        | "rmsnorm_rows" ->
            let rows = int (r.U "rows")
            let cols = int (r.U "cols")
            let n = int64 rows * int64 cols
            let x = inp.F32("x", n, true)
            let w = HostTensor.bf16 (Shape.vector cols) (inp.Bf16("w", int64 cols, true))
            let eps = inp.Arg("eps", true)
            let shape = Shape.matrix rows cols
            let y32 = CpuRef.rmsNorm<float32> exec (HostTensor.f32 shape x) w eps
            let y64 = CpuRef.rmsNorm<float> exec (HostTensor.f64 shape (widen x)) w (float eps)
            let den = inp.AnyDenormal32("x", n) || inp.AnyDenormal16("w", int64 cols)
            [ relative "y" (range n) y32.Data y64.Data den ]
        | "rope_apply" ->
            let rows = r.U "rows"
            let heads = r.U "heads"
            let hd = r.U "head_dim"
            let half = hd / 2L
            let pos0 = r.U "pos0"
            let tableN = (pos0 + rows / heads) * half
            let x = inp.F32("x", rows * hd, true)
            let cos = inp.F32("cos_table", tableN, true)
            let sin = inp.F32("sin_table", tableN, true)
            let y32 = Array.zeroCreate<float32>(int (rows * hd))
            let y64 = Array.zeroCreate<float>(int (rows * hd))
            let x64 = widen x

            for row in 0L .. rows - 1L do
                let pos = pos0 + row / heads
                let at = int (row * hd)
                let c = ReadOnlySpan(cos, int (pos * half), int half)
                let s = ReadOnlySpan(sin, int (pos * half), int half)
                Rows.ropeHead<float32> (ReadOnlySpan(x, at, int hd)) c s (Span(y32, at, int hd))
                Rows.ropeHead<float> (ReadOnlySpan(x64, at, int hd)) c s (Span(y64, at, int hd))

            let den =
                inp.AnyDenormal32("x", rows * hd)
                || inp.AnyDenormal32("cos_table", tableN)
                || inp.AnyDenormal32("sin_table", tableN)

            [ relative "y" (range (rows * hd)) y32 y64 den ]
        | "silu_mul" ->
            let n = r.U "n"
            let shape = Shape.vector (int n)
            let g = inp.F32("gate", n, true)
            let u = inp.F32("up", n, true)

            let y32 =
                CpuRef.siluMul<float32> exec (HostTensor.f32 shape g) (HostTensor.f32 shape u)

            let y64 =
                CpuRef.siluMul<float> exec (HostTensor.f64 shape (widen g)) (HostTensor.f64 shape (widen u))

            let den = inp.AnyDenormal32("gate", n) || inp.AnyDenormal32("up", n)
            [ relative "y" (range n) y32.Data y64.Data den ]
        | "matmul_bf16w" ->
            let m = int (r.U "m")
            let n = int (r.U "n")
            let kk = int (r.U "k")
            let x = inp.F32("x", int64 m * int64 kk, true)

            let w =
                HostTensor.bf16 (Shape.matrix n kk) (inp.Bf16("w", int64 n * int64 kk, true))

            let shape = Shape.matrix m kk
            let y32 = CpuRef.matmulBf16W<float32> exec (HostTensor.f32 shape x) w
            let y64 = CpuRef.matmulBf16W<float> exec (HostTensor.f64 shape (widen x)) w

            let den =
                inp.AnyDenormal32("x", int64 m * int64 kk)
                || inp.AnyDenormal16("w", int64 n * int64 kk)

            [ relative "y" (range (int64 m * int64 n)) y32.Data y64.Data den ]
        | "attention_gqa_causal" ->
            let rows = int (r.U "rows")
            let q0 = int (r.U "q_row0")
            let past = int (r.U "past")
            let qh = int (r.U "q_heads")
            let kvh = int (r.U "kv_heads")
            let hd = int (r.U "head_dim")
            let qc = qh * hd
            let kvc = kvh * hd
            let tk = past + rows
            let qAll = inp.F32("q", int64 (q0 + rows) * int64 qc, true)
            let q = Array.sub qAll (q0 * qc) (rows * qc)
            let kc = inp.F32("k", int64 tk * int64 kvc, true)
            let vc = inp.F32("v", int64 tk * int64 kvc, true)
            let scale = inp.Arg("scale", true)

            let shape =
                {
                    QHeads = qh
                    KvHeads = kvh
                    HeadDim = hd
                }

            let qs = Shape.matrix rows qc
            let ks = Shape.matrix tk kvc

            let y32 =
                CpuRef.attention<float32>
                    exec
                    (HostTensor.f32 qs q)
                    (HostTensor.f32 ks kc)
                    (HostTensor.f32 ks vc)
                    shape
                    past
                    scale

            let y64 =
                CpuRef.attention<float>
                    exec
                    (HostTensor.f64 qs (widen q))
                    (HostTensor.f64 ks (widen kc))
                    (HostTensor.f64 ks (widen vc))
                    shape
                    past
                    (float scale)

            let den =
                q |> Array.exists (BitConverter.SingleToUInt32Bits >> isDenormal32)
                || inp.AnyDenormal32("k", int64 tk * int64 kvc)
                || inp.AnyDenormal32("v", int64 tk * int64 kvc)

            [
                relative "y" (ElementSet.Range(int64 q0 * int64 qc, int64 (q0 + rows) * int64 qc)) y32.Data y64.Data den
            ]
        | other -> invalidArg "r" (sprintf "the harness has no reference for kernel %s" other)

    /// The rows of a case for batch invariance, and how to restrict it to rows [r0, r1). None for kernels without
    /// rows (table_echo, args_echo).
    let private rowsOf (k: KernelSpec) (v: string -> int64) : (int64 * (int64 -> int64 -> Split)) option =
        let views (fields: (string * int64) list) r0 r1 =
            fields |> List.map (fun (f, per) -> f, r0 * per, (r1 - r0) * per)

        let split label scalars vs =
            {
                Label = label
                Scalars = scalars
                Views = vs
            }

        let lbl (r0: int64) (r1: int64) =
            if r1 - r0 = 1L then
                Invariant.format $"row {r0} alone"
            else
                Invariant.format $"rows {r0} to {r1 - 1L}"

        match k.Name with
        | "scale_f32"
        | "bf16_unpack" ->
            Some(v "n", (fun r0 r1 -> split (lbl r0 r1) [ "n", uint32 (r1 - r0) ] (views [ "x", 1L; "y", 1L ] r0 r1)))
        | "silu_mul" ->
            Some(
                v "n",
                fun r0 r1 -> split (lbl r0 r1) [ "n", uint32 (r1 - r0) ] (views [ "gate", 1L; "up", 1L; "y", 1L ] r0 r1)
            )
        | "add" ->
            Some(
                v "n",
                fun r0 r1 -> split (lbl r0 r1) [ "n", uint32 (r1 - r0) ] (views [ "a", 1L; "b", 1L; "y", 1L ] r0 r1)
            )
        | "groups_echo" ->
            Some(v "groups", (fun _ r1 -> split (Invariant.format $"groups 0 to {r1 - 1L}") [ "groups", uint32 r1 ] []))
        | "embed_gather_bf16" ->
            let cols = v "cols"

            Some(
                v "rows",
                (fun r0 r1 -> split (lbl r0 r1) [ "rows", uint32 (r1 - r0) ] (views [ "ids", 1L; "y", cols ] r0 r1))
            )
        | "rmsnorm_rows" ->
            let cols = v "cols"

            Some(
                v "rows",
                (fun r0 r1 -> split (lbl r0 r1) [ "rows", uint32 (r1 - r0) ] (views [ "x", cols; "y", cols ] r0 r1))
            )
        | "argmax_rows" ->
            let cols = v "cols"

            Some(
                v "rows",
                (fun r0 r1 -> split (lbl r0 r1) [ "rows", uint32 (r1 - r0) ] (views [ "x", cols; "y", 1L ] r0 r1))
            )
        | "rope_apply" ->
            let heads = v "heads"
            let per = heads * v "head_dim"

            Some(
                v "rows" / heads,
                fun t0 t1 ->
                    split
                        (if t1 - t0 = 1L then
                             Invariant.format $"token {t0} alone"
                         else
                             Invariant.format $"tokens {t0} to {t1 - 1L}")
                        [ "rows", uint32 ((t1 - t0) * heads); "pos0", uint32 (v "pos0" + t0) ]
                        (views [ "x", per; "y", per ] t0 t1)
            )
        | "copy_rows" ->
            Some(
                v "rows",
                fun r0 r1 ->
                    split
                        (lbl r0 r1)
                        [
                            "rows", uint32 (r1 - r0)
                            "src_row0", uint32 (v "src_row0" + r0)
                            "dst_row0", uint32 (v "dst_row0" + r0)
                        ]
                        []
            )
        | "matmul_bf16w" ->
            let kk = v "k"
            let n = v "n"
            let fields = if kk = 0L then [ "y", n ] else [ "x", kk; "y", n ]

            Some(
                v "m",
                (fun r0 r1 ->
                    split
                        (Invariant.format $"m = {r1 - r0} from row {r0}")
                        [ "m", uint32 (r1 - r0) ]
                        (views fields r0 r1))
            )
        | "attention_gqa_causal" ->
            Some(
                v "rows",
                fun r0 r1 ->
                    let position = v "past" + r0

                    split
                        (if r1 - r0 = 1L then
                             Invariant.format $"query {r0} alone (decode at position {position})"
                         else
                             Invariant.format $"queries {r0} to {r1 - 1L} (chunked prefill)")
                        [
                            "rows", uint32 (r1 - r0)
                            "q_row0", uint32 (v "q_row0" + r0)
                            "past", uint32 (v "past" + r0)
                        ]
                        []
            )
        | _ -> None

    /// The batch-invariance splits of a case (BatchInvariance contract): the first row alone, the last row alone and,
    /// with at least four rows, the middle third; for groups_echo, prefixes of the groups. A split's starts are even
    /// where a bf16 field gets a view, so the view stays 4-byte aligned. Empty for kernels without rows and for cases
    /// with one row.
    let splits (k: KernelSpec) (args: ArgBlock) : Split list =
        let v (name: string) = int64 args[k.Word name]

        match rowsOf k v with
        | None -> []
        | Some(rows, _) when rows < 2L -> []
        | Some(rows, restrict) ->
            let even = k.Name = "bf16_unpack"
            let down (x: int64) = if even then x &&& ~~~1L else x

            let ranges =
                if k.Name = "groups_echo" then
                    [ 0L, 1L; 0L, max 1L (rows / 2L) ]
                else
                    [
                        0L, 1L
                        down (rows - 1L), rows
                        if rows >= 4L then
                            down (rows / 3L), (2L * rows) / 3L
                    ]

            ranges
            |> List.filter (fun (r0, r1) -> r1 > r0 && not (r0 = 0L && r1 = rows))
            |> List.distinct
            |> List.map (fun (r0, r1) -> restrict r0 r1)
