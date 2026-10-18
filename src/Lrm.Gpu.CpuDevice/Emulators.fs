namespace Lrm.Gpu.CpuDevice

open System
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Tensor.CpuReference

/// One data token as an emulator sees it: `Count` elements of `DType` starting at byte `Offset` of `Bytes` (a page, or
/// a flushed copy of the token's elements). Element i is at Offset + i * size (kernel-abi.md, tensor access).
[<Struct>]
type internal TokenView =
    {
        Bytes: byte[]
        Offset: int
        Count: int
        DType: DType
    }

    member private v.Span(size: int) =
        Span<byte>(v.Bytes, v.Offset, v.Count * size)

    member private v.ReadOnly(size: int) =
        ReadOnlySpan<byte>(v.Bytes, v.Offset, v.Count * size)

    member v.F32: Span<float32> = MemoryMarshal.Cast<byte, float32>(v.Span 4)
    member v.RF32: ReadOnlySpan<float32> = MemoryMarshal.Cast<byte, float32>(v.ReadOnly 4)
    member v.U32: Span<uint32> = MemoryMarshal.Cast<byte, uint32>(v.Span 4)
    member v.RU32: ReadOnlySpan<uint32> = MemoryMarshal.Cast<byte, uint32>(v.ReadOnly 4)
    member v.RBf16: ReadOnlySpan<uint16> = MemoryMarshal.Cast<byte, uint16>(v.ReadOnly 2)

/// What an emulator gets for one dispatch: the decoded argument block (word 0 is the physical x), the physical grid,
/// the data tokens by field and the decoded rows of meta tokens.
type internal EmuContext =
    {
        Spec: KernelSpec
        Args: ArgBlock
        Grid: PhysicalGrid
        View: string -> TokenView
        MetaRow: string -> TensorRow
    }

    member c.U(name: string) = c.Args[c.Spec.Word name]
    member c.I64(name: string) = int64 c.Args[c.Spec.Word name]
    member c.F(name: string) = c.Args.F32(c.Spec.Word name)

    member c.ThreadsX =
        let x, _, _ = c.Spec.Threads
        x

    /// The logical group of physical group (gx, gy) of a linear kernel: gy * lrm_groups_x + gx (LRM_GROUP_LINEAR).
    member c.Linear(gx: int, gy: int) =
        int64 gy * int64 c.Args.GroupsX + int64 gx

/// One emulator per catalog kernel (KernelAbiAndCatalog). Each runs one physical thread group, as the HLSL kernel would:
/// it derives its logical group exactly as the kernel does (LRM_GROUP_LINEAR on linear kernels, so a folded grid's extra
/// groups find no work and return), and computes its elements with CpuRef F32's row kernels (Lrm.Tensor.CpuReference
/// Rows), so CpuDevice equals CpuRef F32 bit for bit. Kernels that only move bits do it with integer operations, so
/// every pattern (NaN payloads, denormals) passes unchanged. Read tokens are reached only through read-only spans.
[<RequireQualifiedAccess>]
module internal Emulators =
    let private scaleF32 (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let g = c.Linear(gx, gy)
        let n = c.I64 "n"
        let a = c.F "a"
        let x = (c.View "x").RF32
        let y = (c.View "y").F32

        for t in 0 .. c.ThreadsX - 1 do
            let i = g * int64 c.ThreadsX + int64 t

            if i < n then
                y[int i] <- x[int i] * a

    let private tableEcho (c: EmuContext) (_: int) (_: int) (_: int) =
        let row = c.MetaRow "src"
        let y = (c.View "y").U32

        for t in 0 .. c.ThreadsX - 1 do
            y[t] <- row.Word t

    let private argsEcho (c: EmuContext) (_: int) (_: int) (_: int) =
        let y = (c.View "y").U32

        for t in 0 .. c.ThreadsX - 1 do
            y[t] <- c.Args[t]

    let private groupsEcho (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let g = c.Linear(gx, gy)

        if g < c.I64 "groups" then
            (c.View "y").U32[ int g ] <- uint32 g

    let private bf16Unpack (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let g = c.Linear(gx, gy)
        let n = c.I64 "n"
        let x = (c.View "x").RBf16
        let y = (c.View "y").U32

        for t in 0 .. c.ThreadsX - 1 do
            let i = g * int64 c.ThreadsX + int64 t

            if i < n then
                y[int i] <- Bf16.widenBits x[int i]

    let private embedGather (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let r = int64 gy
        let cols = c.I64 "cols"
        let vocab = c.I64 "vocab"
        let ids = (c.View "ids").RU32
        let table = (c.View "table").RBf16
        let y = (c.View "y").U32

        if r < c.I64 "rows" then
            let id = int64 ids[int r]

            for t in 0 .. c.ThreadsX - 1 do
                let col = int64 gx * int64 c.ThreadsX + int64 t

                if col < cols then
                    y[int (r * cols + col)] <-
                        if id < vocab then
                            Bf16.widenBits table[int (id * cols + col)]
                        else
                            0x7FC00000u

    let private rmsnormRows (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let r = c.Linear(gx, gy)
        let cols = c.I64 "cols"

        if r < c.I64 "rows" then
            let x = (c.View "x").RF32
            let w = (c.View "w").RBf16
            let y = (c.View "y").F32
            let at = int (r * cols)
            Rows.rmsNorm<float32> (x.Slice(at, int cols)) (w.Slice(0, int cols)) (c.F "eps") (y.Slice(at, int cols))

    let private ropeApply (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let r = c.Linear(gx, gy)

        if r < c.I64 "rows" then
            let hd = c.I64 "head_dim"
            let half = hd / 2L
            let pos = c.I64 "pos0" + r / c.I64 "heads"
            let x = (c.View "x").RF32
            let cos = (c.View "cos_table").RF32
            let sin = (c.View "sin_table").RF32
            let y = (c.View "y").F32

            Rows.ropeHead<float32>
                (x.Slice(int (r * hd), int hd))
                (cos.Slice(int (pos * half), int half))
                (sin.Slice(int (pos * half), int half))
                (y.Slice(int (r * hd), int hd))

    /// The elements [lo, hi) of group g of an element-wise kernel with ThreadsX threads, or an empty range.
    let private elements (c: EmuContext) (g: int64) (n: int64) =
        let lo = g * int64 c.ThreadsX
        let hi = min n (lo + int64 c.ThreadsX)
        struct (int lo, int (max 0L (hi - lo)))

    let private siluMul (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let struct (lo, len) = elements c (c.Linear(gx, gy)) (c.I64 "n")

        if len > 0 then
            Rows.siluMul<float32>
                ((c.View "gate").RF32.Slice(lo, len))
                ((c.View "up").RF32.Slice(lo, len))
                ((c.View "y").F32.Slice(lo, len))

    let private add (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let struct (lo, len) = elements c (c.Linear(gx, gy)) (c.I64 "n")

        if len > 0 then
            Rows.add<float32>
                ((c.View "a").RF32.Slice(lo, len))
                ((c.View "b").RF32.Slice(lo, len))
                ((c.View "y").F32.Slice(lo, len))

    let private copyRows (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let g = c.Linear(gx, gy)
        let rows = c.I64 "rows"
        let cols = c.I64 "cols"
        let src = (c.View "src").RU32
        let dst = (c.View "dst").U32

        for t in 0 .. c.ThreadsX - 1 do
            let e = g * int64 c.ThreadsX + int64 t

            if e < rows * cols then
                let r = e / cols
                let col = e % cols

                dst[int ((c.I64 "dst_row0" + r) * c.I64 "dst_stride" + col)] <-
                    src[int ((c.I64 "src_row0" + r) * c.I64 "src_stride" + col)]

    let private argmaxRows (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let r = c.Linear(gx, gy)
        let cols = c.I64 "cols"

        if r < c.I64 "rows" then
            let x = (c.View "x").RF32

            (c.View "y").U32[ int r ] <-
                match Rows.tryArgmax<float32>(x.Slice(int (r * cols), int cols)) with
                | ValueSome i -> uint32 i
                | ValueNone -> UInt32.MaxValue

    let private matmulBf16w (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let m = c.I64 "m"
        let n = c.I64 "n"
        let k = c.I64 "k"
        let x = (c.View "x").RF32
        let w = (c.View "w").RBf16
        let y = (c.View "y").F32
        let i0 = int64 gy * 64L
        let j0 = int64 gx * 64L

        for i in i0 .. (min m (i0 + 64L)) - 1L do
            let xi = x.Slice(int (i * k), int k)

            for j in j0 .. (min n (j0 + 64L)) - 1L do
                y[int (i * n + j)] <- Rows.dotBf16<float32> xi (w.Slice(int (j * k), int k))

    let private attention (c: EmuContext) (gx: int) (gy: int) (_: int) =
        let i = int64 gx
        let h = int64 gy
        let qh = c.I64 "q_heads"

        if i < c.I64 "rows" && h < qh then
            let hd = c.I64 "head_dim"
            let kvh = c.I64 "kv_heads"
            let qc = qh * hd
            let kvc = kvh * hd
            let keys = c.I64 "past" + i + 1L
            let at = (c.I64 "q_row0" + i) * qc + h * hd
            let scores = Array.zeroCreate<float32>(int keys)

            Rows.attention<float32>
                ((c.View "q").RF32.Slice(int at, int hd))
                (c.View "k").RF32
                (c.View "v").RF32
                (int kvc)
                (int ((h / (qh / kvh)) * hd))
                (int keys)
                (c.F "scale")
                (Span<float32>(scores))
                ((c.View "y").F32.Slice(int at, int hd))

    /// The emulator of every catalog kernel, by name.
    let byName: Map<string, EmuContext -> int -> int -> int -> unit> =
        Map.ofList
            [
                "scale_f32", scaleF32
                "table_echo", tableEcho
                "args_echo", argsEcho
                "groups_echo", groupsEcho
                "bf16_unpack", bf16Unpack
                "embed_gather_bf16", embedGather
                "rmsnorm_rows", rmsnormRows
                "rope_apply", ropeApply
                "silu_mul", siluMul
                "add", add
                "copy_rows", copyRows
                "argmax_rows", argmaxRows
                "matmul_bf16w", matmulBf16w
                "attention_gqa_causal", attention
            ]

    /// The kernels that compute on fp32 values (KernelBounds): a flushing model flushes their fp32 and bf16 inputs and
    /// fp32 scalar arguments, and flushes and canonicalizes their fp32 results. The others only move bits.
    let arithmetic =
        set
            [
                "scale_f32"
                "add"
                "rmsnorm_rows"
                "rope_apply"
                "silu_mul"
                "argmax_rows"
                "matmul_bf16w"
                "attention_gqa_causal"
            ]
