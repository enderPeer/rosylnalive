namespace Lrm.Gpu.CpuDevice

open System
open Lrm.Core
open Lrm.Gpu.Core

module internal Fail =
    /// Raises GpuContractException with a formatted message (the shared exception of the GPU contracts).
    let contract fmt =
        Printf.kprintf (fun s -> raise (GpuContractException s)) fmt

/// A set of element indices of one token: `Count` runs of `Length` elements, run r starting at Start + r * Stride.
/// One run (Count = 1) is a plain range. Runs may touch or overlap when Stride < Length (copy_rows sources may).
[<Struct>]
type ElementSet =
    {
        Start: int64
        Stride: int64
        Length: int64
        Count: int64
    }

    /// The range [lo, hi).
    static member Range(lo: int64, hi: int64) =
        {
            Start = lo
            Stride = 0L
            Length = max 0L (hi - lo)
            Count = 1L
        }

    /// True when the set holds no element.
    member s.IsEmpty = s.Count <= 0L || s.Length <= 0L

    /// One past the last element (Start for an empty set).
    member s.End =
        if s.IsEmpty then
            s.Start
        else
            s.Start + (s.Count - 1L) * s.Stride + s.Length

    /// The number of elements in the runs, counted with multiplicity.
    member s.Elements = if s.IsEmpty then 0L else s.Count * s.Length

    /// The runs as [lo, hi) ranges, in increasing order (as many as Count).
    member s.Runs =
        let start = s.Start
        let stride = s.Stride
        let length = s.Length
        let count = if s.IsEmpty then 0L else s.Count

        seq {
            for r in 0L .. count - 1L do
                let lo = start + r * stride
                yield struct (lo, lo + length)
        }

    /// True when some run overlaps [lo, hi).
    member s.Overlaps(lo: int64, hi: int64) =
        if s.IsEmpty || hi <= lo || hi <= s.Start || lo >= s.End then
            false
        elif s.Count = 1L || s.Stride <= 0L then
            true
        else
            // Runs r with Start + r*Stride < hi and Start + r*Stride + Length > lo. Integer division truncates, which
            // equals the floor wherever the clamp to 0 does not decide.
            let first = max 0L ((lo - s.Start - s.Length + s.Stride) / s.Stride)
            let last = min (s.Count - 1L) ((hi - 1L - s.Start) / s.Stride)
            first <= last

    /// True when element `i` is in the set.
    member s.Contains(i: int64) = s.Overlaps(i, i + 1L)

/// The elements one dispatch touches in one data token (KernelAbiAndCatalog, tensor access): the field, how the
/// kernel uses it, and the elements, computed from the kernel's u32 fields.
type FieldFootprint =
    {
        Field: string
        Access: Access
        Elements: ElementSet
        /// The minimum element count rule of kernel-abi.md, as written there, for error messages.
        Rule: string
    }

/// The kernel-specific argument rules of KernelAbiAndCatalog ("Tensor access": minimum element counts and the other
/// argument rules), which Dispatch.prepare deliberately leaves to CpuDevice. GPU kernels assume them; CpuDevice raises
/// GpuContractException before it emulates a dispatch that breaks one. The footprints are also what the conformance
/// harness poisons and compares, and what CpuDevice's barrier checks track.
[<RequireQualifiedAccess>]
module KernelRules =
    let private mul (a: int64) (b: int64) =
        try
            Checked.(*) a b
        with :? OverflowException ->
            Fail.contract "the element count %d * %d overflows" a b

    let private add (a: int64) (b: int64) =
        try
            Checked.(+) a b
        with :? OverflowException ->
            Fail.contract "the element count %d + %d overflows" a b

    let private sub (a: int64) (b: int64) = a - b

    let private range lo hi = ElementSet.Range(lo, hi)

    /// The other argument rules of a kernel (kernel-abi.md, "Other argument rules"), checked before any footprint is
    /// computed so that a footprint never divides by zero. `dtype` gives the dtype of a token field.
    let private argumentRules (k: KernelSpec) (v: string -> int64) (dtype: string -> DType option) =
        let fail (rule: string) = Fail.contract "%s: %s" k.Name rule

        match k.Name with
        | "rmsnorm_rows"
        | "argmax_rows" ->
            if v "cols" < 1L then
                fail (sprintf "cols >= 1 is required (cols = %d)" (v "cols"))
        | "rope_apply" ->
            let heads = v "heads"
            let hd = v "head_dim"

            if heads < 1L then
                fail "heads >= 1 is required (heads = 0)"

            if v "rows" % heads <> 0L then
                fail (sprintf "heads must divide rows (rows = %d, heads = %d)" (v "rows") heads)

            if hd < 2L || hd % 2L <> 0L then
                fail (sprintf "head_dim must be even and at least 2 (head_dim = %d)" hd)
        | "copy_rows" ->
            match dtype "src", dtype "dst" with
            | Some a, Some b when a <> b ->
                fail (
                    sprintf "src and dst must have the same dtype (src is %s, dst is %s)" (DType.name a) (DType.name b)
                )
            | _ -> ()

            if v "rows" > 1L && v "cols" > v "dst_stride" then
                fail (
                    sprintf
                        "cols <= dst_stride is required when rows > 1, so destination rows never overlap (cols = %d, dst_stride = %d)"
                        (v "cols")
                        (v "dst_stride")
                )
        | "attention_gqa_causal" ->
            let qh = v "q_heads"
            let kvh = v "kv_heads"

            if kvh < 1L then
                fail "kv_heads >= 1 is required (kv_heads = 0)"

            if qh % kvh <> 0L then
                fail (sprintf "kv_heads must divide q_heads (q_heads = %d, kv_heads = %d)" qh kvh)

            if v "head_dim" < 1L then
                fail "head_dim >= 1 is required (head_dim = 0)"
        | _ -> ()

    /// The footprint of each data token of a dispatch, from the kernel's u32 fields alone (never from the tokens' dims).
    let private footprintOf (k: KernelSpec) (v: string -> int64) : FieldFootprint list =
        let fp (field: string) (access: Access) (elements: ElementSet) (rule: string) =
            {
                Field = field
                Access = access
                Elements = elements
                Rule = rule
            }

        let r = Access.Read
        let w = Access.Write

        match k.Name with
        | "scale_f32" ->
            let n = v "n"
            [ fp "x" r (range 0L n) "x >= n"; fp "y" w (range 0L n) "y >= n" ]
        | "table_echo" -> [ fp "y" w (range 0L 16L) "y >= 16" ]
        | "args_echo" -> [ fp "y" w (range 0L 32L) "y >= 32" ]
        | "groups_echo" -> [ fp "y" w (range 0L (v "groups")) "y >= groups" ]
        | "bf16_unpack" ->
            let n = v "n"
            [ fp "x" r (range 0L n) "x >= n"; fp "y" w (range 0L n) "y >= n" ]
        | "embed_gather_bf16" ->
            let rows = v "rows"
            let cols = v "cols"

            [
                fp "ids" r (range 0L rows) "ids >= rows"
                fp "table" r (range 0L (mul (v "vocab") cols)) "table >= vocab * cols"
                fp "y" w (range 0L (mul rows cols)) "y >= rows * cols"
            ]
        | "rmsnorm_rows" ->
            let e = mul (v "rows") (v "cols")

            [
                fp "x" r (range 0L e) "x >= rows * cols"
                fp "w" r (range 0L (v "cols")) "w >= cols"
                fp "y" w (range 0L e) "y >= rows * cols"
            ]
        | "rope_apply" ->
            let rows = v "rows"
            let hd = v "head_dim"
            let half = hd / 2L
            let pos0 = v "pos0"
            let e = mul rows hd
            let tables = range (mul pos0 half) (mul (add pos0 (rows / v "heads")) half)
            let tableRule = "(pos0 + rows / heads) * (head_dim / 2)"

            [
                fp "x" r (range 0L e) "x >= rows * head_dim"
                fp "cos_table" r tables ("cos_table >= " + tableRule)
                fp "sin_table" r tables ("sin_table >= " + tableRule)
                fp "y" w (range 0L e) "y >= rows * head_dim"
            ]
        | "silu_mul" ->
            let n = v "n"

            [
                fp "gate" r (range 0L n) "gate >= n"
                fp "up" r (range 0L n) "up >= n"
                fp "y" w (range 0L n) "y >= n"
            ]
        | "add" ->
            let n = v "n"

            [
                fp "a" r (range 0L n) "a >= n"
                fp "b" r (range 0L n) "b >= n"
                fp "y" w (range 0L n) "y >= n"
            ]
        | "copy_rows" ->
            let rows = v "rows"
            let cols = v "cols"

            let rowsOf (row0: int64) (stride: int64) =
                // One run when the rows are packed (or there is one row); otherwise one run per row.
                if rows = 1L || stride = cols then
                    range (mul row0 stride) (add (mul (add row0 (rows - 1L)) stride) cols)
                else
                    // Validates that the last run's end fits in int64.
                    add (mul (add row0 (rows - 1L)) stride) cols |> ignore

                    {
                        Start = mul row0 stride
                        Stride = stride
                        Length = cols
                        Count = rows
                    }

            [
                fp "src" r (rowsOf (v "src_row0") (v "src_stride")) "src >= (src_row0 + rows - 1) * src_stride + cols"
                fp "dst" w (rowsOf (v "dst_row0") (v "dst_stride")) "dst >= (dst_row0 + rows - 1) * dst_stride + cols"
            ]
        | "argmax_rows" ->
            let rows = v "rows"

            [
                fp "x" r (range 0L (mul rows (v "cols"))) "x >= rows * cols"
                fp "y" w (range 0L rows) "y >= rows"
            ]
        | "matmul_bf16w" ->
            let m = v "m"
            let n = v "n"
            let kk = v "k"

            [
                fp "x" r (range 0L (mul m kk)) "x >= m * k"
                fp "w" r (range 0L (mul n kk)) "w >= n * k"
                fp "y" w (range 0L (mul m n)) "y >= m * n"
            ]
        | "attention_gqa_causal" ->
            let rows = v "rows"
            let q0 = v "q_row0"
            let qc = mul (v "q_heads") (v "head_dim")
            let kvc = mul (v "kv_heads") (v "head_dim")
            let qs = range (mul q0 qc) (mul (add q0 rows) qc)
            let kv = range 0L (mul (add (v "past") rows) kvc)

            [
                fp "q" r qs "q >= (q_row0 + rows) * q_heads * head_dim"
                fp "k" r kv "k >= (past + rows) * kv_heads * head_dim"
                fp "v" r kv "v >= (past + rows) * kv_heads * head_dim"
                fp "y" w qs "y >= (q_row0 + rows) * q_heads * head_dim"
            ]
        | other -> Fail.contract "CpuDevice has no footprint for kernel %s" other

    let private lookup (k: KernelSpec) (args: ArgBlock) =
        fun (name: string) -> int64 args[k.Word name]

    /// The elements each data token of a dispatch touches, per field, computed in int64 from the kernel's u32 fields.
    /// Raises GpuContractException when an argument rule is broken (so a footprint is never computed from bad
    /// arguments) or a count overflows. Meta tokens have no footprint.
    let footprint (k: KernelSpec) (args: ArgBlock) (dtype: string -> DType option) : FieldFootprint list =
        let v = lookup k args
        argumentRules k v dtype
        footprintOf k v

    /// Checks a dispatch's kernel-specific rules (kernel-abi.md, "Tensor access"): the other argument rules, then that
    /// every data token holds at least the elements the dispatch touches. `count` gives a token field's element count
    /// (the row's word 12) and `name` the tensor's name for messages. Returns the footprint.
    let check
        (k: KernelSpec)
        (args: ArgBlock)
        (count: string -> int64)
        (dtype: string -> DType option)
        (name: string -> string)
        : FieldFootprint list =
        let fps = footprint k args dtype

        for f in fps do
            let need = f.Elements.End
            let have = count f.Field

            if have < need then
                Fail.contract
                    "%s.%s: tensor '%s' holds %d elements, but the dispatch touches %d (rule %s)"
                    k.Name
                    f.Field
                    (name f.Field)
                    have
                    need
                    f.Rule

        fps
