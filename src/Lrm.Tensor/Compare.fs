namespace Lrm.Tensor

open System
open Lrm.Core

/// How a measured value is judged against its bound (ComparatorsAndGoldenReader contract).
[<RequireQualifiedAccess>]
type BoundOp =
    /// Passes when measured <= bound: errors, distances and divergences.
    | AtMost
    /// Passes when measured >= bound: similarities, agreements and margins.
    | AtLeast

[<RequireQualifiedAccess>]
module BoundOp =
    /// "le" or "ge", as written to telemetry.
    let name =
        function
        | BoundOp.AtMost -> "le"
        | BoundOp.AtLeast -> "ge"

/// The outcome of one comparison (ComparatorsAndGoldenReader contract). A NaN measurement never passes.
type CheckResult =
    {
        /// The metric name: relL2, maxAbs, maxUlp, cosine, kl, topKAgreement, top2Margin or exact.
        Metric: string
        Measured: float
        Bound: float
        Op: BoundOp
        Passed: bool
        /// Elements compared (rows for the row metrics).
        Elements: int64
        /// The caller's context: pair, tap, layer, prompt, op and so on.
        Attrs: Attr list
    }

    override r.ToString() =
        sprintf
            "%s %s %s %s: %s"
            r.Metric
            (Invariant.f64 r.Measured)
            (match r.Op with
             | BoundOp.AtMost -> "<="
             | BoundOp.AtLeast -> ">=")
            (Invariant.f64 r.Bound)
            (if r.Passed then "pass" else "FAIL")

/// The metrics as pure float64 functions (ComparatorsAndGoldenReader contract). Every sum runs sequentially in
/// increasing index order in float64, so a metric is the same number on every machine and at every thread count. The
/// reference is always the second argument.
[<RequireQualifiedAccess>]
module Metrics =
    let private sameLength (a: ReadOnlySpan<'a>) (b: ReadOnlySpan<'b>) =
        if a.Length <> b.Length then
            invalidArg "reference" (sprintf "lengths differ: %d and %d" a.Length b.Length)

    /// ||candidate - reference||2 / ||reference||2. 0 when both are all zero, +inf when only the reference is. NaN when
    /// either holds a NaN (or infinities that cancel).
    let relL2 (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) =
        sameLength candidate reference
        let mutable num = 0.0
        let mutable den = 0.0

        for i in 0 .. candidate.Length - 1 do
            let d = candidate[i] - reference[i]
            num <- num + d * d
            den <- den + reference[i] * reference[i]

        if Double.IsNaN num || Double.IsNaN den then nan
        elif den = 0.0 then (if num = 0.0 then 0.0 else infinity)
        else sqrt num / sqrt den

    /// max |candidate - reference|; 0 for empty input, NaN when any difference is NaN.
    let maxAbs (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) =
        sameLength candidate reference
        let mutable m = 0.0
        let mutable sawNaN = false

        for i in 0 .. candidate.Length - 1 do
            let d = abs (candidate[i] - reference[i])

            if Double.IsNaN d then
                sawNaN <- true
            elif d > m then
                m <- d

        if sawNaN then nan else m

    /// The position of a float32 on the ordered integer line (+0 and -0 both map to 0).
    let private ordered32 (x: float32) =
        let i = BitConverter.SingleToInt32Bits x
        if i >= 0 then int64 i else -(int64 (i &&& 0x7FFFFFFF))

    let private ordered64 (x: float) =
        let i = BitConverter.DoubleToInt64Bits x

        if i >= 0L then
            Int128.op_Implicit i
        else
            -(Int128.op_Implicit (i &&& 0x7FFFFFFFFFFFFFFFL))

    /// The distance in float32 units in the last place: the number of float32 values between a and b, plus one; 0 for
    /// equal values and for +0 and -0. +inf when either is NaN.
    let ulpDistance32 (a: float32) (b: float32) =
        if Single.IsNaN a || Single.IsNaN b then
            infinity
        else
            float (abs (ordered32 a - ordered32 b))

    /// The distance in float64 units in the last place (as for ulpDistance32).
    let ulpDistance64 (a: float) (b: float) =
        if Double.IsNaN a || Double.IsNaN b then
            infinity
        else
            float (Int128.Abs(ordered64 a - ordered64 b))

    /// The distance in bfloat16 units in the last place, on raw bits (as for ulpDistance32).
    let ulpDistanceBf16 (a: uint16) (b: uint16) =
        if Bf16.isNaN a || Bf16.isNaN b then
            infinity
        else
            let ordered (x: uint16) =
                if x &&& 0x8000us = 0us then
                    int x
                else
                    -(int (x &&& 0x7FFFus))

            float (abs (ordered a - ordered b))

    /// Largest float32 ulp distance.
    let maxUlp32 (candidate: ReadOnlySpan<float32>) (reference: ReadOnlySpan<float32>) =
        sameLength candidate reference
        let mutable m = 0.0

        for i in 0 .. candidate.Length - 1 do
            m <- max m (ulpDistance32 candidate[i] reference[i])

        m

    /// Largest float64 ulp distance.
    let maxUlp64 (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) =
        sameLength candidate reference
        let mutable m = 0.0

        for i in 0 .. candidate.Length - 1 do
            m <- max m (ulpDistance64 candidate[i] reference[i])

        m

    /// a.b / (||a|| ||b||); 1 when both are all zero, 0 when exactly one is.
    let cosine (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) =
        sameLength candidate reference
        let mutable ab = 0.0
        let mutable aa = 0.0
        let mutable bb = 0.0

        for i in 0 .. candidate.Length - 1 do
            ab <- ab + candidate[i] * reference[i]
            aa <- aa + candidate[i] * candidate[i]
            bb <- bb + reference[i] * reference[i]

        if aa = 0.0 && bb = 0.0 then 1.0
        elif aa = 0.0 || bb = 0.0 then 0.0
        else ab / (sqrt aa * sqrt bb)

    /// log softmax of one row in float64: x - (m + log sum exp(x - m)), m the row maximum.
    let logSoftmax (row: ReadOnlySpan<float>) : float[] =
        if row.Length = 0 then
            invalidArg "row" "empty row"

        let mutable m = Double.NegativeInfinity
        let mutable sawNaN = false

        for x in row do
            if Double.IsNaN x then
                sawNaN <- true
            elif x > m then
                m <- x

        if sawNaN then
            Array.create row.Length nan
        else
            let mutable s = 0.0

            for x in row do
                s <- s + exp (x - m)

            let lse = m + log s
            let r = Array.zeroCreate row.Length

            for i in 0 .. row.Length - 1 do
                r[i] <- row[i] - lse

            r

    /// KL(P_reference || P_candidate) of one row of logits: sum p_i (log p_i - log q_i) with p and q the softmax of the
    /// reference and candidate rows, in float64.
    let klRow (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) =
        sameLength candidate reference
        let lq = logSoftmax candidate
        let lp = logSoftmax reference
        let mutable s = 0.0

        for i in 0 .. lp.Length - 1 do
            let p = exp lp[i]

            if p > 0.0 then
                s <- s + p * (lp[i] - lq[i])

        if Array.exists Double.IsNaN lq || Array.exists Double.IsNaN lp then
            nan
        else
            s

    /// Indices of the k largest values of a row, largest first; equal values keep index order (lowest first).
    /// Raises ArgumentException on NaN.
    let topK (row: ReadOnlySpan<float>) (k: int) : int[] =
        if k < 1 || k > row.Length then
            invalidArg "k" (sprintf "k = %d for a row of %d" k row.Length)

        for x in row do
            if Double.IsNaN x then
                invalidArg "row" "NaN in a row"

        let values = row.ToArray()
        let idx = Array.init values.Length id

        Array.sortInPlaceWith
            (fun a b ->
                let c = compare values[b] values[a]
                if c <> 0 then c else compare a b)
            idx

        Array.sub idx 0 k

    /// |topK(candidate) intersect topK(reference)| / k for one row.
    let topKAgreementRow (candidate: ReadOnlySpan<float>) (reference: ReadOnlySpan<float>) (k: int) =
        sameLength candidate reference
        let a = topK candidate k |> Set.ofArray
        let b = topK reference k |> Set.ofArray
        float (Set.count (Set.intersect a b)) / float k

    /// The top-1 value minus the top-2 value of one row (at least 2 elements), in float64. Raises on NaN.
    let top2MarginRow (row: ReadOnlySpan<float>) =
        if row.Length < 2 then
            invalidArg "row" "a margin needs at least 2 elements"

        let mutable a = Double.NegativeInfinity
        let mutable b = Double.NegativeInfinity

        for x in row do
            if Double.IsNaN x then
                invalidArg "row" "NaN in a row"

            if x > a then
                b <- a
                a <- x
            elif x > b then
                b <- x

        a - b

/// Comparisons that return a CheckResult and emit it as a golden.compare Check (ComparatorsAndGoldenReader and
/// TelemetryFacade contracts). Candidate first, reference second; shapes must be equal. Row metrics treat a tensor as
/// [Rows, Cols] (the first dimension and the product of the rest) and raise ArgumentException for rank 0 and 1.
[<RequireQualifiedAccess>]
module Compare =
    /// Builds the result for a measured value and emits golden.compare with metric, measured, bound, direction and elements,
    /// then the caller's attributes.
    let check (metric: string) (op: BoundOp) (measured: float) (bound: float) (elements: int64) (attrs: Attr list) =
        let passed =
            not (Double.IsNaN measured)
            && (match op with
                | BoundOp.AtMost -> measured <= bound
                | BoundOp.AtLeast -> measured >= bound)

        Telemetry.check
            TelemetryNames.GoldenCompare
            passed
            ([
                Attr.str "metric" metric
                Attr.float "measured" measured
                Attr.float "bound" bound
                Attr.str "direction" (BoundOp.name op)
                Attr.int64 "elements" elements
             ]
             @ attrs)

        {
            Metric = metric
            Measured = measured
            Bound = bound
            Op = op
            Passed = passed
            Elements = elements
            Attrs = attrs
        }

    let private sameShape (candidate: IHostTensor) (reference: IHostTensor) =
        if not (candidate.Desc.Shape.Equals reference.Desc.Shape) then
            invalidArg "reference" (sprintf "shapes differ: %O and %O" candidate.Desc.Shape reference.Desc.Shape)

    let private pair (candidate: #IHostTensor) (reference: #IHostTensor) =
        sameShape candidate reference
        candidate.ToFloat64(), reference.ToFloat64()

    /// [Rows, Cols] for the row metrics (kl, topKAgreement, top2Margin). A rank-1 [n] has Rows = n and Cols = 1, so a
    /// logits vector would become n one-element rows, and KL would pass trivially: rank 0 and 1 raise instead.
    let private rows (t: IHostTensor) =
        let s = t.Desc.Shape

        if s.Rank < 2 then
            invalidArg
                "candidate"
                (sprintf "a row metric needs [rows, cols] (rank 2 or more), not %O; reshape a vector [n] to [1, n]" s)

        int s.Rows, int s.Cols

    /// relL2(candidate, reference) <= bound.
    let relL2 (bound: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        let a, b = pair candidate reference
        check "relL2" BoundOp.AtMost (Metrics.relL2 (ReadOnlySpan a) (ReadOnlySpan b)) bound a.LongLength attrs

    /// max |candidate - reference| <= bound.
    let maxAbs (bound: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        let a, b = pair candidate reference
        check "maxAbs" BoundOp.AtMost (Metrics.maxAbs (ReadOnlySpan a) (ReadOnlySpan b)) bound a.LongLength attrs

    /// The largest distance in units in the last place of the common dtype (F32, F64 or BF16) <= bound.
    let maxUlp (bound: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        sameShape candidate reference
        let dtype = candidate.Desc.DType

        if reference.Desc.DType <> dtype then
            invalidArg "reference" "maxUlp needs one dtype"

        let ca = candidate.ToBytes()
        let ra = reference.ToBytes()

        let measured =
            match dtype with
            | DType.F32 ->
                Metrics.maxUlp32
                    (Runtime.InteropServices.MemoryMarshal.Cast<byte, float32>(ReadOnlySpan ca))
                    (Runtime.InteropServices.MemoryMarshal.Cast<byte, float32>(ReadOnlySpan ra))
            | DType.F64 ->
                Metrics.maxUlp64
                    (Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(ReadOnlySpan ca))
                    (Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(ReadOnlySpan ra))
            | DType.BF16 ->
                let a = Runtime.InteropServices.MemoryMarshal.Cast<byte, uint16>(ReadOnlySpan ca)
                let b = Runtime.InteropServices.MemoryMarshal.Cast<byte, uint16>(ReadOnlySpan ra)
                let mutable m = 0.0

                for i in 0 .. a.Length - 1 do
                    m <- max m (Metrics.ulpDistanceBf16 a[i] b[i])

                m
            | other ->
                invalidArg "candidate" (sprintf "maxUlp is defined for f32, f64 and bf16, not %s" (DType.name other))

        check "maxUlp" BoundOp.AtMost measured bound candidate.Desc.Shape.Elements attrs

    /// cosine(candidate, reference) >= minimum.
    let cosine (minimum: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        let a, b = pair candidate reference
        check "cosine" BoundOp.AtLeast (Metrics.cosine (ReadOnlySpan a) (ReadOnlySpan b)) minimum a.LongLength attrs

    /// The largest per-row KL(P_reference || P_candidate) of two logit tensors <= bound.
    let kl (bound: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        let r, c = rows candidate
        let a, b = pair candidate reference
        let mutable m = 0.0
        let mutable sawNaN = false

        for i in 0 .. r - 1 do
            let v = Metrics.klRow (ReadOnlySpan(a, i * c, c)) (ReadOnlySpan(b, i * c, c))

            if Double.IsNaN v then
                sawNaN <- true
            elif v > m then
                m <- v

        check "kl" BoundOp.AtMost (if sawNaN then nan else m) bound (int64 r) attrs

    /// The smallest per-row top-k agreement >= minimum.
    let topKAgreement (k: int) (minimum: float) (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        let r, c = rows candidate
        let a, b = pair candidate reference
        let mutable m = 1.0

        for i in 0 .. r - 1 do
            m <- min m (Metrics.topKAgreementRow (ReadOnlySpan(a, i * c, c)) (ReadOnlySpan(b, i * c, c)) k)

        check "topKAgreement" BoundOp.AtLeast m minimum (int64 r) (attrs @ [ Attr.int "k" k ])

    /// The smallest per-row top-2 margin of one logit tensor >= minimum.
    let top2Margin (minimum: float) (attrs: Attr list) (logits: #IHostTensor) =
        let r, c = rows logits
        let a = logits.ToFloat64()
        let mutable m = infinity

        for i in 0 .. r - 1 do
            m <- min m (Metrics.top2MarginRow (ReadOnlySpan(a, i * c, c)))

        check "top2Margin" BoundOp.AtLeast m minimum (int64 r) attrs

    /// Bitwise equality: measured is the number of elements whose bits differ, bound 0. Dtypes and shapes must match.
    let exact (attrs: Attr list) (candidate: #IHostTensor) (reference: #IHostTensor) =
        sameShape candidate reference

        if candidate.Desc.DType <> reference.Desc.DType then
            invalidArg "reference" "exact needs one dtype"

        let a = candidate.ToBytes()
        let b = reference.ToBytes()
        let size = candidate.Desc.ElementSize
        let mutable differing = 0L

        for i in 0..size .. a.Length - 1 do
            if not (ReadOnlySpan(a, i, size).SequenceEqual(ReadOnlySpan(b, i, size))) then
                differing <- differing + 1L

        check "exact" BoundOp.AtMost (float differing) 0.0 candidate.Desc.Shape.Elements attrs
