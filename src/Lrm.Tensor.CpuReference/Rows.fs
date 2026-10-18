namespace Lrm.Tensor.CpuReference

open System
open System.Numerics
open Lrm.Core

/// The row kernels of the CPU reference (CpuReferenceSemantics contract), generic over float32 and float. Each works
/// on spans with no allocation, so CpuDevice emulators can run them over heap memory. Every reduction is sequential in
/// increasing index order and starts from +0; products that feed a reduction are fused (one rounding per step, through
/// 'T.FusedMultiplyAdd); no SIMD type (Vector<T>) is used. Element-wise formulas round after every operation, in the
/// order written.
[<RequireQualifiedAccess>]
module Rows =
    /// bf16 bits widened exactly to 'T.
    let inline widen<'T when 'T :> IFloatingPointIeee754<'T>> (bits: uint16) : 'T = 'T.CreateTruncating(Bf16.widen bits)

    /// A float32 widened exactly to 'T.
    let inline ofF32<'T when 'T :> IFloatingPointIeee754<'T>> (x: float32) : 'T = 'T.CreateTruncating x

    /// IEEE a > b (false when either is NaN), without F# generic comparison.
    let inline private gt<'T when 'T :> IFloatingPointIeee754<'T>> (a: 'T) (b: 'T) = 'T.op_GreaterThan (a, b)

    let private sameLength (a: int) (b: int) (what: string) =
        if a <> b then
            invalidArg what (sprintf "lengths differ: %d and %d" a b)

    /// sum_i x[i] * y[i]: acc = fma(x[i], y[i], acc) for i = 0, 1, ... from acc = +0.
    let dot<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) (y: ReadOnlySpan<'T>) : 'T =
        sameLength x.Length y.Length "y"
        let mutable acc = 'T.Zero

        for i in 0 .. x.Length - 1 do
            acc <- 'T.FusedMultiplyAdd(x[i], y[i], acc)

        acc

    /// sum_i x[i] * widen(w[i]), as dot.
    let dotBf16<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) (w: ReadOnlySpan<uint16>) : 'T =
        sameLength x.Length w.Length "w"
        let mutable acc = 'T.Zero

        for i in 0 .. x.Length - 1 do
            acc <- 'T.FusedMultiplyAdd(x[i], widen<'T> w[i], acc)

        acc

    /// RMSNorm of one row (Qwen3RMSNorm, weight used as w, not 1 + w):
    /// ss = sum fma(x, x); mean = ss / n; r = 1 / sqrt(mean + eps); y[c] = widen(w[c]) * (x[c] * r).
    /// y may be x.
    let rmsNorm<'T when 'T :> IFloatingPointIeee754<'T>>
        (x: ReadOnlySpan<'T>)
        (w: ReadOnlySpan<uint16>)
        (eps: 'T)
        (y: Span<'T>)
        =
        sameLength x.Length w.Length "w"
        sameLength x.Length y.Length "y"

        if x.Length = 0 then
            invalidArg "x" "an empty row has no RMS"

        let mutable ss = 'T.Zero

        for i in 0 .. x.Length - 1 do
            ss <- 'T.FusedMultiplyAdd(x[i], x[i], ss)

        let mean = ss / 'T.CreateTruncating x.Length
        let r = 'T.One / 'T.Sqrt(mean + eps)

        for i in 0 .. x.Length - 1 do
            y[i] <- widen<'T> w[i] * (x[i] * r)

    /// The row maximum by `>` from x[0] (a NaN never wins a comparison; softmax and logSumExp still propagate it).
    let private rowMax<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) =
        let mutable m = x[0]

        for i in 1 .. x.Length - 1 do
            if gt x[i] m then
                m <- x[i]

        m

    /// Softmax of a row in place (see softmax).
    let private softmaxInPlace<'T when 'T :> IFloatingPointIeee754<'T>> (y: Span<'T>) =
        let mutable m = y[0]

        for i in 1 .. y.Length - 1 do
            if gt y[i] m then
                m <- y[i]

        let mutable l = 'T.Zero

        for i in 0 .. y.Length - 1 do
            let e = 'T.Exp(y[i] - m)
            y[i] <- e
            l <- l + e

        for i in 0 .. y.Length - 1 do
            y[i] <- y[i] / l

    /// Softmax of one row: m = max; e[i] = exp(x[i] - m); l = sum e (sequential); y[i] = e[i] / l. y may be x. A NaN
    /// anywhere makes the whole row NaN.
    let softmax<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) (y: Span<'T>) =
        sameLength x.Length y.Length "y"

        if x.Length = 0 then
            invalidArg "x" "empty row"

        x.CopyTo y
        softmaxInPlace y

    /// m + log(sum_i exp(x[i] - m)), m the row maximum, sums sequential.
    let logSumExp<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) : 'T =
        if x.Length = 0 then
            invalidArg "x" "empty row"

        let m = rowMax x
        let mutable l = 'T.Zero

        for i in 0 .. x.Length - 1 do
            l <- l + 'T.Exp(x[i] - m)

        m + 'T.Log l

    /// The lowest index of the row maximum (+0 and -0 tie), or ValueNone when the row holds a NaN.
    let tryArgmax<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) : int voption =
        if x.Length = 0 then
            invalidArg "x" "empty row"

        let mutable best = 0
        let mutable nan = 'T.IsNaN x[0]
        let mutable i = 1

        while not nan && i < x.Length do
            if 'T.IsNaN x[i] then
                nan <- true
            elif gt x[i] x[best] then
                best <- i

            i <- i + 1

        if nan then ValueNone else ValueSome best

    /// The lowest index of the row maximum; raises ArgumentException when the row holds a NaN.
    let argmax<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) : int =
        match tryArgmax x with
        | ValueSome i -> i
        | ValueNone -> invalidArg "x" "argmax of a row that holds a NaN"

    /// The indices of the k largest values, largest first; equal values in index order (lowest first). Raises
    /// ArgumentException when the row holds a NaN.
    let topK<'T when 'T :> IFloatingPointIeee754<'T>> (x: ReadOnlySpan<'T>) (k: int) : int[] =
        if k < 1 || k > x.Length then
            invalidArg "k" (sprintf "k = %d for a row of %d" k x.Length)

        for i in 0 .. x.Length - 1 do
            if 'T.IsNaN x[i] then
                invalidArg "x" "topK of a row that holds a NaN"

        let values = x.ToArray()
        let idx = Array.init values.Length id

        Array.sortInPlaceWith
            (fun a b ->
                if gt values[a] values[b] then -1
                elif gt values[b] values[a] then 1
                else compare a b)
            idx

        Array.sub idx 0 k

    /// y[i] = silu(g[i]) * u[i] with silu(g) = g / (1 + exp(-g)). y may be g or u.
    let siluMul<'T when 'T :> IFloatingPointIeee754<'T>> (g: ReadOnlySpan<'T>) (u: ReadOnlySpan<'T>) (y: Span<'T>) =
        sameLength g.Length u.Length "u"
        sameLength g.Length y.Length "y"

        for i in 0 .. g.Length - 1 do
            let gi = g[i]
            let s = gi / ('T.One + 'T.Exp(-gi))
            y[i] <- s * u[i]

    /// y[i] = a[i] + b[i]. y may be a or b.
    let add<'T when 'T :> IFloatingPointIeee754<'T>> (a: ReadOnlySpan<'T>) (b: ReadOnlySpan<'T>) (y: Span<'T>) =
        sameLength a.Length b.Length "b"
        sameLength a.Length y.Length "y"

        for i in 0 .. a.Length - 1 do
            y[i] <- a[i] + b[i]

    /// NeoX rotate-half RoPE of one head (2 * half elements) from one table row: for i < half, with c = cos[i],
    /// s = sin[i] widened, y[i] = x[i]*c - x[i+half]*s and y[i+half] = x[i+half]*c + x[i]*s, each product rounded.
    /// y may be x.
    let ropeHead<'T when 'T :> IFloatingPointIeee754<'T>>
        (x: ReadOnlySpan<'T>)
        (cos: ReadOnlySpan<float32>)
        (sin: ReadOnlySpan<float32>)
        (y: Span<'T>)
        =
        let half = cos.Length
        sameLength x.Length (2 * half) "x"
        sameLength sin.Length half "sin"
        sameLength y.Length x.Length "y"

        for i in 0 .. half - 1 do
            let c = ofF32<'T> cos[i]
            let s = ofF32<'T> sin[i]
            let a = x[i]
            let b = x[i + half]
            let ac = a * c
            let bs = b * s
            let bc = b * c
            let as' = a * s
            y[i] <- ac - bs
            y[i + half] <- bc + as'

    /// Attention of one query head over keys 0 .. keys - 1 (the causal mask is this loop bound). Key and value row j
    /// starts at j * stride + offset in k and v and is q.Length long. With s[j] = dot(q, k_j) * scale, the weights
    /// are softmax(s) as in `softmax`, and y[d] = sum_j fma(p[j], v_j[d], .) in increasing j from +0. `scores` is
    /// scratch of at least `keys` elements. y must not overlap q, k or v.
    let attention<'T when 'T :> IFloatingPointIeee754<'T>>
        (q: ReadOnlySpan<'T>)
        (k: ReadOnlySpan<'T>)
        (v: ReadOnlySpan<'T>)
        (stride: int)
        (offset: int)
        (keys: int)
        (scale: 'T)
        (scores: Span<'T>)
        (y: Span<'T>)
        =
        let hd = q.Length
        sameLength y.Length hd "y"

        if keys < 1 then
            invalidArg "keys" "attention needs at least one key"

        if scores.Length < keys then
            invalidArg "scores" "scratch shorter than the key count"

        let last = (keys - 1) * stride + offset + hd

        if offset < 0 || offset + hd > stride || last > k.Length || last > v.Length then
            invalidArg "k" "a key or value row is outside the cache"

        for j in 0 .. keys - 1 do
            scores[j] <- dot q (k.Slice(j * stride + offset, hd)) * scale

        softmaxInPlace (scores.Slice(0, keys))

        for d in 0 .. hd - 1 do
            y[d] <- 'T.Zero

        for j in 0 .. keys - 1 do
            let p = scores[j]
            let vj = v.Slice(j * stride + offset, hd)

            for d in 0 .. hd - 1 do
                y[d] <- 'T.FusedMultiplyAdd(p, vj[d], y[d])

    /// KL(P || Q) of two logit rows: with lp = p - logSumExp p and lq = q - logSumExp q,
    /// sum_i fma(exp(lp[i]), lp[i] - lq[i], .) in increasing i from +0.
    let kl<'T when 'T :> IFloatingPointIeee754<'T>> (p: ReadOnlySpan<'T>) (q: ReadOnlySpan<'T>) : 'T =
        sameLength p.Length q.Length "q"
        let lseP = logSumExp p
        let lseQ = logSumExp q
        let mutable acc = 'T.Zero

        for i in 0 .. p.Length - 1 do
            let lp = p[i] - lseP
            let lq = q[i] - lseQ
            acc <- 'T.FusedMultiplyAdd('T.Exp lp, lp - lq, acc)

        acc
