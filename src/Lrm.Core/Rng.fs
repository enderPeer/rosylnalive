namespace Lrm.Core

open System
open System.Buffers.Binary

/// SplitMix64 (Steele, Lea and Flood 2014; the generator behind java.util.SplittableRandom). A small sequential
/// generator for seeding and mixing. Use DetRng for anything that must be reproducible across shards and replays.
[<Sealed>]
type SplitMix64(seed: uint64) =
    let mutable state = seed

    /// The golden-ratio increment added to the state before every output.
    static member Gamma = 0x9E3779B97F4A7C15UL

    /// The SplitMix64 output function (a bijection on uint64).
    static member Mix(z: uint64) =
        let z = (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9UL
        let z = (z ^^^ (z >>> 27)) * 0x94D049BB133111EBUL
        z ^^^ (z >>> 31)

    /// The current state; SplitMix64(state) continues the same sequence.
    member _.State = state

    /// The next 64-bit output.
    member _.Next() =
        state <- state + SplitMix64.Gamma
        SplitMix64.Mix state

/// One Philox4x32 output block: four 32-bit words.
[<Struct>]
type Philox4x32Block =
    {
        X0: uint32
        X1: uint32
        X2: uint32
        X3: uint32
    }

/// Philox4x32 (Salmon, Moraes, Dror and Shaw, SC11; the Random123 philox4x32_R), a counter-based generator: it maps a
/// 128-bit counter and a 64-bit key to 128 random bits, with no state in between.
[<RequireQualifiedAccess>]
module Philox4x32 =
    [<Literal>]
    let M0 = 0xD2511F53u

    [<Literal>]
    let M1 = 0xCD9E8D57u

    [<Literal>]
    let W0 = 0x9E3779B9u

    [<Literal>]
    let W1 = 0xBB67AE85u

    /// Philox4x32 with `rounds` rounds of counter (c0, c1, c2, c3) under key (k0, k1). The first round uses the key
    /// as given; the key is bumped by (W0, W1) before each later round, as in Random123.
    let generate (rounds: int) (c0: uint32, c1: uint32, c2: uint32, c3: uint32) (k0: uint32, k1: uint32) =
        let mutable x0 = c0
        let mutable x1 = c1
        let mutable x2 = c2
        let mutable x3 = c3
        let mutable key0 = k0
        let mutable key1 = k1

        for r in 0 .. rounds - 1 do
            if r > 0 then
                key0 <- key0 + W0
                key1 <- key1 + W1

            let p0 = uint64 M0 * uint64 x0
            let p1 = uint64 M1 * uint64 x2
            let y0 = uint32 (p1 >>> 32) ^^^ x1 ^^^ key0
            let y1 = uint32 p1
            let y2 = uint32 (p0 >>> 32) ^^^ x3 ^^^ key1
            let y3 = uint32 p0
            x0 <- y0
            x1 <- y1
            x2 <- y2
            x3 <- y3

        { X0 = x0; X1 = x1; X2 = x2; X3 = x3 }

    /// Philox4x32-10, the standard 10-round variant.
    let block10 (c0: uint32, c1: uint32, c2: uint32, c3: uint32) (k0: uint32, k1: uint32) =
        generate 10 (c0, c1, c2, c3) (k0, k1)

/// Deterministic random numbers (LrmCore contract): Philox4x32-10 with key (seed low 32 bits, seed high 32 bits) and
/// counter (block low, block high, stream low, stream high). Draw i (counting 32-bit words from 0) is word i % 4 of
/// block i / 4. The sequence depends only on (seed, stream), and any position can be reached in O(1), so a shard
/// or a replay on another node draws exactly the same numbers. A DetRng instance is not thread-safe; give each
/// worker its own stream or its own instance at a known position.
[<Sealed>]
type DetRng(seed: uint64, stream: uint64, position: uint64) =
    let k0 = uint32 seed
    let k1 = uint32 (seed >>> 32)
    let s0 = uint32 stream
    let s1 = uint32 (stream >>> 32)
    let mutable pos = position
    let mutable cached = false
    let mutable cachedBlock = 0UL
    let mutable block = Unchecked.defaultof<Philox4x32Block>

    /// Starts at position 0.
    new(seed: uint64, stream: uint64) = DetRng(seed, stream, 0UL)

    /// The seed (the Philox key).
    member _.Seed = seed

    /// The stream (the high half of the Philox counter).
    member _.Stream = stream

    /// The number of 32-bit words drawn so far (including those consumed by NextU64 and friends).
    /// DetRng(seed, stream, rng.Position) continues the same sequence.
    member _.Position = pos

    /// The Philox block at a block index of this stream.
    member _.Block(index: uint64) =
        Philox4x32.block10 (uint32 index, uint32 (index >>> 32), s0, s1) (k0, k1)

    /// The next 32-bit word.
    member r.NextU32() : uint32 =
        let b = pos >>> 2

        if not cached || b <> cachedBlock then
            block <- r.Block b
            cachedBlock <- b
            cached <- true

        let lane = int (pos &&& 3UL)
        pos <- pos + 1UL

        match lane with
        | 0 -> block.X0
        | 1 -> block.X1
        | 2 -> block.X2
        | _ -> block.X3

    /// The next 64 bits: the first word drawn is the low half.
    member r.NextU64() : uint64 =
        let lo = r.NextU32()
        let hi = r.NextU32()
        (uint64 hi <<< 32) ||| uint64 lo

    /// A float32 in [0, 1): the top 24 bits of one word times 2^-24. Exact; every value is a multiple of 2^-24.
    member r.NextF32() : float32 =
        float32 (r.NextU32() >>> 8) * 5.9604644775390625e-8f

    /// A float in [0, 1): the top 53 bits of NextU64 times 2^-53. Exact; every value is a multiple of 2^-53.
    member r.NextF64() : float =
        float (r.NextU64() >>> 11) * 1.1102230246251565e-16

    /// A uniform integer in [0, bound), without modulo bias: words below (2^32 mod bound) are rejected and redrawn,
    /// so the number of words consumed depends on the values, deterministically.
    member r.NextBelow(bound: uint32) : uint32 =
        if bound = 0u then
            invalidArg "bound" "bound must be positive"

        let threshold = (0u - bound) % bound
        let mutable x = r.NextU32()

        while x < threshold do
            x <- r.NextU32()

        x % bound

    /// A uniform int in [lo, hi).
    member r.NextInt(lo: int, hi: int) : int =
        if hi <= lo then
            invalidArg "hi" "hi must be greater than lo"

        int (int64 lo + int64 (r.NextBelow(uint32 (int64 hi - int64 lo))))

    /// Fills a span with consecutive words.
    member r.Fill(destination: Span<uint32>) =
        for i in 0 .. destination.Length - 1 do
            destination[i] <- r.NextU32()

    /// A generator for another stream under the same seed, at position 0.
    member _.WithStream(other: uint64) = DetRng(seed, other)

/// Helpers for naming DetRng streams.
[<RequireQualifiedAccess>]
module DetRng =
    /// A stream number derived from a name: the first 8 bytes, little-endian, of
    /// Canonical.hash "lrm.rng.stream/1" (string name). Use it to give components stable, distinct streams.
    let streamOf (name: string) : uint64 =
        let id = Canonical.hash "lrm.rng.stream/1" (fun w -> w.String name)
        BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan(id.ToBytes(), 0, 8))

    /// A generator at an arbitrary position of (seed, stream).
    let at (seed: uint64) (stream: uint64) (position: uint64) = DetRng(seed, stream, position)
