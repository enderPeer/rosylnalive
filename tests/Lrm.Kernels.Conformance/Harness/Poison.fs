namespace Lrm.Kernels.Conformance

open System
open System.Buffers.Binary
open Lrm.Core

/// Poisoned outputs (ConformanceHarness contract; VulkanHeapRecipe section 8; ADR R1-heap-recipe pitfall 15). Before
/// every run the harness fills each region a case writes with poison words that differ per configuration, case and
/// run; after the run a written word still equal to its poison fails the case.
[<RequireQualifiedAccess>]
module Poison =
    /// The canonical-hash tag of a poison seed.
    [<Literal>]
    let Tag = "lrm.conformance.poison/1"

    /// The seed of one run: Canonical.hash Tag over the strings configuration, case and run (SHA-256 of
    /// tag || 0x00 || the three length-prefixed UTF-8 strings).
    let seed (configuration: string) (case: string) (run: string) : ContentId =
        Canonical.hash Tag (fun w ->
            w.String configuration
            w.String case
            w.String run)

    /// The DetRng of a seed: the Philox key is the seed's first 8 bytes and the stream its next 8, little-endian.
    let rng (seed: ContentId) =
        let b = seed.ToBytes()

        DetRng(
            BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan(b, 0, 8)),
            BinaryPrimitives.ReadUInt64LittleEndian(ReadOnlySpan(b, 8, 8))
        )

    /// A poison word from a random word r: the sign of r's bit 31, exponent all ones, quiet bit clear, and the mantissa
    /// r's bits 0..21 (1 when those are 0). Every poison word is a float32 signalling NaN, which no arithmetic produces.
    let ofWord (r: uint32) =
        let m = r &&& 0x003FFFFFu
        (r &&& 0x80000000u) ||| 0x7F800000u ||| (if m = 0u then 1u else m)

    /// True for a float32 signalling NaN (exponent all ones, quiet bit clear, mantissa nonzero).
    let isSignallingNaN (w: uint32) =
        w &&& 0x7FC00000u = 0x7F800000u && w &&& 0x003FFFFFu <> 0u

    /// `count` poison words of a seed, in order.
    let words (seed: ContentId) (count: int) : uint32[] =
        let r = rng seed
        Array.init count (fun _ -> ofWord (r.NextU32()))

    /// A poison word that differs from `expected`: the word itself, or with its lowest mantissa bit flipped (2 instead
    /// of a mantissa that would become 0). It stays a signalling NaN.
    let avoiding (expected: uint32) (poison: uint32) =
        if poison <> expected then
            poison
        elif poison &&& 0x003FFFFFu = 1u then
            (poison &&& ~~~0x003FFFFFu) ||| 2u
        else
            poison ^^^ 1u
