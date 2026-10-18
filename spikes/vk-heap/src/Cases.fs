/// The kernel cases and their CPU references. Every gating case is bit-exact by construction: scale_f32 is one fp32
/// multiply of normal inputs with normal results, and the other kernels only move bits. The edge case (denormals,
/// infinities, NaN) is informational: it records what each backend does, it does not gate.
module VkHeapSpike.Cases

open System
open System.IO
open System.Text.Json
open Lrm.Core
open VkHeapSpike.Native
open VkHeapSpike.Heap

type Case =
    {
        Name: string
        Kernel: string
        Job: Job
        /// The expected bytes of each readback.
        Expected: byte[] list
        /// False for informational cases.
        Gating: bool
    }

let private args (groupsX: uint32) (fields: (int * uint32) list) =
    let a = Array.zeroCreate<uint32> 32
    a[0] <- groupsX

    for (i, v) in fields do
        a[i] <- v

    a

let private tableUpload: Upload =
    {
        Slot = 0
        Offset = 0L
        Data = Heap.table
    }

let private upTo (token: int) (data: byte[]) : Upload =
    let r = Heap.row token

    {
        Slot = r.Slot
        Offset = r.ByteOffset
        Data = data
    }

let private readOf (token: int) (bytes: int) : Readback =
    let r = Heap.row token

    {
        Slot = r.Slot
        Offset = r.ByteOffset
        Length = bytes
    }

let private f32bytes (xs: float32[]) =
    let b = Array.zeroCreate<byte> (xs.Length * 4)
    Buffer.BlockCopy(xs, 0, b, 0, b.Length)
    b

let private u16bytes (xs: uint16[]) =
    let b = Array.zeroCreate<byte> (xs.Length * 2)
    Buffer.BlockCopy(xs, 0, b, 0, b.Length)
    b

/// Normal fp32 values with random sign, exponent in [2^-60, 2^60) and random mantissa: products with `a` stay normal.
let normalInputs (seed: uint64) (n: int) =
    let rng = DetRng(seed, 1UL)

    Array.init n (fun _ ->
        let r = rng.NextU32()
        let sign = r &&& 0x80000000u
        let exponent = 67u + (rng.NextU32() % 120u) // biased 67..186
        BitConverter.UInt32BitsToSingle(sign ||| (exponent <<< 23) ||| (r &&& 0x007FFFFFu)))

let ScaleFactor = 0.7853982f

/// The CPU reference for scale_f32: one IEEE fp32 multiply per element.
let scaleRef (xs: float32[]) (a: float32) = xs |> Array.map (fun x -> x * a)

let groupsFor (threads: int) (n: int) = uint32 ((n + threads - 1) / threads)

/// scale_f32 from token tx to token ty. Each case has its own input seed, so no two cases expect the same output.
let scaleCase (name: string) (seed: uint64) (tx: int) (ty: int) (gridY: uint32) =
    let n = ScaleN
    let xs = normalInputs seed n
    let total = groupsFor 256 n
    let gx = total / gridY

    if gx * gridY <> total then
        failwith "grid does not divide"

    {
        Name = name
        Kernel = "scale_f32"
        Job =
            {
                Uploads = [ tableUpload; upTo tx (f32bytes xs) ]
                Dispatches =
                    [
                        {
                            Kernel = "scale_f32"
                            GroupsX = gx
                            GroupsY = gridY
                            Args =
                                args
                                    gx
                                    [
                                        1, uint32 tx
                                        2, uint32 ty
                                        3, uint32 n
                                        4, BitConverter.SingleToUInt32Bits ScaleFactor
                                    ]
                        }
                    ]
                Writes = [ readOf ty (n * 4) ]
                Readbacks = [ readOf ty (n * 4) ]
            }
        Expected = [ f32bytes (scaleRef xs ScaleFactor) ]
        Gating = true
    }

/// Two dependent dispatches in one command buffer: y = x * a, then z = y * a. Checks the barrier between dependent
/// dispatches that V1 inserts (a global memory barrier).
let chainCase () =
    let n = ScaleN
    let xs = normalInputs 0x434841494eUL n
    let g = groupsFor 256 n
    let a = BitConverter.SingleToUInt32Bits ScaleFactor

    let step (src: int) (dst: int) =
        {
            Kernel = "scale_f32"
            GroupsX = g
            GroupsY = 1u
            Args = args g [ 1, uint32 src; 2, uint32 dst; 3, uint32 n; 4, a ]
        }

    {
        Name = "scale_f32.chain2"
        Kernel = "scale_f32"
        Job =
            {
                Uploads = [ tableUpload; upTo Tok.ScaleX (f32bytes xs) ]
                Dispatches = [ step Tok.ScaleX Tok.ScaleY; step Tok.ScaleY Tok.ChainZ ]
                // The intermediate y is poisoned too: a first dispatch that writes nothing must not leave the second
                // one a correct y from an earlier run.
                Writes = [ readOf Tok.ScaleY (n * 4); readOf Tok.ChainZ (n * 4) ]
                Readbacks = [ readOf Tok.ChainZ (n * 4) ]
            }
        Expected = [ f32bytes (scaleRef (scaleRef xs ScaleFactor) ScaleFactor) ]
        Gating = true
    }

/// args_echo with 30 payload dwords: random words plus fixed patterns (all ones, a NaN pattern, the sign bit, 1).
let echoCase (k: int) =
    let rng = DetRng(0x4543484fUL, uint64 k)
    let fixedWords = [| 0xFFFFFFFFu; 0x7FC00001u; 0x80000000u; 0x00000001u |]

    let a =
        args
            1u
            [
                yield 1, uint32 Tok.EchoOut
                for i in 2..31 do
                    yield
                        i,
                        (if i - 2 < fixedWords.Length && k = 0 then
                             fixedWords[i - 2]
                         else
                             rng.NextU32())
            ]

    {
        Name = sprintf "args_echo.%d" k
        Kernel = "args_echo"
        Job =
            {
                Uploads = [ tableUpload ]
                Dispatches =
                    [
                        {
                            Kernel = "args_echo"
                            GroupsX = 1u
                            GroupsY = 1u
                            Args = a
                        }
                    ]
                Writes = [ readOf Tok.EchoOut 128 ]
                Readbacks = [ readOf Tok.EchoOut 128 ]
            }
        Expected = [ ofWords a ]
        Gating = true
    }

let tableCase () =
    let words = TableRows * 16
    let g = groupsFor 256 words

    {
        Name = "table_echo"
        Kernel = "table_echo"
        Job =
            {
                Uploads = [ tableUpload ]
                Dispatches =
                    [
                        {
                            Kernel = "table_echo"
                            GroupsX = g
                            GroupsY = 1u
                            Args = args g [ 1, uint32 Tok.TableOut; 2, 0u; 3, uint32 TableRows ]
                        }
                    ]
                Writes = [ readOf Tok.TableOut (words * 4) ]
                Readbacks = [ readOf Tok.TableOut (words * 4) ]
            }
        Expected = [ Heap.table ]
        Gating = true
    }

let bf16Case (name: string) (tin: int) (tout: int) (bits: uint16[]) =
    let n = bits.Length
    let g = groupsFor 256 n

    {
        Name = name
        Kernel = "bf16_unpack"
        Job =
            {
                Uploads = [ tableUpload; upTo tin (u16bytes bits) ]
                Dispatches =
                    [
                        {
                            Kernel = "bf16_unpack"
                            GroupsX = g
                            GroupsY = 1u
                            Args = args g [ 1, uint32 tin; 2, uint32 tout; 3, uint32 n ]
                        }
                    ]
                Writes = [ readOf tout (n * 4) ]
                Readbacks = [ readOf tout (n * 4) ]
            }
        Expected = [ bits |> Array.map Bf16.widenBits |> ofWords ]
        Gating = true
    }

/// Every bf16 bit pattern, NaNs and denormals included.
let bf16Exhaustive () =
    bf16Case "bf16_unpack.all65536" Tok.Bf16In Tok.Bf16Out (Array.init 65536 uint16)

/// The first QwenN elements of model.embed_tokens.weight from the local Qwen3-0.6B weights, or None when the model
/// is not present.
let qwenEmbeddingBits () : uint16[] option =
    let path = Path.Combine(LrmPaths.model "Qwen3-0.6B", "model.safetensors")

    if not (File.Exists path) then
        None
    else
        use f = File.OpenRead path
        let lenBytes = Array.zeroCreate<byte> 8
        f.ReadExactly(lenBytes, 0, 8)
        let headerLen = BitConverter.ToInt64(lenBytes, 0)
        let header = Array.zeroCreate<byte> (int headerLen)
        f.ReadExactly(header, 0, header.Length)
        use doc = JsonDocument.Parse(header)
        let t = doc.RootElement.GetProperty("model.embed_tokens.weight")

        if t.GetProperty("dtype").GetString() <> "BF16" then
            failwith "embed_tokens is not BF16"

        let offsets = t.GetProperty("data_offsets")
        let start = offsets[0].GetInt64()
        f.Seek(8L + headerLen + start, SeekOrigin.Begin) |> ignore
        let raw = Array.zeroCreate<byte> (QwenN * 2)
        f.ReadExactly(raw, 0, raw.Length)
        let bits = Array.zeroCreate<uint16> QwenN
        Buffer.BlockCopy(raw, 0, bits, 0, raw.Length)
        Some bits

/// Denormal inputs and outputs, infinities and NaN through scale_f32: informational only.
let edgeCase () =
    let n = EdgeN

    let xs =
        Array.init n (fun i ->
            match i % 8 with
            | 0 -> BitConverter.UInt32BitsToSingle(uint32 (i + 1)) // denormal input
            | 1 -> BitConverter.UInt32BitsToSingle(0x00800000u + uint32 i) // normal input, denormal result
            | 2 -> Single.PositiveInfinity
            | 3 -> Single.NegativeInfinity
            | 4 -> -0.0f
            | 5 -> Single.MaxValue
            | 6 -> BitConverter.UInt32BitsToSingle(0x7FC00000u ||| uint32 i) // quiet NaN with payload
            | _ -> 1.5f)

    let g = groupsFor 256 n

    {
        Name = "scale_f32.edges"
        Kernel = "scale_f32"
        Job =
            {
                Uploads = [ tableUpload; upTo Tok.EdgeX (f32bytes xs) ]
                Dispatches =
                    [
                        {
                            Kernel = "scale_f32"
                            GroupsX = g
                            GroupsY = 1u
                            Args =
                                args
                                    g
                                    [
                                        1, uint32 Tok.EdgeX
                                        2, uint32 Tok.EdgeY
                                        3, uint32 n
                                        4, BitConverter.SingleToUInt32Bits ScaleFactor
                                    ]
                        }
                    ]
                Writes = [ readOf Tok.EdgeY (n * 4) ]
                Readbacks = [ readOf Tok.EdgeY (n * 4) ]
            }
        Expected = [ f32bytes (scaleRef xs ScaleFactor) ]
        Gating = false
    }

/// All cases, gating first. The 1 GiB-page case proves the descriptor range at the page-size limit.
let all () =
    [
        scaleCase "scale_f32" 0x5343414c46UL Tok.ScaleX Tok.ScaleY 1u
        scaleCase "scale_f32.grid2d" 0x4752494432UL Tok.ScaleX Tok.ScaleY 4u
        scaleCase "scale_f32.page1gib" 0x5343414c4cUL Tok.BigX Tok.BigY 1u
        chainCase ()
        for k in 0..3 do
            echoCase k
        tableCase ()
        bf16Exhaustive ()
        match qwenEmbeddingBits () with
        | Some bits -> bf16Case "bf16_unpack.qwen_embed" Tok.QwenIn Tok.QwenOut bits
        | None -> ()
        edgeCase ()
    ]

/// The cases the mapping controls run: enough to cover the table, several slots and every push dword.
let controlSubset = set [ "scale_f32"; "args_echo.0"; "table_echo"; "bf16_unpack.all65536" ]

/// "[index] got/expected" in hex, for messages and reports.
let describeMismatches (xs: (int * uint32 * uint32) list) =
    xs
    |> List.map (fun (i, g, e) -> sprintf "[%d] 0x%08x/0x%08x" i g e)
    |> String.concat ", "

/// Mismatching u32 words: count and the first few (index, got, expected). A plain loop: the negative controls compare
/// millions of mismatching words, and host time between jobs lets the GPU drop out of P0 (see the ADR).
let compareWords (got: byte[]) (expected: byte[]) =
    if got.Length <> expected.Length then
        Int32.MaxValue, [ -1, uint32 got.Length, uint32 expected.Length ]
    else
        let g = words got
        let e = words expected
        let first = ResizeArray<int * uint32 * uint32>()
        let mutable n = 0

        for i in 0 .. g.Length - 1 do
            if g[i] <> e[i] then
                if first.Count < 8 then
                    first.Add((i, g[i], e[i]))

                n <- n + 1

        n, List.ofSeq first

/// Byte equality without F#'s generic structural comparison.
let sameBytes (a: byte[]) (b: byte[]) =
    ReadOnlySpan<byte>(a).SequenceEqual(ReadOnlySpan<byte>(b))

// ---- poisoned outputs ----

/// The seed of one run's poison: SHA-256 of "label|case|run", so the poison differs per configuration, per case and
/// per run.
let poisonSeed (label: string) (case: string) (run: string) =
    BitConverter.ToUInt64((Hash.sha256Utf8 (sprintf "%s|%s|%s" label case run)).ToBytes(), 0)

/// The poison words of one write region. Each is a signalling-NaN bit pattern (random sign, quiet bit clear, payload
/// in [1, 0x3FFFFF]), moved to the next payload where it would equal the expected word. Arithmetic never produces a
/// signalling NaN and the poison never equals the expected word, so a word that still equals its poison after a run
/// was not written by the dispatch.
let poisonWords (seed: uint64) (region: int) (expected: uint32[] option) (n: int) =
    let mix = SplitMix64(SplitMix64.Mix(seed + uint64 region))

    Array.init n (fun i ->
        let r = uint32 (mix.Next() >>> 32)
        let p = (r &&& 0x80000000u) ||| 0x7F800000u ||| (1u + (r &&& 0x3FFFFFu) % 0x3FFFFEu)

        match expected with
        | Some e when e[i] = p -> p + 1u
        | _ -> p)

let private sameRange (w: Readback) (r: Readback) =
    w.Slot = r.Slot && w.Offset = r.Offset && w.Length = r.Length

/// The job with every write region poisoned first, and the poison of each readback (to count what was not written).
/// The poison uploads go before the inputs, and Cases.all keeps every write region apart from every input region.
let poisoned (seed: uint64) (c: Case) : Job * uint32[] list =
    let expectedOf (w: Readback) =
        List.zip c.Job.Readbacks c.Expected
        |> List.tryFind (fun (r, _) -> sameRange w r)
        |> Option.map (snd >> words)

    let poison =
        c.Job.Writes
        |> List.mapi (fun k w -> w, poisonWords seed k (expectedOf w) (w.Length / 4))

    let uploads =
        [
            for (w, p) in poison ->
                {
                    Slot = w.Slot
                    Offset = w.Offset
                    Data = ofWords p
                }
        ]

    let perReadback =
        [
            for r in c.Job.Readbacks ->
                match poison |> List.tryFind (fun (w, _) -> sameRange w r) with
                | Some(_, p) -> p
                | None -> failwithf "%s: readback at slot %d offset %d is not one of the job's writes" c.Name r.Slot r.Offset
        ]

    { c.Job with Uploads = uploads @ c.Job.Uploads }, perReadback

/// Problems with a case's regions: a write region that overlaps an input, a readback that is not a write, a length
/// that is not whole words.
let regionProblems (c: Case) =
    [
        for w in c.Job.Writes do
            if w.Length % 4 <> 0 then
                yield sprintf "%s: write region of %d bytes is not whole words" c.Name w.Length

            for u in c.Job.Uploads do
                if overlaps (w.Slot, w.Offset, int64 w.Length) (u.Slot, u.Offset, int64 u.Data.Length) then
                    yield sprintf "%s: write region at slot %d offset %d overlaps an input" c.Name w.Slot w.Offset

        for r in c.Job.Readbacks do
            if not (c.Job.Writes |> List.exists (fun w -> sameRange w r)) then
                yield sprintf "%s: readback at slot %d offset %d is not poisoned" c.Name r.Slot r.Offset
    ]

/// Words still equal to their poison.
let poisonLeft (outputs: byte[] list) (poison: uint32[] list) =
    List.zip outputs poison
    |> List.sumBy (fun (o, p) ->
        let w = words o
        let mutable n = 0

        for i in 0 .. min w.Length p.Length - 1 do
            if w[i] = p[i] then
                n <- n + 1

        n)

/// One case on one backend configuration, run twice with a different poison each time.
type CaseOutcome =
    {
        Case: string
        Kernel: string
        Gating: bool
        /// Mismatching words against the CPU reference (first run).
        Mismatches: int
        FirstMismatches: (int * uint32 * uint32) list
        /// Words of the readbacks still equal to their poison, summed over both runs. Must be 0: every output word was
        /// written by the dispatches in both runs.
        PoisonWordsLeft: int
        /// True when run 1 and run 2 gave identical bytes. The runs use different poisons, so a run that writes
        /// nothing cannot match the other.
        RunTwiceIdentical: bool
        /// SHA-256 of the first run's outputs.
        Digest: string
        /// The faster run's GPU time per dispatch, in ns.
        GpuNs: float list
        /// The first run's start of each dispatch in the GPU timestamp domain, in ns.
        GpuStartNs: int64 list
    }

/// Total words of a case's readbacks.
let readbackWords (c: Case) =
    c.Job.Readbacks |> List.sumBy (fun r -> r.Length / 4)

/// Runs `c` twice on `gpu`, each run with its own poison. `tag` separates the poison of other runs of the same case
/// on the same configuration (the negative controls).
let runCaseTagged (tag: string) (gpu: IGpu) (c: Case) : CaseOutcome =
    let runOnce (run: string) =
        let job, poison = poisoned (poisonSeed gpu.Label c.Name (tag + run)) c
        let r = gpu.Run job
        r, poisonLeft r.Outputs poison

    let r1, left1 = runOnce "1"
    let r2, left2 = runOnce "2"
    let cat (o: byte[] list) = Array.concat o
    let mism, first = compareWords (cat r1.Outputs) (cat c.Expected)

    {
        Case = c.Name
        Kernel = c.Kernel
        Gating = c.Gating
        Mismatches = mism
        FirstMismatches = first
        PoisonWordsLeft = left1 + left2
        RunTwiceIdentical = sameBytes (cat r1.Outputs) (cat r2.Outputs)
        Digest = (Hash.sha256 (cat r1.Outputs)).Hex
        GpuNs = List.map2 min r1.DispatchGpuNs r2.DispatchGpuNs
        GpuStartNs = r1.DispatchGpuStartNs
    }

let runCase (gpu: IGpu) (c: Case) = runCaseTagged "" gpu c

/// The sabotaged variants of a case for the negative control: all dispatches removed and, for a job with several
/// dispatches, each one removed in turn. Every variant must fail the case's checks.
let sabotaged (c: Case) =
    [
        yield "no_dispatches", { c with Job = { c.Job with Dispatches = [] } }

        if c.Job.Dispatches.Length > 1 then
            for k in 0 .. c.Job.Dispatches.Length - 1 do
                yield
                    sprintf "without_dispatch_%d" (k + 1),
                    { c with
                        Job =
                            { c.Job with
                                Dispatches = List.removeAt k c.Job.Dispatches
                            }
                    }
    ]

/// Whether the checks catch a sabotaged run: every readback word differs from the expected word (the poison is never
/// expected) and, when nothing ran at all, every word is still poison in both runs and the two runs differ.
let caught (variant: string) (c: Case) (o: CaseOutcome) =
    let n = readbackWords c

    if variant = "no_dispatches" then
        o.Mismatches = n && o.PoisonWordsLeft = 2 * n && not o.RunTwiceIdentical
    else
        o.Mismatches = n
