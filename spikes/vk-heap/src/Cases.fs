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

let scaleCase (name: string) (tx: int) (ty: int) (gridY: uint32) =
    let n = ScaleN
    let xs = normalInputs (0x5343414c45UL + uint64 tx) n
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
                Readbacks = [ readOf Tok.EdgeY (n * 4) ]
            }
        Expected = [ f32bytes (scaleRef xs ScaleFactor) ]
        Gating = false
    }

/// All cases, gating first. The 1 GiB-page case proves the descriptor range at the page-size limit.
let all () =
    [
        scaleCase "scale_f32" Tok.ScaleX Tok.ScaleY 1u
        scaleCase "scale_f32.grid2d" Tok.ScaleX Tok.ScaleY 4u
        scaleCase "scale_f32.page1gib" Tok.BigX Tok.BigY 1u
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

/// Mismatching u32 words: count and the first few (index, got, expected).
let compareWords (got: byte[]) (expected: byte[]) =
    if got.Length <> expected.Length then
        Int32.MaxValue, [ -1, uint32 got.Length, uint32 expected.Length ]
    else
        let g = words got
        let e = words expected
        let bad = [ for i in 0 .. g.Length - 1 do if g[i] <> e[i] then yield i ]
        bad.Length, bad |> List.truncate 8 |> List.map (fun i -> i, g[i], e[i])

/// One case on one backend configuration, run twice.
type CaseOutcome =
    {
        Case: string
        Kernel: string
        Gating: bool
        /// Mismatching words against the CPU reference (first run).
        Mismatches: int
        FirstMismatches: (int * uint32 * uint32) list
        /// True when run 1 and run 2 gave identical bytes.
        RunTwiceIdentical: bool
        /// SHA-256 of the first run's outputs.
        Digest: string
        /// The faster run's GPU time per dispatch, in ns.
        GpuNs: float list
    }

let runCase (gpu: IGpu) (c: Case) : CaseOutcome =
    let r1 = gpu.Run c.Job
    let r2 = gpu.Run c.Job
    let cat (o: byte[] list) = Array.concat o
    let mism, first = compareWords (cat r1.Outputs) (cat c.Expected)

    {
        Case = c.Name
        Kernel = c.Kernel
        Gating = c.Gating
        Mismatches = mism
        FirstMismatches = first
        RunTwiceIdentical = (cat r1.Outputs) = (cat r2.Outputs)
        Digest = (Hash.sha256 (cat r1.Outputs)).Hex
        GpuNs = List.map2 min r1.DispatchGpuNs r2.DispatchGpuNs
    }
