/// The heap plan both backends build: 64 slots written once, slot 0 the tensor table, the rest arena pages
/// (HeapLayoutAndTensorTable, as planned for R4a). The spike uses a few small pages, one 1 GiB page to prove the
/// descriptor range at the page-size limit, and the last slot, so slot indexing is exercised end to end.
module VkHeapSpike.Heap

open System
open Lrm.Core

[<Literal>]
let SlotCount = 64

[<Literal>]
let TableRows = 16384

[<Literal>]
let RowBytes = 64

let TableBytes = int64 TableRows * int64 RowBytes

let MiB = 1024L * 1024L
let GiB = 1024L * MiB

/// Bytes of each slot that holds its own page; every other slot points at a shared 64 KiB dummy page.
let pageBytes (slot: int) : int64 option =
    match slot with
    | 0 -> Some TableBytes
    | s when s >= 1 && s <= 6 -> Some(32L * MiB)
    | 7 -> Some GiB
    | 63 -> Some(64L * 1024L)
    | _ -> None

let DummyBytes = 64L * 1024L

/// One tensor-table row (TensorRow: 16 u32).
type TensorRow =
    {
        Slot: int
        ByteOffset: int64
        DType: DType
        Count: int
    }

let rowWords (r: TensorRow) : uint32[] =
    [|
        uint32 r.Slot
        uint32 r.ByteOffset
        uint32 (DType.code r.DType)
        1u // rank
        uint32 r.Count
        1u
        1u
        1u // dims
        1u
        0u
        0u
        0u // strides
        uint32 r.Count // elemCount
        0u // flags
        0u // debugId, filled below
        0u
    |]

/// Tokens used by the cases. Token 0 is the null row.
module Tok =
    let ScaleX = 1
    let ScaleY = 2
    let EchoOut = 3
    let TableOut = 4
    let Bf16In = 5
    let Bf16Out = 6
    let BigX = 7
    let BigY = 8
    let QwenIn = 9
    let QwenOut = 10
    let EdgeX = 11
    let EdgeY = 12
    let ChainZ = 13

let ScaleN = 1 <<< 20
let QwenN = 1 <<< 21
let EdgeN = 1024

let rows: (int * TensorRow) list =
    [
        Tok.ScaleX,
        {
            Slot = 1
            ByteOffset = 0L
            DType = DType.F32
            Count = ScaleN
        }
        // An odd page and a non-zero offset, so the row's slot and byteOffset are both used.
        Tok.ScaleY,
        {
            Slot = 2
            ByteOffset = 4L * MiB + 256L
            DType = DType.F32
            Count = ScaleN
        }
        Tok.EchoOut,
        {
            Slot = 63
            ByteOffset = 128L
            DType = DType.U32
            Count = 32
        }
        Tok.TableOut,
        {
            Slot = 3
            ByteOffset = 0L
            DType = DType.U32
            Count = TableRows * 16
        }
        // Starts on a half word (byte 2), so the kernel's odd-element path runs for even indices.
        Tok.Bf16In,
        {
            Slot = 4
            ByteOffset = 2L
            DType = DType.BF16
            Count = 65536
        }
        Tok.Bf16Out,
        {
            Slot = 5
            ByteOffset = 0L
            DType = DType.F32
            Count = 65536
        }
        // The last 8 MiB of the 1 GiB page.
        Tok.BigX,
        {
            Slot = 7
            ByteOffset = GiB - 8L * MiB
            DType = DType.F32
            Count = ScaleN
        }
        Tok.BigY,
        {
            Slot = 7
            ByteOffset = GiB - 4L * MiB
            DType = DType.F32
            Count = ScaleN
        }
        Tok.QwenIn,
        {
            Slot = 6
            ByteOffset = 0L
            DType = DType.BF16
            Count = QwenN
        }
        Tok.QwenOut,
        {
            Slot = 1
            ByteOffset = 8L * MiB
            DType = DType.F32
            Count = QwenN
        }
        Tok.EdgeX,
        {
            Slot = 2
            ByteOffset = 16L * MiB
            DType = DType.F32
            Count = EdgeN
        }
        Tok.EdgeY,
        {
            Slot = 2
            ByteOffset = 20L * MiB
            DType = DType.F32
            Count = EdgeN
        }
        Tok.ChainZ,
        {
            Slot = 3
            ByteOffset = 8L * MiB
            DType = DType.F32
            Count = ScaleN
        }
    ]

let row (token: int) =
    rows |> List.find (fun (t, _) -> t = token) |> snd

/// The full table: real rows for the tokens above, token 0 zero, and every other row a SplitMix64 pattern so that
/// table_echo checks real content in every row.
let table: byte[] =
    let w = Array.zeroCreate<uint32> (TableRows * 16)
    let mix = SplitMix64(0x5231564b48454150UL) // "R1VKHEAP"

    for r in 1 .. TableRows - 1 do
        for k in 0 .. 15 do
            w[r * 16 + k] <- uint32 (mix.Next())

    for (t, r) in rows do
        let rw = rowWords r
        rw[14] <- uint32 (t * 1000 + 1)
        Array.blit rw 0 w (t * 16) 16

    for k in 0 .. 15 do
        w[k] <- 0u

    Native.ofWords w

// ---- one GPU job: uploads, dispatches in order, readbacks (the same shape on both backends) ----

type Upload =
    {
        Slot: int
        Offset: int64
        Data: byte[]
    }

type Readback =
    {
        Slot: int
        Offset: int64
        Length: int
    }

type Dispatch =
    {
        Kernel: string
        GroupsX: uint32
        GroupsY: uint32
        /// The 32-dword ArgBlock; Args[0] must equal GroupsX (lrm_groups_x).
        Args: uint32[]
    }

type Job =
    {
        Uploads: Upload list
        Dispatches: Dispatch list
        Readbacks: Readback list
    }

type JobResult =
    {
        Outputs: byte[] list
        /// GPU time of each dispatch from timestamp queries, in ns.
        DispatchGpuNs: float list
        /// The start of each dispatch in the queue's GPU timestamp domain, in ns.
        DispatchGpuStartNs: int64 list
    }

/// One backend configuration that can run jobs: a device, the heap written once, and the kernels.
type IGpu =
    inherit IDisposable
    /// For example "vulkan/explicit/bar" or "d3d12".
    abstract Label: string
    abstract Run: Job -> JobResult
