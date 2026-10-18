/// Helpers shared by the CpuDevice tests.
module Lrm.Gpu.CpuDevice.Tests.Common

open System
open System.Runtime.InteropServices
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Shaders

/// Lowercase hex.
let hex (bytes: byte[]) = Convert.ToHexStringLower bytes

/// Requires `f` to raise GpuContractException whose message contains `fragment`.
let raisesContract (fragment: string) (f: unit -> 'a) =
    try
        f () |> ignore
        failtestf "expected GpuContractException containing '%s', but nothing was raised" fragment
    with
    | :? GpuContractException as e ->
        if not (e.Message.Contains fragment) then
            failtestf "expected a message containing '%s', got '%s'" fragment e.Message
    | :? AssertException -> reraise ()
    | e -> failtestf "expected GpuContractException containing '%s', got %s: %s" fragment (e.GetType().Name) e.Message

/// Runs `f` with an in-memory telemetry sink installed and returns its records.
let withSink (f: unit -> unit) =
    let sink = InMemorySink()
    use _ = Telemetry.install (Telemetry.ofSinkWithClock (fun () -> 0L) sink)
    f ()
    sink.Records

let bytesOf<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType> (xs: 'T[]) =
    MemoryMarshal.AsBytes(ReadOnlySpan<'T>(xs)).ToArray()

let f32Bits (xs: float32[]) =
    MemoryMarshal.Cast<float32, uint32>(ReadOnlySpan xs).ToArray()

let ofBits (ws: uint32[]) =
    MemoryMarshal.Cast<uint32, float32>(ReadOnlySpan ws).ToArray()

/// A tensor of a fixture: a name, a dtype, dims and flags, placed by the planner (Activations unless `Arena`).
type T =
    {
        Name: string
        DType: DType
        Dims: int64 list
        Flags: TensorFlags
        Arena: ArenaKind
    }

let t (name: string) (dtype: DType) (count: int64) =
    {
        Name = name
        DType = dtype
        Dims = [ count ]
        Flags = TensorFlags.None
        Arena = ArenaKind.Activations
    }

let readOnly (x: T) =
    { x with
        Flags = TensorFlags.ReadOnly
        Arena = ArenaKind.Weights
    }

/// A CpuDevice with its heap, kernels and table: every tensor placed by a HeapPlanner, the embedded shader set loaded.
type Fixture =
    {
        Gpu: IGpuBackend
        Heap: IGpuHeap
        Table: SealedTable
    }

    member f.Token(name: string) = (f.Table.TryFind name).Value
    member f.Row(name: string) = f.Table.Row(f.Token name)

    member f.Upload(name: string, bytes: byte[]) =
        let r = f.Row name
        let padded = Array.zeroCreate<byte>(int (Heap.alignUp 4L (int64 bytes.Length)))
        Buffer.BlockCopy(bytes, 0, padded, 0, bytes.Length)
        f.Heap.Upload(r.Slot, int64 r.ByteOffset, ReadOnlySpan padded)

    member f.UploadF32(name: string, xs: float32[]) = f.Upload(name, bytesOf xs)
    member f.UploadU32(name: string, xs: uint32[]) = f.Upload(name, bytesOf xs)
    member f.UploadBf16(name: string, xs: uint16[]) = f.Upload(name, bytesOf xs)

    /// The first n words of a tensor.
    member f.Words(name: string, n: int) =
        let r = f.Row name
        let b = Array.zeroCreate<byte>(4 * n)
        f.Heap.Download(r.Slot, int64 r.ByteOffset, Span b)
        MemoryMarshal.Cast<byte, uint32>(ReadOnlySpan b).ToArray()

    member f.F32(name: string, n: int) = ofBits (f.Words(name, n))

    /// Records `record` into one command list, submits it and waits.
    member f.Run(record: ICommandList -> unit) =
        let c = f.Gpu.Begin()
        record c
        f.Gpu.Wait(f.Gpu.Submit c)

let fixtureWith (cpu: CpuDeviceOptions) (options: DeviceOptions) (tensors: T list) =
    let planner = HeapPlanner(1L <<< 26)

    let placed =
        tensors
        |> List.map (fun x ->
            let count = List.fold (*) 1L x.Dims
            x, planner.Allocate(x.Arena, max 4L (count * int64 (DType.size x.DType))))

    let layout = planner.Build()
    let gpu = CpuDevice.create cpu options
    let heap = gpu.CreateHeap layout
    gpu.LoadKernels(Shaders.embedded ()) |> ignore
    let table = TensorTable layout

    for x, p in placed do
        table.Add(x.Name, p, x.DType, x.Dims, x.Flags) |> ignore

    let sealedTable = table.Seal()
    heap.WriteTable sealedTable

    {
        Gpu = gpu
        Heap = heap
        Table = sealedTable
    }

let fixture (tensors: T list) =
    fixtureWith CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice) tensors

/// Deterministic test data.
let rng (name: string) =
    DetRng(0x4C524D2D45314454UL, DetRng.streamOf ("cpudevice.tests." + name))

let uniform (r: DetRng) (n: int) (lo: float32) (hi: float32) =
    Array.init n (fun _ -> lo + (hi - lo) * r.NextF32())

let uniformBf16 (r: DetRng) (n: int) (lo: float32) (hi: float32) =
    uniform r n lo hi |> Array.map Bf16.narrowRne

let scaleArgs (x: TensorToken) (y: TensorToken) (n: uint32) (a: float32) : Kernels.ScaleF32Args =
    { X = x; Y = y; N = n; A = a }
