// Read-only probe: D3D12 device, feature support, and one SM 6.6 ResourceDescriptorHeap compute dispatch.
// COM calls go through raw vtable slots (indices from Windows SDK 10.0.26100 d3d12.h).
#nowarn "9"
#nowarn "51"
open System
open System.IO
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop

module Native =
    [<DllImport("d3d12.dll")>]
    extern int D3D12CreateDevice(nativeint adapter, int minFeatureLevel, Guid& riid, nativeint& device)
    [<DllImport("dxgi.dll")>]
    extern int CreateDXGIFactory2(uint32 flags, Guid& riid, nativeint& factory)
    [<DllImport("kernel32.dll")>]
    extern nativeint CreateEventW(nativeint attrs, int manualReset, int initialState, nativeint name)
    [<DllImport("kernel32.dll")>]
    extern uint32 WaitForSingleObject(nativeint h, uint32 ms)

let slot (obj: nativeint) (i: int) =
    let vtbl = NativePtr.read (NativePtr.ofNativeInt<nativeint> obj)
    NativePtr.get (NativePtr.ofNativeInt<nativeint> vtbl) i
let fn<'D when 'D :> Delegate> obj i = Marshal.GetDelegateForFunctionPointer<'D>(slot obj i)
let ok what (hr: int) = if hr < 0 then failwithf "%s failed: 0x%08X" what hr

type D_Release = delegate of nativeint -> uint32
type D_QI = delegate of nativeint * Guid byref * nativeint byref -> int
type D_CheckFeature = delegate of nativeint * int * nativeint * int -> int
type D_Create4 = delegate of nativeint * nativeint * Guid byref * nativeint byref -> int          // (this, desc*, riid, out)
type D_CreateAlloc = delegate of nativeint * int * Guid byref * nativeint byref -> int
type D_CreateRS = delegate of nativeint * uint32 * nativeint * unativeint * Guid byref * nativeint byref -> int
type D_CreateList = delegate of nativeint * uint32 * int * nativeint * nativeint * Guid byref * nativeint byref -> int
type D_IncSize = delegate of nativeint * int -> uint32
type D_CreateUAV = delegate of nativeint * nativeint * nativeint * nativeint * unativeint -> unit
type D_CreateCommitted = delegate of nativeint * nativeint * int * nativeint * int * nativeint * Guid byref * nativeint byref -> int
type D_CreateFence = delegate of nativeint * uint64 * int * Guid byref * nativeint byref -> int
type D_RemovedReason = delegate of nativeint -> int
type D_HeapStart = delegate of nativeint * unativeint byref -> nativeint                          // hidden RetVal ABI
type D_Map = delegate of nativeint * uint32 * nativeint * nativeint byref -> int
type D_Unmap = delegate of nativeint * uint32 * nativeint -> unit
type D_Exec = delegate of nativeint * uint32 * nativeint byref -> unit
type D_Signal = delegate of nativeint * nativeint * uint64 -> int
type D_SetEvent = delegate of nativeint * uint64 * nativeint -> int
type D_Close = delegate of nativeint -> int
type D_SetObj = delegate of nativeint * nativeint -> unit
type D_SetHeaps = delegate of nativeint * uint32 * nativeint byref -> unit
type D_SetConsts = delegate of nativeint * uint32 * uint32 * nativeint * uint32 -> unit
type D_Dispatch = delegate of nativeint * uint32 * uint32 * uint32 -> unit
type D_CopyBuf = delegate of nativeint * nativeint * uint64 * nativeint * uint64 * uint64 -> unit
type D_Barrier = delegate of nativeint * uint32 * nativeint -> unit
type D_GetDescCfg = delegate of nativeint * nativeint -> nativeint                                 // hidden RetVal ABI
type D_EnumByPref = delegate of nativeint * uint32 * int * Guid byref * nativeint byref -> int
type D_GetDesc1 = delegate of nativeint * nativeint -> int

let release (p: nativeint) = if p <> 0n then (fn<D_Release> p 2).Invoke p |> ignore

let mutable IID_Device = Guid("189819f1-1db6-4b57-be54-1821339b85f7")
let mutable IID_Queue = Guid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
let mutable IID_Alloc = Guid("6102dee4-af59-4b09-b999-b44d73f09b24")
let mutable IID_List = Guid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455")
let mutable IID_Fence = Guid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")
let mutable IID_Resource = Guid("696442be-a72e-4059-bc79-5b5c98040fad")
let mutable IID_Heap = Guid("8efb471d-616c-4f49-90f7-127bb763fa51")
let mutable IID_RS = Guid("c54a6b66-72df-4ee8-8be5-a946a1429214")
let mutable IID_PSO = Guid("765a30f3-f624-4c6f-a828-ace948622445")
let mutable IID_DevCfg = Guid("78dbf87b-f766-422b-a61c-c8c446bdb9ad")
let mutable IID_Factory6 = Guid("c1b6694f-ff09-44a9-b03c-77900a0a1d17")
let mutable IID_Adapter1 = Guid("29038f61-3839-4626-91fd-08687901 1a05".Replace(" ", ""))

let mem (n: int) = let p = Marshal.AllocHGlobal n in (for i in 0 .. n - 1 do Marshal.WriteByte(p, i, 0uy)); p
let w32 (p: nativeint) off (v: int) = Marshal.WriteInt32(p, off, v)
let w64 (p: nativeint) off (v: int64) = Marshal.WriteInt64(p, off, v)
let wptr (p: nativeint) off (v: nativeint) = Marshal.WriteIntPtr(p, off, v)

// ---- adapter: high-performance GPU via DXGI ----
let mutable factory = 0n
ok "CreateDXGIFactory2" (Native.CreateDXGIFactory2(0u, &IID_Factory6, &factory))
let mutable adapter = 0n
ok "EnumAdapterByGpuPreference" ((fn<D_EnumByPref> factory 29).Invoke(factory, 0u, 2, &IID_Adapter1, &adapter))
let adesc = mem 512
ok "GetDesc1" ((fn<D_GetDesc1> adapter 10).Invoke(adapter, adesc))
printfn "adapter: %s  dedicatedVRAM=%d MiB" (Marshal.PtrToStringUni adesc) (Marshal.ReadInt64(adesc, 256 + 4 * 4) / 1048576L)

// ---- device (feature level 12_0 = 0xc000) ----
let mutable dev = 0n
ok "D3D12CreateDevice" (Native.D3D12CreateDevice(adapter, 0xc000, &IID_Device, &dev))
let check (feature: int) (size: int) (init: nativeint -> unit) =
    let p = mem size
    init p
    let hr = (fn<D_CheckFeature> dev 13).Invoke(dev, feature, p, size)
    hr, p

// highest shader model: ask from 6.9 down
let rec sm (v: int) =
    let hr, p = check 7 4 (fun p -> w32 p 0 v)
    if hr >= 0 then Marshal.ReadInt32 p elif v > 0x60 then sm (v - 1) else -1
printfn "HighestShaderModel: 0x%x" (sm 0x69)
let _, o0 = check 0 60 ignore
printfn "ResourceBindingTier=%d ResourceHeapTier=%d DoublePrecision=%d" (Marshal.ReadInt32(o0, 16)) (Marshal.ReadInt32(o0, 56)) (Marshal.ReadInt32(o0, 0))
let _, o1 = check 8 24 ignore
printfn "WaveOps=%d WaveLaneCountMin=%d Max=%d TotalLaneCount=%d Int64ShaderOps=%d" (Marshal.ReadInt32(o1, 0)) (Marshal.ReadInt32(o1, 4)) (Marshal.ReadInt32(o1, 8)) (Marshal.ReadInt32(o1, 12)) (Marshal.ReadInt32(o1, 20))
let _, o4 = check 23 12 ignore
printfn "Native16BitShaderOpsSupported=%d" (Marshal.ReadInt32(o4, 8))
let _, o9 = check 37 24 ignore
printfn "AtomicInt64OnTypedResource=%d AtomicInt64OnGroupShared=%d WaveMMATier=%d" (Marshal.ReadInt32(o9, 8)) (Marshal.ReadInt32(o9, 12)) (Marshal.ReadInt32(o9, 20))
let hr11, o11 = check 40 4 ignore
printfn "AtomicInt64OnDescriptorHeapResource=%d (hr 0x%08X)" (Marshal.ReadInt32(o11, 0)) hr11
let hrRS, ors = check 12 4 (fun p -> w32 p 0 0x3) // D3D_ROOT_SIGNATURE_VERSION_1_2 = 3
printfn "RootSignature highest version: %d (hr 0x%08X)" (Marshal.ReadInt32(ors, 0)) hrRS

let mutable cfg = 0n
let hrCfg = (fn<D_QI> dev 0).Invoke(dev, &IID_DevCfg, &cfg)
if hrCfg >= 0 then
    let d = mem 16
    (fn<D_GetDescCfg> cfg 3).Invoke(cfg, d) |> ignore
    printfn "ID3D12DeviceConfiguration: SDKVersion=%d Flags=%d" (Marshal.ReadInt32(d, 8)) (Marshal.ReadInt32(d, 0))
    release cfg
else printfn "QI ID3D12DeviceConfiguration: hr 0x%08X" hrCfg

// ---- compute objects ----
let qd = mem 16
w32 qd 0 2 // D3D12_COMMAND_LIST_TYPE_COMPUTE
let mutable queue = 0n
ok "CreateCommandQueue" ((fn<D_Create4> dev 8).Invoke(dev, qd, &IID_Queue, &queue))
let mutable alloc = 0n
ok "CreateCommandAllocator" ((fn<D_CreateAlloc> dev 9).Invoke(dev, 2, &IID_Alloc, &alloc))

let dxil = File.ReadAllBytes(Path.Combine(__SOURCE_DIRECTORY__, "..", "shaders", "add_heap_rs.dxil"))
let hDxil = GCHandle.Alloc(dxil, GCHandleType.Pinned)
let pDxil = hDxil.AddrOfPinnedObject()
let mutable rs = 0n
ok "CreateRootSignature(from DXIL RTS0)" ((fn<D_CreateRS> dev 16).Invoke(dev, 0u, pDxil, unativeint dxil.Length, &IID_RS, &rs))
let psoDesc = mem 56
wptr psoDesc 0 rs
wptr psoDesc 8 pDxil
w64 psoDesc 16 (int64 dxil.Length)
let mutable pso = 0n
ok "CreateComputePipelineState" ((fn<D_Create4> dev 11).Invoke(dev, psoDesc, &IID_PSO, &pso))
printfn "root signature + compute PSO (cs_6_6, ResourceDescriptorHeap): OK"

let hd = mem 16
w32 hd 0 0 // CBV_SRV_UAV
w32 hd 4 16
w32 hd 8 1 // SHADER_VISIBLE
let mutable heap = 0n
ok "CreateDescriptorHeap" ((fn<D_Create4> dev 14).Invoke(dev, hd, &IID_Heap, &heap))
let inc = (fn<D_IncSize> dev 15).Invoke(dev, 0)
let mutable cpuStart = 0un
(fn<D_HeapStart> heap 9).Invoke(heap, &cpuStart) |> ignore
printfn "CBV_SRV_UAV descriptor increment = %d bytes" inc

let N = 1024
let bytes = N * 4
let buffer (heapType: int) (size: int) (flags: int) (state: int) =
    let hp = mem 20
    w32 hp 0 heapType
    let rd = mem 56
    w32 rd 0 1 // BUFFER
    w64 rd 16 (int64 size)
    w32 rd 24 1
    Marshal.WriteInt16(rd, 28, 1s)
    Marshal.WriteInt16(rd, 30, 1s)
    w32 rd 36 1 // SampleDesc.Count
    w32 rd 44 1 // ROW_MAJOR
    w32 rd 48 flags
    let mutable r = 0n
    ok "CreateCommittedResource" ((fn<D_CreateCommitted> dev 27).Invoke(dev, hp, 0, rd, state, 0n, &IID_Resource, &r))
    r
let A = buffer 1 bytes 4 0
let B = buffer 1 bytes 4 0
let C = buffer 1 bytes 4 0
let up = buffer 2 (2 * bytes) 0 0xac3 // UPLOAD, GENERIC_READ
let rb = buffer 3 bytes 0 0x400       // READBACK, COPY_DEST

let mutable pUp = 0n
ok "Map upload" ((fn<D_Map> up 8).Invoke(up, 0u, 0n, &pUp))
for i in 0 .. N - 1 do
    Marshal.WriteInt32(pUp, i * 4, BitConverter.SingleToInt32Bits(float32 i))
    Marshal.WriteInt32(pUp, bytes + i * 4, BitConverter.SingleToInt32Bits(0.5f))
(fn<D_Unmap> up 9).Invoke(up, 0u, 0n)

let uav (res: nativeint) (index: int) =
    let d = mem 40
    w32 d 0 39 // DXGI_FORMAT_R32_TYPELESS
    w32 d 4 1  // D3D12_UAV_DIMENSION_BUFFER
    w64 d 8 0L
    w32 d 16 N
    w32 d 20 0
    w64 d 24 0L
    w32 d 32 1 // D3D12_BUFFER_UAV_FLAG_RAW
    (fn<D_CreateUAV> dev 19).Invoke(dev, res, 0n, d, cpuStart + unativeint (index * int inc))
uav A 0
uav B 1
uav C 2

let mutable list = 0n
ok "CreateCommandList" ((fn<D_CreateList> dev 12).Invoke(dev, 0u, 2, alloc, pso, &IID_List, &list))
let barrier (res: nativeint) (before: int) (after: int) =
    let b = mem 32
    w32 b 0 0
    wptr b 8 res
    w32 b 16 -1 // ALL_SUBRESOURCES
    w32 b 20 before
    w32 b 24 after
    (fn<D_Barrier> list 26).Invoke(list, 1u, b)
let copy = fn<D_CopyBuf> list 15
barrier A 0 0x400
barrier B 0 0x400
copy.Invoke(list, A, 0UL, up, 0UL, uint64 bytes)
copy.Invoke(list, B, 0UL, up, uint64 bytes, uint64 bytes)
barrier A 0x400 0x8
barrier B 0x400 0x8
barrier C 0 0x8
let mutable heapArr = heap
(fn<D_SetHeaps> list 28).Invoke(list, 1u, &heapArr)   // must precede SetComputeRootSignature for DIRECTLY_INDEXED
(fn<D_SetObj> list 29).Invoke(list, rs)
let consts = mem 16
w32 consts 0 0
w32 consts 4 1
w32 consts 8 2
w32 consts 12 N
(fn<D_SetConsts> list 35).Invoke(list, 0u, 4u, consts, 0u)
(fn<D_Dispatch> list 14).Invoke(list, uint32 (N / 64), 1u, 1u)
barrier C 0x8 0x800
copy.Invoke(list, rb, 0UL, C, 0UL, uint64 bytes)
ok "Close" ((fn<D_Close> list 9).Invoke list)

let mutable fence = 0n
ok "CreateFence" ((fn<D_CreateFence> dev 36).Invoke(dev, 0UL, 0, &IID_Fence, &fence))
let mutable l = list
(fn<D_Exec> queue 10).Invoke(queue, 1u, &l)
ok "Signal" ((fn<D_Signal> queue 14).Invoke(queue, fence, 1UL))
let ev = Native.CreateEventW(0n, 0, 0, 0n)
ok "SetEventOnCompletion" ((fn<D_SetEvent> fence 9).Invoke(fence, 1UL, ev))
let wr = Native.WaitForSingleObject(ev, 10000u)
let removed = (fn<D_RemovedReason> dev 37).Invoke dev
printfn "wait=%d deviceRemovedReason=0x%08X" wr removed

let mutable pRb = 0n
ok "Map readback" ((fn<D_Map> rb 8).Invoke(rb, 0u, 0n, &pRb))
let mutable bad = 0
for i in 0 .. N - 1 do
    let v = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(pRb, i * 4))
    if v <> float32 i + 0.5f then bad <- bad + 1
printfn "dispatch result: %d/%d correct (C[0]=%g C[1023]=%g)" (N - bad) N (BitConverter.Int32BitsToSingle(Marshal.ReadInt32(pRb, 0))) (BitConverter.Int32BitsToSingle(Marshal.ReadInt32(pRb, 1023 * 4)))
(fn<D_Unmap> rb 9).Invoke(rb, 0u, 0n)
for p in [ fence; list; rb; up; C; B; A; heap; pso; rs; alloc; queue; dev; adapter; factory ] do release p
hDxil.Free()
