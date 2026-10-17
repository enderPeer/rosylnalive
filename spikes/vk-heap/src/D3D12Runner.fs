/// The D3D12 side: the same HLSL as DXIL (cs_6_6, ResourceDescriptorHeap) under the one global root signature
/// (CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, 32 root constants at b0), one shader-visible heap of 64 raw UAVs (slot i =
/// descriptor i), a compute queue, a fence and timestamp queries. COM calls go through vtable slots counted from
/// the Windows SDK 10.0.26100 d3d12.h (DECLSPEC_XFGVIRT order), as in the planning probe d3d12probe.fsx.
module VkHeapSpike.D3D12Runner

#nowarn "9"

open System
open System.Runtime.InteropServices
open VkHeapSpike.Native
open VkHeapSpike.Heap

module private Native =
    [<DllImport("d3d12.dll")>]
    extern int D3D12CreateDevice(nativeint adapter, int minFeatureLevel, Guid& riid, nativeint& device)

    [<DllImport("dxgi.dll")>]
    extern int CreateDXGIFactory2(uint32 flags, Guid& riid, nativeint& factory)

    [<DllImport("kernel32.dll")>]
    extern nativeint CreateEventW(nativeint attrs, int manualReset, int initialState, nativeint name)

    [<DllImport("kernel32.dll")>]
    extern uint32 WaitForSingleObject(nativeint h, uint32 ms)

    [<DllImport("kernel32.dll")>]
    extern int CloseHandle(nativeint h)

let private slot (obj: nativeint) (i: int) =
    let vtbl = Marshal.ReadIntPtr obj
    Marshal.ReadIntPtr(vtbl, i * IntPtr.Size)

let private fn<'D when 'D :> Delegate> obj i =
    Marshal.GetDelegateForFunctionPointer<'D>(slot obj i)

let private hr what (h: int) =
    if h < 0 then
        failwithf "%s failed: 0x%08X" what h

type private D_Release = delegate of nativeint -> uint32
type private D_CheckFeature = delegate of nativeint * int * nativeint * int -> int
type private D_Create4 = delegate of nativeint * nativeint * Guid byref * nativeint byref -> int
type private D_CreateAlloc = delegate of nativeint * int * Guid byref * nativeint byref -> int
type private D_CreateRS = delegate of nativeint * uint32 * nativeint * unativeint * Guid byref * nativeint byref -> int
type private D_CreateList = delegate of nativeint * uint32 * int * nativeint * nativeint * Guid byref * nativeint byref -> int
type private D_IncSize = delegate of nativeint * int -> uint32
type private D_CreateUAV = delegate of nativeint * nativeint * nativeint * nativeint * unativeint -> unit

type private D_CreateCommitted =
    delegate of nativeint * nativeint * int * nativeint * int * nativeint * Guid byref * nativeint byref -> int

type private D_CreateFence = delegate of nativeint * uint64 * int * Guid byref * nativeint byref -> int
type private D_RemovedReason = delegate of nativeint -> int
type private D_HeapStart = delegate of nativeint * unativeint byref -> nativeint // hidden RetVal ABI
type private D_Map = delegate of nativeint * uint32 * nativeint * nativeint byref -> int
type private D_Unmap = delegate of nativeint * uint32 * nativeint -> unit
type private D_Exec = delegate of nativeint * uint32 * nativeint byref -> unit
type private D_Signal = delegate of nativeint * nativeint * uint64 -> int
type private D_SetEvent = delegate of nativeint * uint64 * nativeint -> int
type private D_Completed = delegate of nativeint -> uint64
type private D_Close = delegate of nativeint -> int
type private D_ResetAlloc = delegate of nativeint -> int
type private D_ResetList = delegate of nativeint * nativeint * nativeint -> int
type private D_SetObj = delegate of nativeint * nativeint -> unit
type private D_SetHeaps = delegate of nativeint * uint32 * nativeint byref -> unit
type private D_SetConsts = delegate of nativeint * uint32 * uint32 * nativeint * uint32 -> unit
type private D_Dispatch = delegate of nativeint * uint32 * uint32 * uint32 -> unit
type private D_CopyBuf = delegate of nativeint * nativeint * uint64 * nativeint * uint64 * uint64 -> unit
type private D_Barrier = delegate of nativeint * uint32 * nativeint -> unit
type private D_EndQuery = delegate of nativeint * nativeint * int * uint32 -> unit
type private D_Resolve = delegate of nativeint * nativeint * int * uint32 * uint32 * nativeint * uint64 -> unit
type private D_TsFreq = delegate of nativeint * uint64 byref -> int
type private D_EnumByPref = delegate of nativeint * uint32 * int * Guid byref * nativeint byref -> int
type private D_GetDesc1 = delegate of nativeint * nativeint -> int
type private D_CheckIface = delegate of nativeint * Guid byref * int64 byref -> int

let private release (p: nativeint) =
    if p <> 0n then
        (fn<D_Release> p 2).Invoke p |> ignore

let mutable private IID_Device = Guid("189819f1-1db6-4b57-be54-1821339b85f7")
let mutable private IID_Queue = Guid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed")
let mutable private IID_Alloc = Guid("6102dee4-af59-4b09-b999-b44d73f09b24")
let mutable private IID_List = Guid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455")
let mutable private IID_Fence = Guid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76")
let mutable private IID_Resource = Guid("696442be-a72e-4059-bc79-5b5c98040fad")
let mutable private IID_Heap = Guid("8efb471d-616c-4f49-90f7-127bb763fa51")
let mutable private IID_QueryHeap = Guid("0d9658ae-ed45-469e-a61d-970ec583cab4")
let mutable private IID_RS = Guid("c54a6b66-72df-4ee8-8be5-a946a1429214")
let mutable private IID_PSO = Guid("765a30f3-f624-4c6f-a828-ace948622445")
let mutable private IID_Factory6 = Guid("c1b6694f-ff09-44a9-b03c-77900a0a1d17")
let mutable private IID_Adapter1 = Guid("29038f61-3839-4626-91fd-086879011a05")
let mutable private IID_DXGIDevice = Guid("54ec77fa-1377-44e6-8c32-88fd5f44c84c")

// D3D12 enum values used below.
let private ListCompute = 2
let private HeapDefault = 1
let private HeapUpload = 2
let private HeapReadback = 3
let private StateCommon = 0
let private StateUav = 0x8
let private StateCopyDest = 0x400
let private StateCopySource = 0x800
let private StateGenericRead = 0xAC3
let private FlagAllowUav = 0x4
let private stagingBytes = 64L * MiB
let private maxQueries = 64

type D3D12Facts =
    {
        Adapter: string
        /// The user-mode driver version from IDXGIAdapter::CheckInterfaceSupport, for example 32.0.15.6692.
        Driver: string
        HighestShaderModel: int
        ResourceBindingTier: int
        DescriptorIncrement: uint32
        TimestampFrequency: uint64
    }

/// A D3D12 device with the 64-slot heap written once and the four kernels.
type D3D12Gpu(shaders: Shaders.ShaderSet) =
    let created = ResizeArray<nativeint>()

    let track p =
        created.Add p
        p

    let mutable factory = 0n
    do hr "CreateDXGIFactory2" (Native.CreateDXGIFactory2(0u, &IID_Factory6, &factory))
    do track factory |> ignore
    let mutable adapter = 0n

    // DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE = 2
    do
        hr
            "EnumAdapterByGpuPreference"
            ((fn<D_EnumByPref> factory 29).Invoke(factory, 0u, 2, &IID_Adapter1, &adapter))

    do track adapter |> ignore

    let adapterName =
        let d = Marshal.AllocHGlobal 512

        try
            hr "GetDesc1" ((fn<D_GetDesc1> adapter 10).Invoke(adapter, d))
            Marshal.PtrToStringUni d
        finally
            Marshal.FreeHGlobal d

    let umd =
        let mutable v = 0L

        if (fn<D_CheckIface> adapter 9).Invoke(adapter, &IID_DXGIDevice, &v) >= 0 then
            sprintf "%d.%d.%d.%d" (v >>> 48) ((v >>> 32) &&& 0xFFFFL) ((v >>> 16) &&& 0xFFFFL) (v &&& 0xFFFFL)
        else
            "unknown"

    let mutable dev = 0n
    do hr "D3D12CreateDevice" (Native.D3D12CreateDevice(adapter, 0xc000, &IID_Device, &dev))
    do track dev |> ignore

    let check (feature: int) (size: int) (init: nativeint -> unit) =
        let p = Marshal.AllocHGlobal size

        try
            for i in 0 .. size - 1 do
                Marshal.WriteByte(p, i, 0uy)

            init p
            let r = (fn<D_CheckFeature> dev 13).Invoke(dev, feature, p, size)
            r, Array.init (size / 4) (fun i -> Marshal.ReadInt32(p, 4 * i))
        finally
            Marshal.FreeHGlobal p

    let shaderModel =
        let rec sm v =
            let r, o = check 7 4 (fun p -> Marshal.WriteInt32(p, 0, v))
            if r >= 0 then o[0] elif v > 0x60 then sm (v - 1) else -1

        sm 0x69

    let bindingTier =
        let _, o = check 0 60 ignore
        o[4]

    do
        if shaderModel < 0x66 || bindingTier < 3 then
            failwithf "D3D12 device needs SM 6.6 and binding tier 3 (got SM 0x%x, tier %d)" shaderModel bindingTier

    let call4 name (obj: nativeint) slot (desc: nativeint) (iid: Guid) =
        let mutable g = iid
        let mutable out = 0n
        hr name ((fn<D_Create4> obj slot).Invoke(obj, desc, &g, &out))
        track out

    let queue =
        let qd = Marshal.AllocHGlobal 16

        try
            for i in 0..3 do
                Marshal.WriteInt32(qd, 4 * i, 0)

            Marshal.WriteInt32(qd, 0, ListCompute)
            call4 "CreateCommandQueue" dev 8 qd IID_Queue
        finally
            Marshal.FreeHGlobal qd

    let alloc =
        let mutable a = 0n
        hr "CreateCommandAllocator" ((fn<D_CreateAlloc> dev 9).Invoke(dev, ListCompute, &IID_Alloc, &a))
        track a

    let rootSig =
        let h = GCHandle.Alloc(shaders.RootSignature, GCHandleType.Pinned)

        try
            let mutable rs = 0n

            hr
                "CreateRootSignature"
                ((fn<D_CreateRS> dev 16)
                    .Invoke(dev, 0u, h.AddrOfPinnedObject(), unativeint shaders.RootSignature.Length, &IID_RS, &rs))

            track rs
        finally
            h.Free()

    let psos =
        [
            for KeyValue(k, dxil) in shaders.Dxil do
                let h = GCHandle.Alloc(dxil, GCHandleType.Pinned)
                let desc = Marshal.AllocHGlobal 56

                try
                    for i in 0..13 do
                        Marshal.WriteInt32(desc, 4 * i, 0)

                    Marshal.WriteIntPtr(desc, 0, rootSig)
                    Marshal.WriteIntPtr(desc, 8, h.AddrOfPinnedObject())
                    Marshal.WriteInt64(desc, 16, int64 dxil.Length)
                    yield k, call4 (sprintf "CreateComputePipelineState(%s)" k) dev 11 desc IID_PSO
                finally
                    Marshal.FreeHGlobal desc
                    h.Free()
        ]
        |> Map.ofList

    let heap =
        let hd = Marshal.AllocHGlobal 16

        try
            Marshal.WriteInt32(hd, 0, 0) // CBV_SRV_UAV
            Marshal.WriteInt32(hd, 4, SlotCount)
            Marshal.WriteInt32(hd, 8, 1) // SHADER_VISIBLE
            Marshal.WriteInt32(hd, 12, 0)
            call4 "CreateDescriptorHeap" dev 14 hd IID_Heap
        finally
            Marshal.FreeHGlobal hd

    let increment = (fn<D_IncSize> dev 15).Invoke(dev, 0)

    let cpuStart =
        let mutable h = 0un
        (fn<D_HeapStart> heap 9).Invoke(heap, &h) |> ignore
        h

    let buffer (heapType: int) (size: int64) (flags: int) (state: int) =
        let hp = Marshal.AllocHGlobal 20
        let rd = Marshal.AllocHGlobal 56

        try
            for i in 0..4 do
                Marshal.WriteInt32(hp, 4 * i, 0)

            for i in 0..13 do
                Marshal.WriteInt32(rd, 4 * i, 0)

            Marshal.WriteInt32(hp, 0, heapType)
            Marshal.WriteInt32(rd, 0, 1) // DIMENSION_BUFFER
            Marshal.WriteInt64(rd, 16, size)
            Marshal.WriteInt32(rd, 24, 1)
            Marshal.WriteInt16(rd, 28, 1s)
            Marshal.WriteInt16(rd, 30, 1s)
            Marshal.WriteInt32(rd, 36, 1) // SampleDesc.Count
            Marshal.WriteInt32(rd, 44, 1) // ROW_MAJOR
            Marshal.WriteInt32(rd, 48, flags)
            let mutable r = 0n

            hr
                "CreateCommittedResource"
                ((fn<D_CreateCommitted> dev 27).Invoke(dev, hp, 0, rd, state, 0n, &IID_Resource, &r))

            track r
        finally
            Marshal.FreeHGlobal hp
            Marshal.FreeHGlobal rd

    let dummy = buffer HeapDefault DummyBytes FlagAllowUav StateCommon

    /// (resource, bytes) per slot; slots without their own page share the dummy.
    let pages =
        Array.init SlotCount (fun s ->
            match pageBytes s with
            | Some bytes -> buffer HeapDefault bytes FlagAllowUav StateCommon, bytes
            | None -> dummy, DummyBytes)

    let distinctPages = pages |> Array.map fst |> Array.distinct

    let uav (res: nativeint) (bytes: int64) (index: int) =
        let d = Marshal.AllocHGlobal 40

        try
            for i in 0..9 do
                Marshal.WriteInt32(d, 4 * i, 0)

            Marshal.WriteInt32(d, 0, 39) // DXGI_FORMAT_R32_TYPELESS
            Marshal.WriteInt32(d, 4, 1) // D3D12_UAV_DIMENSION_BUFFER
            Marshal.WriteInt64(d, 8, 0L) // FirstElement
            Marshal.WriteInt32(d, 16, int (bytes / 4L)) // NumElements
            Marshal.WriteInt32(d, 20, 0) // StructureByteStride
            Marshal.WriteInt64(d, 24, 0L) // CounterOffsetInBytes
            Marshal.WriteInt32(d, 32, 1) // D3D12_BUFFER_UAV_FLAG_RAW
            (fn<D_CreateUAV> dev 19).Invoke(dev, res, 0n, d, cpuStart + unativeint (index * int increment))
        finally
            Marshal.FreeHGlobal d

    // The heap is written once: descriptor i is slot i.
    do pages |> Array.iteri (fun i (res, bytes) -> uav res bytes i)

    let upload = buffer HeapUpload stagingBytes 0 StateGenericRead
    let readback = buffer HeapReadback stagingBytes 0 StateCopyDest

    let map (res: nativeint) =
        let mutable p = 0n
        hr "Map" ((fn<D_Map> res 8).Invoke(res, 0u, 0n, &p))
        p

    let uploadPtr = map upload
    let readbackPtr = map readback

    let queryHeap =
        let qd = Marshal.AllocHGlobal 12

        try
            Marshal.WriteInt32(qd, 0, 1) // D3D12_QUERY_HEAP_TYPE_TIMESTAMP
            Marshal.WriteInt32(qd, 4, maxQueries)
            Marshal.WriteInt32(qd, 8, 0)
            call4 "CreateQueryHeap" dev 39 qd IID_QueryHeap
        finally
            Marshal.FreeHGlobal qd

    let queryReadback = buffer HeapReadback (int64 maxQueries * 8L) 0 StateCopyDest
    let queryPtr = map queryReadback

    let tsFreq =
        let mutable f = 0UL
        hr "GetTimestampFrequency" ((fn<D_TsFreq> queue 16).Invoke(queue, &f))
        f

    let list =
        let mutable l = 0n
        hr "CreateCommandList" ((fn<D_CreateList> dev 12).Invoke(dev, 0u, ListCompute, alloc, 0n, &IID_List, &l))
        hr "Close" ((fn<D_Close> l 9).Invoke l)
        track l

    let fence =
        let mutable f = 0n
        hr "CreateFence" ((fn<D_CreateFence> dev 36).Invoke(dev, 0UL, 0, &IID_Fence, &f))
        track f

    let event = Native.CreateEventW(0n, 0, 0, 0n)
    let mutable fenceValue = 0UL
    let mutable disposed = false

    let transition (res: nativeint) (before: int) (after: int) =
        let b = Marshal.AllocHGlobal 32

        try
            for i in 0..7 do
                Marshal.WriteInt32(b, 4 * i, 0)

            Marshal.WriteInt32(b, 0, 0) // TRANSITION
            Marshal.WriteIntPtr(b, 8, res)
            Marshal.WriteInt32(b, 16, -1) // ALL_SUBRESOURCES
            Marshal.WriteInt32(b, 20, before)
            Marshal.WriteInt32(b, 24, after)
            (fn<D_Barrier> list 26).Invoke(list, 1u, b)
        finally
            Marshal.FreeHGlobal b

    let uavBarrier () =
        let b = Marshal.AllocHGlobal 32

        try
            for i in 0..7 do
                Marshal.WriteInt32(b, 4 * i, 0)

            Marshal.WriteInt32(b, 0, 2) // UAV, null resource = all UAV accesses
            (fn<D_Barrier> list 26).Invoke(list, 1u, b)
        finally
            Marshal.FreeHGlobal b

    let facts =
        {
            Adapter = adapterName
            Driver = umd
            HighestShaderModel = shaderModel
            ResourceBindingTier = bindingTier
            DescriptorIncrement = increment
            TimestampFrequency = tsFreq
        }

    member _.Facts = facts

    interface IGpu with
        member _.Label = "d3d12"

        member _.Run(job: Job) =
            let nq = 2 * job.Dispatches.Length

            if nq > maxQueries then
                failwith "too many dispatches for the query heap"

            hr "Allocator Reset" ((fn<D_ResetAlloc> alloc 8).Invoke alloc)
            hr "List Reset" ((fn<D_ResetList> list 10).Invoke(list, alloc, 0n))
            let copy = fn<D_CopyBuf> list 15
            // Buffers start every command list in COMMON (they decay at the end of each ExecuteCommandLists).
            let state = Collections.Generic.Dictionary<nativeint, int>()

            for p in distinctPages do
                state[p] <- StateCommon

            let toState (res: nativeint) (s: int) =
                if state[res] <> s then
                    transition res state[res] s
                    state[res] <- s

            let mutable at = 0L

            for u in job.Uploads do
                if at + int64 u.Data.Length > stagingBytes then
                    failwith "upload staging too small"

                copyIn (uploadPtr + nativeint at) u.Data
                let res = fst pages[u.Slot]
                toState res StateCopyDest
                copy.Invoke(list, res, uint64 u.Offset, upload, uint64 at, uint64 u.Data.Length)
                at <- int64 (alignUp (uint64 (at + int64 u.Data.Length)) 256UL)

            for p in distinctPages do
                toState p StateUav

            let mutable heapArr = heap
            // SetDescriptorHeaps must precede SetComputeRootSignature for a directly indexed heap.
            (fn<D_SetHeaps> list 28).Invoke(list, 1u, &heapArr)
            (fn<D_SetObj> list 29).Invoke(list, rootSig)
            let consts = Marshal.AllocHGlobal 128

            try
                job.Dispatches
                |> List.iteri (fun k d ->
                    if d.Args.Length <> 32 || d.Args[0] <> d.GroupsX then
                        failwith "ArgBlock must be 32 dwords with dword0 = groups_x"

                    (fn<D_SetObj> list 25).Invoke(list, psos[d.Kernel])

                    for i in 0..31 do
                        Marshal.WriteInt32(consts, 4 * i, int d.Args[i])

                    (fn<D_SetConsts> list 35).Invoke(list, 0u, 32u, consts, 0u)
                    (fn<D_EndQuery> list 53).Invoke(list, queryHeap, 2, uint32 (2 * k))
                    (fn<D_Dispatch> list 14).Invoke(list, d.GroupsX, d.GroupsY, 1u)
                    (fn<D_EndQuery> list 53).Invoke(list, queryHeap, 2, uint32 (2 * k + 1))
                    uavBarrier ())
            finally
                Marshal.FreeHGlobal consts

            if nq > 0 then
                (fn<D_Resolve> list 54).Invoke(list, queryHeap, 2, 0u, uint32 nq, queryReadback, 0UL)

            let mutable rat = 0L

            let readbacks =
                [
                    for r in job.Readbacks do
                        if rat + int64 r.Length > stagingBytes then
                            failwith "readback staging too small"

                        let res = fst pages[r.Slot]
                        toState res StateCopySource
                        copy.Invoke(list, readback, uint64 rat, res, uint64 r.Offset, uint64 r.Length)
                        yield r, rat
                        rat <- int64 (alignUp (uint64 (rat + int64 r.Length)) 256UL)
                ]

            hr "Close" ((fn<D_Close> list 9).Invoke list)
            let mutable l = list
            (fn<D_Exec> queue 10).Invoke(queue, 1u, &l)
            fenceValue <- fenceValue + 1UL
            hr "Signal" ((fn<D_Signal> queue 14).Invoke(queue, fence, fenceValue))

            if (fn<D_Completed> fence 8).Invoke fence < fenceValue then
                hr "SetEventOnCompletion" ((fn<D_SetEvent> fence 9).Invoke(fence, fenceValue, event))

                if Native.WaitForSingleObject(event, 10000u) <> 0u then
                    failwith "D3D12 fence wait timed out after 10 s"

            let removed = (fn<D_RemovedReason> dev 37).Invoke dev

            if removed <> 0 then
                failwithf "D3D12 device removed: 0x%08X" removed

            let outputs =
                [ for (r, off) in readbacks -> copyOut (readbackPtr + nativeint off) r.Length ]

            let times =
                [
                    for k in 0 .. job.Dispatches.Length - 1 do
                        let t0 = Marshal.ReadInt64(queryPtr, 16 * k) |> uint64
                        let t1 = Marshal.ReadInt64(queryPtr, 16 * k + 8) |> uint64
                        float (t1 - t0) * 1e9 / float tsFreq
                ]

            { Outputs = outputs; DispatchGpuNs = times }

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                Native.CloseHandle event |> ignore

                for p in Seq.rev created do
                    release p
