/// Fake COM objects in unmanaged memory, so the cpu tier can exercise ComPtr, the delegate cache, the generated
/// wrappers, RetVal returns and D3D12Exception without a GPU. Each object is a pointer to a vtable whose IUnknown
/// slots are managed functions over a reference count; tests fill the other slots they call.
module Lrm.Gpu.D3D12.Interop.Tests.FakeCom

open System
open System.Runtime.InteropServices
open Lrm.Gpu.D3D12.Interop

type FakeObject(slotCount: int, iids: Guid list) =
    let keep = ResizeArray<Delegate>()
    let vtbl = Marshal.AllocHGlobal(slotCount * IntPtr.Size)
    let obj = Marshal.AllocHGlobal IntPtr.Size
    let mutable refs = 1
    let mutable queries = 0
    let supported = Iids.IUnknown :: iids

    let set (slot: int) (d: Delegate) =
        keep.Add d
        Marshal.WriteIntPtr(vtbl, slot * IntPtr.Size, Marshal.GetFunctionPointerForDelegate d)

    do
        for i in 0 .. slotCount - 1 do
            Marshal.WriteIntPtr(vtbl, i * IntPtr.Size, 0n)

        Marshal.WriteIntPtr(obj, vtbl)

        set
            Slots.IUnknown.QueryInterface
            (Fn.IUnknown.QueryInterface(fun _ riid ppv ->
                queries <- queries + 1
                let iid = Marshal.PtrToStructure<Guid> riid

                if List.contains iid supported then
                    refs <- refs + 1
                    Marshal.WriteIntPtr(ppv, obj)
                    HResult.S_OK
                else
                    Marshal.WriteIntPtr(ppv, 0n)
                    HResult.E_NOINTERFACE))

        set
            Slots.IUnknown.AddRef
            (Fn.IUnknown.AddRef(fun _ ->
                refs <- refs + 1
                uint32 refs))

        set
            Slots.IUnknown.Release
            (Fn.IUnknown.Release(fun _ ->
                refs <- refs - 1
                uint32 (max refs 0)))

    /// The COM pointer (pointer to the vtable pointer).
    member _.Ptr = obj
    /// The current reference count; it starts at 1, the reference a creator hands out.
    member _.Refs = refs
    /// How many QueryInterface calls were made.
    member _.Queries = queries
    /// Fills a vtable slot with a delegate of the generated Fn type for that method.
    member _.Set(slot: int, d: Delegate) = set slot d

    interface IDisposable with
        member _.Dispose() =
            Marshal.FreeHGlobal obj
            Marshal.FreeHGlobal vtbl
