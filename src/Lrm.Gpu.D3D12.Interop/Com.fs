namespace Lrm.Gpu.D3D12.Interop

// The D3D12ComIdiom contract (docs/contracts/d3d12-com-idiom.md): COM objects are plain pointers. ComPtr<'I> is a
// struct over one nativeint, typed by a phantom marker interface; calls go through generated vtable slots and IIDs
// and delegates cached per function pointer; struct returns use the hidden RetVal pointer; failed HRESULTs raise
// D3D12Exception with the device-removed reason. No ComImport, no runtime-callable wrappers, no ComWrappers.

#nowarn "9"
#nowarn "51"

open System
open System.Collections.Concurrent
open System.Runtime.CompilerServices
open System.Threading
open Microsoft.FSharp.NativeInterop

/// Every P/Invoke and unmanaged delegate in this assembly is blittable; the runtime marshaller is off, so a
/// non-blittable signature fails at once instead of being marshalled silently.
module internal AssemblyAttributes =
    [<assembly: DisableRuntimeMarshalling>]
    do ()

/// HRESULT codes the interop names (winerror.h). The ABI oracle checks each value against the SDK.
[<RequireQualifiedAccess>]
module HResult =
    [<Literal>]
    let S_OK = 0

    [<Literal>]
    let S_FALSE = 1

    [<Literal>]
    let E_NOTIMPL = 0x80004001

    [<Literal>]
    let E_NOINTERFACE = 0x80004002

    [<Literal>]
    let E_POINTER = 0x80004003

    [<Literal>]
    let E_FAIL = 0x80004005

    [<Literal>]
    let E_OUTOFMEMORY = 0x8007000E

    [<Literal>]
    let E_INVALIDARG = 0x80070057

    [<Literal>]
    let DXGI_ERROR_INVALID_CALL = 0x887A0001

    [<Literal>]
    let DXGI_ERROR_NOT_FOUND = 0x887A0002

    [<Literal>]
    let DXGI_ERROR_MORE_DATA = 0x887A0003

    [<Literal>]
    let DXGI_ERROR_UNSUPPORTED = 0x887A0004

    [<Literal>]
    let DXGI_ERROR_DEVICE_REMOVED = 0x887A0005

    [<Literal>]
    let DXGI_ERROR_DEVICE_HUNG = 0x887A0006

    [<Literal>]
    let DXGI_ERROR_DEVICE_RESET = 0x887A0007

    [<Literal>]
    let DXGI_ERROR_WAS_STILL_DRAWING = 0x887A000A

    [<Literal>]
    let DXGI_ERROR_DRIVER_INTERNAL_ERROR = 0x887A0020

    [<Literal>]
    let DXGI_ERROR_NOT_CURRENTLY_AVAILABLE = 0x887A0022

    [<Literal>]
    let DXGI_ERROR_SDK_COMPONENT_MISSING = 0x887A002D

    [<Literal>]
    let D3D12_ERROR_ADAPTER_NOT_FOUND = 0x887E0001

    [<Literal>]
    let D3D12_ERROR_DRIVER_VERSION_MISMATCH = 0x887E0002

    [<Literal>]
    let D3D12_ERROR_INVALID_REDIST = 0x887E0003

    /// Every named code, in the order above.
    let known: (string * int) list =
        [
            "S_OK", S_OK
            "S_FALSE", S_FALSE
            "E_NOTIMPL", E_NOTIMPL
            "E_NOINTERFACE", E_NOINTERFACE
            "E_POINTER", E_POINTER
            "E_FAIL", E_FAIL
            "E_OUTOFMEMORY", E_OUTOFMEMORY
            "E_INVALIDARG", E_INVALIDARG
            "DXGI_ERROR_INVALID_CALL", DXGI_ERROR_INVALID_CALL
            "DXGI_ERROR_NOT_FOUND", DXGI_ERROR_NOT_FOUND
            "DXGI_ERROR_MORE_DATA", DXGI_ERROR_MORE_DATA
            "DXGI_ERROR_UNSUPPORTED", DXGI_ERROR_UNSUPPORTED
            "DXGI_ERROR_DEVICE_REMOVED", DXGI_ERROR_DEVICE_REMOVED
            "DXGI_ERROR_DEVICE_HUNG", DXGI_ERROR_DEVICE_HUNG
            "DXGI_ERROR_DEVICE_RESET", DXGI_ERROR_DEVICE_RESET
            "DXGI_ERROR_WAS_STILL_DRAWING", DXGI_ERROR_WAS_STILL_DRAWING
            "DXGI_ERROR_DRIVER_INTERNAL_ERROR", DXGI_ERROR_DRIVER_INTERNAL_ERROR
            "DXGI_ERROR_NOT_CURRENTLY_AVAILABLE", DXGI_ERROR_NOT_CURRENTLY_AVAILABLE
            "DXGI_ERROR_SDK_COMPONENT_MISSING", DXGI_ERROR_SDK_COMPONENT_MISSING
            "D3D12_ERROR_ADAPTER_NOT_FOUND", D3D12_ERROR_ADAPTER_NOT_FOUND
            "D3D12_ERROR_DRIVER_VERSION_MISMATCH", D3D12_ERROR_DRIVER_VERSION_MISMATCH
            "D3D12_ERROR_INVALID_REDIST", D3D12_ERROR_INVALID_REDIST
        ]

    /// The symbolic name of a code, when the interop knows it.
    let name (hr: int) =
        known |> List.tryFind (fun (_, v) -> v = hr) |> Option.map fst

    /// "0x887A0005 (DXGI_ERROR_DEVICE_REMOVED)", or just the hex for an unnamed code.
    let describe (hr: int) =
        match name hr with
        | Some n -> sprintf "0x%08X (%s)" hr n
        | None -> sprintf "0x%08X" hr

    /// The device is gone: removed, hung, reset or the driver failed internally.
    let isDeviceLost (hr: int) =
        hr = DXGI_ERROR_DEVICE_REMOVED
        || hr = DXGI_ERROR_DEVICE_HUNG
        || hr = DXGI_ERROR_DEVICE_RESET
        || hr = DXGI_ERROR_DRIVER_INTERNAL_ERROR

module internal ExceptionText =
    let format (call: string) (code: int) (removedReason: int option) =
        let removed =
            match removedReason with
            | Some 0 -> "; the device is not removed"
            | Some r -> sprintf "; device removed reason %s" (HResult.describe r)
            | None -> ""

        sprintf "%s failed with %s%s" call (HResult.describe code) removed

/// A failed HRESULT from a D3D12 or DXGI call. RemovedReason is ID3D12Device::GetDeviceRemovedReason for the
/// device that owns the called object, when the object is a D3D12 device or device child.
type D3D12Exception(call: string, code: int, removedReason: int option) as this =
    inherit Exception(ExceptionText.format call code removedReason)
    do this.HResult <- code

    /// The failed call, for example "ID3D12Device::CreateCommittedResource".
    member _.Call = call
    /// The HRESULT.
    member _.Code = code
    /// The device-removed reason (S_OK when the device is fine), if the object led to a D3D12 device.
    member _.RemovedReason = removedReason

    /// True when the HRESULT or the removed reason says the device is gone.
    member _.DeviceLost =
        HResult.isDeviceLost code || (removedReason |> Option.exists (fun r -> r <> 0))

type internal CachedDelegate<'D>(fp: nativeint, d: 'D) =
    member _.Fp = fp
    member _.Delegate = d

/// One delegate per (delegate type, function pointer), created once. The last hit is kept in a field so the common
/// case (every object of one interface shares one vtable) skips the dictionary.
type internal DelegateCache<'D when 'D :> Delegate> private () =
    static let map = ConcurrentDictionary<nativeint, 'D>()

    static let mutable last = CachedDelegate<'D>(0n, Unchecked.defaultof<'D>)

    static member Get(fp: nativeint) : 'D =
        let hit = last

        if hit.Fp = fp && fp <> 0n then
            hit.Delegate
        else
            let d =
                map.GetOrAdd(fp, (fun p -> Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer<'D> p))

            last <- CachedDelegate<'D>(fp, d)
            d

/// Vtable access for generated and hand-written calls.
[<RequireQualifiedAccess>]
module Vtbl =
    /// The function pointer in `slot` of the COM object `this`.
    let slotPointer (this: nativeint) (slot: int) : nativeint =
        let vtbl = NativePtr.read (NativePtr.ofNativeInt<nativeint> this)
        NativePtr.get (NativePtr.ofNativeInt<nativeint> vtbl) slot

    let private nullCall (slot: int) =
        raise (NullReferenceException(sprintf "COM call through vtable slot %d on a null interface pointer" slot))

    /// The cached delegate for `slot` of `this`. Raises NullReferenceException for a null pointer.
    let get<'D when 'D :> Delegate> (this: nativeint) (slot: int) : 'D =
        if this = 0n then
            nullCall slot

        DelegateCache<'D>.Get(slotPointer this slot)

/// Raw IUnknown calls, reference tracking and HRESULT checks.
[<RequireQualifiedAccess>]
module Com =
    let private live = ref 0L

    /// References currently owned through ComPtr (created by Attach, AddRef or QueryInterface, minus releases).
    /// Leak tests compare it before and after.
    let liveReferences () = Interlocked.Read &live.contents

    let internal track (delta: int64) =
        Interlocked.Add(&live.contents, delta) |> ignore

    /// IUnknown::AddRef; returns the new count.
    let addRefRaw (p: nativeint) : uint32 =
        (Vtbl.get<Fn.IUnknown.AddRef> p Slots.IUnknown.AddRef).Invoke p

    /// IUnknown::Release; returns the new count.
    let releaseRaw (p: nativeint) : uint32 =
        (Vtbl.get<Fn.IUnknown.Release> p Slots.IUnknown.Release).Invoke p

    /// IUnknown::QueryInterface; returns the HRESULT and the new pointer (owned by the caller).
    let queryInterfaceRaw (p: nativeint) (iid: Guid) : int * nativeint =
        let mutable iid = iid
        let mutable out = 0n

        let hr =
            (Vtbl.get<Fn.IUnknown.QueryInterface> p Slots.IUnknown.QueryInterface)
                .Invoke(p, NativePtr.toNativeInt &&iid, NativePtr.toNativeInt &&out)

        hr, out

    let private reasonOfDevice (dev: nativeint) =
        (Vtbl.get<Fn.ID3D12Device.GetDeviceRemovedReason> dev Slots.ID3D12Device.GetDeviceRemovedReason).Invoke dev

    /// GetDeviceRemovedReason of the device behind `obj`: obj itself when it is a device, else the device of a
    /// device child. None for other objects (DXGI) or when the queries fail. Used only on failure paths.
    let removedReason (obj: nativeint) : int option =
        if obj = 0n then
            None
        else
            try
                let hr, dev = queryInterfaceRaw obj Iids.ID3D12Device

                if hr >= 0 && dev <> 0n then
                    let r = reasonOfDevice dev
                    releaseRaw dev |> ignore
                    Some r
                else
                    let hr, child = queryInterfaceRaw obj Iids.ID3D12DeviceChild

                    if hr >= 0 && child <> 0n then
                        let mutable iid = Iids.ID3D12Device
                        let mutable dev = 0n

                        let hr =
                            (Vtbl.get<Fn.ID3D12DeviceChild.GetDevice> child Slots.ID3D12DeviceChild.GetDevice)
                                .Invoke(child, NativePtr.toNativeInt &&iid, NativePtr.toNativeInt &&dev)

                        releaseRaw child |> ignore

                        if hr >= 0 && dev <> 0n then
                            let r = reasonOfDevice dev
                            releaseRaw dev |> ignore
                            Some r
                        else
                            None
                    else
                        None
            with _ ->
                None

    let private fail (call: string) (hr: int) (this: nativeint) =
        raise (D3D12Exception(call, hr, removedReason this))

    /// Raises D3D12Exception when `hr` is a failure; `this` is the called object (for the removed reason).
    let check (call: string) (hr: int) (this: nativeint) : unit =
        if hr < 0 then
            fail call hr this

    /// Calls `f` with a pointer to the NUL-terminated UTF-16 text of `s` (null for a null string), pinned for the call.
    let withWide (s: string) (f: nativeint -> 'R) : 'R =
        if isNull s then
            f 0n
        else
            use p = fixed s
            f (NativePtr.toNativeInt p)

/// The IID of a marker interface, looked up once per type.
type IidOf<'I when 'I :> IUnknown> private () =
    static let value =
        match Iids.byType.TryGetValue typeof<'I> with
        | true, g -> g
        | _ -> invalidOp (sprintf "%s is not a bound COM interface" typeof<'I>.FullName)

    static member Value: Guid = value

/// A COM interface pointer, typed by the phantom marker 'I. A struct over one nativeint: copying it copies the
/// pointer, not a reference. Exactly one variable owns each reference and releases it once, with `use`, Dispose or
/// ComPtr.release (which also clears the variable). Wrappers that return a ComPtr hand the caller a new reference.
/// There is no public constructor: Attach takes ownership of a reference (and counts it), Borrow wraps one that
/// something else owns.
[<Struct; CustomEquality; NoComparison>]
type ComPtr<'I when 'I :> IUnknown> =
    val Ptr: nativeint
    private new(ptr: nativeint) = { Ptr = ptr }

    member p.IsNull = p.Ptr = 0n

    static member Null: ComPtr<'I> = new ComPtr<'I>(0n)

    /// Takes ownership of a reference received from COM (an out parameter) and counts it as live.
    static member Attach(ptr: nativeint) : ComPtr<'I> =
        if ptr <> 0n then
            Com.track 1L

        new ComPtr<'I>(ptr)

    /// Wraps a pointer whose reference someone else owns (a pointer inside a D3D12 struct, or one the caller keeps),
    /// without counting it. A borrowed ComPtr must never be released or disposed: that would release a reference it
    /// does not own and undercount Com.liveReferences. Call AddRef for an owned copy.
    static member Borrow(ptr: nativeint) : ComPtr<'I> = new ComPtr<'I>(ptr)

    /// A new owned reference to the same object (IUnknown::AddRef).
    member p.AddRef() : ComPtr<'I> =
        if p.Ptr = 0n then
            ComPtr<'I>.Null
        else
            Com.addRefRaw p.Ptr |> ignore
            ComPtr<'I>.Attach p.Ptr

    /// A new owned reference to interface 'J of the same object. Raises D3D12Exception (E_NOINTERFACE) when the
    /// object does not implement it.
    member p.QueryInterface<'J when 'J :> IUnknown>() : ComPtr<'J> =
        let hr, out = Com.queryInterfaceRaw p.Ptr IidOf<'J>.Value
        Com.check (sprintf "%s::QueryInterface(%s)" typeof<'I>.Name typeof<'J>.Name) hr p.Ptr
        ComPtr<'J>.Attach out

    /// Like QueryInterface, but ValueNone when the object does not implement 'J.
    member p.TryQueryInterface<'J when 'J :> IUnknown>() : ComPtr<'J> voption =
        if p.Ptr = 0n then
            ValueNone
        else
            let hr, out = Com.queryInterfaceRaw p.Ptr IidOf<'J>.Value

            if hr >= 0 && out <> 0n then
                ValueSome(ComPtr<'J>.Attach out)
            else
                ValueNone

    /// Releases this reference and returns the count IUnknown::Release reports (0 for a null pointer).
    /// The value must not be used afterwards; ComPtr.release also clears the variable.
    member p.Release() : uint32 =
        if p.Ptr = 0n then
            0u
        else
            Com.track -1L
            Com.releaseRaw p.Ptr

    member p.Dispose() = p.Release() |> ignore

    interface IDisposable with
        member p.Dispose() = p.Release() |> ignore

    override p.Equals(o: obj) =
        match o with
        | :? ComPtr<'I> as q -> q.Ptr = p.Ptr
        | _ -> false

    override p.GetHashCode() = p.Ptr.GetHashCode()

    override p.ToString() =
        sprintf "ComPtr<%s>(0x%X)" typeof<'I>.Name p.Ptr

/// Ownership helpers for ComPtr.
[<RequireQualifiedAccess; CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module ComPtr =
    /// Releases the reference held by `p` and clears the variable, so the reference is released exactly once
    /// however often this runs.
    let release (p: byref<ComPtr<'I>>) : unit =
        let v = p
        p <- ComPtr<'I>.Null
        v.Release() |> ignore

    /// The raw pointers of `xs`, for the …Raw forms of array methods (the typed ExecuteCommandLists or
    /// SetDescriptorHeaps take a ReadOnlySpan<ComPtr<_>> or a single ComPtr). No references are added.
    let pointers (xs: ComPtr<'I> seq) : nativeint[] = xs |> Seq.map _.Ptr |> Array.ofSeq
