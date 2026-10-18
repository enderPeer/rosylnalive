namespace Lrm.Gpu.D3D12

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open System.Text
open Lrm.Core
open Lrm.Gpu.D3D12.Interop
open Microsoft.FSharp.NativeInterop

/// One command list's DRED auto-breadcrumbs: the operations it recorded and how many of them the GPU finished.
type Breadcrumbs =
    {
        /// The command list's debug name (ID3D12Object::SetName), or "".
        CommandList: string
        /// The queue's debug name, or "".
        CommandQueue: string
        /// Every operation the list recorded, in order.
        Ops: D3D12_AUTO_BREADCRUMB_OP list
        /// Operations the GPU completed (the list's last breadcrumb value), or None when DRED did not say.
        Completed: uint32 option
        /// Breadcrumb context strings (SetMarker text) by operation index.
        Contexts: (uint32 * string) list
    }

/// One allocation named in DRED's page-fault report.
type DredAllocation =
    {
        Name: string
        Kind: D3D12_DRED_ALLOCATION_TYPE
    }

/// DRED's page-fault report: the faulting GPU virtual address and the allocations near it.
type PageFault =
    {
        Address: uint64
        Existing: DredAllocation list
        RecentlyFreed: DredAllocation list
    }

/// What DRED (device removed extended data) reports for a device.
type DredReport =
    {
        /// True when DRED was forced on for the device (Dred.withEnabled).
        Enabled: bool
        Breadcrumbs: Breadcrumbs list
        PageFault: PageFault option
        /// Why a part is missing, for example "DRED is not available from this runtime".
        Notes: string list
    }

/// DRED, the diagnostics D3D12 keeps for a device removal (dev plan: GetDeviceRemovedReason after every submit, with
/// DRED breadcrumbs). Its settings are process-wide and apply to devices created while they are set, so the backend
/// forces them on around its own D3D12CreateDevice call and restores the system defaults afterwards.
[<RequireQualifiedAccess>]
module Dred =
    let private gate = obj ()

    /// Lists longer than this are cut: DRED's lists are runtime memory, and a corrupt list must not hang the report.
    [<Literal>]
    let MaxNodes = 4096

    /// Operations kept per command list.
    [<Literal>]
    let MaxOps = 65536

    let private set (s: ComPtr<ID3D12DeviceRemovedExtendedDataSettings>) (e: D3D12_DRED_ENABLEMENT) =
        s.SetAutoBreadcrumbsEnablement e
        s.SetPageFaultEnablement e

    /// Runs `create` with DRED auto-breadcrumbs, breadcrumb contexts and page-fault reporting forced on, then restores
    /// the system-controlled defaults. Returns the result and whether DRED could be forced on (false when the runtime
    /// does not expose ID3D12DeviceRemovedExtendedDataSettings).
    let withEnabled (create: unit -> 'T) : 'T * bool =
        lock gate (fun () ->
            match D3D12Api.tryGetDebugInterface<ID3D12DeviceRemovedExtendedDataSettings1>() with
            | ValueSome s1 ->
                use s1 = s1
                use s = s1.QueryInterface<ID3D12DeviceRemovedExtendedDataSettings>()
                set s D3D12_DRED_ENABLEMENT.FORCED_ON
                s1.SetBreadcrumbContextEnablement D3D12_DRED_ENABLEMENT.FORCED_ON

                try
                    create (), true
                finally
                    set s D3D12_DRED_ENABLEMENT.SYSTEM_CONTROLLED
                    s1.SetBreadcrumbContextEnablement D3D12_DRED_ENABLEMENT.SYSTEM_CONTROLLED
            | ValueNone ->
                match D3D12Api.tryGetDebugInterface<ID3D12DeviceRemovedExtendedDataSettings>() with
                | ValueSome s ->
                    use s = s
                    set s D3D12_DRED_ENABLEMENT.FORCED_ON

                    try
                        create (), true
                    finally
                        set s D3D12_DRED_ENABLEMENT.SYSTEM_CONTROLLED
                | ValueNone -> create (), false)

    let private wide (p: nativeint) =
        if p = 0n then "" else Marshal.PtrToStringUni p

    let private narrow (p: nativeint) =
        if p = 0n then "" else Marshal.PtrToStringAnsi p

    let private nameOf (w: nativeint) (a: nativeint) =
        match wide w with
        | "" -> narrow a
        | s -> s

    /// Reads a list of D3D12_AUTO_BREADCRUMB_NODE1 from its head (runtime memory; a null head is an empty list).
    let readBreadcrumbs (head: nativeint) : Breadcrumbs list =
        let nodes = ResizeArray<Breadcrumbs>()
        let mutable p = head

        while p <> 0n && nodes.Count < MaxNodes do
            let n = NativePtr.read (NativePtr.ofNativeInt<D3D12_AUTO_BREADCRUMB_NODE1> p)
            let count = int (min n.BreadcrumbCount (uint32 MaxOps))

            let ops =
                if n.pCommandHistory = 0n then
                    []
                else
                    let h = NativePtr.ofNativeInt<int> n.pCommandHistory

                    [
                        for i in 0 .. count - 1 ->
                            LanguagePrimitives.EnumOfValue<int, D3D12_AUTO_BREADCRUMB_OP>(NativePtr.get h i)
                    ]

            let completed =
                if n.pLastBreadcrumbValue = 0n then
                    None
                else
                    Some(NativePtr.read (NativePtr.ofNativeInt<uint32> n.pLastBreadcrumbValue))

            let contexts =
                if n.pBreadcrumbContexts = 0n then
                    []
                else
                    let c = NativePtr.ofNativeInt<D3D12_DRED_BREADCRUMB_CONTEXT> n.pBreadcrumbContexts

                    [
                        for i in 0 .. int (min n.BreadcrumbContextsCount (uint32 MaxOps)) - 1 do
                            let x = NativePtr.get c i
                            yield x.BreadcrumbIndex, wide x.pContextString
                    ]

            nodes.Add
                {
                    CommandList = nameOf n.pCommandListDebugNameW n.pCommandListDebugNameA
                    CommandQueue = nameOf n.pCommandQueueDebugNameW n.pCommandQueueDebugNameA
                    Ops = ops
                    Completed = completed
                    Contexts = contexts
                }

            p <- n.pNext

        List.ofSeq nodes

    /// Reads a list of D3D12_DRED_ALLOCATION_NODE1 from its head.
    let readAllocations (head: nativeint) : DredAllocation list =
        let nodes = ResizeArray<DredAllocation>()
        let mutable p = head

        while p <> 0n && nodes.Count < MaxNodes do
            let n = NativePtr.read (NativePtr.ofNativeInt<D3D12_DRED_ALLOCATION_NODE1> p)

            nodes.Add
                {
                    Name = nameOf n.ObjectNameW n.ObjectNameA
                    Kind = n.AllocationType
                }

            p <- n.pNext

        List.ofSeq nodes

    /// DRED's report for `device`. Never raises: what cannot be read is named in Notes.
    let report (enabled: bool) (device: ComPtr<ID3D12Device>) : DredReport =
        try
            match device.TryQueryInterface<ID3D12DeviceRemovedExtendedData1>() with
            | ValueNone ->
                {
                    Enabled = enabled
                    Breadcrumbs = []
                    PageFault = None
                    Notes = [ "the device has no ID3D12DeviceRemovedExtendedData1" ]
                }
            | ValueSome data ->
                use data = data
                let notes = ResizeArray<string>()
                let mutable crumbs = D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1()
                let hr = data.GetAutoBreadcrumbsOutput1Raw(NativePtr.toNativeInt &&crumbs)

                let breadcrumbs =
                    if hr >= 0 then
                        readBreadcrumbs crumbs.pHeadAutoBreadcrumbNode
                    else
                        notes.Add(sprintf "no breadcrumbs: GetAutoBreadcrumbsOutput1 returned %s" (HResult.describe hr))
                        []

                let mutable fault = D3D12_DRED_PAGE_FAULT_OUTPUT1()
                let hr = data.GetPageFaultAllocationOutput1Raw(NativePtr.toNativeInt &&fault)

                let pageFault =
                    if hr >= 0 then
                        if
                            fault.PageFaultVA = 0UL
                            && fault.pHeadExistingAllocationNode = 0n
                            && fault.pHeadRecentFreedAllocationNode = 0n
                        then
                            None
                        else
                            Some
                                {
                                    Address = fault.PageFaultVA
                                    Existing = readAllocations fault.pHeadExistingAllocationNode
                                    RecentlyFreed = readAllocations fault.pHeadRecentFreedAllocationNode
                                }
                    else
                        notes.Add(
                            sprintf
                                "no page-fault report: GetPageFaultAllocationOutput1 returned %s"
                                (HResult.describe hr)
                        )

                        None

                {
                    Enabled = enabled
                    Breadcrumbs = breadcrumbs
                    PageFault = pageFault
                    Notes = List.ofSeq notes
                }
        with e ->
            {
                Enabled = enabled
                Breadcrumbs = []
                PageFault = None
                Notes = [ "reading DRED failed: " + e.Message ]
            }

    let private opName (op: D3D12_AUTO_BREADCRUMB_OP) =
        match Enum.GetName(typeof<D3D12_AUTO_BREADCRUMB_OP>, op) with
        | null -> Invariant.i32 (int op)
        | n -> n

    /// The report as text: per command list the operation count, how many completed, and the operations around the
    /// first unfinished one (marked with >); then the page fault and the notes. Deterministic.
    let format (r: DredReport) : string =
        let sb = StringBuilder()

        let line (s: string) = sb.Append(s).Append('\n') |> ignore

        if not r.Enabled then
            line "DRED: not forced on for this device"

        for b in r.Breadcrumbs do
            let total = b.Ops.Length

            let completed =
                match b.Completed with
                | Some c -> Invariant.i32 (int c)
                | None -> "?"

            line (
                Invariant.format
                    $"DRED breadcrumbs: list '{b.CommandList}' on queue '{b.CommandQueue}': {total} ops, {completed} completed"
            )

            let first = b.Completed |> Option.map int |> Option.defaultValue 0
            let lo = max 0 (first - 3)
            let hi = min (total - 1) (first + 3)

            for i in lo..hi do
                let mark = if i = first then ">" else " "

                let ctx =
                    b.Contexts
                    |> List.tryFind (fun (k, _) -> int k = i)
                    |> Option.map (fun (_, s) -> " \"" + s + "\"")
                    |> Option.defaultValue ""

                line (Invariant.format $"  {mark} {i}: {opName b.Ops[i]}{ctx}")

        match r.PageFault with
        | Some f ->
            line (Invariant.format $"DRED page fault at GPU VA 0x{f.Address:x16}")

            for a in f.Existing do
                line (Invariant.format $"  existing: '{a.Name}' ({a.Kind})")

            for a in f.RecentlyFreed do
                line (Invariant.format $"  recently freed: '{a.Name}' ({a.Kind})")
        | None -> ()

        for n in r.Notes do
            line ("DRED: " + n)

        sb.ToString().TrimEnd('\n')
