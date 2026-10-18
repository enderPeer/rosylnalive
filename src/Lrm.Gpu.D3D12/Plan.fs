namespace Lrm.Gpu.D3D12

open System
open System.Collections.Generic
open Lrm.Core
open Lrm.Gpu.Core

/// The resource states the backend tracks per page (and for its copy scratch buffer). D3D12 needs a buffer in
/// UNORDERED_ACCESS for kernel access through the heap, COPY_SOURCE or COPY_DEST for CopyBufferRegion, and every
/// buffer is COMMON between command lists: the encoder starts each list with every page in COMMON and returns every
/// page it touched to COMMON at the end, so the invariant does not depend on D3D12's implicit promotion and decay.
[<RequireQualifiedAccess>]
type PageState =
    | Common
    | UnorderedAccess
    | CopySource
    | CopyDest

/// A buffer the encoder names. Pages and the scratch buffer are state-tracked; staging buffers live on UPLOAD heaps
/// (always GENERIC_READ) and READBACK heaps (always COPY_DEST), so they never transition.
[<RequireQualifiedAccess; Struct>]
type Target =
    /// A heap slot: 0 is the tensor table, 1..63 the pages.
    | Page of slot: int
    /// The backend's DEFAULT-heap scratch buffer, which carries copies within one page (Plan.ScratchBytes).
    | Scratch
    /// Upload staging buffer i.
    | Upload of upload: int
    /// Readback staging buffer i.
    | Readback of readback: int

    override t.ToString() =
        match t with
        | Page s -> sprintf "page %d" s
        | Scratch -> "scratch"
        | Upload i -> sprintf "upload %d" i
        | Readback i -> sprintf "readback %d" i

/// One entry of a ResourceBarrier call.
[<RequireQualifiedAccess>]
type BarrierOp =
    /// A transition barrier on every subresource of the target.
    | Transition of target: Target * before: PageState * after: PageState
    /// A UAV barrier with a null resource: every earlier UAV access finishes before any later one starts.
    | AllUav

/// One native command-list call, in recording order.
[<RequireQualifiedAccess>]
type NativeOp =
    /// One ResourceBarrier call.
    | Barriers of BarrierOp list
    /// SetPipelineState with the kernel's pipeline.
    | Pipeline of kernel: KernelId
    /// SetComputeRoot32BitConstants(0, the 32 words, 0): the argument block at b0.
    | Constants of words: uint32[]
    /// Dispatch(x, y, z) of the physical grid.
    | Dispatch of grid: PhysicalGrid
    /// CopyBufferRegion(destination, destinationOffset, source, sourceOffset, bytes).
    | Copy of destination: Target * destinationOffset: int64 * source: Target * sourceOffset: int64 * bytes: int64
    /// EndQuery(TIMESTAMP, query).
    | Timestamp of query: int
    /// ResolveQueryData(TIMESTAMP, 0, count) into the frame's timestamp readback buffer.
    | ResolveTimestamps of count: int

/// A command as an ICommandList recorded it: checked when recorded (Dispatch.prepare, Dispatch.checkCopy, the timer
/// rules), encoded at Submit.
[<RequireQualifiedAccess>]
type Command =
    | Dispatch of PreparedDispatch
    | Barrier
    | Copy of source: BufferRange * destination: BufferRange
    | BeginTimer of timer: int * name: string * attrs: Attr list
    | EndTimer of timer: int

/// A timed region of an encoded list: the two timestamp queries that bracket it.
type TimerSlot =
    {
        Name: string
        Attrs: Attr list
        Begin: int
        End: int
    }

/// An encoded command list.
type Encoded =
    {
        Ops: NativeOp list
        /// Timestamp queries the list writes (0..Queries-1), resolved at its end.
        Queries: int
        /// Timers in the order they were opened.
        Timers: TimerSlot list
        /// gpu.dispatch timings, one per dispatch when dispatches are timed, in dispatch order.
        DispatchTimers: TimerSlot list
        Dispatches: int
        /// The byte count of every Copy command, in order (gpu.bytes_moved, direction copy).
        Copies: int64 list
    }

    /// True when the list dispatches kernels, so it binds the descriptor heap and the root signature first.
    member e.Dispatching = e.Dispatches > 0

/// The tracked state of every target within one command list. Every target starts in COMMON.
[<Sealed>]
type StateTracker() =
    let states = Dictionary<Target, PageState>()

    /// The current state of a target.
    member _.State(t: Target) =
        match states.TryGetValue t with
        | true, s -> s
        | _ -> PageState.Common

    /// Moves `t` to `s`: the transition barrier, or None when it is already there.
    member x.To(t: Target, s: PageState) : BarrierOp option =
        let current = x.State t

        if current = s then
            None
        else
            states[t] <- s
            Some(BarrierOp.Transition(t, current, s))

    /// Every target that is not in COMMON goes back to it, in target order.
    member _.Finish() : BarrierOp list =
        let pending =
            states
            |> Seq.filter (fun kv -> kv.Value <> PageState.Common)
            |> Seq.map (fun kv -> kv.Key, kv.Value)
            |> Seq.sortBy fst
            |> List.ofSeq

        for t, _ in pending do
            states[t] <- PageState.Common

        pending |> List.map (fun (t, s) -> BarrierOp.Transition(t, s, PageState.Common))

/// The command encoder: a pure function from recorded commands to native calls, with per-page state tracking
/// (GpuBackendInterface: Copy is state-correct; D3D12 inserts the UAV-to-COPY transitions and back). The cpu tier
/// checks it with Plan.verify; the backend replays the ops on an ID3D12GraphicsCommandList.
[<RequireQualifiedAccess>]
module Plan =
    /// The scratch buffer that carries copies within one page, and the largest chunk of such a copy.
    [<Literal>]
    let ScratchBytes = 67108864L

    let private barriers (xs: BarrierOp option list) =
        match List.choose id xs with
        | [] -> []
        | bs -> [ NativeOp.Barriers bs ]

    /// The gpu.dispatch attributes of each catalog kernel, one list per kernel shared by all its timed dispatches, so
    /// a retained timing costs no attribute list of its own.
    let private dispatchAttrs: Attr list[] =
        Kernels.all |> Array.map (fun k -> [ Attr.str TelemetryKeys.Kernel k.Name ])

    let private attrsOf (d: PreparedDispatch) =
        let i = d.Kernel.Value

        if i >= 0 && i < dispatchAttrs.Length then
            dispatchAttrs[i]
        else
            [ Attr.str TelemetryKeys.Kernel d.Spec.Name ]

    /// Encodes one command list:
    /// - a dispatch first moves the table (page 0) and the pages of its tokens to UNORDERED_ACCESS, sets the pipeline
    ///   when the kernel changes, sets the 32 root constants and dispatches the physical grid;
    /// - Barrier is a UAV barrier on all resources;
    /// - Copy is a full barrier on both sides: a UAV barrier and the source and destination moved to COPY_SOURCE and
    ///   COPY_DEST, the copy, then both back to UNORDERED_ACCESS. A copy within one page goes through the scratch
    ///   buffer in chunks of at most ScratchBytes, because one buffer cannot be COPY_SOURCE and COPY_DEST at once;
    /// - timers and (with timeDispatches) every dispatch are bracketed by timestamp queries; a timed dispatch also gets
    ///   a UAV barrier before each of its two timestamps, so its interval covers that dispatch alone;
    /// - at the end every touched page returns to COMMON and the queries are resolved.
    let encode (timeDispatches: bool) (commands: Command list) : Encoded =
        let tracker = StateTracker()
        let ops = ResizeArray<NativeOp>()
        let mutable queries = 0
        let mutable pipeline: KernelId option = None
        let opened = Dictionary<int, string * Attr list * int>()
        let timers = SortedDictionary<int, TimerSlot>()
        let dispatchTimers = ResizeArray<TimerSlot>()
        let copies = ResizeArray<int64>()
        let mutable dispatches = 0

        let query () =
            let q = queries
            queries <- queries + 1
            ops.Add(NativeOp.Timestamp q)
            q

        for c in commands do
            match c with
            | Command.Dispatch d ->
                let slots =
                    Heap.TableSlot :: (d.Tokens |> List.map _.Slot) |> List.distinct |> List.sort

                ops.AddRange(barriers [ for s in slots -> tracker.To(Target.Page s, PageState.UnorderedAccess) ])

                if pipeline <> Some d.Kernel then
                    ops.Add(NativeOp.Pipeline d.Kernel)
                    pipeline <- Some d.Kernel

                ops.Add(NativeOp.Constants(d.Args.ToWords()))

                if timeDispatches then
                    // UAV barriers around the timed dispatch, so earlier work has finished when its first timestamp is
                    // taken and it has finished when its second one is: timed dispatches run one at a time.
                    ops.Add(NativeOp.Barriers [ BarrierOp.AllUav ])
                    let q0 = query ()
                    ops.Add(NativeOp.Dispatch d.Physical)
                    ops.Add(NativeOp.Barriers [ BarrierOp.AllUav ])
                    let q1 = query ()

                    dispatchTimers.Add
                        {
                            Name = GpuTelemetry.Dispatch
                            Attrs = attrsOf d
                            Begin = q0
                            End = q1
                        }
                else
                    ops.Add(NativeOp.Dispatch d.Physical)

                dispatches <- dispatches + 1
            | Command.Barrier -> ops.Add(NativeOp.Barriers [ BarrierOp.AllUav ])
            | Command.Copy(src, dst) ->
                copies.Add src.Bytes
                let s = Target.Page src.Slot
                let d = Target.Page dst.Slot

                if src.Slot <> dst.Slot then
                    ops.Add(
                        NativeOp.Barriers(
                            BarrierOp.AllUav
                            :: List.choose id [ tracker.To(s, PageState.CopySource); tracker.To(d, PageState.CopyDest) ]
                        )
                    )

                    ops.Add(NativeOp.Copy(d, dst.Offset, s, src.Offset, src.Bytes))

                    ops.AddRange(
                        barriers
                            [
                                tracker.To(s, PageState.UnorderedAccess)
                                tracker.To(d, PageState.UnorderedAccess)
                            ]
                    )
                else
                    // One page: source -> scratch, then scratch -> destination, chunk by chunk.
                    let mutable first = true
                    let mutable done' = 0L

                    while done' < src.Bytes do
                        let n = min ScratchBytes (src.Bytes - done')

                        let before =
                            List.choose
                                id
                                [
                                    tracker.To(s, PageState.CopySource)
                                    tracker.To(Target.Scratch, PageState.CopyDest)
                                ]

                        ops.Add(NativeOp.Barriers(if first then BarrierOp.AllUav :: before else before))
                        first <- false
                        ops.Add(NativeOp.Copy(Target.Scratch, 0L, s, src.Offset + done', n))

                        ops.AddRange(
                            barriers
                                [
                                    tracker.To(Target.Scratch, PageState.CopySource)
                                    tracker.To(d, PageState.CopyDest)
                                ]
                        )

                        ops.Add(NativeOp.Copy(d, dst.Offset + done', Target.Scratch, 0L, n))
                        done' <- done' + n

                    ops.AddRange(barriers [ tracker.To(d, PageState.UnorderedAccess) ])
            | Command.BeginTimer(id, name, attrs) -> opened[id] <- (name, attrs, query ())
            | Command.EndTimer id ->
                let name, attrs, q0 = opened[id]
                let q1 = query ()

                timers[id] <-
                    {
                        Name = name
                        Attrs = attrs
                        Begin = q0
                        End = q1
                    }

        ops.AddRange(barriers (tracker.Finish() |> List.map Some))

        if queries > 0 then
            ops.Add(NativeOp.ResolveTimestamps queries)

        {
            Ops = List.ofSeq ops
            Queries = queries
            Timers = List.ofSeq timers.Values
            DispatchTimers = List.ofSeq dispatchTimers
            Dispatches = dispatches
            Copies = List.ofSeq copies
        }

    /// An upload chunk: staging buffer `staging` (from byte 0) into `slot` at `offset`.
    let upload (slot: int) (offset: int64) (staging: int) (bytes: int64) : NativeOp list =
        let t = Target.Page slot

        [
            NativeOp.Barriers [ BarrierOp.Transition(t, PageState.Common, PageState.CopyDest) ]
            NativeOp.Copy(t, offset, Target.Upload staging, 0L, bytes)
            NativeOp.Barriers [ BarrierOp.Transition(t, PageState.CopyDest, PageState.Common) ]
        ]

    /// A download chunk: `slot` at `offset` into readback staging buffer `staging` (from byte 0).
    let download (slot: int) (offset: int64) (staging: int) (bytes: int64) : NativeOp list =
        let t = Target.Page slot

        [
            NativeOp.Barriers [ BarrierOp.Transition(t, PageState.Common, PageState.CopySource) ]
            NativeOp.Copy(Target.Readback staging, 0L, t, offset, bytes)
            NativeOp.Barriers [ BarrierOp.Transition(t, PageState.CopySource, PageState.Common) ]
        ]

    /// Checks an op list against the state rules it must keep: every transition starts from the tracked state and
    /// changes it, at most once per target and call; staging buffers never transition; a dispatch finds the table and
    /// the pages of its tokens in UNORDERED_ACCESS (`dispatchSlots`: the token slots of each dispatch, in order); a copy
    /// finds its source in COPY_SOURCE and its destination in COPY_DEST (staging buffers in their fixed states), and
    /// never copies a buffer onto itself; and the list ends with every target in COMMON. Error names the first broken
    /// rule.
    let verify (dispatchSlots: int list list) (ops: NativeOp list) : Result<unit, string> =
        let state = Dictionary<Target, PageState>()

        let get t =
            match state.TryGetValue t with
            | true, s -> s
            | _ -> PageState.Common

        let mutable error: string option = None
        let mutable pending = dispatchSlots

        let fail (s: string) =
            if error.IsNone then
                error <- Some s

        ops
        |> List.iteri (fun i op ->
            match op with
            | NativeOp.Barriers bs ->
                let seen = HashSet<Target>()

                for b in bs do
                    match b with
                    | BarrierOp.AllUav -> ()
                    | BarrierOp.Transition(t, before, after) ->
                        match t with
                        | Target.Upload _
                        | Target.Readback _ -> fail (sprintf "op %d: %O never transitions" i t)
                        | _ -> ()

                        if not (seen.Add t) then
                            fail (sprintf "op %d: %O transitions twice in one call" i t)

                        if before = after then
                            fail (sprintf "op %d: %O transitions from %A to itself" i t before)

                        if get t <> before then
                            fail (sprintf "op %d: %O is %A, not %A" i t (get t) before)

                        state[t] <- after
            | NativeOp.Dispatch _ ->
                match pending with
                | slots :: rest ->
                    pending <- rest

                    for s in Heap.TableSlot :: slots do
                        if get (Target.Page s) <> PageState.UnorderedAccess then
                            fail (sprintf "op %d: dispatch with page %d in %A" i s (get (Target.Page s)))
                | [] -> fail (sprintf "op %d: more dispatches than expected" i)
            | NativeOp.Copy(d, _, s, _, n) ->
                if n <= 0L then
                    fail (sprintf "op %d: empty copy" i)

                match s with
                | Target.Upload _ -> ()
                | Target.Readback _ -> fail (sprintf "op %d: a readback buffer is never a copy source" i)
                | _ ->
                    if get s <> PageState.CopySource then
                        fail (sprintf "op %d: copy source %O is %A" i s (get s))

                match d with
                | Target.Readback _ -> ()
                | Target.Upload _ -> fail (sprintf "op %d: an upload buffer is never a copy destination" i)
                | _ ->
                    if get d <> PageState.CopyDest then
                        fail (sprintf "op %d: copy destination %O is %A" i d (get d))

                if obj.Equals(s, d) then
                    fail (sprintf "op %d: %O is copied onto itself" i s)
            | NativeOp.Pipeline _
            | NativeOp.Constants _
            | NativeOp.Timestamp _
            | NativeOp.ResolveTimestamps _ -> ())

        if not pending.IsEmpty then
            fail (sprintf "%d dispatches expected but not found" pending.Length)

        for kv in state do
            if kv.Value <> PageState.Common then
                fail (sprintf "the list ends with %O in %A" kv.Key kv.Value)

        match error with
        | None -> Ok()
        | Some e -> Error e
