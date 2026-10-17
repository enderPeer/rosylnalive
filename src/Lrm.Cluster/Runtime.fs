namespace Lrm.Cluster

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

type internal HostEvent =
    | FrameIn of Frame
    | WorkResult of token: uint64 * result: obj
    | StopHost

/// Runs one actor in a real process: a dedicated thread takes frames from the transport, timers from the clock and
/// executor results from the thread pool, one at a time, so the actor's code stays single-threaded. Executor work
/// runs on the thread pool, so a long unit never stops heartbeats.
[<Sealed>]
type ActorHost(actor: IActor, transport: ITransport, clock: IClock) =
    let mailbox = new BlockingCollection<HostEvent>(ConcurrentQueue<HostEvent>())
    let timers = PriorityQueue<uint64, struct (int64 * int64)>()

    let finished =
        TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)

    let mutable order = 0L
    let mutable thread: Thread = null

    let post (ev: HostEvent) =
        try
            mailbox.Add ev
        with :? InvalidOperationException ->
            ()

    let ctx =
        { new IActorContext with
            member _.Now = clock.NowNs

            member _.Send(dest: uint64, frame: Frame) =
                transport.Send(dest, FrameCodec.encode frame)

            member _.SetTimer(due: int64, tag: uint64) =
                order <- order + 1L
                timers.Enqueue(tag, struct (due, order))

            member _.Execute(_: WorkCost, work: unit -> obj, token: uint64) =
                Task.Run(fun () ->
                    let result =
                        try
                            work ()
                        with e ->
                            box e

                    post (WorkResult(token, result)))
                |> ignore
        }

    /// The due time of the earliest timer, if any.
    let nextDue () =
        let mutable tag = 0UL
        let mutable priority = struct (0L, 0L)

        if timers.TryPeek(&tag, &priority) then
            let struct (due, _) = priority
            ValueSome due
        else
            ValueNone

    let loop () =
        try
            actor.Start ctx
            let mutable running = true

            while running do
                let mutable firing = true

                while running && firing do
                    match nextDue () with
                    | ValueSome due when due <= clock.NowNs ->
                        let tag = timers.Dequeue()
                        actor.OnTimer(ctx, tag)
                    | _ -> firing <- false

                let timeoutMs =
                    match nextDue () with
                    | ValueSome due -> int (max 0L (min 1000L ((due - clock.NowNs + 999_999L) / 1_000_000L)))
                    | ValueNone -> 1000

                let mutable ev = Unchecked.defaultof<HostEvent>

                if mailbox.TryTake(&ev, timeoutMs) then
                    match ev with
                    | FrameIn frame -> actor.OnFrame(ctx, frame)
                    | WorkResult(token, result) -> actor.OnWorkDone(ctx, token, result)
                    | StopHost -> running <- false

            finished.TrySetResult() |> ignore
        with e ->
            finished.TrySetException e |> ignore

    /// Completes when the host stops; faults with the actor's exception if it threw.
    member _.Completion: Task = finished.Task

    /// Starts the actor thread and the transport.
    member _.Start() =
        if not (isNull thread) then
            invalidOp "the host is already started"

        thread <- Thread(ThreadStart loop, IsBackground = true, Name = sprintf "lrm-actor-%d" actor.Id)
        thread.Start()
        transport.Start(fun frame -> post (FrameIn frame))

    /// Stops the actor thread (pending events are dropped) and waits up to 5 s for it.
    member _.Stop() =
        post StopHost

        if not (isNull thread) && Thread.CurrentThread <> thread then
            thread.Join(TimeSpan.FromSeconds 5.0) |> ignore

    interface IDisposable with
        member h.Dispose() =
            h.Stop()
            mailbox.CompleteAdding()
