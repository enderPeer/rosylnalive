/// Cross-process test mutexes with a visible ticket queue (TestTiers contract).
///
/// - Mutexes are Local\LrmGpu.d3d12, Local\LrmGpu.vulkan and Local\LrmModelCpu, always acquired in that order.
/// - While waiting, a process keeps a ticket file in LRM_DATA/locks and prints its queue position and the holder PID
///   every minute. The holder writes a holder file. The wait limit is 120 minutes.
/// - AbandonedMutexException (the holder died) counts as acquired.
/// - Priority: when LRM_DATA/locks/verify.flag names a repo root, runs from that tree (the integrator's verify
///   worktree) go first; other waiters do not take the mutex while a priority ticket is queued.
/// - Worker processes (LRM_TEST_WORKER=1) never acquire a mutex.
/// Mutex ownership is per thread: acquire and dispose a Held on the same thread.
module Lrm.Testing.Locks

open System
open System.Diagnostics
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open System.Threading

/// The test mutexes. Declaration order is the acquisition order.
type LockId =
    | GpuD3D12
    | GpuVulkan
    | ModelCpu

let stem =
    function
    | GpuD3D12 -> "LrmGpu.d3d12"
    | GpuVulkan -> "LrmGpu.vulkan"
    | ModelCpu -> "LrmModelCpu"

let forBackend =
    function
    | Backend.D3D12 -> GpuD3D12
    | Backend.Vulkan -> GpuVulkan

type LockOptions =
    {
        /// LRM_DATA; tickets and holder files live in its locks/ folder.
        DataRoot: string
        /// Empty for the real locks. Self-tests use a unique prefix so they never touch them.
        Prefix: string
        /// Repo root of this run, compared with verify.flag for priority.
        RepoRoot: string
        /// Shown to other waiters, e.g. "Lrm.Gpu.Vulkan.Tests gpu.vulkan".
        Label: string
        WaitLimit: TimeSpan
        PollInterval: TimeSpan
        ReportInterval: TimeSpan
        Log: string -> unit
    }

/// Defaults from the environment. LRM_LOCK_WAIT_MINUTES overrides the 120 min wait limit.
let defaultOptions (label: string) =
    let waitMinutes =
        match Env.tryGet "LRM_LOCK_WAIT_MINUTES" with
        | Some s ->
            match Double.TryParse(s, Globalization.NumberStyles.Float, Globalization.CultureInfo.InvariantCulture) with
            | true, m when m > 0.0 -> m
            | _ -> failwithf "LRM_LOCK_WAIT_MINUTES must be a positive number, got '%s'" s
        | None -> 120.0

    {
        DataRoot = Env.dataRoot ()
        Prefix = ""
        RepoRoot = Env.repoRoot ()
        Label = label
        WaitLimit = TimeSpan.FromMinutes waitMinutes
        PollInterval = TimeSpan.FromMilliseconds 500.0
        ReportInterval = TimeSpan.FromMinutes 1.0
        Log = fun line -> printfn "[lrm-lock] %s" line
    }

let mutexName (o: LockOptions) id = @"Local\" + o.Prefix + stem id

let locksDir (o: LockOptions) =
    let d = Path.Combine(o.DataRoot, "locks")
    Directory.CreateDirectory d |> ignore
    d

let flagPath (o: LockOptions) =
    Path.Combine(locksDir o, o.Prefix + "verify.flag")

/// True when verify.flag exists and names this run's repo root.
let isPriorityRun (o: LockOptions) =
    let flag = flagPath o

    File.Exists flag
    && (File.ReadAllLines flag
        |> Array.tryFind (fun l -> not (String.IsNullOrWhiteSpace l))
        |> Option.exists (fun target -> Env.samePath (target.Trim()) o.RepoRoot))

type Ticket =
    {
        Path: string
        Pid: int
        Seq: int64
        Priority: bool
        Since: DateTimeOffset
        Label: string
    }

let private pid = Environment.ProcessId
let private seqCounter = ref 0L

let private isAlive (p: int) =
    try
        use proc = Process.GetProcessById p
        not proc.HasExited
    with _ ->
        false

let private writeJson (path: string) (o: JsonObject) =
    let tmp = path + ".tmp" + string pid
    File.WriteAllText(tmp, o.ToJsonString())
    File.Move(tmp, path, true)

let private readTicket (path: string) =
    try
        use doc = JsonDocument.Parse(File.ReadAllText path)
        let r = doc.RootElement

        Some
            {
                Path = path
                Pid = r.GetProperty("pid").GetInt32()
                Seq = r.GetProperty("seq").GetInt64()
                Priority = r.GetProperty("priority").GetBoolean()
                Since =
                    DateTimeOffset.Parse(r.GetProperty("since").GetString(), Globalization.CultureInfo.InvariantCulture)
                Label = r.GetProperty("label").GetString()
            }
    with _ ->
        None

/// Live tickets for one lock in queue order (priority first, then oldest). Tickets of dead processes are removed.
let queue (o: LockOptions) id =
    Directory.GetFiles(locksDir o, sprintf "%s%s.*.ticket" o.Prefix (stem id))
    |> Array.choose (fun f ->
        match readTicket f with
        | Some t when isAlive t.Pid -> Some t
        | Some _ ->
            (try
                File.Delete f
             with _ ->
                 ())

            None
        | None -> None)
    |> Array.sortBy (fun t -> (not t.Priority), t.Since, t.Pid, t.Seq)
    |> List.ofArray

let private holderPath (o: LockOptions) id =
    Path.Combine(locksDir o, sprintf "%s%s.holder" o.Prefix (stem id))

/// "PID 1234 (label since ...)" for the current holder file, if any.
let describeHolder (o: LockOptions) id =
    try
        use doc = JsonDocument.Parse(File.ReadAllText(holderPath o id))
        let r = doc.RootElement
        let p = r.GetProperty("pid").GetInt32()

        sprintf
            "PID %d (%s, since %s)%s"
            p
            (r.GetProperty("label").GetString())
            (r.GetProperty("since").GetString())
            (if isAlive p then "" else ", stale: that process has exited")
    with _ ->
        "unknown (no holder file)"

/// Locks held by this thread, released in reverse order on Dispose.
type Held(items: (LockId * (unit -> unit)) list) =
    let mutable released = false
    member _.Locks = items |> List.map fst

    interface IDisposable with
        member _.Dispose() =
            if not released then
                released <- true

                for (_, release) in List.rev items do
                    release ()

let private acquireOne (o: LockOptions) (id: LockId) : unit -> unit =
    let name = mutexName o id
    let priority = isPriorityRun o
    let seq = Interlocked.Increment &seqCounter.contents

    let ticket =
        Path.Combine(locksDir o, sprintf "%s%s.%d.%d.ticket" o.Prefix (stem id) pid seq)

    let since = DateTimeOffset.UtcNow

    let stamp () =
        let j = JsonObject()
        j["pid"] <- JsonValue.Create pid
        j["seq"] <- JsonValue.Create seq
        j["lock"] <- JsonValue.Create name
        j["label"] <- JsonValue.Create o.Label
        j["repo"] <- JsonValue.Create o.RepoRoot
        j["priority"] <- JsonValue.Create priority
        j["since"] <- JsonValue.Create(since.ToString("o", Globalization.CultureInfo.InvariantCulture))
        j

    let m = new Mutex(false, name)
    writeJson ticket (stamp ())
    let clock = Stopwatch.StartNew()
    let mutable acquired = false
    let mutable nextReport = o.ReportInterval

    try
        try
            while not acquired do
                let blockedByPriority =
                    not priority
                    && queue o id |> List.exists (fun t -> t.Priority && t.Path <> ticket)

                if blockedByPriority then
                    Thread.Sleep o.PollInterval
                else
                    try
                        acquired <- m.WaitOne o.PollInterval
                    with :? AbandonedMutexException ->
                        o.Log(sprintf "%s: the previous holder exited without releasing it; counted as acquired" name)
                        acquired <- true

                if not acquired then
                    if clock.Elapsed >= nextReport then
                        nextReport <- nextReport + o.ReportInterval
                        let q = queue o id

                        let pos =
                            1
                            + (q
                               |> List.tryFindIndex (fun t -> t.Path = ticket)
                               |> Option.defaultValue q.Length)

                        o.Log(
                            sprintf
                                "waiting %.0f min for %s: position %d of %d, held by %s%s"
                                clock.Elapsed.TotalMinutes
                                name
                                pos
                                (max pos q.Length)
                                (describeHolder o id)
                                (if blockedByPriority then
                                     "; a verify run has priority"
                                 else
                                     "")
                        )

                    if clock.Elapsed >= o.WaitLimit then
                        raise (
                            TimeoutException(
                                sprintf
                                    "gave up waiting for %s after %.1f min (limit %.1f min); held by %s"
                                    name
                                    clock.Elapsed.TotalMinutes
                                    o.WaitLimit.TotalMinutes
                                    (describeHolder o id)
                            )
                        )
        with _ ->
            m.Dispose()
            reraise ()
    finally
        try
            File.Delete ticket
        with _ ->
            ()

    let holder = holderPath o id
    writeJson holder (stamp ())

    if clock.Elapsed > TimeSpan.FromSeconds 1.0 then
        o.Log(sprintf "acquired %s after %.1f s" name clock.Elapsed.TotalSeconds)

    fun () ->
        try
            use doc = JsonDocument.Parse(File.ReadAllText holder)
            let r = doc.RootElement

            if r.GetProperty("pid").GetInt32() = pid && r.GetProperty("seq").GetInt64() = seq then
                File.Delete holder
        with _ ->
            ()

        m.ReleaseMutex()
        m.Dispose()

/// Acquires the given locks in canonical order (d3d12, vulkan, model-cpu). Workers acquire nothing.
let acquire (o: LockOptions) (ids: LockId list) : Held =
    if Env.isWorker () || ids.IsEmpty then
        new Held([])
    else
        let taken = ResizeArray<LockId * (unit -> unit)>()

        try
            for id in ids |> List.distinct |> List.sort do
                taken.Add(id, acquireOne o id)

            new Held(List.ofSeq taken)
        with _ ->
            for (_, release) in Seq.rev taken do
                release ()

            reraise ()
