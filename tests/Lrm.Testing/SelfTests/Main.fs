/// Self-tests for Lrm.Testing: tiers, mutexes and tickets, run-twice checks, the fixed FsCheck seed and start-up.
/// They use a private mutex prefix and a temporary data root, so they never touch the real test mutexes.
module Lrm.Testing.SelfTests.Main

open System
open System.Globalization
open System.IO
open System.Threading
open Expecto
open Lrm.Testing

/// Every temporary data root of this run lives under one folder, deleted when the run ends.
let private runRoot =
    Path.Combine(Path.GetTempPath(), sprintf "lrm-selftest-%d-%s" Environment.ProcessId (Guid.NewGuid().ToString "N"))

let private tempRoot () =
    let d = Path.Combine(runRoot, Guid.NewGuid().ToString "N")
    Directory.CreateDirectory d |> ignore
    d

let private options (log: string -> unit) =
    { Locks.defaultOptions "selftest" with
        DataRoot = tempRoot ()
        Prefix = "LrmSelfTest" + Guid.NewGuid().ToString "N" + "."
        RepoRoot = "C:/lrm-selftest/agent"
        WaitLimit = TimeSpan.FromMilliseconds 400.0
        PollInterval = TimeSpan.FromMilliseconds 20.0
        ReportInterval = TimeSpan.FromMilliseconds 100.0
        Log = log
    }

/// Runs `f` on a fresh thread (mutex ownership is per thread) and returns its result.
let private onThread (f: unit -> 'a) : 'a =
    let result = ref Unchecked.defaultof<'a>
    let failure = ref None

    let t =
        Thread(fun () ->
            try
                result.Value <- f ()
            with e ->
                failure.Value <- Some e)

    t.Start()
    t.Join()

    match failure.Value with
    | Some e -> raise (AggregateException(e))
    | None -> result.Value

let private files (o: Locks.LockOptions) pattern =
    Directory.GetFiles(Locks.locksDir o, o.Prefix + pattern)

let private dummy name = testCase name ignore

/// Holds `id` on a background thread until the returned disposable is disposed.
let private holdOnThread (o: Locks.LockOptions) (id: Locks.LockId) : IDisposable =
    let acquired = new ManualResetEventSlim()
    let release = new ManualResetEventSlim()

    let t =
        Thread(fun () ->
            use _ = Locks.acquire o [ id ]
            acquired.Set()
            release.Wait())

    t.Start()
    acquired.Wait()

    { new IDisposable with
        member _.Dispose() =
            release.Set()
            t.Join()
            acquired.Dispose()
            release.Dispose()
    }

let tierTests =
    testList
        "tiers"
        [
            testCase "tier names round-trip and unknown names are rejected"
            <| fun () ->
                for t in Tier.all do
                    Expect.equal (Tier.tryParse (Tier.name t)) (Some t) (Tier.name t)

                Expect.equal (Tier.tryParse " GPU ") (Some Tier.Gpu) "case and spaces are ignored"
                Expect.isNone (Tier.tryParse "fpga") "unknown tier"

            testCase "backend lists parse in mutex order"
            <| fun () ->
                Expect.equal (Backend.parseList "vulkan,d3d12") (Ok [ Backend.D3D12; Backend.Vulkan ]) "sorted"
                Expect.isError (Backend.parseList "d3d12,metal") "unknown backend"

            testCase "the model tier runs CPU cases first; GPU groups take the backend mutex, then LrmModelCpu"
            <| fun () ->
                let tiers =
                    { TestTiers.empty with
                        Model =
                            [
                                Some Backend.Vulkan, [ dummy "v" ]
                                None, [ dummy "c1" ]
                                Some Backend.D3D12, [ dummy "d" ]
                                None, [ dummy "c2" ]
                            ]
                    }

                let groups = Runner.plan Tier.Model None tiers
                Expect.equal (groups |> List.map _.Name) [ "model.cpu"; "model.d3d12"; "model.vulkan" ] "order"
                Expect.equal groups[0].Locks [ Locks.ModelCpu ] "cpu cases hold only LrmModelCpu"
                Expect.equal groups[0].Tests.Length 2 "both cpu lists"
                Expect.equal groups[1].Locks [ Locks.GpuD3D12; Locks.ModelCpu ] "gpu cases"

            testCase "gpu groups merge per backend, honour LRM_BACKENDS and drop empty tiers"
            <| fun () ->
                let tiers =
                    { TestTiers.empty with
                        Gpu =
                            [
                                Backend.Vulkan, [ dummy "a" ]
                                Backend.D3D12, [ dummy "b" ]
                                Backend.Vulkan, [ dummy "c" ]
                            ]
                    }

                let all = Runner.plan Tier.Gpu None tiers
                Expect.equal (all |> List.map _.Name) [ "gpu.d3d12"; "gpu.vulkan" ] "groups"
                Expect.equal all[1].Tests.Length 2 "vulkan lists merged"
                let onlyVulkan = Runner.plan Tier.Gpu (Some [ Backend.Vulkan ]) tiers
                Expect.equal (onlyVulkan |> List.map _.Name) [ "gpu.vulkan" ] "filtered"
                Expect.isEmpty (Runner.plan Tier.Cpu None tiers) "no cpu tests, no group"

            testCase "bench holds every mutex"
            <| fun () ->
                let groups =
                    Runner.plan
                        Tier.Bench
                        None
                        { TestTiers.empty with
                            Bench = [ dummy "x" ]
                        }

                Expect.equal groups[0].Locks [ Locks.GpuD3D12; Locks.GpuVulkan; Locks.ModelCpu ] "all locks"
        ]

let lockTests =
    testList
        "locks"
        [
            testCase "mutex names follow the TestTiers contract"
            <| fun () ->
                let o = Locks.defaultOptions "x"
                Expect.equal (Locks.mutexName o Locks.GpuD3D12) @"Local\LrmGpu.d3d12" "d3d12"
                Expect.equal (Locks.mutexName o Locks.GpuVulkan) @"Local\LrmGpu.vulkan" "vulkan"
                Expect.equal (Locks.mutexName o Locks.ModelCpu) @"Local\LrmModelCpu" "model cpu"

            testCase "locks are taken in canonical order and leave no tickets or holder files"
            <| fun () ->
                let o = options ignore

                onThread (fun () ->
                    use held =
                        Locks.acquire o [ Locks.ModelCpu; Locks.GpuVulkan; Locks.GpuD3D12; Locks.GpuVulkan ]

                    Expect.equal held.Locks [ Locks.GpuD3D12; Locks.GpuVulkan; Locks.ModelCpu ] "order"
                    Expect.equal (files o "*.holder").Length 3 "one holder file per lock"
                    Expect.isEmpty (files o "*.ticket") "tickets are removed once acquired")

                Expect.isEmpty (files o "*.holder") "holder files are removed on release"

            testCase "a waiter gives up at the wait limit, names the holder and removes its ticket"
            <| fun () ->
                let o = options ignore
                use _ = holdOnThread o Locks.GpuVulkan

                let e =
                    Expect.throwsC (fun () -> onThread (fun () -> Locks.acquire o [ Locks.GpuVulkan ] |> ignore)) id

                let inner = e.InnerException
                Expect.isTrue (inner :? TimeoutException) "a TimeoutException"
                Expect.stringContains inner.Message (sprintf "PID %d" Environment.ProcessId) "names the holder"
                Expect.isEmpty (files o "*.ticket") "ticket removed after giving up"

            testCase "the queue position and holder are reported while waiting"
            <| fun () ->
                let lines = Collections.Concurrent.ConcurrentQueue<string>()
                let o = options lines.Enqueue

                do
                    use _ = holdOnThread o Locks.ModelCpu

                    try
                        onThread (fun () -> Locks.acquire o [ Locks.ModelCpu ] |> ignore)
                    with _ ->
                        ()

                let report = lines |> Seq.tryFind (fun l -> l.Contains "position 1 of 1")
                Expect.isSome report "a position line was logged"
                Expect.stringContains report.Value (sprintf "held by PID %d" Environment.ProcessId) "holder pid"

            testCase "an abandoned mutex counts as acquired"
            <| fun () ->
                let lines = Collections.Concurrent.ConcurrentQueue<string>()
                let o = options lines.Enqueue

                onThread (fun () ->
                    // Take the raw mutex and let the thread end without releasing it.
                    let m = new Mutex(false, Locks.mutexName o Locks.GpuD3D12)
                    m.WaitOne() |> ignore)

                onThread (fun () ->
                    use held = Locks.acquire o [ Locks.GpuD3D12 ]
                    Expect.equal held.Locks [ Locks.GpuD3D12 ] "acquired")

                Expect.exists lines (fun l -> l.Contains "counted as acquired") "logged"

            testCase "a queued verify run has priority over other waiters"
            <| fun () ->
                let o = options ignore
                let verifyRoot = "C:/lrm-selftest/verify"
                File.WriteAllText(Locks.flagPath o, verifyRoot + "\n")
                Expect.isFalse (Locks.isPriorityRun o) "an agent tree is not the verify tree"
                Expect.isTrue (Locks.isPriorityRun { o with RepoRoot = verifyRoot + "/" }) "the verify tree is"

                // A live priority ticket (this process) is queued, so the free mutex must not be taken.
                let fake =
                    Path.Combine(
                        Locks.locksDir o,
                        sprintf "%s%s.%d.999.ticket" o.Prefix (Locks.stem Locks.GpuVulkan) Environment.ProcessId
                    )

                File.WriteAllText(
                    fake,
                    sprintf
                        """{"pid":%d,"seq":999,"lock":"x","label":"verify","repo":"%s","priority":true,"since":"2026-01-01T00:00:00+00:00"}"""
                        Environment.ProcessId
                        verifyRoot
                )

                let e =
                    Expect.throwsC (fun () -> onThread (fun () -> Locks.acquire o [ Locks.GpuVulkan ] |> ignore)) id

                Expect.isTrue (e.InnerException :? TimeoutException) "deferred to the verify run"
                File.Delete fake

                onThread (fun () ->
                    use held = Locks.acquire o [ Locks.GpuVulkan ]
                    Expect.equal held.Locks [ Locks.GpuVulkan ] "acquired once the verify run left")

            testCase "tickets of exited processes are dropped from the queue"
            <| fun () ->
                let o = options ignore

                let stale =
                    Path.Combine(Locks.locksDir o, o.Prefix + "LrmModelCpu.2147483646.1.ticket")

                File.WriteAllText(
                    stale,
                    """{"pid":2147483646,"seq":1,"lock":"x","label":"gone","repo":"x","priority":true,"since":"2026-01-01T00:00:00+00:00"}"""
                )

                Expect.isEmpty (Locks.queue o Locks.ModelCpu) "dead pid ignored"
                Expect.isFalse (File.Exists stale) "and its ticket deleted"

            testCase "workers never acquire a mutex"
            <| fun () ->
                let o = options ignore
                Environment.SetEnvironmentVariable("LRM_TEST_WORKER", "1")

                try
                    use held = Locks.acquire o [ Locks.GpuD3D12; Locks.ModelCpu ]
                    Expect.isEmpty held.Locks "no locks in a worker"
                    Expect.isEmpty (files o "*.holder") "no holder files"
                finally
                    Environment.SetEnvironmentVariable("LRM_TEST_WORKER", null)
        ]

let determinismTests =
    testList
        "determinism"
        [
            testCase "run-twice returns identical bytes and the faster time"
            <| fun () ->
                let bytes, time = Determinism.runTwice "same" (fun () -> [| 1uy; 2uy; 3uy |])
                Expect.equal bytes [| 1uy; 2uy; 3uy |] "bytes"
                Expect.isGreaterThanOrEqual time TimeSpan.Zero "time"

            testCase "run-twice rejects differing bytes and points at the first difference"
            <| fun () ->
                let n = ref 0uy

                let e =
                    Expect.throwsC
                        (fun () ->
                            Determinism.runTwice "drift" (fun () ->
                                n.Value <- n.Value + 1uy
                                [| 7uy; n.Value |])
                            |> ignore)
                        id

                Expect.stringContains e.Message "first difference at byte 1" "position"

            testCase "first difference covers length mismatches"
            <| fun () ->
                Expect.equal (Determinism.firstDifference [| 1uy |] [| 1uy |]) None "equal"
                Expect.equal (Determinism.firstDifference [| 1uy |] [| 1uy; 2uy |]) (Some 1) "longer"
                Expect.equal (Determinism.firstDifference [| 0uy |] [| 1uy |]) (Some 0) "first byte"
        ]

let propTests =
    testList
        "fscheck"
        [
            testCase "the seed is fixed and LRM_FSCHECK_SEED parses"
            <| fun () ->
                Expect.equal (Props.parseSeed "42") (Ok(42UL, snd Props.defaultSeed)) "seed only"
                Expect.equal (Props.parseSeed "0x10,0x3") (Ok(16UL, 3UL)) "hex seed and gamma"
                Expect.isError (Props.parseSeed "1,2") "even gamma"
                Expect.isError (Props.parseSeed "x") "garbage"
                let s, g = Props.seed.Force()
                Expect.equal (Props.config.Force().replay) (Some(s, g, None)) "config replays the seed"

            Props.testProperty "reversing a list twice is the identity" (fun (xs: int list) ->
                List.rev (List.rev xs) = xs)
        ]

let processTests =
    testList
        "process"
        [
            testCase "Process.init ran: invariant culture and working NFC"
            <| fun () ->
                Expect.equal CultureInfo.CurrentCulture CultureInfo.InvariantCulture "culture"
                Expect.equal ("e\u0301".Normalize Text.NormalizationForm.FormC) "\u00e9" "NFC"
                Expect.equal (string 1.5) "1.5" "decimal point"
        ]

let tiers: TestTiers =
    { TestTiers.empty with
        // Sequenced: some tests change process-wide environment variables.
        Cpu =
            [
                testSequenced (testList "selftest" [ tierTests; lockTests; determinismTests; propTests; processTests ])
            ]
    }

[<EntryPoint>]
let main argv =
    try
        Runner.main argv tiers
    finally
        if Directory.Exists runRoot then
            Directory.Delete(runRoot, true)
