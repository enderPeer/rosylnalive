/// Entry point of every test exe (TestTiers contract).
///
///     [<EntryPoint>]
///     let main argv = Runner.main argv tiers
///
/// The tier comes from LRM_TEST_TIER (or --tier). A missing or unknown tier fails the run. Each tier is split into
/// groups; a group holds its mutexes while Expecto runs it. LRM_BACKENDS optionally restricts the GPU backends.
/// With --results <dir> the runner writes <exe>.<tier>.summary.json and one JUnit file per group there.
/// Every other argument is passed to Expecto (for example --fail-on-focused-tests, --filter, --debug).
module Lrm.Testing.Runner

open System
open System.IO
open System.Reflection
open System.Text.Json.Nodes
open Expecto


/// One Expecto run under one set of mutexes.
type Group =
    {
        Name: string
        Backend: Backend option
        Locks: Locks.LockId list
        Tests: Test list
    }

/// Splits a tier into groups, in run order. Groups without tests are dropped.
let plan (tier: Tier) (backends: Backend list option) (t: TestTiers) : Group list =
    let selected b =
        backends |> Option.forall (List.contains b)

    let byBackend (groups: (Backend * Test list) list) =
        groups
        |> List.groupBy fst
        |> List.sortBy fst
        |> List.filter (fst >> selected)
        |> List.map (fun (b, xs) -> b, xs |> List.collect snd)

    let groups =
        match tier with
        | Tier.Cpu ->
            [
                {
                    Name = "cpu"
                    Backend = None
                    Locks = []
                    Tests = t.Cpu
                }
            ]
        | Tier.Gpu ->
            byBackend t.Gpu
            |> List.map (fun (b, tests) ->
                {
                    Name = "gpu." + Backend.name b
                    Backend = Some b
                    Locks = [ Locks.forBackend b ]
                    Tests = tests
                })
        | Tier.Model ->
            let onCpu =
                t.Model
                |> List.collect (function
                    | None, xs -> xs
                    | Some _, _ -> [])

            let onGpu =
                t.Model
                |> List.choose (function
                    | Some b, xs -> Some(b, xs)
                    | None, _ -> None)
                |> byBackend

            {
                Name = "model.cpu"
                Backend = None
                Locks = [ Locks.ModelCpu ]
                Tests = onCpu
            }
            :: (onGpu
                |> List.map (fun (b, tests) ->
                    {
                        Name = "model." + Backend.name b
                        Backend = Some b
                        Locks = [ Locks.forBackend b; Locks.ModelCpu ]
                        Tests = tests
                    }))
        | Tier.Bench ->
            [
                {
                    Name = "bench"
                    Backend = None
                    Locks = [ Locks.GpuD3D12; Locks.GpuVulkan; Locks.ModelCpu ]
                    Tests = t.Bench
                }
            ]

    groups |> List.filter (fun g -> not g.Tests.IsEmpty)

type private Options =
    {
        Tier: string option
        Results: string option
        Expecto: string list
    }

let rec private parseArgs (o: Options) =
    function
    | "--tier" :: t :: rest -> parseArgs { o with Tier = Some t } rest
    | "--results" :: d :: rest -> parseArgs { o with Results = Some d } rest
    | a :: rest -> parseArgs { o with Expecto = o.Expecto @ [ a ] } rest
    | [] -> o

type private GroupResult =
    {
        Group: Group
        Summary: Impl.TestRunSummary option
        ExitCode: int
        WaitMs: float
        Error: string option
    }

let private count (s: Impl.TestRunSummary option) (pick: Impl.TestRunSummary -> 'a list) =
    s |> Option.map (pick >> List.length) |> Option.defaultValue 0

let private str (s: string) = JsonValue.Create s :> JsonNode
let private num (n: float) = JsonValue.Create n :> JsonNode
let private int' (n: int) = JsonValue.Create n :> JsonNode

let private arr (xs: JsonNode seq) =
    let a = JsonArray()

    for x in xs do
        a.Add x

    a :> JsonNode

let private writeSummary (path: string) (exe: string) (tier: Tier) (results: GroupResult list) (exitCode: int) =
    let o = JsonObject()
    o["schema"] <- str "lrm.test-summary/1"
    o["exe"] <- str exe
    o["tier"] <- str (Tier.name tier)
    o["fscheckSeed"] <- str (Props.describe (Props.seed.Force()))
    o["exitCode"] <- int' exitCode

    o["groups"] <-
        results
        |> Seq.map (fun r ->
            let g = JsonObject()
            g["name"] <- str r.Group.Name
            g["backend"] <- (r.Group.Backend |> Option.map (Backend.name >> str) |> Option.toObj)
            g["locks"] <- arr (r.Group.Locks |> Seq.map (Locks.stem >> str))
            g["passed"] <- int' (count r.Summary _.passed)
            g["failed"] <- int' (count r.Summary _.failed)
            g["errored"] <- int' (count r.Summary _.errored)
            g["ignored"] <- int' (count r.Summary _.ignored)
            g["durationMs"] <- num (r.Summary |> Option.map _.duration.TotalMilliseconds |> Option.defaultValue 0.0)
            g["lockWaitMs"] <- num r.WaitMs
            g["exitCode"] <- int' r.ExitCode
            g["error"] <- (r.Error |> Option.map str |> Option.toObj)
            g :> JsonNode)
        |> arr

    o["devices"] <-
        Devices.all ()
        |> Seq.map (fun d ->
            let j = JsonObject()
            j["backend"] <- str d.Backend
            j["device"] <- str d.Device
            j["driver"] <- str d.Driver
            j["shaderSetHash"] <- (d.ShaderSetHash |> Option.map str |> Option.toObj)
            j :> JsonNode)
        |> arr

    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath path))
    |> ignore

    File.WriteAllText(path, o.ToJsonString(Text.Json.JsonSerializerOptions(WriteIndented = true)) + "\n")

/// Runs the tier selected by LRM_TEST_TIER (or --tier) and returns the process exit code:
/// 0 passed, 1 test failures (Expecto's code), 2 usage error or missing tier, 3 mutex wait limit reached.
let main (argv: string[]) (tiers: TestTiers) : int =
    Lrm.Core.Process.init ()
    let exe = Assembly.GetEntryAssembly().GetName().Name

    let opts =
        parseArgs
            {
                Tier = Env.tryGet "LRM_TEST_TIER"
                Results = Env.tryGet "LRM_TEST_RESULTS"
                Expecto = []
            }
            (List.ofArray argv)

    let tierResult =
        match opts.Tier with
        | None -> Error "LRM_TEST_TIER is not set (expected cpu, gpu, model or bench); use tools/test.ps1 -Tier <tier>"
        | Some t ->
            match Tier.tryParse t with
            | Some tier -> Ok tier
            | None -> Error(sprintf "unknown test tier '%s' (expected cpu, gpu, model or bench)" t)

    let backendsResult =
        match Env.tryGet "LRM_BACKENDS" with
        | None -> Ok None
        | Some s -> Backend.parseList s |> Result.map Some

    match tierResult, backendsResult with
    | Error e, _
    | _, Error e ->
        eprintfn "%s: %s" exe e
        2
    | Ok tier, Ok backends ->
        let seedText =
            try
                Ok(Props.describe (Props.seed.Force()))
            with e ->
                Error e.Message

        match seedText with
        | Error e ->
            eprintfn "%s: %s" exe e
            2
        | Ok seed ->
            let groups = plan tier backends tiers

            printfn
                "lrm-test: %s tier=%s groups=[%s] fscheck-seed=%s%s"
                exe
                (Tier.name tier)
                (groups |> List.map _.Name |> String.concat ", ")
                seed
                (if Env.isWorker () then " (worker: no mutexes)" else "")

            let results = ResizeArray<GroupResult>()
            let mutable stop = false

            for g in groups do
                if not stop then
                    let lockOptions = Locks.defaultOptions (sprintf "%s %s" exe g.Name)
                    let clock = Diagnostics.Stopwatch.StartNew()

                    let held =
                        try
                            Ok(Locks.acquire lockOptions g.Locks)
                        with :? TimeoutException as e ->
                            Error e.Message

                    let waitMs = clock.Elapsed.TotalMilliseconds

                    match held with
                    | Error e ->
                        eprintfn "%s: %s" exe e

                        results.Add
                            {
                                Group = g
                                Summary = None
                                ExitCode = 3
                                WaitMs = waitMs
                                Error = Some e
                            }

                        stop <- true
                    | Ok held ->
                        use _ = held
                        let captured = ref None

                        let cli =
                            [
                                yield
                                    CLIArguments.Append_Summary_Handler(
                                        SummaryHandler(fun s -> captured.Value <- Some s)
                                    )
                                match opts.Results with
                                | Some dir ->
                                    yield
                                        CLIArguments.JUnit_Summary(
                                            Path.GetFullPath(Path.Combine(dir, sprintf "%s.%s.junit.xml" exe g.Name))
                                        )
                                | None -> ()
                            ]

                        let code =
                            runTestsWithCLIArgs cli (Array.ofList opts.Expecto) (testList g.Name g.Tests)

                        results.Add
                            {
                                Group = g
                                Summary = captured.Value
                                ExitCode = code
                                WaitMs = waitMs
                                Error = None
                            }

            let exitCode =
                results
                |> Seq.map _.ExitCode
                |> Seq.tryFind (fun c -> c <> 0)
                |> Option.defaultValue 0

            if groups.IsEmpty then
                printfn "lrm-test: %s has no %s tests" exe (Tier.name tier)

            match opts.Results with
            | Some dir ->
                writeSummary
                    (Path.Combine(dir, sprintf "%s.%s.summary.json" exe (Tier.name tier)))
                    exe
                    tier
                    (List.ofSeq results)
                    exitCode
            | None -> ()

            exitCode
