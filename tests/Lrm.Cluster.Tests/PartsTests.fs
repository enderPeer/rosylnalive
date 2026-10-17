/// The parts under the coordinator: blob stores, clocks, the deadline rule, the chaos policy and the fake pipeline.
module Lrm.Cluster.Tests.PartsTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Cluster.Tests.Common
open Lrm.Testing

let private storeContract (name: string) (make: unit -> IBlobStore) =
    testList
        name
        [
            testCase "put returns the lrm.blob/1 id and get returns the bytes"
            <| fun () ->
                let s = make ()
                let id = s.Put "abc"B
                Expect.equal id (BlobId.ofBytes "abc"B) "id"
                Expect.equal (s.TryGet id) (Some "abc"B) "get"
                Expect.isTrue (s.Contains id) "contains"
                Expect.equal (s.TryGet(BlobId.ofBytes "abd"B)) None "absent"

            testCase "put is idempotent and counts each blob once"
            <| fun () ->
                let s = make ()

                for _ in 1..3 do
                    s.Put "abc"B |> ignore

                s.Put [||] |> ignore
                Expect.equal s.Count 2 "count"
                Expect.equal s.TotalBytes 3L "bytes"
        ]

let storeTests =
    let root () =
        Path.Combine(Path.GetTempPath(), "lrm-cluster-tests", Guid.NewGuid().ToString("N"))

    testList
        "blobstore"
        [
            storeContract "memory" (fun () -> MemoryBlobStore() :> IBlobStore)
            storeContract "directory" (fun () -> DirectoryBlobStore(root ()) :> IBlobStore)

            testCase "the directory store detects a corrupted blob"
            <| fun () ->
                let dir = root ()
                let s = DirectoryBlobStore dir :> IBlobStore
                let id = s.Put "hello"B

                let file =
                    Directory.GetFiles(dir, "*.blob", SearchOption.AllDirectories)
                    |> Array.exactlyOne

                Expect.equal (Path.GetFileNameWithoutExtension file) id.Value.Hex "file name is the id"
                File.WriteAllBytes(file, "hellp"B)
                Expect.throwsT<InvalidDataException> (fun () -> s.TryGet id |> ignore) "corruption"

            testCase "the memory store keeps its own copy"
            <| fun () ->
                let s = MemoryBlobStore() :> IBlobStore
                let bytes = "abc"B
                let id = s.Put bytes
                bytes[0] <- byte 'x'
                Expect.equal (s.TryGet id) (Some "abc"B) "copy"
        ]

let deadlineTests =
    let p = DeadlinePolicy.standard

    testList
        "deadline"
        [
            testCase "the contract values"
            <| fun () ->
                Expect.equal p.InitialNs (Ns.seconds 30L) "initial"
                Expect.equal p.MinNs (Ns.ms 250L) "min"
                Expect.equal p.MaxNs (Ns.seconds 30L) "max"
                Expect.equal (p.Factor, p.MarginNs, p.MinSamples) (4L, Ns.ms 50L, 8) "rule"

            testCase "30 s until 8 samples exist"
            <| fun () ->
                let e = DeadlineEstimator p

                for i in 1..7 do
                    e.Add(Ns.ms 10L)
                    Expect.equal e.Current (Ns.seconds 30L) (sprintf "%d samples" i)

                e.Add(Ns.ms 10L)
                Expect.equal e.Current (Ns.ms 250L) "8 samples: clamp(4*10+50, 250, ...) = 250 ms"

            testCase "clamp(4 * p95 + 50 ms, 250 ms, 30 s)"
            <| fun () ->
                let e = DeadlineEstimator p

                for ms in 1L .. 20L do
                    e.Add(Ns.ms (ms * 10L))

                // 20 samples 10..200 ms: nearest rank ceil(0.95 * 20) = 19th = 190 ms; 4 * 190 + 50 = 810 ms.
                Expect.equal e.P95 (Ns.ms 190L) "p95"
                Expect.equal e.Current (Ns.ms 810L) "deadline"

                let slow = DeadlineEstimator p

                for _ in 1..8 do
                    slow.Add(Ns.seconds 10L)

                Expect.equal slow.Current (Ns.seconds 30L) "upper clamp"

            testCase "the window keeps the 64 most recent samples"
            <| fun () ->
                let e = DeadlineEstimator p

                for _ in 1..100 do
                    e.Add(Ns.seconds 5L)

                for _ in 1..64 do
                    e.Add(Ns.ms 1L)

                Expect.equal e.Count 64 "window"
                Expect.equal e.Current (Ns.ms 250L) "old samples forgotten"

            testCase "nearest-rank p95 for small windows"
            <| fun () ->
                for n in 1..30 do
                    let e = DeadlineEstimator p

                    for i in 1..n do
                        e.Add(int64 i)

                    let rank = int (ceil (0.95 * float n))
                    Expect.equal e.P95 (int64 rank) (sprintf "n = %d" n)
        ]

let clockTests =
    testList
        "clock"
        [
            testCase "virtual time moves only forward"
            <| fun () ->
                let c = VirtualClock 5L
                c.Advance 10L
                Expect.equal c.NowNs 15L "advance"
                Expect.throwsT<ArgumentException> (fun () -> c.AdvanceTo 14L) "backwards"
                Expect.equal (c.TelemetryClock()) 15L "telemetry clock"

            testCase "the system clock is monotonic"
            <| fun () ->
                let a = SystemClock.Instance.NowNs
                let b = SystemClock.Instance.NowNs
                Expect.isGreaterThanOrEqual b a "monotonic"
        ]

let chaosTests =
    testList
        "chaos"
        [
            testCase "one seed gives the same fates; another seed differs"
            <| fun () ->
                let fates seed =
                    let p = ChaosPolicy(seed, ChaosConfig.standard, [])
                    [ for i in 0..999 -> p.Decide(uint64 (i % 3), uint64 (i % 5), int64 i) ]

                Expect.equal (fates 1UL) (fates 1UL) "same seed"
                Expect.notEqual (fates 1UL) (fates 2UL) "other seed"

            testCase "each link has its own stream"
            <| fun () ->
                let a = ChaosPolicy(9UL, ChaosConfig.standard, [])
                let b = ChaosPolicy(9UL, ChaosConfig.standard, [])
                // Interleaving another link's frames does not change this link's fates.
                let alone = [ for _ in 1..200 -> a.Decide(1UL, 0UL, 0L) ]

                let mixed =
                    [
                        for _ in 1..200 do
                            b.Decide(2UL, 0UL, 0L) |> ignore
                            yield b.Decide(1UL, 0UL, 0L)
                    ]

                Expect.equal mixed alone "independent links"

            testCase "rates and delays follow the config"
            <| fun () ->
                let p = ChaosPolicy(3UL, ChaosConfig.standard, [])
                let fates = [ for _ in 1..100_000 -> p.Decide(1UL, 0UL, 0L) ]
                let dropped = fates |> List.filter (fun f -> f.Copies.IsEmpty) |> List.length
                let duplicated = fates |> List.filter (fun f -> f.Copies.Length = 2) |> List.length
                Expect.isTrue (dropped > 1500 && dropped < 2500) (sprintf "2%% drops: %d" dropped)
                Expect.isTrue (duplicated > 1500 && duplicated < 2500) (sprintf "2%% duplicates: %d" duplicated)
                let delays = fates |> List.collect _.Copies
                Expect.isGreaterThanOrEqual (List.min delays) (Ns.ms 1L) "min delay"
                Expect.isLessThanOrEqual (List.max delays) (Ns.ms 15L * 2L + Ns.ms 40L) "max delay"

            testCase "partitions drop both directions inside their window"
            <| fun () ->
                let p = ChaosPolicy(1UL, ChaosConfig.none, [ Fault.Partition(4UL, 100L, 200L) ])
                Expect.equal (p.Decide(4UL, 0UL, 99L)).Copies [ 0L ] "before"
                Expect.equal (p.Decide(4UL, 0UL, 100L)).Copies [] "from node"
                Expect.equal (p.Decide(0UL, 4UL, 150L)).Copies [] "to node"
                Expect.equal (p.Decide(4UL, 0UL, 200L)).Copies [ 0L ] "after"
                Expect.equal (p.Decide(5UL, 0UL, 150L)).Copies [ 0L ] "other node"

            testCase "lag windows multiply executor cost"
            <| fun () ->
                let p =
                    ChaosPolicy(1UL, ChaosConfig.none, [ Fault.Lag(2UL, 10L, 20L, 7); Fault.Lag(2UL, 15L, 30L, 3) ])

                Expect.equal (p.LagFactor(2UL, 5L)) 1 "before"
                Expect.equal (p.LagFactor(2UL, 16L)) 7 "overlap takes the max"
                Expect.equal (p.LagFactor(2UL, 25L)) 3 "second window"

            testCase "random fault schedules are a function of the seed"
            <| fun () ->
                let nodes = [ 1UL .. 7UL ]

                let schedule seed =
                    ChaosPolicy.RandomFaults(seed, nodes, Ns.seconds 2L)

                Expect.equal (schedule 11UL) (schedule 11UL) "same seed"

                let all = [ for s in 1UL .. 200UL -> schedule s ]
                Expect.isGreaterThan (all |> List.distinct |> List.length) 150 "seeds differ"

                for faults in all do
                    let kills =
                        faults
                        |> List.choose (function
                            | Fault.Kill(n, _) -> Some n
                            | _ -> None)

                    Expect.isLessThanOrEqual kills.Length 2 "at most two kills"
                    Expect.equal (List.distinct kills) kills "distinct kill targets"

            testCase "invalid configs are refused"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () ->
                        ChaosPolicy(
                            1UL,
                            { ChaosConfig.none with
                                DropPermille = 1001
                            },
                            []
                        )
                        |> ignore)
                    "permille"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        ChaosPolicy(
                            1UL,
                            { ChaosConfig.none with
                                MinDelayNs = 5L
                                MaxDelayNs = 1L
                            },
                            []
                        )
                        |> ignore)
                    "delays"
        ]

let fakeTests =
    testList
        "fake pipeline"
        [
            testCase "small job known answers: job id, run digest, replica digest, tokens"
            <| fun () ->
                let r = FakeReference.run Vectors.smallJob
                Expect.equal (FakeJob.id Vectors.smallJob).Hex Vectors.smallJobId "job id"
                Expect.equal r.Units 18 "2 lanes x 3 steps x 3 stages"
                Expect.equal r.RunDigest.Hex Vectors.smallRun "run digest"
                Expect.equal r.ReplicaDigest.Hex Vectors.smallReplicas "replica digest"
                Expect.equal r.Tokens Vectors.smallTokens "tokens"

            testCase "standard job known answers"
            <| fun () ->
                let r = reference FakeJob.standard
                Expect.equal (FakeJob.id FakeJob.standard).Hex Vectors.standardJobId "job id"
                Expect.equal r.Units 72 "3 lanes x 8 steps x 3 stages"
                Expect.equal r.RunDigest.Hex Vectors.standardRun "run digest"
                Expect.equal r.ReplicaDigest.Hex Vectors.standardReplicas "replica digest"

            testCase "replaying a chain on a fresh state gives the same outputs and digest"
            <| fun () ->
                let job = Vectors.smallJob
                let w = FakeWorkload job :> IWorkload
                let ex = FakeExecutor([ 0u; 1u; 2u ]) :> IStageExecutor
                let first = ex.Open(w.Job, { Lane = 0u; Stage = 1u })
                let again = ex.Open(w.Job, { Lane = 0u; Stage = 1u })
                let other = ex.Open(w.Job, { Lane = 0u; Stage = 1u })

                let u0 =
                    WorkUnit.fwdStage w.Job w.Class 0u 1u 0u UnitId.none (BlobId.ofBytes "x"B) (FakeJob.args job 1)

                let u1 =
                    WorkUnit.fwdStage w.Job w.Class 0u 1u 1u (WorkUnit.id u0) (BlobId.ofBytes "y"B) (FakeJob.args job 1)

                let a0 = first.Apply(u0, "x"B)
                let a1 = first.Apply(u1, "y"B)
                Expect.equal (again.Apply(u0, "x"B)) a0 "step 0"
                Expect.equal (again.Apply(u1, "y"B)) a1 "step 1"
                Expect.equal again.Digest first.Digest "digest"
                // Skipping a step changes the state, so the output differs: the hash chain matters.
                Expect.notEqual (other.Apply(u1, "y"B)) a1 "skipped step"

            testCase "a corrupting executor changes the output but not the digest"
            <| fun () ->
                let job = Vectors.smallJob
                let w = FakeWorkload job :> IWorkload

                let honest =
                    (FakeExecutor([ 0u ]) :> IStageExecutor).Open(w.Job, { Lane = 0u; Stage = 0u })

                let faulty =
                    (FakeExecutor([ 0u ], (fun _ -> true), ignore) :> IStageExecutor)
                        .Open(w.Job, { Lane = 0u; Stage = 0u })

                let u =
                    WorkUnit.fwdStage w.Job w.Class 0u 0u 0u UnitId.none (BlobId.ofBytes "x"B) (FakeJob.args job 0)

                Expect.notEqual (faulty.Apply(u, "x"B)) (honest.Apply(u, "x"B)) "output"
                Expect.equal faulty.Digest honest.Digest "digest"

            testCase "the workload plans every unit exactly once, in chain order"
            <| fun () ->
                let job = Vectors.smallJob
                let r = FakeReference.run job
                Expect.equal r.Chains.Length (job.Lanes * job.Stages) "one chain per lane and stage"
                Expect.equal r.Tokens.Length job.Lanes "lanes"

                for t in r.Tokens do
                    Expect.equal t.Length job.Steps "steps"

            testCase "job validation"
            <| fun () ->
                Expect.isError (FakeJob.validate { FakeJob.standard with Lanes = 0 }) "lanes"
                Expect.isError (FakeJob.validate { FakeJob.standard with Stages = 0 }) "stages"
                Expect.isError (FakeJob.validate { FakeJob.standard with Steps = 0 }) "steps"
        ]

let tests =
    testList "parts" [ storeTests; deadlineTests; clockTests; chaosTests; fakeTests ]
