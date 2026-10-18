/// The StagePipeline encodings against the independent vectors (vectors/serving_vectors.py), the split and layer
/// rules, job validation and the command-line job spec.
module Lrm.Serving.Tests.JobTests

open System
open Expecto
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen
open Lrm.Serving
open Lrm.Serving.Tests.Common

/// The values printed by vectors/serving_vectors.py (Python hashlib and struct, from the contract text alone).
module Vectors =
    let m1 = "ee4be75724ddf15bb8355336c14b7b236ba502b565e936b0658cb3efd3bc067a"
    let j1 = "ca0e63adc155bf3494bfd8dbc86c13ee9899db27f1a859e183c1f175dec6218c"
    let j2 = "32826f379e592d5ca236116421c33be5411adbe777a75e087cc678dd0894ef7a"

    let a1 =
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2002000000000000000000000000000000030000000001"

    let a2 =
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2002000000010000000100000003000000010000000101"

    let p1 = "01000000ffffffff"
    let p2 = "070000000000003f"
    let u1 = "2d17dac3e03fdabc60aa9f49ad1d24807ddd3b20e2be951f5f949f213a610a68"
    let d0 = "5abefe293d5d6edd36288bb4d94ba4b14d0bee4a3e129dc7384761a0d1665301"
    let k1 = "49a80723ee6cbe18b3a31a10017d8772ab50fff97a819719e9d3e671f1874c6f"
    let d1 = "d336f89922584e44d681fadecad6bbd426a7ddf2e931d93d5498effc855f5174"
    let l1 = "21bdec2fd4ce03bf97abbbe86e0286995322cb4bdac042c1a16be68aa8f4ebd2"

let private modelBytes = ContentId.OfBytes(Array.init 32 (fun i -> byte (i + 1)))

/// The vectors' job: model 01..20, the fake class, 2 layers of 64 over 256 ids, split [1], lanes [1; 2; 3] and [4],
/// 4 steps, EOS 7 ignored.
let private vectorJob: ServingJob =
    {
        Model = modelBytes
        Layers = 2
        Hidden = 64
        Vocab = 256
        Boundaries = [ 1 ]
        Class = FakeJob.executionClass
        Prompts = [ [| 1; 2; 3 |]; [| 4 |] ]
        Steps = 4
        StopAtEos = false
        Eos = [ 7 ]
    }

let private args0: StageArgs =
    {
        Model = modelBytes
        Layers = 2
        FirstLayer = 0
        LastLayer = 0
        Pos0 = 0
        Rows = 3
        Head = false
        DType = DType.F32
    }

let private f32Bytes (xs: float32 list) =
    xs |> List.map BitConverter.GetBytes |> Array.concat

let private throwsFormat (f: unit -> unit) (label: string) = Expect.throwsT<FormatException> f label

let tests =
    testList
        "job"
        [
            testCase "ModelId sorts the SHA256SUMS lines by name (M1)"
            <| fun () ->
                let sums =
                    [
                        "model.safetensors", ContentId.OfBytes(Array.create 32 0x11uy)
                        "REVISION", ContentId.OfBytes(Array.create 32 0x22uy)
                        "config.json", ContentId.OfBytes(Array.create 32 0x33uy)
                    ]

                Expect.equal (ModelId.ofSums sums).Hex Vectors.m1 "M1"
                Expect.equal (ModelId.ofSums (List.rev sums)).Hex Vectors.m1 "order-free"

            testCase "the job id covers every field (J1, J2)"
            <| fun () ->
                Expect.equal (ServingJob.id vectorJob).Hex Vectors.j1 "J1"
                Expect.equal (ServingJob.id { vectorJob with StopAtEos = true }).Hex Vectors.j2 "J2"

                Expect.notEqual
                    (ServingJob.id { vectorJob with Steps = 5 })
                    (ServingJob.id vectorJob)
                    "steps change the id"

                Expect.notEqual
                    (ServingJob.id
                        { vectorJob with
                            Class = ExecutionClass.forCpu ExecutionBackend.CpuF64
                        })
                    (ServingJob.id vectorJob)
                    "the class changes the id"

            testCase "stage args encode to 54 bytes and decode back (A1, A2)"
            <| fun () ->
                let a2 =
                    { args0 with
                        FirstLayer = 1
                        LastLayer = 1
                        Pos0 = 3
                        Rows = 1
                        Head = true
                    }

                Expect.equal (Convert.ToHexStringLower(StageArgs.encode args0)) Vectors.a1 "A1"
                Expect.equal (Convert.ToHexStringLower(StageArgs.encode a2)) Vectors.a2 "A2"
                Expect.equal (StageArgs.decode (StageArgs.encode args0)) args0 "A1 round trip"
                Expect.equal (StageArgs.decode (StageArgs.encode a2)) a2 "A2 round trip"

            testCase "stage args refuse a wrong head flag, dtype, size or trailing bytes"
            <| fun () ->
                let bytes = StageArgs.encode args0

                throwsFormat
                    (fun () -> StageArgs.decode (StageArgs.encode { args0 with Head = true }) |> ignore)
                    "head on a stage without the last layer"

                throwsFormat
                    (fun () ->
                        StageArgs.decode (
                            StageArgs.encode
                                { args0 with
                                    LastLayer = 1
                                    Head = false
                                }
                        )
                        |> ignore)
                    "no head on the stage with the last layer"

                throwsFormat
                    (fun () -> StageArgs.decode (StageArgs.encode { args0 with DType = DType.BF16 }) |> ignore)
                    "BF16 payloads"

                throwsFormat (fun () -> StageArgs.decode (StageArgs.encode { args0 with Rows = 0 }) |> ignore) "0 rows"
                throwsFormat (fun () -> StageArgs.decode (Array.append bytes [| 0uy |]) |> ignore) "trailing byte"
                throwsFormat (fun () -> StageArgs.decode bytes[.. bytes.Length - 2] |> ignore) "truncated"

            testCase "payloads: ids are i32, a head output is the token then the logits (P1, P2)"
            <| fun () ->
                Expect.equal (Convert.ToHexStringLower(StagePayload.ofIds [| 1; -1 |])) Vectors.p1 "P1"
                let head = StagePayload.ofHead 7u (f32Bytes [ 0.5f ])
                Expect.equal (Convert.ToHexStringLower head) Vectors.p2 "P2"
                Expect.equal (StagePayload.token head) 7u "token"
                Expect.equal (StagePayload.logits head) (f32Bytes [ 0.5f ]) "logits"
                Expect.equal (StagePayload.ids (StagePayload.ofIds [| 5; 6 |]) 2) [| 5; 6 |] "ids round trip"
                throwsFormat (fun () -> StagePayload.ids [| 0uy; 0uy; 0uy |] 1 |> ignore) "3 bytes for one id"
                Expect.equal (StagePayload.headBytes DType.F32 256) 1028 "tiny head output"
                Expect.equal (StagePayload.residualBytes DType.F64 3 64) 1536 "f64 residual"

            testCase "the first unit of a lane (U1)"
            <| fun () ->
                let jobId = ServingJob.id vectorJob
                let stage = (ServingJob.stages vectorJob).Head

                let p =
                    ServingUnit.create vectorJob jobId stage 0 0 UnitId.none 0 3 (StagePayload.ofIds [| 1; 2; 3 |])

                Expect.equal (WorkUnit.id p.Unit).Value.Hex Vectors.u1 "U1"
                Expect.equal p.Unit.Args (StageArgs.encode args0) "args"
                Expect.equal p.Unit.Class FakeJob.classId "class"

            testCase "chain digests: initial, KV rows, after a unit, logits (D0, K1, D1, L1)"
            <| fun () ->
                let jobId = ContentId.parse Vectors.j1
                let d0 = ChainDigest.initial jobId { Lane = 1u; Stage = 1u }
                Expect.equal d0.Hex Vectors.d0 "D0"

                let kv =
                    ChainDigest.kvRows 1 1 2 1 DType.F32 (fun layer isValue ->
                        Expect.equal layer 1 "layer"

                        if isValue then
                            f32Bytes [ 0.25f; 8.0f ]
                        else
                            f32Bytes [ 1.5f; -2.0f ])

                Expect.equal kv.Hex Vectors.k1 "K1"
                let d1 = ChainDigest.after d0 3u (BlobId.ofBytes "abc"B) (BlobId.ofBytes [||]) kv
                Expect.equal d1.Hex Vectors.d1 "D1"

                let l1 =
                    ChainDigest.logits [ [ StagePayload.ofHead 5u (f32Bytes [ 1.0f; 2.0f ]) ]; [] ]

                Expect.equal l1.Hex Vectors.l1 "L1"

            testCase "the split: the midpoint, Stage.split's stages, and the plan's decision table"
            <| fun () ->
                let c = (tinyWeights ()).Config
                Expect.equal (StageLayout.midpoint 2) [ 1 ] "tiny midpoint"
                Expect.equal (StageLayout.midpoint 28) [ 14 ] "Qwen3-0.6B midpoint"
                Expect.equal (StageLayout.midpoint 1) [] "one layer"
                Expect.equal (StageLayout.stages c.Layers [ 1 ]) (Stage.split c [ 1 ]) "tiny = Stage.split"
                Expect.equal (StageLayout.stages c.Layers []) (Stage.split c []) "whole = Stage.split"

                let q = StageLayout.stages 28 [ 14 ]

                Expect.equal
                    (q |> List.map (fun s -> s.Index, s.FirstLayer, s.LastLayer))
                    [ 0, 0, 13; 1, 14, 27 ]
                    "the plan's split"

                Expect.equal (StageLayout.stages 28 [ 7; 14; 21 ]).Length 4 "four stages of 7"

                for bad in [ [ 0 ]; [ 28 ]; [ 14; 14 ]; [ 14; 7 ] ] do
                    Expect.isError (StageLayout.validate 28 bad) (sprintf "%A refused" bad)

            testCase "--layers: whole stages only; a node hosts the stages they cover"
            <| fun () ->
                let q = StageLayout.stages 28 [ 14 ]

                let parse t =
                    StageLayout.parseLayers 28 t |> Result.bind (StageLayout.hosted q)

                let indices t =
                    parse t |> Result.map (List.map _.Index)

                Expect.equal (indices "0-13") (Ok [ 0 ]) "stage 0"
                Expect.equal (indices "14-27") (Ok [ 1 ]) "stage 1"
                Expect.equal (indices "all") (Ok [ 0; 1 ]) "all"
                Expect.equal (indices "0-27") (Ok [ 0; 1 ]) "one range over both"
                Expect.equal (indices "0-13,14-27") (Ok [ 0; 1 ]) "two ranges"
                Expect.isError (parse "0-6") "part of a stage"
                Expect.isError (parse "5-20") "parts of two stages"
                Expect.isError (parse "0-28") "beyond the model"
                Expect.isError (parse "x") "not a range"
                let tiny = StageLayout.stages 2 [ 1 ]

                Expect.equal
                    (StageLayout.parseLayers 2 "1"
                     |> Result.bind (StageLayout.hosted tiny)
                     |> Result.map (List.map _.Index))
                    (Ok [ 1 ])
                    "one layer"

            testCase "jobs are refused for empty lanes, ids outside the vocabulary, no steps or a bad split"
            <| fun () ->
                Expect.isOk (ServingJob.validate vectorJob) "the vector job"
                Expect.isError (ServingJob.validate { vectorJob with Prompts = [] }) "no lanes"
                Expect.isError (ServingJob.validate { vectorJob with Prompts = [ [||] ] }) "an empty prompt"

                Expect.isError
                    (ServingJob.validate
                        { vectorJob with
                            Prompts = [ [| 256 |] ]
                        })
                    "id 256"

                Expect.isError
                    (ServingJob.validate
                        { vectorJob with
                            Prompts = [ [| -1 |] ]
                        })
                    "id -1"

                Expect.isError (ServingJob.validate { vectorJob with Steps = 0 }) "no steps"
                Expect.isError (ServingJob.validate { vectorJob with Boundaries = [ 2 ] }) "split at the last layer"
                Expect.equal (ServingJob.maxPositions vectorJob) 6 "3 + 4 - 1 positions"
                Expect.equal (ServingJob.maxRows vectorJob) 3 "the longest prompt"

            testCase "the lane rule: steps, EOS and NaN"
            <| fun () ->
                let out (t: uint32) =
                    StagePayload.ofHead t (Array.zeroCreate (256 * 4))

                Expect.equal (LaneRule.after vectorJob 0 (out 7u)) LaneEnd.Running "EOS ignored"
                Expect.equal (LaneRule.after vectorJob 3 (out 9u)) LaneEnd.Done "the last step"

                Expect.equal (LaneRule.after { vectorJob with StopAtEos = true } 0 (out 7u)) LaneEnd.Eos "EOS stops"

                Expect.equal (LaneRule.after vectorJob 0 (out UInt32.MaxValue)) LaneEnd.NaN "NaN stops"
                Expect.equal (LaneRule.after vectorJob 0 [| 1uy |]) LaneEnd.Invalid "a short output stops"

            testCase "the job spec survives the trip to another process's command line"
            <| fun () ->
                let dir = tinyDir ()

                let spec =
                    { tinySpec with
                        Boundaries = Some [ 1 ]
                    }

                let args = JobSpec.toArgs dir [ 1 ] spec

                let parsed =
                    CliArgs.parse JobSpec.values JobSpec.switches args |> Result.bind JobSpec.parse

                let expected =
                    { spec with
                        Model = dir
                        Prompts =
                            spec.Prompts
                            |> List.map (function
                                | PromptSpec.IdsFrom f -> PromptSpec.IdsFrom(IO.Path.GetFullPath f)
                                | p -> p)
                    }

                Expect.equal parsed (Ok expected) "round trip (paths made absolute)"

                let built = parsed |> Result.bind (JobSpec.build (tinyWeights ()))
                Expect.equal (built |> Result.map ServingJob.id) (Ok(ServingJob.id (jobOf tinySpec))) "same job"

                let eos =
                    CliArgs.parse
                        JobSpec.values
                        JobSpec.switches
                        (JobSpec.toArgs dir [ 1 ] { spec with StopAtEos = true })
                    |> Result.bind JobSpec.parse

                Expect.equal (eos |> Result.map _.StopAtEos) (Ok true) "--stop-at-eos"

            testCase "the tiny prompt set is p0, p1 and p2 of the golden set"
            <| fun () ->
                let job = jobOf tinySpec
                Expect.equal (job.Prompts |> List.map _.Length) [ 13; 1; 70 ] "prompt lengths"
                Expect.equal job.Boundaries [ 1 ] "midpoint"
                Expect.equal job.Eos [ 7 ] "the tiny config's EOS"
                Expect.equal job.Class (ExecutionClass.forCpu ExecutionBackend.CpuF32) "cpu-f32"
                Expect.equal job.Model (ModelId.ofWeights (tinyWeights ())) "model id"
        ]
