namespace Lrm.Serving

open System
open System.Buffers.Binary
open System.Collections.Generic
open System.Text
open Lrm.Core
open Lrm.Cluster
open Lrm.Models.Qwen

/// The domain tags and DetRng streams of the StagePipeline contract (docs/contracts/stage-pipeline.md, "Tags").
[<RequireQualifiedAccess>]
module ServingTags =
    /// ModelId: the verified SHA256SUMS lines of a model folder.
    [<Literal>]
    let Model = "lrm.serving.model/1"

    /// The job id.
    [<Literal>]
    let Job = "lrm.serving.job/1"

    /// The digest of an empty chain.
    [<Literal>]
    let Init = "lrm.serving.init/1"

    /// The digest after a unit.
    [<Literal>]
    let State = "lrm.serving.state/1"

    /// The digest of the KV-cache rows one unit wrote.
    [<Literal>]
    let KvRows = "lrm.serving.kvrows/1"

    /// The digest of every head output of a run (logits and greedy ids).
    [<Literal>]
    let Logits = "lrm.serving.logits/1"

    /// The DetRng stream of the harness's fault schedules.
    [<Literal>]
    let Faults = "lrm.serving.chaos.faults"

/// The identity of a model folder: Canonical.hash "lrm.serving.model/1" over its verified SHA256SUMS lines (name and
/// SHA-256), sorted by name. It names the content, never the path, so every host of one checkpoint agrees on it.
[<RequireQualifiedAccess>]
module ModelId =
    /// The id over (file name, SHA-256) pairs.
    let ofSums (sums: (string * ContentId) list) : ContentId =
        let sorted =
            sums |> List.sortWith (fun (a, _) (b, _) -> String.CompareOrdinal(a, b))

        Canonical.hash ServingTags.Model (fun w ->
            w.Seq(
                sorted,
                fun (name, sha) ->
                    w.String name
                    w.Id sha
            ))

    /// The id of verified weights.
    let ofWeights (weights: Qwen3Weights) = ofSums weights.Sums

/// The split of a model into pipeline stages (StagePipeline, "Stages"): boundaries are the first layers of every stage
/// but the first, the rule of Stage.split. The plan's split of Qwen3-0.6B is [14]; the tiny model's is [1].
[<RequireQualifiedAccess>]
module StageLayout =
    /// The midpoint split: [layers / 2] for two or more layers, no boundary for one.
    let midpoint (layers: int) =
        if layers >= 2 then [ layers / 2 ] else []

    /// The boundaries when they increase strictly inside 1 .. layers - 1.
    let validate (layers: int) (boundaries: int list) : Result<int list, string> =
        if layers < 1 then
            Error(sprintf "%d layers" layers)
        else
            let starts = 0 :: boundaries

            if List.pairwise starts |> List.exists (fun (a, b) -> b <= a) then
                Error(sprintf "boundaries %A do not increase strictly from layer 1" boundaries)
            elif boundaries |> List.exists (fun b -> b >= layers) then
                Error(sprintf "boundaries %A are not all below %d layers" boundaries layers)
            else
                Ok boundaries

    /// The stages of the split, in order; raises ArgumentException for invalid boundaries.
    let stages (layers: int) (boundaries: int list) : Stage list =
        match validate layers boundaries with
        | Error e -> invalidArg "boundaries" e
        | Ok bs ->
            let starts = 0 :: bs
            let ends = (bs |> List.map (fun b -> b - 1)) @ [ layers - 1 ]

            List.zip starts ends
            |> List.mapi (fun i (a, b) ->
                {
                    Index = i
                    FirstLayer = a
                    LastLayer = b
                })

    /// Parses "14" or "7,14,21" (empty or "none" for a single stage).
    let parseBoundaries (text: string) : Result<int list, string> =
        match text.Trim() with
        | ""
        | "none" -> Ok []
        | t ->
            let parts = t.Split(',', StringSplitOptions.RemoveEmptyEntries)
            let parsed = parts |> Array.map (fun p -> Invariant.tryParseI32 (p.Trim()))

            if parsed |> Array.forall Option.isSome then
                Ok [ for p in parsed -> p.Value ]
            else
                Error(sprintf "bad split '%s' (expected layer numbers such as 14 or 7,14,21)" text)

    /// Parses layer ranges "0-13", "14-27", "0-13,14-27" or one layer "5"; "all" is every layer of the model.
    let parseLayers (layers: int) (text: string) : Result<(int * int) list, string> =
        match text.Trim() with
        | "all" -> Ok [ 0, layers - 1 ]
        | t ->
            let range (p: string) =
                match p.Split('-') with
                | [| a |] -> Invariant.tryParseI32 (a.Trim()) |> Option.map (fun a -> a, a)
                | [| a; b |] ->
                    match Invariant.tryParseI32 (a.Trim()), Invariant.tryParseI32 (b.Trim()) with
                    | Some a, Some b -> Some(a, b)
                    | _ -> None
                | _ -> None

            let parsed = t.Split(',', StringSplitOptions.RemoveEmptyEntries) |> Array.map range

            if parsed.Length = 0 || parsed |> Array.exists Option.isNone then
                Error(sprintf "bad --layers '%s' (expected ranges such as 0-13 or 14-27, or all)" text)
            else
                let rs = [ for r in parsed -> r.Value ]

                match rs |> List.tryFind (fun (a, b) -> a < 0 || b < a || b >= layers) with
                | Some(a, b) -> Error(sprintf "layers %d-%d are not inside the model's 0-%d" a b (layers - 1))
                | None -> Ok rs

    /// The stages a node hosts: every stage of the split whose layers lie inside one of the ranges. Every range must
    /// be a union of whole stages, so a node never claims part of a stage.
    let hosted (stages: Stage list) (ranges: (int * int) list) : Result<Stage list, string> =
        let inside (s: Stage) =
            ranges |> List.exists (fun (a, b) -> a <= s.FirstLayer && s.LastLayer <= b)

        let starts = stages |> List.map _.FirstLayer |> Set.ofList
        let ends = stages |> List.map _.LastLayer |> Set.ofList

        match
            ranges
            |> List.tryFind (fun (a, b) -> not (starts.Contains a && ends.Contains b))
        with
        | Some(a, b) ->
            Error(
                sprintf
                    "layers %d-%d are not whole stages of the split (stages: %s)"
                    a
                    b
                    (stages
                     |> List.map (fun s -> Invariant.format $"{s.FirstLayer}-{s.LastLayer}")
                     |> String.concat ", ")
            )
        | None -> Ok(stages |> List.filter inside)

/// A request the pipeline serves (StagePipeline, "The job"): greedy decoding of one prompt per lane on the canonical
/// schedule (a prefill, then single-token decode steps), split into stages, in one execution class.
type ServingJob =
    {
        /// ModelId of the weights.
        Model: ContentId
        /// The model's dimensions the units and payloads need.
        Layers: int
        Hidden: int
        Vocab: int
        /// The split (StageLayout).
        Boundaries: int list
        /// The class every unit runs in. Its backend fixes the payload dtype: F64 for cpu-f64, F32 otherwise.
        Class: ExecutionClass
        /// One prompt per lane, each at least one id.
        Prompts: int[] list
        /// Greedy ids per lane, at least 1: the prefill gives the first, each decode step one more.
        Steps: int
        /// Stop a lane after the first id in Eos (kept as its last id). Parity runs pass false.
        StopAtEos: bool
        /// The model's EOS ids (from its config).
        Eos: int list
    }

[<RequireQualifiedAccess>]
module ServingJob =
    /// The dtype of activations, caches and payloads in a class: F64 for cpu-f64, F32 for every other class.
    let dtypeOf (cls: ExecutionClass) =
        match cls.Backend with
        | ExecutionBackend.CpuF64 -> DType.F64
        | _ -> DType.F32

    /// The job's payload dtype.
    let dtype (job: ServingJob) = dtypeOf job.Class

    /// The stages of the job's split.
    let stages (job: ServingJob) =
        StageLayout.stages job.Layers job.Boundaries

    /// The positions lane `lane` uses: its prompt and every generated id but the last.
    let positions (job: ServingJob) (lane: int) =
        job.Prompts[lane].Length + job.Steps - 1

    /// The most positions any lane uses: the cache size every node must hold.
    let maxPositions (job: ServingJob) =
        [ 0 .. job.Prompts.Length - 1 ] |> List.map (positions job) |> List.max

    /// The longest prompt: the most rows of one forward.
    let maxRows (job: ServingJob) =
        job.Prompts |> List.map _.Length |> List.max

    /// The job when it is well formed: a valid split, 1 to 4096 lanes of 1 or more ids inside the vocabulary, and at
    /// least one step.
    let validate (job: ServingJob) : Result<ServingJob, string> =
        let badId =
            job.Prompts
            |> List.indexed
            |> List.tryPick (fun (lane, p) ->
                p
                |> Array.tryFind (fun id -> id < 0 || id >= job.Vocab)
                |> Option.map (fun id -> lane, id))

        match StageLayout.validate job.Layers job.Boundaries with
        | Error e -> Error e
        | Ok _ ->
            if job.Hidden < 1 || job.Vocab < 1 then
                Error "hidden and vocab must be at least 1"
            elif job.Prompts.IsEmpty || job.Prompts.Length > 4096 then
                Error "a job has 1 to 4096 lanes"
            elif job.Prompts |> List.exists (fun p -> isNull p || p.Length = 0) then
                Error "every lane needs a prompt of at least one id"
            elif job.Steps < 1 then
                Error "a job needs at least one step"
            else
                match badId with
                | Some(lane, id) -> Error(sprintf "lane %d: id %d is outside the vocabulary of %d" lane id job.Vocab)
                | None -> Ok job

    let private check (job: ServingJob) =
        match validate job with
        | Ok j -> j
        | Error e -> invalidArg "job" e

    /// A job over verified weights, split at `boundaries`.
    let create
        (weights: Qwen3Weights)
        (cls: ExecutionClass)
        (boundaries: int list)
        (prompts: int[] list)
        (steps: int)
        (stopAtEos: bool)
        =
        let c = weights.Config

        check
            {
                Model = ModelId.ofWeights weights
                Layers = c.Layers
                Hidden = c.Hidden
                Vocab = c.Vocab
                Boundaries = boundaries
                Class = cls
                Prompts = prompts |> List.map Array.copy
                Steps = steps
                StopAtEos = stopAtEos
                Eos = c.EosTokenIds
            }

    /// Canonical.hash "lrm.serving.job/1": model id, class id (32 raw bytes each); layers, hidden, vocab (u32); the
    /// boundaries (seq of u32); the prompts (seq of seq of i32); steps (u32); stop at EOS (bool); the EOS ids (seq of
    /// i32).
    let id (job: ServingJob) : ContentId =
        let job = check job

        Canonical.hash ServingTags.Job (fun w ->
            w.Id job.Model
            w.Id (ExecutionClass.id job.Class).Value
            w.U32(uint32 job.Layers)
            w.U32(uint32 job.Hidden)
            w.U32(uint32 job.Vocab)
            w.Seq(job.Boundaries, (fun b -> w.U32(uint32 b)))
            w.Seq(job.Prompts, (fun p -> w.Seq(p, (fun id -> w.I32 id))))
            w.U32(uint32 job.Steps)
            w.Bool job.StopAtEos
            w.Seq(job.Eos, (fun id -> w.I32 id)))

/// The Args of a Qwen fwd.stage v1 unit (StagePipeline, "Units"): which stage of which model runs which rows.
type StageArgs =
    {
        Model: ContentId
        /// The model's layer count.
        Layers: int
        FirstLayer: int
        /// Inclusive.
        LastLayer: int
        /// The position of the unit's first row.
        Pos0: int
        /// The rows (T) of the unit.
        Rows: int
        /// True on the stage that ends at the last layer: the unit outputs a token and the last row's logits.
        Head: bool
        /// The payload dtype (F32, or F64 for cpu-f64).
        DType: DType
    }

[<RequireQualifiedAccess>]
module StageArgs =
    /// The canonical bytes: model id (32 raw bytes); layers, first layer, last layer, pos0, rows (u32); head (bool);
    /// the dtype code (u8). 55 bytes.
    let encode (a: StageArgs) : byte[] =
        Canonical.encode (fun w ->
            w.Id a.Model
            w.U32(uint32 a.Layers)
            w.U32(uint32 a.FirstLayer)
            w.U32(uint32 a.LastLayer)
            w.U32(uint32 a.Pos0)
            w.U32(uint32 a.Rows)
            w.Bool a.Head
            w.U8(DType.code a.DType))

    /// Reads exactly the bytes encode writes and checks them (FormatException otherwise).
    let decode (bytes: byte[]) : StageArgs =
        let r = CanonicalReader(ReadOnlyMemory bytes)
        let model = r.Id()
        let layers = r.U32()
        let first = r.U32()
        let last = r.U32()
        let pos0 = r.U32()
        let rows = r.U32()
        let head = r.Bool()
        let code = r.U8()
        r.ExpectEnd()

        let dtype =
            match DType.tryOfCode (int code) with
            | Some(DType.F32 as d)
            | Some(DType.F64 as d) -> d
            | _ -> raise (FormatException(sprintf "stage args: dtype code %d is not F32 or F64" code))

        if layers < 1u || layers > 4096u || last < first || last >= layers then
            raise (FormatException(sprintf "stage args: layers %d .. %d of %d" first last layers))

        if rows < 1u || rows > uint32 Int32.MaxValue || pos0 > uint32 Int32.MaxValue then
            raise (FormatException(sprintf "stage args: %d rows at position %d" rows pos0))

        if head <> (last = layers - 1u) then
            raise (FormatException "stage args: the head flag must be set exactly on the stage with the last layer")

        {
            Model = model
            Layers = int layers
            FirstLayer = int first
            LastLayer = int last
            Pos0 = int pos0
            Rows = int rows
            Head = head
            DType = dtype
        }

/// The bytes units carry (StagePipeline, "Payloads"). Every value is little-endian.
[<RequireQualifiedAccess>]
module StagePayload =
    /// The input of the first stage: the ids, 4 bytes each (i32).
    let ofIds (ids: int[]) : byte[] =
        let b = Array.zeroCreate<byte>(4 * ids.Length)

        for i in 0 .. ids.Length - 1 do
            BinaryPrimitives.WriteInt32LittleEndian(Span(b, 4 * i, 4), ids[i])

        b

    /// The ids of a first-stage input of `rows` rows (FormatException for another length).
    let ids (bytes: byte[]) (rows: int) : int[] =
        if bytes.Length <> 4 * rows then
            raise (FormatException(sprintf "%d input bytes for %d ids" bytes.Length rows))

        Array.init rows (fun i -> BinaryPrimitives.ReadInt32LittleEndian(ReadOnlySpan(bytes, 4 * i, 4)))

    /// The size of a residual of `rows` rows: rows * hidden elements of the dtype.
    let residualBytes (dtype: DType) (rows: int) (hidden: int) = rows * hidden * DType.size dtype

    /// The size of a head output: the token (u32), then the last row's logits (vocab elements of the dtype).
    let headBytes (dtype: DType) (vocab: int) = 4 + vocab * DType.size dtype

    /// A head output: the token (the argmax of the last row, 0xFFFFFFFF for a row that holds a NaN), then the logits.
    let ofHead (token: uint32) (logits: byte[]) : byte[] =
        let b = Array.zeroCreate<byte>(4 + logits.Length)
        BinaryPrimitives.WriteUInt32LittleEndian(Span(b, 0, 4), token)
        Array.blit logits 0 b 4 logits.Length
        b

    /// The token of a head output.
    let token (output: byte[]) : uint32 =
        if output.Length < 4 then
            raise (FormatException(sprintf "a head output of %d bytes" output.Length))

        BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(output, 0, 4))

    /// The logits bytes of a head output.
    let logits (output: byte[]) : byte[] =
        if output.Length < 4 then
            raise (FormatException(sprintf "a head output of %d bytes" output.Length))

        Array.sub output 4 (output.Length - 4)

/// The state digests of a Qwen chain (StagePipeline, "Chain digests"). A digest chains over the previous digest, the
/// unit's input and output ids and the digest of the KV rows the unit wrote, so it changes whenever an output or a
/// cache row changes and stays changed: a chain's digest commits to every output on it and to every valid cache row.
[<RequireQualifiedAccess>]
module ChainDigest =
    /// Canonical.hash "lrm.serving.init/1" (job id, lane u32, stage u32).
    let initial (job: ContentId) (chain: ChainKey) =
        Canonical.hash ServingTags.Init (fun w ->
            w.Id job
            w.U32 chain.Lane
            w.U32 chain.Stage)

    /// Canonical.hash "lrm.serving.state/1" (digest, step u32, input blob id, output blob id, KV rows digest).
    let after (digest: ContentId) (step: uint32) (input: BlobId) (output: BlobId) (kvRows: ContentId) =
        Canonical.hash ServingTags.State (fun w ->
            w.Id digest
            w.U32 step
            w.Id input.Value
            w.Id output.Value
            w.Id kvRows)

    /// SHA-256("lrm.serving.kvrows/1" || 0x00 || payload), where the payload is first layer, layer count, pos0, rows
    /// and the dtype code (u32 each), then for each layer in order its K rows pos0 .. pos0 + rows - 1 and then its V
    /// rows, as their little-endian element bytes. `rowsOf layer isValue` returns those bytes.
    let kvRows
        (firstLayer: int)
        (layerCount: int)
        (pos0: int)
        (rows: int)
        (dtype: DType)
        (rowsOf: int -> bool -> byte[])
        : ContentId =
        use h = new Hash.Incremental()
        h.Append(Encoding.ASCII.GetBytes(ServingTags.KvRows + "\000"))

        let u32 (v: int) =
            let b = Array.zeroCreate<byte> 4
            BinaryPrimitives.WriteUInt32LittleEndian(Span b, uint32 v)
            h.Append b

        u32 firstLayer
        u32 layerCount
        u32 pos0
        u32 rows
        u32 (int (DType.code dtype))

        for l in firstLayer .. firstLayer + layerCount - 1 do
            h.Append(rowsOf l false)
            h.Append(rowsOf l true)

        h.Finish()

    /// Canonical.hash "lrm.serving.logits/1": a sequence over the lanes in order, each its lane (u32) and the sequence
    /// of its head outputs (bytes), step by step. Equal digests mean equal logits and equal greedy ids.
    let logits (outputs: byte[] list list) : ContentId =
        Canonical.hash ServingTags.Logits (fun w ->
            w.Seq(
                List.indexed outputs,
                fun (lane, xs) ->
                    w.U32(uint32 lane)
                    w.Seq(xs, (fun (o: byte[]) -> w.Bytes o))
            ))

/// Planning one Qwen unit (StagePipeline, "Units").
[<RequireQualifiedAccess>]
module ServingUnit =
    /// The fwd.stage v1 unit of `job` (id `jobId`) at (lane, stage, step), rows pos0 .. pos0 + rows - 1, after `prev`.
    let create
        (job: ServingJob)
        (jobId: ContentId)
        (stage: Stage)
        (lane: int)
        (step: int)
        (prev: UnitId)
        (pos0: int)
        (rows: int)
        (input: byte[])
        : PlannedUnit =
        let args =
            StageArgs.encode
                {
                    Model = job.Model
                    Layers = job.Layers
                    FirstLayer = stage.FirstLayer
                    LastLayer = stage.LastLayer
                    Pos0 = pos0
                    Rows = rows
                    Head = (stage.LastLayer = job.Layers - 1)
                    DType = ServingJob.dtype job
                }

        {
            Unit =
                WorkUnit.fwdStage
                    jobId
                    (ExecutionClass.id job.Class)
                    (uint32 lane)
                    (uint32 stage.Index)
                    (uint32 step)
                    prev
                    (BlobId.ofBytes input)
                    args
            Input = input
        }

/// How a lane of a job ended.
[<RequireQualifiedAccess>]
type LaneEnd =
    /// Still decoding.
    | Running
    /// It produced Steps ids.
    | Done
    /// StopAtEos stopped it at an EOS id.
    | Eos
    /// A step's logits held a NaN (the token 0xFFFFFFFF); the lane stopped there.
    | NaN
    /// A head output had the wrong size; the lane stopped there.
    | Invalid

[<RequireQualifiedAccess>]
module LaneEnd =
    /// The name the outcome lines and reports print.
    let name (e: LaneEnd) =
        match e with
        | LaneEnd.Running -> "running"
        | LaneEnd.Done -> "done"
        | LaneEnd.Eos -> "eos"
        | LaneEnd.NaN -> "nan"
        | LaneEnd.Invalid -> "invalid"

/// What the head stage's outputs of one lane mean for its schedule. Shared by the workload and the single-process
/// reference, so both stop a lane by the same rule.
[<RequireQualifiedAccess>]
module LaneRule =
    /// The end of a lane after head output `output` at `step`, or Running when it decodes on.
    let after (job: ServingJob) (step: int) (output: byte[]) : LaneEnd =
        if output.Length <> StagePayload.headBytes (ServingJob.dtype job) job.Vocab then
            LaneEnd.Invalid
        else
            let token = StagePayload.token output

            if token = UInt32.MaxValue then
                LaneEnd.NaN
            elif job.StopAtEos && List.contains (int token) job.Eos then
                LaneEnd.Eos
            elif step + 1 >= job.Steps then
                LaneEnd.Done
            else
                LaneEnd.Running

/// The tokens and head outputs of a finished or running job, lane by lane: what a run is compared on.
type ServingResult =
    {
        /// The greedy ids per lane.
        Tokens: int list list
        /// The head outputs per lane (token, then logits), step by step.
        HeadOutputs: byte[] list list
        Ends: LaneEnd list
        /// ChainDigest.logits over HeadOutputs.
        LogitsDigest: ContentId
    }

[<RequireQualifiedAccess>]
module ServingResult =
    /// The result over per-lane head outputs and ends.
    let ofOutputs (outputs: byte[] list list) (ends: LaneEnd list) =
        {
            Tokens =
                outputs
                |> List.map (fun xs ->
                    xs
                    |> List.choose (fun o ->
                        if o.Length < 4 then
                            None
                        else
                            match StagePayload.token o with
                            | t when t = UInt32.MaxValue -> None
                            | t -> Some(int t)))
            HeadOutputs = outputs
            Ends = ends
            LogitsDigest = ChainDigest.logits outputs
        }

/// The coordinator's plan of a Qwen job (IWorkload): stage 0 of every lane starts with its prompt; each stage's output
/// is the next stage's input at the same step; the head's token is stage 0's input at the next step, at position
/// prompt + step. Deterministic: the units it plans depend only on accepted outputs.
[<Sealed>]
type QwenWorkload(job: ServingJob) =
    let jobId = ServingJob.id job
    let stages = ServingJob.stages job |> Array.ofList
    let lanes = job.Prompts.Length
    let prompts = job.Prompts |> Array.ofList
    let lastPlanned = Dictionary<ChainKey, UnitId>()
    let outputs = Array.init lanes (fun _ -> ResizeArray<byte[]>())
    let ends = Array.create lanes LaneEnd.Running
    let mutable lanesLeft = lanes

    let plan (lane: int) (stage: int) (step: int) (pos0: int) (rows: int) (input: byte[]) =
        let chain =
            {
                Lane = uint32 lane
                Stage = uint32 stage
            }

        let prev =
            match lastPlanned.TryGetValue chain with
            | true, p -> p
            | _ -> UnitId.none

        let p = ServingUnit.create job jobId stages[stage] lane step prev pos0 rows input
        lastPlanned[chain] <- WorkUnit.id p.Unit
        p

    /// The job.
    member _.Job = job

    /// The job id.
    member _.JobId = jobId

    /// The tokens, head outputs and lane ends accepted so far (from the coordinator's thread, or after the run).
    member _.Result =
        ServingResult.ofOutputs [ for xs in outputs -> List.ofSeq xs ] (List.ofArray ends)

    interface IWorkload with
        member _.Job = jobId
        member _.Class = ExecutionClass.id job.Class
        member _.Stages = uint32 stages.Length

        member _.Start() =
            [
                for lane in 0 .. lanes - 1 -> plan lane 0 0 0 prompts[lane].Length (StagePayload.ofIds prompts[lane])
            ]

        member _.Accept(work: WorkUnit, output: byte[]) =
            let lane = int work.Lane
            let stage = int work.Stage
            let step = int work.Step

            if stage < stages.Length - 1 then
                let a = StageArgs.decode work.Args
                [ plan lane (stage + 1) step a.Pos0 a.Rows output ]
            else
                outputs[lane].Add output

                match LaneRule.after job step output with
                | LaneEnd.Running ->
                    let token = int (StagePayload.token output)

                    [
                        plan lane 0 (step + 1) (prompts[lane].Length + step) 1 (StagePayload.ofIds [| token |])
                    ]
                | e ->
                    ends[lane] <- e
                    lanesLeft <- lanesLeft - 1
                    []

        member _.IsComplete = lanesLeft = 0
