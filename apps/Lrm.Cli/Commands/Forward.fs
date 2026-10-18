/// One forward pass (a prefill from position 0) on a registered backend (Backends.fs): prints the logits, writes them
/// and the taps in the golden layout, and compares them with a golden file (ForwardOpsAndQwenModel contract). Owner:
/// M2a.
///
///     lrm forward [--model DIR|tiny|qwen3-0.6b] [--backend cpu-f32]
///                 (--ids 1,2,3 | --ids-from FILE[:TENSOR] | --prompt TEXT) [--taps none|all|NAME,...] [--head all|last]
///                 [--out FILE.safetensors] [--compare FILE.safetensors] [--pair NAME] [--bound RELL2]
///                 [--tolerances FILE] [--case NAME] [--threads 8] [--run-log] [--title TEXT]
///
/// --out writes input_ids (I64 [T]), the taps and logits, as the golden fwd_*.safetensors files hold them. --compare
/// measures every written tensor that the file also holds (relL2; max |Δ| and KL for logits) as golden.compare
/// checks, with the ToleranceSpec pair of the backend and the file's dtype (F32: torch32, F64: torch64p) unless --pair
/// names one; it exits 1 when one fails. --run-log writes a run log (kind parity with --compare, else inference).
module Lrm.Cli.Commands.Forward

open System
open System.IO
open Lrm.Core
open Lrm.Tensor
open Lrm.Telemetry
open Lrm.Models.Qwen
open Lrm.Cli.Commands.Backends

let private usage () =
    eprintfn "usage: lrm forward [--model DIR|tiny|qwen3-0.6b] [--backend NAME]"
    eprintfn "                   (--ids 1,2,3 | --ids-from FILE[:TENSOR] | --prompt TEXT)"
    eprintfn "                   [--taps none|all|NAME,...] [--head all|last] [--out FILE.safetensors]"
    eprintfn "                   [--compare FILE.safetensors] [--pair NAME] [--bound RELL2] [--tolerances FILE]"
    eprintfn "                   [--case NAME] [--threads 8] [--run-log] [--title TEXT]"
    eprintfn ""
    eprintfn "  --taps        ActivationNames taps to record (default none; all for every tap)"
    eprintfn "  --head        logits for every row (all, the default) or the last row only"
    eprintfn "  --out         write input_ids, the taps and logits as safetensors (the golden layout)"
    eprintfn "  --compare     compare with a golden fwd_*.safetensors file; exit 1 when a check fails"
    eprintfn "  --pair        the ToleranceSpec pair (default: from the backend and the file's dtype)"
    eprintfn "  --bound       the relL2 bound of every tap (default: the pair's floor or fixed bound)"
    eprintfn "  --tolerances  a tolerances.json (ToleranceSpec) for the bounds"
    eprintfn "  --run-log     write a run log under LRM_DATA/runs (kind parity with --compare)"
    2

let private values =
    [
        "model"
        "backend"
        "ids"
        "ids-from"
        "prompt"
        "taps"
        "head"
        "out"
        "compare"
        "pair"
        "bound"
        "tolerances"
        "case"
        "threads"
        "title"
    ]

let private switches = [ "run-log" ]

/// The default bound of a pair and metric: ToleranceSpec's fixed bounds and floors (the full-model cpu64 bound).
let private defaultBound (pair: TolerancePair option) (metric: ToleranceMetric) =
    match pair, metric with
    | Some TolerancePair.Cpu64VsTorch64p, ToleranceMetric.RelL2 -> Some 1e-10
    | Some TolerancePair.Cpu64VsTorch64p, _ -> None
    | Some(TolerancePair.Cpu32VsTorch32 | TolerancePair.Cpu32VsTorch64p), ToleranceMetric.RelL2 -> Some 1e-5
    | Some(TolerancePair.Cpu32VsTorch32 | TolerancePair.Cpu32VsTorch64p), ToleranceMetric.MaxAbs -> Some 2e-3
    | Some(TolerancePair.Cpu32VsTorch32 | TolerancePair.Cpu32VsTorch64p), ToleranceMetric.Kl -> Some 1e-7
    | Some TolerancePair.VulkanVsD3D12, ToleranceMetric.RelL2 -> Some 1e-4
    | Some TolerancePair.VulkanVsD3D12, ToleranceMetric.MaxAbs -> Some 1e-2
    | Some TolerancePair.VulkanVsD3D12, ToleranceMetric.Kl -> None
    | Some _, ToleranceMetric.RelL2 -> Some 1e-4
    | Some _, ToleranceMetric.MaxAbs -> Some 1e-2
    | Some _, ToleranceMetric.Kl -> Some 1e-6
    | None, ToleranceMetric.RelL2 -> Some 1e-5
    | None, _ -> None

/// The last row of a [rows, cols] tensor, in its dtype.
let private lastRow (t: IHostTensor) : IHostTensor =
    let rows = int t.Desc.Shape.Rows
    let cols = int t.Desc.Shape.Cols
    let size = t.Desc.ElementSize
    let bytes = ReadOnlySpan(t.ToBytes(), (rows - 1) * cols * size, cols * size)

    match t.Desc.DType with
    | DType.F32 -> HostTensor.ofBytes<float32> DType.F32 (Shape.matrix 1 cols) bytes :> IHostTensor
    | DType.F64 -> HostTensor.ofBytes<float> DType.F64 (Shape.matrix 1 cols) bytes :> IHostTensor
    | d -> invalidArg "t" (DType.name d)

type private Comparison =
    {
        Name: string
        Checks: CheckResult list
        Note: string
    }

let private compare
    (file: SafetensorsFile)
    (outputs: (string * IHostTensor) list)
    (pair: TolerancePair option)
    (bound: float option)
    (spec: ToleranceSpec option)
    (case: string)
    =
    let boundOf (name: string) (metric: ToleranceMetric) =
        match metric, bound with
        | ToleranceMetric.RelL2, Some b -> Some b
        | _ ->
            match pair, spec with
            | Some p, Some s ->
                s.TryBound(p, ActivationNames.kind name, metric)
                |> Option.map _.Bound
                |> Option.orElse (defaultBound pair metric)
            | _ -> defaultBound pair metric

    [
        for name, ours in outputs do
            match file.TryEntry name with
            | None -> ()
            | Some _ ->
                let golden = file.Tensor name
                let ourRows = ours.Desc.Shape.Rows

                // A last-row head against a golden of every row: compare with the golden's last row.
                let golden, note =
                    if
                        ourRows = 1L
                        && golden.Desc.Shape.Rows > 1L
                        && ours.Desc.Shape.Cols = golden.Desc.Shape.Cols
                    then
                        lastRow golden, " (last row)"
                    else
                        golden, ""

                if not (ours.Desc.Shape.Equals golden.Desc.Shape) then
                    yield
                        {
                            Name = name
                            Checks = []
                            Note = sprintf "shape %O, golden %O: not compared" ours.Desc.Shape golden.Desc.Shape
                        }
                else
                    let attrs =
                        [
                            match pair with
                            | Some p -> yield Attr.str "pair" (TolerancePair.name p)
                            | None -> ()
                            yield Attr.str "prompt" case
                            yield Attr.str "tap" name
                            yield Attr.int TelemetryKeys.Layer (ActivationNames.layer name)
                            yield Attr.str "kind" "chain"
                        ]

                    let checks =
                        [
                            match boundOf name ToleranceMetric.RelL2 with
                            | Some b -> yield Compare.relL2 b attrs ours golden
                            | None -> ()
                            if name = "logits" then
                                match boundOf name ToleranceMetric.MaxAbs with
                                | Some b -> yield Compare.maxAbs b attrs ours golden
                                | None -> ()

                                match boundOf name ToleranceMetric.Kl with
                                | Some b -> yield Compare.kl b attrs ours golden
                                | None -> ()
                        ]

                    yield
                        {
                            Name = name
                            Checks = checks
                            Note = note
                        }
    ]

let private run (argv: string[]) : int =
    match Cli.parse values switches argv with
    | Error e ->
        eprintfn "lrm forward: %s" e
        usage ()
    | Ok p ->
        let setup =
            Cli.result {
                let! dir = Cli.modelDir (p.Get "model")
                let! backends = Cli.backends (p.Get "backend")
                let! threads = Cli.intOption p "threads" (min 8 Environment.ProcessorCount)

                let! head =
                    match p.Get "head" with
                    | None
                    | Some "all" -> Ok HeadRows.All
                    | Some "last" -> Ok HeadRows.Last
                    | Some h -> Error(sprintf "--head '%s' is neither all nor last" h)

                let! pair =
                    match p.Get "pair" with
                    | None -> Ok None
                    | Some n ->
                        match TolerancePair.tryParse n with
                        | Some x -> Ok(Some x)
                        | None -> Error(sprintf "--pair '%s' is not a ToleranceSpec pair" n)

                let! bound =
                    match p.Get "bound" with
                    | None -> Ok None
                    | Some b ->
                        match Invariant.tryParseF64 b with
                        | Some x when x > 0.0 && Double.IsFinite x -> Ok(Some x)
                        | _ -> Error(sprintf "--bound '%s' is not a positive number" b)

                match backends with
                | [ f ] when threads >= 1 -> return dir, f, threads, head, pair, bound
                | [ _ ] -> return! Error "--threads must be at least 1"
                | _ -> return! Error "lrm forward runs one backend; lrm generate takes several"
            }

        match setup with
        | Error e ->
            eprintfn "lrm forward: %s" e
            2
        | Ok(dir, factory, threads, head, pairArg, bound) ->
            try
                use w = Qwen3Weights.Open dir

                match Cli.promptIds p w with
                | Error e ->
                    eprintfn "lrm forward: %s" e
                    2
                | Ok(ids, source) ->
                    let taps = TapFilter.Parse(defaultArg (p.Get "taps") "none")
                    let spec = p.Get "tolerances" |> Option.map ToleranceSpec.load

                    let goldenFile =
                        p.Get "compare"
                        |> Option.map (fun f -> SafetensorsFile.Open(Path.GetFullPath f))

                    use _ =
                        { new IDisposable with
                            member _.Dispose() =
                                goldenFile |> Option.iter (fun f -> f.Dispose())
                        }

                    let case =
                        p.Get "case"
                        |> Option.orElse (
                            p.Get "compare"
                            |> Option.map (fun f ->
                                match (Path.GetFileName f).Split('.') with
                                | [| _; prompt; "safetensors" |] -> prompt
                                | _ -> Path.GetFileNameWithoutExtension f)
                        )
                        |> Option.defaultValue "cli"

                    let runner =
                        factory.Create
                            w
                            {
                                Threads = threads
                                MaxContext = max 1 ids.Length
                            }

                    let host =
                        Cli.RunHost(
                            (if goldenFile.IsSome then "parity" else "inference"),
                            p.Get "title"
                            |> Option.defaultValue (
                                Invariant.format
                                    $"lrm forward: {Path.GetFileName w.Dir} ({w.Revision.Text}) on {factory.Name}, {ids.Length} tokens"
                            ),
                            p.Has "run-log",
                            [
                                "model", w.Dir
                                "revision", w.Revision.Text
                                "model_sha256", w.Sha256.Hex
                                "prompt", source
                                "prompt_tokens", Invariant.i32 ids.Length
                                "backend", factory.Name
                                "threads", Invariant.i32 threads
                                "command", "lrm forward"
                                match p.Get "compare" with
                                | Some f -> "compare", f
                                | None -> ()
                            ]
                        )

                    printfn "lrm forward: %s (%s), %d tokens (%s)" w.Dir w.Revision.Text ids.Length source
                    printfn "%s: %s" factory.Name (Cli.describe runner.ExecutionClass)

                    let mutable failed = false

                    do
                        use _ = host.Install(0, runner.ExecutionClass)
                        let r = runner.Forward(ids, taps, head)
                        let logits = r.Logits
                        let rows = int logits.Desc.Shape.Rows
                        let values = logits.ToFloat64()
                        let cols = int logits.Desc.Shape.Cols
                        let last = ReadOnlySpan(values, (rows - 1) * cols, cols)
                        let top = Metrics.topK last 5

                        printfn
                            "  logits %O %s; last row top-5: %s"
                            logits.Desc.Shape
                            (DType.name logits.Desc.DType)
                            (top
                             |> Array.map (fun i ->
                                 Invariant.format $"{i} ({Invariant.f64 values[(rows - 1) * cols + i]})")
                             |> String.concat ", ")

                        let argmax =
                            [|
                                for t in 0 .. rows - 1 -> (Metrics.topK (ReadOnlySpan(values, t * cols, cols)) 1)[0]
                            |]

                        printfn "  argmax per row: %s" (argmax |> Array.map Invariant.i32 |> String.concat " ")

                        if not r.Taps.IsEmpty then
                            printfn "  taps: %d (%s .. %s)" r.Taps.Length (fst r.Taps.Head) (fst (List.last r.Taps))

                        let outputs =
                            r.Taps
                            @ (if r.Taps |> List.exists (fun (n, _) -> n = "logits") then
                                   []
                               else
                                   [ "logits", logits ])

                        match p.Get "out" with
                        | None -> ()
                        | Some out ->
                            let path = Path.GetFullPath out

                            let tensors =
                                SafetensorsTensor.ofHost
                                    "input_ids"
                                    (HostTensor.i64 (Shape.vector ids.Length) (ids |> Array.map int64))
                                :: (outputs |> List.map (fun (n, t) -> SafetensorsTensor.ofHost n t))

                            Safetensors.writeFile
                                path
                                [
                                    "lrm.backend", factory.Name
                                    "lrm.xclass", (ExecutionClass.id runner.ExecutionClass).Value.Hex
                                    "lrm.model.revision", w.Revision.Text
                                    "lrm.model.sha256", w.Sha256.Hex
                                    "lrm.head", (if head = HeadRows.All then "all" else "last")
                                ]
                                tensors

                            printfn "  wrote %s (%d tensors)" path tensors.Length

                        match goldenFile with
                        | None -> ()
                        | Some f ->
                            let oracle =
                                match f.TryEntry "logits" with
                                | Some e when e.Desc.DType = DType.F64 -> "torch64p"
                                | _ -> "torch32"

                            let pair = pairArg |> Option.orElse (Cli.pairFor factory.Name oracle)

                            if f.Contains "input_ids" && (f.ReadI64 "input_ids").Data |> Array.map int <> ids then
                                printfn "  note: the file's input_ids differ from the prompt"

                            let results = compare f outputs pair bound spec case

                            printfn
                                "  compare with %s (%s, pair %s):"
                                (p.Get "compare").Value
                                oracle
                                (pair |> Option.map TolerancePair.name |> Option.defaultValue "none")

                            for c in results do
                                match c.Checks with
                                | [] -> printfn "    %-34s %s" c.Name c.Note
                                | checks ->
                                    for k in checks do
                                        printfn
                                            "    %-34s %-7s %-24s <= %-10s %s%s"
                                            c.Name
                                            k.Metric
                                            (Invariant.f64 k.Measured)
                                            (Invariant.f64 k.Bound)
                                            (if k.Passed then "ok" else "FAIL")
                                            c.Note

                            let all = results |> List.collect _.Checks
                            let bad = all |> List.filter (fun k -> not k.Passed)
                            printfn "  %d checks, %d failed" all.Length bad.Length

                            if not bad.IsEmpty || all.IsEmpty then
                                failed <- true

                    host.PrintRates 0
                    host.Finish(if failed then RunStatus.Failed else RunStatus.Ok)
                    if failed then 1 else 0
            with
            | :? Qwen3WeightsException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? Qwen3ConfigException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? IntegrityException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? ToleranceFormatException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? SafetensorsFormatException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? ForwardException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? ArgumentException as e ->
                eprintfn "lrm forward: %s" e.Message
                1
            | :? IOException as e ->
                eprintfn "lrm forward: %s" e.Message
                1

/// Commands this module adds to `lrm`. Program.fs concatenates every module's list in compile order.
let commands
    : {|
          Name: string
          Summary: string
          Run: string[] -> int
      |} list =
    [
        {|
            Name = "forward"
            Summary = "Run a forward pass; print, write and compare logits and taps."
            Run = run
        |}
    ]
