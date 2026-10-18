/// The GPU backends of the forward-ops registry (GpuExecutor contract): each runs the Qwen forward through the GPU
/// executor (Lrm.Gpu.Exec, GpuForwardOps) on a fresh backend per runner, which the runner owns. This file registers
/// them through Backends.register, after cpu-f32 and cpu-f64 (Lrm.Cli's fixed compile order: Backends, GpuBackends,
/// Forward, Generate):
///
///     cpu-device   the CPU emulation of the kernel catalog (E1), bit-exact to CpuRef F32
///     d3d12        Direct3D 12 (D1), on the first accepted adapter
///     vulkan       not in this build: Lrm.Gpu.Vulkan (V1) has no provider yet; X1b registers it with its provider
///                  (a plan change against X1a's deliverable, recorded in the GpuExecutor contract and PR #31)
///
/// A backend that cannot run the executor (no accepted device, a shader set without the executor's kernels) fails the
/// runner's creation with GpuBackendUnavailableException, a ForwardException that the commands print as one line.
///
/// Apps wire backends and libraries never do (ProjectGraph): the executor knows only IGpuBackend, and this file is
/// where the concrete providers meet it. Owner: X1 (X1a).
///
///     lrm exec-plan [--model DIR|tiny|qwen3-0.6b] [--max-rows 512] [--max-context 512] [--taps none|all|NAMES]
///                   [--caches 1] [--split 14 [--stage 1]] [--max-logits-rows N]
///                                          prints the executor's static plan for a model's config.json
module Lrm.Cli.Commands.GpuBackends

open System
open System.IO
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.D3D12
open Lrm.Gpu.Exec
open Lrm.Models.Qwen
open Lrm.Cli.Commands.Backends

/// The device providers of this build, registered with Lrm.Gpu.Core's registry (GpuBackendInterface: apps do).
let private providers: IGpuBackendProvider list = [ D3D12.provider ]

/// A shader backend from Lrm.Gpu.Core's registry: the first accepted device, without the backend's own dispatch
/// timing (the executor times every dispatch with its kernel, layer and step).
let private shaderDevice (backend: ExecutionBackend) (_: OpsOptions) : IGpuBackend =
    match Backend.tryCreate (DeviceOptions.For backend) with
    | Ok b -> b
    | Error e -> raise (GpuContractException(sprintf "%s: %s" (ExecutionBackend.name backend) e))

/// cpu-device: a CpuDevice with the runner's threads, preserve, CpuRef's NaN and barrier checks on.
let cpuDevice: OpsFactory =
    GpuBackendFactory.create
        ExecutionBackend.CpuDevice
        "GPU executor on the CPU emulation of the kernel catalog (E1), bit-exact to CpuRef F32"
        GpuExecOptions.Default
        Lrm.Shaders.Shaders.embedded
        (fun o ->
            CpuDevice.create
                { CpuDeviceOptions.Default with
                    Threads = o.Threads
                }
                (DeviceOptions.For ExecutionBackend.CpuDevice))

/// d3d12: Direct3D 12 on the first accepted adapter.
let d3d12: OpsFactory =
    GpuBackendFactory.create
        ExecutionBackend.D3D12
        "GPU executor on Direct3D 12 (D1), first accepted adapter"
        GpuExecOptions.Default
        Lrm.Shaders.Shaders.embedded
        (shaderDevice ExecutionBackend.D3D12)

/// The GPU entries, in registry order.
let all = [ cpuDevice; d3d12 ]

do
    for p in providers do
        Backend.register p

    for f in all do
        Backends.register f

let private usage () =
    eprintfn "usage: lrm exec-plan [--model DIR|tiny|qwen3-0.6b] [--max-rows 512] [--max-context 512]"
    eprintfn "                     [--taps none|all|NAMES] [--caches 1] [--split LAYER [--stage I]]"
    eprintfn "                     [--max-logits-rows N]"
    eprintfn ""
    eprintfn "  Prints the GPU executor's static plan (GpuExecutor contract) for the model's config.json: the arenas,"
    eprintfn "  the table, each stage's static order and pool, and the dispatches of a prefill of --max-rows rows and"
    eprintfn "  of a decode step at the last position. No weights are read and no device is created. --split plans"
    eprintfn "  both stages of a split, --stage I only stage I of it (a pipeline node: only its layers' weights)."
    eprintfn "  --max-logits-rows caps the rows of an All-head forward (GpuExecOptions.MaxLogitsRows; 1 for a plan"
    eprintfn "  that only decodes greedily)."
    2

let private mib (bytes: int64) =
    Invariant.format $"{float bytes / 1048576.0:F2} MiB"

/// The dispatches one forward of `rows` rows at `pos0` makes, by kernel, through a stage variant's static order.
let private dispatchesOf (plan: ExecPlan) (v: PlanVariant) (rows: int) (pos0: int) =
    let shapes = ExecPlan.weightShapes plan.Model |> dict

    v.Ops
    |> Array.collect (fun o ->
        let r = if o.Rows = 1 then 1 else rows

        match o.Kind with
        | PlanOpKind.MatMul w ->
            let d = shapes[w]

            Chunking.matmul plan.Exec.MatMulMacsPerDispatch r (int d[0]) (int d[1])
            |> List.map (fun _ -> "matmul_bf16w")
            |> Array.ofList
        | PlanOpKind.Attention _ ->
            Chunking.attention plan.Exec.AttentionPairsPerDispatch r pos0
            |> List.map (fun _ -> "attention_gqa_causal")
            |> Array.ofList
        | k -> [| PlanOpKind.kernel k |])
    |> Array.countBy id
    |> List.ofArray

let private plan (argv: string[]) =
    let parsed =
        Cli.parse
            [
                "model"
                "max-rows"
                "max-context"
                "taps"
                "caches"
                "split"
                "stage"
                "max-logits-rows"
            ]
            []
            argv

    let build () =
        Cli.result {
            let! p = parsed
            let! dir = Cli.modelDir (p.Get "model")
            let config = Qwen3Config.load (Path.Combine(dir, "config.json"))

            let! maxContext =
                Cli.intOption p "max-context" (min 512 config.MaxPositionEmbeddings)

            let! maxRows = Cli.intOption p "max-rows" maxContext
            let! caches = Cli.intOption p "caches" 1

            let! logitsRows =
                Cli.intOption p "max-logits-rows" GpuExecOptions.Default.MaxLogitsRows

            let! stages =
                match p.Get "split", p.Get "stage" with
                | None, None -> Ok [ Stage.whole config ]
                | None, Some _ -> Error "--stage needs --split"
                | Some s, only ->
                    match Invariant.tryParseI32 s with
                    | None -> Error(sprintf "--split '%s' is not a layer" s)
                    | Some b ->
                        let stages = Stage.split config [ b ]

                        match only with
                        | None -> Ok stages
                        | Some i ->
                            match Invariant.tryParseI32 i with
                            | Some n when n >= 0 && n < stages.Length -> Ok [ stages[n] ]
                            | _ ->
                                Error(
                                    sprintf "--stage '%s' is not a stage of the split (0 .. %d)" i (stages.Length - 1)
                                )

            let model =
                {
                    Config = config
                    Params = Qwen3Params.ofConfig config config.TieWordEmbeddings
                }

            let options =
                { OpsOptions.defaults config with
                    MaxContext = maxContext
                    MaxRows = maxRows
                    Stages = stages
                    Taps = TapFilter.Parse(defaultArg (p.Get "taps") "none")
                    Caches = caches
                }

            let exec =
                { GpuExecOptions.Default with
                    MaxLogitsRows = logitsRows
                }

            return dir, model, options, exec
        }

    let run =
        try
            build ()
        with
        | :? FormatException as e -> Error e.Message
        | :? IOException as e -> Error e.Message

    match run with
    | Error e ->
        eprintfn "lrm exec-plan: %s" e
        usage ()
    | Ok(dir, model, options, exec) ->
        try
            let plan = ExecPlan.build model options exec
            let c = model.Config

            printfn
                "lrm exec-plan: %s: %d layers, hidden %d, vocab %d; MaxRows %d, MaxContext %d, %d cache(s), taps %s"
                (Cli.displayPath dir)
                c.Layers
                c.Hidden
                c.Vocab
                plan.MaxRows
                options.MaxContext
                options.Caches
                (match options.Taps with
                 | TapFilter.Off -> "none"
                 | TapFilter.All -> "all"
                 | TapFilter.Names n -> Invariant.format $"{n.Length} named")

            printfn
                "heap: %s in %d page(s) and the table; %d of %d table rows"
                (mib plan.Layout.TotalBytes)
                plan.Layout.PageCount
                plan.Table.Count
                (Heap.TableRows - 1)

            for a in plan.Layout.Arenas do
                let pages = plan.Layout.Pages |> List.filter (fun pg -> pg.Arena = a)
                printfn "  %-12s %12s  %d page(s)" (ArenaKind.name a) (mib (plan.Layout.ArenaBytes a)) pages.Length

            for sp in plan.Stages do
                printfn
                    "stage %d (layers %d .. %d)%s: %d tap snapshot(s)"
                    sp.Stage.Index
                    sp.Stage.FirstLayer
                    sp.Stage.LastLayer
                    (if sp.HasHead then
                         Invariant.format $", with the head (HeadRows.All up to {plan.LogitsRows} rows)"
                     else
                         "")
                    sp.Taps.Count

                for v in sp.Variants do
                    // The all head serves at most LogitsRows rows; a decode step (one row) always runs `all`
                    // (BeginForward picks `last` only for HeadRows.Last over more than one row).
                    let rows =
                        if sp.HasHead && not v.LastRow then
                            min plan.MaxRows plan.LogitsRows
                        else
                            plan.MaxRows

                    let prefill = dispatchesOf plan v rows 0
                    let total (xs: (string * int) list) = xs |> List.sumBy snd

                    let decode =
                        if v.LastRow then
                            "n/a (a decode step runs all)"
                        else
                            Invariant.i32 (total (dispatchesOf plan v 1 (options.MaxContext - 1)))

                    printfn
                        "  %-4s %d ops, pool %s; a prefill of %d rows: %d dispatches; a decode step: %s"
                        v.Name
                        v.Ops.Length
                        (mib v.PoolBytes)
                        rows
                        (total prefill)
                        decode

                    for k, n in prefill do
                        printfn "         %-22s %5d" k n

            0
        with
        | :? ArgumentException as e ->
            eprintfn "lrm exec-plan: %s" e.Message
            2
        | :? GpuContractException as e ->
            eprintfn "lrm exec-plan: %s" e.Message
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
            Name = "exec-plan"
            Summary = "Print the GPU executor's static plan (arenas, table, dispatches) for a model's config."
            Run = plan
        |}
    ]
