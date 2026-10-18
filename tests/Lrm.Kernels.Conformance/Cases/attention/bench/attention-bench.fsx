// The attention bench (K3, ADR K3-0001): attention_gqa_causal on the D3D12 backend with a warm GPU, for the
// Qwen3-0.6B layout (16 query heads, 8 KV heads, head_dim 128) at decode, chunked-prefill and prefill sizes up to 4096
// keys, plus one-head head_dim-1 grids of 8,192, 16,384 and 65,535 queries. Informational, like the bench tier
// (TestTiers): it gates nothing; the conformance cases gate correctness and the budget.
//
// Two modes, one run log each:
// - warm (the default): each configuration is dispatched `reps` times (8) back to back in one submission (a barrier
//   between dispatches), first with telemetry off to raise the GPU clocks, then again inside the run log;
// - cold: the configurations X1's chunking rule rests on (ADR K3-0001, decision 8), each `reps` times (3) as a single
//   dispatch after 2.5 s of idle GPU. That is how the conformance harness meets a case: with the clock still down.
// The backend's gpu.dispatch intervals carry `step` (the configuration's number in `configs`, the same in both modes,
// as printed) and `case` (its name). The run log is of kind `kernels`, so `lrm report <runId>` renders it with the
// kernels template (GPU time by kernel, timeline, dispatch histogram, slowest dispatches).
//
// Usage, after `dotnet build Lrm.slnx -c Release`, from the repository root:
//     dotnet fsi --tailcalls- tests/Lrm.Kernels.Conformance/Cases/attention/bench/attention-bench.fsx [warm|cold] [reps]
// It takes the d3d12 test mutex through Lrm.Testing, so it waits for (and never overlaps) GPU test runs. The cold mode
// holds it for about a minute, mostly idle.
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Core.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Gpu.Core.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Gpu.D3D12.Interop.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Gpu.D3D12.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Shaders.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Telemetry.dll"
#r "../../../../../artifacts/bin/Lrm.Kernels.Conformance/release/Lrm.Testing.dll"

open System
open System.Diagnostics
open System.Runtime.InteropServices
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12
open Lrm.Telemetry
open Lrm.Testing

Process.init ()

/// Warm: back-to-back dispatches after an untimed submission. Cold: one dispatch after `IdleMs` of idle GPU.
type Mode =
    | Warm
    | Cold

let mode, reps =
    let mode, rest =
        match fsi.CommandLineArgs |> Array.tail |> List.ofArray with
        | "cold" :: rest -> Cold, rest
        | "warm" :: rest -> Warm, rest
        | rest -> Warm, rest

    match rest, mode with
    | [ r ], _ -> mode, int r
    | [], Warm -> mode, 8
    | [], Cold -> mode, 3
    | _ -> failwith "usage: attention-bench.fsx [warm|cold] [reps]"

/// The idle time before each cold dispatch: long enough for the RTX 4080 to drop to its idle clock.
[<Literal>]
let IdleMs = 2500

/// One configuration: rows queries at positions past .. past + rows - 1 with the given heads.
type Config =
    {
        Name: string
        Rows: int
        Past: int
        QHeads: int
        KvHeads: int
        HeadDim: int
    }

let qwen name rows past =
    {
        Name = name
        Rows = rows
        Past = past
        QHeads = 16
        KvHeads = 8
        HeadDim = 128
    }

let oneHead name rows =
    {
        Name = name
        Rows = rows
        Past = 0
        QHeads = 1
        KvHeads = 1
        HeadDim = 1
    }

let configs =
    [
        qwen "decode_p511" 1 511
        qwen "decode_p1023" 1 1023
        qwen "decode_p2047" 1 2047
        qwen "decode_p4095" 1 4095
        qwen "chunk_t128_past3968" 128 3968
        qwen "chunk_t256_past3840" 256 3840
        qwen "chunk_t512_past3584" 512 3584
        qwen "chunk_t1024_past3072" 1024 3072
        qwen "prefill_t512" 512 0
        qwen "prefill_t900" 900 0
        qwen "prefill_t1024" 1024 0
        qwen "prefill_t2048" 2048 0
        qwen "prefill_t4096" 4096 0
        oneHead "hd1_rows8192" 8192
        oneHead "hd1_rows16384" 16384
        oneHead "hd1_rows65535" 65535
    ]

/// The configurations the cold mode times: decode at 4095, the chunk sizes at 4096 keys, a 900-token gate prompt, and
/// the prefills of 1024 and 4096 tokens in one dispatch.
let coldConfigs =
    set
        [
            "decode_p4095"
            "chunk_t128_past3968"
            "chunk_t256_past3840"
            "chunk_t512_past3584"
            "chunk_t1024_past3072"
            "prefill_t900"
            "prefill_t1024"
            "prefill_t4096"
        ]

/// One line for the run manifest: what a configuration dispatches.
let describe (c: Config) =
    Invariant.format $"{c.Name}: rows {c.Rows}, past {c.Past}, heads {c.QHeads}:{c.KvHeads}, head_dim {c.HeadDim}"

let git (args: string) =
    try
        let p =
            Process.Start(ProcessStartInfo("git", args, RedirectStandardOutput = true, UseShellExecute = false))

        let out = p.StandardOutput.ReadToEnd().Trim()
        p.WaitForExit()
        if p.ExitCode = 0 then Some out else None
    with _ ->
        None

let held =
    Locks.acquire (Locks.defaultOptions "K3 attention bench") [ Locks.LockId.GpuD3D12 ]

let backend =
    match
        D3D12.tryCreate
            { DeviceOptions.For ExecutionBackend.D3D12 with
                TimeDispatches = true
            }
    with
    | Ok b -> b
    | Error e -> failwithf "no d3d12 device: %s" e

let kernels = backend.LoadKernels(Lrm.Shaders.Shaders.embedded ())

if not (kernels.Contains Kernels.AttentionGqaCausal.Id) then
    failwith "the shader set has no attention_gqa_causal blob"

// One set of tensors large enough for every configuration: q and y for 4096 rows of 16 heads, the caches for 4096
// positions of 8 heads (Qwen3-0.6B), and 65,536 elements each for the one-head grids.
let qwenQ = 4096L * 16L * 128L
let qwenKv = 4096L * 8L * 128L
let oneN = 65536L

let planner = HeapPlanner()

let placed =
    [
        "q", qwenQ
        "k", qwenKv
        "v", qwenKv
        "y", qwenQ
        "q1", oneN
        "k1", oneN
        "v1", oneN
        "y1", oneN
    ]
    |> List.map (fun (name, n) -> name, n, planner.Allocate(ArenaKind.Scratch, n * 4L))

let layout = planner.Build()
let table = TensorTable layout

let tokens =
    placed
    |> List.map (fun (name, n, p) ->
        let slot = layout.SlotOf(p.Arena, p.Page)

        let row =
            TensorRow.create slot p.Offset DType.F32 [ n ] TensorFlags.None (TensorRow.debugIdOf name)

        name, (table.Add("bench/" + name, row), n, slot, p.Offset))
    |> Map.ofList

let heap = backend.CreateHeap layout
heap.WriteTable(table.Seal())

// Deterministic inputs in [-2, 2): the timing does not depend on the values, and a rerun reads the same ones.
let rng = DetRng(0x4B33424E43480001UL, DetRng.streamOf "attention-bench")

for KeyValue(_, (_, n, slot, offset)) in tokens do
    let data = Array.init (int n) (fun _ -> 4.0f * rng.NextF32() - 2.0f)
    heap.Upload(slot, offset, MemoryMarshal.AsBytes(ReadOnlySpan data))

let token name =
    let t, _, _, _ = tokens[name]
    t

let argsOf (c: Config) =
    let s = if c.HeadDim = 128 then "" else "1"

    Kernels.AttentionGqaCausal.encode
        {
            Q = token ("q" + s)
            K = token ("k" + s)
            V = token ("v" + s)
            Y = token ("y" + s)
            Rows = uint32 c.Rows
            QRow0 = 0u
            Past = uint32 c.Past
            QHeads = uint32 c.QHeads
            KvHeads = uint32 c.KvHeads
            HeadDim = uint32 c.HeadDim
            Scale = float32 (1.0 / sqrt (float c.HeadDim))
        }

/// `n` dispatches of one block in one submission; the dispatch times in ms.
let submit (n: int) (args: ArgBlock) =
    let commands = backend.Begin()

    for r in 1..n do
        if r > 1 then
            commands.Barrier()

        commands.Dispatch(Kernels.AttentionGqaCausal.Id, args)

    (backend.Wait(backend.Submit commands)).Timers
    |> List.filter (fun t -> t.Name = GpuTelemetry.Dispatch)
    |> List.map (fun t -> float t.DurationNs / 1e6)

let cls = backend.ExecutionClass

let title, modeText =
    match mode with
    | Warm -> "attention_gqa_causal bench on d3d12 (warm GPU)", "warm: back to back after an untimed submission"
    | Cold ->
        "attention_gqa_causal bench on d3d12 (cold GPU)",
        Invariant.format $"cold: one dispatch after {IdleMs} ms of idle GPU"

/// The configurations this mode times, with their step numbers.
let timed =
    configs
    |> List.indexed
    |> List.filter (fun (_, c) -> mode = Warm || coldConfigs.Contains c.Name)

let run =
    RunContext.Create(
        { RunOptions.create "kernels" title with
            ExecutionClass = Some cls
            GitSha = git "rev-parse HEAD"
            Params =
                [
                    "source", "tests/Lrm.Kernels.Conformance/Cases/attention/bench/attention-bench.fsx"
                    "backend", ExecutionBackend.name cls.Backend
                    "device", backend.Info.Name
                    "mode", modeText
                    "reps", Invariant.i32 reps
                    yield! timed |> List.map (fun (i, c) -> Invariant.format $"step {i}", describe c)
                ]
        }
    )

printfn "run log: %s" run.Directory

printfn
    "device %s, driver %s, dxil ShaderSetHash %s"
    backend.Info.Name
    cls.Driver
    (cls.ShaderSetHash |> Option.map _.Hex |> Option.defaultValue "")

printfn "%-4s %-22s %10s %10s %10s" "step" "config" "min ms" "median ms" "max ms"

/// `n` dispatches of configuration `i` in one submission, inside the run log; the dispatch times in ms.
let measure (i: int) (c: Config) (args: ArgBlock) (n: int) =
    use _ = run.Install()
    use _ = Telemetry.scope [ Attr.int "step" i; Attr.str "case" c.Name ]
    submit n args

for i, c in timed do
    let args = argsOf c

    let times =
        match mode with
        | Warm ->
            submit reps args |> ignore
            measure i c args reps |> List.sort
        | Cold ->
            List.init reps (fun _ ->
                Threading.Thread.Sleep IdleMs
                measure i c args 1)
            |> List.concat
            |> List.sort

    printfn "%-4d %-22s %10.3f %10.3f %10.3f" i c.Name times.Head times[times.Length / 2] (List.last times)

run.Finish RunStatus.Ok
(backend :> IDisposable).Dispose()
(held :> IDisposable).Dispose()
