/// Synthetic run logs for every template (ReportTemplates contract, "Synthetic runs"), so the reports can be reviewed
/// before the real runs exist. Each generator writes a real run folder through RunContext and the JSONL sink, on a
/// virtual clock, with numbers drawn from DetRng and shaped with DetMath only: the same call writes the same bytes on
/// every machine. Every manifest says synthetic = true, and every report then carries a banner saying so.
module Lrm.Reports.Synthetic

open System
open Lrm.Core
open Lrm.Telemetry

/// A clock that moves only when told to.
[<Sealed>]
type VirtualClock(start: int64) =
    let mutable now = start
    member _.Now() = now
    member _.Advance(ns: int64) = now <- now + ns
    member _.Set(ns: int64) = now <- ns

/// 2026-10-18T09:00:00Z in Unix ns, the start of every synthetic run.
let epochNs =
    DateTimeOffset(2026, 10, 18, 9, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds()
    * 1_000_000L

let private seed = 0x4C524D2D54315359UL

/// A deterministic generator for one synthetic run.
let private rngFor (name: string) =
    DetRng(seed, DetRng.streamOf ("synthetic." + name))

/// An approximately standard normal draw: the sum of four uniforms, centred and scaled to variance 1.
let private normal (r: DetRng) =
    (r.NextF64() + r.NextF64() + r.NextF64() + r.NextF64() - 2.0)
    * 1.7320508075688772

/// A log-normal factor exp(sigma * z).
let private lognormal (r: DetRng) (sigma: float) = DetMath.exp (sigma * normal r)

let private ns (x: float) = int64 (Math.Round x)

/// Illustrative execution classes (the GPU values follow the ExecutionClass contract's vector X2).
let private cpuClass backend =
    {
        Backend = backend
        DeviceId = "cpu"
        Driver = ".NET 10.0.13"
        ShaderSetHash = None
        Arch = "win-x64+avx2+fma"
    }

let private shaderSet = Hash.sha256Utf8 "synthetic shader set"

let private vulkanClass =
    {
        Backend = ExecutionBackend.Vulkan
        DeviceId = "pci:10de:2704"
        Driver = "616.92"
        ShaderSetHash = Some shaderSet
        Arch = "win-x64"
    }

let private d3d12Class =
    { vulkanClass with
        Backend = ExecutionBackend.D3D12
    }

let private options (kind: string) (title: string) (root: string) (clock: VirtualClock) xclass pars =
    { RunOptions.create kind title with
        Node =
            (if kind = "cluster" then
                 NodeId.parse "coord"
             else
                 NodeId.local)
        ExecutionClass = xclass
        Params = ("generator", "lrm report --synthetic " + kind) :: pars
        Synthetic = true
        RunId = Some(RunId.parse ("synthetic-" + kind))
        RunsRoot = Some root
        Clock = Some clock.Now
        FlushInterval = TimeSpan.Zero
    }

let private qwen = TransformerDims.qwen3_0_6b

let private modelParams =
    [
        "model", "Qwen3-0.6B"
        "revision", "c1899de289a04d12100db370d81485cdf75e47ca"
    ]

// ---- parity -------------------------------------------------------------------------------------------------------

let private parity (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "parity"

    use run =
        RunContext.Create(
            options
                "parity"
                "CPU reference vs the goldens (synthetic)"
                root
                clock
                (Some(cpuClass ExecutionBackend.CpuF32))
                (modelParams @ [ "goldens", "qwen3-0.6b/g1"; "tolerances", "v1"; "prompts", "8" ])
        )

    let t = run.Telemetry.Child [ Attr.str TelemetryKeys.Component "parity" ]

    let pairs =
        [
            "cpu64-vs-torch64p", 1e-10, 2e-13, 1e-8, 3e-11
            "cpu32-vs-torch32", 2e-5, 3e-7, 2e-3, 2.5e-4
        ]

    let taps =
        [
            "q_proj", 0.8
            "k_proj", 0.9
            "v_proj", 0.7
            "o_proj.out", 1.1
            "down.out", 1.4
            "out", 1.0
        ]

    let emit pair case tap (layer: int option) metric measured bound =
        let attrs =
            [
                Attr.str "pair" pair
                Attr.str TelemetryKeys.Case case
                Attr.str "tap" tap
                Attr.str "metric" metric
                Attr.float "measured" measured
                Attr.float "bound" bound
                match layer with
                | Some l -> Attr.int TelemetryKeys.Layer l
                | None -> ()
            ]

        clock.Advance 40_000L
        t.Check(TelemetryNames.GoldenCompare, measured <= bound, attrs)

    for c in 1..8 do
        let case = Invariant.format $"g{c:D2}"
        use _ = t.Span("parity.case", [ Attr.str TelemetryKeys.Case case ])
        clock.Advance 2_000_000L

        for pair, bound, base', absBound, absBase in pairs do
            emit pair case "embed.out" None "rel_l2" 0.0 bound

            for layer in 0 .. int qwen.Layers - 1 do
                for tap, factor in taps do
                    let depth = 1.0 + 0.12 * float layer
                    let v = base' * depth * factor * lognormal r 0.35
                    // One tap of one prompt runs at about 80% of its bound, so the review sees a near miss.
                    let v =
                        if c = 5 && layer = 27 && tap = "down.out" then
                            bound * (0.82 + 0.02 * normal r)
                        else
                            v

                    emit pair case ("L" + Invariant.i32 layer + "." + tap) (Some layer) "rel_l2" v bound

            emit pair case "final_norm.out" None "rel_l2" (base' * 4.2 * lognormal r 0.3) bound
            emit pair case "logits" None "rel_l2" (base' * 5.0 * lognormal r 0.3) bound
            emit pair case "logits" None "max_abs" (absBase * lognormal r 0.4) absBound
            emit pair case "greedy" None "greedy_match" 64.0 64.0

    // The oracle self-tests and greedy margins that G2's golden tools record (ReferenceOracles, "Telemetry").
    for check, measured, bound in [ "rmsnorm_fp64", 1.2e-16, 1e-15; "attention_fp64", 4.5e-16, 1e-13 ] do
        clock.Advance 40_000L

        t.Check(
            TelemetryNames.GoldenCompare,
            true,
            [
                Attr.str "pair" "oracle-selftest"
                Attr.str "check" check
                Attr.str "metric" "relL2"
                Attr.float "measured" measured
                Attr.float "bound" bound
            ]
        )

    for check in [ "bf16_rne"; "rope_t_spec"; "safetensors" ] do
        t.Check("goldens.selftest", true, [ Attr.str "check" check ])

    for c in 1..8 do
        let prompt = Invariant.format $"g{c:D2}"
        let margin32 = 0.06 + 0.5 * r.NextF64()

        for oracle, margin in [ "torch32", margin32; "torch64p", margin32 * (1.0 + 1e-6 * normal r) ] do
            let attrs = [ Attr.str "prompt" prompt; Attr.str "oracle" oracle ]
            clock.Advance 40_000L
            t.Gauge("goldens.greedy.min_margin", margin, "", attrs)
            // One float32 ulp of a top logit near 20 is 2^-19.
            t.Gauge("goldens.greedy.min_margin_f32_ulps", margin * 524288.0, "ulp", attrs)
            t.Check("goldens.greedy.f32_argmax_agrees", true, attrs)

    run.Finish RunStatus.Ok
    run.Directory

// ---- inference ----------------------------------------------------------------------------------------------------

let private inference (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "inference"

    use run =
        RunContext.Create(
            options
                "inference"
                "Prefill 512 tokens, then 64 greedy steps on Vulkan (synthetic)"
                root
                clock
                (Some vulkanClass)
                (modelParams @ [ "prompt_tokens", "512"; "decode_steps", "64" ])
        )

    let t =
        run.Telemetry.Child [ Attr.str TelemetryKeys.Component "forward"; Attr.int TelemetryKeys.Stage 0 ]

    let prompt = 512
    let gib = 1073741824.0

    let arena name (bytes: float) =
        t.Gauge(TelemetryNames.MemArena, bytes, TelemetryUnits.Bytes, [ Attr.str TelemetryKeys.Arena name ])

    clock.Advance 5_000_000L
    arena "weights" (1.40 * gib)
    arena "rope" 4194304.0
    arena "scratch" 16777216.0
    arena "logits" (float qwen.Vocab * 4.0)

    let kvPerToken = float (qwen.Layers * qwen.KvHeads * qwen.HeadDim * 2L * 4L)
    let mutable gpu = 2_000_000_000L

    // The GPU work of one layer: most of its host time, split over its kernels.
    let gpuWork (step: int) (layer: int) (hostNs: int64) =
        for kernel, share in [ "rmsnorm_rows", 0.06; "matmul_bf16w", 0.62; "attention_gqa_causal", 0.17 ] do
            let d = ns (float hostNs * share * lognormal r 0.03)

            t.GpuInterval(
                "gpu.dispatch",
                gpu,
                d,
                [
                    Attr.str TelemetryKeys.Kernel kernel
                    Attr.int TelemetryKeys.Layer layer
                    Attr.int "step" step
                ]
            )

            gpu <- gpu + d + 700L

    for step in 0..64 do
        let prefill = step = 0
        let rows = if prefill then prompt else 1
        let past = if prefill then 0 else prompt + step - 1
        let phase = if prefill then "prefill" else "decode"

        use fwd =
            t.Span("model.forward", [ Attr.str "phase" phase; Attr.int "step" step; Attr.int "tokens" rows ])

        arena "activations" (float rows * float qwen.Intermediate * 4.0 * 4.0 + 1048576.0)

        for layer in 0 .. int qwen.Layers - 1 do
            use _ =
                t.Span("model.layer", [ Attr.int TelemetryKeys.Layer layer; Attr.str "phase" phase ])
            // Attention grows with the context; the MLP dominates; layer 0 also pays for warm caches.
            let perToken = if prefill then 5_200.0 else 88_000.0
            let attention = if prefill then 1.8 * float rows else 0.035 * float past
            let warm = if layer = 0 && prefill then 1.35 else 1.0

            let hostNs =
                ns ((perToken * float rows + attention * 1_000.0) * warm * lognormal r 0.04)

            gpuWork step layer hostNs
            clock.Advance hostNs

        use _ = t.Span("model.lm_head", [ Attr.str "phase" phase ])
        clock.Advance(ns (920_000.0 * lognormal r 0.05))
        let cost = OpCost.forward qwen (int64 rows) (int64 past) 1L
        t.Meter("model.flops", float cost.Flops, TelemetryUnits.Flops, [ Attr.str "phase" phase ])
        t.Meter("model.bytes_moved", float cost.BytesMoved, TelemetryUnits.Bytes, [ Attr.str "phase" phase ])
        t.Meter("model.tokens", float rows, TelemetryUnits.Tokens, [ Attr.str "phase" phase ])
        arena "kv_cache" (kvPerToken * float (past + rows))
        // A sampling pause between steps, outside the forward span.
        clock.Advance 150_000L

    run.Finish RunStatus.Ok
    run.Directory

// ---- kernels ------------------------------------------------------------------------------------------------------

let private kernelList =
    [
        // kernel, conformance shapes, bit-exact, decode microseconds
        "embed_gather_bf16", [ "T=1"; "T=37"; "T=512" ], true
        "rmsnorm_rows", [ "1x1024"; "37x1024"; "512x1024"; "16x128" ], false
        "rope_apply", [ "T=1 h=16"; "T=37 h=16"; "T=512 h=8" ], false
        "silu_mul", [ "1x3072"; "512x3072" ], false
        "add", [ "1x1024"; "512x1024" ], true
        "copy_rows", [ "1x1024"; "64x1024" ], true
        "argmax_rows", [ "1x151936"; "8x151936" ], true
        "matmul_bf16w",
        [
            "M=1 N=2048 K=1024"
            "M=37 N=3072 K=1024"
            "M=512 N=1024 K=3072"
            "M=1 N=151936 K=1024"
        ],
        false
        "attention_gqa_causal", [ "T=1 past=511"; "T=37 past=0"; "T=512 past=0"; "T=1024 past=3072" ], false
    ]

let private layerKernels =
    [
        "rmsnorm_rows", 2.1
        "matmul_bf16w", 9.4
        "matmul_bf16w", 5.1
        "matmul_bf16w", 5.1
        "rmsnorm_rows", 1.3
        "rmsnorm_rows", 1.1
        "rope_apply", 1.6
        "rope_apply", 1.2
        "attention_gqa_causal", 6.3
        "matmul_bf16w", 9.2
        "add", 1.2
        "rmsnorm_rows", 2.1
        "matmul_bf16w", 13.8
        "matmul_bf16w", 13.8
        "silu_mul", 1.7
        "matmul_bf16w", 14.1
        "add", 1.2
    ]

let private kernels (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "kernels"

    use vk =
        RunContext.Create(
            options
                "kernels"
                "Kernel conformance and decode dispatch timing (synthetic)"
                root
                clock
                (Some vulkanClass)
                [ "catalog", "contracts-v1"; "steps", "2" ]
        )

    use dx =
        RunContext.Join(vk.Directory, NodeId.parse "n1", Some d3d12Class, Some clock.Now)

    for node, (ctx: RunContext), speed in [ "vulkan", vk, 1.0; "d3d12", dx, 1.08 ] do
        // The denormal probe each backend runs after LoadKernels (GpuBackendInterface): the two modes differ here so
        // the review sees both rows.
        ctx.Telemetry.Event(
            "gpu.fp32_denormals",
            [
                Attr.str TelemetryKeys.Backend node
                Attr.str "mode" (if node = "vulkan" then "flush" else "preserve")
            ]
        )

        let t = ctx.Telemetry.Child [ Attr.str TelemetryKeys.Component "conformance" ]

        for kernel, shapes, exact in kernelList do
            for shape in shapes do
                let bound = if exact then 0.0 else 2.4e-6
                // One synthetic failure on d3d12, so the review sees how a failing case renders.
                let failing =
                    node = "d3d12" && kernel = "attention_gqa_causal" && shape = "T=1024 past=3072"

                let measured =
                    if exact then 0.0
                    elif failing then 3.1e-6
                    else 2.4e-6 * 0.25 * lognormal r 0.4 |> min 2.2e-6

                clock.Advance 3_000_000L

                t.Check(
                    TelemetryNames.ConformanceCase,
                    measured <= bound,
                    [
                        Attr.str TelemetryKeys.Kernel kernel
                        Attr.str TelemetryKeys.Case shape
                        Attr.float "measured" measured
                        Attr.float "bound" bound
                        Attr.bool "bit_exact" exact
                        Attr.float "ms" (0.05 * speed * lognormal r 0.6)
                    ]
                )

        let g = ctx.Telemetry.Child [ Attr.str TelemetryKeys.Component "executor" ]
        let mutable gpu = 1_000_000_000L

        let dispatch (kernel: string) (us: float) (attrs: Attr list) =
            let d = ns (us * 1000.0 * speed * lognormal r 0.05)
            clock.Advance(d + 900L)
            g.GpuInterval("gpu.dispatch", gpu, d, Attr.str TelemetryKeys.Kernel kernel :: attrs)
            gpu <- gpu + d + 800L

        for step in 0..1 do
            let s = Attr.int "step" step
            dispatch "embed_gather_bf16" 2.0 [ s ]

            for layer in 0 .. int qwen.Layers - 1 do
                for kernel, us in layerKernels do
                    let us =
                        if kernel = "attention_gqa_causal" then
                            us * (1.0 + 0.002 * float layer)
                        else
                            us

                    dispatch kernel us [ s; Attr.int TelemetryKeys.Layer layer ]

            dispatch "rmsnorm_rows" 2.1 [ s ]
            dispatch "matmul_bf16w" 310.0 [ s ]
            dispatch "argmax_rows" 21.0 [ s ]
            gpu <- gpu + 400_000L

    dx.Finish RunStatus.Failed
    vk.Finish RunStatus.Failed
    vk.Directory

// ---- cluster ------------------------------------------------------------------------------------------------------

/// A two-stage pipeline in C1's record names (ClusterProtocol, "Telemetry"): the coordinator writes every cluster.*
/// record, with the worker in `node`, as C1's coordinator does; the workers join as run-log nodes with their GPU
/// classes, and the spare writes its replay count. s0a is killed at step 9 while s0b lags, so the unit is reissued to
/// the spare, which replays the chain first; s0a is demoted and later leaves when its lease runs out.
let private cluster (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "cluster"

    use coord =
        RunContext.Create(
            options
                "cluster"
                "Two-stage pipeline, 2 replicas per stage and a spare, one node killed (synthetic)"
                root
                clock
                None
                (modelParams
                 @ [ "split", "14"; "replicas", "2"; "spares", "1"; "steps", "24"; "seed", "7" ])
        )

    let join name xclass =
        RunContext.Join(coord.Directory, NodeId.parse name, Some xclass, Some clock.Now)

    use s0a = join "s0a" vulkanClass
    use s0b = join "s0b" vulkanClass
    use s1a = join "s1a" d3d12Class
    use s1b = join "s1b" d3d12Class
    use spare = join "spare" vulkanClass

    let ct =
        coord.Telemetry.Child
            [
                Attr.str TelemetryKeys.Component "cluster.coordinator"
                Attr.str TelemetryKeys.Node "coord"
            ]

    let node (n: string) = Attr.str TelemetryKeys.Node n
    let stageAttr (s: int) = Attr.int TelemetryKeys.Stage s
    let names = [ "s0a"; "s0b"; "s1a"; "s1b"; "spare" ]
    let mutable epoch = 0

    for i, n in List.indexed names do
        clock.Set(epochNs + 200_000L + int64 i * 150_000L)
        epoch <- epoch + 1
        ct.Event("cluster.member.join", [ node n; Attr.str "incarnation" "1"; Attr.str "reason" "join" ])
        ct.Gauge("cluster.epoch", float epoch, "", [ Attr.str "reason" "join"; Attr.int "members" (i + 1) ])
        ct.Meter("cluster.bytes_sent", 70.0, TelemetryUnits.Bytes, [ Attr.str "kind" "membership" ])

    for i, n in List.indexed names do
        clock.Set(epochNs + 8_000_000L + int64 i * 400_000L)
        ct.Event("cluster.ready", [ node n; Attr.str "scope" "node" ])

    let samples = [| 0; 0 |]

    let deadline (stage: int) (value: float) =
        ct.Gauge("cluster.deadline", value, "ns", [ stageAttr stage; Attr.int "samples" samples[stage] ])

    let assign (at: int64) (stage: int) (n: string) (bytes: int) =
        clock.Set at
        ct.Counter("cluster.assign", 1.0, "count", [ stageAttr stage; node n; Attr.bool "retransmit" false ])
        ct.Meter("cluster.bytes_sent", float bytes, TelemetryUnits.Bytes, [ Attr.str "kind" "assign" ])

    let accept (step: int) (stage: int) (n: string) (start: int64) (finish: int64) =
        clock.Set finish

        ct.Event(
            "cluster.unit.done",
            [
                Attr.int "lane" 0
                stageAttr stage
                Attr.int "step" step
                node n
                Attr.int64 "start_ns" start
                Attr.int64 "end_ns" finish
                Attr.int64 "latency_ns" (finish - start)
            ]
        )

        ct.Histogram("cluster.unit.latency", float (finish - start), "ns", [ stageAttr stage; node n ])
        samples[stage] <- samples[stage] + 1

        if samples[stage] = 8 then
            deadline stage 250e6

    let confirm (stage: int) (n: string) (at: int64) (lagNs: int64) =
        clock.Set at
        ct.Counter("cluster.duplicate", 1.0, "count", [ stageAttr stage; node n ])
        ct.Histogram("cluster.lag", float lagNs, "ns", [ stageAttr stage; node n ])

    // Runs one stage of one step on its replicas: the first result is accepted, the other confirms it.
    let runStage (step: int) (stage: int) (replicas: string list) (start: int64) (baseNs: float) (bytes: int) =
        for n in replicas do
            assign start stage n bytes

        let timed =
            replicas
            |> List.map (fun n -> n, start + ns (baseNs * lognormal r 0.06))
            |> List.sortBy snd

        let first, firstEnd = List.head timed
        accept step stage first start firstEnd

        for n, finish in List.tail timed do
            confirm stage n finish (finish - firstEnd)

        firstEnd

    let killStep = 9
    let mutable s0Replicas = [ "s0a"; "s0b" ]
    let mutable leaveAt = Int64.MaxValue
    let mutable t = epochNs + 12_000_000L

    for step in 0..23 do
        let prefill = step = 0
        let base0 = if prefill then 41e6 else 9.2e6
        let base1 = if prefill then 46e6 else 11.1e6
        let tokens = if prefill then 512 else 1

        if t >= leaveAt then
            clock.Set leaveAt
            ct.Event("cluster.member.leave", [ node "s0a"; Attr.str "reason" "lease" ])
            epoch <- epoch + 1
            ct.Gauge("cluster.epoch", float epoch, "", [ Attr.str "reason" "lease"; Attr.int "members" 4 ])
            leaveAt <- Int64.MaxValue

        let accepted0 =
            if step <> killStep then
                runStage step 0 s0Replicas t base0 (tokens * 4)
            else
                // s0a dies 3 ms into the unit and s0b runs 45 times slower, so both miss the 250 ms deadline. The
                // coordinator reissues the unit to the spare, which first replays the 9 earlier units of the chain,
                // and drops s0a from the stage; the late result of s0b then confirms the result of the spare.
                for n in s0Replicas do
                    assign t 0 n 4

                clock.Set(t + 3_000_000L)
                ct.Counter("cluster.chaos", 1.0, "count", [ Attr.str "fault" "kill"; node "s0a" ])
                let missed = t + 250_000_000L
                clock.Set missed

                for n in s0Replicas do
                    ct.Counter("cluster.deadline.miss", 1.0, "count", [ stageAttr 0; node n ])

                ct.Counter(
                    "cluster.reissue",
                    1.0,
                    "count",
                    [ Attr.int "lane" 0; stageAttr 0; Attr.int "step" step; node "spare" ]
                )

                assign missed 0 "spare" 4

                ct.Counter(
                    "cluster.replay.served",
                    float step,
                    "count",
                    [ Attr.int "lane" 0; stageAttr 0; node "spare" ]
                )

                ct.Meter("cluster.bytes_sent", float (step * 4096), TelemetryUnits.Bytes, [ Attr.str "kind" "replay" ])
                ct.Event("cluster.demote", [ stageAttr 0; node "s0a" ])
                let replayed = missed + ns (float step * 4.8e6 * lognormal r 0.05)
                clock.Set replayed

                spare.Telemetry.Counter(
                    "cluster.replay.units",
                    float step,
                    "count",
                    [
                        Attr.str TelemetryKeys.Component "cluster.node"
                        node "spare"
                        Attr.int "lane" 0
                        stageAttr 0
                    ]
                )

                let finish = replayed + ns (base0 * lognormal r 0.06)
                accept step 0 "spare" t finish
                let late = t + 420_000_000L
                confirm 0 "s0b" late (late - finish)
                deadline 0 1.32e9
                s0Replicas <- [ "s0b"; "spare" ]
                leaveAt <- t + 500_000_000L
                finish

        if step = killStep + 9 then
            clock.Set accepted0
            deadline 0 250e6

        let start1 = accepted0 + 1_100_000L

        let accepted1 =
            runStage step 1 [ "s1a"; "s1b" ] start1 base1 (tokens * int qwen.Hidden * 4)

        t <- accepted1 + 300_000L

    clock.Set(t + 2_000_000L)

    ct.Event(
        "cluster.run",
        [
            Attr.str "status" "completed"
            Attr.int "units" 48
            Attr.int "epoch" epoch
            Attr.int "reissues" 1
            Attr.int "duplicates" 48
            Attr.int "divergences" 0
            Attr.int "replays" 1
            Attr.int "cross_checked" 48
            Attr.int "unconfirmed" 0
            Attr.str "run_digest" (Hash.sha256Utf8 "synthetic cluster run").Hex
            Attr.int64 "end_ns" (clock.Now())
        ]
    )

    for ctx in [ s1b; s1a; spare; s0b ] do
        ctx.Finish RunStatus.Ok

    // s0a was killed: it never writes its trailer. Its file simply stops (the reader reports it).
    coord.Finish RunStatus.Ok
    s0a.Sink.Close()
    coord.Directory

// ---- gate1 --------------------------------------------------------------------------------------------------------

let private gate1 (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "gate1"

    use run =
        RunContext.Create(
            options
                "gate1"
                "Fresh clone of main (synthetic)"
                root
                clock
                (Some(cpuClass ExecutionBackend.CpuF32))
                (modelParams @ [ "goldens", "qwen3-0.6b/g1"; "tolerances", "v1" ])
        )

    let t = run.Telemetry.Child [ Attr.str TelemetryKeys.Component "gate1" ]

    let check group name backend detail =
        clock.Advance(ns (2.5e9 * lognormal r 0.8))

        t.Check(
            "gate1.check",
            true,
            [
                Attr.str "group" group
                Attr.str "check" name
                if backend <> "" then
                    Attr.str TelemetryKeys.Backend backend
                Attr.str "detail" detail
            ]
        )

    check "environment" "SDK 10.0.402 pinned" "" "global.json"
    check "environment" "driver matches the manifest" "" "616.92"
    check "environment" "DXC 1.9.2609.5" "" "--version verified"
    check "environment" "weights SHA256SUMS" "" "model.safetensors, tokenizer.json"
    check "tokenizer" "corpus exact" "" "2,048 / 2,048 cases"
    check "cpuref" "cpu64 vs torch64p" "" "max rel_l2 3.1e-12 <= 1e-10"
    check "cpuref" "cpu32 vs torch32" "" "max rel_l2 6.2e-06 <= 2e-05"
    check "cpuref" "greedy ids" "" "8 prompts x 64 steps identical"

    for b in [ "cpu-device"; "vulkan"; "d3d12" ] do
        check "conformance" "conformance matrix" b "every catalog kernel within KernelBounds"

    for b in [ "vulkan"; "d3d12" ] do
        check "backend" "logits within tolerance" b "8 prompts"
        check "backend" "greedy ids identical" b "8 prompts x 64 steps"
        check "backend" "teacher-forced agreement" b "where margin >= 2 tau"

    check "cluster" "vulkan|vulkan split bit-identical" "vulkan" "digest equal"
    check "cluster" "d3d12|d3d12 split bit-identical" "d3d12" "digest equal"
    check "cluster" "vulkan|d3d12 split within tolerance" "" "rel_l2 4.1e-05 <= 1e-04"
    check "cluster" "GPU chaos 25 seeds bit-identical" "" "25 / 25"
    check "determinism" "second run identical" "" "every digest equal"

    for b, scale in [ "vulkan", 1.0; "d3d12", 1.15 ] do
        for p in 1..8 do
            let case = Invariant.format $"g{p:D2}"

            for step in 0..63 do
                let v = 1.1e-5 * scale * (1.0 + 0.004 * float step) * lognormal r 0.25
                clock.Advance 1_000_000L

                t.Check(
                    TelemetryNames.GoldenCompare,
                    v <= 1e-4,
                    [
                        Attr.str "pair" "gpu-vs-torch32"
                        Attr.str TelemetryKeys.Backend b
                        Attr.str TelemetryKeys.Case case
                        Attr.str "tap" "logits"
                        Attr.str "metric" "rel_l2"
                        Attr.int "step" step
                        Attr.float "measured" v
                        Attr.float "bound" 1e-4
                    ]
                )

    run.Finish RunStatus.Ok
    run.Directory

// ---- training -----------------------------------------------------------------------------------------------------

let private training (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "training"

    use run =
        RunContext.Create(
            options
                "training"
                "Pretraining curriculum, 600 steps (synthetic, Phase 2 preview)"
                root
                clock
                (Some vulkanClass)
                (modelParams @ [ "optimizer", "AdamW"; "lr", "3e-4"; "batch_tokens", "65536" ])
        )

    let t = run.Telemetry.Child [ Attr.str TelemetryKeys.Component "train" ]

    let stages =
        [ "pretrain-csharp", 0, 249; "pretrain-il", 250, 449; "alignment", 450, 599 ]

    let stageOf step =
        stages |> List.find (fun (_, lo, hi) -> step >= lo && step <= hi)

    for step in 0..599 do
        let name, lo, _ = stageOf step
        let since = float (step - lo)
        clock.Advance(ns (410e6 * lognormal r 0.03))

        let floor' =
            1.55 - 0.12 * float (stages |> List.findIndex (fun (n, _, _) -> n = name))

        let bump = if lo > 0 then 0.55 * DetMath.exp (-since / 18.0) else 0.0

        let decay =
            if lo = 0 then
                2.9 * DetMath.exp (-float step / 70.0)
            else
                0.0

        let train =
            floor'
            + decay
            + bump
            + 0.25 * DetMath.exp (-float step / 300.0)
            + 0.035 * normal r

        let stageAttr = Attr.str TelemetryKeys.Stage name
        t.Gauge("train.loss", train, "nat", [ Attr.int "step" step; Attr.str "split" "train"; stageAttr ])

        let lr =
            3e-4 * (min 1.0 (float (step + 1) / 50.0)) * (1.0 - 0.6 * float step / 600.0)

        t.Gauge("train.lr", lr, TelemetryUnits.NoUnit, [ Attr.int "step" step ])

        if step % 25 = 24 then
            let eval = train + 0.08 + 0.02 * normal r
            t.Gauge("train.loss", eval, "nat", [ Attr.int "step" step; Attr.str "split" "eval"; stageAttr ])

        if step % 20 = 0 then
            for layer in 0 .. int qwen.Layers - 1 do
                let la = Attr.int TelemetryKeys.Layer layer
                let depth = 1.0 + 0.6 * float (27 - layer) / 27.0
                let spike = if since < 6.0 && lo > 0 then 2.4 else 1.0

                let g = 0.85 * depth * DetMath.exp (-float step / 900.0) * spike * lognormal r 0.22

                t.Gauge("train.grad_norm", g, TelemetryUnits.NoUnit, [ Attr.int "step" step; la ])

                t.Gauge(
                    "train.update_ratio",
                    lr * 4.0 * g * lognormal r 0.35,
                    TelemetryUnits.NoUnit,
                    [ Attr.int "step" step; la ]
                )

        if step % 100 = 99 then
            for layer in 0 .. int qwen.Layers - 1 do
                let la = Attr.int TelemetryKeys.Layer layer
                let l = float layer

                for site, v in
                    [
                        "resid", 1.0 + 0.32 * l + 0.004 * l * l
                        "attn_out", 0.55 + 0.018 * l
                        "mlp_out", 0.7 + 0.06 * l + (if layer = 27 then 1.4 else 0.0)
                    ] do
                    t.Gauge(
                        "train.act_rms",
                        v * lognormal r 0.03,
                        TelemetryUnits.NoUnit,
                        [ Attr.int "step" step; la; Attr.str "site" site ]
                    )

    run.Finish RunStatus.Ok
    run.Directory

// ---- routing ------------------------------------------------------------------------------------------------------

let private routing (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "routing"

    use run =
        RunContext.Create(
            options
                "routing"
                "Roslyn expert routing through training (synthetic, Phase 3 preview)"
                root
                clock
                (Some vulkanClass)
                [ "experts", "6 routed + none"; "tokens_per_checkpoint", "4096" ]
        )

    let t = run.Telemetry.Child [ Attr.str TelemetryKeys.Component "router" ]

    // Where each routed expert is preferred: a centre layer and a strength.
    let experts =
        [
            "GetDiagnostics", 6.0, 1.0
            "AnalyzeDataFlow", 12.0, 0.8
            "ReplaceNode", 18.0, 0.9
            "InsertNodesAfter", 20.0, 0.6
            "CompileToIL", 24.0, 1.1
            "DecompileIL", 15.0, 0.5
            "none", -1.0, 0.0
        ]

    let tokens = 4096.0

    for step in [ 0; 500; 1000; 1500; 2000; 2500 ] do
        clock.Advance 60_000_000_000L
        let sharp = 0.15 + 0.95 * float step / 2500.0

        for layer in 0 .. int qwen.Layers - 1 do
            let l = float layer

            let logits =
                experts
                |> List.map (fun (name, centre, strength) ->
                    let pref =
                        if centre < 0.0 then
                            0.9 + 0.04 * l
                        else
                            strength * (2.0 - abs (l - centre) / 4.0)

                    name, sharp * pref + 0.15 * normal r)

            let top = logits |> List.map snd |> List.max
            let weights = logits |> List.map (fun (n, z) -> n, DetMath.exp (z - top))
            let total = weights |> List.sumBy snd
            let probs = weights |> List.map (fun (n, w) -> n, w / total)

            let ent =
                probs |> List.sumBy (fun (_, p) -> if p > 0.0 then -p * DetMath.ln p else 0.0)

            let la = Attr.int TelemetryKeys.Layer layer

            for name, p in probs do
                t.Counter(
                    "route.usage",
                    Math.Round(tokens * p),
                    TelemetryUnits.Tokens,
                    [ Attr.int "step" step; la; Attr.str "expert" name ]
                )

            t.Gauge("route.entropy", ent, "nat", [ Attr.int "step" step; la ])

    run.Finish RunStatus.Ok
    run.Directory

// ---- thinking -----------------------------------------------------------------------------------------------------

let private thinking (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "thinking"

    use run =
        RunContext.Create(
            options
                "thinking"
                "Roslyn calls while writing one method (synthetic, Phase 3 preview)"
                root
                clock
                (Some vulkanClass)
                [ "task", "level 3: parse a CSV line into a record"; "tokens", "220" ]
        )

    let t = run.Telemetry.Child [ Attr.str TelemetryKeys.Component "roslyn" ]

    let apis =
        [
            "SemanticModel.GetTypeInfo", 0.35, 0.06
            "SemanticModel.GetSymbolInfo", 0.3, 0.05
            "SemanticModel.LookupSymbols", 0.6, 0.12
            "Compilation.GetDiagnostics", 3.5, 0.2
            "SemanticModel.AnalyzeDataFlow", 1.4, 0.1
            "SyntaxNode.ReplaceNode", 0.9, 0.05
            "Compilation.Emit", 18.0, 0.6
        ]

    let step kind detail i =
        t.Event("think.step", [ Attr.int "step" i; Attr.str "kind" kind; Attr.str "detail" detail ])

    step "plan" "goal: ParseLine(string) -> Row; stubs Split, ParseField" 0
    let mutable token = 0

    for i in 0..71 do
        token <- token + 2 + int (r.NextBelow 4u)
        let api, ms, _ = apis[int (r.NextBelow(uint32 apis.Length))]

        let api, ms =
            match i with
            | 26
            | 35 -> "Compilation.GetDiagnostics", 3.5
            | 44 -> "Compilation.Emit", 18.0
            | 58 -> "SemanticModel.AnalyzeDataFlow", 1.4
            | _ when i % 9 = 8 -> "Compilation.GetDiagnostics", 3.5
            | _ -> api, ms

        let layer = 8 + 4 * int (r.NextBelow 4u)
        clock.Advance(ns (2.2e6 * lognormal r 0.5))

        let outcome, detail, dur =
            match i with
            | 26 -> "diagnostic", "CS0103: The name 'fields' does not exist", ms
            | 35 -> "diagnostic", "CS1503: cannot convert from 'string' to 'int'", ms
            | 44 -> "timeout", "Emit over the 250 ms budget", 250.0
            | 58 -> "error", "InvalidOperationException in AnalyzeDataFlow (node not in tree)", ms
            | _ -> "ok", "", ms

        use _ =
            t.Span(
                "roslyn.call",
                [
                    Attr.str "api" api
                    Attr.int TelemetryKeys.Layer layer
                    Attr.int "token" token
                    Attr.str "outcome" outcome
                    if detail <> "" then
                        Attr.str "detail" detail
                ]
            )

        clock.Advance(ns (dur * 1e6 * lognormal r 0.3))

        if i = 27 then
            step "revise" "declare 'fields' from Split before use" 1
        elif i = 36 then
            step "revise" "int.Parse the age field" 2
        elif i = 60 then
            step "check" "all tests compile; run hidden tests" 3

    run.Finish RunStatus.Ok
    run.Directory

// ---- goldens ------------------------------------------------------------------------------------------------------

/// A tokenizer golden run with --check, in the records of tools/goldens (ReferenceOracles, "Telemetry").
let private goldens (root: string) =
    let clock = VirtualClock epochNs
    let r = rngFor "goldens"

    use run =
        RunContext.Create(
            options
                "goldens"
                "make_tokenizer --check (synthetic)"
                root
                clock
                None
                [ "tool", "make_tokenizer"; "set", "goldens/tokenizer/g1" ]
        )

    let t =
        run.Telemetry.Child [ Attr.str TelemetryKeys.Component "goldens"; Attr.str "tool" "make_tokenizer" ]

    let set = Attr.str "set" "tokenizer"

    let categories =
        [
            "added", 520
            "astral", 320
            "casefold", 200
            "csharp", 360
            "decode", 520
            "digits", 130
            "emoji", 440
            "il", 360
            "multilingual", 190
            "nfc", 170
            "unicode-space", 400
            "utf8", 140
            "whitespace", 370
        ]
        |> List.map (fun (c, n) -> c, n + int (40.0 * (r.NextF64() - 0.5)))

    let total = categories |> List.sumBy snd

    do
        use _ = t.Span("goldens.tokenizer.build", [ set ])
        clock.Advance(ns (4.7e9 * lognormal r 0.05))

    do
        use _ = t.Span("goldens.tokenizer.crosscheck", [ set ])
        clock.Advance(ns (0.6e9 * lognormal r 0.05))

    t.Check("goldens.double_run_identical", true, [ set; Attr.int "threads" 8 ])

    t.Check(
        "goldens.tokenizer.transformers",
        true,
        [
            Attr.int "agree" (total - 520)
            Attr.int "differ" 0
            Attr.str "tokenizer_class" "Qwen2Tokenizer"
        ]
    )

    for c, n in categories do
        t.Gauge("goldens.tokenizer.cases", float n, "count", [ Attr.str "category" c ])

    for kind, n in
        [
            "decode", 520
            "encode", total - 520 - 150
            "encode_bytes", 142
            "encode_reject", 8
        ] do
        t.Gauge("goldens.tokenizer.case_kinds", float n, "count", [ Attr.str "kind" kind ])

    let files =
        [
            "goldens/tokenizer/g1/cases.jsonl", float total * 172.0
            "goldens/tokenizer/g1/manifest.json", 12_400.0
        ]

    for f, b in files do
        t.Gauge("goldens.file_bytes", Math.Round b, TelemetryUnits.Bytes, [ Attr.str "file" f ])

    t.Gauge(
        "goldens.set_bytes",
        files |> List.sumBy (fun (_, b) -> Math.Round b),
        TelemetryUnits.Bytes,
        [ set; Attr.int64 "budget" 2097152L ]
    )

    t.Check("goldens.regen_identical", true, [ set ])
    run.Finish RunStatus.Ok
    run.Directory

/// The template names that have a synthetic generator, in report order.
let kinds =
    [
        "parity"
        "inference"
        "kernels"
        "cluster"
        "gate1"
        "goldens"
        "training"
        "routing"
        "thinking"
    ]

/// Writes the synthetic run for `kind` under `runsRoot` (as runsRoot/synthetic-<kind>, which must not exist) and
/// returns its folder.
let write (kind: string) (runsRoot: string) : string =
    match kind with
    | "parity" -> parity runsRoot
    | "inference" -> inference runsRoot
    | "kernels" -> kernels runsRoot
    | "cluster" -> cluster runsRoot
    | "gate1" -> gate1 runsRoot
    | "goldens" -> goldens runsRoot
    | "training" -> training runsRoot
    | "routing" -> routing runsRoot
    | "thinking" -> thinking runsRoot
    | other ->
        invalidArg "kind" (sprintf "no synthetic generator for '%s' (known: %s)" other (String.concat ", " kinds))
