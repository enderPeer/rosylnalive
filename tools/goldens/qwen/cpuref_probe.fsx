// The CpuRef calibration probe of goldens/qwen3-0.6b/g1 (G2): runs O2's CpuRef ops, composed into the Qwen3-0.6B
// forward exactly as tests/Lrm.Tensor.Tests/TinyModel.fs composes them for the tiny model, on the gate prompts, and
// compares them with the goldens under the calibrated tolerances. It is evidence that the tolerances hold for an
// independent fp32 implementation in sequential FMA order, not the gate1-cpu acceptance (M2b, with M2a's forward).
//
//     dotnet fsi tools/goldens/qwen/cpuref_probe.fsx [--set <folder>] [--data <root>] [--threads 8] [--long]
//
// Per gate prompt: CpuRef F32 against torch32 and torch64p (logits relL2, max |delta logit| and KL; every tap for the
// taps prompts), CpuRef F64 against torch64p (relL2), 64 greedy steps in both modes (greedy_match on the gated steps),
// and teacher-forced argmax on the tf_* sequences where the reference margin is at least 2 * tau_logit. With --long,
// the long prompts' final_norm.out and stored logit rows too. Every comparison is a golden.compare Check in a parity
// run log (n0 cpu-f64, n1 cpu-f32) that `lrm report <runId>` renders; the exit code is 1 if one fails.

#r "../../../artifacts/bin/Lrm.Core/release/Lrm.Core.dll"
#r "../../../artifacts/bin/Lrm.Tensor/release/Lrm.Tensor.dll"
#r "../../../artifacts/bin/Lrm.Tensor.CpuReference/release/Lrm.Tensor.CpuReference.dll"
#r "../../../artifacts/bin/Lrm.Telemetry/release/Lrm.Telemetry.dll"

open System
open System.Collections.Generic
open System.IO
open System.Numerics
open System.Text.Json
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Telemetry

Lrm.Core.Process.init ()

let argv = fsi.CommandLineArgs |> Array.skip 1

let opt (name: string) =
    argv |> Array.tryFindIndex ((=) name) |> Option.map (fun i -> argv[i + 1])

let setDir =
    opt "--set"
    |> Option.defaultValue (Path.Combine(__SOURCE_DIRECTORY__, "..", "..", "..", "goldens", "qwen3-0.6b", "g1"))

let threads = opt "--threads" |> Option.map int |> Option.defaultValue 8
let withLong = argv |> Array.contains "--long"
let exec = CpuExec.WithThreads threads

let set =
    match opt "--data" with
    | Some root -> Goldens.Open(setDir, root)
    | None -> Goldens.Open setDir

let spec = ToleranceSpec.load (Path.Combine(set.Dir, "tolerances.json"))
let greedyRules = spec.Greedy.Value
let tau = greedyRules.GetProperty("tau_logit").GetDouble()
let tfMargin = greedyRules.GetProperty("teacher_forced_min_margin").GetDouble()

let say (s: string) =
    printfn "%s %s" (DateTime.Now.ToString("HH:mm:ss", Globalization.CultureInfo.InvariantCulture)) s

// ----- the model -----------------------------------------------------------------------------------------------------

type LayerWeights =
    {
        InputNorm: HostTensor<uint16>
        Q: HostTensor<uint16>
        K: HostTensor<uint16>
        V: HostTensor<uint16>
        O: HostTensor<uint16>
        QNorm: HostTensor<uint16>
        KNorm: HostTensor<uint16>
        PostNorm: HostTensor<uint16>
        Gate: HostTensor<uint16>
        Up: HostTensor<uint16>
        Down: HostTensor<uint16>
    }

let modelDir = LrmPaths.model "Qwen3-0.6B"

let config =
    JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDir, "config.json"))).RootElement

let cfgInt (k: string) = config.GetProperty(k).GetInt32()
let layers = cfgInt "num_hidden_layers"
let heads = cfgInt "num_attention_heads"
let kvHeads = cfgInt "num_key_value_heads"
let headDim = cfgInt "head_dim"
let epsilon = config.GetProperty("rms_norm_eps").GetDouble()

let modelSha =
    File.ReadAllLines(Path.Combine(modelDir, "SHA256SUMS"))
    |> Array.pick (fun l ->
        let parts = l.Split([| " *" |], StringSplitOptions.None)

        if parts.Length = 2 && parts[1] = "model.safetensors" then
            ContentId.tryParse parts[0]
        else
            None)

let path = Path.Combine(modelDir, "model.safetensors")

let weightsFile =
    SafetensorsFile.OpenVerified(path, FileInfo(path).Length, modelSha)

let w (name: string) = weightsFile.ReadBf16 name

let layerWeights =
    Array.init layers (fun i ->
        let p = Invariant.format $"model.layers.{i}."

        {
            InputNorm = w (p + "input_layernorm.weight")
            Q = w (p + "self_attn.q_proj.weight")
            K = w (p + "self_attn.k_proj.weight")
            V = w (p + "self_attn.v_proj.weight")
            O = w (p + "self_attn.o_proj.weight")
            QNorm = w (p + "self_attn.q_norm.weight")
            KNorm = w (p + "self_attn.k_norm.weight")
            PostNorm = w (p + "post_attention_layernorm.weight")
            Gate = w (p + "mlp.gate_proj.weight")
            Up = w (p + "mlp.up_proj.weight")
            Down = w (p + "mlp.down_proj.weight")
        })

let embedW = w "model.embed_tokens.weight"
let normW = w "model.norm.weight"
weightsFile.Dispose()

let rope =
    use r = set.OpenSafetensors "rope.safetensors"
    CpuRef.ropeTable (r.ReadU32 "inv_freq.bits").Data 8192

let shape =
    {
        QHeads = heads
        KvHeads = kvHeads
        HeadDim = headDim
    }

type Cache<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType> =
    {
        Keys: HostTensor<'T>[]
        Values: HostTensor<'T>[]
    }

let newCache<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (maxPos: int)
    : Cache<'T> =
    let dtype =
        if typeof<'T> = typeof<float32> then
            DType.F32
        else
            DType.F64

    let s = Shape.matrix maxPos (kvHeads * headDim)

    {
        Keys = Array.init layers (fun _ -> HostTensor.zeros<'T> dtype s)
        Values = Array.init layers (fun _ -> HostTensor.zeros<'T> dtype s)
    }

let rowsOf<'T when 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (t: HostTensor<'T>)
    (lo: int)
    (hi: int)
    =
    HostTensor.create t.DType (Shape.matrix (hi - lo) t.Cols) (Array.sub t.Data (lo * t.Cols) ((hi - lo) * t.Cols))

/// One forward at positions pos0 .. pos0 + T - 1 (TinyModel.forward with Qwen3-0.6B's weights): writes K and V into
/// the cache, records every ActivationNames tap into `taps` when given, and returns the logits.
let forward<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (cache: Cache<'T>)
    (pos0: int)
    (ids: int[])
    (taps: Dictionary<string, HostTensor<'T>> option)
    : HostTensor<'T> =
    let eps = 'T.CreateTruncating epsilon
    let scale = CpuRef.attentionScale<'T> headDim

    let tap (name: string) (t: HostTensor<'T>) =
        taps |> Option.iter (fun d -> d[name] <- t)
        t

    let mutable x = tap "embed.out" (CpuRef.embed<'T> exec embedW ids)

    for i in 0 .. layers - 1 do
        let lw = layerWeights[i]
        let p = Invariant.format $"L{i}."
        let h = tap (p + "input_layernorm.out") (CpuRef.rmsNorm exec x lw.InputNorm eps)
        let q = tap (p + "attn.q_proj.out") (CpuRef.matmulBf16W exec h lw.Q)
        let k = tap (p + "attn.k_proj.out") (CpuRef.matmulBf16W exec h lw.K)
        let v = tap (p + "attn.v_proj.out") (CpuRef.matmulBf16W exec h lw.V)
        let qn = tap (p + "attn.q_norm.out") (CpuRef.rmsNorm exec q lw.QNorm eps)
        let kn = tap (p + "attn.k_norm.out") (CpuRef.rmsNorm exec k lw.KNorm eps)
        let qr = tap (p + "attn.q_rope") (CpuRef.ropeApply exec rope qn heads pos0)
        let kr = tap (p + "attn.k_rope") (CpuRef.ropeApply exec rope kn kvHeads pos0)
        CpuRef.copyRows kr 0 cache.Keys[i] pos0 ids.Length
        CpuRef.copyRows v 0 cache.Values[i] pos0 ids.Length

        tap (p + "attn.v_cache") (rowsOf cache.Values[i] pos0 (pos0 + ids.Length))
        |> ignore

        let o =
            tap (p + "attn.o_proj.in") (CpuRef.attention exec qr cache.Keys[i] cache.Values[i] shape pos0 scale)

        let oo = tap (p + "attn.o_proj.out") (CpuRef.matmulBf16W exec o lw.O)
        let r = tap (p + "resid_mid") (CpuRef.add exec x oo)

        let h2 =
            tap (p + "post_attention_layernorm.out") (CpuRef.rmsNorm exec r lw.PostNorm eps)

        let g = tap (p + "mlp.gate.out") (CpuRef.matmulBf16W exec h2 lw.Gate)
        let u = tap (p + "mlp.up.out") (CpuRef.matmulBf16W exec h2 lw.Up)
        let d = tap (p + "mlp.down.in") (CpuRef.siluMul exec g u)
        let dd = tap (p + "mlp.down.out") (CpuRef.matmulBf16W exec d lw.Down)
        x <- tap (p + "out") (CpuRef.add exec r dd)

    let f = tap "final_norm.out" (CpuRef.rmsNorm exec x normW eps)
    tap "logits" (CpuRef.matmulBf16W exec f embedW)

/// The canonical greedy schedule: one prefill, then single-token decode steps; EOS is never checked.
let greedy<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (prompt: int[])
    (steps: int)
    =
    let cache = newCache<'T>(prompt.Length + steps)
    let logits = forward cache 0 prompt None
    let mutable next = (CpuRef.argmax (rowsOf logits (logits.Rows - 1) logits.Rows))[0]
    let ids = ResizeArray [ next ]

    for s in 1 .. steps - 1 do
        let l = forward cache (prompt.Length + s - 1) [| next |] None
        next <- (CpuRef.argmax l)[0]
        ids.Add next

    ids.ToArray()

// ----- the run log ---------------------------------------------------------------------------------------------------

let runF64 =
    RunContext.Create(
        { RunOptions.create "parity" "CpuRef calibration probe (G2): CpuRef F32 and F64 on the Qwen3-0.6B gate prompts" with
            ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuF64)
            Params =
                [
                    "golden", "goldens/qwen3-0.6b/g1"
                    "source", "tools/goldens/qwen/cpuref_probe.fsx (G2 calibration evidence, not gate1-cpu)"
                    "nodes", "n0 cpu-f64, n1 cpu-f32"
                    "threads", string threads
                ]
        }
    )

let runF32 =
    RunContext.Join(
        runF64.Directory,
        { JoinOptions.create (NodeId.parse "n1") with
            ExecutionClass = Some(ExecutionClass.forCpu ExecutionBackend.CpuF32)
        }
    )

say ("run log: " + runF64.Directory)
let mutable failures = 0

let record (run: RunContext) (f: unit -> CheckResult) =
    use _ = run.Install()
    let r = f ()

    if not r.Passed then
        failures <- failures + 1
        say (sprintf "FAIL %O %A" r r.Attrs)

let bound (pair: TolerancePair) (tap: string) (metric: ToleranceMetric) =
    spec.Bound(pair, ActivationNames.kind tap, metric)

let attrs (pair: TolerancePair) (prompt: string) (tap: string) =
    [
        Attr.str "pair" (TolerancePair.name pair)
        Attr.str "prompt" prompt
        Attr.str "tap" tap
        Attr.int "layer" (ActivationNames.layer tap)
    ]

let compareLogits
    (run: RunContext)
    pair
    prompt
    (cand: #IHostTensor)
    (refr: #IHostTensor)
    (metrics: ToleranceMetric list)
    =
    for m in metrics do
        let b = bound pair "logits" m
        let a = attrs pair prompt "logits"

        match m with
        | ToleranceMetric.RelL2 -> record run (fun () -> Compare.relL2 b a cand refr)
        | ToleranceMetric.MaxAbs -> record run (fun () -> Compare.maxAbs b a cand refr)
        | ToleranceMetric.Kl -> record run (fun () -> Compare.kl b a cand refr)

let greedyMatch (run: RunContext) pair prompt (ids: int[]) (want: int64[]) (gated: int) =
    let matched =
        Seq.zip ids want
        |> Seq.takeWhile (fun (a, b) -> int64 a = b)
        |> Seq.length
        |> min gated

    record run (fun () ->
        Compare.check
            "greedy_match"
            BoundOp.AtLeast
            (float matched)
            (float gated)
            (int64 gated)
            (attrs pair prompt "logits"))

/// Teacher-forced argmax: mismatching rows among those whose reference margin is at least 2 * tau_logit.
let teacherForced (run: RunContext) pair prompt (logits: HostTensor<'T>) (argmax: int64[]) (margin: float[]) =
    let mine = CpuRef.argmax logits
    let mutable rows, wrong = 0, 0

    for r in 0 .. argmax.Length - 1 do
        if margin[r] >= tfMargin then
            rows <- rows + 1

            if int64 mine[r] <> argmax[r] then
                wrong <- wrong + 1

    record run (fun () ->
        Compare.check
            "teacher_forced_mismatch"
            BoundOp.AtMost
            (float wrong)
            0.0
            (int64 rows)
            (attrs pair prompt "logits" @ [ Attr.float "min_margin" tfMargin ]))

// ----- the probe -----------------------------------------------------------------------------------------------------

let prompts = set.Manifest.GetProperty("prompts").EnumerateArray() |> List.ofSeq
let greedyFile = set.OpenSafetensors "greedy.safetensors"

for p in prompts do
    let name = p.GetProperty("name").GetString()
    let role = p.GetProperty("role").GetString()

    let ids =
        p.GetProperty("ids").EnumerateArray() |> Seq.map _.GetInt32() |> Array.ofSeq

    let clock = System.Diagnostics.Stopwatch.StartNew()

    if role = "gate" then
        let taps = p.GetProperty("dumps").GetString() = "taps"
        let gated = p.GetProperty("gated_steps").GetInt32()
        use g32 = set.OpenSafetensors(Invariant.format $"fwd_f32.{name}.safetensors")
        use g64 = set.OpenSafetensors(Invariant.format $"fwd_f64p.{name}.safetensors")
        // CpuRef F32 against torch32 and torch64p
        let t32 = Dictionary<string, HostTensor<float32>>(StringComparer.Ordinal)

        let l32 =
            forward (newCache<float32> ids.Length) 0 ids (if taps then Some t32 else None)

        let ref32 = g32.ReadF32 "logits"
        let ref64 = g64.ReadF64 "logits"
        let all = [ ToleranceMetric.RelL2; ToleranceMetric.MaxAbs; ToleranceMetric.Kl ]
        compareLogits runF32 TolerancePair.Cpu32VsTorch32 name l32 ref32 all
        compareLogits runF32 TolerancePair.Cpu32VsTorch64p name l32 ref64 all

        if taps then
            for KeyValue(tap, t) in t32 do
                if tap <> "logits" then
                    record runF32 (fun () ->
                        Compare.relL2
                            (bound TolerancePair.Cpu32VsTorch32 tap ToleranceMetric.RelL2)
                            (attrs TolerancePair.Cpu32VsTorch32 name tap)
                            t
                            (g32.ReadF32 tap))

                    record runF32 (fun () ->
                        Compare.relL2
                            (bound TolerancePair.Cpu32VsTorch64p tap ToleranceMetric.RelL2)
                            (attrs TolerancePair.Cpu32VsTorch64p name tap)
                            t
                            (g64.ReadF64 tap))

        t32.Clear()
        // CpuRef F64 against torch64p
        let t64 = Dictionary<string, HostTensor<float>>(StringComparer.Ordinal)

        let l64 =
            forward (newCache<float> ids.Length) 0 ids (if taps then Some t64 else None)

        compareLogits runF64 TolerancePair.Cpu64VsTorch64p name l64 ref64 [ ToleranceMetric.RelL2 ]

        if taps then
            for KeyValue(tap, t) in t64 do
                if tap <> "logits" then
                    record runF64 (fun () ->
                        Compare.relL2
                            (bound TolerancePair.Cpu64VsTorch64p tap ToleranceMetric.RelL2)
                            (attrs TolerancePair.Cpu64VsTorch64p name tap)
                            t
                            (g64.ReadF64 tap))

        t64.Clear()
        // greedy, 64 steps, against each oracle's ids on the gated steps
        let want32 = (greedyFile.ReadI64(name + ".ids.torch32")).Data
        let want64 = (greedyFile.ReadI64(name + ".ids.torch64p")).Data
        greedyMatch runF32 TolerancePair.Cpu32VsTorch32 name (greedy<float32> ids 64) want32 gated
        greedyMatch runF64 TolerancePair.Cpu64VsTorch64p name (greedy<float> ids 64) want64 gated
        // teacher-forced argmax
        use tf32 = set.OpenSafetensors(Invariant.format $"tf_f32.{name}.safetensors")
        use tf64 = set.OpenSafetensors(Invariant.format $"tf_f64p.{name}.safetensors")
        let teacher = (tf32.ReadI64 "input_ids").Data |> Array.map int
        let tl32 = forward (newCache<float32> teacher.Length) 0 teacher None

        teacherForced
            runF32
            TolerancePair.Cpu32VsTorch32
            name
            tl32
            (tf32.ReadI64 "argmax").Data
            (tf32.ReadF64 "margin").Data

        let tl64 = forward (newCache<float> teacher.Length) 0 teacher None

        teacherForced
            runF64
            TolerancePair.Cpu64VsTorch64p
            name
            tl64
            (tf64.ReadI64 "argmax").Data
            (tf64.ReadF64 "margin").Data

        say (Invariant.format $"{name} ({ids.Length} tokens) in {clock.Elapsed.TotalSeconds:F0} s")
    elif withLong then
        use g32 = set.OpenSafetensors(Invariant.format $"long_f32.{name}.safetensors")
        use g64 = set.OpenSafetensors(Invariant.format $"long_f64p.{name}.safetensors")
        let rows = (g32.ReadI64 "logits.rows").Data |> Array.map int

        let pick (l: HostTensor<'T>) =
            HostTensor.create
                l.DType
                (Shape.matrix rows.Length l.Cols)
                (rows |> Array.collect (fun r -> Array.sub l.Data (r * l.Cols) l.Cols))

        let t32 = Dictionary<string, HostTensor<float32>>(StringComparer.Ordinal)
        let l32 = forward (newCache<float32> ids.Length) 0 ids (Some t32)
        let f32 = t32["final_norm.out"]
        t32.Clear()
        let s32 = pick l32

        compareLogits
            runF32
            TolerancePair.Cpu32VsTorch32
            name
            s32
            (g32.ReadF32 "logits")
            [ ToleranceMetric.RelL2; ToleranceMetric.MaxAbs; ToleranceMetric.Kl ]

        compareLogits
            runF32
            TolerancePair.Cpu32VsTorch64p
            name
            s32
            (g64.ReadF64 "logits")
            [ ToleranceMetric.RelL2; ToleranceMetric.MaxAbs; ToleranceMetric.Kl ]

        record runF32 (fun () ->
            Compare.relL2
                (bound TolerancePair.Cpu32VsTorch32 "final_norm.out" ToleranceMetric.RelL2)
                (attrs TolerancePair.Cpu32VsTorch32 name "final_norm.out")
                f32
                (g32.ReadF32 "final_norm.out"))

        teacherForced
            runF32
            TolerancePair.Cpu32VsTorch32
            name
            l32
            (g32.ReadI64 "argmax").Data
            (g32.ReadF64 "margin").Data

        say (Invariant.format $"{name} ({ids.Length} tokens) in {clock.Elapsed.TotalSeconds:F0} s")

greedyFile.Dispose()
let status = if failures = 0 then RunStatus.Ok else RunStatus.Failed
runF32.Finish status
runF64.Finish status
say (Invariant.format $"cpuref_probe: {failures} failed; run {runF64.RunId}")
exit (if failures = 0 then 0 else 1)
