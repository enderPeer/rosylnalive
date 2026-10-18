/// O2's acceptance on the tiny model (dev plan, O2 row): per op, CpuRef F64 vs torch64p within 1e-12 and CpuRef F32 vs
/// torch32 within 1e-6; the chained forward and greedy decoding against both oracles; bit-identity at 1, 7 and 32
/// threads; prefill equal to decode.
module Lrm.Tensor.Tests.TinyModelTests

open System
open System.Collections.Generic
open System.Numerics
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference
open Lrm.Tensor.Tests.Common

/// The per-op bounds of the O2 acceptance.
let F64OpBound = 1e-12
let F32OpBound = 1e-6

/// The CPU32 floors of the ToleranceSpec contract, which bound the chained float32 forward.
let Cpu32ChainRelL2 = 1e-5
let Cpu32LogitMaxAbs = 2e-3
let Cpu32LogitKl = 1e-7

let promptIds (prompt: string) =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"fwd_f32.{prompt}.safetensors")

    (f.ReadI64 "input_ids").Data |> Array.map int

let private attrs (pair: TolerancePair) (prompt: string) (op: string) (tapName: string) (kind: string) =
    [
        Attr.str "pair" (TolerancePair.name pair)
        Attr.str "prompt" prompt
        Attr.str "op" op
        Attr.str "tap" tapName
        Attr.int TelemetryKeys.Layer (ActivationNames.layer tapName)
        Attr.str "kind" kind
    ]

/// Every op of the tiny forward, fed with the oracle's own input taps and compared with its output tap.
let perOp<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (exec: CpuExec)
    (oracleFile: string)
    (pair: TolerancePair)
    (bound: float)
    (prompt: string)
    : (string * CheckResult) list =
    let m = TinyModel.model.Force()
    let c = m.Config
    let w = m.Weights

    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"{oracleFile}.{prompt}.safetensors")

    let tap (name: string) = f.Read<'T> name
    let ids = (f.ReadI64 "input_ids").Data |> Array.map int
    let eps = 'T.CreateTruncating c.Eps
    let scale = CpuRef.attentionScale<'T> c.HeadDim

    let shape =
        {
            QHeads = c.Heads
            KvHeads = c.KvHeads
            HeadDim = c.HeadDim
        }

    let checks = ResizeArray<string * CheckResult>()

    let check (op: string) (name: string) (computed: HostTensor<'T>) =
        checks.Add(op, Compare.relL2 bound (attrs pair prompt op name "op") computed (tap name))

    check "embed" "embed.out" (CpuRef.embed<'T> exec w.Embed ids)

    for i in 0 .. c.Layers - 1 do
        let lw = w.Layers[i]
        let p = Invariant.format $"L{i}."

        let x =
            tap (
                if i = 0 then
                    "embed.out"
                else
                    Invariant.format $"L{i - 1}.out"
            )

        check "rmsnorm" (p + "input_layernorm.out") (CpuRef.rmsNorm exec x lw.InputNorm eps)
        let h = tap (p + "input_layernorm.out")
        check "matmul" (p + "attn.q_proj.out") (CpuRef.matmulBf16W exec h lw.Q)
        check "matmul" (p + "attn.k_proj.out") (CpuRef.matmulBf16W exec h lw.K)
        check "matmul" (p + "attn.v_proj.out") (CpuRef.matmulBf16W exec h lw.V)
        check "rmsnorm" (p + "attn.q_norm.out") (CpuRef.rmsNorm exec (tap (p + "attn.q_proj.out")) lw.QNorm eps)
        check "rmsnorm" (p + "attn.k_norm.out") (CpuRef.rmsNorm exec (tap (p + "attn.k_proj.out")) lw.KNorm eps)
        check "rope" (p + "attn.q_rope") (CpuRef.ropeApply exec m.Rope (tap (p + "attn.q_norm.out")) c.Heads 0)
        check "rope" (p + "attn.k_rope") (CpuRef.ropeApply exec m.Rope (tap (p + "attn.k_norm.out")) c.KvHeads 0)
        // The KV cache stores V raw: the golden v_cache tap is the v_proj output itself.
        check "cache" (p + "attn.v_cache") (tap (p + "attn.v_proj.out"))

        check
            "attention"
            (p + "attn.o_proj.in")
            (CpuRef.attention
                exec
                (tap (p + "attn.q_rope"))
                (tap (p + "attn.k_rope"))
                (tap (p + "attn.v_cache"))
                shape
                0
                scale)

        check "matmul" (p + "attn.o_proj.out") (CpuRef.matmulBf16W exec (tap (p + "attn.o_proj.in")) lw.O)
        check "add" (p + "resid_mid") (CpuRef.add exec x (tap (p + "attn.o_proj.out")))

        check
            "rmsnorm"
            (p + "post_attention_layernorm.out")
            (CpuRef.rmsNorm exec (tap (p + "resid_mid")) lw.PostNorm eps)

        let h2 = tap (p + "post_attention_layernorm.out")
        check "matmul" (p + "mlp.gate.out") (CpuRef.matmulBf16W exec h2 lw.Gate)
        check "matmul" (p + "mlp.up.out") (CpuRef.matmulBf16W exec h2 lw.Up)
        check "silu_mul" (p + "mlp.down.in") (CpuRef.siluMul exec (tap (p + "mlp.gate.out")) (tap (p + "mlp.up.out")))
        check "matmul" (p + "mlp.down.out") (CpuRef.matmulBf16W exec (tap (p + "mlp.down.in")) lw.Down)
        check "add" (p + "out") (CpuRef.add exec (tap (p + "resid_mid")) (tap (p + "mlp.down.out")))

    check "rmsnorm" "final_norm.out" (CpuRef.rmsNorm exec (tap (Invariant.format $"L{c.Layers - 1}.out")) w.Norm eps)
    check "matmul" "logits" (CpuRef.matmulBf16W exec (tap "final_norm.out") w.Embed)
    List.ofSeq checks

let private tapOf (r: CheckResult) =
    r.Attrs
    |> List.tryPick (fun a ->
        match a.Value with
        | AttrValue.Str s when a.Key = "tap" -> Some s
        | _ -> None)
    |> Option.defaultValue "?"

/// Prints the worst measurement per key (op, or "chain") in one block, then fails on the first check that did not pass.
let private report (label: string) (checks: (string * CheckResult) list) =
    let lines =
        checks
        |> List.groupBy fst
        |> List.map (fun (key, xs) ->
            let worst = xs |> List.maxBy (fun (_, r) -> r.Measured) |> snd

            Invariant.format
                $"  {key, -10} worst {worst.Metric} {Invariant.f64 worst.Measured, -24} at {tapOf worst} ({xs.Length} checks, bound {Invariant.f64 worst.Bound})")

    printfn "%s" (String.Join("\n", label :: lines))

    for op, r in checks do
        if not r.Passed then
            failwithf "%s: %s %s: %O" label op (tapOf r) r

let private chainChecks<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (taps: Dictionary<string, HostTensor<'T>>)
    (oracleFile: string)
    (prompt: string)
    (pair: TolerancePair)
    (bound: float)
    (names: string[])
    =
    use f =
        tiny.Force().OpenSafetensors(Invariant.format $"{oracleFile}.{prompt}.safetensors")

    [
        for name in names do
            yield "chain", Compare.relL2 bound (attrs pair prompt "chain" name "chain") taps[name] (f.Tensor name)
    ]

/// A prefill of `ids` and the same ids decoded one at a time give the same logits rows and caches, bit for bit.
let private prefillEqualsDecode<'T
    when 'T :> IFloatingPointIeee754<'T> and 'T: unmanaged and 'T: struct and 'T: (new: unit -> 'T) and 'T :> ValueType>
    (label: string)
    (ids: int[])
    =
    let m = TinyModel.model.Force()
    let pc = TinyModel.newCache<'T> m
    let prefill = TinyModel.forward CpuExec.Machine m pc 0 ids None
    let dc = TinyModel.newCache<'T> m

    for t in 0 .. ids.Length - 1 do
        let step = TinyModel.forward CpuExec.Machine m dc t [| ids[t] |] None
        expectSameBits (Invariant.format $"{label} logits row {t}") step (rowOf prefill t)

    for l in 0 .. m.Config.Layers - 1 do
        expectSameBits (Invariant.format $"{label} K cache L{l}") pc.Keys[l] dc.Keys[l]
        expectSameBits (Invariant.format $"{label} V cache L{l}") pc.Values[l] dc.Values[l]

let tests =
    testList
        "tiny"
        [
            testList
                "per op"
                [
                    for prompt in [ "p0"; "p1" ] do
                        testCase (Invariant.format $"CpuRef F64 vs torch64p <= 1e-12 ({prompt})")
                        <| fun () ->
                            let checks, records =
                                withSink (fun () ->
                                    perOp<float>
                                        CpuExec.Machine
                                        "fwd_f64p"
                                        TolerancePair.Cpu64VsTorch64p
                                        F64OpBound
                                        prompt)

                            report (Invariant.format $"CpuRef F64 vs torch64p, {prompt}") checks

                            let compares =
                                records |> List.filter (fun r -> r.Name = TelemetryNames.GoldenCompare)

                            Expect.equal compares.Length checks.Length "one golden.compare per op check"

                            for r in compares do
                                Expect.equal (r.Attrs |> List.distinctBy _.Key).Length r.Attrs.Length "unique keys"

                        testCase (Invariant.format $"CpuRef F32 vs torch32 <= 1e-6 ({prompt})")
                        <| fun () ->
                            perOp<float32> CpuExec.Machine "fwd_f32" TolerancePair.Cpu32VsTorch32 F32OpBound prompt
                            |> report (Invariant.format $"CpuRef F32 vs torch32, {prompt}")
                ]

            testList
                "chain"
                [
                    for prompt in [ "p0"; "p1" ] do
                        testCase (Invariant.format $"CpuRef F64 forward vs torch64p, every tap <= 1e-12 ({prompt})")
                        <| fun () ->
                            let taps, _ = TinyModel.prefillTaps<float> CpuExec.Machine (promptIds prompt)

                            chainChecks
                                taps
                                "fwd_f64p"
                                prompt
                                TolerancePair.Cpu64VsTorch64p
                                F64OpBound
                                (ActivationNames.all 2)
                            |> report (Invariant.format $"CpuRef F64 chain vs torch64p, {prompt}")

                        testCase (
                            Invariant.format $"CpuRef F32 forward vs both oracles within the CPU32 floors ({prompt})"
                        )
                        <| fun () ->
                            let taps, _ = TinyModel.prefillTaps<float32> CpuExec.Machine (promptIds prompt)
                            let names = ActivationNames.all 2

                            chainChecks taps "fwd_f32" prompt TolerancePair.Cpu32VsTorch32 Cpu32ChainRelL2 names
                            |> report (Invariant.format $"CpuRef F32 chain vs torch32, {prompt}")

                            chainChecks taps "fwd_f64p" prompt TolerancePair.Cpu32VsTorch64p Cpu32ChainRelL2 names
                            |> report (Invariant.format $"CpuRef F32 chain vs torch64p, {prompt}")

                            use f =
                                tiny.Force().OpenSafetensors(Invariant.format $"fwd_f32.{prompt}.safetensors")

                            let a = attrs TolerancePair.Cpu32VsTorch32 prompt "chain" "logits" "chain"

                            let r = Compare.maxAbs Cpu32LogitMaxAbs a taps["logits"] (f.ReadF32 "logits")
                            Expect.isTrue r.Passed (string r)
                            let r = Compare.kl Cpu32LogitKl a taps["logits"] (f.ReadF32 "logits")
                            Expect.isTrue r.Passed (string r)

                    testCase "p2 (70 tokens, across a 64-key block): logits of both modes"
                    <| fun () ->
                        let ids = promptIds "p2"
                        let t64, _ = TinyModel.prefillTaps<float> CpuExec.Machine ids
                        let t32, _ = TinyModel.prefillTaps<float32> CpuExec.Machine ids

                        chainChecks t64 "fwd_f64p" "p2" TolerancePair.Cpu64VsTorch64p F64OpBound [| "logits" |]
                        |> report "CpuRef F64 chain vs torch64p, p2"

                        chainChecks t32 "fwd_f32" "p2" TolerancePair.Cpu32VsTorch32 Cpu32ChainRelL2 [| "logits" |]
                        |> report "CpuRef F32 chain vs torch32, p2"
                ]

            testList
                "greedy"
                [
                    for prompt in [ "p0"; "p1" ] do
                        testCase (Invariant.format $"64 greedy steps equal both oracles in F32 and F64 ({prompt})")
                        <| fun () ->
                            use g = tiny.Force().OpenSafetensors "greedy.safetensors"

                            let golden32 =
                                (g.ReadI64(Invariant.format $"{prompt}.ids.torch32")).Data |> Array.map int

                            let golden64 =
                                (g.ReadI64(Invariant.format $"{prompt}.ids.torch64p")).Data |> Array.map int

                            let ids = promptIds prompt

                            Expect.sequenceEqual
                                (TinyModel.greedy<float32> CpuExec.Machine ids 64)
                                golden32
                                "F32 = torch32"

                            Expect.sequenceEqual
                                (TinyModel.greedy<float> CpuExec.Machine ids 64)
                                golden64
                                "F64 = torch64p"
                ]

            testList
                "determinism"
                [
                    testCase "the F32 and F64 forwards are bit-identical at 1, 7 and 32 threads"
                    <| fun () ->
                        let ids = promptIds "p2"

                        let run32 n =
                            fst (TinyModel.prefillTaps<float32> (CpuExec.WithThreads n) ids)

                        let run64 n =
                            fst (TinyModel.prefillTaps<float> (CpuExec.WithThreads n) ids)

                        let a32, a64 = run32 1, run64 1

                        for n in [ 7; 32 ] do
                            let b32, b64 = run32 n, run64 n

                            for name in ActivationNames.all 2 do
                                expectSameBits (Invariant.format $"F32 {name}, 1 vs {n} threads") a32[name] b32[name]
                                expectSameBits (Invariant.format $"F64 {name}, 1 vs {n} threads") a64[name] b64[name]

                    testCase "prefill equals token-by-token decode, logits and caches bit for bit"
                    <| fun () ->
                        let ids = promptIds "p2"
                        prefillEqualsDecode<float32> "F32" ids
                        prefillEqualsDecode<float> "F64" ids
                ]
        ]
