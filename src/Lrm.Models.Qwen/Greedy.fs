namespace Lrm.Models.Qwen

open System
open Lrm.Core
open Lrm.Tensor
open Lrm.Telemetry

/// The options of Greedy.run.
type GreedyOptions =
    {
        /// Tokens to generate, at least 1: the prefill gives the first, each decode step one more.
        Steps: int
        /// Stop after the first EOS id of the config (it is kept as the last id). Parity runs pass false: the goldens
        /// ignore EOS (ReferenceOracles, "Greedy decoding and EOS").
        StopAtEos: bool
        /// Also read every step's logits row back and record its top-2 margin (a full row readback on a GPU).
        Margins: bool
    }

[<RequireQualifiedAccess>]
module GreedyOptions =
    /// `steps` tokens, EOS ignored, no margins: the parity schedule.
    let parity (steps: int) =
        {
            Steps = steps
            StopAtEos = false
            Margins = false
        }

/// The result of Greedy.run.
type GreedyResult =
    {
        Prompt: int[]
        /// The generated ids, in order (fewer than Steps only when StopAtEos stopped at an EOS id).
        Ids: int[]
        /// Each step's top-1 minus top-2 logit, in float64 from the ops' DType, when GreedyOptions.Margins is set;
        /// else empty.
        Margins: float[]
        /// True when the last id is an EOS id and StopAtEos stopped there.
        StoppedAtEos: bool
    }

/// Greedy decoding on the canonical schedule (ForwardOpsAndQwenModel contract): one prefill of the prompt with the
/// head on its last row, then single-token decode steps, each feeding back the argmax of the previous step's logits
/// (lowest index on ties). A step whose logits hold a NaN raises ForwardException.
[<RequireQualifiedAccess>]
module Greedy =
    /// `model.generate` span: one Greedy.run.
    [<Literal>]
    let GenerateSpan = "model.generate"

    /// The `component` attribute of the model.generate span (the steps' own records are the forward's).
    [<Literal>]
    let Component = "generate"

    /// The positions a greedy run uses: the prompt and every generated id but the last.
    let positions (promptLength: int) (steps: int) = promptLength + steps - 1

    /// Generates options.Steps ids after `prompt` into a new cache of exactly the positions it uses.
    let run (ops: IForwardOps<'t>) (model: Qwen3Model) (prompt: int[]) (options: GreedyOptions) : GreedyResult =
        let c = model.Config

        if isNull prompt || prompt.Length = 0 then
            invalidArg "prompt" "greedy decoding needs a prompt of at least one token"

        if options.Steps < 1 then
            invalidArg "options" (sprintf "%d steps; at least 1" options.Steps)

        let maxPos = positions prompt.Length options.Steps

        if maxPos > ops.MaxContext then
            raise (
                ForwardException(
                    sprintf
                        "%d prompt tokens and %d steps need %d positions; the ops hold %d"
                        prompt.Length
                        options.Steps
                        maxPos
                        ops.MaxContext
                )
            )

        let stage = Stage.whole c
        let cache = KvCache.create ops model stage maxPos
        let eos = Set.ofList c.EosTokenIds

        use _ =
            Telemetry.span
                GenerateSpan
                [
                    Attr.str TelemetryKeys.Component Component
                    Attr.int "prompt_tokens" prompt.Length
                    Attr.int "steps" options.Steps
                    Attr.bool "stop_at_eos" options.StopAtEos
                ]

        let ids = ResizeArray<int>(options.Steps)
        let margins = ResizeArray<float>()

        let pick (step: int) (logits: 't) =
            let a = ops.Argmax logits

            if a.Length <> 1 then
                raise (ForwardException(sprintf "step %d: %d logits rows, expected 1" step a.Length))

            if a[0] = UInt32.MaxValue then
                raise (ForwardException(sprintf "step %d: the logits hold a NaN" step))

            if Telemetry.enabled () then
                Metering.op
                    "model"
                    (Op.Argmax(1L, int64 c.Vocab))
                    [
                        Attr.str TelemetryKeys.Component Forward.Component
                        Attr.int TelemetryKeys.Stage stage.Index
                    ]

            if options.Margins then
                margins.Add(Metrics.top2MarginRow (ReadOnlySpan((ops.Read logits).ToFloat64())))

            let id = int a[0]
            ids.Add id
            id

        let mutable next =
            pick
                0
                (Forward.run
                    ops
                    model
                    cache
                    0
                    prompt
                    { ForwardOptions.last with
                        Step = Some 0
                    })

        let mutable stopped = options.StopAtEos && eos.Contains next
        let mutable step = 1

        while step < options.Steps && not stopped do
            let pos = prompt.Length + step - 1

            next <-
                pick
                    step
                    (Forward.run
                        ops
                        model
                        cache
                        pos
                        [| next |]
                        { ForwardOptions.last with
                            Step = Some step
                        })

            stopped <- options.StopAtEos && eos.Contains next
            step <- step + 1

        {
            Prompt = Array.copy prompt
            Ids = ids.ToArray()
            Margins = margins.ToArray()
            StoppedAtEos = stopped
        }
