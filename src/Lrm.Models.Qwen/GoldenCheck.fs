namespace Lrm.Models.Qwen

open System
open Lrm.Core
open Lrm.Tensor

/// The comparisons of one tensor with a golden file (GoldenCheck.tensors).
type GoldenTensorCheck =
    {
        /// The tensor (tap) name.
        Name: string
        /// Its golden.compare checks; empty only for a tensor the file does not hold.
        Checks: CheckResult list
        /// A remark for the printout, such as " (last row)".
        Note: string
    }

/// The comparisons behind `lrm forward --compare` and `lrm generate --expect` (ForwardOpsAndQwenModel, "The
/// commands"), as functions the tests cover. Every comparison is a golden.compare check (ComparatorsAndGoldenReader);
/// nothing the caller asked to compare is skipped silently: a tensor that cannot be compared is a failed check.
[<RequireQualifiedAccess>]
module GoldenCheck =
    /// The `detail` attribute of a comparison that measured nothing.
    [<Literal>]
    let DetailKey = "detail"

    /// The last row of a [rows, cols] tensor, in its dtype (F32 or F64).
    let lastRow (t: IHostTensor) : IHostTensor =
        let rows = int t.Desc.Shape.Rows
        let cols = int t.Desc.Shape.Cols
        let size = t.Desc.ElementSize
        let bytes = ReadOnlySpan(t.ToBytes(), (rows - 1) * cols * size, cols * size)

        match t.Desc.DType with
        | DType.F32 -> HostTensor.ofBytes<float32> DType.F32 (Shape.matrix 1 cols) bytes :> IHostTensor
        | DType.F64 -> HostTensor.ofBytes<float> DType.F64 (Shape.matrix 1 cols) bytes :> IHostTensor
        | d -> invalidArg "t" (sprintf "the last row of a %s tensor" (DType.name d))

    /// A failed comparison that measured nothing: relL2 NaN, which a report counts as an invalid comparison (NaN never
    /// passes), with the reason in `detail`.
    let invalid (bound: float) (attrs: Attr list) (detail: string) =
        Compare.check "relL2" BoundOp.AtMost Double.NaN bound 0L (attrs @ [ Attr.str DetailKey detail ])

    /// Compares each output with the tensor of the same name in a golden file: relL2 for every tensor, and max |Δ| and
    /// KL for logits, each where `boundOf name metric` gives a bound. A [1, cols] output against a golden of more rows
    /// (a last-row head) is compared with the golden's last row. An output whose shape differs from the golden's is a
    /// failed check; with `requireAll`, so is every golden tensor but `input_ids` that the outputs lack. An output
    /// the file does not hold has no checks.
    let tensors
        (file: SafetensorsFile)
        (outputs: (string * IHostTensor) list)
        (boundOf: string -> ToleranceMetric -> float option)
        (attrsOf: string -> Attr list)
        (requireAll: bool)
        : GoldenTensorCheck list =
        let relBound (name: string) =
            boundOf name ToleranceMetric.RelL2 |> Option.defaultValue 0.0

        [
            for name, ours in outputs do
                match file.TryEntry name with
                | None ->
                    yield
                        {
                            Name = name
                            Checks = []
                            Note = "not in the golden file"
                        }
                | Some _ ->
                    let golden = file.Tensor name

                    let golden, note =
                        if
                            ours.Desc.Shape.Rank = 2
                            && ours.Desc.Shape.Rows = 1L
                            && golden.Desc.Shape.Rank = 2
                            && golden.Desc.Shape.Rows > 1L
                            && ours.Desc.Shape.Cols = golden.Desc.Shape.Cols
                        then
                            lastRow golden, " (last row)"
                        else
                            golden, ""

                    let attrs = attrsOf name

                    if not (ours.Desc.Shape.Equals golden.Desc.Shape) then
                        let detail =
                            sprintf "shape %O, golden %O: not comparable" ours.Desc.Shape golden.Desc.Shape

                        yield
                            {
                                Name = name
                                Checks = [ invalid (relBound name) attrs detail ]
                                Note = " (" + detail + ")"
                            }
                    else
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

            if requireAll then
                let produced = outputs |> List.map fst |> Set.ofList

                for name in file.Names do
                    if name <> "input_ids" && not (produced.Contains name) then
                        let detail = "the golden file holds it, the run did not produce it"

                        yield
                            {
                                Name = name
                                Checks = [ invalid (relBound name) (attrsOf name) detail ]
                                Note = " (" + detail + ")"
                            }
        ]

    /// The greedy_match check (ReportTemplates, parity) of generated ids against golden ids: the steps that match
    /// before the first difference, against the steps generated, so a run of fewer steps than the golden (or one
    /// that stopped at EOS) is compared with the golden's prefix. It passes only when every generated id matches.
    /// Raises ArgumentException when more ids were generated than the golden holds.
    let greedy (attrs: Attr list) (got: int[]) (golden: int[]) : CheckResult =
        if got.Length > golden.Length then
            invalidArg "golden" (sprintf "%d ids generated; the golden holds %d" got.Length golden.Length)

        let matched =
            Seq.zip got golden |> Seq.takeWhile (fun (a, b) -> a = b) |> Seq.length

        Compare.check
            "greedy_match"
            BoundOp.AtLeast
            (float matched)
            (float got.Length)
            (int64 got.Length)
            (attrs @ [ Attr.int "steps" got.Length ])
