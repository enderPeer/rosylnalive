// Reads goldens/qwen3-0.6b/g1 through LRM's own F# readers (G2), after a Release build:
//
//     dotnet fsi tools/goldens/qwen/check_set.fsx [<set folder>] [<data root>]
//
// - Goldens.Open validates the manifest (ComparatorsAndGoldenReader) and VerifyAll checks the size and SHA-256 of every
//   listed file, the LRM_DATA ones included;
// - ToleranceSpec.load parses tolerances.json with its exact calibration check, and every pair answers a lookup;
// - CpuRef's RoPE table for the pinned inv_freq bits equals the dumped T_spec bit for bit (CpuReferenceSemantics);
// - every data safetensors file opens through the validating reader, and each fwd_* file holds the prompt's ids.
// Exit code 0 when everything holds, 1 otherwise.

#r "../../../artifacts/bin/Lrm.Core/release/Lrm.Core.dll"
#r "../../../artifacts/bin/Lrm.Tensor/release/Lrm.Tensor.dll"
#r "../../../artifacts/bin/Lrm.Tensor.CpuReference/release/Lrm.Tensor.CpuReference.dll"

open System
open System.IO
open System.Text.Json
open Lrm.Core
open Lrm.Tensor
open Lrm.Tensor.CpuReference

let argv = fsi.CommandLineArgs |> Array.skip 1

let dir =
    if argv.Length > 0 then
        argv[0]
    else
        Path.Combine(__SOURCE_DIRECTORY__, "..", "..", "..", "goldens", "qwen3-0.6b", "g1")

let set =
    if argv.Length > 1 then
        Goldens.Open(dir, argv[1])
    else
        Goldens.Open dir

let mutable failures = 0

let check (ok: bool) (what: string) =
    printfn "%s %s" (if ok then "ok  " else "FAIL") what

    if not ok then
        failures <- failures + 1

printfn "%s/%s (%s): %d files" set.Set set.Version set.Kind set.Files.Length
set.VerifyAll()
check true "every listed file has its manifest size and SHA-256"

let spec = ToleranceSpec.load (Path.Combine(set.Dir, "tolerances.json"))
check (spec.Set = "qwen3-0.6b/g1") "tolerances.json names its set"

for pair in TolerancePair.all do
    match spec.TryBound(pair, "logits", ToleranceMetric.RelL2) with
    | Some b ->
        let eRef = b.ERef |> Option.map Invariant.f64 |> Option.defaultValue "fixed"

        printfn "  %-18s logits relL2 bound %s (e_ref %s)" (TolerancePair.name pair) (Invariant.f64 b.Bound) eRef
    | None -> check false (TolerancePair.name pair + " has no logits relL2 bound")

for kind in ActivationNames.all 28 |> Array.map ActivationNames.kind |> Array.distinct do
    check
        (spec.TryBound(TolerancePair.Cpu32VsTorch32, kind, ToleranceMetric.RelL2)).IsSome
        ("cpu32-vs-torch32 bounds " + kind)

let rope = set.OpenSafetensors "rope.safetensors"
let bits = (rope.ReadU32 "inv_freq.bits").Data
let table = CpuRef.ropeTable bits 8192
check (table.Cos = (rope.ReadF32 "t_spec.cos").Data) "CpuRef's cos table equals the dumped T_spec bit for bit"
check (table.Sin = (rope.ReadF32 "t_spec.sin").Data) "CpuRef's sin table equals the dumped T_spec bit for bit"

check
    (table.Digest().Hex = "da820d286cb666557f4cae3fa7f806c07d941653769e10c09021555a8f6ebfaf")
    "the T_spec digest is the ReferenceOracles vector"

rope.Dispose()

let prompts =
    set.Manifest.GetProperty("prompts").EnumerateArray()
    |> Seq.map (fun p ->
        p.GetProperty("name").GetString(), p.GetProperty("ids").EnumerateArray() |> Seq.map _.GetInt64() |> Array.ofSeq)
    |> Map.ofSeq

for f in set.Files do
    if f.Name.EndsWith(".safetensors", StringComparison.Ordinal) then
        use st = set.OpenSafetensors f.Name
        let parts = f.Name.Split('.')

        if parts[0].StartsWith("fwd_", StringComparison.Ordinal) then
            let ids = (st.ReadI64 "input_ids").Data
            check (ids = prompts[parts[1]]) (f.Name + ": input_ids are the manifest's prompt")
        else
            check (st.Entries.Length > 0) (f.Name + " opens")

printfn
    "check_set: %s"
    (if failures = 0 then
         "all passed"
     else
         Invariant.format $"{failures} FAILED")

exit (if failures = 0 then 0 else 1)
