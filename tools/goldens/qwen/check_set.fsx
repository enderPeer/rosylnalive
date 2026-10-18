// Reads goldens/qwen3-0.6b/g1 through LRM's own F# readers (G2), after a Release build:
//
//     dotnet fsi tools/goldens/qwen/check_set.fsx [<set folder>] [<data root>]
//
// - Goldens.Open validates the manifest (ComparatorsAndGoldenReader) and VerifyAll checks the size and SHA-256 of every
//   listed file, the LRM_DATA ones included;
// - ToleranceSpec.load parses tolerances.json with its exact calibration check, and every pair answers a lookup;
//   vulkan-vs-d3d12 bounds maxAbs on logits only (ADR G2-0001, item 5);
// - CpuRef's RoPE table for the pinned inv_freq bits equals the dumped T_spec bit for bit (CpuReferenceSemantics);
// - every data safetensors file opens through the validating reader, and each fwd_* file holds the prompt's ids;
// - each prompt's files follow its role and `dumps`: gate prompts fwd_* and tf_*, long prompts long_* only.
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

// ADR G2-0001, item 5: vulkan-vs-d3d12 bounds max |delta| on logits only, so activations have no maxAbs bound there.
check
    ((spec.TryBound(TolerancePair.VulkanVsD3D12, "logits", ToleranceMetric.MaxAbs)
      |> Option.map _.Bound)
        =
        Some 0.01)
    "vulkan-vs-d3d12 bounds max |delta logit| at 0.01"

check
    (ActivationNames.all 28
     |> Array.map ActivationNames.kind
     |> Array.distinct
     |> Array.filter ((<>) "logits")
     |> Array.forall (fun kind -> (spec.TryBound(TolerancePair.VulkanVsD3D12, kind, ToleranceMetric.MaxAbs)).IsNone))
    "vulkan-vs-d3d12 has no maxAbs bound for any activation kind"

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

// ADR G2-0002, Consequences: consumers branch on `role` first. A gate prompt's `dumps` is "taps" or "logits" and it
// has fwd_* and tf_* files; a long prompt's is "long" and it has only long_* files.
let fileNames = set.Files |> Array.map _.Name |> Set.ofArray

let filesOf (prefix: string) (name: string) =
    [ "f32"; "f64p" ]
    |> List.map (fun o -> fileNames.Contains(prefix + "_" + o + "." + name + ".safetensors"))

for p in set.Manifest.GetProperty("prompts").EnumerateArray() do
    let name = p.GetProperty("name").GetString()
    let role = p.GetProperty("role").GetString()
    let dumps = p.GetProperty("dumps").GetString()
    let all prefix = filesOf prefix name |> List.forall id
    let none prefix = filesOf prefix name |> List.forall not

    match role with
    | "gate" ->
        check
            ((dumps = "taps" || dumps = "logits") && all "fwd" && all "tf" && none "long")
            (Invariant.format $"{name}: a gate prompt dumps {dumps} in fwd_* and has tf_*")
    | "long" ->
        check (dumps = "long" && all "long" && none "fwd" && none "tf") (name + ": a long prompt has long_* files only")
    | other -> check false (Invariant.format $"{name}: unknown role {other}")

printfn
    "check_set: %s"
    (if failures = 0 then
         "all passed"
     else
         Invariant.format $"{failures} FAILED")

exit (if failures = 0 then 0 else 1)
