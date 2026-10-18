// Cross-checks Lrm.Gpu.Vulkan.Interop against R1's hand-written layouts: spikes/vk-heap/src/Layout.fs transcribes
// the C declaration of every struct the spike used from the Vulkan reference pages, with its own x64 layout
// calculator. Two independent sources (vk.xml through vkgen, and the reference pages through R1) must agree on every
// size, alignment, member name, order and offset. The spike's file is read as data (#load), never built.
//
//     dotnet fsi tools/abi-oracle/vulkan/spike-crosscheck.fsx <path to Lrm.Gpu.Vulkan.Interop.dll>
//
// Exit codes: 0 every declaration agrees; 1 a mismatch; 2 usage error. Owner: R2 (VulkanInteropApi contract).
#load "../../../spikes/vk-heap/src/Layout.fs"

open System
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open VkHeapSpike

let args = fsi.CommandLineArgs |> Array.skip 1

if args.Length <> 1 then
    eprintfn "usage: dotnet fsi spike-crosscheck.fsx <Lrm.Gpu.Vulkan.Interop.dll>"
    exit 2

let asm = Assembly.LoadFrom(args[0])

let nameOf (t: Type) =
    t.GetCustomAttributesData()
    |> Seq.tryPick (fun a ->
        match a.AttributeType.Name with
        | "VkStructAttribute"
        | "VkUnionAttribute" -> Some(string a.ConstructorArguments[0].Value)
        | _ -> None)

let byName =
    asm.GetTypes()
    |> Array.choose (fun t -> nameOf t |> Option.map (fun n -> n, t))
    |> dict

// Aliases (VkDeviceAddressRangeEXT is VkDeviceAddressRangeKHR): VkMeta.Aliases holds (alias, type).
let aliases =
    let meta = asm.GetType "Lrm.Gpu.Vulkan.Interop.VkMeta"
    let p = meta.GetProperty("Aliases", BindingFlags.Public ||| BindingFlags.Static)

    (p.GetValue null :?> Collections.IEnumerable)
    |> Seq.cast<obj>
    |> Seq.map (fun o ->
        let items = Reflection.FSharpValue.GetTupleFields o
        string items[0], items[1] :?> Type)
    |> dict

let alignOf (t: Type) =
    // The offset of a field of type t after one byte, measured on a dynamic two-field struct.
    let ab =
        Reflection.Emit.AssemblyBuilder.DefineDynamicAssembly(
            AssemblyName "probe",
            Reflection.Emit.AssemblyBuilderAccess.Run
        )

    let mb = ab.DefineDynamicModule "probe"

    let tb =
        mb.DefineType(
            "Probe",
            TypeAttributes.Public
            ||| TypeAttributes.SequentialLayout
            ||| TypeAttributes.Sealed,
            typeof<ValueType>
        )

    tb.DefineField("pad", typeof<byte>, FieldAttributes.Public) |> ignore
    tb.DefineField("value", t, FieldAttributes.Public) |> ignore
    let probe = tb.CreateType()
    int (Marshal.OffsetOf(probe, "value"))

let mutable structs = 0
let mutable members = 0
let problems = ResizeArray<string>()

for (name, decl) in Layout.decls do
    let t =
        match byName.TryGetValue name with
        | true, t -> Some t
        | _ ->
            match aliases.TryGetValue name with
            | true, t -> Some t
            | _ -> None

    match t with
    | None -> problems.Add(sprintf "%s: the spike declares it, the binding does not" name)
    | Some t ->
        structs <- structs + 1
        let size, align, offsets = Layout.layoutOf (Layout.S name)
        let runtimeSize = RuntimeHelpers.SizeOf t.TypeHandle

        if runtimeSize <> size then
            problems.Add(sprintf "%s: binding %d bytes, spike %d" name runtimeSize size)

        let runtimeAlign = alignOf t

        if runtimeAlign <> align then
            problems.Add(sprintf "%s: binding aligned to %d, spike %d" name runtimeAlign align)

        let fields =
            t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
            |> Array.map (fun f -> f.Name, int (Marshal.OffsetOf(t, f.Name)))
            |> dict

        match decl with
        | Layout.Union _ ->
            // The spike binds a subset of each union's members; each must exist in the binding at offset 0.
            for (m, _) in offsets do
                members <- members + 1

                match fields.TryGetValue m with
                | true, 0 -> ()
                | true, o -> problems.Add(sprintf "%s.%s: binding offset %d in a union" name m o)
                | _ -> problems.Add(sprintf "%s.%s: not in the binding" name m)
        | Layout.Struct _ ->
            let runtimeOrder =
                fields
                |> Seq.sortBy (fun kv -> kv.Value)
                |> Seq.map (fun kv -> kv.Key)
                |> List.ofSeq

            let spikeOrder = offsets |> List.map fst

            if runtimeOrder <> spikeOrder then
                problems.Add(sprintf "%s: binding members %A, spike %A" name runtimeOrder spikeOrder)
            else
                for (m, o) in offsets do
                    members <- members + 1

                    if fields[m] <> o then
                        problems.Add(sprintf "%s.%s: binding offset %d, spike %d" name m fields[m] o)

for p in problems do
    printfn "MISMATCH %s" p

printfn
    "spike cross-check: %d structs and unions, %d members of spikes/vk-heap/src/Layout.fs against the binding; %d mismatches"
    structs
    members
    problems.Count

exit (if problems.Count = 0 then 0 else 1)
