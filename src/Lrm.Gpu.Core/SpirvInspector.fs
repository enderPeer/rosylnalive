namespace Lrm.Gpu.Core

open System
open System.Collections.Generic
open System.Text
open Lrm.Core

/// A rule of the SPIR-V inspector (ShaderBuildAndSourceConvention v1.1, "The SPIR-V inspector").
[<RequireQualifiedAccess>]
type SpirvRule =
    /// A well-formed SPIR-V 1.6 module with the Logical GLSL450 memory model.
    | Module
    /// Only the capabilities the recipe's device features allow, Shader and RuntimeDescriptorArray among them.
    | Capabilities
    /// Only the SPV_EXT_descriptor_indexing extension and the GLSL.std.450 instruction set.
    | Extensions
    /// Exactly one entry point: GLCompute "main".
    | EntryPoint
    /// Exactly one execution mode: LocalSize, equal to the catalog's thread counts.
    | ExecutionMode
    /// Exactly one resource variable: the writable runtime array of raw buffers at set 0, binding 0.
    | Resource
    /// Exactly one push-constant block: lrm_args_<kernel>, word i at offset 4 * i with the catalog's name and type.
    | PushBlock
    /// Module-scope variables only in StorageBuffer, PushConstant, Input, Workgroup and Private storage.
    | StorageClass
    /// Input variables are only the WorkgroupId, LocalInvocationId and LocalInvocationIndex built-ins.
    | BuiltIns
    /// No atomic operation on a float.
    | FloatAtomics
    /// No atomic operation at all (contracts v1).
    | Atomics
    /// No 16-bit integer or float type.
    | SixteenBit
    /// No 8-bit or 64-bit integer or float type.
    | ScalarWidth
    /// No wave, subgroup or group operation.
    | WaveOps
    /// No image, sampler or sampled-image type.
    | TypedViews
    /// Workgroup (groupshared) variables total at most 32 KiB.
    | WorkgroupSize

[<RequireQualifiedAccess>]
module SpirvRule =
    let all =
        [
            SpirvRule.Module
            SpirvRule.Capabilities
            SpirvRule.Extensions
            SpirvRule.EntryPoint
            SpirvRule.ExecutionMode
            SpirvRule.Resource
            SpirvRule.PushBlock
            SpirvRule.StorageClass
            SpirvRule.BuiltIns
            SpirvRule.FloatAtomics
            SpirvRule.Atomics
            SpirvRule.SixteenBit
            SpirvRule.ScalarWidth
            SpirvRule.WaveOps
            SpirvRule.TypedViews
            SpirvRule.WorkgroupSize
        ]

    /// The rule's name in findings, telemetry and the fixtures' `expect:` lines.
    let name (r: SpirvRule) =
        match r with
        | SpirvRule.Module -> "module"
        | SpirvRule.Capabilities -> "capabilities"
        | SpirvRule.Extensions -> "extensions"
        | SpirvRule.EntryPoint -> "entry-point"
        | SpirvRule.ExecutionMode -> "execution-mode"
        | SpirvRule.Resource -> "resource"
        | SpirvRule.PushBlock -> "push-block"
        | SpirvRule.StorageClass -> "storage-class"
        | SpirvRule.BuiltIns -> "built-ins"
        | SpirvRule.FloatAtomics -> "float-atomics"
        | SpirvRule.Atomics -> "atomics"
        | SpirvRule.SixteenBit -> "16-bit-types"
        | SpirvRule.ScalarWidth -> "scalar-width"
        | SpirvRule.WaveOps -> "wave-ops"
        | SpirvRule.TypedViews -> "typed-views"
        | SpirvRule.WorkgroupSize -> "workgroup-size"

    let tryParse (text: string) =
        all
        |> List.tryFind (fun r -> String.Equals(name r, text, StringComparison.Ordinal))

/// One broken rule.
type SpirvFinding = { Rule: SpirvRule; Message: string }

/// One member of the push-constant block, as the module declares it.
type SpirvPushMember =
    {
        /// OpMemberName, when present.
        Name: string option
        /// The Offset decoration, when present.
        Offset: int option
        /// "u32", "i32", "f32", or another type's description.
        Type: string
    }

/// A module-scope variable with a DescriptorSet or Binding decoration.
type SpirvResource =
    {
        Name: string option
        StorageClass: uint32
        Set: uint32 option
        Binding: uint32 option
        NonWritable: bool
    }

/// What the inspector read from a module; also the data of the shader reports.
type SpirvFacts =
    {
        Version: uint32
        Capabilities: uint32 list
        Extensions: string list
        ExtInstImports: string list
        /// The LocalSize execution mode, when there is exactly one.
        LocalSize: (int * int * int) option
        Resources: SpirvResource list
        /// The OpName of the push-constant block's type.
        PushBlock: string option
        PushMembers: SpirvPushMember list
        /// The bytes the push-constant block spans (0 without one).
        PushBytes: int
        /// The bytes of all Workgroup variables (-1 when a size cannot be computed).
        WorkgroupBytes: int64
        /// The built-ins the module's Input variables carry.
        BuiltIns: uint32 list
    }

/// The inspector's verdict on one SPIR-V blob.
type SpirvReport =
    {
        Kernel: string
        /// Facts, unless the module could not be parsed.
        Facts: SpirvFacts option
        /// Every broken rule, in rule order.
        Findings: SpirvFinding list
    }

    member r.Passed = r.Findings.IsEmpty

/// The SPIR-V inspector (ShaderBuildAndSourceConvention v1.1): checks one compiled SPIR-V blob against its catalog
/// entry and the binding of VulkanHeapRecipe. The shader build runs it on every SPIR-V blob it builds or reuses, the
/// cpu tier on every embedded blob, and CI's shader step on the built set and on negative fixtures.
[<RequireQualifiedAccess>]
module SpirvInspector =
    /// The groupshared limit of KernelAbiAndCatalog, in bytes.
    [<Literal>]
    let MaxWorkgroupBytes = 32768L

    /// Capabilities a blob may declare: Shader (core), RuntimeDescriptorArray (feature runtimeDescriptorArray) and
    /// StorageBufferArrayDynamicIndexing (feature shaderStorageBufferArrayDynamicIndexing). The recipe's device
    /// enables exactly these features, so any other capability could need one it lacks.
    let allowedCapabilities =
        [
            Spirv.Capability.Shader
            Spirv.Capability.StorageBufferArrayDynamicIndexing
            Spirv.Capability.RuntimeDescriptorArray
        ]

    /// Capabilities every blob declares: g_buf[] is a runtime array.
    let requiredCapabilities =
        [ Spirv.Capability.Shader; Spirv.Capability.RuntimeDescriptorArray ]

    /// Extensions a blob may declare (DXC declares descriptor indexing for g_buf[]; it is core in SPIR-V 1.5).
    let allowedExtensions = [ "SPV_EXT_descriptor_indexing" ]

    /// Extended instruction sets a blob may import.
    let allowedExtInstImports = [ "GLSL.std.450" ]

    /// Built-ins of the kernels' inputs: SV_GroupID, SV_GroupThreadID and SV_GroupIndex.
    let allowedBuiltIns =
        [
            Spirv.BuiltIn.WorkgroupId
            Spirv.BuiltIn.LocalInvocationId
            Spirv.BuiltIn.LocalInvocationIndex
        ]

    /// Storage classes of module-scope variables.
    let allowedGlobalStorage =
        [
            Spirv.StorageClass.StorageBuffer
            Spirv.StorageClass.PushConstant
            Spirv.StorageClass.Input
            Spirv.StorageClass.Workgroup
            Spirv.StorageClass.Private
        ]

    /// The OpName DXC gives the push-constant block of `LRM_ARGS(kernel)`.
    let pushBlockName (kernel: KernelSpec) =
        "type.ConstantBuffer.lrm_args_" + kernel.Name

    /// The push block's members as the catalog defines them: word 0 `lrm_groups_x` (u32), then the fields.
    let expectedPushMembers (kernel: KernelSpec) : (string * string) list =
        ("lrm_groups_x", "u32")
        :: [
            for f in kernel.Fields ->
                f.Name,
                match f.Kind with
                | FieldKind.Token _
                | FieldKind.U32 -> "u32"
                | FieldKind.I32 -> "i32"
                | FieldKind.F32 -> "f32"
        ]

    let private isAtomic (op: int) =
        (op >= Spirv.Op.AtomicLoad && op <= Spirv.Op.AtomicXor)
        || op = Spirv.Op.AtomicFlagTestAndSet
        || op = Spirv.Op.AtomicFlagClear
        || op = Spirv.Op.AtomicFMinEXT
        || op = Spirv.Op.AtomicFMaxEXT
        || op = Spirv.Op.AtomicFAddEXT

    let private isWaveOp (op: int) =
        (op >= 261 && op <= 271)
        || (op >= Spirv.Op.GroupNonUniformElect && op <= Spirv.Op.GroupNonUniformQuadSwap)
        || (op >= 4421 && op <= 4432)

    let private sixteenBitCapabilities = [ 8u; 9u; 22u; 4433u; 4434u; 4435u; 4436u ]

    let private hex (v: uint32) = sprintf "0x%08x" v

    /// Inspects one SPIR-V blob against its catalog entry.
    let inspect (kernel: KernelSpec) (bytes: byte[]) : SpirvReport =
        match Spirv.parse bytes with
        | Error e ->
            {
                Kernel = kernel.Name
                Facts = None
                Findings =
                    [
                        {
                            Rule = SpirvRule.Module
                            Message = "not a SPIR-V module: " + e
                        }
                    ]
            }
        | Ok m ->
            let findings = ResizeArray<SpirvFinding>()

            let add rule fmt =
                Printf.kprintf (fun s -> findings.Add { Rule = rule; Message = s }) fmt

            // ---- index the module ----
            let names = Dictionary<uint32, string>()
            let memberNames = Dictionary<struct (uint32 * uint32), string>()
            let decorations = Dictionary<uint32, ResizeArray<uint32 * uint32[]>>()

            let memberDecorations =
                Dictionary<struct (uint32 * uint32), ResizeArray<uint32 * uint32[]>>()

            let types = Dictionary<uint32, SpirvInstruction>()
            let constants = Dictionary<uint32, SpirvInstruction>()
            let typeOf = Dictionary<uint32, uint32>()
            let globals = ResizeArray<uint32 * uint32 * uint32>() // type, id, storage class
            let capabilities = ResizeArray<uint32>()
            let extensions = ResizeArray<string>()
            let extInst = ResizeArray<string>()
            let memoryModels = ResizeArray<uint32 * uint32>()
            let entryPoints = ResizeArray<uint32 * string>()
            let modes = ResizeArray<int * uint32 * uint32[]>() // opcode, mode, operands after the mode
            let atomics = ResizeArray<SpirvInstruction>()
            let waveOps = ResizeArray<int>()
            let mutable inFunction = false

            let listAt (d: Dictionary<'k, ResizeArray<'v>>) (k: 'k) =
                match d.TryGetValue k with
                | true, l -> l
                | _ ->
                    let l = ResizeArray<'v>()
                    d[k] <- l
                    l

            let str (ins: SpirvInstruction) (start: int) =
                match Spirv.literalString ins.Operands start with
                | Some(s, _) -> s
                | None ->
                    add SpirvRule.Module "word %d: a literal string without its terminator" ins.Word
                    ""

            let operandsOk (ins: SpirvInstruction) (n: int) =
                if ins.Operands.Length >= n then
                    true
                else
                    add
                        SpirvRule.Module
                        "word %d: opcode %d needs %d operands, has %d"
                        ins.Word
                        ins.Opcode
                        n
                        ins.Operands.Length

                    false

            for ins in m.Instructions do
                let o = ins.Operands

                match ins.Opcode with
                | Spirv.Op.Name when operandsOk ins 2 -> names[o[0]] <- str ins 1
                | Spirv.Op.MemberName when operandsOk ins 3 -> memberNames[struct (o[0], o[1])] <- str ins 2
                | Spirv.Op.Capability when operandsOk ins 1 -> capabilities.Add o[0]
                | Spirv.Op.Extension when operandsOk ins 1 -> extensions.Add(str ins 0)
                | Spirv.Op.ExtInstImport when operandsOk ins 2 -> extInst.Add(str ins 1)
                | Spirv.Op.MemoryModel when operandsOk ins 2 -> memoryModels.Add(o[0], o[1])
                | Spirv.Op.EntryPoint when operandsOk ins 3 -> entryPoints.Add(o[0], str ins 2)
                | Spirv.Op.ExecutionMode
                | Spirv.Op.ExecutionModeId when operandsOk ins 2 ->
                    modes.Add(ins.Opcode, o[1], Array.sub o 2 (o.Length - 2))
                | Spirv.Op.Decorate when operandsOk ins 2 ->
                    (listAt decorations o[0]).Add(o[1], Array.sub o 2 (o.Length - 2))
                | Spirv.Op.MemberDecorate when operandsOk ins 3 ->
                    (listAt memberDecorations (struct (o[0], o[1]))).Add(o[2], Array.sub o 3 (o.Length - 3))
                | op when op >= Spirv.Op.TypeVoid && op <= 39 && operandsOk ins 1 -> types[o[0]] <- ins
                | Spirv.Op.Constant
                | Spirv.Op.SpecConstant when operandsOk ins 3 -> constants[o[1]] <- ins
                | Spirv.Op.Function -> inFunction <- true
                | Spirv.Op.FunctionEnd -> inFunction <- false
                | Spirv.Op.Variable when operandsOk ins 3 ->
                    typeOf[o[1]] <- o[0]

                    if inFunction then
                        if o[2] <> Spirv.StorageClass.Function then
                            add
                                SpirvRule.StorageClass
                                "a variable inside a function has storage class %s, not Function"
                                (Spirv.StorageClass.name o[2])
                    else
                        globals.Add(o[0], o[1], o[2])
                | Spirv.Op.FunctionParameter
                | Spirv.Op.Load
                | Spirv.Op.AccessChain
                | Spirv.Op.InBoundsAccessChain
                | Spirv.Op.PtrAccessChain
                | Spirv.Op.CopyObject when operandsOk ins 2 -> typeOf[o[1]] <- o[0]
                | op when isAtomic op -> atomics.Add ins
                | op when isWaveOp op -> waveOps.Add op
                | _ -> ()

            let decoration (id: uint32) (d: uint32) =
                match decorations.TryGetValue id with
                | true, l -> l |> Seq.tryFind (fun (k, _) -> k = d) |> Option.map snd
                | _ -> None

            let memberDecoration (s: uint32) (i: uint32) (d: uint32) =
                match memberDecorations.TryGetValue(struct (s, i)) with
                | true, l -> l |> Seq.tryFind (fun (k, _) -> k = d) |> Option.map snd
                | _ -> None

            let firstLiteral (v: uint32[] option) =
                v |> Option.bind (fun a -> if a.Length > 0 then Some a[0] else None)

            let nameOf (id: uint32) =
                match names.TryGetValue id with
                | true, n -> Some n
                | _ -> None

            let typeIns (id: uint32) =
                match types.TryGetValue id with
                | true, t -> Some t
                | _ -> None

            let pointee (pointerType: uint32) =
                match typeIns pointerType with
                | Some t when t.Opcode = Spirv.Op.TypePointer && t.Operands.Length >= 3 -> Some t.Operands[2]
                | _ -> None

            let constantValue (id: uint32) =
                match constants.TryGetValue id with
                | true, c when c.Opcode = Spirv.Op.Constant && c.Operands.Length = 3 -> Some(int64 c.Operands[2])
                | true, c when c.Opcode = Spirv.Op.Constant && c.Operands.Length = 4 ->
                    Some(int64 c.Operands[2] ||| (int64 c.Operands[3] <<< 32))
                | _ -> None

            let rec typeText (depth: int) (id: uint32) : string =
                match typeIns id with
                | _ when depth > 8 -> "..."
                | Some t ->
                    let o = t.Operands

                    match t.Opcode with
                    | Spirv.Op.TypeInt when o.Length >= 3 -> (if o[2] = 0u then "u" else "i") + string o[1]
                    | Spirv.Op.TypeFloat when o.Length >= 2 -> "f" + string o[1]
                    | Spirv.Op.TypeBool -> "bool"
                    | Spirv.Op.TypeVector when o.Length >= 3 -> sprintf "%s x%d" (typeText (depth + 1) o[1]) o[2]
                    | Spirv.Op.TypeArray when o.Length >= 3 ->
                        let n = constantValue o[2] |> Option.map string |> Option.defaultValue "?"

                        sprintf "%s[%s]" (typeText (depth + 1) o[1]) n
                    | Spirv.Op.TypeRuntimeArray when o.Length >= 2 -> typeText (depth + 1) o[1] + "[]"
                    | Spirv.Op.TypeStruct ->
                        nameOf id |> Option.map (fun n -> "struct " + n) |> Option.defaultValue "struct"
                    | Spirv.Op.TypePointer when o.Length >= 3 ->
                        sprintf "ptr %s %s" (Spirv.StorageClass.name o[1]) (typeText (depth + 1) o[2])
                    | op -> sprintf "type (opcode %d)" op
                | None -> sprintf "%%%d" id

            let rec sizeOf (depth: int) (id: uint32) : Result<int64, string> =
                match typeIns id with
                | _ when depth > 16 -> Error "type nesting too deep"
                | Some t ->
                    let o = t.Operands

                    match t.Opcode with
                    | Spirv.Op.TypeInt
                    | Spirv.Op.TypeFloat when o.Length >= 2 && o[1] % 8u = 0u -> Ok(int64 (o[1] / 8u))
                    | Spirv.Op.TypeBool -> Ok 4L
                    | Spirv.Op.TypeVector
                    | Spirv.Op.TypeMatrix when o.Length >= 3 ->
                        sizeOf (depth + 1) o[1] |> Result.map (fun e -> e * int64 o[2])
                    | Spirv.Op.TypeArray when o.Length >= 3 ->
                        match constantValue o[2] with
                        | Some n -> sizeOf (depth + 1) o[1] |> Result.map (fun e -> e * n)
                        | None -> Error(sprintf "the length of %s is not a constant" (typeText 0 id))
                    | Spirv.Op.TypeStruct ->
                        o
                        |> Array.skip 1
                        |> Array.fold
                            (fun acc mt ->
                                match acc, sizeOf (depth + 1) mt with
                                | Ok a, Ok b -> Ok(a + b)
                                | (Error _ as e), _
                                | _, (Error _ as e) -> e)
                            (Ok 0L)
                    | _ -> Error(sprintf "%s has no fixed size" (typeText 0 id))
                | None -> Error(sprintf "%%%d is not a type" id)

            let isFloatType (id: uint32) =
                match typeIns id with
                | Some t when t.Opcode = Spirv.Op.TypeFloat -> true
                | Some t when t.Opcode = Spirv.Op.TypeVector && t.Operands.Length >= 2 ->
                    match typeIns t.Operands[1] with
                    | Some e -> e.Opcode = Spirv.Op.TypeFloat
                    | None -> false
                | _ -> false

            // ---- module ----
            if m.Version <> Spirv.Version16 then
                add
                    SpirvRule.Module
                    "version %s, expected SPIR-V 1.6 (%s, -fspv-target-env=vulkan1.3)"
                    (hex m.Version)
                    (hex Spirv.Version16)

            match List.ofSeq memoryModels with
            | [ (0u, 1u) ] -> ()
            | [ (a, mm) ] -> add SpirvRule.Module "memory model (%d, %d), expected Logical GLSL450 (0, 1)" a mm
            | l -> add SpirvRule.Module "%d OpMemoryModel instructions, expected 1" l.Length

            // ---- capabilities and extensions ----
            for c in capabilities |> Seq.distinct do
                if not (List.contains c allowedCapabilities) then
                    add
                        SpirvRule.Capabilities
                        "capability %s is not allowed: the recipe's device enables no feature for it"
                        (Spirv.Capability.label c)

            for c in requiredCapabilities do
                if not (capabilities.Contains c) then
                    add SpirvRule.Capabilities "capability %s is missing" (Spirv.Capability.label c)

            for e in extensions |> Seq.distinct do
                if not (List.contains e allowedExtensions) then
                    add SpirvRule.Extensions "extension %s is not allowed" e

            for e in extInst |> Seq.distinct do
                if not (List.contains e allowedExtInstImports) then
                    add SpirvRule.Extensions "extended instruction set %s is not allowed" e

            // ---- entry point and execution mode ----
            match List.ofSeq entryPoints with
            | [ (model, name) ] ->
                if model <> Spirv.GLCompute then
                    add SpirvRule.EntryPoint "execution model %d, expected GLCompute (5)" model

                if name <> "main" then
                    add SpirvRule.EntryPoint "entry point '%s', expected 'main'" name
            | l -> add SpirvRule.EntryPoint "%d entry points, expected 1" l.Length

            let tx, ty, tz = kernel.Threads

            let localSize =
                match List.ofSeq modes with
                | [ (op, mode, lits) ] when
                    op = Spirv.Op.ExecutionMode
                    && mode = Spirv.ExecutionMode.LocalSize
                    && lits.Length = 3
                    ->
                    Some(int lits[0], int lits[1], int lits[2])
                | _ -> None

            match localSize with
            | Some(x, y, z) when (x, y, z) <> (tx, ty, tz) ->
                add
                    SpirvRule.ExecutionMode
                    "LocalSize %d x %d x %d, but the catalog's threads are %d x %d x %d"
                    x
                    y
                    z
                    tx
                    ty
                    tz
            | Some _ -> ()
            | None ->
                let listed =
                    modes
                    |> Seq.map (fun (_, mode, _) -> Spirv.ExecutionMode.name mode)
                    |> String.concat ", "

                add
                    SpirvRule.ExecutionMode
                    "the execution modes are [%s], expected exactly one, LocalSize %d x %d x %d"
                    listed
                    tx
                    ty
                    tz

            // ---- storage classes ----
            for _, id, sc in globals do
                if not (List.contains sc allowedGlobalStorage) then
                    add
                        SpirvRule.StorageClass
                        "variable %s has storage class %s"
                        (nameOf id |> Option.defaultValue (sprintf "%%%d" id))
                        (Spirv.StorageClass.name sc)

            // ---- the resource variable ----
            let bound =
                globals
                |> Seq.filter (fun (_, id, _) ->
                    (decoration id Spirv.Decoration.DescriptorSet).IsSome
                    || (decoration id Spirv.Decoration.Binding).IsSome)
                |> List.ofSeq

            let resources =
                [
                    for _, id, sc in bound do
                        let blockNonWritable =
                            pointee (typeOf[id])
                            |> Option.bind (fun arr ->
                                match typeIns arr with
                                | Some a when a.Opcode = Spirv.Op.TypeRuntimeArray && a.Operands.Length >= 2 ->
                                    Some a.Operands[1]
                                | _ -> Some arr)
                            |> Option.map (fun s ->
                                match typeIns s with
                                | Some st when st.Opcode = Spirv.Op.TypeStruct ->
                                    [ 0 .. st.Operands.Length - 2 ]
                                    |> List.exists (fun i ->
                                        (memberDecoration s (uint32 i) Spirv.Decoration.NonWritable).IsSome)
                                | _ -> false)
                            |> Option.defaultValue false

                        yield
                            {
                                Name = nameOf id
                                StorageClass = sc
                                Set = firstLiteral (decoration id Spirv.Decoration.DescriptorSet)
                                Binding = firstLiteral (decoration id Spirv.Decoration.Binding)
                                NonWritable = (decoration id Spirv.Decoration.NonWritable).IsSome || blockNonWritable
                            }
                ]

            let storageBuffers =
                globals
                |> Seq.filter (fun (_, _, sc) -> sc = Spirv.StorageClass.StorageBuffer)
                |> List.ofSeq

            match bound with
            | [ (ptr, id, sc) ] ->
                let r = resources.Head
                let label = r.Name |> Option.defaultValue (sprintf "%%%d" id)

                if sc <> Spirv.StorageClass.StorageBuffer then
                    add
                        SpirvRule.Resource
                        "%s has storage class %s, expected StorageBuffer"
                        label
                        (Spirv.StorageClass.name sc)

                if
                    r.Set <> Some SpirvBinding.DescriptorSet
                    || r.Binding <> Some SpirvBinding.Binding
                then
                    add
                        SpirvRule.Resource
                        "%s is at set %s binding %s, expected set %d binding %d"
                        label
                        (r.Set |> Option.map string |> Option.defaultValue "none")
                        (r.Binding |> Option.map string |> Option.defaultValue "none")
                        SpirvBinding.DescriptorSet
                        SpirvBinding.Binding

                if r.NonWritable then
                    add SpirvRule.Resource "%s is NonWritable; kernels write through LRM_BUF" label

                // The type: a runtime array of Block structs whose one member is a runtime array of u32 with stride 4
                // at offset 0 (RWByteAddressBuffer).
                let raw =
                    match pointee ptr |> Option.bind typeIns with
                    | Some arr when arr.Opcode = Spirv.Op.TypeRuntimeArray && arr.Operands.Length >= 2 ->
                        let s = arr.Operands[1]

                        match typeIns s with
                        | Some st when
                            st.Opcode = Spirv.Op.TypeStruct
                            && st.Operands.Length = 2
                            && (decoration s Spirv.Decoration.Block).IsSome
                            && firstLiteral (memberDecoration s 0u Spirv.Decoration.Offset) = Some 0u
                            ->
                            let words = st.Operands[1]

                            match typeIns words with
                            | Some w when w.Opcode = Spirv.Op.TypeRuntimeArray && w.Operands.Length >= 2 ->
                                typeText 0 w.Operands[1] = "u32"
                                && firstLiteral (decoration words Spirv.Decoration.ArrayStride) = Some 4u
                            | _ -> false
                        | _ -> false
                    | _ -> false

                if not raw then
                    add
                        SpirvRule.Resource
                        "%s is %s, expected a runtime array of raw buffers (RWByteAddressBuffer g_buf[])"
                        label
                        (typeText 0 ptr)
            | [] -> add SpirvRule.Resource "no resource variable, expected g_buf[] at set 0 binding 0"
            | l ->
                let listed =
                    resources
                    |> List.map (fun r ->
                        sprintf
                            "%s (set %s, binding %s)"
                            (r.Name |> Option.defaultValue "?")
                            (r.Set |> Option.map string |> Option.defaultValue "-")
                            (r.Binding |> Option.map string |> Option.defaultValue "-"))
                    |> String.concat ", "

                add SpirvRule.Resource "%d resource variables, expected exactly 1: %s" l.Length listed

            for _, id, _ in storageBuffers do
                if not (bound |> List.exists (fun (_, b, _) -> b = id)) then
                    add
                        SpirvRule.Resource
                        "StorageBuffer variable %s has no set and binding"
                        (nameOf id |> Option.defaultValue (sprintf "%%%d" id))

            // ---- the push block ----
            let pushVars =
                globals
                |> Seq.filter (fun (_, _, sc) -> sc = Spirv.StorageClass.PushConstant)
                |> List.ofSeq

            let mutable pushName = None
            let mutable pushMembers = []
            let mutable pushBytes = 0

            match pushVars with
            | [ (ptr, _, _) ] ->
                match pointee ptr with
                | Some s when (typeIns s |> Option.map _.Opcode) = Some Spirv.Op.TypeStruct ->
                    let st = types[s]
                    let memberTypes = Array.skip 1 st.Operands
                    pushName <- nameOf s

                    pushMembers <-
                        [
                            for i in 0 .. memberTypes.Length - 1 ->
                                let ui = uint32 i

                                {
                                    Name =
                                        match memberNames.TryGetValue(struct (s, ui)) with
                                        | true, n -> Some n
                                        | _ -> None
                                    Offset =
                                        firstLiteral (memberDecoration s ui Spirv.Decoration.Offset) |> Option.map int
                                    Type = typeText 0 memberTypes[i]
                                }
                        ]

                    pushBytes <-
                        Array.zip (Array.ofList pushMembers) memberTypes
                        |> Array.fold
                            (fun acc (pm, t) ->
                                match pm.Offset, sizeOf 0 t with
                                | Some o, Ok size -> max acc (o + int size)
                                | _ -> acc)
                            0

                    if (decoration s Spirv.Decoration.Block).IsNone then
                        add SpirvRule.PushBlock "the push block is not decorated Block"

                    match pushName with
                    | Some n when n = pushBlockName kernel -> ()
                    | Some n ->
                        add
                            SpirvRule.PushBlock
                            "the push block is %s, expected %s (LRM_ARGS(%s))"
                            n
                            (pushBlockName kernel)
                            kernel.Name
                    | None -> add SpirvRule.PushBlock "the push block has no OpName, expected %s" (pushBlockName kernel)

                    let expected = expectedPushMembers kernel

                    if pushMembers.Length <> expected.Length then
                        add
                            SpirvRule.PushBlock
                            "the push block has %d members, expected %d (lrm_groups_x and %d catalog fields)"
                            pushMembers.Length
                            expected.Length
                            kernel.Fields.Length

                    List.zip (List.truncate expected.Length pushMembers) (List.truncate pushMembers.Length expected)
                    |> List.iteri (fun i (pm, (ename, etype)) ->
                        if pm.Offset <> Some(4 * i) then
                            add
                                SpirvRule.PushBlock
                                "member %d (%s) is at offset %s, expected %d (argument word %d)"
                                i
                                ename
                                (pm.Offset |> Option.map string |> Option.defaultValue "none")
                                (4 * i)
                                i

                        if pm.Type <> etype then
                            add SpirvRule.PushBlock "member %d (%s) is %s, expected %s" i ename pm.Type etype

                        if pm.Name <> Some ename then
                            add
                                SpirvRule.PushBlock
                                "member %d is named %s, expected %s"
                                i
                                (pm.Name |> Option.defaultValue "(no name)")
                                ename)

                    if int64 pushBytes > int64 SpirvBinding.PushDataBytes then
                        add
                            SpirvRule.PushBlock
                            "the push block spans %d bytes, more than the %d-byte ArgBlock"
                            pushBytes
                            SpirvBinding.PushDataBytes
                | _ -> add SpirvRule.PushBlock "the push-constant variable is %s, expected a struct" (typeText 0 ptr)
            | [] -> add SpirvRule.PushBlock "no push-constant block, expected LRM_ARGS(%s)" kernel.Name
            | l -> add SpirvRule.PushBlock "%d push-constant variables, expected 1" l.Length

            // ---- built-ins ----
            let builtIns = ResizeArray<uint32>()

            for ptr, id, sc in globals do
                if sc = Spirv.StorageClass.Input then
                    let label = nameOf id |> Option.defaultValue (sprintf "%%%d" id)

                    let found =
                        match firstLiteral (decoration id Spirv.Decoration.BuiltIn) with
                        | Some b -> [ b ]
                        | None ->
                            match pointee ptr |> Option.bind typeIns with
                            | Some st when st.Opcode = Spirv.Op.TypeStruct ->
                                [
                                    for i in 0 .. st.Operands.Length - 2 do
                                        match
                                            firstLiteral (
                                                memberDecoration st.Operands[0] (uint32 i) Spirv.Decoration.BuiltIn
                                            )
                                        with
                                        | Some b -> b
                                        | None -> ()
                                ]
                            | _ -> []

                    if found.IsEmpty then
                        add SpirvRule.BuiltIns "Input variable %s is not a built-in" label

                    for b in found do
                        builtIns.Add b

                        if b = Spirv.BuiltIn.GlobalInvocationId then
                            add
                                SpirvRule.BuiltIns
                                "%s uses GlobalInvocationId (SV_DispatchThreadID), which is wrong on folded grids"
                                label
                        elif not (List.contains b allowedBuiltIns) then
                            add
                                SpirvRule.BuiltIns
                                "%s uses built-in %s; kernels take only SV_GroupID, SV_GroupThreadID and SV_GroupIndex"
                                label
                                (Spirv.BuiltIn.name b)

            // ---- atomics, wave ops, types ----
            let valueIsFloat (id: uint32) =
                match typeOf.TryGetValue id with
                | true, t -> isFloatType t
                | _ -> false

            let pointsToFloat (id: uint32) =
                match typeOf.TryGetValue id with
                | true, t -> pointee t |> Option.exists isFloatType
                | _ -> false

            // OpAtomicStore has no result type: look at its pointer and value. The flag operations act on integers.
            // Every other atomic has its result type first, which is the type of the memory it changes.
            let onFloat (a: SpirvInstruction) =
                let o = a.Operands

                match a.Opcode with
                | Spirv.Op.AtomicFAddEXT
                | Spirv.Op.AtomicFMinEXT
                | Spirv.Op.AtomicFMaxEXT -> true
                | Spirv.Op.AtomicStore -> o.Length >= 4 && (pointsToFloat o[0] || valueIsFloat o[3])
                | Spirv.Op.AtomicFlagClear
                | Spirv.Op.AtomicFlagTestAndSet -> false
                | _ -> o.Length >= 1 && isFloatType o[0]

            for a in atomics do
                if onFloat a then
                    add SpirvRule.FloatAtomics "word %d: %s on a float" a.Word (Spirv.opName a.Opcode)
                else
                    add SpirvRule.Atomics "word %d: %s; contracts v1 allow no atomics" a.Word (Spirv.opName a.Opcode)

            for op in waveOps |> Seq.distinct do
                add SpirvRule.WaveOps "wave or group operation %s" (Spirv.opName op)

            for KeyValue(id, t) in types |> Seq.sortBy (fun kv -> kv.Key) do
                let o = t.Operands

                match t.Opcode with
                | Spirv.Op.TypeInt
                | Spirv.Op.TypeFloat when o.Length >= 2 ->
                    if o[1] = 16u then
                        add SpirvRule.SixteenBit "16-bit type %s" (typeText 0 id)
                    elif o[1] <> 32u then
                        add SpirvRule.ScalarWidth "%d-bit type %s; kernels use 32-bit scalars only" o[1] (typeText 0 id)
                | Spirv.Op.TypeImage
                | Spirv.Op.TypeSampler
                | Spirv.Op.TypeSampledImage ->
                    add SpirvRule.TypedViews "typed view or sampler type (opcode %d)" t.Opcode
                | _ -> ()

            for c in capabilities |> Seq.distinct do
                if List.contains c sixteenBitCapabilities then
                    add SpirvRule.SixteenBit "16-bit capability %s" (Spirv.Capability.label c)

            // ---- groupshared ----
            let mutable workgroupBytes = 0L

            for ptr, id, sc in globals do
                if sc = Spirv.StorageClass.Workgroup && workgroupBytes >= 0L then
                    match pointee ptr |> Option.map (sizeOf 0) with
                    | Some(Ok n) -> workgroupBytes <- workgroupBytes + n
                    | Some(Error e) ->
                        add
                            SpirvRule.WorkgroupSize
                            "cannot size Workgroup variable %s: %s"
                            (nameOf id |> Option.defaultValue (sprintf "%%%d" id))
                            e

                        workgroupBytes <- -1L
                    | None ->
                        add SpirvRule.WorkgroupSize "Workgroup variable %%%d has no pointer type" id
                        workgroupBytes <- -1L

            if workgroupBytes > MaxWorkgroupBytes then
                add
                    SpirvRule.WorkgroupSize
                    "Workgroup variables take %d bytes, more than %d (32 KiB)"
                    workgroupBytes
                    MaxWorkgroupBytes

            let order = SpirvRule.all |> List.mapi (fun i r -> r, i) |> dict

            {
                Kernel = kernel.Name
                Facts =
                    Some
                        {
                            Version = m.Version
                            Capabilities = List.ofSeq capabilities
                            Extensions = List.ofSeq extensions
                            ExtInstImports = List.ofSeq extInst
                            LocalSize = localSize
                            Resources = resources
                            PushBlock = pushName
                            PushMembers = pushMembers
                            PushBytes = pushBytes
                            WorkgroupBytes = workgroupBytes
                            BuiltIns = builtIns |> Seq.distinct |> Seq.sort |> List.ofSeq
                        }
                Findings =
                    findings
                    |> Seq.mapi (fun i f -> i, f)
                    |> Seq.sortBy (fun (i, f) -> order[f.Rule], i)
                    |> Seq.map snd
                    |> List.ofSeq
            }

    /// "[rule] message".
    let describe (f: SpirvFinding) =
        sprintf "[%s] %s" (SpirvRule.name f.Rule) f.Message

    /// The distinct rules a report breaks, in rule order.
    let rules (r: SpirvReport) =
        r.Findings |> List.map _.Rule |> List.distinct

    /// Raises GpuContractException listing every finding when the blob breaks a rule.
    let check (kernel: KernelSpec) (bytes: byte[]) =
        let r = inspect kernel bytes

        if not r.Passed then
            Fail.contract
                "%s (spirv) breaks the SPIR-V rules:\n  %s"
                kernel.Name
                (r.Findings |> List.map describe |> String.concat "\n  ")

        r

    /// Inspects every SPIR-V blob of a set, in kernel order. A blob whose kernel is not in `catalog` gets a Module
    /// finding.
    let inspectSet (catalog: Catalog) (set: ShaderSet) : SpirvReport list =
        [
            for b in ShaderSet.blobsFor ShaderTarget.Spirv set do
                match Catalog.tryFind b.Kernel catalog with
                | Some k -> inspect k b.Bytes
                | None ->
                    {
                        Kernel = b.Kernel
                        Facts = None
                        Findings =
                            [
                                {
                                    Rule = SpirvRule.Module
                                    Message = sprintf "%s is not a catalog kernel" b.Kernel
                                }
                            ]
                    }
        ]

    /// Emits the report: a gpu.shader.spirv_check Check, and the gpu.shader.push_bytes and gpu.shader.workgroup_bytes
    /// gauges when the module could be read.
    let record (r: SpirvReport) =
        let attrs =
            [
                Attr.str TelemetryKeys.Kernel r.Kernel
                Attr.str "target" (ShaderTarget.name ShaderTarget.Spirv)
            ]

        Telemetry.check
            GpuTelemetry.SpirvCheck
            r.Passed
            (attrs
             @ [
                 Attr.int "findings" r.Findings.Length
                 Attr.str "rules" (rules r |> List.map SpirvRule.name |> String.concat ",")
             ])

        match r.Facts with
        | Some f ->
            Telemetry.gauge GpuTelemetry.PushBytes (float f.PushBytes) TelemetryUnits.Bytes attrs

            if f.WorkgroupBytes >= 0L then
                Telemetry.gauge GpuTelemetry.WorkgroupBytes (float f.WorkgroupBytes) TelemetryUnits.Bytes attrs
        | None -> ()

    /// A Markdown table of reports: kernel, LocalSize, push block, groupshared bytes, capabilities and the verdict.
    let markdown (reports: SpirvReport list) : string =
        let sb = StringBuilder()
        let line (s: string) = sb.Append(s).Append('\n') |> ignore
        line "| Kernel | LocalSize | Push block (words, bytes) | Workgroup bytes | Capabilities | Result |"
        line "| --- | --- | --- | ---: | --- | --- |"

        for r in reports do
            match r.Facts with
            | Some f ->
                let ls =
                    match f.LocalSize with
                    | Some(x, y, z) -> sprintf "%dx%dx%d" x y z
                    | None -> "-"

                let caps = f.Capabilities |> List.map Spirv.Capability.name |> String.concat ", "

                let verdict =
                    if r.Passed then
                        "pass"
                    else
                        "FAIL: " + (rules r |> List.map SpirvRule.name |> String.concat ", ")

                line (
                    sprintf
                        "| `%s` | %s | %d, %d | %d | %s | %s |"
                        r.Kernel
                        ls
                        f.PushMembers.Length
                        f.PushBytes
                        f.WorkgroupBytes
                        caps
                        verdict
                )
            | None -> line (sprintf "| `%s` | - | - | - | - | FAIL: module |" r.Kernel)

        sb.ToString()
