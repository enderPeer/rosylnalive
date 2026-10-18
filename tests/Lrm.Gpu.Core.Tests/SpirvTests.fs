/// The SPIR-V binding (SpirvBinding), the SPIR-V reader and the SPIR-V inspector (ShaderBuildAndSourceConvention
/// v1.1). Every embedded SPIR-V blob must pass; each rule is shown to fire by mutating a real blob word by word, so
/// these tests need no DXC. CI's shader step adds real DXC fixtures (src/Lrm.Shaders.Build/fixtures/spirv).
module Lrm.Gpu.Core.Tests.SpirvTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common
open Lrm.Shaders
open Lrm.Testing

let private kernel (name: string) =
    Kernels.all |> Array.find (fun k -> k.Name = name)

let private blob (name: string) =
    match ShaderSet.tryBlob name ShaderTarget.Spirv (Shaders.embedded ()) with
    | Some b -> b.Bytes
    | None -> failtestf "no SPIR-V blob for %s" name

let private parse (bytes: byte[]) =
    match Spirv.parse bytes with
    | Ok m -> m
    | Error e -> failtestf "parse: %s" e

let private ins (op: int) (operands: uint32 list) = Spirv.instruction op operands

/// The id an OpName gives `name`.
let private idNamed (m: SpirvModule) (name: string) =
    m.Instructions
    |> List.pick (fun i ->
        if
            i.Opcode = Spirv.Op.Name
            && (Spirv.literalString i.Operands 1 |> Option.map fst) = Some name
        then
            Some i.Operands[0]
        else
            None)

/// The result id of the first instruction with `opcode` whose operands after the result id start with `tail`.
let private typeId (m: SpirvModule) (opcode: int) (tail: uint32 list) =
    m.Instructions
    |> List.pick (fun i ->
        if
            i.Opcode = opcode
            && List.ofArray i.Operands.[1..] |> List.truncate tail.Length = tail
        then
            Some i.Operands[0]
        else
            None)

let private f32 m = typeId m Spirv.Op.TypeFloat [ 32u ]
let private u32 m = typeId m Spirv.Op.TypeInt [ 32u; 0u ]

/// The type of a module-scope variable.
let private varType (m: SpirvModule) (id: uint32) =
    m.Instructions
    |> List.pick (fun i ->
        if i.Opcode = Spirv.Op.Variable && i.Operands[1] = id then
            Some i.Operands[0]
        else
            None)

/// The variable that carries built-in `b`.
let private builtInVar (m: SpirvModule) (b: uint32) =
    m.Instructions
    |> List.pick (fun i ->
        if
            i.Opcode = Spirv.Op.Decorate
            && i.Operands[1] = Spirv.Decoration.BuiltIn
            && i.Operands[2] = b
        then
            Some i.Operands[0]
        else
            None)

/// `n` fresh ids, and the module with its bound raised past them.
let private fresh (n: int) (m: SpirvModule) =
    [ for k in 0 .. n - 1 -> m.Bound + uint32 k ], { m with Bound = m.Bound + uint32 n }

let private insertBefore (pred: SpirvInstruction -> bool) (added: SpirvInstruction list) (m: SpirvModule) =
    let i = m.Instructions |> List.findIndex pred

    { m with
        Instructions = List.take i m.Instructions @ added @ List.skip i m.Instructions
    }

/// Adds module-scope instructions (before the first function; the inspector reads them wherever they are).
let private addGlobal (added: SpirvInstruction list) (m: SpirvModule) =
    insertBefore (fun i -> i.Opcode = Spirv.Op.Function) added m

/// Adds instructions to the body of the entry point, after its first OpLabel (248).
let private addInBody (added: SpirvInstruction list) (m: SpirvModule) =
    let i = m.Instructions |> List.findIndex (fun i -> i.Opcode = 248)

    { m with
        Instructions = List.take (i + 1) m.Instructions @ added @ List.skip (i + 1) m.Instructions
    }

let private mapWhere (pred: SpirvInstruction -> bool) (f: uint32[] -> uint32[]) (m: SpirvModule) =
    let mutable hits = 0

    let out =
        { m with
            Instructions =
                m.Instructions
                |> List.map (fun i ->
                    if pred i then
                        hits <- hits + 1

                        { i with
                            Operands = f (Array.copy i.Operands)
                        }
                    else
                        i)
        }

    if hits = 0 then
        failtest "the mutation matched no instruction"

    out

let private removeWhere (pred: SpirvInstruction -> bool) (m: SpirvModule) =
    let kept = m.Instructions |> List.filter (pred >> not)

    if kept.Length = m.Instructions.Length then
        failtest "the removal matched no instruction"

    { m with Instructions = kept }

let private set (k: int) (v: uint32) (o: uint32[]) =
    o[k] <- v
    o

let private isDecoration (target: uint32) (d: uint32) (i: SpirvInstruction) =
    i.Opcode = Spirv.Op.Decorate && i.Operands[0] = target && i.Operands[1] = d

let private isMemberDecoration (s: uint32) (mem: uint32) (d: uint32) (i: SpirvInstruction) =
    i.Opcode = Spirv.Op.MemberDecorate
    && i.Operands[0] = s
    && i.Operands[1] = mem
    && i.Operands[2] = d

let private ruleNames (r: SpirvReport) =
    SpirvInspector.rules r |> List.map SpirvRule.name |> List.sort

let private expectRules (expected: string list) (r: SpirvReport) (what: string) =
    Expect.equal
        (ruleNames r)
        (List.sort expected)
        (sprintf "%s: %s" what (r.Findings |> List.map SpirvInspector.describe |> String.concat "; "))

/// One word-level mutation of scale_f32.spv and the rules it must break, exactly.
type private Mutation =
    {
        Name: string
        Expect: string list
        Apply: SpirvModule -> SpirvModule
    }

let private mutations: Mutation list =
    let gbuf m = idNamed m "g_buf"

    let push m =
        idNamed m "type.ConstantBuffer.lrm_args_scale_f32"

    let pushVar m = idNamed m "lrm_args"
    let rwb m = idNamed m "type.RWByteAddressBuffer"
    let main m = idNamed m "main"

    let m (name: string) (expect: string list) (apply: SpirvModule -> SpirvModule) =
        {
            Name = name
            Expect = expect
            Apply = apply
        }

    let withFresh n (f: uint32 list -> SpirvModule -> SpirvModule) (x: SpirvModule) =
        let ids, x' = fresh n x
        f ids x'

    /// A Workgroup array of `count` elements of `elem`.
    let workgroupArray (count: uint32) (elem: SpirvModule -> uint32) =
        withFresh 4 (fun ids x ->
            match ids with
            | [ c; arr; ptr; v ] ->
                addGlobal
                    [
                        ins Spirv.Op.Constant [ u32 x; c; count ]
                        ins Spirv.Op.TypeArray [ arr; elem x; c ]
                        ins Spirv.Op.TypePointer [ ptr; Spirv.StorageClass.Workgroup; arr ]
                        ins Spirv.Op.Variable [ ptr; v; Spirv.StorageClass.Workgroup ]
                    ]
                    x
            | _ -> failwith "ids")

    let globalVariable (sc: uint32) =
        withFresh 2 (fun ids x ->
            match ids with
            | [ p; v ] ->
                addGlobal
                    [
                        ins Spirv.Op.TypePointer [ p; sc; f32 x ]
                        ins Spirv.Op.Variable [ p; v; sc ]
                    ]
                    x
            | _ -> failwith "ids")

    let inBody (op: int) (operands: SpirvModule -> uint32 -> uint32 list) =
        withFresh 1 (fun ids x -> addInBody [ ins op (operands x ids.Head) ] x)

    let newType (op: int) (tail: uint32 list) =
        withFresh 1 (fun ids x -> addGlobal [ ins op (ids.Head :: tail) ] x)

    /// The id of a PushConstant access chain to a float (the load of `a`).
    let floatPointer (x: SpirvModule) =
        let ptrFloat =
            typeId x Spirv.Op.TypePointer [ Spirv.StorageClass.PushConstant; f32 x ]

        x.Instructions
        |> List.pick (fun i ->
            if i.Opcode = Spirv.Op.AccessChain && i.Operands[0] = ptrFloat then
                Some i.Operands[1]
            else
                None)

    [
        // resource
        m "binding 1" [ "resource" ] (fun x -> mapWhere (isDecoration (gbuf x) Spirv.Decoration.Binding) (set 2 1u) x)
        m "set 1" [ "resource" ] (fun x -> mapWhere (isDecoration (gbuf x) Spirv.Decoration.DescriptorSet) (set 2 1u) x)
        m "g_buf NonWritable" [ "resource" ] (fun x ->
            addGlobal [ ins Spirv.Op.Decorate [ gbuf x; Spirv.Decoration.NonWritable ] ] x)
        m "buffer member NonWritable" [ "resource" ] (fun x ->
            addGlobal [ ins Spirv.Op.MemberDecorate [ rwb x; 0u; Spirv.Decoration.NonWritable ] ] x)
        m
            "a second buffer at binding 1"
            [ "resource" ]
            (withFresh 1 (fun ids x ->
                let v = ids.Head

                addGlobal
                    [
                        ins Spirv.Op.Variable [ varType x (gbuf x); v; Spirv.StorageClass.StorageBuffer ]
                        ins Spirv.Op.Decorate [ v; Spirv.Decoration.DescriptorSet; 0u ]
                        ins Spirv.Op.Decorate [ v; Spirv.Decoration.Binding; 1u ]
                    ]
                    x))
        m
            "a StorageBuffer variable without set and binding"
            [ "resource" ]
            (withFresh 1 (fun ids x ->
                addGlobal
                    [
                        ins Spirv.Op.Variable [ varType x (gbuf x); ids.Head; Spirv.StorageClass.StorageBuffer ]
                    ]
                    x))
        m
            "g_buf is one buffer, not an array"
            [ "resource" ]
            (withFresh 1 (fun ids x ->
                let p = ids.Head

                addGlobal [ ins Spirv.Op.TypePointer [ p; Spirv.StorageClass.StorageBuffer; rwb x ] ] x
                |> mapWhere (fun i -> i.Opcode = Spirv.Op.Variable && i.Operands[1] = gbuf x) (set 0 p)))
        m "no resource variable" [ "resource" ] (fun x ->
            removeWhere (fun i -> i.Opcode = Spirv.Op.Variable && i.Operands[1] = gbuf x) x)
        // push block
        m "push offset of word 2" [ "push-block" ] (fun x ->
            mapWhere (isMemberDecoration (push x) 2u Spirv.Decoration.Offset) (set 3 12u) x)
        m "push word 4 typed u32" [ "push-block" ] (fun x ->
            mapWhere (fun i -> i.Opcode = Spirv.Op.TypeStruct && i.Operands[0] = push x) (set 5 (u32 x)) x)
        m "push member renamed" [ "push-block" ] (fun x ->
            mapWhere
                (fun i -> i.Opcode = Spirv.Op.MemberName && i.Operands[0] = push x && i.Operands[1] = 1u)
                (fun o -> Array.append o[0..1] (Array.ofList (Spirv.stringOperands "z")))
                x)
        m "push block not Block" [ "push-block" ] (fun x ->
            removeWhere (isDecoration (push x) Spirv.Decoration.Block) x)
        m "no push block" [ "push-block" ] (fun x ->
            removeWhere (fun i -> i.Opcode = Spirv.Op.Variable && i.Operands[1] = pushVar x) x)
        m
            "a second push block"
            [ "push-block" ]
            (withFresh 1 (fun ids x ->
                addGlobal
                    [
                        ins Spirv.Op.Variable [ varType x (pushVar x); ids.Head; Spirv.StorageClass.PushConstant ]
                    ]
                    x))
        // execution mode and entry point
        m "LocalSize 128" [ "execution-mode" ] (fun x ->
            mapWhere (fun i -> i.Opcode = Spirv.Op.ExecutionMode) (set 2 128u) x)
        m "DenormPreserve mode" [ "execution-mode" ] (fun x ->
            addGlobal [ ins Spirv.Op.ExecutionMode [ main x; 4459u; 32u ] ] x)
        m "LocalSizeId" [ "execution-mode" ] (fun x ->
            { x with
                Instructions =
                    x.Instructions
                    |> List.map (fun i ->
                        if i.Opcode = Spirv.Op.ExecutionMode then
                            { i with
                                Opcode = Spirv.Op.ExecutionModeId
                                Operands = Array.copy i.Operands |> set 1 38u
                            }
                        else
                            i)
            })
        m "two entry points" [ "entry-point" ] (fun x ->
            addGlobal [ x.Instructions |> List.find (fun i -> i.Opcode = Spirv.Op.EntryPoint) ] x)
        m "entry point named kernel_main" [ "entry-point" ] (fun x ->
            mapWhere
                (fun i -> i.Opcode = Spirv.Op.EntryPoint)
                (fun o ->
                    let _, used = (Spirv.literalString o 2).Value

                    Array.concat [ o[0..1]; Array.ofList (Spirv.stringOperands "kernel_main"); o[2 + used ..] ])
                x)
        m "Kernel execution model" [ "entry-point" ] (fun x ->
            mapWhere (fun i -> i.Opcode = Spirv.Op.EntryPoint) (set 0 6u) x)
        // capabilities, extensions, module
        m "Float16 capability" [ "capabilities"; "16-bit-types" ] (fun x ->
            addGlobal [ ins Spirv.Op.Capability [ 9u ] ] x)
        m "Int64 capability" [ "capabilities" ] (fun x -> addGlobal [ ins Spirv.Op.Capability [ 11u ] ] x)
        m "no RuntimeDescriptorArray" [ "capabilities" ] (fun x ->
            removeWhere (fun i -> i.Opcode = Spirv.Op.Capability && i.Operands[0] = 5302u) x)
        m "an extension" [ "extensions" ] (fun x ->
            addGlobal [ ins Spirv.Op.Extension (Spirv.stringOperands "SPV_KHR_shader_clock") ] x)
        m
            "an extended instruction set"
            [ "extensions" ]
            (withFresh 1 (fun ids x ->
                addGlobal
                    [
                        ins Spirv.Op.ExtInstImport (ids.Head :: Spirv.stringOperands "NonSemantic.DebugPrintf")
                    ]
                    x))
        m "SPIR-V 1.5" [ "module" ] (fun x -> { x with Version = 0x00010500u })
        m "Vulkan memory model" [ "module" ] (fun x -> mapWhere (fun i -> i.Opcode = Spirv.Op.MemoryModel) (set 1 3u) x)
        m "a string without its terminator" [ "module" ] (fun x ->
            addGlobal [ ins Spirv.Op.Name [ gbuf x; 0x41414141u ] ] x)
        m "a decoration without its operands" [ "module" ] (fun x -> addGlobal [ ins Spirv.Op.Decorate [ gbuf x ] ] x)
        // atomics
        m
            "OpAtomicExchange on a float"
            [ "float-atomics" ]
            (inBody Spirv.Op.AtomicExchange (fun x r -> [ f32 x; r; gbuf x; 2u; 0u; gbuf x ]))
        m
            "OpAtomicFAddEXT"
            [ "float-atomics" ]
            (inBody Spirv.Op.AtomicFAddEXT (fun x r -> [ f32 x; r; gbuf x; 2u; 0u; gbuf x ]))
        m "OpAtomicStore to a float" [ "float-atomics" ] (fun x ->
            addInBody [ ins Spirv.Op.AtomicStore [ floatPointer x; 2u; 0u; gbuf x ] ] x)
        m "OpAtomicIAdd" [ "atomics" ] (inBody Spirv.Op.AtomicIAdd (fun x r -> [ u32 x; r; gbuf x; 2u; 0u; gbuf x ]))
        // types, wave ops
        m "a 16-bit int type" [ "16-bit-types" ] (newType Spirv.Op.TypeInt [ 16u; 0u ])
        m "a 64-bit float type" [ "scalar-width" ] (newType Spirv.Op.TypeFloat [ 64u ])
        m "an 8-bit int type" [ "scalar-width" ] (newType Spirv.Op.TypeInt [ 8u; 1u ])
        m "an image type" [ "typed-views" ] (fun x -> newType Spirv.Op.TypeImage [ f32 x; 0u; 0u; 0u; 0u; 2u; 0u ] x)
        m "a sampler type" [ "typed-views" ] (newType Spirv.Op.TypeSampler [])
        m
            "OpGroupNonUniformFAdd"
            [ "wave-ops" ]
            (inBody Spirv.Op.GroupNonUniformFAdd (fun x r -> [ f32 x; r; 3u; 0u; gbuf x ]))
        // groupshared
        m "40 KiB of Workgroup" [ "workgroup-size" ] (workgroupArray 10240u f32)
        m "exactly 32 KiB of Workgroup" [] (workgroupArray 8192u f32)
        m
            "32 KiB + 4 in two Workgroup variables"
            [ "workgroup-size" ]
            (workgroupArray 8192u f32 >> workgroupArray 1u u32)
        m
            "a Workgroup array with a spec-constant length"
            [ "workgroup-size" ]
            (withFresh 4 (fun ids x ->
                match ids with
                | [ c; arr; ptr; v ] ->
                    addGlobal
                        [
                            ins Spirv.Op.SpecConstant [ u32 x; c; 16u ]
                            ins Spirv.Op.TypeArray [ arr; f32 x; c ]
                            ins Spirv.Op.TypePointer [ ptr; Spirv.StorageClass.Workgroup; arr ]
                            ins Spirv.Op.Variable [ ptr; v; Spirv.StorageClass.Workgroup ]
                        ]
                        x
                | _ -> failwith "ids"))
        // storage classes and built-ins
        m "a Uniform variable" [ "storage-class" ] (globalVariable Spirv.StorageClass.Uniform)
        m "an Output variable" [ "storage-class" ] (globalVariable Spirv.StorageClass.Output)
        m "a module-scope Function variable" [ "storage-class" ] (globalVariable Spirv.StorageClass.Function)
        m "a Private variable (allowed)" [] (globalVariable Spirv.StorageClass.Private)
        m
            "a Private variable inside the function"
            [ "storage-class" ]
            (withFresh 2 (fun ids x ->
                match ids with
                | [ p; v ] ->
                    addGlobal [ ins Spirv.Op.TypePointer [ p; Spirv.StorageClass.Private; f32 x ] ] x
                    |> addInBody [ ins Spirv.Op.Variable [ p; v; Spirv.StorageClass.Private ] ]
                | _ -> failwith "ids"))
        m "GlobalInvocationId" [ "built-ins" ] (fun x ->
            mapWhere (isDecoration (builtInVar x Spirv.BuiltIn.WorkgroupId) Spirv.Decoration.BuiltIn) (set 2 28u) x)
        m "SubgroupSize" [ "built-ins" ] (fun x ->
            mapWhere (isDecoration (builtInVar x Spirv.BuiltIn.WorkgroupId) Spirv.Decoration.BuiltIn) (set 2 36u) x)
        m "an Input variable without a built-in" [ "built-ins" ] (fun x ->
            removeWhere (isDecoration (builtInVar x Spirv.BuiltIn.WorkgroupId) Spirv.Decoration.BuiltIn) x)
    ]

/// Facts of the contracts-v1.1 selftests: LocalSize, push-block words and bytes, built-ins.
let private selftestFacts =
    [
        "args_echo", (32, 1, 1), 32, 128, [ 27u ]
        "bf16_unpack", (256, 1, 1), 4, 16, [ 26u; 27u ]
        "groups_echo", (1, 1, 1), 3, 12, [ 26u ]
        "scale_f32", (256, 1, 1), 5, 20, [ 26u; 27u ]
        "table_echo", (16, 1, 1), 3, 12, [ 27u ]
    ]

let tests =
    testList
        "spirv"
        [
            testCase "SpirvBinding holds the recipe's mapping values"
            <| fun () ->
                Expect.equal SpirvBinding.DescriptorSet 0u "set 0"
                Expect.equal SpirvBinding.Binding 0u "binding 0"
                Expect.equal SpirvBinding.FirstBinding 0u "firstBinding 0"
                Expect.equal SpirvBinding.BindingCount 1u "bindingCount 1"
                Expect.equal SpirvBinding.ResourceMask 0xC0u "READ_ONLY | READ_WRITE storage buffer"
                Expect.equal SpirvBinding.MappingSourceHeapWithConstantOffset 0u "HEAP_WITH_CONSTANT_OFFSET"
                Expect.equal SpirvBinding.HeapOffset 0u "heapOffset 0"
                Expect.equal SpirvBinding.PushDataOffset 0u "push data at offset 0"
                Expect.equal SpirvBinding.PushDataBytes (uint32 ArgBlock.ByteCount) "the whole ArgBlock"

                for i in 0..63 do
                    Expect.equal (SpirvBinding.resolve 16u i) (16L * int64 i) "slot i at 16 * i (RTX 4080)"
                    Expect.equal (SpirvBinding.resolve 32u i) (32L * int64 i) "the stride is the descriptor size"

                raisesContract "outside" (fun () -> SpirvBinding.resolve 16u 64)
                raisesContract "positive" (fun () -> SpirvBinding.resolve 0u 1)

            testCase "the SPIR-V flags are the recipe's"
            <| fun () ->
                let flags = ShaderSet.flagsFor ShaderTarget.Spirv (Shaders.embedded ())
                Expect.equal (SpirvBinding.spirvFlagProblems flags) [] "the embedded set's flags"

                let has (fragment: string) (ps: string list) =
                    Expect.isTrue (ps |> List.exists (fun p -> p.Contains fragment)) (sprintf "'%s' in %A" fragment ps)

                has
                    "-fvk-bind-resource-heap"
                    (SpirvBinding.spirvFlagProblems (flags @ [ "-fvk-bind-resource-heap"; "0"; "0" ]))

                has
                    "-fspv-use-descriptor-heap"
                    (SpirvBinding.spirvFlagProblems (flags @ [ "-fspv-use-descriptor-heap" ]))

                has "-enable-16bit-types" (SpirvBinding.spirvFlagProblems (flags @ [ "-enable-16bit-types" ]))
                has "-denorm" (SpirvBinding.spirvFlagProblems (flags @ [ "-denorm"; "preserve" ]))
                has "lack -spirv" (SpirvBinding.spirvFlagProblems (flags |> List.filter ((<>) "-spirv")))

            testCase "lrm.hlsli declares the final binding"
            <| fun () ->
                let text =
                    File.ReadAllText(Path.Combine(Env.repoRoot (), "src", "Lrm.Shaders", "include", "lrm.hlsli"))

                Expect.stringContains
                    text
                    "[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);"
                    "the explicit array at set 0 binding 0"

                Expect.stringContains text "#define LRM_BUF(slot) g_buf[(slot)]" "LRM_BUF on SPIR-V"
                Expect.isFalse (text.Contains "PROVISIONAL") "no provisional marker"
                Expect.isFalse (text.Contains "fvk-bind-resource-heap") "no emulated heap"

            testCase "the reader parses every embedded blob and writes it back byte for byte"
            <| fun () ->
                for b in ShaderSet.blobsFor ShaderTarget.Spirv (Shaders.embedded ()) do
                    let m = parse b.Bytes
                    Expect.equal m.Version Spirv.Version16 (b.Kernel + ": SPIR-V 1.6")
                    Expect.isTrue (m.Bound > 0u) "a bound"
                    Expect.equal (Spirv.toBytes m) b.Bytes (b.Kernel + ": round trip")
                    Expect.equal (Spirv.localSize b.Bytes) (Some (kernel b.Kernel).Threads) (b.Kernel + ": localSize")

            testCase "the reader rejects malformed modules"
            <| fun () ->
                let good = blob "scale_f32"

                let rejects (fragment: string) (bytes: byte[]) =
                    match Spirv.parse bytes with
                    | Ok _ -> failtestf "expected '%s'" fragment
                    | Error e -> Expect.stringContains e fragment fragment

                rejects "no bytes" null
                rejects "header" [| 3uy; 2uy; 35uy; 7uy |]
                rejects "whole number of words" good[.. good.Length - 2]
                // The header and the first word of OpCapability Shader, which has 2 words.
                rejects "runs past the end" good[..23]
                rejects "big-endian" (Array.concat [ [| 7uy; 35uy; 2uy; 3uy |]; good[4..] ])
                rejects "no SPIR-V magic" (Array.concat [ [| 0uy; 0uy; 0uy; 0uy |]; good[4..] ])
                let zero = Array.copy good
                // The first instruction's word count (bytes 22..23) set to 0.
                zero[22] <- 0uy
                zero[23] <- 0uy
                rejects "word count 0" zero

                let r = SpirvInspector.inspect (kernel "scale_f32") good[..23]
                expectRules [ "module" ] r "a truncated blob"
                Expect.isNone r.Facts "no facts"

            testCase "literal strings round-trip"
            <| fun () ->
                for s in
                    [
                        ""
                        "a"
                        "abc"
                        "abcd"
                        "main"
                        "GLSL.std.450"
                        "SPV_EXT_descriptor_indexing"
                        "é∑"
                    ] do
                    let w = Array.ofList (Spirv.stringOperands s)
                    Expect.equal (Spirv.literalString w 0) (Some(s, w.Length)) s
                    Expect.equal w.Length (Text.Encoding.UTF8.GetByteCount s / 4 + 1) "nul-terminated, padded"

                Expect.isNone (Spirv.literalString [| 0x41414141u |] 0) "no terminator"

            testCase "every embedded SPIR-V blob passes the inspector"
            <| fun () ->
                let reports = SpirvInspector.inspectSet Kernels.catalog (Shaders.embedded ())

                Expect.equal
                    (reports |> List.map _.Kernel)
                    (ShaderSet.blobsFor ShaderTarget.Spirv (Shaders.embedded ()) |> List.map _.Kernel)
                    "one report per blob"

                for r in reports do
                    expectRules [] r r.Kernel

            testCase "the facts of every blob match the catalog and the binding"
            <| fun () ->
                for b in ShaderSet.blobsFor ShaderTarget.Spirv (Shaders.embedded ()) do
                    let k = kernel b.Kernel
                    let r = SpirvInspector.inspect k b.Bytes

                    match r.Facts with
                    | None -> failtestf "%s: no facts" b.Kernel
                    | Some f ->
                        Expect.equal f.Version Spirv.Version16 "1.6"

                        for c in SpirvInspector.requiredCapabilities do
                            Expect.contains f.Capabilities c "a required capability"

                        for c in f.Capabilities do
                            Expect.contains SpirvInspector.allowedCapabilities c "an allowed capability"

                        for e in f.Extensions do
                            Expect.contains SpirvInspector.allowedExtensions e "an allowed extension"

                        Expect.equal f.LocalSize (Some k.Threads) "LocalSize"

                        Expect.equal
                            f.Resources
                            [
                                {
                                    Name = Some "g_buf"
                                    StorageClass = Spirv.StorageClass.StorageBuffer
                                    Set = Some SpirvBinding.DescriptorSet
                                    Binding = Some SpirvBinding.Binding
                                    NonWritable = false
                                }
                            ]
                            "one resource variable at set 0 binding 0"

                        Expect.equal f.PushBlock (Some(SpirvInspector.pushBlockName k)) "the push block"

                        Expect.equal
                            (f.PushMembers |> List.map (fun pm -> pm.Name, pm.Offset, pm.Type))
                            (SpirvInspector.expectedPushMembers k
                             |> List.mapi (fun i (n, t) -> Some n, Some(4 * i), t))
                            "push members: catalog names, offsets 4 * i, types"

                        Expect.equal f.PushBytes (4 * (1 + k.Fields.Length)) "push bytes"
                        Expect.isLessThanOrEqual f.PushBytes 128 "within the ArgBlock"

                        Expect.isLessThanOrEqual
                            f.WorkgroupBytes
                            SpirvInspector.MaxWorkgroupBytes
                            "32 KiB of groupshared"

                        for bi in f.BuiltIns do
                            Expect.contains SpirvInspector.allowedBuiltIns bi "allowed built-in"

            testCase "the selftests' SPIR-V facts (contracts-v1.1 vector)"
            <| fun () ->
                for name, threads, words, bytes, builtIns in selftestFacts do
                    let r = SpirvInspector.inspect (kernel name) (blob name)
                    let f = r.Facts.Value
                    Expect.equal f.LocalSize (Some threads) (name + " LocalSize")
                    Expect.equal f.PushMembers.Length words (name + " push words")
                    Expect.equal f.PushBytes bytes (name + " push bytes")
                    Expect.equal f.BuiltIns builtIns (name + " built-ins")
                    Expect.equal f.WorkgroupBytes 0L (name + " has no groupshared")

                    Expect.equal
                        f.Capabilities
                        [ Spirv.Capability.Shader; Spirv.Capability.RuntimeDescriptorArray ]
                        (name + " capabilities")

                    Expect.equal f.Extensions [ "SPV_EXT_descriptor_indexing" ] (name + " extensions")

            testCase "every rule fires on its mutation, and only its rules"
            <| fun () ->
                let original = parse (blob "scale_f32")

                for mu in mutations do
                    let bytes = Spirv.toBytes (mu.Apply original)
                    expectRules mu.Expect (SpirvInspector.inspect (kernel "scale_f32") bytes) mu.Name

                let covered = mutations |> List.collect _.Expect |> Set.ofList

                for r in SpirvRule.all do
                    Expect.isTrue
                        (covered.Contains(SpirvRule.name r))
                        (sprintf "rule %s has a mutation" (SpirvRule.name r))

            testCase "a blob inspected as another kernel breaks the push block and LocalSize"
            <| fun () ->
                let r = SpirvInspector.inspect (kernel "table_echo") (blob "scale_f32")
                expectRules [ "execution-mode"; "push-block" ] r "scale_f32 as table_echo"

                raisesContract "breaks the SPIR-V rules" (fun () ->
                    SpirvInspector.check (kernel "add") (blob "scale_f32"))

                Expect.isTrue (SpirvInspector.check (kernel "scale_f32") (blob "scale_f32")).Passed "check passes"

            testCase "a blob of a kernel outside the catalog is a module finding"
            <| fun () ->
                let set = Shaders.embedded ()
                let b = (ShaderSet.tryBlob "scale_f32" ShaderTarget.Spirv set).Value

                let other =
                    { set with
                        Blobs = { b with Kernel = "not_a_kernel" } :: set.Blobs
                    }

                let r =
                    SpirvInspector.inspectSet Kernels.catalog other
                    |> List.find (fun r -> r.Kernel = "not_a_kernel")

                expectRules [ "module" ] r "unknown kernel"

            testCase "rule names are unique and parse back"
            <| fun () ->
                let names = SpirvRule.all |> List.map SpirvRule.name
                Expect.equal (List.distinct names).Length names.Length "unique"

                for r in SpirvRule.all do
                    Expect.equal (SpirvRule.tryParse (SpirvRule.name r)) (Some r) (SpirvRule.name r)

                Expect.isNone (SpirvRule.tryParse "Resource") "case-sensitive"

            testCase "the report is emitted as gpu.shader telemetry"
            <| fun () ->
                let pass = SpirvInspector.inspect (kernel "scale_f32") (blob "scale_f32")
                let fail = SpirvInspector.inspect (kernel "table_echo") (blob "scale_f32")

                let records =
                    withSink (fun () ->
                        SpirvInspector.record pass
                        SpirvInspector.record fail)

                let attrs (r: TelemetryRecord) =
                    r.Attrs |> List.map (fun a -> a.Key, a.Value)

                match records with
                | [ c1; p1; w1; c2; p2; w2 ] ->
                    for r in records do
                        Expect.isTrue (TelemetryNames.isReserved r.Name) (r.Name + " is a reserved gpu.* name")

                    Expect.equal
                        (c1.Kind, c1.Name, c1.Value)
                        (TelemetryKind.Check, "gpu.shader.spirv_check", 1.0)
                        "check passed"

                    Expect.equal
                        (attrs c1)
                        [
                            "kernel", AttrValue.Str "scale_f32"
                            "target", AttrValue.Str "spirv"
                            "findings", AttrValue.Int 0L
                            "rules", AttrValue.Str ""
                        ]
                        "check attrs"

                    Expect.equal
                        (p1.Kind, p1.Name, p1.Value, p1.Unit)
                        (TelemetryKind.Gauge, "gpu.shader.push_bytes", 20.0, "B")
                        "push bytes"

                    Expect.equal (w1.Name, w1.Value, w1.Unit) ("gpu.shader.workgroup_bytes", 0.0, "B") "workgroup bytes"
                    Expect.equal c2.Value 0.0 "check failed"

                    Expect.equal
                        (attrs c2 |> List.filter (fun (k, _) -> k = "rules"))
                        [ "rules", AttrValue.Str "execution-mode,push-block" ]
                        "failed rules"

                    Expect.equal (p2.Name, w2.Name) ("gpu.shader.push_bytes", "gpu.shader.workgroup_bytes") "gauges"
                | rs -> failtestf "expected 6 records, got %d" rs.Length

            testCase "the Markdown table lists every blob"
            <| fun () ->
                let reports = SpirvInspector.inspectSet Kernels.catalog (Shaders.embedded ())
                let table = SpirvInspector.markdown reports
                let lines = table.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                Expect.equal lines.Length (2 + reports.Length) "header, rule and one row per blob"

                Expect.stringContains
                    table
                    "| `scale_f32` | 256x1x1 | 5, 20 | 0 | Shader, RuntimeDescriptorArray | pass |"
                    "the scale_f32 row"

                Expect.isFalse (table.Contains "\r") "LF only"
        ]
