namespace Lrm.Gpu.D3D12.Interop

open System
open System.IO
open System.Reflection
open System.Reflection.Emit
open System.Reflection.Metadata
open System.Reflection.Metadata.Ecma335
open System.Reflection.PortableExecutable

/// One call compiled with the IL tail. prefix.
type TailCallSite =
    {
        /// The calling method, "Namespace.Type.Method" (nested types joined with '+').
        Caller: string
        /// The called method, "Type.Method", as far as the metadata names it.
        Callee: string
    }

/// Finds IL tail. prefixes in an assembly (ADR R3-0001). A tail call frees the caller's frame before the callee runs,
/// so a callee that receives the address of a caller's local (NativePtr.toNativeInt &&x, or a span over it) reads a
/// dead stack slot. The generated wrappers cannot be tail-called (they take `this` by reference); projects that
/// also pass stack addresses to their own functions build with Tailcalls=false and check the result with `scan`.
[<RequireQualifiedAccess>]
module TailCalls =
    let private opcodes =
        typeof<OpCodes>.GetFields(BindingFlags.Public ||| BindingFlags.Static)
        |> Array.map (fun f -> f.GetValue null :?> OpCode)
        |> Array.map (fun op -> uint16 op.Value, op)
        |> dict

    let private operandSize (op: OpCode) (il: byte[]) (at: int) =
        match op.OperandType with
        | OperandType.InlineNone -> 0
        | OperandType.ShortInlineBrTarget
        | OperandType.ShortInlineI
        | OperandType.ShortInlineVar -> 1
        | OperandType.InlineVar -> 2
        | OperandType.InlineI8
        | OperandType.InlineR -> 8
        | OperandType.InlineSwitch -> 4 + 4 * BitConverter.ToInt32(il, at)
        | _ -> 4

    let rec private typeDefName (md: MetadataReader) (h: TypeDefinitionHandle) =
        let t = md.GetTypeDefinition h
        let name = md.GetString t.Name
        let outer = t.GetDeclaringType()

        if outer.IsNil then
            let ns = md.GetString t.Namespace
            if ns = "" then name else ns + "." + name
        else
            typeDefName md outer + "+" + name

    let private parentName (md: MetadataReader) (h: EntityHandle) =
        match h.Kind with
        | HandleKind.TypeDefinition -> typeDefName md (TypeDefinitionHandle.op_Explicit h)
        | HandleKind.TypeReference -> md.GetString (md.GetTypeReference(TypeReferenceHandle.op_Explicit h)).Name
        | HandleKind.TypeSpecification -> "(generic type)"
        | _ -> "?"

    let rec private methodName (md: MetadataReader) (h: EntityHandle) =
        match h.Kind with
        | HandleKind.MethodDefinition ->
            let m = md.GetMethodDefinition(MethodDefinitionHandle.op_Explicit h)
            typeDefName md (m.GetDeclaringType()) + "." + md.GetString m.Name
        | HandleKind.MemberReference ->
            let m = md.GetMemberReference(MemberReferenceHandle.op_Explicit h)
            parentName md m.Parent + "." + md.GetString m.Name
        | HandleKind.MethodSpecification ->
            methodName md (md.GetMethodSpecification(MethodSpecificationHandle.op_Explicit h)).Method
        | HandleKind.StandaloneSignature -> "(calli)"
        | _ -> "?"

    /// Every call in the assembly at `path` that carries the tail. prefix.
    let scan (path: string) : TailCallSite list =
        use fs = File.OpenRead path
        use pe = new PEReader(fs)
        let md = pe.GetMetadataReader()

        [
            for h in md.MethodDefinitions do
                let m = md.GetMethodDefinition h

                if m.RelativeVirtualAddress <> 0 then
                    let il = pe.GetMethodBody(m.RelativeVirtualAddress).GetILBytes()
                    let caller = typeDefName md (m.GetDeclaringType()) + "." + md.GetString m.Name
                    let mutable i = 0
                    let mutable tail = false

                    while i < il.Length do
                        let value, len =
                            if il[i] = 0xFEuy then
                                (0xFE00us ||| uint16 il[i + 1]), 2
                            else
                                uint16 il[i], 1

                        let op = opcodes[value]

                        if tail && op.FlowControl = FlowControl.Call then
                            let token = BitConverter.ToInt32(il, i + len)

                            yield
                                {
                                    Caller = caller
                                    Callee = methodName md (MetadataTokens.EntityHandle token)
                                }

                        tail <- op = OpCodes.Tailcall
                        i <- i + len + operandSize op il (i + len)
        ]
