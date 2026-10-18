namespace Lrm.Gpu.Vulkan.Interop

#nowarn "9"

open System
open System.Collections.Generic
open Microsoft.FSharp.NativeInterop

/// pNext chains (VulkanInteropApi contract): every struct of a chain is a local of the frame that makes the call, so
/// the chain never outlives it. A chain is built by prepending: `VkChain.prepend &&s &s.pNext &head`.
[<RequireQualifiedAccess>]
module VkChain =
    /// Prepends `s`, a struct in the caller's frame, to the chain whose first pointer is `head`.
    let inline prepend (s: nativeptr<'T>) (pNextOfS: byref<nativeint>) (head: byref<nativeint>) =
        pNextOfS <- head
        head <- NativePtr.toNativeInt s

    let private bySType =
        lazy
            (let d = Dictionary<VkStructureType, VkStructMeta>()

             for s in VkMeta.Structs do
                 match s.SType with
                 | ValueSome t -> d[t] <- s
                 | ValueNone -> ()

             d)

    let private byName = lazy (VkMeta.Structs |> List.map (fun s -> s.Name, s) |> dict)

    /// The bound struct whose sType is `t`.
    let tryStruct (t: VkStructureType) =
        match bySType.Value.TryGetValue t with
        | true, s -> Some s
        | _ -> None

    /// The sType values of a chain, from its first pointer: each node's sType (offset 0) and pNext (offset 8).
    /// Stops at 64 nodes so that a cyclic chain cannot hang the walk.
    let walk (head: nativeint) : VkStructureType list =
        [
            let mutable p = head
            let mutable n = 0

            while p <> 0n && n < 64 do
                yield enum<VkStructureType>(NativePtr.read (NativePtr.ofNativeInt<int32> p))
                p <- NativePtr.read (NativePtr.ofNativeInt<nativeint>(p + 8n))
                n <- n + 1

            if p <> 0n then
                yield enum<VkStructureType> -1
        ]

    /// What is wrong with the chain starting at `head` for the struct `parent` (vk.xml rules): every node is a bound
    /// struct that may extend `parent`, none appears twice unless it allows duplicates, and the chain ends. Empty when
    /// the chain is valid. Tests and debug checks call it; it reads the chain and changes nothing.
    let problems (parent: string) (head: nativeint) : string list =
        let nodes = walk head
        let seen = HashSet<VkStructureType>()

        [
            if not (byName.Value.ContainsKey parent) then
                yield sprintf "%s is not a bound struct" parent

            for t in nodes do
                if t = enum<VkStructureType> -1 then
                    yield sprintf "the chain of %s has more than 64 nodes (a cycle?)" parent
                else
                    match tryStruct t with
                    | None -> yield sprintf "sType %d is not a bound struct" (int t)
                    | Some s ->
                        if not (List.contains parent s.Extends) then
                            yield sprintf "%s does not extend %s" s.Name parent

                        if not (seen.Add t) && not s.AllowDuplicate then
                            yield sprintf "%s appears twice in the chain of %s" s.Name parent
        ]
