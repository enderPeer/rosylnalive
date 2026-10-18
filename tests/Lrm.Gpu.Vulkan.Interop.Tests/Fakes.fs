/// Fake entry points for the cpu tier: managed delegates whose function pointers stand in for the driver's, so the
/// generated dispatch tables and wrappers are tested end to end without a GPU.
module Lrm.Gpu.Vulkan.Interop.Tests.Fakes

open System
open System.Collections.Generic
open System.Runtime.InteropServices

/// name -> function pointer of a managed delegate, each delegate rooted for the loader's lifetime.
type FakeLoader() =
    let roots = ResizeArray<Delegate>()
    let table = Dictionary<string, nativeint>(StringComparer.Ordinal)

    member _.Add(name: string, d: Delegate) =
        roots.Add d
        table[name] <- Marshal.GetFunctionPointerForDelegate d

    /// The loader function a dispatch table takes: 0n for names never added.
    member _.Load(name: string) =
        match table.TryGetValue name with
        | true, p -> p
        | _ -> 0n

    member _.Count = table.Count
