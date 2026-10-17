/// Compares the CLR layout of every bound struct with the C layout derived from vk.xml.
module LayoutCheck

open System
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Lrm.Gpu.Vulkan

/// Returns a list of mismatches; empty means every size, member name, member order and offset agrees.
let run () : string list =
    let asm = typeof<VkApplicationInfo>.Assembly

    [
        if not Environment.Is64BitProcess then
            "not a 64-bit process: the bindings and the expected table assume 64-bit"
        for (name, size, members) in VkXmlLayout.expected do
            match asm.GetType("Lrm.Gpu.Vulkan." + name) with
            | null -> sprintf "%s: type not found in binding" name
            | t ->
                let managed = RuntimeHelpers.SizeOf t.TypeHandle
                let marshalled = Marshal.SizeOf t

                if managed <> size then
                    sprintf "%s: managed sizeof %d, vk.xml %d" name managed size

                if marshalled <> size then
                    sprintf "%s: marshalled sizeof %d, vk.xml %d" name marshalled size
                // Pinning only succeeds for blittable types, i.e. ones the native side sees byte-for-byte.
                match
                    (try
                        Ok(GCHandle.Alloc(Activator.CreateInstance t, GCHandleType.Pinned))
                     with e ->
                         Error e.Message)
                with
                | Ok h -> h.Free()
                | Error msg -> sprintf "%s: not blittable (%s)" name msg

                let fields =
                    t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
                    |> Array.map (fun f -> f.Name)
                    |> Array.sortBy (fun f -> Marshal.OffsetOf(t, f))
                    |> List.ofArray

                let expectedNames = members |> List.map fst

                if fields <> expectedNames then
                    sprintf "%s: members %A, vk.xml %A" name fields expectedNames
                else
                    for (m, off) in members do
                        let actual = int (Marshal.OffsetOf(t, m))

                        if actual <> off then
                            sprintf "%s.%s: offset %d, vk.xml %d" name m actual off
    ]
