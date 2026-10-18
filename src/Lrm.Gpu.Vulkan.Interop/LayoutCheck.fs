/// Compares the CLR layout of every bound struct and union with the generator's own C layout (VkMeta.Structs,
/// computed from vk.xml member types). The ABI oracle checks the same runtime layout against the Vulkan headers with
/// cl.exe; this check needs no compiler, so the device check and every node run it before touching a device.
module Lrm.Gpu.Vulkan.Interop.LayoutCheck

open System
open System.Reflection
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

/// The mismatches for one struct; empty when its size, member names, member order, offsets and member sizes agree
/// and it is blittable.
let checkStruct (s: VkStructMeta) : string list =
    let t = s.Type

    [
        let managed = RuntimeHelpers.SizeOf t.TypeHandle
        let marshalled = Marshal.SizeOf t

        if managed <> s.Size || marshalled <> s.Size then
            sprintf "%s: managed sizeof %d, marshalled %d, vk.xml %d" s.Name managed marshalled s.Size

        // Pinning succeeds only for blittable types, which the native side sees byte for byte.
        match
            (try
                Ok(GCHandle.Alloc(Activator.CreateInstance t, GCHandleType.Pinned))
             with e ->
                 Error e.Message)
        with
        | Ok h -> h.Free()
        | Error msg -> sprintf "%s: not blittable (%s)" s.Name msg

        let fields =
            t.GetFields(BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic)
            |> Array.map (fun f ->
                f.Name, int (Marshal.OffsetOf(t, f.Name)), RuntimeHelpers.SizeOf f.FieldType.TypeHandle)
            |> List.ofArray

        let ordered =
            if s.IsUnion then
                fields // every member at 0: declaration order is the only order
            else
                fields |> List.sortBy (fun (_, o, _) -> o)

        let names = ordered |> List.map (fun (n, _, _) -> n)
        let expected = s.Fields |> List.map (fun (n, _, _) -> n)

        if names <> expected then
            sprintf "%s: members %A, vk.xml %A" s.Name names expected
        else
            for ((n, o, z), (_, eo, ez)) in List.zip ordered s.Fields do
                if o <> eo then
                    sprintf "%s.%s: offset %d, vk.xml %d" s.Name n o eo

                if z <> ez then
                    sprintf "%s.%s: size %d, vk.xml %d" s.Name n z ez
    ]

/// Every mismatch over every bound struct; empty means the binding matches vk.xml. Fails at once on a 32-bit process,
/// which the binding does not support.
let run () : string list =
    [
        if not Environment.Is64BitProcess then
            "not a 64-bit process: the binding and vk.xml layouts assume x64"
        else
            for s in VkMeta.Structs do
                yield! checkStruct s
    ]
