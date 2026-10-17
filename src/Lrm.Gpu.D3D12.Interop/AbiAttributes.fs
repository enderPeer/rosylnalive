namespace Lrm.Gpu.D3D12.Interop

open System

/// Marks a generated struct with the C type it mirrors. The ABI oracle (tools/abi-oracle/d3d12) asserts its size
/// and every field offset against the SDK headers with cl.exe.
[<Sealed; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type CStructAttribute(name: string) =
    inherit Attribute()
    /// The C type name, for example D3D12_RESOURCE_DESC.
    member _.Name = name

/// Marks a generated enum with the C enum it mirrors and the prefix its member names dropped: the C name of member
/// M is Prefix + M. The ABI oracle asserts every member's value.
[<Sealed; AttributeUsage(AttributeTargets.Enum, AllowMultiple = false)>]
type CEnumAttribute(name: string, prefix: string) =
    inherit Attribute()
    member _.Name = name
    member _.Prefix = prefix

/// Marks a generated inline array that stands for a C array field such as WCHAR Description[128].
[<Sealed; AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)>]
type CArrayAttribute(element: string, count: int) =
    inherit Attribute()
    member _.Element = element
    member _.Count = count
