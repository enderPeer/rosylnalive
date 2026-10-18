namespace Lrm.Gpu.Core

open System
open System.Buffers.Binary

/// The 128-byte kernel argument block (KernelAbiAndCatalog contract): 32 little-endian u32 words. Word 0 is
/// lrm_groups_x, which ICommandList.Dispatch writes; the kernel's catalog fields follow from word 1 in catalog order.
/// Words after the last field are 0. D3D12 passes the block as 32 root constants at b0; Vulkan as push data at
/// offset 0 (vkCmdPushDataEXT, 128 bytes).
[<Sealed>]
type ArgBlock private (words: uint32[]) =
    /// u32 words in a block.
    static member WordCount = 32

    /// Bytes in a block.
    static member ByteCount = 128

    /// The all-zero block.
    static member Zero = ArgBlock(Array.zeroCreate 32)

    /// A block from exactly 32 words (copied).
    static member OfWords(w: uint32[]) =
        if isNull w || w.Length <> 32 then
            Fail.contract "an argument block is 32 words"

        ArgBlock(Array.copy w)

    /// A block from exactly 128 little-endian bytes.
    static member OfBytes(bytes: ReadOnlySpan<byte>) =
        if bytes.Length <> 128 then
            Fail.contract "an argument block is 128 bytes, got %d" bytes.Length

        let w = Array.zeroCreate<uint32> 32

        for i in 0..31 do
            w[i] <- BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(4 * i, 4))

        ArgBlock w

    /// Word `i` (0..31).
    member _.Item
        with get (i: int) = words[i]

    /// Word 0: the number of thread groups in x of the physical dispatch.
    member _.GroupsX = words[0]

    /// A copy with word `i` replaced.
    member _.With(i: int, value: uint32) =
        let w = Array.copy words
        w[i] <- value
        ArgBlock w

    /// A copy with word 0 (lrm_groups_x) replaced.
    member x.WithGroupsX(groupsX: uint32) = x.With(0, groupsX)

    /// Word `i` read as the raw bits of a float32.
    member _.F32(i: int) =
        BitConverter.UInt32BitsToSingle words[i]

    /// Word `i` read as an int32.
    member _.I32(i: int) = int words[i]

    /// A copy of the 32 words.
    member _.ToWords() = Array.copy words

    /// Writes the 128 bytes to the start of `destination`.
    member _.WriteTo(destination: Span<byte>) =
        if destination.Length < 128 then
            Fail.contract "an argument block needs 128 bytes"

        for i in 0..31 do
            BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(4 * i, 4), words[i])

    /// The 128 bytes.
    member x.ToBytes() =
        let b = Array.zeroCreate<byte> 128
        x.WriteTo(Span b)
        b

    override _.Equals(other: obj) =
        match other with
        | :? ArgBlock as o -> o.ToWords() = words
        | _ -> false

    override _.GetHashCode() = hash words

    override _.ToString() =
        words
        |> Array.map (sprintf "%08x")
        |> String.concat " "
        |> sprintf "ArgBlock(%s)"

[<RequireQualifiedAccess>]
module ArgBlock =
    /// The raw bits of a float32, as stored in an argument word. NaN payloads and negative zero are kept.
    let f32Bits (x: float32) = BitConverter.SingleToUInt32Bits x

    /// An int32 as stored in an argument word (two's complement).
    let i32Bits (x: int) = uint32 x

/// A logical grid of thread groups, as the catalog's grid formula gives it.
[<Struct>]
type Grid = { X: int64; Y: int64; Z: int64 }

/// The grid actually dispatched. X, Y and Z are each 1..65,535, and lrm_groups_x (argument word 0) equals X.
[<Struct>]
type PhysicalGrid = { X: int; Y: int; Z: int }

[<RequireQualifiedAccess>]
module Grid =
    /// The largest group count in one dimension of a dispatch (D3D12 and Vulkan minimum).
    [<Literal>]
    let MaxDimension = 65535L

    /// A logical grid.
    let create (x: int64) (y: int64) (z: int64) : Grid = { X = x; Y = y; Z = z }

    /// ceil(a / b) for a >= 0 and b >= 1.
    let ceilDiv (a: int64) (b: int64) =
        if b < 1L then
            Fail.contract "ceildiv by %d" b

        if a < 0L then
            Fail.contract "ceildiv of negative %d" a

        (a + b - 1L) / b

    /// The physical grid for a logical one (KernelAbiAndCatalog contract, grid folding):
    /// - every logical dimension must be at least 1, and Y and Z at most 65,535;
    /// - X at most 65,535: dispatched as it is;
    /// - X above 65,535: allowed only for a linear kernel (catalog grid y and z are 1). It is folded into
    ///   y' = ceil(X / 65535) rows of x' = ceil(X / y') groups. The kernel's logical group is
    ///   gid.y * lrm_groups_x + gid.x; the x' * y' - X (< y') extra groups find no work and return.
    let physical (linear: bool) (g: Grid) : Result<PhysicalGrid, string> =
        if g.X < 1L || g.Y < 1L || g.Z < 1L then
            Error(sprintf "empty grid %dx%dx%d (every dimension must be at least 1)" g.X g.Y g.Z)
        elif g.Y > MaxDimension || g.Z > MaxDimension then
            Error(sprintf "grid %dx%dx%d: y and z must be at most 65535" g.X g.Y g.Z)
        elif g.X <= MaxDimension then
            Ok
                {
                    X = int g.X
                    Y = int g.Y
                    Z = int g.Z
                }
        elif not linear then
            Error(sprintf "grid x of %d exceeds 65535 and the kernel's grid is not linear, so it cannot be folded" g.X)
        elif g.X > MaxDimension * MaxDimension then
            Error(sprintf "grid x of %d exceeds 65535 * 65535" g.X)
        else
            let y = ceilDiv g.X MaxDimension
            let x = ceilDiv g.X y
            Ok { X = int x; Y = int y; Z = 1 }

    /// The logical group index of physical group (gx, gy) of a linear kernel: gy * physical.X + gx.
    let linearGroup (p: PhysicalGrid) (gx: int) (gy: int) = int64 gy * int64 p.X + int64 gx

    /// Thread groups in a physical grid.
    let groups (p: PhysicalGrid) = int64 p.X * int64 p.Y * int64 p.Z
