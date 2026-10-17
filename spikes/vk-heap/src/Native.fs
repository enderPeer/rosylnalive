/// Native-memory helpers for the spike: a scratch arena whose blocks live until it is disposed, so pNext chains and
/// arrays handed to the driver stay valid for the whole call.
module VkHeapSpike.Native

#nowarn "9"

open System
open System.Runtime.InteropServices
open System.Text
open Microsoft.FSharp.NativeInterop

/// Zeroed native blocks, freed together on Dispose.
type Scratch() =
    let blocks = ResizeArray<nativeint>()

    /// `bytes` zeroed bytes (at least 1), 16-byte aligned.
    member _.Alloc(bytes: int) : nativeint =
        let p = NativeMemory.AlignedAlloc(unativeint (max bytes 1), 16un)
        NativeMemory.Clear(p, unativeint (max bytes 1))
        let a = NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte> p)
        blocks.Add a
        a

    /// A copy of `value` in native memory.
    member s.Struct<'T when 'T: unmanaged>(value: 'T) : nativeint =
        let p = s.Alloc(sizeof<'T>)
        NativePtr.write (NativePtr.ofNativeInt<'T> p) value
        p

    /// A copy of `values` in native memory (a C array).
    member s.Array<'T when 'T: unmanaged>(values: 'T[]) : nativeint =
        let size = sizeof<'T>
        let p = s.Alloc(size * values.Length)

        for i in 0 .. values.Length - 1 do
            NativePtr.set (NativePtr.ofNativeInt<'T> p) i values[i]

        p

    /// A null-terminated UTF-8 string.
    member s.Utf8(text: string) : nativeint =
        let b = Encoding.UTF8.GetBytes(text + "\000")
        let p = s.Alloc b.Length
        Marshal.Copy(b, 0, p, b.Length)
        p

    /// An array of pointers to null-terminated UTF-8 strings.
    member s.Utf8Array(texts: string list) : nativeint =
        texts |> List.map s.Utf8 |> Array.ofList |> s.Array

    interface IDisposable with
        member _.Dispose() =
            for b in blocks do
                NativeMemory.AlignedFree(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> b))

            blocks.Clear()

let read<'T when 'T: unmanaged> (p: nativeint) = NativePtr.read (NativePtr.ofNativeInt<'T> p)

let readAt<'T when 'T: unmanaged> (p: nativeint) (i: int) =
    NativePtr.get (NativePtr.ofNativeInt<'T> p) i

let copyIn (dst: nativeint) (src: byte[]) = Marshal.Copy(src, 0, dst, src.Length)

let copyOut (src: nativeint) (length: int) =
    let a = Array.zeroCreate<byte> length
    Marshal.Copy(src, a, 0, length)
    a

let alignUp (x: uint64) (a: uint64) = (x + a - 1UL) / a * a

/// Little-endian u32 words of a byte array.
let words (b: byte[]) =
    Array.init (b.Length / 4) (fun i -> BitConverter.ToUInt32(b, 4 * i))

let ofWords (w: uint32[]) =
    let b = Array.zeroCreate<byte> (w.Length * 4)
    Buffer.BlockCopy(w, 0, b, 0, b.Length)
    b
