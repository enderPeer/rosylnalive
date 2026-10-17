namespace Lrm.Cluster

open System
open System.Collections.Concurrent
open System.IO
open System.Threading
open Lrm.Core

/// A content-addressed store of raw bytes, keyed by BlobId (Canonical.hashBytes "lrm.blob/1"). Blobs are immutable:
/// a Put of bytes that are already stored is a no-op, and a get returns bytes whose id was checked. Implementations
/// are thread-safe.
type IBlobStore =
    /// Stores `bytes` and returns their id.
    abstract Put: bytes: byte[] -> BlobId
    /// The bytes of `id`, or None when absent. Callers must not modify the returned array.
    abstract TryGet: id: BlobId -> byte[] option
    /// True when `id` is stored.
    abstract Contains: id: BlobId -> bool
    /// The number of blobs stored.
    abstract Count: int
    /// The total size of the stored blobs in bytes.
    abstract TotalBytes: int64

/// An in-memory blob store.
[<Sealed>]
type MemoryBlobStore() =
    let blobs = ConcurrentDictionary<BlobId, byte[]>()
    let mutable total = 0L

    interface IBlobStore with
        member _.Put(bytes: byte[]) =
            let id = BlobId.ofBytes bytes

            if blobs.TryAdd(id, Array.copy bytes) then
                Interlocked.Add(&total, int64 bytes.Length) |> ignore

            id

        member _.TryGet(id: BlobId) =
            match blobs.TryGetValue id with
            | true, b -> Some b
            | _ -> None

        member _.Contains(id: BlobId) = blobs.ContainsKey id
        member _.Count = blobs.Count
        member _.TotalBytes = Interlocked.Read(&total)

/// A blob store in a directory: one file per blob at <root>/<first two hex digits>/<64 hex digits>.blob, written to a
/// temporary file and renamed into place, so a reader never sees a partial blob. Every read re-checks the id; a file
/// whose bytes no longer match raises InvalidDataException.
[<Sealed>]
type DirectoryBlobStore(root: string) =
    let root = Path.GetFullPath root
    do Directory.CreateDirectory root |> ignore

    let pathOf (id: BlobId) =
        let hex = id.Value.Hex
        Path.Combine(root, hex.Substring(0, 2), hex + ".blob")

    /// The store's directory.
    member _.Root = root

    interface IBlobStore with
        member _.Put(bytes: byte[]) =
            let id = BlobId.ofBytes bytes
            let path = pathOf id

            if not (File.Exists path) then
                Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
                let temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp"
                File.WriteAllBytes(temp, bytes)

                try
                    File.Move(temp, path, false)
                with :? IOException when File.Exists path ->
                    // Another writer stored the same content first; the blob is immutable, so either copy is fine.
                    File.Delete temp

            id

        member _.TryGet(id: BlobId) =
            let path = pathOf id

            if not (File.Exists path) then
                None
            else
                let bytes = File.ReadAllBytes path

                if BlobId.ofBytes bytes <> id then
                    raise (InvalidDataException(sprintf "blob %s is corrupt (%s)" id.Value.Hex path))

                Some bytes

        member _.Contains(id: BlobId) = File.Exists(pathOf id)

        member _.Count =
            if Directory.Exists root then
                Directory.EnumerateFiles(root, "*.blob", SearchOption.AllDirectories)
                |> Seq.length
            else
                0

        member _.TotalBytes =
            if Directory.Exists root then
                Directory.EnumerateFiles(root, "*.blob", SearchOption.AllDirectories)
                |> Seq.sumBy (fun f -> FileInfo(f).Length)
            else
                0L
