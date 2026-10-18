namespace Lrm.Tokenizer

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Text
open System.Threading

/// Per-thread work arrays of the BPE merge loop.
[<Sealed; AllowNullLiteral>]
type internal BpeWork() =
    member val Bytes: byte[] = Array.zeroCreate 256 with get, set
    member val Sym: int[] = Array.zeroCreate 256 with get, set
    member val Next: int[] = Array.zeroCreate 256 with get, set
    member val Prev: int[] = Array.zeroCreate 256 with get, set
    member val Queue = PriorityQueue<int, int64>()

    [<ThreadStatic; DefaultValue>]
    static val mutable private current: BpeWork

    /// This thread's work arrays, with room for `bytes` bytes.
    static member Get(bytes: int) =
        let mutable w = BpeWork.current

        if isNull w then
            w <- BpeWork()
            BpeWork.current <- w

        if w.Bytes.Length < bytes then
            let n = max bytes (2 * w.Bytes.Length)
            w.Bytes <- Array.zeroCreate n
            w.Sym <- Array.zeroCreate n
            w.Next <- Array.zeroCreate n
            w.Prev <- Array.zeroCreate n

        w

/// Byte-level BPE over merge ranks (TokenizerApi contract, "BPE").
///
/// A piece's UTF-8 bytes start as one symbol each (the vocabulary id of the byte's map character). While some adjacent
/// pair has a merge, the pair with the lowest rank is merged, the leftmost one among equal ranks, exactly as
/// tokenizers' Word::merge_all pops its heap ordered by (rank, position). Results are cached per piece; the cache
/// never changes a result.
[<Sealed>]
type internal BpeModel
    (
        // The bytes of every vocabulary id 0 .. n-1.
        tokenBytes: byte[][],
        // The vocabulary id of each single byte.
        byteIds: int[],
        // (left id << 32 | right id) -> (rank << 32 | merged id).
        merges: Dictionary<int64, int64>,
        cacheCapacity: int
    ) =
    let cache = ConcurrentDictionary<string, int[]>(StringComparer.Ordinal)
    let lookup = cache.GetAlternateLookup<ReadOnlySpan<char>>()
    let mutable cached = 0

    static let key (left: int) (right: int) =
        (int64 left <<< 32) ||| int64 (uint32 right)

    /// The merged ids of the first `m` bytes of `w.Bytes`.
    let merge (w: BpeWork) (m: int) : int[] =
        let bytes = w.Bytes

        if m = 1 then
            [| byteIds[int bytes[0]] |]
        else
            let sym = w.Sym
            let next = w.Next
            let prev = w.Prev
            let queue = w.Queue
            queue.Clear()

            for i in 0 .. m - 1 do
                sym[i] <- byteIds[int bytes[i]]
                next[i] <- if i + 1 < m then i + 1 else -1
                prev[i] <- i - 1

            let mutable alive = m

            // Priority: rank << 32 | position of the left symbol, so ties go to the leftmost pair.
            let push (i: int) =
                let j = next[i]

                if j >= 0 then
                    match merges.TryGetValue(key sym[i] sym[j]) with
                    | true, v -> queue.Enqueue(i, ((v >>> 32) <<< 32) ||| int64 i)
                    | _ -> ()

            for i in 0 .. m - 2 do
                push i

            let mutable i = 0
            let mutable priority = 0L

            while queue.TryDequeue(&i, &priority) do
                let j = next[i]
                // Stale entries: the left symbol was absorbed (prev = -2), it has no right neighbour any more, or the
                // pair changed. Ranks are unique per pair, so an unchanged rank means an unchanged pair.
                if prev[i] <> -2 && j >= 0 then
                    match merges.TryGetValue(key sym[i] sym[j]) with
                    | true, v when (v >>> 32) = (priority >>> 32) ->
                        sym[i] <- int (v &&& 0xFFFFFFFFL)
                        let k = next[j]
                        next[i] <- k

                        if k >= 0 then
                            prev[k] <- i

                        prev[j] <- -2
                        alive <- alive - 1

                        if prev[i] >= 0 then
                            push prev[i]

                        push i
                    | _ -> ()

            let out = Array.zeroCreate<int> alive
            let mutable s = 0
            let mutable n = 0

            while s >= 0 do
                out[n] <- sym[s]
                n <- n + 1
                s <- next[s]

            out

    /// The number of vocabulary ids.
    member _.Count = tokenBytes.Length

    /// The bytes of vocabulary id `id` (0 <= id < Count).
    member _.BytesOf(id: int) = tokenBytes[id]

    /// Appends the ids of the piece text[start .. start + length - 1] to `output`.
    member _.EncodePiece(text: string, start: int, length: int, output: ResizeArray<int>) =
        let piece = text.AsSpan(start, length)
        let mutable ids = null

        if lookup.TryGetValue(piece, &ids) then
            output.AddRange ids
        else
            let w = BpeWork.Get(Encoding.UTF8.GetMaxByteCount length)
            let m = Encoding.UTF8.GetBytes(piece, Span<byte>(w.Bytes))
            let merged = merge w m

            if Volatile.Read &cached < cacheCapacity && lookup.TryAdd(piece, merged) then
                Interlocked.Increment &cached |> ignore

            output.AddRange merged

    /// The number of cached pieces.
    member _.CacheCount = Volatile.Read &cached

    /// Forgets every cached piece (the bench's cold runs).
    member _.ClearCache() =
        cache.Clear()
        Volatile.Write(&cached, 0)
