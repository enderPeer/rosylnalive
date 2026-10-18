namespace Lrm.Tokenizer

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Text
open System.Threading

/// The keys of the merge table: (left id << 32 | right id).
///
/// Int64's default hash is (low ^ high), which is left ^ right for these keys. All 444 self-pairs of Qwen3 ('a' 'a',
/// U+0120 U+0120, '=' '=' and so on) share the hash 0, and pairs of nearby ids collide in groups of a hundred, so the
/// dictionary walked chains of 12 on average and of 444 for a run of one repeated character. The comparer multiplies by
/// 2^64 / phi and keeps the high half, which depends on every bit of both ids.
module internal MergeKey =
    let inline pack (left: int) (right: int) =
        (int64 left <<< 32) ||| int64 (uint32 right)

    [<Sealed>]
    type private Comparer() =
        interface IEqualityComparer<int64> with
            member _.Equals(a: int64, b: int64) = a = b

            member _.GetHashCode(k: int64) =
                int ((uint64 k * 0x9E3779B97F4A7C15UL) >>> 32)

    /// The comparer of every merge table.
    let comparer: IEqualityComparer<int64> = Comparer()

/// One generation of the piece cache: a concurrent map, and the pieces and UTF-16 units added to it.
[<Sealed; AllowNullLiteral>]
type internal PieceGeneration() =
    let map = ConcurrentDictionary<string, int[]>(StringComparer.Ordinal)

    [<DefaultValue>]
    val mutable Count: int

    [<DefaultValue>]
    val mutable Units: int

    member val Lookup = map.GetAlternateLookup<ReadOnlySpan<char>>()

/// Per-thread work arrays of the BPE merge loop.
[<Sealed; AllowNullLiteral>]
type internal BpeWork(size: int) =
    member val Bytes: byte[] = Array.zeroCreate size with get, set
    member val Sym: int[] = Array.zeroCreate size with get, set
    member val Next: int[] = Array.zeroCreate size with get, set
    member val Prev: int[] = Array.zeroCreate size with get, set
    member val Queue = PriorityQueue<int, int64>()

    [<ThreadStatic; DefaultValue>]
    static val mutable private current: BpeWork

    /// Pieces up to this many bytes reuse the thread's arrays; longer ones get arrays of their own, so a thread never
    /// keeps the memory of one huge piece.
    static member Kept = 1 <<< 16

    /// Work arrays with room for `bytes` bytes: this thread's, or fresh ones for a piece longer than Kept.
    static member Get(bytes: int) =
        if bytes > BpeWork.Kept then
            BpeWork bytes
        else
            let mutable w = BpeWork.current

            if isNull w then
                w <- BpeWork 256
                BpeWork.current <- w

            if w.Bytes.Length < bytes then
                let n = min BpeWork.Kept (max bytes (2 * w.Bytes.Length))
                w.Bytes <- Array.zeroCreate n
                w.Sym <- Array.zeroCreate n
                w.Next <- Array.zeroCreate n
                w.Prev <- Array.zeroCreate n

            w

/// Byte-level BPE over merge ranks (TokenizerApi contract, "BPE").
///
/// A piece's UTF-8 bytes start as one symbol each (the vocabulary id of the byte's map character). While some adjacent
/// pair has a merge, the pair with the lowest rank is merged, the leftmost one among equal ranks, exactly as
/// tokenizers' Word::merge_all pops its heap ordered by (rank, position).
///
/// Results are cached per piece of at most `maxPieceUnits` UTF-16 units, in two generations of at most `capacity / 2`
/// pieces and `units / 2` units each. New pieces go into the current generation; a piece found only in the previous one
/// is copied into the current one. When the current generation is full it becomes the previous one and the old previous
/// one is dropped. So the cache holds at most `capacity` pieces and `units + 2 * maxPieceUnits` units (a generation is
/// full once it reaches its share, so its last piece may overshoot), plus at most one piece per thread and generation
/// from threads racing a swap. Pieces in
/// use survive a swap, and a change of input domain refills the cache. The cache never changes a result.
[<Sealed>]
type internal BpeModel
    (
        // The bytes of every vocabulary id 0 .. n-1.
        tokenBytes: byte[][],
        // The vocabulary id of each single byte.
        byteIds: int[],
        // (left id << 32 | right id) -> (rank << 32 | merged id), with MergeKey.comparer.
        merges: Dictionary<int64, int64>,
        capacity: int,
        units: int,
        maxPieceUnits: int
    ) =
    let generationSize = max 1 (capacity / 2)
    let generationUnits = max 1 (units / 2)
    let swap = obj ()
    let mutable current = PieceGeneration()
    let mutable previous = PieceGeneration()

    static let key (left: int) (right: int) = MergeKey.pack left right

    /// Adds a piece to generation `g`, and makes a new current generation when `g` is the current one and full.
    let remember (g: PieceGeneration) (piece: ReadOnlySpan<char>) (ids: int[]) =
        if g.Lookup.TryAdd(piece, ids) then
            let pieces = Interlocked.Increment &g.Count
            let units = Interlocked.Add(&g.Units, piece.Length)

            if pieces >= generationSize || units >= generationUnits then
                lock swap (fun () ->
                    if obj.ReferenceEquals(current, g) then
                        Volatile.Write(&previous, g)
                        Volatile.Write(&current, PieceGeneration()))

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
        let cacheable = length <= maxPieceUnits
        let g = Volatile.Read &current

        if cacheable && g.Lookup.TryGetValue(piece, &ids) then
            output.AddRange ids
        elif cacheable && (Volatile.Read &previous).Lookup.TryGetValue(piece, &ids) then
            remember g piece ids
            output.AddRange ids
        else
            let w = BpeWork.Get(Encoding.UTF8.GetMaxByteCount length)
            let m = Encoding.UTF8.GetBytes(piece, Span<byte>(w.Bytes))
            let merged = merge w m

            if cacheable then
                remember g piece merged

            output.AddRange merged

    /// The number of pieces the cache holds, over both generations (a piece copied from the previous generation into
    /// the current one counts twice).
    member _.CacheCount = (Volatile.Read &current).Count + (Volatile.Read &previous).Count

    /// Forgets every cached piece (the bench's cold runs).
    member _.ClearCache() =
        lock swap (fun () ->
            Volatile.Write(&previous, PieceGeneration())
            Volatile.Write(&current, PieceGeneration()))
