namespace Lrm.Tokenizer

open System
open System.Collections.Generic

/// One added token of tokenizer.json, such as <|im_start|> (TokenizerApi contract, "Added tokens").
type AddedToken =
    {
        Id: int
        Content: string
        /// True for the 14 special tokens (<|endoftext|> to <|video_pad|>); Encode and Decode treat all 26 alike.
        Special: bool
    }

/// A character trie over the added tokens' contents. Encode scans the raw text left to right and, at the first
/// position where any added token starts, takes the longest one there (leftmost-longest, as tokenizers'
/// Aho-Corasick matcher with MatchKind::LeftmostLongest). The text between matches goes on to NFC and the pre-tokenizer.
[<Sealed>]
type internal AddedTokenTrie(tokens: AddedToken list) =
    // Node 0 is the root. Children per node by character; Terminal holds the token id ending there, or -1.
    let children = ResizeArray<Dictionary<char, int>>()
    let terminal = ResizeArray<int>()

    let addNode () =
        children.Add(Dictionary<char, int>())
        terminal.Add -1
        children.Count - 1

    do
        addNode () |> ignore

        for t in tokens do
            if String.IsNullOrEmpty t.Content then
                invalidArg "tokens" "an added token has empty content"

            let mutable node = 0

            for c in t.Content do
                match children[node].TryGetValue c with
                | true, next -> node <- next
                | _ ->
                    let next = addNode ()
                    children[node].[c] <- next
                    node <- next

            if terminal[node] >= 0 then
                invalidArg "tokens" (sprintf "the added token %s appears twice" t.Content)

            terminal[node] <- t.Id

    /// The longest added token that starts at `start`: struct (id, length), or (-1, 0).
    member _.LongestAt(text: string, start: int) : struct (int * int) =
        let mutable node = 0
        let mutable i = start
        let mutable bestId = -1
        let mutable bestLength = 0
        let mutable go = true

        while go && i < text.Length do
            match children[node].TryGetValue text[i] with
            | true, next ->
                node <- next
                i <- i + 1

                if terminal[node] >= 0 then
                    bestId <- terminal[node]
                    bestLength <- i - start
            | _ -> go <- false

        struct (bestId, bestLength)

    /// True when some added token starts with character `c`.
    member _.Starts(c: char) = children[0].ContainsKey c

    /// Calls `added id` for each added-token match and `segment start length` for each stretch of text between them,
    /// in order. Empty stretches are skipped.
    member this.Split(text: string, added: int -> unit, segment: int -> int -> unit) =
        let mutable start = 0
        let mutable i = 0

        while i < text.Length do
            if this.Starts text[i] then
                let struct (id, length) = this.LongestAt(text, i)

                if id >= 0 then
                    if i > start then
                        segment start (i - start)

                    added id
                    i <- i + length
                    start <- i
                else
                    i <- i + 1
            else
                i <- i + 1

        if start < text.Length then
            segment start (text.Length - start)
