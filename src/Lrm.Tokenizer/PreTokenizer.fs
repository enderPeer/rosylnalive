/// The Qwen Split pre-tokenizer as a state machine over Unicode scalar values (TokenizerApi contract,
/// "Pre-tokenizer").
///
/// tokenizer.json splits with the Oniguruma regex
///
///     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
///
/// with behaviour Isolated. Oniguruma tries the alternatives in order and keeps the first that matches, with greedy,
/// backtracking quantifiers. Every scalar is a letter, a number, white space or other, so some alternative always
/// matches and the matches tile the text. At position p the match ends at:
///
///  1. p+2 or p+3 when U+0027 is followed by s, t, m or d, or by re, ve or ll, under simple case folding;
///  2. the end of the letter run when p is a letter, or when p is neither \r, \n nor a number and p+1 is a letter;
///  3. p+1 when p is a number;
///  4. when p is other, or a U+0020 space followed by other: the end of the run of others, then of \r and \n;
///  5. when p starts a white-space run [p, q) holding \r or \n: just after the last \r or \n in it;
///  6. otherwise q when the run ends the text, or q-1 when the run has at least two scalars;
///  7. otherwise q (a single white-space scalar).
///
/// System.Text.RegularExpressions cannot do this: it matches UTF-16 code units, so \p{L} never matches an astral
/// letter and its surrogates fall into [^\s\p{L}\p{N}] (the "Regex regression" test pins the difference).
module Lrm.Tokenizer.PreTokenizer

open System

/// The Qwen Split regex, character for character as tokenizer.json holds it (the loader requires exactly this).
[<Literal>]
let SplitPattern =
    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

/// Per-thread scratch arrays: scalar values, UTF-16 offsets and classes of the current text.
[<Sealed; AllowNullLiteral>]
type private Scratch() =
    member val Cp: int[] = Array.zeroCreate 256 with get, set
    member val Off: int[] = Array.zeroCreate 257 with get, set
    member val Cls: CharClass[] = Array.zeroCreate 256 with get, set

    [<ThreadStatic; DefaultValue>]
    static val mutable private current: Scratch

    static member Get(length: int) =
        let mutable s = Scratch.current

        if isNull s then
            s <- Scratch()
            Scratch.current <- s

        if s.Cp.Length < length then
            let n = max length (2 * s.Cp.Length)
            s.Cp <- Array.zeroCreate n
            s.Off <- Array.zeroCreate (n + 1)
            s.Cls <- Array.zeroCreate n

        s

let private isCrLf (v: int) = v = 0x0D || v = 0x0A

/// The first index at or after `from` whose class is not `k`.
let private runEnd (cls: CharClass[]) (n: int) (k: CharClass) (from: int) =
    let mutable i = from

    while i < n && cls[i] = k do
        i <- i + 1

    i

/// The end of the match that starts at `p` (rules 1 to 7 above).
let private matchEnd (cp: int[]) (cls: CharClass[]) (n: int) (p: int) =
    let c = cp[p]
    let k = cls[p]

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    let contraction =
        if c = 0x27 && p + 1 < n then
            match CharClass.contractionLetter cp[p + 1] with
            | 's'
            | 't'
            | 'm'
            | 'd' -> p + 2
            | 'r'
            | 'v' when p + 2 < n && CharClass.contractionLetter cp[p + 2] = 'e' -> p + 3
            | 'l' when p + 2 < n && CharClass.contractionLetter cp[p + 2] = 'l' -> p + 3
            | _ -> -1
        else
            -1

    if contraction >= 0 then
        contraction
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    elif k = CharClass.Letter then
        runEnd cls n CharClass.Letter (p + 1)
    elif
        k <> CharClass.Number
        && not (isCrLf c)
        && p + 1 < n
        && cls[p + 1] = CharClass.Letter
    then
        runEnd cls n CharClass.Letter (p + 2)
    // 3. \p{N}
    elif k = CharClass.Number then
        p + 1
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    elif k = CharClass.Other || (c = 0x20 && p + 1 < n && cls[p + 1] = CharClass.Other) then
        let mutable e =
            runEnd cls n CharClass.Other (if k = CharClass.Other then p + 1 else p + 2)

        while e < n && isCrLf cp[e] do
            e <- e + 1

        e
    else
        // White space: the run [p, q) and its last \r or \n.
        let mutable q = p
        let mutable lastCrLf = -1

        while q < n && cls[q] = CharClass.Space do
            if isCrLf cp[q] then
                lastCrLf <- q

            q <- q + 1

        // 5. \s*[\r\n]+
        if lastCrLf >= 0 then
            lastCrLf + 1
        // 6. \s+(?!\S)
        elif q = n then
            n
        elif q - p >= 2 then
            q - 1
        // 7. \s+
        else
            q

/// Calls `piece start length` (in UTF-16 code units) for every piece of `text`, in order. `text` must hold no lone
/// surrogate.
let split (text: string) (piece: int -> int -> unit) =
    let s = Scratch.Get text.Length
    let cp = s.Cp
    let off = s.Off
    let cls = s.Cls
    let mutable n = 0
    let mutable i = 0

    while i < text.Length do
        let c = text[i]
        off[n] <- i

        if Char.IsHighSurrogate c && i + 1 < text.Length && Char.IsLowSurrogate text[i + 1] then
            cp[n] <- Char.ConvertToUtf32(c, text[i + 1])
            i <- i + 2
        else
            cp[n] <- int c
            i <- i + 1

        cls[n] <- CharClass.get cp[n]
        n <- n + 1

    off[n] <- text.Length
    let mutable p = 0

    while p < n do
        let e = matchEnd cp cls n p
        piece off[p] (off[e] - off[p])
        p <- e

/// The pieces of `text` as strings.
let pieces (text: string) : string list =
    let out = ResizeArray<string>()
    split text (fun start length -> out.Add(text.Substring(start, length)))
    List.ofSeq out
