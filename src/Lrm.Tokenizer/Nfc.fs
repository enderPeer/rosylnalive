/// The oracle's NFC (TokenizerApi contract, "Normalization"; ADR M1-0001).
///
/// tokenizers 0.23.3 normalizes with the unicode-normalization-alignments crate, whose tables are Unicode 9.0. A scalar
/// it does not know is inert there: combining class 0, no decomposition, part of no composition, so the oracle's NFC
/// of a b c, with b inert, is NFC(a) b NFC(c). The runtime's NFC (string.Normalize with ICU) knows newer marks and
/// composites and would reorder or compose around them. So this module splits a string at every scalar of
/// OracleNfcData.splitRanges (each inert in the oracle and possibly not in a newer Unicode) and at U+FFFE (which
/// string.Normalize rejects although NFC leaves it alone), normalizes the parts with string.Normalize(FormC), and keeps
/// the split scalars as they are. Where neither side knows more than the other, this is the runtime's NFC.
module Lrm.Tokenizer.Nfc

open System
open System.Text

/// The BMP split points as a bit table, and the astral ranges as inclusive pairs.
let private tables =
    let bits = Array.zeroCreate<bool> 0x10000
    let astral = ResizeArray<int>()
    let ranges = OracleNfcData.splitRanges

    for i in 0..2 .. ranges.Length - 2 do
        let first = ranges[i]
        let last = ranges[i + 1]

        for c in first .. min last 0xFFFF do
            bits[c] <- true

        if last >= 0x10000 then
            astral.Add(max first 0x10000)
            astral.Add last

    bits[0xFFFE] <- true
    bits, astral.ToArray()

let private bmpSplit: bool[] = fst tables
let private astral: int[] = snd tables

/// True when the string is split at scalar `v` before string.Normalize.
let isSplitPoint (v: int) =
    if v < 0x10000 then
        bmpSplit[v]
    else
        // Binary search over the inclusive astral pairs.
        let mutable lo = 0
        let mutable hi = astral.Length / 2 - 1
        let mutable found = false

        while not found && lo <= hi do
            let mid = (lo + hi) / 2

            if v < astral[2 * mid] then hi <- mid - 1
            elif v > astral[2 * mid + 1] then lo <- mid + 1
            else found <- true

        found

/// The oracle's NFC of `text`, which must hold no lone surrogate.
let normalize (text: string) : string =
    // Find the first split point; most text has none, and then this is string.Normalize alone.
    let mutable i = 0
    let mutable first = -1

    while first < 0 && i < text.Length do
        let c = text[i]

        if Char.IsHighSurrogate c && i + 1 < text.Length then
            if isSplitPoint (Char.ConvertToUtf32(c, text[i + 1])) then
                first <- i

            i <- i + 2
        else
            if c >= '̀' && bmpSplit[int c] then
                first <- i

            i <- i + 1

    if first < 0 then
        text.Normalize NormalizationForm.FormC
    else
        let sb = StringBuilder(text.Length + 8)
        let mutable start = 0
        let mutable i = first

        while i < text.Length do
            let c = text[i]

            let width, v =
                if Char.IsHighSurrogate c && i + 1 < text.Length then
                    2, Char.ConvertToUtf32(c, text[i + 1])
                else
                    1, int c

            if isSplitPoint v then
                if i > start then
                    sb.Append(text.Substring(start, i - start).Normalize NormalizationForm.FormC)
                    |> ignore

                sb.Append(text, i, width) |> ignore
                start <- i + width

            i <- i + width

        if start < text.Length then
            sb.Append(text.Substring(start).Normalize NormalizationForm.FormC) |> ignore

        sb.ToString()
