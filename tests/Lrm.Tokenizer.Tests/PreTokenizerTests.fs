/// The Split pre-tokenizer and the byte map (TokenizerApi contract, vectors P1 to P18 and B1). Neither needs
/// tokenizer.json, so these run in CI.
module Lrm.Tokenizer.Tests.PreTokenizerTests

open System
open Expecto
open Lrm.Core
open Lrm.Tokenizer
open Lrm.Tokenizer.Tests.Common

/// Vectors P1 to P18 of docs/contracts/tokenizer.md, from the oracle's Split pre-tokenizer.
let splitVectors: (string * string list) list =
    let astral (v: int) = Char.ConvertFromUtf32 v

    [
        "Hello world", [ "Hello"; " world" ]
        "it's", [ "it"; "'s" ]
        "WE'RE 'Sa", [ "WE"; "'RE"; " '"; "Sa" ]
        "x'\u017fa", [ "x"; "'\u017f"; "a" ]
        "  hi", [ " "; " hi" ]
        "a  \n\n  b", [ "a"; "  \n\n"; " "; " b" ]
        "end   ", [ "end"; "   " ]
        "12345", [ "1"; "2"; "3"; "4"; "5" ]
        "x\u00bd\u216b", [ "x"; "\u00bd"; "\u216b" ]
        " !!\r\n\r\nx", [ " !!\r\n\r\n"; "x" ]
        "\tfoo", [ "\tfoo" ]
        "\u3000\u4e2d\u6587", [ "\u3000\u4e2d\u6587" ]
        astral 0x1D400 + astral 0x1D401, [ astral 0x1D400 + astral 0x1D401 ]
        "a\r\nb", [ "a"; "\r\n"; "b" ]
        "x \u0301y", [ "x"; " \u0301"; "y" ]
        "'ll've", [ "'ll"; "'ve" ]
        "don't'", [ "don"; "'t"; "'" ]
        astral 0x1D7CE + astral 0x1D7CF, [ astral 0x1D7CE; astral 0x1D7CF ]
    ]

let private splitTests =
    testList
        "split"
        [
            testCase "vectors P1 to P18"
            <| fun () ->
                for input, pieces in splitVectors do
                    Expect.equal (PreTokenizer.pieces input) pieces (describe input)

            testCase "every oracle split case (tests/Lrm.Tokenizer.Tests/oracle/split-cases.jsonl)"
            <| fun () ->
                let cases = jsonLines (oraclePath "split-cases.jsonl")
                Expect.isGreaterThanOrEqual cases.Length 4000 "the probe wrote at least 4,000 cases"

                let bad =
                    [
                        for e in cases do
                            let t = text e "text"
                            let want = [ for p in e.GetProperty("pieces").EnumerateArray() -> p.GetString() ]
                            let got = PreTokenizer.pieces t

                            if got <> want then
                                yield sprintf "%s: %A, the oracle has %A" (describe t) got want
                    ]

                use _ = TokenizerRun.install ()

                Telemetry.check
                    "tokenizer.split"
                    bad.IsEmpty
                    [ Attr.int "cases" cases.Length; Attr.int "mismatches" bad.Length ]

                if not bad.IsEmpty then
                    failtestf "%d split cases differ: %s" bad.Length (bad |> List.truncate 10 |> String.concat "\n")

            testCase "pieces tile the text"
            <| fun () ->
                for e in jsonLines (oraclePath "split-cases.jsonl") do
                    let t = text e "text"
                    Expect.equal (String.Concat(PreTokenizer.pieces t)) t (describe t)

            testCase "the empty text has no pieces"
            <| fun () -> Expect.isEmpty (PreTokenizer.pieces "") "no pieces"
        ]

let private byteTests =
    testList
        "byte level"
        [
            testCase "vector B1: the GPT-2 byte map"
            <| fun () ->
                for b, c in
                    [
                        0x00, '\u0100'
                        0x09, '\u0109'
                        0x0A, '\u010a'
                        0x20, '\u0120'
                        0x21, '!'
                        0x7E, '~'
                        0x7F, '\u0121'
                        0xA0, '\u0142'
                        0xA1, '\u00a1'
                        0xAC, '\u00ac'
                        0xAD, '\u0143'
                        0xFF, '\u00ff'
                    ] do
                    Expect.equal (ByteLevel.charOf (byte b)) c (sprintf "byte 0x%02X" b)

            testCase "the map is a bijection onto 256 characters, 68 of them moved to U+0100..U+0143"
            <| fun () ->
                let chars = [ for b in 0..255 -> ByteLevel.charOf (byte b) ]
                Expect.equal (List.distinct chars).Length 256 "distinct"

                Expect.equal
                    (chars |> List.filter (fun c -> c >= '\u0100') |> List.sort)
                    [ for c in 0x100..0x143 -> char c ]
                    "moved bytes"

                for b in 0..255 do
                    Expect.equal (ByteLevel.tryByteOf (ByteLevel.charOf (byte b))) (Some(byte b)) "round trip"

                let all = Array.init 256 byte
                Expect.equal (ByteLevel.tryDecode (ByteLevel.encode all)) (Some all) "string round trip"
                Expect.isNone (ByteLevel.tryByteOf '\u0144') "U+0144 is not in the map"
        ]

let tests = testList "pretokenizer" [ splitTests; byteTests ]
