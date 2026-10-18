/// The oracle's Unicode behaviour, scalar by scalar (ADR M1-0001): the Split regex's classes, the case-folded
/// contraction letters, the apostrophe, the oracle's NFC, and the Regex regression. The expected values come from
/// tests/Lrm.Tokenizer.Tests/oracle/unicode-oracle.json, which probe_oracle.py wrote from HF tokenizers 0.23.3, so these
/// tests run without tokenizer.json (in CI too).
module Lrm.Tokenizer.Tests.UnicodeTests

open System
open System.IO
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open Expecto
open Lrm.Core
open Lrm.Tokenizer
open Lrm.Tokenizer.Tests.Common

let private oracle =
    lazy
        (use doc = JsonDocument.Parse(File.ReadAllText(oraclePath "unicode-oracle.json"))
         doc.RootElement.Clone())

let private maxScalar = 0x110000

let private scalars =
    seq {
        for c in 0 .. maxScalar - 1 do
            if c < 0xD800 || c > 0xDFFF then
                yield c
    }

let private str (v: int) = Rune(v).ToString()

/// Inclusive [first, last] pairs from the JSON.
let private ranges (name: string) =
    [
        for r in oracle.Force().GetProperty(name).EnumerateArray() -> r[0].GetInt32(), r[1].GetInt32()
    ]

let private inRanges (rs: (int * int) list) =
    let set = Collections.Generic.HashSet<int>()

    for a, b in rs do
        for c in a..b do
            set.Add c |> ignore

    set

/// Emits one tokenizer.unicode check into the run log and fails with the first mismatches.
let private report (probe: string) (checked': int) (mismatches: string list) =
    use _ = TokenizerRun.install ()

    Telemetry.check
        "tokenizer.unicode"
        mismatches.IsEmpty
        [
            Attr.str "probe" probe
            Attr.int "checked" checked'
            Attr.int "mismatches" mismatches.Length
        ]

    if not mismatches.IsEmpty then
        failtestf
            "%s: %d of %d differ from the oracle, first: %s"
            probe
            mismatches.Length
            checked'
            (mismatches |> List.truncate 20 |> String.concat "; ")

let private classOf (k: string) =
    match k with
    | "L" -> CharClass.Letter
    | "N" -> CharClass.Number
    | "S" -> CharClass.Space
    | "O" -> CharClass.Other
    | other -> failwithf "unknown class %s" other

let private classTests =
    testList
        "classes"
        [
            testCase "every scalar's class equals the oracle's (Oniguruma's \\p{L}, \\p{N} and \\s)"
            <| fun () ->
                let runs =
                    [|
                        for r in oracle.Force().GetProperty("classes").EnumerateArray() ->
                            r[0].GetInt32(), r[1].GetString()
                    |]

                let bad = ResizeArray<string>()
                let mutable n = 0

                for i in 0 .. runs.Length - 1 do
                    let start, k = runs[i]
                    let stop = if i + 1 < runs.Length then fst runs[i + 1] else maxScalar

                    if k <> "X" then
                        let want = classOf k

                        for c in start .. stop - 1 do
                            n <- n + 1
                            let got = CharClass.get c

                            if got <> want then
                                bad.Add(sprintf "U+%04X is %A, the oracle says %A" c got want)

                Expect.equal n 1_112_064 "every scalar value is classified"
                report "classes" n (List.ofSeq bad)

            testCase "the contraction group matches exactly the oracle's letters at each position"
            <| fun () ->
                let letters = oracle.Force().GetProperty("contraction_letters")

                let probes =
                    [
                        "s|t|m|d", (fun ch -> "'" + ch + "a"), 2
                        "r|v (first of 're 've)", (fun ch -> "'" + ch + "ea"), 3
                        "e (second of 're)", (fun ch -> "'r" + ch + "a"), 3
                        "e (second of 've)", (fun ch -> "'v" + ch + "a"), 3
                        "l (first of 'll)", (fun ch -> "'" + ch + "la"), 3
                        "l (second of 'll)", (fun ch -> "'l" + ch + "a"), 3
                    ]

                let bad = ResizeArray<string>()
                let mutable n = 0

                for name, build, length in probes do
                    let want =
                        set [ for x in letters.GetProperty(name).EnumerateArray() -> x.GetInt32() ]

                    let got =
                        set
                            [
                                for c in scalars do
                                    if CharClass.get c = CharClass.Letter then
                                        n <- n + 1

                                        match PreTokenizer.pieces (build (str c)) with
                                        | [ first; _ ] when first.EnumerateRunes() |> Seq.length = length -> yield c
                                        | _ -> ()
                            ]

                    for c in Set.difference want got do
                        bad.Add(sprintf "%s: U+%04X matches in the oracle only" name c)

                    for c in Set.difference got want do
                        bad.Add(sprintf "%s: U+%04X matches here only" name c)

                report "contraction_letters" n (List.ofSeq bad)

            testCase "only U+0027 opens a contraction"
            <| fun () ->
                let want =
                    set
                        [
                            for x in oracle.Force().GetProperty("apostrophes").EnumerateArray() -> x.GetInt32()
                        ]

                Expect.equal want (set [ 0x27 ]) "the oracle's apostrophes"
                let bad = ResizeArray<string>()
                let mutable n = 0

                for c in scalars do
                    n <- n + 1
                    let ch = str c
                    let opens = PreTokenizer.pieces ("x" + ch + "sa") = [ "x"; ch + "s"; "a" ]

                    if opens <> want.Contains c then
                        bad.Add(sprintf "U+%04X" c)

                report "apostrophes" n (List.ofSeq bad)
        ]

/// The probe families of probe_oracle.py: the lines "<family> <hex> <hex output>\n" of every changed input.
let private families: (string * (string -> string)) list =
    [
        "F1", id
        "F2", (fun ch -> "a" + ch + "\u0334")
        "F3", (fun ch -> "a\u0345" + ch)
        "F5", (fun ch -> ch + "\u0301")
        "F6", (fun ch -> "a" + ch)
        "F7", (fun ch -> ch + "\u0308")
        "F8", (fun ch -> "\u1100" + ch)
        "F9", (fun ch -> "\uac00" + ch)
    ]

let private hexs (s: string) =
    s.EnumerateRunes()
    |> Seq.map (fun r -> sprintf "%x" r.Value)
    |> String.concat " "

/// Count and SHA-256 of a family's lines.
type private Digest() =
    let h = IncrementalHash.CreateHash HashAlgorithmName.SHA256
    let mutable count = 0

    member _.Add(line: string) =
        h.AppendData(Encoding.UTF8.GetBytes line)
        count <- count + 1

    member _.Count = count
    member _.Hex = Convert.ToHexStringLower(h.GetHashAndReset())

let private expectFamily (name: string) (d: Digest) =
    let want = oracle.Force().GetProperty("nfc").GetProperty(name)
    let wantCount = want.GetProperty("count").GetInt32()
    let wantSha = want.GetProperty("sha256").GetString()
    let count = d.Count
    let sha = d.Hex

    report
        ("nfc." + name)
        count
        (if count = wantCount && sha = wantSha then
             []
         else
             [
                 sprintf "%d changed inputs with SHA-256 %s, the oracle has %d with %s" count sha wantCount wantSha
             ])

let private nfcTests =
    testList
        "nfc"
        [
            for name, build in families do
                testCase (sprintf "family %s equals the oracle's NFC on every scalar" name)
                <| fun () ->
                    let d = Digest()

                    for c in scalars do
                        let s = build (str c)
                        let out = Nfc.normalize s

                        if out <> s then
                            d.Add(sprintf "%s %x %s\n" name c (hexs out))

                    expectFamily name d

            testCase "family P (every pair of the oracle's reordering marks) equals the oracle's NFC"
            <| fun () ->
                let marks =
                    [|
                        for a, b in ranges "marks" do
                            yield! [| a..b |]
                    |]

                let d = Digest()

                for m1 in marks do
                    for m2 in marks do
                        let s = "a" + str m1 + str m2
                        let out = Nfc.normalize s

                        if out <> s then
                            d.Add(sprintf "P %x %x %s\n" m1 m2 (hexs out))

                expectFamily "P" d

            testCase "family K (every canonical decomposition, recomposed) equals the oracle's NFC"
            <| fun () ->
                let inputs =
                    [|
                        for x in oracle.Force().GetProperty("composition_inputs").EnumerateArray() ->
                            x.GetString().Split(' ')
                            |> Array.map (fun h -> str (Convert.ToInt32(h, 16)))
                            |> String.concat ""
                    |]

                let d = Digest()

                inputs
                |> Array.iteri (fun i s ->
                    let out = Nfc.normalize s

                    if out <> s then
                        d.Add(sprintf "K %d %s\n" i (hexs out)))

                expectFamily "K" d

            testCase "the split points are the generated set plus U+FFFE"
            <| fun () ->
                let want = inRanges (ranges "split_points")
                let bad = ResizeArray<string>()
                let mutable n = 0

                for c in scalars do
                    n <- n + 1

                    if Nfc.isSplitPoint c <> want.Contains c then
                        bad.Add(sprintf "U+%04X" c)

                Expect.isTrue (Nfc.isSplitPoint 0xFFFE) "U+FFFE"
                report "split_points" n (List.ofSeq bad)

            testCase "U+FFFE, which string.Normalize rejects, passes through NFC unchanged"
            <| fun () ->
                let s = "e\u0301\ufffea\u0308"

                Expect.throwsT<ArgumentException>
                    (fun () -> s.Normalize NormalizationForm.FormC |> ignore)
                    "string.Normalize rejects U+FFFE (the reason it is a split point)"

                Expect.equal (Nfc.normalize s) "\u00e9\ufffe\u00e4" "NFC on both sides of U+FFFE"

            testCase "a mark newer than the oracle's Unicode 9 tables is not reordered"
            <| fun () ->
                // U+07FD NKO DANTAYALAN (Unicode 11, ccc 220) is inert in the oracle; string.Normalize would put
                // U+0334 (ccc 1) before it.
                let s = "a\u07fd\u0334"
                Expect.equal (s.Normalize NormalizationForm.FormC) "a\u0334\u07fd" "the runtime reorders"
                Expect.equal (Nfc.normalize s) s "the oracle does not"
        ]

/// The Qwen pattern under System.Text.RegularExpressions: what the plan verified and M1 does not use.
let private dotnetPieces (text: string) =
    [ for m in Regex.Matches(text, PreTokenizer.SplitPattern) -> m.Value ]

let private regexTests =
    testList
        "regex regression"
        [
            testCase "System.Text.RegularExpressions splits astral letters and numbers; the state machine does not"
            <| fun () ->
                let bold = Char.ConvertFromUtf32 0x1D400 + Char.ConvertFromUtf32 0x1D401
                let digit = Char.ConvertFromUtf32 0x1D7CE

                for text, oracle in
                    [
                        "a" + bold + "b", [ "a" + bold + "b" ]
                        "x" + digit + digit, [ "x"; digit; digit ]
                        Char.ConvertFromUtf32 0x20000 + "a", [ Char.ConvertFromUtf32 0x20000 + "a" ]
                    ] do
                    Expect.equal (PreTokenizer.pieces text) oracle (describe text)

                    Expect.notEqual
                        (dotnetPieces text)
                        oracle
                        (sprintf
                            "%s: .NET Regex now agrees with the oracle; the planning fact behind the state machine no longer holds (update ADR M1-0001)"
                            (describe text))

            testCase "over every split case, .NET Regex disagrees with the oracle and the state machine never does"
            <| fun () ->
                let cases = jsonLines (oraclePath "split-cases.jsonl")
                let mutable regexBad = 0
                let mutable oursBad = 0

                for e in cases do
                    let t = text e "text"
                    let want = [ for p in e.GetProperty("pieces").EnumerateArray() -> p.GetString() ]

                    if dotnetPieces t <> want then
                        regexBad <- regexBad + 1

                    if PreTokenizer.pieces t <> want then
                        oursBad <- oursBad + 1

                use _ = TokenizerRun.install ()

                Telemetry.check
                    "tokenizer.regex_regression"
                    (oursBad = 0 && regexBad > 0)
                    [
                        Attr.int "cases" cases.Length
                        Attr.int "regex_mismatches" regexBad
                        Attr.int "state_machine_mismatches" oursBad
                    ]

                printfn
                    "regex regression: .NET Regex differs from the oracle on %d of %d split cases, the state machine on %d"
                    regexBad
                    cases.Length
                    oursBad

                Expect.equal oursBad 0 "the state machine equals the oracle on every split case"
                Expect.isGreaterThan regexBad 0 ".NET Regex differs somewhere"
        ]

/// The pins of docs/contracts/tokenizer.md ("Vectors"): the oracle data and the generated split points change only
/// together with the contract.
let dataPins =
    [
        "tests/Lrm.Tokenizer.Tests/oracle/unicode-oracle.json",
        "f9339ab0e55acabd0509b4f562ae831d97d167398aaeacbd47c4a49324acbb89"
        "tests/Lrm.Tokenizer.Tests/oracle/split-cases.jsonl",
        "39bb2555c870896ecb4bc4c907c90a1dcb912aaf8ed75a143aee114c35bb3b95"
        "tests/Lrm.Tokenizer.Tests/oracle/encode-cases.jsonl",
        "3ce0e0e7473f4e6465b92ba3482c31682d61d7c48645b535a987940bdd6743dd"
        "src/Lrm.Tokenizer/OracleNfc.g.fs", "52e1ea35744313d89115f904babf8ce0c37b957e1b2eb264fb20217bab74301c"
    ]

let private dataTests =
    testList
        "oracle data"
        [
            testCase "the oracle data and the generated split points have the contract's SHA-256"
            <| fun () ->
                for path, sha in dataPins do
                    Expect.equal (ContentId.hex (Hash.sha256 (File.ReadAllBytes(repoPath path)))) sha path

            testCase "the probe's facts: the oracle's versions, classes and pattern"
            <| fun () ->
                let meta = oracle.Force().GetProperty("meta")
                Expect.equal (meta.GetProperty("tokenizers").GetString()) "0.23.3" "tokenizers"
                Expect.equal (meta.GetProperty("tokenizer_sha256").GetString()) QwenTokenizer.PinnedSha256 "pin"
                Expect.equal (meta.GetProperty("pattern").GetString()) PreTokenizer.SplitPattern "pattern"

            testCase "every source and data file of M1 is ASCII, so no editor or tool can normalize a test string"
            <| fun () ->
                let roots = [ "src/Lrm.Tokenizer"; "tests/Lrm.Tokenizer.Tests" ]
                let kinds = set [ ".fs"; ".fsproj"; ".targets"; ".py"; ".json"; ".jsonl" ]

                let files =
                    [
                        for root in roots do
                            for f in Directory.EnumerateFiles(repoPath root, "*", SearchOption.AllDirectories) do
                                if kinds.Contains(Path.GetExtension f) then
                                    yield f
                    ]

                Expect.isGreaterThan files.Length 15 "files found"

                for f in files do
                    let bytes = File.ReadAllBytes f

                    match Array.tryFindIndex (fun (b: byte) -> b >= 0x80uy) bytes with
                    | Some at -> failtestf "%s has a non-ASCII byte at offset %d; write escapes instead" f at
                    | None -> ()
        ]

let tests = testList "unicode" [ dataTests; classTests; nfcTests; regexTests ]
