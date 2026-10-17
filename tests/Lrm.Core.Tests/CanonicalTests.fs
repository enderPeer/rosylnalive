module Lrm.Core.Tests.CanonicalTests

open System
open System.Text
open Expecto
open Lrm.Core
open Lrm.Core.Tests.Common
open Lrm.Testing

/// The KAT payload of the LrmCore contract (docs/contracts/lrm-core.md, "Canonical encoding vector").
let katWrite (w: CanonicalWriter) =
    w.U8 0x2Auy
    w.Bool true
    w.U16 0x1234us
    w.U32 0xDEADBEEFu
    w.I32 -2
    w.U64 0x0123456789ABCDEFUL
    w.I64 -1L
    w.F32 1.5f
    w.F64 -0.0
    w.String "h\u00e9llo"
    w.Bytes [| 1uy; 2uy; 3uy |]
    w.Option(None, w.U32)
    w.Option(Some 7u, w.U32)
    w.Seq([ 1u; 2u ], w.U32)
    w.Map([ "b", 2uy; "a", 1uy ], w.U8)
    w.Id(Hash.sha256Utf8 "abc")

/// The bytes katWrite must produce, field by field.
let katPayload =
    String.concat
        ""
        [
            "2a" // u8 0x2a
            "01" // bool true
            "3412" // u16 0x1234
            "efbeadde" // u32 0xdeadbeef
            "feffffff" // i32 -2
            "efcdab8967452301" // u64 0x0123456789abcdef
            "ffffffffffffffff" // i64 -1
            "0000c03f" // f32 1.5
            "0000000000000080" // f64 -0.0
            "06000000" + "68c3a96c6c6f" // string "h\u00e9llo" (6 UTF-8 bytes)
            "03000000" + "010203" // bytes [1; 2; 3]
            "00" // None
            "01" + "07000000" // Some 7u
            "02000000" + "01000000" + "02000000" // seq [1u; 2u]
            "02000000" + "01000000" + "61" + "01" + "01000000" + "62" + "02" // map {a: 1, b: 2}, keys sorted
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" // ContentId SHA-256("abc")
        ]

/// SHA-256("lrm.test/1" || 0x00 || katPayload), computed independently with sha256sum.
let katHash = "bcaeed69c2d44298c10172d350032c34ab1e5eb45e12051ba7c3869a4933b23e"

/// SHA-256("lrm.test/1" || 0x00), the hash of an empty payload.
let emptyHash = "7390733e381c00f7903c802456310b720fdec8109fe53d67f28df45d3db9aad4"

let private isValidUtf16 (s: string) =
    try
        UTF8Encoding(false, true).GetBytes s |> ignore
        true
    with _ ->
        false

let private roundTrip (write: CanonicalWriter -> unit) (read: CanonicalReader -> 'a) =
    let bytes = Canonical.encode write
    let r = CanonicalReader bytes
    let value = read r
    r.ExpectEnd()
    value

let private rejects (bytes: string) (read: CanonicalReader -> unit) =
    Expect.throwsT<FormatException> (fun () -> read (CanonicalReader(unhex bytes))) bytes

let tests =
    testList
        "canonical"
        [
            testCase "KAT payload bytes"
            <| fun () -> Expect.equal (hex (Canonical.encode katWrite)) katPayload "payload"

            testCase "KAT hash is SHA-256(tag || 0x00 || payload)"
            <| fun () ->
                Expect.equal (Canonical.hash "lrm.test/1" katWrite).Hex katHash "hash"
                Expect.equal (Canonical.hashBytes "lrm.test/1" (unhex katPayload)).Hex katHash "hashBytes"

                let manual =
                    Array.concat [ Encoding.ASCII.GetBytes "lrm.test/1"; [| 0uy |]; unhex katPayload ]

                Expect.equal (Hash.sha256 manual).Hex katHash "explicit concatenation"

            testCase "empty payload"
            <| fun () -> Expect.equal (Canonical.hash "lrm.test/1" ignore).Hex emptyHash "empty"

            testCase "domain separation"
            <| fun () ->
                let payload = Canonical.encode (fun w -> w.String "x")
                let a = Canonical.hashBytes "lrm.a/1" payload
                let b = Canonical.hashBytes "lrm.b/1" payload
                Expect.notEqual a b "different tags give different ids"
                Expect.notEqual a (Hash.sha256 payload) "a tagged hash is not the plain hash"
                // The 0x00 separator keeps (tag "ab", payload "c") apart from (tag "a", payload "bc").
                let ab = Canonical.hashBytes "ab" (Encoding.ASCII.GetBytes "c")
                let a' = Canonical.hashBytes "a" (Encoding.ASCII.GetBytes "bc")
                Expect.notEqual ab a' "separator"

            testCase "tags must be printable ASCII"
            <| fun () ->
                for bad in [ ""; "has space"; "tab\t"; "nul\000"; "caf\u00e9"; String('a', 129) ] do
                    Expect.isFalse (Canonical.isValidTag bad) (sprintf "%A" bad)

                    Expect.throwsT<ArgumentException>
                        (fun () -> Canonical.hash bad ignore |> ignore)
                        (sprintf "hash %A" bad)

                Expect.isTrue (Canonical.isValidTag "lrm.xclass/1") "lrm.xclass/1"

            testCase "map keys sort by UTF-8 bytes (code point order), not UTF-16"
            <| fun () ->
                // U+FF61 sorts before U+1F600 by code point, but after it by UTF-16 code unit (0xFF61 > 0xD83D).
                let bytes =
                    Canonical.encode (fun w -> w.Map([ "\U0001F600", 2uy; "\uff61", 1uy ], w.U8))

                Expect.equal (hex bytes) "0200000003000000efbda10104000000f09f988002" "order"

                Expect.equal
                    (Canonical.hashBytes "lrm.test/1" bytes).Hex
                    "6658a5c6133053fc57c29797182eae1f2fb6f773c658691de0cf468dad570edb"
                    "hash"

            testCase "maps reject duplicate keys"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () -> Canonical.encode (fun w -> w.Map([ "a", 1uy; "a", 2uy ], w.U8)) |> ignore)
                    "duplicate"

            testCase "strings are strict UTF-8 and not normalized"
            <| fun () ->
                // Built at run time: the F# compiler turns a lone-surrogate escape in a literal into U+FFFD.
                Expect.throwsT<ArgumentException>
                    (fun () -> Canonical.encode (fun w -> w.String(String(char 0xD800, 1))) |> ignore)
                    "lone surrogate"

                let nfc = Canonical.encode (fun w -> w.String "\u00e9")
                let nfd = Canonical.encode (fun w -> w.String "e\u0301")
                Expect.notEqual nfc nfd "NFC and NFD stay distinct"

                Expect.throwsT<ArgumentNullException>
                    (fun () -> Canonical.encode (fun w -> w.String null) |> ignore)
                    "null"

            testCase "floats keep their raw bits"
            <| fun () ->
                let enc (x: float32) =
                    hex (Canonical.encode (fun w -> w.F32 x))

                Expect.equal (enc 0.0f) "00000000" "+0"
                Expect.equal (enc -0.0f) "00000080" "-0"
                Expect.equal (enc (BitConverter.UInt32BitsToSingle 0x7FC00001u)) "0100c07f" "NaN payload"

            testCase "the reader rejects non-canonical input"
            <| fun () ->
                rejects "02" (fun r -> r.Bool() |> ignore)
                rejects "02" (fun r -> r.Option(fun () -> r.U8()) |> ignore)
                rejects "0500000061" (fun r -> r.String() |> ignore)
                rejects "01000000ff" (fun r -> r.String() |> ignore)
                rejects "02000000010000006201010000006102" (fun r -> r.Map(fun () -> r.U8()) |> ignore)
                rejects "02000000010000006101010000006102" (fun r -> r.Map(fun () -> r.U8()) |> ignore)
                rejects "0000" (fun r -> r.U32() |> ignore)

                Expect.throwsT<FormatException>
                    (fun () ->
                        let r = CanonicalReader(unhex "0000")
                        r.U8() |> ignore
                        r.ExpectEnd())
                    "trailing bytes"

            testCase "the KAT payload reads back"
            <| fun () ->
                let r = CanonicalReader(unhex katPayload)
                Expect.equal (r.U8()) 0x2Auy "u8"
                Expect.isTrue (r.Bool()) "bool"
                Expect.equal (r.U16()) 0x1234us "u16"
                Expect.equal (r.U32()) 0xDEADBEEFu "u32"
                Expect.equal (r.I32()) -2 "i32"
                Expect.equal (r.U64()) 0x0123456789ABCDEFUL "u64"
                Expect.equal (r.I64()) -1L "i64"
                Expect.equal (r.F32()) 1.5f "f32"
                Expect.equal (BitConverter.DoubleToUInt64Bits(r.F64())) 0x8000000000000000UL "-0.0"
                Expect.equal (r.String()) "h\u00e9llo" "string"
                Expect.equal (r.Bytes()) [| 1uy; 2uy; 3uy |] "bytes"
                Expect.equal (r.Option(fun () -> r.U32())) None "None"
                Expect.equal (r.Option(fun () -> r.U32())) (Some 7u) "Some"
                Expect.equal (r.Seq(fun () -> r.U32())) [ 1u; 2u ] "seq"
                Expect.equal (r.Map(fun () -> r.U8())) [ "a", 1uy; "b", 2uy ] "map"
                Expect.equal (r.Id()) (Hash.sha256Utf8 "abc") "id"
                r.ExpectEnd()

            Props.testProperty "integers and floats round-trip"
            <| fun (a: byte, b: uint16, c: uint32, d: int, e: uint64, f: int64, g: float32, h: float) ->
                let back =
                    roundTrip
                        (fun w ->
                            w.U8 a
                            w.U16 b
                            w.U32 c
                            w.I32 d
                            w.U64 e
                            w.I64 f
                            w.F32 g
                            w.F64 h)
                        (fun r -> r.U8(), r.U16(), r.U32(), r.I32(), r.U64(), r.I64(), r.F32(), r.F64())

                let a', b', c', d', e', f', g', h' = back

                a' = a
                && b' = b
                && c' = c
                && d' = d
                && e' = e
                && f' = f
                && BitConverter.SingleToUInt32Bits g' = BitConverter.SingleToUInt32Bits g
                && BitConverter.DoubleToUInt64Bits h' = BitConverter.DoubleToUInt64Bits h

            Props.testProperty "strings round-trip or are rejected for lone surrogates"
            <| fun (s: string) ->
                if isNull s then
                    true
                elif isValidUtf16 s then
                    roundTrip (fun w -> w.String s) (fun r -> r.String()) = s
                else
                    try
                        Canonical.encode (fun w -> w.String s) |> ignore
                        false
                    with :? ArgumentException ->
                        true

            Props.testProperty "collections round-trip"
            <| fun (bytes: byte[], xs: int list, o: uint64 option, m: Map<string, int>) ->
                let entries =
                    m |> Map.toList |> List.filter (fun (k, _) -> not (isNull k) && isValidUtf16 k)

                let b', xs', o', m' =
                    roundTrip
                        (fun w ->
                            w.Bytes bytes
                            w.Seq(xs, w.I32)
                            w.Option(o, w.U64)
                            w.Map(entries, w.I32))
                        (fun r ->
                            r.Bytes(), r.Seq(fun () -> r.I32()), r.Option(fun () -> r.U64()), r.Map(fun () -> r.I32()))

                let sortedKeys =
                    entries
                    |> List.map fst
                    |> List.sortWith (fun a b ->
                        ReadOnlySpan(Encoding.UTF8.GetBytes a)
                            .SequenceCompareTo(ReadOnlySpan(Encoding.UTF8.GetBytes b)))

                b' = bytes
                && xs' = xs
                && o' = o
                && List.map fst m' = sortedKeys
                && Map.ofList m' = Map.ofList entries

            Props.testProperty "map encoding does not depend on entry order"
            <| fun (m: Map<string, int>) ->
                let entries =
                    m |> Map.toList |> List.filter (fun (k, _) -> not (isNull k) && isValidUtf16 k)

                let forward = Canonical.encode (fun w -> w.Map(entries, w.I32))
                let backward = Canonical.encode (fun w -> w.Map(List.rev entries, w.I32))
                forward = backward
        ]
