module Lrm.Core.Tests.HashTests

open System
open System.IO
open System.Text
open Expecto
open Lrm.Core
open Lrm.Testing

/// FIPS 180-2 / NIST CSRC SHA-256 example vectors.
let private sha256Kats =
    [
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    ]

let private ofWords (a: uint64) (b: uint64) (c: uint64) (d: uint64) =
    let bytes = Array.zeroCreate<byte> 32

    for i, w in [ a; b; c; d ] |> List.indexed do
        Buffers.Binary.BinaryPrimitives.WriteUInt64BigEndian(Span(bytes, 8 * i, 8), w)

    bytes

let tests =
    testList
        "hash"
        [
            testCase "SHA-256 known answers"
            <| fun () ->
                for text, expected in sha256Kats do
                    Expect.equal (Hash.sha256Utf8 text).Hex expected (sprintf "sha256 %A" text)

            testCase "SHA-256 of one million 'a'"
            <| fun () ->
                let data = Array.create 1_000_000 (byte 'a')

                Expect.equal
                    (Hash.sha256 data).Hex
                    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
                    "1e6 a"

            testCase "stream, file and incremental hashing agree"
            <| fun () ->
                let data =
                    Encoding.ASCII.GetBytes "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"

                let expected = Hash.sha256 data
                use ms = new MemoryStream(data)
                Expect.equal (Hash.sha256Stream ms) expected "stream"
                let path = Path.GetTempFileName()

                try
                    File.WriteAllBytes(path, data)
                    Expect.equal (Hash.sha256File path) expected "file"
                finally
                    File.Delete path

                use inc = new Hash.Incremental()
                inc.Append(data.AsSpan(0, 10))
                inc.Append(data[10..])
                Expect.equal (inc.Finish()) expected "incremental"
                Expect.equal (inc.Finish()) (Hash.sha256 [||]) "Finish resets"

            testCase "ContentId text form"
            <| fun () ->
                let id = Hash.sha256Utf8 "abc"
                Expect.equal (string id) id.Hex "ToString is hex"
                Expect.equal id.Short "ba7816bf8f01" "short"
                Expect.equal (ContentId.parse (id.Hex.ToUpperInvariant())) id "parse accepts upper case"
                Expect.equal (ContentId.tryParse (id.Hex.Substring 2)) None "62 digits"
                Expect.equal (ContentId.tryParse null) None "null"
                Expect.equal ContentId.zero.Hex (String('0', 64)) "zero"
                Expect.throwsT<ArgumentException> (fun () -> ContentId.ofBytes [| 1uy |] |> ignore) "31 bytes"

            Props.testProperty "ContentId bytes round-trip"
            <| fun (a: uint64, b: uint64, c: uint64, d: uint64) ->
                let bytes = ofWords a b c d
                let id = ContentId.ofBytes bytes
                id.ToBytes() = bytes && ContentId.parse id.Hex = id

            Props.testProperty "ContentId order is lexicographic byte order"
            <| fun (a: uint64, b: uint64, c: uint64, d: uint64, e: uint64) ->
                let x = ofWords a b c d
                let y = ofWords a b e d
                let expected = sign (ReadOnlySpan(x).SequenceCompareTo(ReadOnlySpan y))
                let ix, iy = ContentId.ofBytes x, ContentId.ofBytes y

                sign (compare ix iy) = expected
                && (ix = iy) = (expected = 0)
                && (expected <> 0 || hash ix = hash iy)
        ]
