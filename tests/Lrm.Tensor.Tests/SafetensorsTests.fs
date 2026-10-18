/// HostTensorAndSafetensors: the deterministic writer, the validating memory-mapped reader, and cross-checks with the
/// golden files written by O1's Python writer.
module Lrm.Tensor.Tests.SafetensorsTests

open System
open System.IO
open System.Text
open Expecto
open Lrm.Core
open Lrm.Tensor
open Lrm.Testing
open Lrm.Tensor.Tests.Common

/// The writer known-answer vector S1 of the contract. Python's json.dumps(ensure_ascii=True) writer (the convention
/// of tools/goldens/common.py) gives these bytes for the same input.
module S1 =
    let eAcute = string (char 0xE9)
    let name = eAcute + "\U0001F600"
    let value = "\"q\"\\\n" + string (char 0x7F) + "\t"
    let metadata = [ "lrm.test", "sa" + string (char 0xEF) + "tensors!"; "a", value ]

    let tensors =
        [
            SafetensorsTensor.ofHost "x" (HostTensor.f32 (Shape.vector 2) [| 1.0f; -2.0f |])
            SafetensorsTensor.ofHost name (HostTensor.bf16 (Shape.matrix 1 1) [| 0x3F80us |])
            SafetensorsTensor.ofHost "b" (HostTensor.i64 Shape.Scalar [| 7L |])
            SafetensorsTensor.ofHost "e" (HostTensor.u8 (Shape.vector 0) [||])
        ]

    let headerJson =
        "{\"__metadata__\":{\"a\":\"\\\"q\\\"\\\\\\n\\u007f\\t\",\"lrm.test\":\"sa\\u00eftensors!\"},"
        + "\"b\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[0,8]},"
        + "\"e\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[8,8]},"
        + "\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]},"
        + "\"\\u00e9\\ud83d\\ude00\":{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[16,18]}}"
        + String(' ', 7)

    let sha256 = "49fc00c6aa63a8f23a5488fe2bd49a8768779e8f5b943a166013f4b2140666ce"

/// A hand-built file: the 8-byte length, the header text and the data.
let rawFile (header: string) (data: byte[]) =
    let h = Encoding.UTF8.GetBytes header
    Array.concat [ BitConverter.GetBytes(uint64 h.Length); h; data ]

let withFile (bytes: byte[]) (f: string -> 'a) =
    let dir = tempDir "st"

    try
        let path = Path.Combine(dir, "t.safetensors")
        File.WriteAllBytes(path, bytes)
        f path
    finally
        Directory.Delete(dir, true)

let expectFormatError (label: string) (bytes: byte[]) =
    withFile bytes (fun path ->
        Expect.throwsT<SafetensorsFormatException> (fun () -> SafetensorsFile.Open(path).Dispose()) label)

let private entry (dtype: string) (shape: string) (b: int) (e: int) =
    Invariant.format $"{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{b},{e}]}}"

/// A random set of tensors for the round-trip properties.
type Sample =
    {
        Tensors: SafetensorsTensor list
        Metadata: (string * string) list
    }

let private sampleOf (seed: int) (count: int) =
    let rng = DetRng(uint64 seed, 9UL)
    let dtypes = DType.all |> Array.ofList

    let tensors =
        [
            for i in 0 .. count - 1 do
                let dtype = dtypes[int (rng.NextBelow(uint32 dtypes.Length))]
                let rank = int (rng.NextBelow 4u)
                let dims = List.init rank (fun _ -> int (rng.NextBelow 5u))
                let shape = Shape.ofList dims
                let bytes = Array.zeroCreate<byte>(shape.ElementCount * DType.size dtype)

                rng.Fill(
                    Runtime.InteropServices.MemoryMarshal.Cast<byte, uint32>(Span<byte>(bytes, 0, bytes.Length / 4 * 4))
                )
                // Names with characters that need escaping and astral characters, unique by index.
                let name =
                    Invariant.format $"t{i}"
                    + (if i % 3 = 0 then "\U0001F600" else "")
                    + (if i % 4 = 1 then "\"" else "")

                yield SafetensorsTensor.ofBytes name dtype shape bytes
        ]

    {
        Tensors = tensors
        Metadata =
            [
                for i in 0 .. int (rng.NextBelow 3u) - 1 -> Invariant.format $"k{i}", Invariant.format $"v{i}\n"
            ]
    }

let tests =
    testList
        "safetensors"
        [
            testCase "writer known-answer vector S1 (matches Python json.dumps ensure_ascii)"
            <| fun () ->
                let bytes = Safetensors.toBytes S1.metadata S1.tensors
                let n = BitConverter.ToUInt64(bytes, 0)
                Expect.equal n 312UL "header length, padded to 8"
                Expect.equal (Encoding.ASCII.GetString(bytes, 8, 312)) S1.headerJson "header JSON"
                Expect.equal bytes.Length 338 "file length"
                Expect.equal (Hash.sha256 bytes).Hex S1.sha256 "SHA-256"

            testCase "writer output is independent of input order"
            <| fun () ->
                let a = Safetensors.toBytes S1.metadata S1.tensors
                let b = Safetensors.toBytes (List.rev S1.metadata) (List.rev S1.tensors)
                Expect.sequenceEqual b a "same bytes"

            testCase "JSON strings escape like Python ensure_ascii"
            <| fun () ->
                let s =
                    String([| char 0; char 0x1F; char 0x7F; char 0x80; char 0xFFFF |])
                    + "\U0010FFFF"

                Expect.equal
                    (Safetensors.jsonString s)
                    "\"\\u0000\\u001f\\u007f\\u0080\\uffff\\udbff\\udfff\""
                    "escapes"

                Expect.equal
                    (Safetensors.jsonString "a\"b\\c\n\r\t\b\f~ ")
                    "\"a\\\"b\\\\c\\n\\r\\t\\b\\f~ \""
                    "short escapes"

            testCase "reader opens S1 and returns every tensor, its metadata and the header"
            <| fun () ->
                withFile (Safetensors.toBytes S1.metadata S1.tensors) (fun path ->
                    use f = SafetensorsFile.Open path
                    Expect.equal f.HeaderLength 312L "N"
                    Expect.equal f.DataOffset 320L "data offset"
                    Expect.sequenceEqual f.Names [| "b"; "e"; "x"; S1.name |] "names in code point order"
                    Expect.sequenceEqual f.Metadata (List.sortBy fst S1.metadata |> Array.ofList) "metadata"
                    Expect.sequenceEqual (f.ReadF32 "x").Data [| 1.0f; -2.0f |] "F32"
                    Expect.sequenceEqual (f.ReadBf16 S1.name).Data [| 0x3F80us |] "BF16 bits"
                    Expect.equal (f.ReadI64 "b").Shape.Rank 0 "scalar"
                    Expect.equal (f.ReadU8 "e").Length 0 "empty tensor"
                    Expect.isTrue f.IsAligned "aligned"
                    Expect.sequenceEqual (f.RebuildHeader()) (f.HeaderBytes()) "header rebuilt"
                    Expect.equal (f.Sha256()).Hex S1.sha256 "Sha256 of the mapped file"
                    Expect.throwsT<ArgumentException> (fun () -> f.ReadF64 "x" |> ignore) "dtype checked")

            testCase "every tiny golden file is rebuilt byte for byte by the F# writer"
            <| fun () ->
                let set = tiny.Force()

                for g in set.Files do
                    if g.Name.EndsWith(".safetensors", StringComparison.Ordinal) then
                        use f = set.OpenSafetensors g.Name
                        Expect.isTrue f.IsAligned (g.Name + " aligned")

                        let tensors =
                            f.Names |> Array.map (fun n -> SafetensorsTensor.ofHost n (f.Tensor n))

                        let bytes = Safetensors.toBytes f.Metadata tensors
                        Expect.equal (Hash.sha256 bytes) g.Sha256 (g.Name + " rewritten")

                use m = set.OpenSafetensors "model/model.safetensors"
                Expect.equal (m.Entries.Length, m.HeaderLength) (24, 2480L) "tiny model: 24 tensors, N = 2,480"
                Expect.sequenceEqual m.Metadata [| "format", "pt" |] "tiny model metadata"
                Expect.isTrue (m.Entries |> Array.forall (fun e -> e.Desc.DType = DType.BF16)) "all BF16"
                Expect.isFalse (m.Contains "lm_head.weight") "no lm_head (tied)"

            testCase "while a file is open nobody can write it, and spans end with Dispose"
            <| fun () ->
                withFile (Safetensors.toBytes S1.metadata S1.tensors) (fun path ->
                    let f = SafetensorsFile.Open path
                    Expect.throwsT<IOException> (fun () -> File.OpenWrite(path).Dispose()) "writers are refused"
                    (f :> IDisposable).Dispose()
                    Expect.throwsT<ObjectDisposedException> (fun () -> f.Span("x").Length |> ignore) "disposed"
                    File.WriteAllBytes(path, [| 0uy |]))

            testCase "OpenVerified checks size and SHA-256 before parsing"
            <| fun () ->
                let bytes = Safetensors.toBytes S1.metadata S1.tensors

                withFile bytes (fun path ->
                    let sha = Hash.sha256 bytes
                    (SafetensorsFile.OpenVerified(path, int64 bytes.Length, sha) :> IDisposable).Dispose()

                    Expect.throwsT<IntegrityException>
                        (fun () -> SafetensorsFile.OpenVerified(path, int64 bytes.Length + 1L, sha).Dispose())
                        "size"

                    Expect.throwsT<IntegrityException>
                        (fun () ->
                            SafetensorsFile.OpenVerified(path, int64 bytes.Length, Hash.sha256Utf8 "x").Dispose())
                        "hash")

                // A file whose header is garbage but whose hash is pinned fails the hash check first.
                let junk = Array.create 64 0xFFuy

                withFile junk (fun path ->
                    Expect.throwsT<IntegrityException>
                        (fun () -> SafetensorsFile.OpenVerified(path, 64L, Hash.sha256Utf8 "x").Dispose())
                        "hash before parse")

            testList
                "the reader rejects malformed files"
                [
                    let ok = entry "F32" "[1]" 0 4
                    let data4 = Array.zeroCreate<byte> 4

                    yield testCase "shorter than 8 bytes" (fun () -> expectFormatError "short" [| 1uy; 2uy |])

                    yield
                        testCase "header length beyond the file" (fun () ->
                            expectFormatError "N" (Array.append (BitConverter.GetBytes 100UL) (Array.zeroCreate 8)))

                    yield
                        testCase "header length above 100 MB" (fun () ->
                            expectFormatError
                                "cap"
                                (Array.append (BitConverter.GetBytes 100_000_001UL) (Array.zeroCreate 8)))

                    for label, header, data in
                        [
                            "not JSON", "{\"a\":", data4
                            "not an object", "[1, 2]", data4
                            "leading space", " {}", [||]
                            "duplicate name", "{\"a\":" + ok + ",\"a\":" + ok + "}", data4
                            "unknown dtype", "{\"a\":" + entry "BOOL" "[4]" 0 4 + "}", data4
                            "negative dimension", "{\"a\":" + entry "U8" "[-4]" 0 4 + "}", data4
                            "fractional dimension", "{\"a\":" + entry "U8" "[4.0]" 0 4 + "}", data4
                            "size mismatch", "{\"a\":" + entry "F32" "[2]" 0 4 + "}", data4
                            "end beyond the data", "{\"a\":" + entry "U8" "[8]" 0 8 + "}", data4
                            "reversed offsets", "{\"a\":" + entry "U8" "[0]" 4 0 + "}", data4
                            "hole", "{\"a\":" + entry "U8" "[2]" 2 4 + "}", data4
                            "overlap", "{\"a\":" + entry "U8" "[3]" 0 3 + ",\"b\":" + entry "U8" "[2]" 2 4 + "}", data4
                            "trailing bytes", "{\"a\":" + entry "U8" "[2]" 0 2 + "}", data4
                            "offsets not a pair", "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0]}}", data4
                            "missing field", "{\"a\":{\"dtype\":\"U8\",\"shape\":[4]}}", data4
                            "unknown field",
                            "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4],\"x\":1}}",
                            data4
                            "empty name", "{\"\":" + ok + "}", data4
                            "metadata not strings", "{\"__metadata__\":{\"k\":1},\"a\":" + ok + "}", data4
                            "metadata not an object", "{\"__metadata__\":[],\"a\":" + ok + "}", data4
                        ] do
                        yield testCase label (fun () -> expectFormatError label (rawFile header data))
                ]

            testCase "the reader accepts any header length and records alignment"
            <| fun () ->
                // N = 47 is not a multiple of 8: foreign writers may do this; IsAligned reports it.
                let header = "{\"a\":" + entry "U8" "[4]" 0 4 + "}"
                let bytes = rawFile header [| 1uy; 2uy; 3uy; 4uy |]

                withFile bytes (fun path ->
                    use f = SafetensorsFile.Open path
                    Expect.equal f.HeaderLength (int64 header.Length) "N"
                    Expect.isFalse f.IsAligned "data start not a multiple of 8"
                    Expect.sequenceEqual (f.ReadU8 "a").Data [| 1uy; 2uy; 3uy; 4uy |] "data")

            testCase "the writer rejects duplicate, reserved and invalid names and short data"
            <| fun () ->
                let t n =
                    SafetensorsTensor.ofBytes n DType.U8 (Shape.vector 1) [| 0uy |]

                Expect.throwsT<ArgumentException> (fun () -> Safetensors.toBytes [] [ t "a"; t "a" ] |> ignore) "dup"

                Expect.throwsT<ArgumentException>
                    (fun () -> Safetensors.toBytes [] [ t "__metadata__" ] |> ignore)
                    "reserved"

                Expect.throwsT<ArgumentException> (fun () -> Safetensors.toBytes [] [ t "" ] |> ignore) "empty"

                Expect.throwsT<ArgumentException>
                    (fun () -> Safetensors.toBytes [] [ t (string (char 0xD800)) ] |> ignore)
                    "lone surrogate"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        Safetensors.toBytes [] [ SafetensorsTensor.ofBytes "a" DType.F32 (Shape.vector 2) [| 0uy |] ]
                        |> ignore)
                    "short data"

                Expect.throwsT<ArgumentException>
                    (fun () -> Safetensors.toBytes [ "k", "1"; "k", "2" ] [] |> ignore)
                    "dup key"

            testCase "writeFile replaces the target atomically and leaves no temporary file"
            <| fun () ->
                let dir = tempDir "stw"

                try
                    let path = Path.Combine(dir, "w.safetensors")
                    File.WriteAllText(path, "old")
                    Safetensors.writeFile path S1.metadata S1.tensors
                    Expect.equal (Hash.sha256File path).Hex S1.sha256 "written"

                    Expect.sequenceEqual
                        (Directory.GetFiles dir |> Array.map Path.GetFileName)
                        [| "w.safetensors" |]
                        "no temp"
                finally
                    Directory.Delete(dir, true)

            Props.testProperty "round trip: write, read, rewrite gives the same tensors and bytes"
            <| fun (seed: int) (count: byte) ->
                let s = sampleOf seed (int count % 12)
                let bytes = Safetensors.toBytes s.Metadata s.Tensors
                let n, metadata, entries = SafetensorsFile.Validate bytes

                n % 8L = 0L
                && entries.Length = s.Tensors.Length
                && (metadata |> Array.sortBy fst) = (s.Metadata |> List.sortBy fst |> Array.ofList)
                && withFile bytes (fun path ->
                    use f = SafetensorsFile.Open path

                    let same =
                        s.Tensors
                        |> List.forall (fun t ->
                            let e = f.Entry t.Name

                            e.Desc.DType = t.DType
                            && e.Desc.Shape.Equals t.Shape
                            && (f.Span t.Name).SequenceEqual(t.Data.Span))

                    let again =
                        Safetensors.toBytes
                            f.Metadata
                            (f.Names |> Array.map (fun n -> SafetensorsTensor.ofHost n (f.Tensor n)))

                    same && again = bytes && f.RebuildHeader() = f.HeaderBytes())
        ]
