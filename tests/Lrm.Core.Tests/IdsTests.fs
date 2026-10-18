module Lrm.Core.Tests.IdsTests

open System
open System.IO
open System.Text.Json
open Expecto
open Lrm.Core
open Lrm.Testing

let private toolchainDataRoot () =
    let path = Path.Combine(Env.repoRoot (), "tools", "bootstrap", "toolchain.json")
    use doc = JsonDocument.Parse(File.ReadAllText path)
    doc.RootElement.GetProperty("dataRoot").GetString()

let tests =
    testList
        "ids"
        [
            testCase "safe names"
            <| fun () ->
                for good in
                    [
                        "n0"
                        "coord"
                        "20261017-231500-3fa2c1d0"
                        "gate1-cpu"
                        "a.b_c-d"
                        String('a', 64)
                    ] do
                    Expect.isTrue (SafeName.isValid good) good

                for bad in
                    [
                        ""
                        "N0"
                        "-x"
                        ".hidden"
                        "x."
                        ".."
                        "a/b"
                        "a b"
                        "caf\u00e9"
                        String('a', 65)
                        null
                    ] do
                    Expect.isFalse (SafeName.isValid bad) (sprintf "%A" bad)

            testCase "run ids"
            <| fun () ->
                let id =
                    RunId.create (DateTimeOffset(2026, 10, 17, 23, 15, 0, TimeSpan.Zero)) 0x3fa2c1d0u

                Expect.equal id.Value "20261017-231500-3fa2c1d0" "format"

                let local = DateTimeOffset(2026, 10, 18, 1, 15, 0, TimeSpan.FromHours 2.0)
                Expect.equal (RunId.create local 0u).Value "20261017-231500-00000000" "UTC"

                let fresh = RunId.fresh ()
                Expect.isTrue (SafeName.isValid fresh.Value) "fresh is a safe name"
                Expect.equal fresh.Value.Length 24 "fresh length"
                Expect.equal (RunId.tryParse "Bad") None "upper case"
                Expect.throwsT<ArgumentException> (fun () -> RunId.parse "a/b" |> ignore) "parse"

            testCase "node ids"
            <| fun () ->
                Expect.equal (NodeId.value NodeId.local) "n0" "local"
                Expect.equal (NodeId.parse "coord").Value "coord" "parse"
                Expect.equal (NodeId.tryParse "x y") None "space"

            testCase "the default data root is toolchain.json's dataRoot"
            <| fun () ->
                Expect.equal LrmPaths.defaultDataRoot (toolchainDataRoot ()) "toolchain.json"
                Expect.equal LrmPaths.defaultDataRoot Env.DefaultDataRoot "same as Lrm.Testing"

            testCase "LRM_DATA overrides the default"
            <| fun () ->
                Expect.equal (LrmPaths.resolveDataRoot null) (Path.GetFullPath LrmPaths.defaultDataRoot) "unset"
                Expect.equal (LrmPaths.resolveDataRoot "  ") (Path.GetFullPath LrmPaths.defaultDataRoot) "blank"
                let custom = Path.Combine(Path.GetTempPath(), "lrm-data-x")
                Expect.equal (LrmPaths.resolveDataRoot (" " + custom + " ")) (Path.GetFullPath custom) "set"

            testCase "paths stay under the data root"
            <| fun () ->
                let root = Path.Combine(Path.GetTempPath(), "lrm-root")
                let full = Path.GetFullPath root

                Expect.equal
                    (LrmPaths.underRoot root (Path.Combine("runs", "r1")))
                    (Path.Combine(full, "runs", "r1"))
                    "runs/r1"

                let sep = string Path.DirectorySeparatorChar

                let bad =
                    [
                        ".."
                        Path.Combine("..", "x")
                        Path.Combine("a", "..", "..")
                        full
                        ""
                        // The root itself is not under the root, with or without a trailing separator.
                        "."
                        Path.Combine("a", "..") + sep
                    ]

                for b in bad do
                    Expect.throwsT<ArgumentException> (fun () -> LrmPaths.underRoot root b |> ignore) b

            testCase "a drive root works as the data root"
            <| fun () ->
                // C:\ on Windows, / elsewhere: the root keeps its separator when trimmed.
                let drive = Path.GetPathRoot(Path.GetTempPath())
                Expect.isTrue (Path.EndsInDirectorySeparator drive) drive
                Expect.equal (LrmPaths.underRoot drive "x") (Path.Combine(drive, "x")) "x"

                Expect.equal
                    (LrmPaths.underRoot drive (Path.Combine("runs", "r1")))
                    (Path.Combine(drive, "runs", "r1"))
                    "runs/r1"

                for b in [ ".."; "."; Path.Combine("a", "..", "..") ] do
                    Expect.throwsT<ArgumentException> (fun () -> LrmPaths.underRoot drive b |> ignore) b

            testCase "fixed locations"
            <| fun () ->
                Expect.equal (LrmPaths.run (RunId.parse "r1")) (Path.Combine(LrmPaths.dataRoot (), "runs", "r1")) "run"

                Expect.equal
                    (LrmPaths.model "Qwen3-0.6B")
                    (Path.Combine(LrmPaths.dataRoot (), "models", "Qwen3-0.6B"))
                    "model"
        ]
