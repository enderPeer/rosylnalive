module Lrm.Gpu.Core.Tests.CatalogTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common
open Lrm.Testing

/// Catalog.hash of contracts-v1, recomputed independently in Python from catalog.json
/// (SHA-256("lrm.catalog/1" || 0x00 || encoding)).
let catalogHashKat =
    "2b17293091faa39ed7267c35c6226ebf1f57e263c63629a257b268acfce1cbfe"

/// The frozen kernel list, in catalog (KernelId) order.
let kernelNames =
    [
        "scale_f32"
        "table_echo"
        "args_echo"
        "groups_echo"
        "bf16_unpack"
        "embed_gather_bf16"
        "rmsnorm_rows"
        "rope_apply"
        "silu_mul"
        "add"
        "copy_rows"
        "argmax_rows"
        "matmul_bf16w"
        "attention_gqa_causal"
    ]

let catalogJson () =
    File.ReadAllText(Path.Combine(Env.repoRoot (), CatalogGen.CatalogPath))

let private withKernel (f: KernelSpec -> KernelSpec) =
    let k = Kernels.all[0]
    KernelSpec.problems (f k)

let tests =
    testList
        "catalog"
        [
            testCase "the catalog holds the frozen kernel list"
            <| fun () ->
                Expect.equal (Kernels.all |> Array.map _.Name |> List.ofArray) kernelNames "names and order"
                Expect.equal Kernels.Version 1 "version"
                Expect.equal (Catalog.problems Kernels.catalog) [] "valid"

            testCase "threads and grids of every kernel"
            <| fun () ->
                let shape (k: KernelSpec) =
                    let gx, gy, gz = k.Grid
                    k.Name, k.Threads, (GridExpr.print gx, GridExpr.print gy, GridExpr.print gz)

                Expect.equal
                    (Kernels.all |> Array.map shape |> List.ofArray)
                    [
                        "scale_f32", (256, 1, 1), ("ceildiv(n, 256)", "1", "1")
                        "table_echo", (16, 1, 1), ("1", "1", "1")
                        "args_echo", (32, 1, 1), ("1", "1", "1")
                        "groups_echo", (1, 1, 1), ("groups", "1", "1")
                        "bf16_unpack", (256, 1, 1), ("ceildiv(n, 256)", "1", "1")
                        "embed_gather_bf16", (256, 1, 1), ("ceildiv(cols, 256)", "rows", "1")
                        "rmsnorm_rows", (256, 1, 1), ("rows", "1", "1")
                        "rope_apply", (64, 1, 1), ("rows", "1", "1")
                        "silu_mul", (256, 1, 1), ("ceildiv(n, 256)", "1", "1")
                        "add", (256, 1, 1), ("ceildiv(n, 256)", "1", "1")
                        "copy_rows", (256, 1, 1), ("ceildiv(rows * cols, 256)", "1", "1")
                        "argmax_rows", (256, 1, 1), ("rows", "1", "1")
                        "matmul_bf16w", (16, 16, 1), ("ceildiv(n, 64)", "ceildiv(m, 64)", "1")
                        "attention_gqa_causal", (64, 1, 1), ("rows", "q_heads", "1")
                    ]
                    "shapes"

            testCase "the catalog hash matches the independent vector"
            <| fun () ->
                Expect.equal (Catalog.hash Kernels.catalog).Hex catalogHashKat "hash"
                Expect.equal Kernels.CatalogHashHex catalogHashKat "generated constant"

            testCase "docs do not change the catalog hash; ABI changes do"
            <| fun () ->
                let k0 = Kernels.all[0]

                let withK (k: KernelSpec) =
                    { Kernels.catalog with
                        Kernels = k :: List.tail Kernels.catalog.Kernels
                    }

                let h0 = Catalog.hash Kernels.catalog
                Expect.equal (Catalog.hash (withK { k0 with Doc = "different" })) h0 "doc"
                Expect.notEqual (Catalog.hash (withK { k0 with Threads = (128, 1, 1) })) h0 "threads"

                Expect.notEqual (Catalog.hash (withK { k0 with Fields = List.rev k0.Fields })) h0 "field order"

            testCase "generated files are diff-clean against catalog.json"
            <| fun () ->
                match CatalogGen.stale (Env.repoRoot ()) with
                | Ok [] -> ()
                | Ok files -> failtestf "stale generated files: %A (run Lrm.Shaders.Build gen-catalog)" files
                | Error e -> failtest e

            testCase "catalog.json parses to the generated catalog"
            <| fun () ->
                match CatalogGen.parse (catalogJson ()) with
                | Ok c -> Expect.equal c Kernels.catalog "same catalog"
                | Error e -> failtest e

            testCase "the parser is strict"
            <| fun () ->
                let json = catalogJson ()

                let bad (edit: string -> string) (fragment: string) =
                    match CatalogGen.parse (edit json) with
                    | Ok _ -> failtestf "expected '%s'" fragment
                    | Error e -> Expect.stringContains e fragment "reason"

                bad (fun s -> s.Replace("lrm.catalog/1", "lrm.catalog/2")) "schema"
                bad (fun s -> s.Replace("\"threads\": [256, 1, 1]", "\"threads\": [256, 1]")) "three integers"

                bad
                    (fun s -> s.Replace("\"family\": \"selftest\"", "\"family\": \"selftest\", \"extra\": 1"))
                    "unknown property"

                bad (fun s -> s.Replace("\"type\": \"u32\"", "\"type\": \"u64\"")) "unknown type"
                bad (fun s -> s.Replace("\"dtypes\": [\"f32\"]", "\"dtypes\": [\"f99\"]")) "unknown dtype"
                bad (fun s -> s.Replace("ceildiv(n, 256)", "ceildiv(n 256)")) "grid formula"
                bad (fun s -> s.Replace("\"name\": \"add\"", "\"name\": \"scale_f32\"")) "unique"
                bad (fun s -> s.Replace("Element count.", "Element\\tcount.")) "printable ASCII"
                bad (fun s -> s + "x") "not valid JSON"

            testCase "every kernel rule is enforced"
            <| fun () ->
                let has (fragment: string) (ps: string list) =
                    Expect.isTrue (ps |> List.exists (fun p -> p.Contains fragment)) (sprintf "%s in %A" fragment ps)

                has "snake_case" (withKernel (fun k -> { k with Name = "Scale" }))
                has "reserved" (withKernel (fun k -> { k with Name = "lrm_x" }))
                has "the source must be" (withKernel (fun k -> { k with Source = "hlsl/x.hlsl" }))
                has "threads" (withKernel (fun k -> { k with Threads = (1024, 2, 1) }))
                has "threads" (withKernel (fun k -> { k with Threads = (1, 1, 65) }))

                has
                    "1 to 31 fields"
                    (withKernel (fun k ->
                        { k with
                            Fields = List.init 32 (fun i -> FieldSpec.u32 (sprintf "f%d" i) "d")
                        }))

                has "1 to 31 fields" (withKernel (fun k -> { k with Fields = [] }))

                has
                    "reserved HLSL word"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.u32 "out" "d" ]
                        }))

                has
                    "unique"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.u32 "n" "d" ]
                        }))

                has
                    "accepts any dtype"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.token "m" Access.Meta [ DType.F32 ] [] "d" ]
                        }))

                has
                    "must list its dtypes"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.token "z" Access.Read [] [] "d" ]
                        }))

                has
                    "only written fields may alias"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.token "z" Access.Read [ DType.F32 ] [ "x" ] "d" ]
                        }))

                has
                    "may alias only read token fields"
                    (withKernel (fun k ->
                        { k with
                            Fields = k.Fields @ [ FieldSpec.token "z" Access.Write [ DType.F32 ] [ "y" ] "d" ]
                        }))

                has
                    "writes at least one token"
                    (withKernel (fun k ->
                        { k with
                            Fields = [ FieldSpec.token "x" Access.Read [ DType.F32 ] [] "d"; FieldSpec.u32 "n" "d" ]
                        }))

                has
                    "not a u32 field"
                    (withKernel (fun k ->
                        { k with
                            Grid = (GridExpr.Ref "a", GridExpr.Lit 1L, GridExpr.Lit 1L)
                        }))

                has
                    "unknown field"
                    (withKernel (fun k ->
                        { k with
                            Grid = (GridExpr.Ref "q", GridExpr.Lit 1L, GridExpr.Lit 1L)
                        }))

                has
                    "at least 1"
                    (withKernel (fun k ->
                        { k with
                            Grid = (GridExpr.Lit 0L, GridExpr.Lit 1L, GridExpr.Lit 1L)
                        }))

            testCase "field words follow the catalog order"
            <| fun () ->
                for k in Kernels.all do
                    k.Fields
                    |> List.iteri (fun i f -> Expect.equal (k.Word f.Name) (i + 1) (k.Name + "." + f.Name))

                Expect.equal
                    (Kernels.all |> Array.map _.Fields.Length |> Array.max)
                    31
                    "args_echo fills all 31 field words"

            testCase "linear kernels are exactly those with y and z grids of 1"
            <| fun () ->
                let linear =
                    Kernels.all |> Array.filter _.Linear |> Array.map _.Name |> List.ofArray

                Expect.equal
                    linear
                    [
                        "scale_f32"
                        "table_echo"
                        "args_echo"
                        "groups_echo"
                        "bf16_unpack"
                        "rmsnorm_rows"
                        "rope_apply"
                        "silu_mul"
                        "add"
                        "copy_rows"
                        "argmax_rows"
                    ]
                    "linear kernels"

            testCase "typed records encode and decode"
            <| fun () ->
                let echo: Kernels.ArgsEchoArgs =
                    {
                        Y = TensorToken 3u
                        V2 = 2u
                        V3 = 3u
                        V4 = 4u
                        V5 = 5u
                        V6 = 6u
                        V7 = 7u
                        V8 = 8u
                        V9 = 9u
                        V10 = 10u
                        V11 = 11u
                        V12 = 12u
                        V13 = 13u
                        V14 = 14u
                        V15 = 15u
                        V16 = 16u
                        V17 = 17u
                        V18 = 18u
                        V19 = 19u
                        V20 = 20u
                        V21 = 21u
                        V22 = 22u
                        V23 = 23u
                        V24 = 24u
                        V25 = 25u
                        V26 = 26u
                        V27 = 27u
                        V28 = 28u
                        V29 = 29u
                        V30 = 30u
                        V31 = 0xFFFFFFFFu
                    }

                let b = Kernels.ArgsEcho.encode echo

                Expect.equal
                    (b.ToWords())
                    (Array.append [| 0u; 3u |] [| 2u .. 30u |]
                     |> fun a -> Array.append a [| 0xFFFFFFFFu |])
                    "words"

                Expect.equal (Kernels.ArgsEcho.decode b) echo "decode"

                let rope: Kernels.RopeApplyArgs =
                    {
                        X = TensorToken 1u
                        CosTable = TensorToken 2u
                        SinTable = TensorToken 3u
                        Y = TensorToken 1u
                        Rows = 7u * 16u
                        Heads = 16u
                        HeadDim = 128u
                        Pos0 = 5u
                    }

                Expect.equal (Kernels.RopeApply.decode (Kernels.RopeApply.encode rope)) rope "rope"
                Expect.equal (Kernels.RopeApply.grid rope) (Grid.create 112L 1L 1L) "grid"

            Props.testProperty "typed grids equal the catalog formula"
            <| fun (m: uint16) (n: uint32) (k: uint16) ->
                let a: Kernels.MatmulBf16wArgs =
                    {
                        X = TensorToken 1u
                        W = TensorToken 2u
                        Y = TensorToken 3u
                        M = uint32 m + 1u
                        N = n % 200000u + 1u
                        K = uint32 k + 1u
                    }

                let g = Kernels.MatmulBf16w.grid a
                g = Grid.create (Grid.ceilDiv (int64 a.N) 64L) (Grid.ceilDiv (int64 a.M) 64L) 1L

            Props.testProperty "grid formulas print and parse back to the same tree"
            <| fun (a: uint16) (b: uint16) (c: bool) ->
                let lit (v: uint16) = GridExpr.Lit(int64 v + 1L)

                let e =
                    if c then
                        GridExpr.CeilDiv(
                            GridExpr.Mul(GridExpr.Ref "rows", GridExpr.Add(lit a, GridExpr.Ref "cols")),
                            lit b
                        )
                    else
                        GridExpr.Add(lit a, GridExpr.Mul(GridExpr.Ref "n", GridExpr.Add(lit b, GridExpr.Lit 1L)))

                GridExpr.parse (GridExpr.print e) = Ok e

            testCase "grid formulas evaluate with overflow checks"
            <| fun () ->
                let e =
                    match GridExpr.parse "ceildiv(rows * cols, 256) + 1" with
                    | Ok e -> e
                    | Error err -> failtest err

                let v =
                    function
                    | "rows" -> 3L
                    | "cols" -> 1000L
                    | _ -> 0L

                Expect.equal (GridExpr.eval v e) 13L "value"
                raisesContract "overflows" (fun () -> GridExpr.eval (fun _ -> Int64.MaxValue) e)
                Expect.isError (GridExpr.parse "ceildiv(n)") "one argument"
                Expect.isError (GridExpr.parse "n - 1") "no minus"
                Expect.isError (GridExpr.parse "N") "lowercase names"
        ]
