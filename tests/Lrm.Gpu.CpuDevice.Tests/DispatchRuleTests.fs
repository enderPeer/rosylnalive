/// Every dispatch rule raises on CpuDevice: the shared rules of Dispatch.prepare, the kernel-specific minimum element
/// counts and argument rules of kernel-abi.md "Tensor access" (which only CpuDevice checks), integer overflow, and the
/// barrier rule between dependent dispatches.
module Lrm.Gpu.CpuDevice.Tests.DispatchRuleTests

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.CpuDevice.Tests.Common

/// A dispatch of one kernel with every data token holding exactly the elements the dispatch touches.
type Baseline =
    {
        Kernel: string
        Scalars: (string * uint32) list
        /// (field, dtype, elements); a meta field's tensor is named too.
        Tensors: (string * DType * int64) list
    }

let private f32 (x: float32) = BitConverter.SingleToUInt32Bits x

let baselines =
    [
        {
            Kernel = "scale_f32"
            Scalars = [ "n", 300u; "a", f32 2.0f ]
            Tensors = [ "x", DType.F32, 300L; "y", DType.F32, 300L ]
        }
        {
            Kernel = "table_echo"
            Scalars = []
            Tensors = [ "src", DType.F32, 3L; "y", DType.U32, 16L ]
        }
        {
            Kernel = "args_echo"
            Scalars = [ for i in 2..31 -> sprintf "v%d" i, uint32 i ]
            Tensors = [ "y", DType.U32, 32L ]
        }
        {
            Kernel = "groups_echo"
            Scalars = [ "groups", 5u ]
            Tensors = [ "y", DType.U32, 5L ]
        }
        {
            Kernel = "bf16_unpack"
            Scalars = [ "n", 301u ]
            Tensors = [ "x", DType.BF16, 301L; "y", DType.F32, 301L ]
        }
        {
            Kernel = "embed_gather_bf16"
            Scalars = [ "rows", 3u; "cols", 10u; "vocab", 4u ]
            Tensors = [ "ids", DType.U32, 3L; "table", DType.BF16, 40L; "y", DType.F32, 30L ]
        }
        {
            Kernel = "rmsnorm_rows"
            Scalars = [ "rows", 3u; "cols", 10u; "eps", f32 1e-6f ]
            Tensors = [ "x", DType.F32, 30L; "w", DType.BF16, 10L; "y", DType.F32, 30L ]
        }
        {
            Kernel = "rope_apply"
            Scalars = [ "rows", 6u; "heads", 2u; "head_dim", 4u; "pos0", 2u ]
            Tensors =
                [
                    "x", DType.F32, 24L
                    "cos_table", DType.F32, 10L
                    "sin_table", DType.F32, 10L
                    "y", DType.F32, 24L
                ]
        }
        {
            Kernel = "silu_mul"
            Scalars = [ "n", 50u ]
            Tensors = [ "gate", DType.F32, 50L; "up", DType.F32, 50L; "y", DType.F32, 50L ]
        }
        {
            Kernel = "add"
            Scalars = [ "n", 50u ]
            Tensors = [ "a", DType.F32, 50L; "b", DType.F32, 50L; "y", DType.F32, 50L ]
        }
        {
            Kernel = "copy_rows"
            Scalars =
                [
                    "rows", 3u
                    "cols", 4u
                    "src_row0", 1u
                    "dst_row0", 0u
                    "src_stride", 5u
                    "dst_stride", 6u
                ]
            Tensors = [ "src", DType.F32, 19L; "dst", DType.F32, 16L ]
        }
        {
            Kernel = "argmax_rows"
            Scalars = [ "rows", 3u; "cols", 7u ]
            Tensors = [ "x", DType.F32, 21L; "y", DType.U32, 3L ]
        }
        {
            Kernel = "matmul_bf16w"
            Scalars = [ "m", 3u; "n", 5u; "k", 4u ]
            Tensors = [ "x", DType.F32, 12L; "w", DType.BF16, 20L; "y", DType.F32, 15L ]
        }
        {
            Kernel = "attention_gqa_causal"
            Scalars =
                [
                    "rows", 2u
                    "q_row0", 1u
                    "past", 3u
                    "q_heads", 2u
                    "kv_heads", 1u
                    "head_dim", 4u
                    "scale", f32 0.5f
                ]
            Tensors =
                [
                    "q", DType.F32, 24L
                    "k", DType.F32, 20L
                    "v", DType.F32, 20L
                    "y", DType.F32, 24L
                ]
        }
    ]

/// The fixture and block of a baseline, with its tensors resized by `resize` and its scalars overridden by `scalars`.
let setup (b: Baseline) (resize: string -> DType -> int64 -> DType * int64) (scalars: (string * uint32) list) =
    let spec = Kernels.all[(Kernels.tryFind b.Kernel).Value.Value]

    let tensors =
        b.Tensors |> List.map (fun (f, d, n) -> let d', n' = resize f d n in f, d', n')

    let f = fixture [ for name, d, n in tensors -> t name d n ]
    let r = rng b.Kernel

    for name, d, n in tensors do
        match d with
        | DType.F32 -> f.UploadF32(name, uniform r (int n) -1.0f 1.0f)
        | DType.BF16 -> f.UploadBf16(name, uniformBf16 r (int n) -1.0f 1.0f)
        | _ -> f.UploadU32(name, Array.zeroCreate (int n))

    let value (name: string) =
        match scalars |> List.tryFind (fun (n, _) -> n = name) with
        | Some(_, v) -> v
        | None -> b.Scalars |> List.find (fun (n, _) -> n = name) |> snd

    let w = Array.zeroCreate<uint32> 32

    spec.Fields
    |> List.iteri (fun i fs ->
        w[i + 1] <-
            match fs.Kind with
            | FieldKind.Token _ -> (f.Token fs.Name).Value
            | _ -> value fs.Name)

    f, (Kernels.tryFind b.Kernel).Value, ArgBlock.OfWords w

let private same _ d n = d, n

let private dispatchOf (b: Baseline) resize scalars =
    let f, k, args = setup b resize scalars
    fun () -> f.Run(fun c -> c.Dispatch(k, args))

let private minimumCounts =
    testList
        "minimum element counts"
        [
            for b in baselines do
                testCase (
                    b.Kernel
                    + " runs with exactly the elements it touches, and with more (a capacity)"
                )
                <| fun () ->
                    dispatchOf b same [] () |> ignore
                    dispatchOf b (fun _ d n -> d, n + 7L) [] () |> ignore

                for field, _, n in b.Tensors do
                    let spec = Kernels.all[(Kernels.tryFind b.Kernel).Value.Value]

                    let isMeta =
                        spec.Fields
                        |> List.exists (fun fs ->
                            fs.Name = field
                            && (match fs.Kind with
                                | FieldKind.Token(Access.Meta, _, _) -> true
                                | _ -> false))

                    if n >= 2L && not isMeta then
                        testCase (sprintf "%s.%s one element short raises" b.Kernel field)
                        <| fun () ->
                            let run = dispatchOf b (fun f d c -> d, (if f = field then c - 1L else c)) []

                            raisesContract
                                (sprintf
                                    "%s.%s: tensor '%s' holds %d elements, but the dispatch touches %d"
                                    b.Kernel
                                    field
                                    field
                                    (n - 1L)
                                    n)
                                run
        ]

let private baseline (k: string) =
    baselines |> List.find (fun b -> b.Kernel = k)

let private argumentRules =
    let rule (name: string) (kernel: string) (scalars: (string * uint32) list) (fragment: string) =
        testCase name
        <| fun () -> raisesContract fragment (dispatchOf (baseline kernel) same scalars)

    testList
        "argument rules"
        [
            rule "rmsnorm_rows: cols >= 1" "rmsnorm_rows" [ "cols", 0u ] "cols >= 1 is required"
            rule "argmax_rows: cols >= 1" "argmax_rows" [ "cols", 0u ] "cols >= 1 is required"
            rule "rope_apply: heads >= 1" "rope_apply" [ "heads", 0u ] "heads >= 1 is required"
            rule "rope_apply: heads divides rows" "rope_apply" [ "heads", 4u ] "heads must divide rows"
            rule "rope_apply: head_dim even" "rope_apply" [ "head_dim", 3u ] "head_dim must be even"
            rule
                "rope_apply: head_dim at least 2"
                "rope_apply"
                [ "head_dim", 0u ]
                "head_dim must be even and at least 2"
            rule "copy_rows: destination rows never overlap" "copy_rows" [ "cols", 7u ] "cols <= dst_stride is required"
            rule "attention: kv_heads >= 1" "attention_gqa_causal" [ "kv_heads", 0u ] "kv_heads >= 1 is required"
            rule
                "attention: kv_heads divides q_heads"
                "attention_gqa_causal"
                [ "q_heads", 3u; "kv_heads", 2u ]
                "kv_heads must divide q_heads"
            rule "attention: head_dim >= 1" "attention_gqa_causal" [ "head_dim", 0u ] "head_dim >= 1 is required"

            testCase "copy_rows: src and dst have one dtype"
            <| fun () ->
                raisesContract
                    "src and dst must have the same dtype"
                    (dispatchOf (baseline "copy_rows") (fun f d n -> (if f = "dst" then DType.U32 else d), n) [])

            testCase "copy_rows: cols may exceed dst_stride for one row"
            <| fun () ->
                dispatchOf
                    (baseline "copy_rows")
                    (fun f d n -> d, (if f = "dst" then 7L else n + 3L))
                    [ "rows", 1u; "cols", 7u ]
                    ()
                |> ignore

            testCase "an element count that overflows int64 raises"
            <| fun () ->
                raisesContract
                    "overflows"
                    (dispatchOf
                        (baseline "copy_rows")
                        same
                        [
                            "rows", 2u
                            "src_row0", UInt32.MaxValue
                            "src_stride", UInt32.MaxValue
                            "cols", 1u
                        ])
        ]

/// The shared rules of Dispatch.prepare raise the same GpuContractException on CpuDevice.
let private sharedRules =
    let scaleFixture () =
        let f =
            fixture
                [
                    t "x" DType.F32 1024L
                    t "y" DType.F32 1024L
                    t "u" DType.U32 64L
                    readOnly (t "r" DType.F32 64L)
                ]

        f

    let scale (f: Fixture) (x: TensorToken) (y: TensorToken) (n: uint32) =
        fun () -> f.Run(fun c -> Kernels.ScaleF32.dispatch c (scaleArgs x y n 1.0f))

    testList
        "shared rules"
        [
            testCase "word 0 is lrm_groups_x and must be passed as 0"
            <| fun () ->
                let f = scaleFixture ()

                let a =
                    (Kernels.ScaleF32.encode (scaleArgs (f.Token "x") (f.Token "y") 4u 1.0f)).WithGroupsX 1u

                raisesContract "argument word 0" (fun () -> f.Run(fun c -> c.Dispatch(Kernels.ScaleF32.Id, a)))

            testCase "words past the last field must be 0"
            <| fun () ->
                let f = scaleFixture ()

                let a =
                    (Kernels.ScaleF32.encode (scaleArgs (f.Token "x") (f.Token "y") 4u 1.0f)).With(9, 1u)

                raisesContract "past the last field" (fun () -> f.Run(fun c -> c.Dispatch(Kernels.ScaleF32.Id, a)))

            testCase "a token past the table, the null token and an unassigned token"
            <| fun () ->
                let f = scaleFixture ()
                raisesContract "outside 0..16383" (scale f (TensorToken 16384u) (f.Token "y") 4u)
                raisesContract "the null token is not allowed" (scale f TensorToken.Null (f.Token "y") 4u)
                raisesContract "is not in the table" (scale f (TensorToken 200u) (f.Token "y") 4u)

            testCase "a dtype the field does not accept"
            <| fun () ->
                let f = scaleFixture ()
                raisesContract "the kernel accepts f32" (scale f (f.Token "u") (f.Token "y") 4u)

            testCase "a read-only tensor is never written"
            <| fun () ->
                let f = scaleFixture ()
                raisesContract "is read only" (scale f (f.Token "x") (f.Token "r") 4u)
                scale f (f.Token "r") (f.Token "y") 4u () |> ignore

            testCase "a non-contiguous tensor is refused"
            <| fun () ->
                let layout = HeapLayout.ofPages [ ArenaKind.Activations, 65536L ]

                let g =
                    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap layout
                g.LoadKernels(Lrm.Shaders.Shaders.embedded ()) |> ignore
                let table = TensorTable layout

                let x =
                    table.Add("x", TensorRow.strided 1 0L DType.F32 [ 4L; 4L ] [ 8L; 1L ] TensorFlags.None 0u)

                let y =
                    table.Add("y", TensorRow.create 1 4096L DType.F32 [ 16L ] TensorFlags.None 0u)

                h.WriteTable(table.Seal())
                let c = g.Begin()
                raisesContract "is not contiguous" (fun () -> Kernels.ScaleF32.dispatch c (scaleArgs x y 16u 1.0f))

            testCase "overlapping tokens alias only as the catalog declares, over the same bytes"
            <| fun () ->
                let layout = HeapLayout.ofPages [ ArenaKind.Activations, 65536L ]

                let g =
                    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap layout
                g.LoadKernels(Lrm.Shaders.Shaders.embedded ()) |> ignore
                let table = TensorTable layout
                let x = table.Add("x", TensorRow.create 1 0L DType.F32 [ 64L ] TensorFlags.None 0u)
                let shifted = table.AddView("x.shifted", x, [ 60L ], 4L)

                let w =
                    table.Add("w", TensorRow.create 1 1024L DType.BF16 [ 64L ] TensorFlags.None 0u)

                h.WriteTable(table.Seal())
                let c = g.Begin()

                raisesContract "must span exactly the same bytes" (fun () ->
                    Kernels.ScaleF32.dispatch c (scaleArgs shifted x 60u 1.0f))

                raisesContract "the catalog does not let them alias" (fun () ->
                    Kernels.MatmulBf16w.dispatch
                        c
                        {
                            X = x
                            W = w
                            Y = shifted
                            M = 1u
                            N = 1u
                            K = 1u
                        })

                // In place, as declared: fine.
                Kernels.ScaleF32.dispatch c (scaleArgs x x 64u 1.0f)

            testCase "an empty grid, y above 65535, and x above 65535 on a non-linear kernel"
            <| fun () ->
                let f = scaleFixture ()
                raisesContract "empty grid" (scale f (f.Token "x") (f.Token "y") 0u)
                let e = baseline "embed_gather_bf16"
                raisesContract "y and z must be at most 65535" (dispatchOf e same [ "rows", 65536u ])
                let m = baseline "matmul_bf16w"
                raisesContract "is not linear" (dispatchOf m same [ "n", 65536u * 64u + 1u ])

            testCase "x above 65535 on a linear kernel is folded, not refused"
            <| fun () ->
                let g = baseline "groups_echo"
                dispatchOf g (fun _ d _ -> d, 70000L) [ "groups", 70000u ] () |> ignore
        ]

/// The barrier rule (GpuBackendInterface, Barrier): every dependent dispatch is preceded by a barrier. CpuDevice
/// tracks the bytes each dispatch reads and writes since the last Barrier or Copy.
let private barriers =
    let f () =
        fixture [ t "x" DType.F32 512L; t "y" DType.F32 512L; t "z" DType.F32 512L ]

    let s (f: Fixture) (c: ICommandList) x y n =
        Kernels.ScaleF32.dispatch c (scaleArgs (f.Token x) (f.Token y) n 2.0f)

    testList
        "barriers"
        [
            testCase "read after write without a barrier raises; with one it runs"
            <| fun () ->
                let f = f ()

                raisesContract "scale_f32.x reads bytes that dispatch 0 (scale_f32.y) wrote" (fun () ->
                    f.Run(fun c ->
                        s f c "x" "y" 512u
                        s f c "y" "z" 512u))

                f.UploadF32("x", Array.create 512 1.0f)

                f.Run(fun c ->
                    s f c "x" "y" 512u
                    c.Barrier()
                    s f c "y" "z" 512u)
                |> ignore

                Expect.equal (f.F32("z", 512)) (Array.create 512 4.0f) "1 * 2 * 2"

            testCase "write after read and write after write without a barrier raise"
            <| fun () ->
                let f = f ()

                raisesContract "writes bytes that dispatch 0 (scale_f32.x) read" (fun () ->
                    f.Run(fun c ->
                        s f c "x" "y" 512u
                        s f c "z" "x" 512u))

                raisesContract "writes bytes that dispatch 0 (scale_f32.y) wrote" (fun () ->
                    f.Run(fun c ->
                        s f c "x" "y" 512u
                        s f c "z" "y" 512u))

            testCase "independent dispatches need no barrier; reads of one input may overlap"
            <| fun () ->
                let f = f ()

                f.Run(fun c ->
                    s f c "x" "y" 512u
                    s f c "x" "z" 512u)
                |> ignore

            testCase "disjoint writes into one tensor need no barrier"
            <| fun () ->
                let layout = HeapLayout.ofPages [ ArenaKind.Activations, 65536L ]

                let g =
                    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap layout
                g.LoadKernels(Lrm.Shaders.Shaders.embedded ()) |> ignore
                let table = TensorTable layout
                let x = table.Add("x", TensorRow.create 1 0L DType.F32 [ 64L ] TensorFlags.None 0u)

                let y =
                    table.Add("y", TensorRow.create 1 4096L DType.F32 [ 64L ] TensorFlags.None 0u)

                let lo = table.AddView("y.lo", y, [ 32L ], 0L)
                let hi = table.AddView("y.hi", y, [ 32L ], 32L)
                h.WriteTable(table.Seal())
                let c = g.Begin()
                Kernels.ScaleF32.dispatch c (scaleArgs x lo 32u 1.0f)
                Kernels.ScaleF32.dispatch c (scaleArgs x hi 32u 1.0f)
                raisesContract "wrote" (fun () -> Kernels.ScaleF32.dispatch c (scaleArgs x y 64u 1.0f))

            testCase "strided rows: copy_rows into interleaved rows of one tensor needs no barrier"
            <| fun () ->
                let layout = HeapLayout.ofPages [ ArenaKind.Activations, 65536L ]

                let g =
                    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap layout
                g.LoadKernels(Lrm.Shaders.Shaders.embedded ()) |> ignore
                let table = TensorTable layout

                let src =
                    table.Add("src", TensorRow.create 1 0L DType.F32 [ 16L ] TensorFlags.None 0u)

                let dst =
                    table.Add("dst", TensorRow.create 1 4096L DType.F32 [ 64L ] TensorFlags.None 0u)

                let odd = table.AddView("dst.odd", dst, [ 60L ], 4L)
                h.WriteTable(table.Seal())

                let copy (into: TensorToken) : Kernels.CopyRowsArgs =
                    {
                        Src = src
                        Dst = into
                        Rows = 4u
                        Cols = 4u
                        SrcRow0 = 0u
                        DstRow0 = 0u
                        SrcStride = 4u
                        DstStride = 8u
                    }

                let c = g.Begin()
                // Elements 0..3, 8..11, 16..19, 24..27, then 4..7, 12..15, 20..23, 28..31: interleaved, disjoint.
                Kernels.CopyRows.dispatch c (copy dst)
                Kernels.CopyRows.dispatch c (copy odd)
                // The same rows again overlap the first dispatch's.
                raisesContract "wrote" (fun () -> Kernels.CopyRows.dispatch c (copy dst))

            testCase "Copy is a barrier on both sides"
            <| fun () ->
                let f = f ()
                let x = f.Row "y"
                let z = f.Row "z"

                f.Run(fun c ->
                    s f c "x" "y" 512u

                    c.Copy(
                        {
                            Slot = x.Slot
                            Offset = int64 x.ByteOffset
                            Bytes = 2048L
                        },
                        {
                            Slot = z.Slot
                            Offset = int64 z.ByteOffset
                            Bytes = 2048L
                        }
                    )

                    s f c "y" "x" 512u)
                |> ignore

            testCase "command lists are independent: the boundary is a barrier"
            <| fun () ->
                let f = f ()
                f.Run(fun c -> s f c "x" "y" 512u) |> ignore
                f.Run(fun c -> s f c "y" "z" 512u) |> ignore

            testCase "CheckBarriers = false allows what the device would race on"
            <| fun () ->
                let f =
                    fixtureWith
                        { CpuDeviceOptions.Default with
                            CheckBarriers = false
                        }
                        (DeviceOptions.For ExecutionBackend.CpuDevice)
                        [ t "x" DType.F32 64L; t "y" DType.F32 64L ]

                f.Run(fun c ->
                    s f c "x" "y" 64u
                    s f c "y" "x" 64u)
                |> ignore
        ]

let tests =
    testList "dispatch rules" [ sharedRules; minimumCounts; argumentRules; barriers ]
