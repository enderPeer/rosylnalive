module Lrm.Gpu.Core.Tests.HeapLayoutTests

open Expecto
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common

let MiB = 1048576L

let tests =
    testList
        "heap-layout"
        [
            testCase "arena kinds keep their contract order and codes"
            <| fun () ->
                let names = ArenaKind.all |> List.map ArenaKind.name

                Expect.equal
                    names
                    [
                        "weights"
                        "rope"
                        "kv_cache"
                        "activations"
                        "scratch"
                        "logits"
                        "taps"
                        "master"
                        "grad"
                        "adam_m"
                        "adam_v"
                        "roslyn_in"
                        "roslyn_out"
                    ]
                    "names"

                Expect.equal (ArenaKind.all |> List.map ArenaKind.code) [ 1..13 ] "codes 1..13 in order"

                Expect.equal
                    (ArenaKind.all |> List.filter ArenaKind.isReserved)
                    [
                        ArenaKind.Master
                        ArenaKind.Grad
                        ArenaKind.AdamM
                        ArenaKind.AdamV
                        ArenaKind.RoslynIn
                        ArenaKind.RoslynOut
                    ]
                    "reserved kinds"

                for k in ArenaKind.all do
                    Expect.equal (ArenaKind.tryOfCode (ArenaKind.code k)) (Some k) "code round trip"
                    Expect.equal (ArenaKind.tryParse (ArenaKind.name k)) (Some k) "name round trip"

                    Expect.isTrue
                        (Lrm.Core.TelemetryNames.isValid (ArenaKind.name k))
                        "arena names are valid telemetry attribute values"

            testCase "fixed numbers"
            <| fun () ->
                Expect.equal Heap.SlotCount 64 "slots"
                Expect.equal Heap.TableSlot 0 "table slot"
                Expect.equal Heap.TableRows 16384 "rows"
                Expect.equal Heap.RowBytes 64 "row bytes"
                Expect.equal Heap.TableBytes (16384L * 64L) "table bytes"
                Expect.equal Heap.MaxPageBytes (1L <<< 30) "1 GiB pages"
                Expect.equal Heap.PageGranularity 65536L "64 KiB granularity"

            testCase "Vulkan slot i sits at byte 16 i; D3D12 slot i is descriptor i"
            <| fun () ->
                for i in 0..63 do
                    Expect.equal (Heap.vulkanDescriptorOffset i) (16L * int64 i) "vulkan offset"
                    Expect.equal (Heap.d3d12DescriptorIndex i) i "d3d12 index"

                Expect.equal Heap.vulkanReservedRangeOffset 1024L "reserved range starts after the 64 descriptors"
                Expect.equal (Heap.vulkanReservedRangeOffset % 64L) 0L "64-byte aligned"
                Expect.equal (Heap.vulkanHeapBytes 4096L) 5120L "heap bytes"
                raisesContract "outside" (fun () -> Heap.vulkanDescriptorOffset 64)
                raisesContract "outside" (fun () -> Heap.d3d12DescriptorIndex -1)

            testCase "ofPages assigns slots 1..N in order with per-arena page indices"
            <| fun () ->
                let l =
                    HeapLayout.ofPages
                        [
                            ArenaKind.Weights, 1024L * MiB
                            ArenaKind.Weights, 512L * MiB
                            ArenaKind.Rope, MiB
                            ArenaKind.Activations, 64L * MiB
                        ]

                Expect.equal
                    (l.Pages |> List.map (fun p -> p.Slot, p.Arena, p.Index))
                    [
                        1, ArenaKind.Weights, 0
                        2, ArenaKind.Weights, 1
                        3, ArenaKind.Rope, 0
                        4, ArenaKind.Activations, 0
                    ]
                    "slots"

                Expect.equal (l.SlotOf(ArenaKind.Weights, 1)) 2 "SlotOf"
                Expect.equal (l.ArenaBytes ArenaKind.Weights) (1536L * MiB) "arena bytes"
                Expect.equal l.TotalBytes (Heap.TableBytes + 1601L * MiB) "total"
                Expect.isNone (l.TryPage 0) "slot 0 is the table, not a page"
                Expect.isNone (l.TryPage 5) "unused slot"
                raisesContract "holds no page" (fun () -> l.Page 5)

            testCase "ofPages rejects bad page lists"
            <| fun () ->
                let bad (pages: (ArenaKind * int64) list) (fragment: string) =
                    match HeapLayout.tryOfPages pages with
                    | Ok _ -> failtestf "expected a rejection containing '%s'" fragment
                    | Error e -> Expect.stringContains e fragment "reason"

                bad [ ArenaKind.Master, 65536L ] "reserved until contracts-v2"
                bad [ ArenaKind.RoslynOut, 65536L ] "reserved until contracts-v2"
                bad [ ArenaKind.Weights, 0L ] "outside 64 KiB..1 GiB"
                bad [ ArenaKind.Weights, Heap.MaxPageBytes + 65536L ] "outside 64 KiB..1 GiB"
                bad [ ArenaKind.Weights, 100000L ] "not a multiple of 64 KiB"
                bad [ ArenaKind.Scratch, 65536L; ArenaKind.Weights, 65536L ] "contract order"
                bad (List.replicate 64 (ArenaKind.Scratch, 65536L)) "do not fit"

                Expect.isOk
                    (HeapLayout.tryOfPages (List.replicate 63 (ArenaKind.Scratch, 65536L)))
                    "63 pages fill slots 1..63"

            testCase "the planner packs next-fit, 256-byte aligned, never straddling a page"
            <| fun () ->
                let p = HeapPlanner(131072L)
                let a = p.Allocate(ArenaKind.Activations, 1000L)
                let b = p.Allocate(ArenaKind.Activations, 4L)
                let c = p.Allocate(ArenaKind.Activations, 130000L)
                let d = p.Allocate(ArenaKind.Activations, 100L)
                let w = p.Allocate(ArenaKind.Weights, 70000L)
                Expect.equal (a.Page, a.Offset) (0, 0L) "first"
                Expect.equal (b.Page, b.Offset) (0, 1024L) "aligned to 256"
                Expect.equal (c.Page, c.Offset) (1, 0L) "does not fit: a new page"
                Expect.equal (d.Page, d.Offset) (1, 130048L) "next fit on the last page"
                Expect.equal (w.Page, w.Offset) (0, 0L) "own arena"
                let l = p.Build()

                Expect.equal
                    (l.Pages |> List.map (fun pg -> pg.Slot, pg.Arena, pg.Index, pg.Bytes))
                    [
                        1, ArenaKind.Weights, 0, 131072L
                        2, ArenaKind.Activations, 0, 65536L
                        3, ArenaKind.Activations, 1, 131072L
                    ]
                    "weights first (contract order), pages rounded up to 64 KiB"

                raisesContract "after Build" (fun () -> p.Allocate(ArenaKind.Scratch, 4L))

            testCase "the planner refuses reserved arenas and oversized allocations"
            <| fun () ->
                let p = HeapPlanner()
                raisesContract "reserved until contracts-v2" (fun () -> p.Allocate(ArenaKind.Grad, 4L))
                raisesContract "one page" (fun () -> p.Allocate(ArenaKind.Weights, Heap.MaxPageBytes + 1L))
                raisesContract "one page" (fun () -> p.Allocate(ArenaKind.Weights, 0L))
                raisesContract "multiple of 64 KiB" (fun () -> HeapPlanner(100000L))

            testCase "the Qwen3-0.6B embedding (311 MB bf16) fits one 1 GiB page"
            <| fun () ->
                let p = HeapPlanner()
                let e = p.Allocate(ArenaKind.Weights, 151936L * 1024L * 2L)
                Expect.equal (e.Page, e.Offset) (0, 0L) "placed"
                let l = p.Build()
                Expect.equal (l.Page 1).Bytes (Heap.alignUp 65536L (151936L * 1024L * 2L)) "page size"
        ]
