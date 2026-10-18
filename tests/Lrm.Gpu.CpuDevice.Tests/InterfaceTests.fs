/// The GpuBackendInterface rules on CpuDevice: once-only calls, the class, the kernel set, submissions, timers,
/// transfers, copies, disposal, telemetry and the provider. Every broken rule raises GpuContractException, as on
/// RecordingBackend, D3D12 and Vulkan.
module Lrm.Gpu.CpuDevice.Tests.InterfaceTests

open System
open System.IO
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.CpuDevice
open Lrm.Gpu.CpuDevice.Tests.Common
open Lrm.Shaders

let private small () =
    fixture [ t "x" DType.F32 1024L; t "y" DType.F32 1024L; t "o" DType.U32 64L ]

let private fresh () =
    CpuDevice.create CpuDeviceOptions.Default (DeviceOptions.For ExecutionBackend.CpuDevice)

let private oneLayout () =
    HeapLayout.ofPages [ ArenaKind.Activations, 65536L ]

let tests =
    testList
        "interface"
        [
            testCase "the class is ExecutionClass.forCpu cpu-device, known before LoadKernels"
            <| fun () ->
                use g = fresh ()
                let c = g.ExecutionClass
                Expect.equal c (ExecutionClass.forCpu ExecutionBackend.CpuDevice) "forCpu"
                Expect.isNone c.ShaderSetHash "a CPU class has no ShaderSetHash"
                Expect.isOk (ExecutionClass.validate c) "valid"

            testCase "an arithmetic model gets its own class (+ftz, +nan-canonical), so no bit claim mixes them"
            <| fun () ->
                let classOf cpu =
                    use g = CpuDevice.create cpu (DeviceOptions.For ExecutionBackend.CpuDevice)
                    g.ExecutionClass

                let plain = classOf CpuDeviceOptions.Default

                let ftz =
                    classOf
                        { CpuDeviceOptions.Default with
                            Fp32Denormals = Fp32DenormalMode.Flush
                        }

                let nan =
                    classOf
                        { CpuDeviceOptions.Default with
                            CanonicalNaN = true
                        }

                Expect.stringEnds ftz.Arch "+ftz" "flush"
                Expect.stringEnds nan.Arch "+nan-canonical" "canonical NaN"
                Expect.notEqual (ExecutionClass.id ftz) (ExecutionClass.id plain) "different ids"
                Expect.notEqual (ExecutionClass.id nan) (ExecutionClass.id plain) "different ids"

            testCase "a mixed model, zero threads and another backend's device are refused"
            <| fun () ->
                Expect.throwsT<ArgumentException>
                    (fun () ->
                        CpuDevice.create
                            { CpuDeviceOptions.Default with
                                Fp32Denormals = Fp32DenormalMode.Mixed
                            }
                            (DeviceOptions.For ExecutionBackend.CpuDevice)
                        |> ignore)
                    "mixed"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        CpuDevice.create
                            { CpuDeviceOptions.Default with
                                Threads = 0
                            }
                            (DeviceOptions.For ExecutionBackend.CpuDevice)
                        |> ignore)
                    "threads"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        new CpuDeviceBackend(
                            { CpuDeviceBackend.DefaultDevice with
                                Backend = ExecutionBackend.D3D12
                            },
                            DeviceOptions.For ExecutionBackend.D3D12,
                            CpuDeviceOptions.Default
                        )
                        |> ignore)
                    "not cpu-device"

            testCase "CreateHeap and LoadKernels run once, WriteTable once and only with its layout"
            <| fun () ->
                use g = fresh ()
                let h = g.CreateHeap(oneLayout ())
                raisesContract "CreateHeap runs once" (fun () -> g.CreateHeap(oneLayout ()))
                g.LoadKernels(Shaders.embedded ()) |> ignore
                raisesContract "LoadKernels runs once" (fun () -> g.LoadKernels(Shaders.embedded ()))
                let other = TensorTable(HeapLayout.ofPages [ ArenaKind.Scratch, 65536L ])
                raisesContract "different layout" (fun () -> h.WriteTable(other.Seal()))
                h.WriteTable((TensorTable(h.Layout)).Seal())
                Expect.isSome h.Table "the table is kept"
                raisesContract "written once" (fun () -> h.WriteTable((TensorTable(h.Layout)).Seal()))

            testCase "a dispatch needs the heap, the kernels and the table"
            <| fun () ->
                let args =
                    Kernels.ScaleF32.encode (scaleArgs (TensorToken 1u) (TensorToken 2u) 1u 1.0f)

                use g = fresh ()
                raisesContract "CreateHeap has not run" (fun () -> g.Begin().Dispatch(Kernels.ScaleF32.Id, args))
                let h = g.CreateHeap(oneLayout ())
                raisesContract "LoadKernels has not run" (fun () -> g.Begin().Dispatch(Kernels.ScaleF32.Id, args))
                g.LoadKernels(Shaders.embedded ()) |> ignore
                raisesContract "WriteTable has not run" (fun () -> g.Begin().Dispatch(Kernels.ScaleF32.Id, args))
                ignore h

            testCase "LoadKernels refuses a shader set built for another catalog"
            <| fun () ->
                use g = fresh ()

                let set =
                    { Shaders.embedded () with
                        CatalogHash = Hash.sha256Utf8 "another catalog"
                    }

                raisesContract "built for catalog" (fun () -> g.LoadKernels set)

            testCase "every catalog kernel has an emulator; the kernel set's hash names the emulators and the model"
            <| fun () ->
                let setOf cpu =
                    use g = CpuDevice.create cpu (DeviceOptions.For ExecutionBackend.CpuDevice)
                    g.LoadKernels(Shaders.embedded ())

                let k = setOf CpuDeviceOptions.Default

                for i in 0 .. Kernels.all.Length - 1 do
                    Expect.isTrue (k.Contains(KernelId i)) Kernels.all[i].Name

                Expect.isFalse (k.Contains(KernelId Kernels.all.Length)) "past the catalog"
                Expect.isFalse (k.Contains(KernelId -1)) "negative"
                Expect.equal (setOf CpuDeviceOptions.Default).ShaderSetHash k.ShaderSetHash "stable"

                Expect.notEqual
                    (setOf
                        { CpuDeviceOptions.Default with
                            CanonicalNaN = true
                        })
                        .ShaderSetHash
                    k.ShaderSetHash
                    "the model changes the hash"

            testCase "MirrorShaderSet: a kernel without a blob is missing, and dispatching it raises"
            <| fun () ->
                let full = Shaders.embedded ()

                let set =
                    { full with
                        Blobs = full.Blobs |> List.filter (fun b -> b.Kernel <> "scale_f32")
                        Missing = "scale_f32" :: full.Missing
                    }

                use g =
                    CpuDevice.create
                        { CpuDeviceOptions.Default with
                            MirrorShaderSet = true
                        }
                        (DeviceOptions.For ExecutionBackend.CpuDevice)

                let h = g.CreateHeap(oneLayout ())
                let ks = g.LoadKernels set

                for i in 0 .. Kernels.all.Length - 1 do
                    let name = Kernels.all[i].Name
                    Expect.equal (ks.Contains(KernelId i)) (not (List.contains name set.Missing)) name

                let table = TensorTable(h.Layout)
                let x = table.Add("x", TensorRow.create 1 0L DType.F32 [ 64L ] TensorFlags.None 0u)

                let y =
                    table.Add("y", TensorRow.create 1 256L DType.F32 [ 64L ] TensorFlags.None 0u)

                h.WriteTable(table.Seal())

                raisesContract "not in the loaded shader set" (fun () ->
                    Kernels.ScaleF32.dispatch (g.Begin()) (scaleArgs x y 64u 1.0f))

            testCase "submissions count from 1; a list is used once and only on its backend; Wait repeats"
            <| fun () ->
                let f = small ()
                let s1 = f.Gpu.Submit(f.Gpu.Begin())
                let c2 = f.Gpu.Begin()
                let s2 = f.Gpu.Submit c2
                Expect.equal (s1.Value, s2.Value) (1UL, 2UL) "fence values"
                raisesContract "already submitted" (fun () -> f.Gpu.Submit c2)
                raisesContract "already submitted" (fun () -> c2.Barrier())
                let other = small ()
                raisesContract "this backend's Begin" (fun () -> f.Gpu.Submit(other.Gpu.Begin()))
                raisesContract "was not made by this backend" (fun () -> f.Gpu.Wait(Submission 99UL))
                Expect.equal (f.Gpu.Wait s1) (f.Gpu.Wait s1) "the same completion"

            testCase "timers nest, close in reverse order, and must be closed at Submit"
            <| fun () ->
                let f = small ()
                let c = f.Gpu.Begin()
                let a = c.BeginTimer("gpu.outer", [])
                let b = c.BeginTimer("gpu.inner", [ Attr.int "layer" 3 ])
                raisesContract "most recently opened" (fun () -> c.EndTimer a)
                c.EndTimer b
                c.EndTimer a
                raisesContract "not a valid telemetry name" (fun () -> c.BeginTimer("Bad Name", []))
                let done' = f.Gpu.Wait(f.Gpu.Submit c)
                Expect.equal (done'.Timers |> List.map _.Name) [ "gpu.outer"; "gpu.inner" ] "in opening order"
                let open' = f.Gpu.Begin()
                open'.BeginTimer("gpu.open", []) |> ignore
                raisesContract "still open at Submit" (fun () -> f.Gpu.Submit open')

            testCase "TimeDispatches reports each dispatch as gpu.dispatch with its kernel"
            <| fun () ->
                let f =
                    fixtureWith
                        CpuDeviceOptions.Default
                        { DeviceOptions.For ExecutionBackend.CpuDevice with
                            TimeDispatches = true
                        }
                        [ t "x" DType.F32 256L; t "y" DType.F32 256L ]

                let c =
                    f.Run(fun c ->
                        Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "y") 256u 2.0f)
                        c.Barrier()
                        Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "y") (f.Token "x") 256u 2.0f))

                Expect.equal c.Dispatches 2 "two dispatches"
                let ds = c.Timers |> List.filter (fun t -> t.Name = GpuTelemetry.Dispatch)
                Expect.equal ds.Length 2 "two timings"
                Expect.isTrue (ds |> List.forall (fun t -> t.DurationNs > 0L)) "positive durations"

                Expect.isTrue
                    (ds
                     |> List.forall (fun t -> t.Attrs = [ Attr.str TelemetryKeys.Kernel "scale_f32" ]))
                    "kernel attribute"

            testCase "transfers: page slots only for uploads, multiples of 4, inside the page"
            <| fun () ->
                let f = small ()

                raisesContract "slot 0 holds the tensor table" (fun () ->
                    f.Heap.Upload(0, 0L, ReadOnlySpan(Array.zeroCreate<byte> 4)))

                raisesContract "multiples of 4" (fun () -> f.Heap.Upload(1, 2L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
                raisesContract "multiples of 4" (fun () -> f.Heap.Upload(1, 0L, ReadOnlySpan(Array.zeroCreate<byte> 3)))
                raisesContract "outside 0..63" (fun () -> f.Heap.Upload(64, 0L, ReadOnlySpan(Array.zeroCreate<byte> 4)))
                let page = (f.Heap.Layout.Page 1).Bytes

                raisesContract "past the" (fun () ->
                    f.Heap.Upload(1, page - 4L, ReadOnlySpan(Array.zeroCreate<byte> 8)))

                raisesContract "past the" (fun () -> f.Heap.Download(1, page, Span(Array.zeroCreate<byte> 4)))
                raisesContract "holds no page" (fun () -> f.Heap.Download(9, 0L, Span(Array.zeroCreate<byte> 4)))

            testCase "Download of slot 0 returns the table image; unwritten memory reads as the uninitialized word"
            <| fun () ->
                let f = small ()
                let image = Array.zeroCreate<byte>(int Heap.TableBytes)
                f.Heap.Download(0, 0L, Span image)
                Expect.equal image (f.Table.Image.ToArray()) "the table image"
                Expect.equal (f.Words("x", 3)) [| 0x7FBADBADu; 0x7FBADBADu; 0x7FBADBADu |] "never written"

                let z =
                    fixtureWith
                        { CpuDeviceOptions.Default with
                            UninitializedWord = 0u
                        }
                        (DeviceOptions.For ExecutionBackend.CpuDevice)
                        [ t "x" DType.F32 16L ]

                Expect.equal (z.Words("x", 2)) [| 0u; 0u |] "configurable"

            testCase "UploadFrom streams in chunks and equals Upload; round trips are byte-exact"
            <| fun () ->
                let f = small ()
                let r = rng "upload-from"
                let data = Array.init 4096 (fun _ -> byte (r.NextU32()))
                use s = new MemoryStream(data)
                let row = f.Row "x"
                f.Heap.UploadFrom(row.Slot, int64 row.ByteOffset, s, 4096L)
                let back = Array.zeroCreate<byte> 4096
                f.Heap.Download(row.Slot, int64 row.ByteOffset, Span back)
                Expect.equal back data "round trip"

                raisesContract "multiples of 4" (fun () -> f.Heap.UploadFrom(row.Slot, 0L, new MemoryStream(data), 6L))

            testCase "Copy: page slots, equal sizes, multiples of 4, inside the pages, no overlap; it moves bytes"
            <| fun () ->
                let f = small ()
                let x = f.Row "x"
                let y = f.Row "y"
                f.UploadU32("x", Array.init 1024 uint32)

                let r (row: TensorRow) off bytes : BufferRange =
                    {
                        Slot = row.Slot
                        Offset = int64 row.ByteOffset + off
                        Bytes = bytes
                    }

                f.Run(fun c -> c.Copy(r x 0L 4096L, r y 0L 4096L)) |> ignore
                Expect.equal (f.Words("y", 1024)) (Array.init 1024 uint32) "copied"
                let c = f.Gpu.Begin()
                raisesContract "not a page slot" (fun () -> c.Copy({ Slot = 0; Offset = 0L; Bytes = 4L }, r y 0L 4L))
                raisesContract "source has 8 bytes" (fun () -> c.Copy(r x 0L 8L, r y 0L 4L))
                raisesContract "multiples of 4" (fun () -> c.Copy(r x 2L 4L, r y 0L 4L))
                raisesContract "overlap" (fun () -> c.Copy(r x 0L 64L, r x 32L 64L))

                raisesContract "past the" (fun () ->
                    c.Copy(
                        r x 0L 8L,
                        {
                            Slot = y.Slot
                            Offset = (f.Heap.Layout.Page y.Slot).Bytes - 4L
                            Bytes = 8L
                        }
                    ))

            testCase "after Dispose every call raises ObjectDisposedException"
            <| fun () ->
                let g = fresh ()
                g.Dispose()
                Expect.throwsT<ObjectDisposedException> (fun () -> g.Begin() |> ignore) "Begin"
                Expect.throwsT<ObjectDisposedException> (fun () -> g.ExecutionClass |> ignore) "class"
                g.Dispose()

            testCase "telemetry: mem.arena, gpu.heap.slots, gpu.bytes_moved, gpu.dispatches, gpu.wait, gpu.dispatch"
            <| fun () ->
                let records =
                    withSink (fun () ->
                        let f =
                            fixtureWith
                                CpuDeviceOptions.Default
                                { DeviceOptions.For ExecutionBackend.CpuDevice with
                                    TimeDispatches = true
                                }
                                [ t "x" DType.F32 64L; t "y" DType.F32 64L ]

                        f.UploadF32("x", Array.create 64 1.0f)

                        f.Run(fun c -> Kernels.ScaleF32.dispatch c (scaleArgs (f.Token "x") (f.Token "y") 64u 2.0f))
                        |> ignore

                        f.Words("y", 64) |> ignore)

                let named n =
                    records |> List.filter (fun r -> r.Name = n)

                let backendIs (r: TelemetryRecord) =
                    r.Attrs |> List.contains (Attr.str TelemetryKeys.Backend "cpu-device")

                Expect.isTrue
                    (named "mem.arena"
                     |> List.exists (fun r -> r.Attrs |> List.contains (Attr.str TelemetryKeys.Arena "table")))
                    "table arena"

                Expect.isTrue
                    (named "mem.arena"
                     |> List.exists (fun r -> r.Attrs |> List.contains (Attr.str TelemetryKeys.Arena "activations")))
                    "activations arena"

                Expect.equal (named "gpu.heap.slots" |> List.map _.Value) [ 2.0 ] "table and one page"

                for direction in [ "upload"; "download" ] do
                    Expect.isTrue
                        (named "gpu.bytes_moved"
                         |> List.exists (fun r -> r.Attrs |> List.contains (Attr.str "direction" direction)))
                        direction

                Expect.equal (named "gpu.dispatches" |> List.map _.Value) [ 1.0 ] "one submission, one dispatch"
                Expect.equal (named "gpu.wait" |> List.length) 1 "one wait"
                let d = named "gpu.dispatch"
                Expect.equal d.Length 1 "one dispatch interval"
                Expect.equal d.Head.Kind TelemetryKind.GpuInterval "a GpuInterval"

                Expect.isTrue
                    (records
                     |> List.filter (fun r -> r.Name.StartsWith "gpu.")
                     |> List.forall backendIs)
                    "backend cpu-device"

            testCase "the provider enumerates one accepted CPU device and creates a CpuDevice"
            <| fun () ->
                let reg = BackendRegistry()
                reg.Register CpuDevice.provider
                let all = reg.Enumerate()
                Expect.equal all.Length 1 "one device"
                Expect.isTrue all.Head.Accepted "accepted"
                Expect.equal all.Head.Info.Backend ExecutionBackend.CpuDevice "cpu-device"

                match reg.TryCreate(DeviceOptions.For ExecutionBackend.CpuDevice) with
                | Ok b ->
                    Expect.isTrue (b :? CpuDeviceBackend) "a CpuDeviceBackend"
                    b.Dispose()
                | Error e -> failtest e

            testCase "the bits do not depend on the emulation thread count"
            <| fun () ->
                let run threads =
                    let f =
                        fixtureWith
                            { CpuDeviceOptions.Default with
                                Threads = threads
                            }
                            (DeviceOptions.For ExecutionBackend.CpuDevice)
                            [
                                t "x" DType.F32 (130L * 96L)
                                t "w" DType.BF16 (200L * 96L)
                                t "y" DType.F32 (130L * 200L)
                            ]

                    let r = rng "threads"
                    f.UploadF32("x", uniform r (130 * 96) -1.0f 1.0f)
                    f.UploadBf16("w", uniformBf16 r (200 * 96) -1.0f 1.0f)

                    f.Run(fun c ->
                        Kernels.MatmulBf16w.dispatch
                            c
                            {
                                X = f.Token "x"
                                W = f.Token "w"
                                Y = f.Token "y"
                                M = 130u
                                N = 200u
                                K = 96u
                            })
                    |> ignore

                    f.Words("y", 130 * 200)

                Expect.equal (run 1) (run 8) "1 and 8 threads"
        ]
