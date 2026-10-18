module Lrm.Gpu.Core.Tests.BackendTests

open System
open Expecto
open Lrm.Core
open Lrm.Gpu.Core
open Lrm.Gpu.Core.Tests.Common
open Lrm.Shaders

/// KAT G1 (docs/contracts/gpu-backend.md): the d3d12 class of an RTX 4080 under driver 616.92 (UMD 32.0.16.1692),
/// with SHA-256("abc") standing in for the ShaderSetHash. Payload and id computed independently in Python.
let g1Payload =
    "05000000"
    + "6433643132" // "d3d12"
    + "0d000000"
    + "7063693a313064653a32373034" // "pci:10de:2704"
    + "0c000000"
    + "33322e302e31362e31363932" // "32.0.16.1692"
    + "01"
    + "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    + "07000000"
    + "77696e2d783634" // "win-x64"

let g1Id = "8a35b90b41224bbd538c3da9841e754973ae0358dff3bc2908d807cd0825034f"

/// A recording backend with the selftest layout: weights 64 KiB (slot 1), activations 256 KiB (slot 2), scratch
/// 64 KiB (slot 3), the embedded shader set loaded, and a table with
///   t1 x f32 [1024] (slot 2), t2 y f32 [1024] (slot 2 at 4096), t3 w bf16 [1024] ReadOnly (slot 1),
///   t4 x.shifted f32 [1000], a view of x 16 elements in, t5 o u32 [64] (slot 3), t6 r f32 [16] ReadOnly (slot 1).
type Fixture =
    {
        Backend: RecordingBackend
        Gpu: IGpuBackend
        Heap: IGpuHeap
        X: TensorToken
        Y: TensorToken
        W: TensorToken
        Shifted: TensorToken
        Out: TensorToken
        ReadOnly: TensorToken
    }

let fixture (options: DeviceOptions) =
    let b = new RecordingBackend(RecordingBackend.DefaultDevice, options)
    let gpu = b :> IGpuBackend
    let layout = smallLayout ()
    let heap = gpu.CreateHeap layout
    gpu.LoadKernels(Shaders.embedded ()) |> ignore
    let t = TensorTable layout

    let row slot offset dtype dims flags name =
        TensorRow.create slot offset dtype dims flags (TensorRow.debugIdOf name)

    let x = t.Add("x", row 2 0L DType.F32 [ 1024L ] TensorFlags.None "x")
    let y = t.Add("y", row 2 4096L DType.F32 [ 1024L ] TensorFlags.None "y")
    let w = t.Add("w", row 1 0L DType.BF16 [ 1024L ] TensorFlags.ReadOnly "w")
    let shifted = t.AddView("x.shifted", x, [ 1000L ], 16L)
    let o = t.Add("o", row 3 0L DType.U32 [ 64L ] TensorFlags.None "o")
    let r = t.Add("r", row 1 4096L DType.F32 [ 16L ] TensorFlags.ReadOnly "r")
    heap.WriteTable(t.Seal())

    {
        Backend = b
        Gpu = gpu
        Heap = heap
        X = x
        Y = y
        W = w
        Shifted = shifted
        Out = o
        ReadOnly = r
    }

let scale (x: TensorToken) (y: TensorToken) n : Kernels.ScaleF32Args = { X = x; Y = y; N = n; A = 2.0f }

let tests =
    testList
        "backend"
        [
            testCase "KAT G1: the GPU fill rule"
            <| fun () ->
                Expect.equal (GpuClass.deviceId 0x10deu 0x2704u) "pci:10de:2704" "DeviceId"
                let umd = (32L <<< 48) ||| (0L <<< 32) ||| (16L <<< 16) ||| 1692L
                Expect.equal umd 9007199255791260L "the LARGE_INTEGER"
                Expect.equal (GpuClass.d3d12Driver umd) "32.0.16.1692" "D3D12 driver"
                Expect.equal (GpuClass.vulkanDriver "616.92" 0x9a2e0000u) "616.92" "Vulkan driverInfo"
                Expect.equal (GpuClass.vulkanDriver " " 0x9a2e0000u) "0x9a2e0000" "fallback to driverVersion"
                Expect.equal (GpuClass.vulkanDriver "a\u0001b" 7u) "0x00000007" "non-printable driverInfo"
                Expect.isFalse ((GpuClass.arch ExecutionBackend.D3D12).Contains "+") "no ISA flags for a GPU"

                Expect.equal
                    (GpuClass.arch ExecutionBackend.D3D12Warp)
                    (ExecutionClass.hostArch ())
                    "WARP runs on the CPU, so its class keeps the ISA flags"

                let c =
                    {
                        Backend = ExecutionBackend.D3D12
                        DeviceId = GpuClass.deviceId 0x10deu 0x2704u
                        Driver = GpuClass.d3d12Driver umd
                        ShaderSetHash = Some(Hash.sha256Utf8 "abc")
                        Arch = "win-x64"
                    }

                Expect.equal (hex (Canonical.encode (fun w -> ExecutionClass.encode w c))) g1Payload "payload"
                Expect.equal (ExecutionClass.id c).Value.Hex g1Id "id"

            testCase "fill uses the ShaderSetHash of the backend's own target"
            <| fun () ->
                let set = Shaders.embedded ()

                let dev =
                    { RecordingBackend.DefaultDevice with
                        VendorId = 0x10deu
                        PciDeviceId = 0x2704u
                    }

                let d = GpuClass.fill dev set

                let v =
                    GpuClass.fill
                        { dev with
                            Backend = ExecutionBackend.Vulkan
                            Driver = "616.92"
                        }
                        set

                Expect.equal d.ShaderSetHash (Some(ShaderSet.setHash ShaderTarget.Dxil set)) "d3d12 runs DXIL"
                Expect.equal v.ShaderSetHash (Some(ShaderSet.setHash ShaderTarget.Spirv set)) "vulkan runs SPIR-V"
                Expect.equal d.DeviceId "pci:10de:2704" "device"
                Expect.isOk (ExecutionClass.validate d) "valid"

                Expect.throwsT<ArgumentException>
                    (fun () ->
                        GpuClass.fill
                            { dev with
                                Backend = ExecutionBackend.CpuDevice
                            }
                            set
                        |> ignore)
                    "CPU backends use ExecutionClass.forCpu"

            testCase "the recording backend's class is known after LoadKernels"
            <| fun () ->
                use b = new RecordingBackend()
                let g = b :> IGpuBackend
                raisesContract "after LoadKernels" (fun () -> g.ExecutionClass)
                g.LoadKernels(Shaders.embedded ()) |> ignore

                Expect.equal
                    g.ExecutionClass
                    (GpuClass.fill RecordingBackend.DefaultDevice (Shaders.embedded ()))
                    "class"

                raisesContract "once per backend" (fun () -> g.LoadKernels(Shaders.embedded ()))

            testCase "heap, table and kernels are created once"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                raisesContract "once per backend" (fun () -> f.Gpu.CreateHeap(smallLayout ()))

                raisesContract "written once" (fun () -> f.Heap.WriteTable((TensorTable f.Heap.Layout).Seal()))

                use b = new RecordingBackend()
                let g = b :> IGpuBackend
                let h = g.CreateHeap(smallLayout ())
                g.LoadKernels(Shaders.embedded ()) |> ignore
                let cl = g.Begin()

                raisesContract "WriteTable has not run" (fun () ->
                    Kernels.ScaleF32.dispatch cl (scale (TensorToken 1u) (TensorToken 2u) 4u))

                raisesContract "different layout" (fun () ->
                    h.WriteTable((TensorTable(HeapLayout.ofPages [ ArenaKind.Weights, 65536L ])).Seal()))

            testCase "a dispatch is checked, folded and recorded with word 0 written"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                let cl = f.Gpu.Begin()
                Kernels.ScaleF32.dispatch cl (scale f.X f.Y 1000u)
                cl.Barrier()
                Kernels.GroupsEcho.dispatch cl { Y = f.Out; Groups = 70000u }
                let s = f.Gpu.Submit cl
                let c = f.Gpu.Wait s
                Expect.equal c.Dispatches 2 "two dispatches"
                let ds = Recording.dispatches f.Backend
                Expect.equal (ds |> List.map _.Spec.Name) [ "scale_f32"; "groups_echo" ] "order"
                let d0 = ds[0]
                Expect.equal (d0.Physical.X, d0.Physical.Y, d0.Physical.Z) (4, 1, 1) "scale grid"
                Expect.equal d0.Args[0] 4u "lrm_groups_x"
                let d1 = ds[1]
                Expect.equal (d1.Physical.X, d1.Physical.Y, d1.Physical.Z) (35000, 2, 1) "folded"
                Expect.equal d1.Args[0] 35000u "lrm_groups_x of the folded grid"
                Expect.equal d1.Grid (Grid.create 70000L 1L 1L) "logical grid"

                Expect.equal
                    (f.Backend.Submissions.Head.Commands
                     |> List.map (function
                         | RecordedCommand.Dispatch _ -> "dispatch"
                         | RecordedCommand.Barrier -> "barrier"
                         | _ -> "other"))
                    [ "dispatch"; "barrier"; "dispatch" ]
                    "commands"

            testCase "dispatch rules raise the same contract errors everywhere"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                let cl = f.Gpu.Begin()
                let d (a: Kernels.ScaleF32Args) = Kernels.ScaleF32.dispatch cl a
                raisesContract "null token" (fun () -> d (scale TensorToken.Null f.Y 4u))
                raisesContract "not in the table" (fun () -> d (scale (TensorToken 99u) f.Y 4u))
                raisesContract "outside 0..16383" (fun () -> d (scale (TensorToken 70000u) f.Y 4u))
                raisesContract "the kernel accepts f32" (fun () -> d (scale f.W f.Y 4u))
                raisesContract "read only" (fun () -> d (scale f.X f.ReadOnly 4u))
                raisesContract "empty grid" (fun () -> d (scale f.X f.Y 0u))

                raisesContract "pass 0" (fun () ->
                    cl.Dispatch(Kernels.ScaleF32.Id, (Kernels.ScaleF32.encode (scale f.X f.Y 4u)).WithGroupsX 1u))

                raisesContract "past the last field" (fun () ->
                    cl.Dispatch(Kernels.ScaleF32.Id, (Kernels.ScaleF32.encode (scale f.X f.Y 4u)).With(5, 1u)))

                raisesContract "not in the catalog" (fun () -> cl.Dispatch(KernelId 99, ArgBlock.Zero))

                raisesContract "not in the loaded shader set" (fun () ->
                    Kernels.Add.dispatch cl { A = f.X; B = f.X; Y = f.Y; N = 4u })

            testCase "aliasing is allowed only where the catalog declares it, over the same bytes"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                let cl = f.Gpu.Begin()
                Kernels.ScaleF32.dispatch cl (scale f.X f.X 4u)

                raisesContract "exactly the same bytes" (fun () ->
                    Kernels.ScaleF32.dispatch cl (scale f.X f.Shifted 4u))

                Kernels.TableEcho.dispatch cl { Src = TensorToken.Null; Y = f.Out }
                Kernels.TableEcho.dispatch cl { Src = f.X; Y = f.Out }

                raisesContract "not in the table" (fun () ->
                    Kernels.TableEcho.dispatch cl { Src = TensorToken 77u; Y = f.Out })

                Expect.equal (f.Gpu.Submit cl).Value 1UL "submitted"

            testCase "uploads, downloads and copies move bytes exactly"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                let data = Array.init 4096 byte
                f.Heap.Upload(2, 0L, ReadOnlySpan data)
                let cl = f.Gpu.Begin()

                cl.Copy(
                    { Slot = 2; Offset = 0L; Bytes = 4096L },
                    {
                        Slot = 3
                        Offset = 1024L
                        Bytes = 4096L
                    }
                )

                f.Gpu.Wait(f.Gpu.Submit cl) |> ignore
                let back = Array.zeroCreate<byte> 4096
                f.Heap.Download(3, 1024L, Span back)
                Expect.equal back data "copied"
                let table = Array.zeroCreate<byte> 128
                f.Heap.Download(0, 0L, Span table)
                Expect.equal table[64..127] ((f.Heap.Table.Value.Row f.X).ToBytes()) "slot 0 reads the table"
                use stream = new IO.MemoryStream(data)
                f.Heap.UploadFrom(1, 8192L, stream, 4096L)
                let page1 = Recording.pageBytes f.Backend 1
                Expect.equal page1[8192 .. 8192 + 4095] data "UploadFrom"
                let untouched = Array.create 16 0xFFuy
                let fresh = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                fresh.Heap.Download(3, 0L, Span untouched)
                Expect.equal untouched (Array.zeroCreate 16) "pages start zeroed"

            testCase "transfer and copy rules"
            <| fun () ->
                let f = fixture (DeviceOptions.For ExecutionBackend.D3D12)
                let one = [| 1uy; 2uy; 3uy; 4uy |]
                raisesContract "WriteTable" (fun () -> f.Heap.Upload(0, 0L, ReadOnlySpan one))
                raisesContract "holds no page" (fun () -> f.Heap.Upload(9, 0L, ReadOnlySpan one))
                raisesContract "multiples of 4" (fun () -> f.Heap.Upload(2, 2L, ReadOnlySpan one))
                raisesContract "past the" (fun () -> f.Heap.Upload(3, 65536L, ReadOnlySpan one))
                let cl = f.Gpu.Begin()

                let r slot offset bytes : BufferRange =
                    {
                        Slot = slot
                        Offset = offset
                        Bytes = bytes
                    }

                raisesContract "not a page slot" (fun () -> cl.Copy(r 0 0L 64L, r 2 0L 64L))
                raisesContract "overlap" (fun () -> cl.Copy(r 2 0L 64L, r 2 32L 64L))
                raisesContract "source has" (fun () -> cl.Copy(r 2 0L 64L, r 3 0L 128L))
                raisesContract "past the" (fun () -> cl.Copy(r 3 65532L 8L, r 2 0L 8L))
                raisesContract "multiples of 4" (fun () -> cl.Copy(r 2 1L 4L, r 3 0L 4L))

            testCase "command lists are single use and timers nest"
            <| fun () ->
                let f =
                    fixture
                        { DeviceOptions.For ExecutionBackend.D3D12 with
                            TimeDispatches = true
                        }

                let cl = f.Gpu.Begin()
                let outer = cl.BeginTimer("gpu.layer", [ Attr.int TelemetryKeys.Layer 0 ])
                let inner = cl.BeginTimer("gpu.attention", [])
                raisesContract "most recently opened" (fun () -> cl.EndTimer outer)
                Kernels.ScaleF32.dispatch cl (scale f.X f.Y 16u)
                cl.EndTimer inner
                raisesContract "still open" (fun () -> f.Gpu.Submit cl)
                cl.EndTimer outer
                raisesContract "valid telemetry name" (fun () -> cl.BeginTimer("Bad Name", []))
                let s = f.Gpu.Submit cl
                raisesContract "already submitted" (fun () -> cl.Barrier())
                raisesContract "already submitted" (fun () -> f.Gpu.Submit cl)
                let c = f.Gpu.Wait s

                Expect.equal
                    (c.Timers |> List.map (fun t -> t.Name, t.GpuStartNs, t.DurationNs))
                    [
                        "gpu.layer", 1000L, 4000L
                        "gpu.attention", 2000L, 2000L
                        "gpu.dispatch", 3000L, 1000L
                    ]
                    "synthetic, deterministic timings"

                raisesContract "not made by this backend" (fun () -> f.Gpu.Wait(Submission 9UL))
                use other = new RecordingBackend()
                raisesContract "this backend's Begin" (fun () -> (other :> IGpuBackend).Submit(f.Gpu.Begin()))

            testCase "the backend reports through the telemetry facade with reserved names"
            <| fun () ->
                let records =
                    withSink (fun () ->
                        let f =
                            fixture
                                { DeviceOptions.For ExecutionBackend.D3D12 with
                                    TimeDispatches = true
                                }

                        f.Heap.Upload(2, 0L, ReadOnlySpan(Array.zeroCreate 256))
                        let cl = f.Gpu.Begin()
                        Kernels.ScaleF32.dispatch cl (scale f.X f.Y 16u)
                        f.Gpu.Wait(f.Gpu.Submit cl) |> ignore)

                let arenas =
                    records
                    |> List.filter (fun r -> r.Name = TelemetryNames.MemArena)
                    |> List.map (fun r -> (r.Attrs |> List.find (fun a -> a.Key = TelemetryKeys.Arena)).Value, r.Value)

                Expect.equal
                    arenas
                    [
                        AttrValue.Str "table", 1048576.0
                        AttrValue.Str "weights", 65536.0
                        AttrValue.Str "activations", 262144.0
                        AttrValue.Str "scratch", 65536.0
                    ]
                    "mem.arena per arena"

                let names = records |> List.map _.Name |> Set.ofList

                for n in
                    [
                        GpuTelemetry.HeapSlots
                        GpuTelemetry.BytesMoved
                        GpuTelemetry.Dispatches
                        GpuTelemetry.Dispatch
                        GpuTelemetry.Wait
                    ] do
                    Expect.contains names n (sprintf "emits %s" n)

                Expect.isTrue (records |> List.forall (fun r -> TelemetryNames.isReserved r.Name)) "reserved names only"

            testCase "the registry enumerates providers and creates accepted devices"
            <| fun () ->
                let reg = BackendRegistry()

                let good =
                    { RecordingBackend.DefaultDevice with
                        Name = "good"
                    }

                let rejected =
                    { RecordingBackend.DefaultDevice with
                        Index = 1
                        Name = "old"
                    }

                reg.Register(
                    { new IGpuBackendProvider with
                        member _.Backend = ExecutionBackend.D3D12

                        member _.Enumerate() =
                            [
                                {
                                    Info = rejected
                                    Rejections = [ Requirements.D3D12ShaderModel66 ]
                                }
                                { Info = good; Rejections = [] }
                            ]

                        member _.Create(d, o) =
                            new RecordingBackend(d, o) :> IGpuBackend
                    }
                )

                Expect.equal reg.Backends [ ExecutionBackend.D3D12 ] "registered"
                Expect.equal (reg.Enumerate() |> List.map _.Accepted) [ false; true ] "candidates"

                match reg.TryCreate(DeviceOptions.For ExecutionBackend.D3D12) with
                | Ok b ->
                    Expect.equal b.Info.Name "good" "the first accepted device"
                    b.Dispose()
                | Error e -> failtest e

                match
                    reg.TryCreate
                        { DeviceOptions.For ExecutionBackend.D3D12 with
                            Index = Some 1
                        }
                with
                | Ok _ -> failtest "a rejected device was created"
                | Error e -> Expect.stringContains e Requirements.D3D12ShaderModel66 "reason"

                match reg.TryCreate(DeviceOptions.For ExecutionBackend.Vulkan) with
                | Ok _ -> failtest "no vulkan provider"
                | Error e -> Expect.stringContains e "no provider" "reason"

                reg.Register(
                    Recording.provider
                        { good with
                            Backend = ExecutionBackend.Vulkan
                        }
                )

                Expect.equal reg.Backends [ ExecutionBackend.Vulkan; ExecutionBackend.D3D12 ] "ExecutionBackend order"
        ]
