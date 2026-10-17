/// Execution-class facts that GPU tests report, so tools/test.ps1 summaries show device, driver and ShaderSetHash.
module Lrm.Testing.Devices

open System.Collections.Concurrent

type DeviceRecord =
    {
        Backend: string
        Device: string
        Driver: string
        ShaderSetHash: string option
    }

let private records = ConcurrentQueue<DeviceRecord>()

/// Records the device a test ran on. Duplicates are collapsed in the summary.
let record (backend: Backend) (device: string) (driver: string) (shaderSetHash: string option) =
    records.Enqueue
        {
            Backend = Backend.name backend
            Device = device
            Driver = driver
            ShaderSetHash = shaderSetHash
        }

/// Everything recorded so far, without duplicates.
let all () =
    records.ToArray() |> Array.distinct |> List.ofArray
