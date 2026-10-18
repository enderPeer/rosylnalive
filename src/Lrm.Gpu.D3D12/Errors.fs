namespace Lrm.Gpu.D3D12

open System
open Lrm.Gpu.Core
open Lrm.Gpu.D3D12.Interop

/// The device stopped working: it was removed, hung or reset, or a submission did not finish in time. Device loss is
/// not a contract violation (GpuBackendInterface, "Errors"), so it is its own exception. It carries the call that saw
/// it, ID3D12Device::GetDeviceRemovedReason, and DRED's breadcrumbs and page-fault report (Dred.format).
type D3D12DeviceLostException(call: string, removedReason: int, diagnostics: string) =
    inherit
        Exception(
            sprintf
                "d3d12: the device was lost at %s (removed reason %s)%s"
                call
                (HResult.describe removedReason)
                (if String.IsNullOrEmpty diagnostics then
                     ""
                 else
                     "\n" + diagnostics)
        )

    /// The call after which the loss was seen, for example "ExecuteCommandLists" or "fence wait".
    member _.Call = call
    /// GetDeviceRemovedReason: S_OK (0) when the device still answers but work did not finish in time.
    member _.RemovedReason = removedReason
    /// DRED's report, as Dred.format writes it ("" when DRED had nothing).
    member _.Diagnostics = diagnostics

/// Raising GpuContractException with the texts the other backends use (GpuBackendInterface, "Errors").
module internal Contract =
    let fail fmt =
        Printf.kprintf (fun s -> raise (GpuContractException s)) fmt
