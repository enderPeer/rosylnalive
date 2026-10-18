namespace Lrm.Gpu.Core

open System

/// Raised when a caller breaks a rule of the GPU contracts (HeapLayoutAndTensorTable, KernelAbiAndCatalog,
/// GpuBackendInterface). Every backend raises it for the same violations, so tests can rely on it.
type GpuContractException(message: string) =
    inherit Exception(message)

module internal Fail =
    /// Raises GpuContractException with a formatted message.
    let contract fmt =
        Printf.kprintf (fun s -> raise (GpuContractException s)) fmt
