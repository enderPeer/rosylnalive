namespace Lrm.Gpu.Vulkan.Interop

open System.Runtime.CompilerServices

// Blittable by construction (VulkanInteropApi contract): with runtime marshalling disabled, a delegate or struct that
// would need marshalling fails at once instead of being marshalled silently.
[<assembly: DisableRuntimeMarshalling>]
do ()
