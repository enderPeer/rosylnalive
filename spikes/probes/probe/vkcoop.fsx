#r @"C:\Users\end\dev\rosylnalive\gpu\vulkan\app\VkDeviceCheck\bin\Debug\net10.0\Lrm.Gpu.Vulkan.dll"
#nowarn "9"
#nowarn "51"
open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open Lrm.Gpu.Vulkan

[<Struct; StructLayout(LayoutKind.Sequential)>]
type CoopProps =
  val mutable sType: uint32
  val mutable pNext: nativeint
  val mutable M: uint32
  val mutable N: uint32
  val mutable K: uint32
  val mutable A: uint32
  val mutable B: uint32
  val mutable C: uint32
  val mutable R: uint32
  val mutable sat: uint32
  val mutable scope: uint32

[<UnmanagedFunctionPointer(CallingConvention.Winapi)>]
type PFN_coop = delegate of nativeint * nativeptr<uint32> * nativeptr<CoopProps> -> int

let compName = function
  | 0u -> "f16" | 1u -> "f32" | 2u -> "f64" | 3u -> "s8" | 4u -> "s16" | 5u -> "s32" | 6u -> "s64"
  | 7u -> "u8" | 8u -> "u16" | 9u -> "u32" | 10u -> "u64"
  | 1000141000u -> "bf16" | 1000491002u -> "e4m3" | 1000491003u -> "e5m2" | n -> string n
let scopeName = function 1u -> "device" | 2u -> "workgroup" | 3u -> "subgroup" | 5u -> "queuefamily" | n -> string n

let loader = (VulkanLoader.TryOpen()).Value
let inst = loader.CreateInstance("probe", VkVersion.VK_API_VERSION_1_4)
let pd = inst.PhysicalDevices.[0]
let exts = inst.DeviceExtensions pd
printfn "device extensions: %d" exts.Count
let interesting = ["VK_EXT_descriptor_heap";"VK_KHR_cooperative_matrix";"VK_NV_cooperative_matrix";"VK_NV_cooperative_matrix2";"VK_NV_cooperative_vector";"VK_KHR_shader_bfloat16";"VK_EXT_shader_float8";"VK_KHR_shader_untyped_pointers";"VK_KHR_shader_integer_dot_product";"VK_KHR_16bit_storage";"VK_KHR_8bit_storage";"VK_KHR_shader_float16_int8";"VK_EXT_subgroup_size_control";"VK_KHR_shader_subgroup_extended_types";"VK_KHR_maintenance5";"VK_KHR_maintenance6";"VK_KHR_maintenance7";"VK_KHR_maintenance8";"VK_KHR_maintenance9";"VK_KHR_maintenance10";"VK_KHR_shader_maximal_reconvergence";"VK_KHR_shader_relaxed_extended_instruction";"VK_KHR_vulkan_memory_model";"VK_EXT_shader_atomic_float";"VK_EXT_shader_atomic_float2";"VK_NV_shader_atomic_float16_vector";"VK_KHR_shader_atomic_int64";"VK_EXT_memory_budget";"VK_EXT_memory_priority";"VK_KHR_external_memory_win32";"VK_KHR_external_semaphore_win32";"VK_KHR_pipeline_binary";"VK_KHR_shader_expect_assume";"VK_EXT_descriptor_buffer";"VK_KHR_push_descriptor";"VK_NV_device_diagnostic_checkpoints";"VK_EXT_device_fault";"VK_KHR_calibrated_timestamps";"VK_KHR_shader_clock";"VK_KHR_compute_shader_derivatives";"VK_NV_shader_sm_builtins";"VK_EXT_shader_object";"VK_KHR_internally_synchronized_queues";"VK_KHR_device_address_commands";"VK_EXT_shader_64bit_indexing";"VK_KHR_shader_fma";"VK_KHR_copy_memory_indirect";"VK_NV_copy_memory_indirect"]
for e in interesting do
  match exts.TryFind e with
  | Some v -> printfn "  %-45s spec %d" e v
  | None -> printfn "  %-45s -" e
printfn "other KHR/EXT/NV extensions containing coop/matrix/vector/float/tensor/descriptor:"
for KeyValue(k,v) in exts do
  let l = k.ToLowerInvariant()
  if (l.Contains "coop" || l.Contains "matrix" || l.Contains "vector" || l.Contains "tensor" || l.Contains "float" || l.Contains "descriptor" || l.Contains "bfloat") && not (List.contains k interesting) then printfn "  %s %d" k v
match loader.LoadInstanceProc<PFN_coop>(inst.Handle, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR") with
| ValueNone -> printfn "no coop fn"
| ValueSome f ->
  let mutable n = 0u
  let h = pd.Handle
  f.Invoke(h, &&n, NativePtr.nullPtr) |> ignore
  let arr = Array.init (int n) (fun _ -> let mutable p = CoopProps() in p.sType <- 1000506001u; p)
  use p = fixed arr
  let r = f.Invoke(h, &&n, p)
  printfn "VK_KHR_cooperative_matrix configurations: %d (result %d)" n r
  for c in arr do
    printfn "  M%d N%d K%d  A=%s B=%s C=%s R=%s sat=%d scope=%s" c.M c.N c.K (compName c.A) (compName c.B) (compName c.C) (compName c.R) c.sat (scopeName c.scope)
(inst :> IDisposable).Dispose()
