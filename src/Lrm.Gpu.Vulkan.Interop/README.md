# Lrm.Gpu.Vulkan.Interop — F# Vulkan binding foundation + device check

No third-party bindings. `vulkan-1.dll` is opened with `NativeLibrary`; only `vkGetInstanceProcAddr`
is bound by symbol, everything else is fetched through it (`vkGetDeviceProcAddr` is loaded for later use).

- `src/Lrm.Gpu.Vulkan.Interop` — library (namespace `Lrm.Gpu.Vulkan` for now)
  - `VkTypes.fs` — constants, versions, enums, handles, fixed-array stand-ins, instance structs
  - `VkCore.Generated.fs` — VkPhysicalDeviceLimits / SparseProperties / Features, transcribed from vk.xml
  - `VkStructs.fs` — Properties2/Features2, Driver props, Vulkan12/13 features, VK_EXT_descriptor_heap,
    VK_KHR_shader_untyped_pointers, VK_KHR_cooperative_matrix query structs
  - `VkCommands.fs` — PFN delegates, `VulkanLoader`, `Instance`
  - `DeviceCheck.fs` — `DeviceVerdict = Accepted of Capabilities | Rejected of reasons: string list`
- `apps/VkDeviceCheck` — prints each device + verdict; first verifies every struct's size, member order
  and offsets against `VkXmlLayout.fs` (C-ABI layout computed from vk.xml).
- `tools/vkgen/legacy/vkxml_layout.py` — generator for both generated files (reads vk.xml as data), until vkgen v2 replaces it.

```
dotnet run --project apps/VkDeviceCheck -c Release
```

Exit code: 0 = at least one device accepted, 3 = none accepted, 2 = layout mismatch, 1 = no loader.

Regenerate (vk.xml from Khronos Vulkan-Docs `xml/vk.xml`, currently VK_HEADER_VERSION 365):

```
python -I tools/vkgen/legacy/vkxml_layout.py <vk.xml> --fsharp VkPhysicalDeviceLimits,VkPhysicalDeviceSparseProperties,VkPhysicalDeviceFeatures
python -I tools/vkgen/legacy/vkxml_layout.py <vk.xml> --layout <comma-separated struct list, see VkXmlLayout.fs>
```

Assumes a 64-bit process (`PlatformTarget` x64, set for every project in `Directory.Build.props`); the layout table is 64-bit only.
