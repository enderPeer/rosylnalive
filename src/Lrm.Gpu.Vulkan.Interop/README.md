# Lrm.Gpu.Vulkan.Interop

The F# binding to the Vulkan C API (namespace `Lrm.Gpu.Vulkan.Interop`), generated from vk.xml 1.4.365 by `tools/vkgen`, checked by the cl.exe ABI oracle in `tools/abi-oracle/vulkan`, and the shared Vulkan device check. No third-party bindings. Owner: R2. Contract: [VulkanInteropApi](../../docs/contracts/vulkan-interop.md).

| File | Contents |
| --- | --- |
| `Vk.g.fs` | Generated: constants, basetypes, handles, enums, bitmasks, inline arrays, structs and unions, funcpointer and command delegates |
| `VkCommands.g.fs` | Generated: `VkGlobalCommands`, `VkInstanceCommands`, `VkDeviceCommands` (raw delegates and wrappers) |
| `VkMeta.g.fs` | Generated: what is bound, with the generator's C layout of every struct |
| `AbiAttributes.fs` | The markers on generated items that the oracle reads |
| `Support.fs` | `VulkanException`, `VulkanMissingCommandException`, the loading helper and the metadata types |
| `Text.fs` | UTF-8 strings at the boundary, `Array256_byte.AsString`, `VkVersion` |
| `Chain.fs` | pNext chains: `VkChain.prepend`, `walk` and `problems` |
| `LayoutCheck.fs` | Runtime layout against VkMeta, run before any device is touched |
| `Loader.fs` | `VulkanLoader` (only `vkGetInstanceProcAddr` by symbol) and `Instance` |
| `Device.fs` | `FeatureChain` and `PropertiesChain` (byref-like), `DeviceOptions`, `QueueFamily.recipe`, `Device.create` |
| `DeviceCheck.fs` | The VulkanHeapRecipe acceptance rules, the report, text and JSON |
| `Diagnostics.fs` | `DebugMessenger`, `DebugUtils` names and labels, `DeviceFault.query` |

```bash
dotnet artifacts/bin/VkDeviceCheck/release/VkDeviceCheck.dll [--json]      # layouts, then the device check
dotnet artifacts/bin/Lrm.Cli/release/Lrm.Cli.dll devices-vulkan [--json]   # the same from the lrm CLI
```

VkDeviceCheck exits 0 when a device is accepted, 1 without a loader, 2 on a layout mismatch and 3 when no device is accepted. The binding assumes a 64-bit process (`PlatformTarget` x64 for every project).
