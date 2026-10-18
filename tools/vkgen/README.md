# tools/vkgen

Generates the Vulkan interop (`src/Lrm.Gpu.Vulkan.Interop/Vk.g.fs`, `VkCommands.g.fs` and `VkMeta.g.fs`) from the Khronos registry vk.xml 1.4.365 and `bindings.json`. Owner: R2. Contract: [VulkanInteropApi](../../docs/contracts/vulkan-interop.md). Rationale: ADR [R2-0001](../../docs/decisions/R2-0001-vkgen-and-vulkan-interop.md).

```bash
pwsh tools/bootstrap/bootstrap.ps1                                   # installs the pinned vk.xml under LRM_DATA
dotnet run --project tools/vkgen/Lrm.Tools.VkGen -c Release           # regenerate
pwsh tools/vkgen/check.ps1                                           # codegen diff: 0 current, 1 stale, 4 vk.xml missing
pwsh tools/abi-oracle/vulkan/run.ps1 -Update                         # then refresh the ABI oracle sources
```

- **`bindings.json`** is the single list of what is bound: the core version (1.4), the extensions whose values and commands count, and the root structs, enums, funcpointers and commands (global, instance and device). The generator adds every type the roots reach by value, and the structs and handles command parameters point to.
- **The pin.** vk.xml is the `vkxml` entry of `tools/bootstrap/toolchain.json`; the generator checks its size and SHA-256 before reading it. `--vkxml <path>` reads another copy (it must still match the pin).
- **Pipeline.** `Registry.fs` reads vk.xml for the `vulkan` API (types, enums, constants, commands, features, extensions). `Bind.fs` decides the closure, the enum values (`<enums>` blocks plus the bound `<require>` additions, `depends` evaluated), the x64 C layout of every struct, the F# types and the wrapper shapes. `Emit.fs` writes the files, sorted by name and in dependency order, with LF endings.
- **`legacy/vkxml_layout.py`** is the Python layout script that S0a moved here. Nothing runs it any more; it stays for reference.
