# tools/abi-oracle/vulkan

The ABI oracle for `Lrm.Gpu.Vulkan.Interop`. Owner: R2. Contract: [VulkanInteropApi](../../../docs/contracts/vulkan-interop.md).

`Lrm.Gpu.Vulkan.Interop.Tests abi-oracle` loads the interop and dumps what the runtime uses:
- struct and union sizes, alignments, member offsets, member sizes and member kinds (arrays with their extent and element kind);
- enum values (64-bit flag bits in C++), handle sizes, API and extension constants;
- every command and funcpointer delegate's signature, against the `PFN_*` typedefs;
- every struct's sType, structextends relation and type alias, against vulkan.hpp.

They become C11 and C++17 static assertions over Vulkan-Headers 1.4.365, and cl.exe compiles them.

```bash
pwsh tools/abi-oracle/vulkan/run.ps1            # build, check the committed sources are current, compile and run
pwsh tools/abi-oracle/vulkan/run.ps1 -Update    # rewrite vk_abi.g.c and vk_sig.g.cpp after a bindings change
pwsh tools/abi-oracle/vulkan/run.ps1 -NoBuild   # CI, after the Release build
```

| File | Written by |
| --- | --- |
| `lrm_vk_abi.h`, `lrm_vk_sig.hpp`, `oracle_main.c` | Hand-written: the assertion macros and the summary |
| `vk_abi.g.c`, `vk_sig.g.cpp` | Generated from the runtime; never edited by hand |
| `spike-crosscheck.fsx` | Hand-written: compares R1's C declarations (`spikes/vk-heap/src/Layout.fs`, read as data) with the binding |

- **Headers.** The `vkheaders` entry of `tools/bootstrap/toolchain.json` (installed by bootstrap under `LRM_DATA`); `vulkan_core.h` is checked against bootstrap's install record before compiling.
- **Toolchain.** VS 2022 Build Tools, found through vswhere; objects and the exe go to `artifacts/abi-oracle/vulkan/`.
- **Exit codes.** 0 passed; 1 failed or stale; 2 no C++ toolchain or no headers.
- **Today.** 4,340 compile-time checks (2,801 C, 1,539 C++), 0 failures; the spike cross-check covers 48 structs and unions and 249 members, 0 mismatches.
- **In the cpu tier.** The same oracle, plus 15 mutants that must each be reported, and the spike cross-check.
