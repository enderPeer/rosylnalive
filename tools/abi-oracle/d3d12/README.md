# tools/abi-oracle/d3d12

The ABI oracle for `Lrm.Gpu.D3D12.Interop`. Owner: R3. Contract: [D3D12ComIdiom](../../../docs/contracts/d3d12-com-idiom.md).

`Lrm.Gpu.D3D12.Interop.Tests abi-oracle` loads the interop and dumps what the runtime uses:
- struct sizes and field offsets;
- enum values;
- every vtable slot and slot count;
- every delegate and P/Invoke signature;
- HRESULTs and IIDs.

They become C11 and C++17 static assertions over the real SDK headers. cl.exe compiles them, and `oracle_main.c` compares the IIDs at run time.

```bash
pwsh tools/abi-oracle/d3d12/run.ps1            # build, check the committed sources are current, compile and run
pwsh tools/abi-oracle/d3d12/run.ps1 -Update    # rewrite d3d12_abi.g.c and d3d12_sig.g.cpp after a bindings change
pwsh tools/abi-oracle/d3d12/run.ps1 -NoBuild   # CI, after the Release build
```

| File | Written by |
| --- | --- |
| `lrm_abi.h`, `lrm_abi_sig.hpp`, `oracle_main.c` | Hand-written: the assertion macros and the run-time IID check |
| `d3d12_abi.g.c`, `d3d12_sig.g.cpp` | Generated from the runtime; never edited by hand |

- **Toolchain.** VS 2022 Build Tools, found through vswhere, with `vcvars64.bat 10.0.26100.0` when that SDK is installed. Objects and the exe go to `artifacts/abi-oracle/d3d12/`.
- **Exit codes.** 0 passed; 1 failed or stale; 2 no C++ toolchain.
- **In the cpu tier.** The cpu tier runs the same oracle. It also compiles mutants (a wrong size, offset, slot, enum, HRESULT, signature or IID) that must fail.
