# tools/d3d12gen

Generates the D3D12 and DXGI interop (`src/Lrm.Gpu.D3D12.Interop/D3D12.g.fs` and `D3D12Methods.g.fs`) from the Windows SDK 10.0.26100.0 headers pinned in `bindings.json`. Owner: R3. Contract: [D3D12ComIdiom](../../docs/contracts/d3d12-com-idiom.md).

```bash
dotnet run --project tools/d3d12gen/Lrm.Tools.D3D12Gen -c Release    # regenerate
pwsh tools/d3d12gen/check.ps1                                        # codegen diff: 0 current, 1 stale, 4 headers differ
pwsh tools/abi-oracle/d3d12/run.ps1 -Update                          # then refresh the ABI oracle sources
```

- **`bindings.json`** is the single list of what is bound:
  - interfaces, with every base;
  - root structs, whose field types are bound too;
  - extra enums and `opaqueEnums`;
  - `renames` for methods an intrinsic member would hide;
  - `hresultValues` for methods whose HRESULT is a value;
  - the SDK pin: the version, the servicing build that produced the hashes (`servicing`, 10.0.26100.7705), and a SHA-256 per header.
- **Headers.** They are found through `--sdk`, then `LRM_WINSDK_INCLUDE`, then the `KitsRoot10` registry value plus `Include/<version>`, then `Program Files (x86)/Windows Kits/10/Include/<version>`. When they differ from the pin, the exit code is 4. The message names the pinned servicing build and the ones installed here.
- **Slots.** A method's slot is the position of its `DECLSPEC_XFGVIRT(base, method)` marker in the SDK's C vtable, inherited methods included.
- **Wrapper shapes.** These follow the C types and SAL annotations (the full table is in the contract):
  - `this` is always `inref<ComPtr<'I>>`, so no wrapper call can be a tail call;
  - `_In_reads_(n)` arrays become `ReadOnlySpan`s with single-element overloads;
  - a `RetVal` right after `This` is returned by value;
  - `_Outptr_` `void**` pointers are returned.
- **Pipeline.** `Preprocessor.fs` reads the headers as x64 MSVC C. `Header.fs` parses enums, structs, typedefs, vtables and `DEFINE_GUID`. `CppPass.fs` reads `MIDL_INTERFACE` bases and IIDs from a C++ pass. `Bind.fs` resolves types, layouts, the closure and the wrapper shapes (including SAL `_In_reads_` counts). `Emit.fs` writes the files.
