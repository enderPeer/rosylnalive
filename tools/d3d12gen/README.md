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
  - the SDK pin: a SHA-256 per header.
- **Slots.** A method's slot is the position of its `DECLSPEC_XFGVIRT(base, method)` marker in the SDK's C vtable, inherited methods included.
- **Pipeline.** `Preprocessor.fs` reads the headers as x64 MSVC C. `Header.fs` parses enums, structs, typedefs, vtables and `DEFINE_GUID`. `CppPass.fs` reads `MIDL_INTERFACE` bases and IIDs from a C++ pass. `Bind.fs` resolves types, layouts, the closure and the wrapper shapes. `Emit.fs` writes the files.
