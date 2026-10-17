# R3-0001: D3D12 interop shape, and no F# tail calls around stack addresses

- Status: accepted (R3, 2026-10-18)
- Workstream: R3, D3D12 COM interop and device check
- Contract: [D3D12ComIdiom](../contracts/d3d12-com-idiom.md)

## Context

D1 builds the D3D12 backend on this interop, in F#, without third-party bindings, ComImport or RCWs (fixed decisions in `CLAUDE.md`). The planning probe (`spikes-staging/probe/d3d12probe.fsx`) proved the approach by hand. It ran a ResourceDescriptorHeap dispatch on the RTX 4080 through raw vtable slots read from `d3d12.h` and delegates from `GetDelegateForFunctionPointer`. R3 turns that into a checked, generated layer.

## Decisions

1. **Generate from the SDK headers, then check the runtime independently.**
   - `tools/d3d12gen` reads the MIDL-generated C vtables of the pinned Windows SDK 10.0.26100.0 headers. It takes a slot from the position of the method's `DECLSPEC_XFGVIRT` marker, which MIDL emits once per function pointer, inherited methods included. It takes the IIDs from `DEFINE_GUID`, cross-checked against `MIDL_INTERFACE`.
   - The ABI oracle does not trust the generator. It dumps what the CLR actually lays out and calls, and cl.exe checks that against the same headers.
   - Alternatives rejected: hand-written slot numbers, which the probe needed but nobody can review at 779 slots; and win32metadata or TerraFX, which are third-party bindings and excluded.
2. **Cached delegates, not `calli` stubs.**
   - F# cannot call an unmanaged function pointer directly. The options were a delegate from `GetDelegateForFunctionPointer` or a `DynamicMethod` emitting `calli`.
   - Delegates are cached per (delegate type, function pointer), with the last hit kept in a field, so the common case skips the dictionary.
   - One COM call costs about 13 ns in the bench tier (AddRef and Release on a DXGI factory). That is negligible next to a dispatch, and it keeps the stack free of emitted IL.
3. **Phantom marker interfaces with constrained extension methods.**
   - Each COM interface is an empty .NET interface whose inheritance mirrors COM's. `ComPtr<'I>` is a struct over one `nativeint`.
   - Each method is generated once, as an extension method on `ComPtr<'I> when 'I :> Declarer`. Derived interfaces get their bases' methods through the constraint, and F# picks among same-named methods (`GetDesc` on a resource, a heap or a queue) by the constraint.
   - Type mistakes (a fence where a resource belongs) fail to compile, and no wrapper is duplicated per derived interface.
4. **The hidden RetVal convention comes from the header.** The generator reads the `_WIN32` branch of each vtable, where MIDL already writes struct returns as `T *M(This, T *RetVal)`. No method needs special-casing.
5. **Blittable by construction.** The interop assembly carries `DisableRuntimeMarshalling`, so a signature that would need marshalling fails at once instead of being marshalled silently.
6. **No tail calls in code that passes stack addresses.**
   - **What happened.** On the GPU, the copy and dispatch tests failed: `ID3D12GraphicsCommandList::Close` returned `E_INVALIDARG` with no device removal. Close only fails when a recorded command was invalid. The barrier bytes, dumped just before the call, were correct. The same barrier recorded at script top level, or followed by any statement in the same method, was accepted.
   - **The cause.** F# compiled `list.ResourceBarrier(1u, NativePtr.toNativeInt &&b)`, the last expression of a method, with the `tail.` prefix. A tail call frees the caller's frame first, so the native code read `b` from a dead stack slot.
   - **Why F# did it.** The compiler suppresses tail calls when a byref escapes, but a `nativeint` taken from `&&` is opaque to it.
   - **Confirmed.** With `<Tailcalls>false</Tailcalls>` every test passes unchanged.
   - **Rule.** Projects that call the interop with `&&` set `Tailcalls` to false in their `<Project>.lrm.targets`. The interop and its tests do, and a cpu-tier test scans their IL for `tail.` prefixes. For comparison, `Lrm.Core`, built with tail calls, has 67 methods with the prefix.
   - **Why not mitigate in the wrapper.** The hazard is in the caller's frame, so no wrapper-side change can remove it.

## Consequences

- D1 (`src/Lrm.Gpu.D3D12`) and its tests must set `<Tailcalls>false</Tailcalls>` in their `.lrm.targets` before passing `&&locals`. V1 and R2 have the same exposure with Vulkan structs. A repo-wide default for the GPU projects in `Directory.Build.props` is requested from the integrator.
- New interfaces, structs or enums are added in `tools/d3d12gen/bindings.json`, then regenerated, then `tools/abi-oracle/d3d12/run.ps1 -Update`, all in one PR.
- A different SDK servicing build changes the header hashes. The codegen diff then reports exit 4 instead of guessing. The oracle still compiles against whatever SDK the machine has, because COM vtables and struct layouts only grow.
