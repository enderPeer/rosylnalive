# R3-0001: D3D12 interop shape, and no F# tail calls around stack addresses

- Status: accepted (R3, 2026-10-18); decision 6 revised in review the same day
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
   - One COM call costs about 10 to 13 ns in the bench tier (AddRef and Release on a DXGI factory). That is negligible next to a dispatch, and it keeps the stack free of emitted IL.
3. **Phantom marker interfaces with constrained extension methods.**
   - Each COM interface is an empty .NET interface whose inheritance mirrors COM's. `ComPtr<'I>` is a struct over one `nativeint`.
   - Each method is generated once, as an extension method on `ComPtr<'I> when 'I :> Declarer`. Derived interfaces get their bases' methods through the constraint, and F# picks among same-named methods (`GetDesc` on a resource, a heap or a queue) by the constraint.
   - Type mistakes (a fence where a resource belongs) fail to compile, and no wrapper is duplicated per derived interface.
4. **The hidden RetVal convention comes from the header.** The generator reads the `_WIN32` branch of each vtable, where MIDL already writes struct returns as `T *M(This, T *RetVal, …)`, with the result pointer first. The wrapper returns `T` whatever parameters follow (`GetResourceAllocationInfo`, `GetCustomHeapProperties`). No method needs special-casing.
5. **Blittable by construction.** The interop assembly carries `DisableRuntimeMarshalling`, so a signature that would need marshalling fails at once instead of being marshalled silently.
6. **Wrapper calls are never tail calls: the wrappers take `this` by reference.**
   - **What happened.** On the GPU, the copy and dispatch tests failed: `ID3D12GraphicsCommandList::Close` returned `E_INVALIDARG` with no device removal. Close only fails when a recorded command was invalid. The barrier bytes, dumped just before the call, were correct. The same barrier recorded at script top level, or followed by any statement in the same method, was accepted.
   - **The cause.** F# compiled `list.ResourceBarrier(1u, NativePtr.toNativeInt &&b)`, the last expression of a method, with the `tail.` prefix. A tail call frees the caller's frame first, so the native code read `b` from a dead stack slot. Most of the time the bytes are still intact, so the failure is intermittent. That would break the bit-identical gates.
   - **Why F# did it.** F# leaves out `tail.` only when an argument of the call is a byref (`inref`, `outref`, `byref`) or the call is on a struct instance. A `nativeint` taken from `&&` is opaque to it, and so is a `ReadOnlySpan`, which is a struct.
   - **The fix is in the wrapper.** The first version of this ADR said no wrapper-side change could remove the hazard. Review showed that was wrong. Every generated wrapper now takes `this` as `inref<ComPtr<'I>>`. That is a byref argument on every call, so F# compiles every wrapper call as a normal call, whatever the caller's `Tailcalls` setting and whatever the other arguments are. Call sites do not change: `list.ResourceBarrier &b` passes `&list` implicitly.
   - **Measured.**
     - Before the fix, a scratch consumer built with `<Tailcalls>true</Tailcalls>` had `tail.` on `ResourceBarrier`, `SetComputeRoot32BitConstants` and `ExecuteCommandLists` called with `&&local`. The `inref` call `CreateCommandQueue(&d)` had none.
     - A first variant with span parameters on by-value `this` still had `tail.` on every span call.
     - With `inref` `this`, nothing has it. The cpu tier now checks this on every run: it compiles `TailCallConsumer.fs` with `--tailcalls+` and requires that no wrapper call carries `tail.`. Its control, a direct `Fn` delegate call, must carry it.
   - **Spans and single elements.** The generator also reads the SAL annotations, so the hot calls need no `&&` at all:
     - `_In_reads_(n)` arrays become `ReadOnlySpan<T>` (or `ReadOnlySpan<ComPtr<'P>>` for interface arrays), with a single-element `inref<T>` or `ComPtr<'P>` overload;
     - `n * sizeof(UINT)` buffers become `ReadOnlySpan<uint32>`, and byte buffers `ReadOnlySpan<byte>`;
     - `Values[4]` becomes a length-checked span;
     - `Map` returns its pointer.
     The C-shaped `…Raw` forms remain for data in native memory.
   - **What stays the caller's job.** Direct `Vtbl.get` and `Fn` delegate calls, and a project's own functions that pass `&&local` to each other, can still be tail-called. The interop and its tests keep `<Tailcalls>false</Tailcalls>` in their `.lrm.targets`, and a cpu-tier test scans their IL for `tail.` prefixes. For comparison, `Lrm.Core`, built with tail calls, has 67 methods with the prefix. `TailCalls.scan` is public, so D1's tests can run the same scan on D1.

## Consequences

- D1 (`src/Lrm.Gpu.D3D12`) calls the wrappers without any build setting. It should still set `<Tailcalls>false</Tailcalls>` in its `.lrm.targets` as defence in depth if it passes `&&locals` between its own functions, and check that with `TailCalls.scan`. V1 and R2 have the same exposure with Vulkan structs. A repo-wide default for the GPU projects in `Directory.Build.props` is requested from the integrator. Moving the scanner to `Lrm.Testing` for every backend is requested too.
- New interfaces, structs or enums are added in `tools/d3d12gen/bindings.json`, then regenerated, then `tools/abi-oracle/d3d12/run.ps1 -Update`, all in one PR.
- A different SDK servicing build changes the header hashes. The codegen diff then reports exit 4 instead of guessing, and the cpu tier records the skip as a `gpu.codegen.d3d12.skipped` event. `bindings.json` names the servicing build that reproduces the pin (`sdk.servicing`, 10.0.26100.7705). The oracle still compiles against whatever SDK the machine has, because COM vtables and struct layouts only grow.
