# R1: the Vulkan descriptor-heap recipe

- **Status:** accepted. This ADR records the VulkanHeapRecipe contract (normative text: [`docs/contracts/vulkan-heap-recipe.md`](../contracts/vulkan-heap-recipe.md)).
- **Date:** 2026-10-18.
- **Owner:** R1.
- **Consumers:**
  - V1 implements the recipe exactly;
  - R4b finalizes the SPIR-V `LRM_BUF` from it;
  - R4a's HeapLayout and KernelAbi contracts refer to it.
- **Evidence:** the spike in [`spikes/vk-heap`](../../spikes/vk-heap), with its generated results in [`spikes/vk-heap/results/results.md`](../../spikes/vk-heap/results/results.md).

## Context

The plan's top risk was that the `VK_EXT_descriptor_heap` binding path was unproven at run time. That path is the set/binding → heap mapping, with push data and a `NULL` pipeline layout. The fixed decisions add three constraints:
- start with the mapping path, not direct heap indexing;
- use one resource heap written once at startup;
- pass kernel arguments as 32-bit tokens in at most 128 bytes.

The spike had to show four things on the RTX 4080:
1. the whole path works from our own F# bindings;
2. which SPIR-V binding variant to use;
3. which features and memory types it needs;
4. that results are bit-exact against the CPU and identical to the same HLSL on D3D12.

## What was run

The test machine is one NVIDIA GeForce RTX 4080. On Vulkan it reports driver 616.92, device API 1.4.351 and loader 1.4.341. On D3D12 it reports UMD 32.0.16.1692, highest shader model 6.8 and resource binding tier 3.

| Item | Detail |
| --- | --- |
| Kernels | `scale_f32`, `args_echo` (all 32 ArgBlock dwords), `table_echo` (the whole 16,384-row tensor table), `bf16_unpack` (all 65,536 patterns, and 2 Mi values of the real Qwen3-0.6B `embed_tokens`). There are also two dependent dispatches in one command buffer, a scale at the end of a 1 GiB page, a 2-D grid that uses `lrm_groups_x`, and an informational case with denormals, infinities and NaN |
| Compiler | The pinned DXC 1.9.2609.5: `-T cs_6_6 -HV 2021 -O3 -WX`. SPIR-V adds `-spirv -fspv-target-env=vulkan1.3`. ShaderSetHash `6017356f…5caebd` |
| Vulkan configurations | Two binding variants (explicit, emulated) × three heap placements (BAR written in place, host memory written in place, device-local filled by a copy), plus two mapping controls (heapOffset 512; stride 32) |
| D3D12 | The same HLSL as DXIL. It uses `ResourceDescriptorHeap`, the global root signature (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, 32 root constants at b0), one shader-visible heap of 64 raw UAVs and a compute queue |
| Checks | Every case runs twice and must give identical bytes. Gating cases must match the F# CPU reference word for word. Every configuration must give the same SHA-256 as D3D12 |
| Layout | Every hand-written struct is checked at run time against the C declaration transcribed from the reference pages: names, order, offsets, size and blittability. cl.exe compiles `static_assert`s that check the layout calculator |

**Result:** 115 of 115 gpu-tier tests and 12 of 12 cpu-tier tests pass.
- Every gating case is bit-exact against the CPU on all 9 Vulkan configurations and on D3D12.
- Every configuration produces one SHA-256 per case.
- Every case is identical across its two runs.

The whole GPU run takes about 2 seconds.

## Decision

### Binding variant: explicit array

Use `[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0)` on SPIR-V, so that `LRM_BUF(i)` is `g_buf[i]`. DXIL keeps `ResourceDescriptorHeap[i]`.

Both variants are correct, and they are equally fast within timer noise. The explicit variant wins on the properties the project depends on:

| | Explicit `g_buf[]` | Emulated `-fvk-bind-resource-heap 0 0` |
| --- | --- | --- |
| Results on the RTX 4080 | Bit-exact; same digests as D3D12 | Bit-exact; same digests as D3D12 |
| Resource variables per module | Exactly 1, at set 0 binding 0 | One per distinct `ResourceDescriptorHeap` use, all aliasing set 0 binding 0 (2 for `args_echo`, 3 for the others) |
| R4b inspector rule "one variable at set 0 binding 0" | Holds | Fails by construction |
| Where the binding is decided | In our source (`lrm.hlsli`) | In a DXC emulation flag, whose behaviour the proposal describes as current DXC behaviour that may change |
| Shader source | `#if __spirv__` in one macro | Identical on both targets |

The third option, direct heap indexing through `-fspv-use-descriptor-heap` (`SPV_EXT_descriptor_heap`), is not usable with the pinned DXC. It warns "SPV_EXT_descriptor_heap support is incomplete" and fails with "UAV support not implemented with non-emulated heaps". The fixed decision rules it out for now in any case.

### The recipe

The normative text, with known-answer vectors, is the contract [`docs/contracts/vulkan-heap-recipe.md`](../contracts/vulkan-heap-recipe.md). In short:

1. **Device.**
   - Vulkan 1.4, with one extension enabled: `VK_EXT_descriptor_heap`. `VK_KHR_shader_untyped_pointers` must be supported but is not enabled; the binding interface does not need it.
   - Features enabled, exactly:
     - `descriptorHeap`;
     - Vulkan 1.2 `bufferDeviceAddress`, `timelineSemaphore` and `runtimeDescriptorArray`. The SPIR-V declares `RuntimeDescriptorArray` for `g_buf[]`;
     - Vulkan 1.3 `synchronization2`;
     - `maintenance5`, for `VkPipelineCreateFlags2CreateInfo`.
2. **Queue.** A compute-only family: no GRAPHICS bit, and timestamps present. On the RTX 4080 this is family 2 (COMPUTE | TRANSFER | SPARSE), with 64 timestamp bits and 1 ns per tick.
3. **Memory types.** Selected by flags, never by index; the indices below are the RTX 4080 values with driver 616.92.

   | Use | Flags | RTX 4080 type |
   | --- | --- | --- |
   | Arena pages (device addresses) | DEVICE_LOCAL, not HOST_VISIBLE | 1 |
   | Resource heap | DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT (BAR; heap 0, the full 16,048 MiB) | 4 |
   | Upload staging | HOST_VISIBLE \| HOST_COHERENT, not HOST_CACHED | 2 |
   | Readback staging | HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED | 3 |

4. **Pages.**
   - Each page is a buffer with STORAGE_BUFFER | SHADER_DEVICE_ADDRESS | TRANSFER_SRC | TRANSFER_DST usage, allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`.
   - Each page address is a multiple of `minStorageBufferOffsetAlignment` (16).
   - Each page is at most 1 GiB, well under `maxStorageBufferRange` (4 GiB − 1).
5. **Heap.**
   - **The buffer.** Usage is `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT` (0x10000000) | SHADER_DEVICE_ADDRESS. It is mapped persistently.
   - **Base address.** `heapRange.address` is the buffer's device address, aligned up to `resourceHeapAlignment` (64 B on the RTX 4080). The driver returned 64-aligned addresses.
   - **Slots.** Slot *i* is at byte *i* × `bufferDescriptorSize`. That size is 16 on the RTX 4080, and `vkGetPhysicalDeviceDescriptorSizeEXT(STORAGE_BUFFER)` agrees. The 64 slots take 1,024 bytes.
   - **Reserved range.** It follows the slots, at the slot bytes aligned up to max(64, `bufferDescriptorAlignment`, `imageDescriptorAlignment`) = 1,024. Its size is `minResourceHeapReservedRange` = 96,768. `heapRange.size` is therefore 97,792, within `maxResourceHeapSize` (32 MiB).
6. **Writing.**
   - All 64 slots are written once at startup, in one `vkWriteResourceDescriptorsEXT` call. Each slot is a `STORAGE_BUFFER` descriptor whose `VkDeviceAddressRangeEXT` is {page address, page bytes}. The destinations are {mapped heap + 16·*i*, 16}.
   - Every slot must hold a valid descriptor. Slots without a page point at a 64 KiB dummy page, because null descriptors need `robustness2`.
   - Host writes made before the first submit need no barrier, since the memory is coherent.
7. **Pipelines.**
   - `VkComputePipelineCreateInfo.pNext` carries `VkPipelineCreateFlags2CreateInfo` with `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` (0x1000000000). `layout` is `VK_NULL_HANDLE`.
   - The stage's `pNext` carries one `VkDescriptorSetAndBindingMappingEXT`, with these fields:
     - descriptor set 0, firstBinding 0, bindingCount 1;
     - `resourceMask` = READ_ONLY_STORAGE_BUFFER | READ_WRITE_STORAGE_BUFFER (0xC0);
     - source `HEAP_WITH_CONSTANT_OFFSET`;
     - `heapOffset` 0, `heapArrayStride` = `bufferDescriptorSize`, and no embedded sampler.
   - `g_buf[i]` then resolves to heapOffset + (binding − firstBinding + *i*) × stride = 16·*i*.
8. **Recording.**
   - Once per command buffer, before the first dispatch: `vkCmdBindResourceHeapEXT` {heapRange, reservedRangeOffset 1,024, reservedRangeSize 96,768}.
   - Per dispatch: `vkCmdBindPipeline`, then `vkCmdPushDataEXT` {offset 0, the 128-byte ArgBlock}, then `vkCmdDispatch`.
   - Between dependent dispatches: one `VkMemoryBarrier2`, from COMPUTE_SHADER / SHADER_STORAGE_WRITE to COMPUTE_SHADER | COPY / SHADER_STORAGE_READ | SHADER_STORAGE_WRITE | TRANSFER_READ.
9. **Completion.** `vkQueueSubmit2` signals one timeline semaphore at ALL_COMMANDS. The host waits with `vkWaitSemaphores` and a timeout, so a hung dispatch fails the run instead of blocking it. Timestamps come from `vkCmdWriteTimestamp2(ALL_COMMANDS)` around each dispatch.
10. **Shaders.**
    - The ArgBlock is `struct { uint4 d[8]; }` in a `ConstantBuffer` at `register(b0)`. It carries `[[vk::push_constant]]` only under `__spirv__`, and dword *k* is `d[k >> 2][k & 3]`.
    - With heap pipelines, the shader's PushConstant block reads push data at the same offsets.
    - The SPIR-V must have:
      - exactly one resource variable: StorageBuffer, set 0, binding 0, not NonWritable;
      - a push block with one member at Offset 0, ArrayStride 16 and 128 bytes;
      - LocalSize equal to the catalog value;
      - the `RuntimeDescriptorArray` capability.

### Fallback heap placements

These are validated alternatives, not the recipe.
- **No BAR type.** If a device has no DEVICE_LOCAL | HOST_VISIBLE type, the heap goes in DEVICE_LOCAL memory. The descriptors are written to host memory and uploaded with `vkCmdCopyBuffer`. A barrier then follows, from COPY / TRANSFER_WRITE to COMPUTE_SHADER / `VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT` (0x0400000000000000). The barrier's second scope covers later submissions on the queue.
- **Host memory.** A heap in HOST_VISIBLE | HOST_COHERENT system memory also works. Every descriptor fetch then crosses PCIe, which tends to add up to about 2 µs to the smallest dispatches (see Measurements).

## Measurements

GPU time per job, in microseconds, as the minimum of two runs (run `20261017-232545-c79db331`, the run behind the committed results page). Timestamps are taken around each dispatch; the full matrix and the chart are in the results page.

| Case | d3d12 | explicit/bar | explicit/host | explicit/copy | emulated/bar |
| --- | --- | --- | --- | --- | --- |
| `scale_f32` (1 Mi floats) | 7.2 | 6.1 | 6.3 | 6.7 | 6.1 |
| `scale_f32.page1gib` | 6.1 | 6.1 | 6.2 | 5.1 | 6.1 |
| `scale_f32.chain2` (two dispatches) | 12.3 | 11.3 | 12.3 | 10.7 | 10.2 |
| `args_echo.0` to `.3` (one group each) | 1.0 | 1.0 to 2.0 | 1.7 to 3.1 | 0.8 to 1.5 | 0.7 to 1.0 |
| `table_echo` (1 MiB) | 2.0 | 3.1 | 2.7 | 1.6 | 2.6 |
| `bf16_unpack.qwen_embed` (2 Mi) | 11.3 | 9.9 | 11.3 | 9.8 | 10.2 |

- Across repeated runs the numbers move by about ±1 µs. Within that noise, neither the binding variant nor the heap placement makes a resolvable difference.
- The one consistent tendency is that a heap in host memory makes the smallest dispatches slower, by up to about 2 µs, because their descriptors are fetched over PCIe.
- These numbers cover the binding path only and are not a performance baseline; the bench tier owns that.

## Pitfalls found

1. **`[[vk::push_constant]]` on DXIL is an error under `-WX`.** DXC reports it as an ignored attribute. Guard it with `#if defined(__spirv__)`.
2. **`uint d[32]` is the wrong ArgBlock type.**
   - DXIL lays it out as a 500-byte legacy cbuffer (one 16-byte register per element), which does not fit 32 root constants.
   - SPIR-V lays the same block out with stride 4 and 128 bytes.
   - `uint4 d[8]` is 128 bytes with stride 16 on both. The spike's `flat_args_probe.hlsl` test records this.
3. **`vkCmdPushConstants` is incompatible with heap pipelines.** Use `vkCmdPushDataEXT`. No push-constant range or layout is declared anywhere.
4. **The mapping must cover every set/binding variable.**
   - The emulated variant emits several aliasing variables. One mapping covers them all because they share set, binding and resource kind.
   - `resourceMask` includes READ_ONLY_STORAGE_BUFFER as well, so that a NonWritable decoration from a future DXC would still be mapped.
5. **There are no null descriptors without `robustness2`.** All 64 slots get a real descriptor; unused slots point at the dummy page.
6. **Reserved-range alignment.**
   - Its offset must be a multiple of both `bufferDescriptorAlignment` (8) and `imageDescriptorAlignment` (32), even in a heap that holds only buffers.
   - Its size is about 95 times the slot area. The canary written before the first bind was intact afterwards, so the driver did not touch it in these runs.
   - It is still implementation-owned while any command buffer has it bound. Never read or write it after the first bind.
7. **The value of `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT` is 0x10000000.** Take it from the registry: 0x08000000 is `VK_BUFFER_USAGE_TILE_MEMORY_BIT_QCOM`.
8. **`resourceHeapAlignment` is 64 bytes on the RTX 4080, not 64 KiB.** Align up to the queried value anyway.
9. **Descriptor bytes are opaque and encode the address and size.** Slot 1 is `00001707 00000000 00000002 00000000`. The same {address, size} gives the same bits, so heap bytes depend on allocation addresses. Never hash them into a content id.
10. **Denormals and NaN differ from the CPU, identically on both GPU backends.**
    - fp32 denormals are flushed to zero: a denormal operand, or a product in the denormal range, comes out as zero.
    - A NaN operand gives 0x7FFFFFFF, whatever its payload.
    - The .NET CPU path keeps denormals and propagates NaN payloads, for example 0x7FC00006.
    - Vulkan and D3D12 give the same digest for these values on this device.

    `scale_f32` is therefore bit-exact against the CPU only for normal inputs with normal results. KernelBounds (O2), K1's cases and E1's CpuDevice must either generate such inputs or model the flush.
11. **D3D12 resource states.** On D3D12, buffers decay to COMMON after every `ExecuteCommandLists`, so each command list starts its transitions from COMMON. `SetDescriptorHeaps` must precede `SetComputeRootSignature` for a directly indexed heap.
12. **Descriptor sizes differ between backends.**
    - The D3D12 CBV/SRV/UAV descriptor increment is 32 bytes on the RTX 4080; the Vulkan buffer descriptor is 16 bytes.
    - Slot indices are the shared abstraction; byte offsets are private to each backend.
13. **The heap address was the same across recreation (0x2ffe30000), because the allocator reused it.** A stale-heap bug would not show up. V1 creates the heap once and asserts that it never changes.
14. **pNext chains must not outlive their stack frame.** Chains built from stack locals are valid only inside that frame, as the VulkanInteropApi contract states. The spike keeps every create-info in a scratch arena that lives for the whole call.

## Not covered

- **No validation layers.** AP-3 is not approved, so correctness rests on the reference pages and on the results. If AP-3 is approved, V1 should run its suite under validation once.
- **One device and driver.** Only the RTX 4080 with driver 616.92 was tested. Gates use the driver recorded in the manifest.
- **Push data across pipeline binds.** The spike did not test whether push data survives a pipeline bind. V1 pushes after every bind, so it does not rely on that.
- **Non-uniform slot indices.** These were not exercised; every LRM index is dynamically uniform. A kernel that needs non-uniform indices would need `NonUniformResourceIndex` and the `shaderStorageBufferArrayNonUniformIndexing` feature on this path.
- **Secondary command buffers, samplers and images** are not used by LRM in Phase 1.

## Consequences

- **R4b.**
  - Make the explicit array the final SPIR-V `LRM_BUF`, with no `NonUniformResourceIndex`.
  - The inspector asserts the SPIR-V facts in recipe step 10.
  - The ArgBlock HLSL type is `uint4 d[8]`, which R4a's generated `lrm_args.g.hlsli` must keep.
- **V1.** Implement the recipe exactly, and assert at device creation:
  - `bufferDescriptorSize` equals `vkGetPhysicalDeviceDescriptorSizeEXT(STORAGE_BUFFER)`;
  - the heap fits `maxResourceHeapSize`;
  - the page addresses are aligned.

  Record the memory type indices and the driver in the run manifest.
- **R4a's HeapLayout.** The text "Vulkan slot i at byte 16·i" holds on the RTX 4080. The general rule is *i* × `bufferDescriptorSize`, and shaders never see it.
- **O2, K1 and E1.** Account for pitfall 10 (denormal flush, canonical NaN).
