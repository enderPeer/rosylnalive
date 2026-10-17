# VulkanHeapRecipe contract

- **Owner:** R1.
- **Status:** v1, proposed for contracts-v1.1, together with R4b.
- **Consumers:**
  - V1 (`src/Lrm.Gpu.Vulkan`) implements this exactly;
  - R4b finalizes the SPIR-V `LRM_BUF` and its inspector from it;
  - R4a's HeapLayoutAndTensorTable and KernelAbiAndCatalog contracts refer to it.
- **Evidence and rationale:** [`docs/decisions/R1-vulkan-heap-recipe.md`](../decisions/R1-vulkan-heap-recipe.md), with the spike in [`spikes/vk-heap`](../../spikes/vk-heap).

This contract fixes how the Vulkan backend binds LRM's one resource heap, using `VK_EXT_descriptor_heap` through the set/binding → heap mapping path (not direct heap indexing). Kernels see the same binding model on both backends:
- `LRM_BUF(slot)` is heap slot `slot`;
- the 128-byte ArgBlock arrives at push offset 0 (Vulkan) or as root constants at b0 (D3D12).

The numbers in brackets are the values measured on the RTX 4080 with driver 616.92. V1 must query them, not hard-code them.

## 1. Device

| Item | Rule |
| --- | --- |
| API | Vulkan 1.4 |
| Extensions enabled | `VK_EXT_descriptor_heap` only. `VK_KHR_shader_untyped_pointers` must be supported (an extension dependency) but is not enabled; the binding interface does not need it |
| Features enabled | `VkPhysicalDeviceDescriptorHeapFeaturesEXT.descriptorHeap`; Vulkan 1.2 `bufferDeviceAddress`, `timelineSemaphore` and `runtimeDescriptorArray`; Vulkan 1.3 `synchronization2`; `maintenance5` |
| Not required | `nullDescriptor` / `robustness2`, `shaderStorageBufferArrayNonUniformIndexing`, `vulkanMemoryModel`, untyped pointers |
| Limits asserted at device creation | `maxPushDataSize` ≥ 128 [256]. `vkGetPhysicalDeviceDescriptorSizeEXT(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)` equals `bufferDescriptorSize` [16]. The heap (section 3) fits `maxResourceHeapSize` [33,554,432]. A 1 GiB page fits `maxStorageBufferRange` [4,294,967,295] |
| Queue | The first family with COMPUTE, without GRAPHICS, and with `timestampValidBits` > 0 [family 2: COMPUTE \| TRANSFER \| SPARSE, 64 bits, `timestampPeriod` 1 ns] |

## 2. Memory and buffers

Memory types are chosen by property flags among the buffer's `memoryTypeBits`, never by index. The selected indices are recorded in the run manifest.

| Buffer | Usage | Memory flags | [RTX 4080 type] |
| --- | --- | --- | --- |
| Arena page (≤ 1 GiB) | STORAGE_BUFFER \| SHADER_DEVICE_ADDRESS \| TRANSFER_SRC \| TRANSFER_DST | DEVICE_LOCAL, not HOST_VISIBLE | [1] |
| Resource heap | `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT` (0x10000000) \| SHADER_DEVICE_ADDRESS | DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT, mapped persistently | [4] |
| Upload staging | TRANSFER_SRC | HOST_VISIBLE \| HOST_COHERENT, preferably not HOST_CACHED | [2] |
| Readback staging | TRANSFER_DST | HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED if present | [3] |

- Every allocation behind a buffer with SHADER_DEVICE_ADDRESS usage chains `VkMemoryAllocateFlagsInfo` with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`.
- A page's device address (`vkGetBufferDeviceAddress`) must be a multiple of `minStorageBufferOffsetAlignment` [16].
- **Fallback heap placement**, only when no DEVICE_LOCAL | HOST_VISIBLE type exists:
  1. put the heap in DEVICE_LOCAL memory, with TRANSFER_DST added to its usage;
  2. write the descriptors to host memory;
  3. copy them in with `vkCmdCopyBuffer`;
  4. record one `VkMemoryBarrier2`: src COPY / TRANSFER_WRITE, dst COMPUTE_SHADER / `VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT` (0x0400000000000000).

## 3. Heap geometry

| Quantity | Rule | [RTX 4080] |
| --- | --- | --- |
| `heapRange.address` | The heap buffer's device address, aligned up to `resourceHeapAlignment` (allocate that much slack) | [alignment 64; addresses came back aligned] |
| Slot *i* (0 ≤ *i* < 64) | Byte *i* × `bufferDescriptorSize` from `heapRange.address` | [16·*i*] |
| Slot area | 64 × `bufferDescriptorSize` | [1,024 B] |
| `reservedRangeOffset` | The slot area aligned up to max(64, `bufferDescriptorAlignment`, `imageDescriptorAlignment`) | [1,024] |
| `reservedRangeSize` | `minResourceHeapReservedRange` | [96,768] |
| `heapRange.size` | `reservedRangeOffset` + `reservedRangeSize`, at most `maxResourceHeapSize` | [97,792] |

Slot 0 is the tensor table. Slots 1 to 63 are arena pages (HeapLayoutAndTensorTable). A slot without a page holds a descriptor of a small dummy page; there are no null descriptors.

## 4. Writing the heap

- **Once.** The heap is written once at startup, before the first submit, with **one** `vkWriteResourceDescriptorsEXT(device, 64, pResources, pDescriptors)` call:
  - `pResources[i]` is a `VkResourceDescriptorInfoEXT` with `type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` and `data.pAddressRange` pointing at {address of slot *i*'s page, its size in bytes};
  - `pDescriptors[i]` is the `VkHostAddressRangeEXT` {mapped heap + *i* × `bufferDescriptorSize`, `bufferDescriptorSize`}.
- **Visibility.** Host writes to the coherent heap made before the submit need no barrier.
- **The reserved range is never accessed after the first bind.** The application does not read or write it while any command buffer that bound it is pending or unreset.
- **No hashing.** Descriptor bytes are opaque and depend on page addresses, so they never enter a content hash.

## 5. Pipelines

Each kernel is one compute pipeline:

```
VkComputePipelineCreateInfo
  pNext  -> VkPipelineCreateFlags2CreateInfo { flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT (0x1000000000) }
  flags  = 0
  layout = VK_NULL_HANDLE
  stage.pNext -> VkShaderDescriptorSetAndBindingMappingInfoEXT { mappingCount = 1, pMappings = &mapping }

mapping = VkDescriptorSetAndBindingMappingEXT {
  descriptorSet = 0, firstBinding = 0, bindingCount = 1,
  resourceMask  = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT | ..._READ_WRITE_STORAGE_BUFFER_BIT_EXT (0xC0),
  source        = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT (0),
  sourceData.constantOffset = { heapOffset = 0, heapArrayStride = bufferDescriptorSize,
                                pEmbeddedSampler = NULL, samplerHeapOffset = 0, samplerHeapArrayStride = 0 } }
```

`g_buf[i]` resolves to heapOffset + (Binding − firstBinding + *i*) × heapArrayStride = *i* × `bufferDescriptorSize`.

## 6. Recording and submission

1. **Bind the heap.** Once per command buffer, before the first dispatch, call `vkCmdBindResourceHeapEXT` with `VkBindHeapInfoEXT` {heapRange, reservedRangeOffset, reservedRangeSize} from section 3.
2. **Per dispatch:**
   - `vkCmdBindPipeline`;
   - `vkCmdPushDataEXT` with `VkPushDataInfoEXT` {offset = 0, data = the 128-byte ArgBlock}, whose dword 0 is `lrm_groups_x`;
   - `vkCmdDispatch`.

   Push data is pushed after every pipeline bind. `vkCmdPushConstants` is never used, since it is incompatible with heap pipelines.
3. **Between dependent dispatches:** `vkCmdPipelineBarrier2` with one `VkMemoryBarrier2`:
   - src COMPUTE_SHADER / SHADER_STORAGE_WRITE;
   - dst COMPUTE_SHADER | COPY / SHADER_STORAGE_READ | SHADER_STORAGE_WRITE | TRANSFER_READ.
4. **Submit.** `vkQueueSubmit2` signals one timeline semaphore at ALL_COMMANDS with a new value. The host waits with `vkWaitSemaphores` and a finite timeout.
5. **Timing.** `vkCmdWriteTimestamp2(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)` before and after each dispatch, masked to `timestampValidBits` and scaled by `timestampPeriod`. The results go out as `gpu.dispatch` GpuIntervals.

## 7. Shader side (for R4b)

- **SPIR-V binding.** `[[vk::binding(0, 0)]] RWByteAddressBuffer g_buf[] : register(u0, space0);` with `#define LRM_BUF(slot) g_buf[slot]`, and no `NonUniformResourceIndex`. Slot indices must be dynamically uniform. DXIL keeps `ResourceDescriptorHeap[slot]`.
- **ArgBlock.** `struct LrmArgBlock { uint4 d[8]; };`, declared as `ConstantBuffer<LrmArgBlock> : register(b0, space0)`, with `[[vk::push_constant]]` only under `#if defined(__spirv__)`. Dword *k* is `d[k >> 2][k & 3]`. A flat `uint d[32]` is not allowed: DXIL packs it into 500 bytes.
- **DXC flags.** `-T cs_6_6 -HV 2021 -O3 -WX -spirv -fspv-target-env=vulkan1.3`, with neither `-fvk-bind-resource-heap` nor `-fspv-use-descriptor-heap`.
- **What every module must satisfy:**
  1. exactly one variable with DescriptorSet or Binding decorations: storage class StorageBuffer, set 0, binding 0, not NonWritable;
  2. the PushConstant block has one member at Offset 0, ArrayStride 16, 128 bytes in total;
  3. LocalSize equals the catalog's thread count;
  4. the module declares `RuntimeDescriptorArray`.

## 8. Known-answer vectors

The spike's cpu and gpu tiers check each of these.

**KAT-1, the mapping struct (96 bytes, x64, `pNext` and `pEmbeddedSampler` NULL, `bufferDescriptorSize` = 16):**

```
5dd99c3b000000000000000000000000000000000000000001000000c0000000
0000000000000000000000001000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000
```

Bytes 0 to 3 are sType 1000135005. The first line also holds bindingCount 1 at byte 24 and resourceMask 0xC0 at byte 28. `sourceData` starts at byte 40, and its heapArrayStride of 16 sits at byte 44.

**KAT-2, the geometry for the RTX 4080 limits.** Inputs: `bufferDescriptorSize` 16, `bufferDescriptorAlignment` 8, `imageDescriptorAlignment` 32, `minResourceHeapReservedRange` 96,768. Outputs:
- slots at 16·*i*, slot area 1,024;
- `reservedRangeOffset` 1,024, `reservedRangeSize` 96,768;
- `heapRange.size` 97,792.

**KAT-3, struct layouts on x64** (size; member offsets):

| Struct | Size | Offsets |
| --- | --- | --- |
| `VkDescriptorSetAndBindingMappingEXT` | 96 | descriptorSet 16, firstBinding 20, bindingCount 24, resourceMask 28, source 32, sourceData 40 |
| `VkDescriptorMappingSourceDataEXT` (union) | 56 | all members 0 (the largest is `VkDescriptorMappingSourceIndirectIndexEXT`, 56) |
| `VkDescriptorMappingSourceConstantOffsetEXT` | 24 | heapOffset 0, heapArrayStride 4, pEmbeddedSampler 8, samplerHeapOffset 16, samplerHeapArrayStride 20 |
| `VkShaderDescriptorSetAndBindingMappingInfoEXT` | 32 | mappingCount 16, pMappings 24 |
| `VkBindHeapInfoEXT` | 48 | heapRange 16 (address 16, size 24), reservedRangeOffset 32, reservedRangeSize 40 |
| `VkPushDataInfoEXT` | 40 | offset 16, data 24 (address 24, size 32) |
| `VkResourceDescriptorInfoEXT` | 32 | type 16, data 24 |
| `VkHostAddressRangeEXT`, `VkDeviceAddressRangeEXT` | 16 | address 0, size 8 |
| `VkPipelineCreateFlags2CreateInfo` | 24 | flags 16 |

**KAT-4, the push data echo.** `args_echo` with this ArgBlock writes the same 32 words, in order, to its output tensor:

```
00000001 00000003 ffffffff 7fc00001 80000000 00000001 b519c20e 7fd49868
12508679 4c777b2c 3775418c 6b0e50d0 34a1269a 7f79bf4a 9a8da36d 2f6087d0
33f8d54a 40511b2b d98077b5 ac105150 dfd2c9fd d5b8cf0e 12ee2896 43de8e57
28dca649 65c6007e 610e2eea ae4d91c4 d2a32be2 86dc79be c1d53a9c 97b2b7a7
```

The output's SHA-256 is `74f5e898a05c7e2063639a11cbdaed2cd75d077e98c5c27296542f577afc3531`.

**KAT-5, the bf16 widening of every pattern.** `bf16_unpack` over the bf16 bit patterns 0 to 65,535 gives the u32 words `bits << 16`. Their little-endian bytes have SHA-256 `9207d7eb28680a098c73dbe536d1ff7b94311dc417b9a385e0af6660683e93ca`. Vulkan, D3D12 and the CPU agree.

## 9. Change rules

- This contract changes only through a `contract-change` PR, with a CHANGELOG entry and updated KATs.
- A new device or driver whose limits differ (section 3) does not change the contract: the rules are written in terms of the queried limits.
- Moving to direct heap indexing (`SPV_EXT_descriptor_heap`) would be a new contract version. It needs a DXC that supports UAVs on non-emulated heaps; 1.9.2609.5 does not.
