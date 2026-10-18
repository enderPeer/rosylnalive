// spirv-fixture: a raw device-address load (vk::RawBufferLoad), which bypasses the heap: the PhysicalStorageBuffer64
// addressing model, the PhysicalStorageBufferAddresses capability and a 64-bit address. SPIR-V 1.6 has the
// addressing in core, so DXC declares no extension for it.
// kernel: scale_f32
// expect: module, capabilities, scalar-width
#include "lrm.hlsli"

LRM_ARGS(scale_f32)

LRM_NUMTHREADS(scale_f32)
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint64_t address = ((uint64_t)lrm_args.x << 32) | (uint64_t)lrm_args.y;
    uint v = vk::RawBufferLoad<uint>(address + tid.x * 4u);
    lrm_store_u32(lrm_tensor(lrm_args.y), tid.x, v);
}
