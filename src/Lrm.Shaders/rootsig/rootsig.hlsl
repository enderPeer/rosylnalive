// The global root signature of the D3D12 backend (ShaderBuildAndSourceConvention contract): a directly indexed
// CBV/SRV/UAV heap and 32 root constants at b0 (the 128-byte argument block). The shader build compiles it with
// -T rootsig_1_1 -E LRM_ROOTSIG into rootsig.bin and fails unless the bytes equal RootSignature.canonical.
#define LRM_ROOTSIG "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)"
