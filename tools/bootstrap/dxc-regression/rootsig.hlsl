// The global root signature of the D3D12 backend: a directly indexed resource heap and 32 root constants at b0
// (the 128-byte argument block). Compiled with -T rootsig_1_1 -E LRM_ROOTSIG.
#define LRM_ROOTSIG "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)"
