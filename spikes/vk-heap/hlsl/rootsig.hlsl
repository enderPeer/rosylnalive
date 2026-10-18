// The one global root signature: a directly indexed CBV/SRV/UAV heap and the 128-byte ArgBlock as 32 root constants.
// Compiled with dxc -T rootsig_1_1 -E LRM_ROOTSIG.
#define LRM_ROOTSIG "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=32, b0)"
