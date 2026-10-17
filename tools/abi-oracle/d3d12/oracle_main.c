/* ABI oracle for Lrm.Gpu.D3D12.Interop: the run-time part. Compile-time checks live in the generated
   d3d12_abi.g.c (C11) and d3d12_sig.g.cpp (C++17); reaching main means they all passed. Here the IIDs are compared,
   because a GUID's bytes are not a C constant expression. Owner: R3. Hand-written. */
#include "lrm_abi.h"
#include <stdio.h>
#include <string.h>

extern const lrm_iid_check lrm_iids[];
extern const int lrm_iid_count;
extern const int lrm_static_checks_c;
extern int lrm_static_checks_cpp(void);

static void print_guid(const IID *g)
{
    printf("%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3, g->Data4[0],
           g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
}

int main(void)
{
    int failures = 0;
    int i;

    for (i = 0; i < lrm_iid_count; i++)
    {
        if (memcmp(lrm_iids[i].header, &lrm_iids[i].runtime, sizeof(IID)) != 0)
        {
            printf("FAIL IID_%s: header ", lrm_iids[i].name);
            print_guid(lrm_iids[i].header);
            printf(", runtime ");
            print_guid(&lrm_iids[i].runtime);
            printf("\n");
            failures++;
        }
    }

    printf("d3d12 ABI oracle: %d compile-time checks (%d C, %d C++) and %d IID checks passed against the SDK "
           "headers; %d failures\n",
           lrm_static_checks_c + lrm_static_checks_cpp(), lrm_static_checks_c, lrm_static_checks_cpp(),
           lrm_iid_count - failures, failures);
    return failures == 0 ? 0 : 1;
}
