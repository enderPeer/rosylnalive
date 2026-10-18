/* ABI oracle for Lrm.Gpu.Vulkan.Interop: reaching main means every compile-time check in vk_abi.g.c (C11) and
   vk_sig.g.cpp (C++17) passed against Vulkan-Headers. Owner: R2. Hand-written. */
#include "lrm_vk_abi.h"
#include <stdio.h>

extern const int lrm_static_checks_c;
extern int lrm_static_checks_cpp(void);

int main(void)
{
    printf("vulkan ABI oracle: %d compile-time checks (%d C, %d C++) passed against Vulkan-Headers %u.%u.%u; "
           "0 failures\n",
           lrm_static_checks_c + lrm_static_checks_cpp(), lrm_static_checks_c, lrm_static_checks_cpp(),
           VK_API_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE), VK_API_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE),
           VK_API_VERSION_PATCH(VK_HEADER_VERSION_COMPLETE));
    return 0;
}
