/* ABI oracle for Lrm.Gpu.Vulkan.Interop: the C11 checks that the generated vk_abi.g.c makes, as static assertions
   over the Vulkan-Headers 1.4.365 vulkan_core.h. Owner: R2 (VulkanInteropApi contract). Hand-written;
   vk_abi.g.c is generated from the runtime layout of the loaded interop assembly. */
#ifndef LRM_VK_ABI_H
#define LRM_VK_ABI_H

#include <stddef.h>
#include <vulkan/vulkan.h>

#if !defined(__cplusplus)
#define LRM_STATIC_ASSERT(c, m) _Static_assert(c, m)
#define LRM_ALIGNOF(T) _Alignof(T)
#else
#define LRM_STATIC_ASSERT(c, m) static_assert(c, m)
#define LRM_ALIGNOF(T) alignof(T)
#endif

/* The headers are the pinned release. */
#define LRM_HEADER_VERSION(n) LRM_STATIC_ASSERT(VK_HEADER_VERSION == (n), "VK_HEADER_VERSION == " #n)
/* sizeof(T) of a bound struct or union equals the runtime size of its F# struct. */
#define LRM_SIZE(T, n) LRM_STATIC_ASSERT(sizeof(T) == (n), "sizeof(" #T ") == " #n)
/* _Alignof(T) equals the alignment the runtime gives the F# struct (its offset after one byte in a struct). */
#define LRM_ALIGN(T, n) LRM_STATIC_ASSERT(LRM_ALIGNOF(T) == (n), "_Alignof(" #T ") == " #n)
/* offsetof(T, f) equals the runtime offset (Marshal.OffsetOf) of the F# field. */
#define LRM_OFFSET(T, f, n) LRM_STATIC_ASSERT(offsetof(T, f) == (n), "offsetof(" #T ", " #f ") == " #n)
/* sizeof(T::f) equals the runtime size of the F# field's type, so a field of the wrong width fails even where
   alignment keeps every offset and the struct size unchanged. */
#define LRM_FIELD_SIZE(T, f, n) LRM_STATIC_ASSERT(sizeof(((T *)0)->f) == (n), "sizeof(" #T "." #f ") == " #n)
/* A 32-bit enum member has the value of the F# enum case. */
#define LRM_ENUM(e, v) LRM_STATIC_ASSERT((long long)(e) == (v##LL), #e " == " #v)
/* A 32-bit FlagBits member has the value of the F# case, compared unsigned (bit 31 does not fit an int). */
#define LRM_FLAG(e, v) LRM_STATIC_ASSERT((unsigned long long)(unsigned int)(e) == (v##ULL), #e " == " #v)
/* A 32-bit enum or FlagBits type is 4 bytes, as its F# enum. */
#define LRM_ENUM_SIZE(T, n) LRM_STATIC_ASSERT(sizeof(T) == (n), "sizeof(" #T ") == " #n)
/* A handle is pointer-sized: both handle kinds are pointers on x64. */
#define LRM_HANDLE(T, n) LRM_STATIC_ASSERT(sizeof(T) == (n), "sizeof(" #T ") == " #n)
/* An integer API or extension constant has the F# literal's value. */
#define LRM_CONST(c, v) LRM_STATIC_ASSERT((unsigned long long)(c) == (v##ULL), #c " == " #v)

#endif
