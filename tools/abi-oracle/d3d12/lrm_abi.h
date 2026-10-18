/* ABI oracle for Lrm.Gpu.D3D12.Interop: the checks the generated d3d12_abi.g.c makes, as C11 static assertions
   over the Windows SDK headers. Owner: R3 (D3D12ComIdiom contract). Hand-written; d3d12_abi.g.c is generated. */
#ifndef LRM_ABI_H
#define LRM_ABI_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef LRM_INSTANTIATE_IIDS
#include <initguid.h>
#endif
#include <stddef.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>

#if !defined(__cplusplus)
#define LRM_STATIC_ASSERT(c, m) _Static_assert(c, m)
#else
#define LRM_STATIC_ASSERT(c, m) static_assert(c, m)
#endif

/* sizeof(T) of a bound struct equals the runtime size of its F# struct. */
#define LRM_SIZE(T, n) LRM_STATIC_ASSERT(sizeof(T) == (n), "sizeof(" #T ") == " #n)
/* offsetof(T, f) equals the runtime offset (Marshal.OffsetOf) of the F# field. */
#define LRM_OFFSET(T, f, n) LRM_STATIC_ASSERT(offsetof(T, f) == (n), "offsetof(" #T ", " #f ") == " #n)
/* sizeof(T::f) equals the runtime size of the F# field's type, so a field of the wrong width fails even where
   alignment keeps every offset and the struct size unchanged. */
#define LRM_FIELD_SIZE(T, f, n) LRM_STATIC_ASSERT(sizeof(((T *)0)->f) == (n), "sizeof(" #T "." #f ") == " #n)
/* A bound enum member has the value of the F# enum case. */
#define LRM_ENUM(e, v) LRM_STATIC_ASSERT((long long)(e) == (v), #e " == " #v)
/* Method m of interface I sits in vtable slot n (the generated Slots constant). */
#define LRM_SLOT(I, m, n) LRM_STATIC_ASSERT(offsetof(I##Vtbl, m) == (n) * sizeof(void *), #I "::" #m " is slot " #n)
/* Interface I has exactly n slots (Slots.I.Count). */
#define LRM_SLOTS(I, n) LRM_STATIC_ASSERT(sizeof(I##Vtbl) == (n) * sizeof(void *), #I " has " #n " slots")
/* A named HRESULT has the value of the F# literal. */
#define LRM_HRESULT(h, v) LRM_STATIC_ASSERT((h) == (HRESULT)(v), #h " == " #v)

/* One IID compared at run time: the header's IID_<name> against the F# Iids value. */
typedef struct lrm_iid_check
{
    const char *name;
    const IID *header;
    IID runtime;
} lrm_iid_check;

#endif
