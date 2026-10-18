// ABI oracle for Lrm.Gpu.D3D12.Interop: call signatures and field kinds. The generated d3d12_sig.g.cpp states, for
// every unmanaged delegate and P/Invoke the interop declares, the kind and size of its return value and each
// parameter as the .NET runtime sees them, and the kind and size of every bound struct field. These templates derive
// the same facts from the SDK's own declarations (the C vtables with CINTERFACE, and the struct members) and
// static_assert that they agree. Owner: R3 (D3D12ComIdiom contract). Hand-written.
#pragma once

#include "lrm_abi.h"
#include <cstddef>
#include <type_traits>

namespace lrm
{
    // Kinds: 'p' pointer or reference, 'i' integer or enum, 'f' floating point, 's' struct by value, 'v' void.
    // On the .NET side 'w' is a pointer-sized integer (nativeint, unativeint or a pointer type), which stands for
    // a C pointer or a 64-bit integer: both travel in one general-purpose register on x64.
    struct k
    {
        char kind;
        std::size_t size;
    };

    template <class T> constexpr k of()
    {
        if constexpr (std::is_void_v<T>)
            return {'v', 0};
        else if constexpr (std::is_pointer_v<T> || std::is_reference_v<T>)
            return {'p', sizeof(void *)};
        else if constexpr (std::is_floating_point_v<T>)
            return {'f', sizeof(T)};
        else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>)
            return {'i', sizeof(T)};
        else
            return {'s', sizeof(T)};
    }

    template <class F> struct sig;

    template <class R, class... A> struct sig<R (*)(A...)>
    {
        static constexpr std::size_t arity = sizeof...(A);
        static constexpr k ret = of<R>();
        static constexpr k args[sizeof...(A) + 1] = {of<A>()..., {'v', 0}};
    };

    constexpr bool accepts(k header, k runtime)
    {
        if (runtime.kind == 'w')
            return header.size == sizeof(void *) && (header.kind == 'p' || header.kind == 'i');
        return header.kind == runtime.kind && header.size == runtime.size;
    }
} // namespace lrm

#define LRM_VFN(I, M) decltype(((I##Vtbl *)nullptr)->M)
#define LRM_XFN(F) decltype(&F)

// The function type F returns `ret` and takes `n` parameters (This included for vtable entries).
#define LRM_SIG_SHAPE(F, NAME, RK, RS, N)                                                                            \
    static_assert(lrm::accepts(lrm::sig<F>::ret, lrm::k{RK, RS}), NAME ": return " #RK #RS);                          \
    static_assert(lrm::sig<F>::arity == (N), NAME ": " #N " parameters")
// Parameter `i` of F has the runtime kind and size.
#define LRM_SIG_ARG(F, NAME, I, K, S)                                                                                \
    static_assert(lrm::accepts(lrm::sig<F>::args[I], lrm::k{K, S}), NAME ": parameter " #I " is " #K #S)
// Field f of struct T has the runtime kind and size: an integer is not a float, a pointer not a struct.
#define LRM_FIELD_KIND(T, f, K, S)                                                                                   \
    static_assert(lrm::accepts(lrm::of<decltype(((T *)nullptr)->f)>(), lrm::k{K, S}), #T "." #f " is " #K #S)
