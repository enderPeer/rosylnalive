// ABI oracle for Lrm.Gpu.Vulkan.Interop: what C++ can check that C cannot. The generated vk_sig.g.cpp states, for
// every unmanaged delegate the interop declares, the kind and size of its return value and each parameter as the
// .NET runtime sees them, and the kind of every bound struct field. It also checks the 64-bit flag bits (static
// const in C), the extension-name strings, the sType of every struct and every pNext extension that the runtime
// metadata claims, against vulkan_core.h and vulkan.hpp (Vulkan-Headers 1.4.365). Owner: R2 (VulkanInteropApi
// contract). Hand-written.
#pragma once

#include "lrm_vk_abi.h"
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <vulkan/vulkan.hpp>

namespace lrm
{
    // Kinds: 'p' pointer (data, function, or handle), 'i' integer or enum, 'f' floating point, 's' struct or union by
    // value, 'a' array, 'v' void. On the .NET side 'h' is a handle struct over one nativeint or uint64, which C
    // declares as a pointer: both travel in one general-purpose register on x64.
    struct k
    {
        char kind;
        std::size_t size;
    };

    template <class T> constexpr k of()
    {
        if constexpr (std::is_void_v<T>)
            return {'v', 0};
        else if constexpr (std::is_array_v<T>)
            return {'a', sizeof(T)};
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
        if (runtime.kind == 'h')
            return header.kind == 'p' && header.size == runtime.size;
        return header.kind == runtime.kind && header.size == runtime.size;
    }

    template <class T> using element = std::remove_all_extents_t<T>;
} // namespace lrm

#define LRM_MEMBER(T, f) decltype(((T *)nullptr)->f)

// The function type F returns `ret` and takes `n` parameters.
#define LRM_SIG_SHAPE(F, RK, RS, N)                                                                                   \
    static_assert(lrm::accepts(lrm::sig<F>::ret, lrm::k{RK, RS}), #F ": return " #RK #RS);                            \
    static_assert(lrm::sig<F>::arity == (N), #F ": " #N " parameters")
// Parameter `i` of F has the runtime kind and size.
#define LRM_SIG_ARG(F, I, K, S)                                                                                       \
    static_assert(lrm::accepts(lrm::sig<F>::args[I], lrm::k{K, S}), #F ": parameter " #I " is " #K #S)
// Field f of struct T has the runtime kind and size: an integer is not a float, a pointer not a struct.
#define LRM_FIELD_KIND(T, f, K, S)                                                                                    \
    static_assert(lrm::accepts(lrm::of<LRM_MEMBER(T, f)>(), lrm::k{K, S}), #T "." #f " is " #K #S)
// Array field f of struct T has N elements of the runtime element kind and size.
#define LRM_FIELD_ARRAY(T, f, K, S, N)                                                                                \
    static_assert(std::is_array_v<LRM_MEMBER(T, f)> && std::rank_v<LRM_MEMBER(T, f)> == 1 &&                         \
                      std::extent_v<LRM_MEMBER(T, f)> == (N) &&                                                       \
                      lrm::accepts(lrm::of<lrm::element<LRM_MEMBER(T, f)>>(), lrm::k{K, S}),                          \
                  #T "." #f " is " #K #S "[" #N "]")
// A 64-bit flag bit (static const VkFlags64 in vulkan_core.h) has the F# case's value.
#define LRM_FLAG64(e, v) static_assert((unsigned long long)(e) == (v##ULL), #e " == " #v)
// An extension-name constant has the F# literal's text.
#define LRM_STRING(c, s) static_assert(std::string_view(c) == std::string_view(s), #c " == " #s)
// The struct T, whose vulkan.hpp form is vk::C, has sType v (vulkan.hpp's structureType), and C::NativeType is T.
#define LRM_STYPE(T, C, v)                                                                                            \
    static_assert(std::is_same_v<VULKAN_HPP_NAMESPACE::C::NativeType, T>, #T " is vk::" #C);                         \
    static_assert(VULKAN_HPP_NAMESPACE::C::structureType == static_cast<VULKAN_HPP_NAMESPACE::StructureType>(v),     \
                  #T " has sType " #v)
// vk.xml lets struct C extend struct P (vulkan.hpp's StructExtends<C, P>), as the runtime metadata claims.
#define LRM_EXTENDS(C, P)                                                                                             \
    static_assert(VULKAN_HPP_NAMESPACE::StructExtends<VULKAN_HPP_NAMESPACE::C, VULKAN_HPP_NAMESPACE::P>::value,      \
                  #C " extends " #P)
// A type alias names the same C type.
#define LRM_ALIAS(A, T) static_assert(std::is_same_v<A, T>, #A " is " #T)
