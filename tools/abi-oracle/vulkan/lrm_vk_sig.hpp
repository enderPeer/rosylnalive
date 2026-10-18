// ABI oracle for Lrm.Gpu.Vulkan.Interop: what C++ can check that C cannot. The generated vk_sig.g.cpp states, for
// every unmanaged delegate the interop declares, the kind and size of its return value and each parameter as the
// .NET runtime sees them, and the kind of every bound struct field. Then the exact types: every field, return value
// and parameter is exactly the C type its runtime type stands for (a handle is that handle, a struct that struct, an
// enum that enum, an integer of that width and signedness, a nativeint a pointer that is no handle), and every
// wrapper parameter that is an inref, byref or span points to exactly its element type, const exactly for inref and
// ReadOnlySpan. It also checks the 64-bit flag bits (static const in C), the extension-name strings, the sType of
// every struct and every pNext extension that the runtime metadata claims, against vulkan_core.h and vulkan.hpp
// (Vulkan-Headers 1.4.365). Owner: R2 (VulkanInteropApi contract). Hand-written.
#pragma once

#include "lrm_vk_abi.h"
#include <cstddef>
#include <string_view>
#include <tuple>
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
        using ret_type = R;
        using arg_types = std::tuple<A...>;
    };

    constexpr bool accepts(k header, k runtime)
    {
        if (runtime.kind == 'h')
            return header.kind == 'p' && header.size == runtime.size;
        return header.kind == runtime.kind && header.size == runtime.size;
    }

    template <class T> using element = std::remove_all_extents_t<T>;

    // Handles. VK_DEFINE_HANDLE(X) and, on x64, VK_DEFINE_NON_DISPATCHABLE_HANDLE(X) both declare
    // `typedef struct X_T* X;`, and no header defines struct X_T. So a handle is exactly a pointer to a class type that
    // is incomplete here: a data pointer (const VkDeviceCreateInfo*, void*, const char*, VkDevice*) or a function
    // pointer (PFN_*) is no handle. The checks below hold the trait to that.
    template <class T, class = void> struct is_complete : std::false_type
    {
    };
    template <class T> struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type
    {
    };
    template <class T>
    constexpr bool is_handle_v = std::conjunction_v<std::is_pointer<T>, std::is_class<std::remove_pointer_t<T>>,
                                                    std::negation<is_complete<std::remove_pointer_t<T>>>>;

    static_assert(is_handle_v<VkInstance> && is_handle_v<VkDevice> && is_handle_v<VkBuffer>, "handles");
    static_assert(!is_handle_v<void *> && !is_handle_v<const char *> && !is_handle_v<PFN_vkVoidFunction>, "pointers");
    static_assert(!is_handle_v<const VkDeviceCreateInfo *> && !is_handle_v<VkDevice *>, "pointers to data");
    static_assert(!is_handle_v<uint64_t> && !is_handle_v<VkExtent3D> && !is_handle_v<VkResult>, "values");

    // Exact types. The C type is one of E: the struct, union, enum, handle or scalar the runtime type stands for. A
    // FlagBits enum stands for itself and for its Flags typedef (VkFlags or VkFlags64 underneath), and a byte for
    // uint8_t and char (and for void behind a span's pointer: `const void* pData` is a ReadOnlySpan<byte>).
    template <class... E> struct is
    {
        template <class T> static constexpr bool test = (std::is_same_v<T, E> || ...);
    };

    // What a runtime nativeint stands for: a data or function pointer, never a handle (a handle is a handle struct).
    struct pointer
    {
        template <class T> static constexpr bool test = std::is_pointer_v<T> && !is_handle_v<T>;
    };

    template <class F> using ret_t = typename sig<F>::ret_type;
    template <class F, std::size_t I> using arg_t = std::tuple_element_t<I, typename sig<F>::arg_types>;

    // A wrapper's inref<T>, byref<T>, ReadOnlySpan<T> or Span<T> parameter: a pointer (no handle) to exactly P's type,
    // const exactly when Const (inref and ReadOnlySpan).
    template <class T, bool Const, class P>
    constexpr bool points_to = std::is_pointer_v<T> && !is_handle_v<T> &&
                               std::is_const_v<std::remove_pointer_t<T>> == Const &&
                               P::template test<std::remove_cv_t<std::remove_pointer_t<T>>>;
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
// The exact types. The predicate comes last (a variadic argument) because lrm::is<A, B> holds a comma.
// Field f of struct T is exactly the runtime's type: lrm::is<C types> or lrm::pointer.
#define LRM_FIELD_TYPE(T, f, ...)                                                                                     \
    static_assert(__VA_ARGS__::test<LRM_MEMBER(T, f)>, #T "." #f " is " #__VA_ARGS__)
// The elements of array field f of struct T are exactly the runtime's element type.
#define LRM_FIELD_ELEMENT_TYPE(T, f, ...)                                                                             \
    static_assert(__VA_ARGS__::test<lrm::element<LRM_MEMBER(T, f)>>, #T "." #f " has elements " #__VA_ARGS__)
// The bound handle H is a handle in C (VK_DEFINE_HANDLE or VK_DEFINE_NON_DISPATCHABLE_HANDLE), not a mere pointer.
#define LRM_IS_HANDLE(H) static_assert(lrm::is_handle_v<H>, #H " is a handle")
// F returns exactly the runtime's return type.
#define LRM_SIG_RET_TYPE(F, ...) static_assert(__VA_ARGS__::test<lrm::ret_t<F>>, #F ": returns " #__VA_ARGS__)
// Parameter I of F is exactly the runtime's parameter type.
#define LRM_SIG_ARG_TYPE(F, I, ...)                                                                                   \
    static_assert(__VA_ARGS__::test<lrm::arg_t<F, I>>, #F ": parameter " #I " is " #__VA_ARGS__)
// The wrapper's parameter `name`, C parameter I of F, is an inref<T> or ReadOnlySpan<T>: a const T* with T exactly
// the runtime element type.
#define LRM_WRAP_CONST(F, I, name, ...)                                                                               \
    static_assert(lrm::points_to<lrm::arg_t<F, I>, true, __VA_ARGS__>,                                              \
                  #F ": wrapper parameter " #name " (" #I ") points to const " #__VA_ARGS__)
// The wrapper's parameter `name`, C parameter I of F, is a byref<T> or Span<T>: a T* (not const) with T exactly the
// runtime element type.
#define LRM_WRAP_MUT(F, I, name, ...)                                                                                 \
    static_assert(lrm::points_to<lrm::arg_t<F, I>, false, __VA_ARGS__>,                                             \
                  #F ": wrapper parameter " #name " (" #I ") points to mutable " #__VA_ARGS__)
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
