#ifndef STRF_DETAIL_COMMMON_HPP
#define STRF_DETAIL_COMMMON_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <cstddef>

#if ! defined(STRF_ASSERT)
#  if ! defined(STRF_FREESTANDING) && defined(__STDC_HOSTED__) && __STDC_HOSTED__ == 1
#    include <cassert>
#    define STRF_ASSERT(x) assert(x)
#  else
#    define STRF_ASSERT(x)
#  endif
#endif // ! defined(STRF_ASSERT)

#if defined(STRF_SOURCE) && !defined(STRF_SEPARATE_COMPILATION)
#define STRF_SEPARATE_COMPILATION
#endif

#if defined(STRF_SEPARATE_COMPILATION) && !defined(STRF_SOURCE)
#define STRF_OMIT_IMPL
#endif

#if defined(STRF_SOURCE)
#define STRF_EXPLICIT_TEMPLATE template
#elif defined(STRF_SEPARATE_COMPILATION)
#define STRF_EXPLICIT_TEMPLATE extern template
#endif

#if defined(STRF_SOURCE)
#define STRF_FUNC_IMPL
#elif !defined(STRF_SEPARATE_COMPILATION)
#define STRF_FUNC_IMPL inline
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define STRF_HAS_NODISCARD
#endif //__has_cpp_attribute(nodiscard)
#endif // defined(__has_cpp_attribute)

#if defined(STRF_HAS_NODISCARD)
#define STRF_NODISCARD [[nodiscard]]
#else
#define STRF_NODISCARD
#endif //defined(STRF_HAS_NODISCARD)

#if defined(__cpp_if_constexpr)
#define STRF_IF_CONSTEXPR if constexpr
#else
#define STRF_IF_CONSTEXPR if
#endif

// cxx17 guaranteed copy elision
#if defined(_MSC_VER)
#  if ((_MSC_VER < 1913) || (_MSVC_LANG < 201703))
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif defined(__clang__)
#  if (__clang_major__ < 4) || (__cplusplus < 201703L)
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif defined(__GNUC__)
#  if (__GNUC__ < 7) || (__cplusplus < 201703)
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif (__cplusplus < 201703)
#  define STRF_NO_CXX17_COPY_ELISION

#elif ! defined(__cpp_deduction_guides) || (__cpp_deduction_guides < 201703)
   // compilers that dont support Class template argument deductions
   // usually also dont support guaranteed copy elision
#  define STRF_NO_CXX17_COPY_ELISION
#endif

#if defined(__GNUC__) && (__cplusplus > 201703L) && !defined(__cpp_lib_bitopts)
// some versions of GCC forgot to define __cpp_lib_bitopts
#  define __cpp_lib_bitopts  	201907
#endif

// Define CUDA-related host/device execution scope specifiers/decorators

#ifdef __CUDACC__

#define STRF_HOST    __forceinline__ __host__
#define STRF_DEVICE  __forceinline__          __device__
#define STRF_FD      __forceinline__          __device__
#define STRF_FH      __forceinline__ __host__
#define STRF_FHD     __forceinline__ __host__ __device__
#define STRF_HD                      __host__ __device__

#else // __CUDACC__

#define STRF_FD inline
#define STRF_FH inline
#define STRF_FHD inline
#define STRF_HD
#define STRF_HOST
#define STRF_DEVICE

#endif // __CUDACC__

#ifdef __CUDA_ARCH__
#define STRF_NO_GLOBAL_CONSTEXPR_VARIABLE
#endif

namespace strf {

namespace detail
{

#if defined(__cpp_fold_expressions)

template <bool ... C> constexpr bool fold_and = (C && ...);
template <bool ... C> constexpr bool fold_or  = (C || ...);

#else //defined(__cpp_fold_expressions)

template <bool ... > struct fold_and_impl;
template <bool ... > struct fold_or_impl;

template <> struct fold_and_impl<>
{
    constexpr static bool value = true;
};

template <> struct fold_or_impl<>
{
    constexpr static bool value = false;
};

template <bool C0, bool ... C>
struct fold_and_impl<C0, C...>
{
     constexpr static bool value = fold_and_impl<C...>::value && C0;
};

template <bool C0, bool ... C>
struct fold_or_impl<C0, C...>
{
     constexpr static bool value = fold_or_impl<C...>::value || C0;
};

template <bool ... C> constexpr bool fold_and = fold_and_impl<C...>::value;
template <bool ... C> constexpr bool fold_or = fold_or_impl<C...>::value;

#endif // defined(__cpp_fold_expressions)

template <std::size_t CharSize>
struct wchar_equiv_impl;

template <> struct wchar_equiv_impl<2> { using type = char16_t; };
template <> struct wchar_equiv_impl<4> { using type = char32_t; };

using wchar_equiv = typename wchar_equiv_impl<sizeof(wchar_t)>::type;

} // namespace detail

struct absolute_lowest_rank
{
    explicit constexpr STRF_HD absolute_lowest_rank() noexcept { };
};

template <std::size_t N>
struct rank: rank<N - 1>
{
    explicit constexpr STRF_HD rank() noexcept { };
};

template <>
struct rank<0>: absolute_lowest_rank
{
    explicit constexpr STRF_HD rank() noexcept { }
};

template <typename ... >
struct tag
{
    explicit constexpr STRF_HD tag() noexcept { }
};

template <typename T>
struct tag<T>
{
    explicit constexpr STRF_HD tag() noexcept { }
    using type = T;
};

} // namespace strf

#endif  // STRF_DETAIL_COMMMON_HPP

