#ifndef STRF_DETAIL_COMMMON_HPP
#define STRF_DETAIL_COMMMON_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
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

#if defined(STRF_SOURCE)
// When building static library
#  define STRF_FUNC
#  define STRF_FUNC_IMPL
#  define STRF_EXPLICIT_TEMPLATE template
#elif defined(STRF_SEPARATE_COMPILATION)
// When using static library
#  define STRF_OMIT_IMPL
#  define STRF_FUNC
#  define STRF_EXPLICIT_TEMPLATE extern template
#else
// When using header-only library
#  define STRF_FUNC inline
#  define STRF_FUNC_IMPL inline
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

#if defined(__CUDACC__)
#  if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 3)
#    if defined(__cpp_variable_templates)
#      define STRF_HAS_VARIABLE_TEMPLATES
#    endif
#  endif
#elif defined(__cpp_variable_templates)
#  define STRF_HAS_VARIABLE_TEMPLATES
#endif

#if defined(__CUDACC__)
#  if (__CUDACC_VER_MAJOR__ >= 11)
#    define STRF_HAS_ATTR_DEPRECATED
#  endif
#elif defined(__has_cpp_attribute)
#  if __has_cpp_attribute(deprecated)
#    define STRF_HAS_ATTR_DEPRECATED
#  endif
#endif

#if defined(STRF_HAS_ATTR_DEPRECATED)
#  define STRF_DEPRECATED [[deprecated]]
#  define STRF_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#  define STRF_DEPRECATED
#  define STRF_DEPRECATED_MSG(msg)
#endif

#if defined(__GNUC__) || defined (__clang__)
#  define STRF_IF_LIKELY(x)   if(__builtin_expect(!!(x), 1))
#  define STRF_IF_UNLIKELY(x) if(__builtin_expect(!!(x), 0))

#elif defined(_MSC_VER) && (_MSC_VER >= 1926) && (_MSVC_LANG > 201703L)
#  define STRF_IF_LIKELY(x)   if(x)   [[likely]]
#  define STRF_IF_UNLIKELY(x) if(x) [[unlikely]]

#else
#  define STRF_IF_LIKELY(x)   if(x)
#  define STRF_IF_UNLIKELY(x) if(x)
#endif

// Define CUDA-related host/device execution scope specifiers/decorators

#if __cpp_constexpr >= 201304
#  define STRF_CONSTEXPR_IN_CXX14 constexpr
#  define STRF_ASSERT_IN_CONSTEXPR(X) STRF_ASSERT(X)
#else
#  define STRF_CONSTEXPR_IN_CXX14 inline
#  define STRF_ASSERT_IN_CONSTEXPR(X)
#endif

#if __cpp_decltype_auto >= 201304
#  define STRF_DECLTYPE_AUTO(X) decltype(auto)
#else
#  define STRF_DECLTYPE_AUTO(X) decltype((X))
#endif

//#define STRF_NOEXCEPT_NOEXCEPT(X) noexcept(noexcept(X))

#ifdef __CUDACC__

#define STRF_HOST    __host__
#define STRF_DEVICE  __device__
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

namespace strf { namespace detail {

template <typename T>
inline STRF_HD void pretend_to_use(const T& arg) noexcept { (void)arg; }

}}

#define STRF_MAYBE_UNUSED(X) strf::detail::pretend_to_use((X));

namespace strf {

namespace detail
{

#if defined(__cpp_fold_expressions)

template <bool... C>
constexpr STRF_HD bool fold_and() noexcept
{
    return (C && ...);
}

template <bool... C>
constexpr STRF_HD bool fold_or() noexcept
{
    return (C || ...);
}

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

template <bool ... C>
constexpr STRF_HD bool fold_and()
{
    return fold_and_impl<C...>::value;
}

template <bool ... C>
constexpr STRF_HD bool fold_or()
{
    return fold_or_impl<C...>::value;
}

#endif // defined(__cpp_fold_expressions)

template <std::size_t...> struct index_sequence
{
    using value_type = std::size_t;
};

namespace idxseq {

template <typename S>
struct increment_seq;

template <std::size_t... I>
struct increment_seq<index_sequence<I...> >
{
    using type = index_sequence<I..., sizeof...(I)>;
};
template <std::size_t N>
struct index_seq_maker
{
    using previous_seq = typename index_seq_maker<N - 1>::type;
    using type = typename increment_seq<previous_seq>::type;
};
template <>
struct index_seq_maker<0>
{
    using type = index_sequence<>;
};

} // namespace idxseq

template <std::size_t N>
using make_index_sequence = typename idxseq::index_seq_maker<N>::type;

template <typename T>
using remove_cvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B,T>::type;

template <bool B, typename TrueT, typename FalseT>
using conditional_t = typename std::conditional<B, TrueT, FalseT>::type;

template <typename T>
using make_signed_t = typename std::make_signed<T>::type;

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

