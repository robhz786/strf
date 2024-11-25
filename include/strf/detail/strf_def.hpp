#ifndef STRF_DETAIL_STRF_DEF_HPP
#define STRF_DETAIL_STRF_DEF_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <cstddef>
#include <cstdint>

#if ! defined(STRF_ASSERT)
#  if ! defined(STRF_FREESTANDING) && defined(__STDC_HOSTED__) && __STDC_HOSTED__ == 1
#    include <cassert>
#    define STRF_ASSERT(x) assert(x)
#  else
#    define STRF_ASSERT(x)
#  endif
#endif // ! defined(STRF_ASSERT)

#if defined(__CUDACC__)
#  undef STRF_SEPARATE_COMPILATION
#endif

#if defined(STRF_SOURCE) || (defined(STRF_CUDA_SEPARATE_COMPILATION) && defined(__CUDACC__))
#  if ! defined(STRF_SEPARATE_COMPILATION)
#     define STRF_SEPARATE_COMPILATION
#  endif
#endif

#if defined(STRF_SOURCE)
// When building static library
#  define STRF_FUNC_IMPL
#  define STRF_EXPLICIT_TEMPLATE template
#elif defined(STRF_SEPARATE_COMPILATION)
// When using static library
#  define STRF_OMIT_IMPL
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
#    define STRF_HAS_ATTR_FALLTHROUGH
#  endif
#elif defined(__has_cpp_attribute)
#  if __has_cpp_attribute(deprecated) && __cplusplus >= 201402L
#    define STRF_HAS_ATTR_DEPRECATED
#  endif
#  if __has_cpp_attribute(fallthrough) && __cplusplus >= 201703L
#    define STRF_HAS_ATTR_FALLTHROUGH
#  endif
#endif

#if defined(STRF_HAS_ATTR_DEPRECATED)
#  define STRF_DEPRECATED [[deprecated]]
#  define STRF_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#  define STRF_DEPRECATED
#  define STRF_DEPRECATED_MSG(msg)
#endif

#if defined(STRF_HAS_ATTR_FALLTHROUGH)
#  define STRF_FALLTHROUGH [[fallthrough]]
#else
#  define STRF_FALLTHROUGH
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
#  define STRF_FORCEINLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#  define STRF_FORCEINLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#  define STRF_FORCEINLINE __forceinline
#else
#  define STRF_FORCEINLINE inline
#endif

#ifdef __CUDACC__
#  define STRF_HOST    __host__
#  define STRF_DEVICE  __device__
#  define STRF_HD      __host__ __device__
#  define STRF_DEFAULT_IMPL {}
#else // __CUDACC__
#  define STRF_HD
#  define STRF_HOST
#  define STRF_DEVICE
#  define STRF_DEFAULT_IMPL = default
#endif // __CUDACC__

#ifdef __CUDA_ARCH__
#  define STRF_NO_GLOBAL_CONSTEXPR_VARIABLE
#endif

namespace strf { namespace detail {

template <typename T>
inline STRF_HD void pretend_to_use(const T& arg) noexcept { (void)arg; }

} // namespace detail
} // namespace strf

#define STRF_MAYBE_UNUSED(X) strf::detail::pretend_to_use((X));

namespace strf {

namespace detail
{

template <typename IntT>
constexpr STRF_HD IntT max(IntT a, IntT b)
{
    return a > b ? a : b;
}

template <typename IntT>
constexpr STRF_HD IntT min(IntT a, IntT b)
{
    return a < b ? a : b;
}

#if defined(__cpp_fold_expressions) && (!defined(_MSC_VER) || _MSC_VER >= 1921)

template <bool... C>
struct fold_and
{
    static constexpr bool value = (C && ...);
};

template <bool... C>
struct fold_or
{
    static constexpr bool value = (C || ...);
};

#else //defined(__cpp_fold_expressions)

template <bool ... > struct fold_and;
template <bool ... > struct fold_or;

template <> struct fold_and<>
{
    constexpr static bool value = true;
};

template <> struct fold_or<>
{
    constexpr static bool value = false;
};

template <bool C0, bool ... C>
struct fold_and<C0, C...>
{
     constexpr static bool value = fold_and<C...>::value && C0;
};

template <bool C0, bool ... C>
struct fold_or<C0, C...>
{
     constexpr static bool value = fold_or<C...>::value || C0;
};

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

template <typename... T> struct mp_type_list
{
    template <typename F>
    using add_front = mp_type_list<F, T...>;

    template <typename B>
    using add_back = mp_type_list<T..., B>;

    static constexpr std::size_t size = sizeof...(T);
};

template <typename A, typename B>
struct mp_type_pair
{
    using first = A;
    using second = B;
};

} // namespace detail

struct absolute_lowest_rank
{
    explicit constexpr absolute_lowest_rank() noexcept = default;
};

template <std::size_t N>
struct rank: rank<N - 1>
{
    constexpr static std::size_t number = N;
    explicit constexpr rank() noexcept = default;
};

template <>
struct rank<0>: absolute_lowest_rank
{
    constexpr static std::size_t number = 0;
    explicit constexpr rank() noexcept = default;
};

template <typename ... >
struct tag
{
    explicit constexpr tag() noexcept = default;
};

template <typename T>
struct tag<T>
{
    explicit constexpr tag() noexcept = default;
    using type = T;
};

template <typename> struct is_char: public std::false_type {};

#if defined(__cpp_char8_t)
template <> struct is_char<char8_t>: public std::true_type {};
#endif
template <> struct is_char<char>: public std::true_type {};
template <> struct is_char<char16_t>: public std::true_type {};
template <> struct is_char<char32_t>: public std::true_type {};
template <> struct is_char<wchar_t>: public std::true_type {};


namespace detail {

inline namespace cast_sugars {

template <typename T>
STRF_HD constexpr T zero_if_negative(T x)
{
    return x >= 0 ? x : 0;
}

template <typename T>
STRF_HD constexpr std::ptrdiff_t cast_ssize(T x)
{
    return static_cast<std::ptrdiff_t>(x);
}

template <typename T>
STRF_HD constexpr std::size_t safe_cast_size_t_(std::true_type, T x)
{
    return static_cast<std::size_t>(x >= 0 ? x : 0);
}
template <typename T>
STRF_HD constexpr std::size_t safe_cast_size_t_(std::false_type, T x)
{
    return x;
}
template <typename T>
STRF_HD constexpr std::size_t safe_cast_size_t(T x)
{
    return safe_cast_size_t_(std::is_signed<T>(), x);
}

template <typename IntT, typename UIntT = typename std::make_unsigned<IntT>::type>
constexpr STRF_HD UIntT cast_unsigned(IntT x)
{
    return static_cast<UIntT>(x);
}

template <typename T>
STRF_HD constexpr std::uint64_t cast_int(T x)
{
    return static_cast<int>(x);
}

template <typename T>
STRF_HD constexpr std::uint64_t cast_u64(T x)
{
    return static_cast<std::uint64_t>(x);
}

template <typename T>
STRF_HD constexpr std::int64_t  cast_i64(T x)
{
    return static_cast<std::int64_t>(x);
}

template <typename T>
STRF_HD constexpr std::uint32_t cast_u32(T x)
{
    return static_cast<std::uint32_t>(x);
}

template <typename T>
STRF_HD constexpr std::int32_t  cast_i32(T x)
{
    return static_cast<std::int32_t>(x);
}

template <typename T>
STRF_HD constexpr std::uint16_t cast_u16(T x)
{
    return static_cast<std::uint16_t>(x);
}

template <typename T>
STRF_HD constexpr std::int16_t  cast_i16(T x)
{
    return static_cast<std::int16_t>(x);
}

template <typename T>
STRF_HD constexpr std::uint8_t cast_u8(T x)
{
    return static_cast<std::uint8_t>(x);
}

template <typename T>
STRF_HD constexpr std::int8_t  cast_i8(T x)
{
    return static_cast<std::int8_t>(x);
}

}  // namespace cast_sugars


template <typename T>
STRF_HD constexpr bool ge_zero_(std::true_type, T x)
{
    return x >= 0;
}
template <typename T>
STRF_HD constexpr bool ge_zero_(std::false_type, T)
{
    return true;
}

template <typename T>
STRF_HD constexpr bool ge_zero(T x)
{
    return ge_zero_(std::is_signed<T>(), x);
}

} // namespace detail

} // namespace strf

#endif // STRF_DETAIL_STRF_DEF_HPP

