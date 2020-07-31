//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

#include <strf/detail/strf_def.hpp>

#if defined(__CUDA_ARCH__) && ! defined(STRF_FREESTANDING)
#define STRF_FREESTANDING
#endif

#include <type_traits>
#include <limits>
#include <new>
#include <utility>    // not freestanding, but almost
                      // std::declval, std::move, std::forward

#if ! defined(STRF_FREESTANDING)
#    define STRF_WITH_CSTRING
#    include <cstring>
#    include <algorithm>   // for std::copy_n
#    if defined(__cpp_lib_string_view)
#        define STRF_HAS_STD_STRING_VIEW
#        define STRF_HAS_STD_STRING_DECLARATION
#        define STRF_CONSTEXPR_CHAR_TRAITS constexpr
#        include <string_view> //for char_traits
#    else
#        define STRF_HAS_STD_STRING_DECLARATION
#        define STRF_HAS_STD_STRING_DEFINITION
#        include <string>      //for char_traits

#    endif // defined(__cpp_lib_string_view)
#endif

#if ! defined(STRF_CONSTEXPR_CHAR_TRAITS)
#    define STRF_CONSTEXPR_CHAR_TRAITS inline
#endif

#if defined(STRF_WITH_CSTRING)
#    include <cstring>
#endif

#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace strf {
namespace detail {

#ifdef __cpp_lib_bitops

template< class To, class From >
constexpr To STRF_HD bit_cast(const From& from) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "");
    return std::bit_cast<To, From>(from);
}

#else // __cpp_lib_bitops

template< class To, class From >
To STRF_HD bit_cast(const From& from) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "");

#if defined(STRF_WITH_CSTRING)
    To to;
    memcpy(&to, &from, sizeof(to));
    return to;
#else
    return *reinterpret_cast<const To*>(&from);
#endif
}

#endif //__cpp_lib_bitops

#if ! defined(STRF_FREESTANDING)
template <typename It>
using iterator_value_type = typename std::iterator_traits<It>::value_type;

#else

template <typename It>
struct iterator_value_type_impl
{
    using type = typename It::value_type;
    //std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<It>())>>;
};

template <typename T>
struct iterator_value_type_impl<T*>
{
    using type = std::remove_cv_t<T>;
};

template <typename It>
using iterator_value_type = typename iterator_value_type_impl<It>::type;

#endif

template <typename IntT>
constexpr IntT max(IntT a, IntT b)
{
    return a > b ? a : b;
}

template<class CharT>
STRF_CONSTEXPR_CHAR_TRAITS
STRF_HD void str_fill_n(CharT* str, std::size_t count, CharT value)
{
#if !defined(STRF_FREESTANDING)
    std::char_traits<CharT>::assign(str, count, value);
#else
    auto p = str;
    for (std::size_t i = 0; i != count; ++i, ++p) {
        *p = value;
    }
#endif
}

template<>
inline
STRF_HD void str_fill_n<char>(char* str, std::size_t count, char value)
{
#if !defined(STRF_FREESTANDING)
    std::char_traits<char>::assign(str, count, value);
#elif defined(STRF_WITH_CSTRING)
    memset(str, value, count);
#else
    auto p = str;
    for (std::size_t i = 0; i != count; ++i, ++p) {
        *p = value;
    }
#endif
}

template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD std::size_t
str_length(const CharT* str)
{
#if !defined(STRF_FREESTANDING)
    return std::char_traits<CharT>::length(str);
#else
    std::size_t length { 0 };
    while (*str != CharT{0}) { ++str, ++length; }
    return length;
#endif
}

template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD const CharT*
str_find(const CharT* p, std::size_t count, const CharT& ch) noexcept
{
#if !defined(STRF_FREESTANDING)
    return std::char_traits<CharT>::find(p, count, ch);
#else
    for (std::size_t i = 0; i != count; ++i, ++p) {
        if (*p == ch) {
            return p;
        }
    }
    return nullptr;
#endif
}


template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD bool
str_equal(const CharT* str1, const CharT* str2, std::size_t count)
{
#if !defined(STRF_FREESTANDING)
    return 0 == std::char_traits<CharT>::compare(str1, str2, count);
#else
    for(;count != 0; ++str1, ++str2, --count) {
        if (*str1 != *str2) {
            return false;
        }
    }
    return true;
#endif
}

template <bool SameCharSize, typename SrcCharT, typename DestCharT>
struct interchar_copy_n_impl;

template <typename CharT>
struct interchar_copy_n_impl<true, CharT, CharT>
{
    static inline STRF_HD void copy(const CharT* src, std::size_t count, CharT* dest)
    {
#if defined(STRF_WITH_CSTRING)
        memcpy(dest, src, count * sizeof(CharT));
#else
        for(;count != 0; ++dest, ++src, --count) {
            *dest = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DestCharT>
struct interchar_copy_n_impl<true, SrcCharT, DestCharT>
{
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    static inline STRF_HD void copy(const SrcCharT* src, std::size_t count, DestCharT* dest)
    {
#if defined(STRF_WITH_CSTRING)
        memcpy(dest, src, count * sizeof(SrcCharT));
#else
        for(;count != 0; ++dest, ++src, --count) {
            *dest = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DestCharT>
struct interchar_copy_n_impl<false, SrcCharT, DestCharT>
{
    static_assert(sizeof(SrcCharT) != sizeof(DestCharT), "");

    static inline STRF_HD void copy(const SrcCharT* src, std::size_t count, DestCharT* dest)
    {
#if !defined(STRF_FREESTANDING)
        std::copy_n(src, count, dest);
#else
        for(;count != 0; ++dest, ++src, --count) {
            *dest = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DestCharT>
inline STRF_HD void copy_n
    ( const SrcCharT* src
    , std::size_t count
    , DestCharT* dest )
{
    using impl = strf::detail::interchar_copy_n_impl
        < sizeof(SrcCharT) == sizeof(DestCharT), SrcCharT, DestCharT >;
    impl::copy(src, count, dest);
}

// template< class T >
// constexpr T&& forward( std::remove_reference_t<T>&& t ) noexcept
// {
//     return static_cast<T&&>(t);
// }


namespace detail_tag_invoke_ns {

STRF_HD inline void tag_invoke(){};

struct tag_invoke_fn
{
    template <typename Cpo, typename ... Args>
    constexpr STRF_HD auto operator()(Cpo cpo, Args&&... args) const
        noexcept(noexcept(tag_invoke(cpo, (Args&&)(args)...)))
        -> decltype(tag_invoke(cpo, (Args&&)(args)...))
    {
        return tag_invoke(cpo, (Args&&)(args)...);
    }
};

} // namespace detail_tag_invoke_ns

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename Cpo, typename ... Args>
constexpr STRF_HD auto tag_invoke(Cpo cpo, Args&&... args)
    noexcept(noexcept(tag_invoke(cpo, (Args&&)(args)...)))
    -> decltype(tag_invoke(cpo, (Args&&)(args)...))
{
    return tag_invoke(cpo, (Args&&)(args)...);
}

#else

constexpr strf::detail::detail_tag_invoke_ns::tag_invoke_fn tag_invoke {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

} // namespace detail
} // namespace strf

#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
