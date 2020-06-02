//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

#include <strf/outbuff.hpp>

#if defined(__CUDA_ARCH__) && ! defined(STRF_FREESTANDING)
#define STRF_FREESTANDING
#endif

#include <type_traits>
#include <utility>    // not freestanding, but almost
#include <limits>

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


namespace strf {
namespace detail {

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


template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD void
str_copy_n(CharT* destination, const CharT* source, std::size_t count)
{
#if ! defined(STRF_FREESTANDING)
    std::char_traits<CharT>::copy(destination, source, count);
#elif defined(STRF_WITH_CSTRING)
    memcpy(destination, source, count * sizeof(CharT));
#else
    for(;count != 0; ++destination, ++source, --count) {
        *destination = *source;
    }
#endif
}

template <class InputIt, class Size, class OutputIt>
inline STRF_HD void copy_n(InputIt src_it, Size count, OutputIt dest_it)
{
#if !defined(STRF_FREESTANDING)
    std::copy_n(src_it, count, dest_it);
#else
    for(; count != 0; ++src_it, ++dest_it, --count) {
        *dest_it = *src_it;
    }
#endif
}

} // namespace detail
} // namespace strf

#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
