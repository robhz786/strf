//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

#include <strf/detail/common.hpp>
#  include <cstring> // for std::memcpy
#  include <algorithm> // for std::fill_n
#  ifdef STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
#    if defined(__cpp_lib_string_view)
#      define STRF_HAS_STD_STRING_VIEW
#      define STRF_CONSTEXPR_CHAR_TRAITS constexpr
#      include <string_view>
#    else
#      include <string> // char_traits
#    endif // defined(__cpp_lib_string_view)
#endif

#ifndef STRF_CONSTEXPR_CHAR_TRAITS
#  define STRF_CONSTEXPR_CHAR_TRAITS inline
#endif

namespace strf {

namespace detail {

template<class CharT, class Size, class T>
inline STRF_HD CharT*
str_fill_n(CharT* str, Size count, const T& value)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return std::char_traits<CharT>::assign(str, count, value);
#else
    // TODO: Should we consider CUDA's built-in memset?
    auto p = str;
    for (Size i = 0; i != count; ++i, ++p) {
        *p = value;
    }
    return p;
#endif
}

// std::char_traits<CharT>::length()
template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD std::size_t
str_length(const CharT* str)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
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
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
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
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return 0 == std::char_traits<CharT>::compare(str1, str2, count);
#elif defined(__CUDA_ARCH__)
    for(;count != 0; ++str1, ++str2, --count;) {
        if (*str1 != *str2) {
            return false;
        }
    }
    return true;
#endif
}


template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD CharT*
str_copy_n(CharT* destination, const CharT* source, std::size_t count)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return std::char_traits<CharT>::copy(destination, source, count);
#elif defined(__CUDA_ARCH__)
    // CUDA has a built-in device-side memcpy(); see:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
    // but it is not necessarily that fast, see:
    // https://stackoverflow.com/q/10456728/1593077
    auto result = memcpy(destination, source, count);
    return static_cast<CharT*>(result);
#else
    CharT* ret  = destination;
    for(;count != 0; ++destination, ++source, --count;) {
        *destination = *source;
    }
    return ret;
#endif
}

template <class InputIt, class Size, class OutputIt>
inline STRF_HD constexpr
OutputIt copy_n(InputIt first, Size count, OutputIt result)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return std::copy_n(first, count, result);
#else
    auto src_it { first };
    auto dest_it { result };
    for(; count != 0; ++src_it, ++dest_it, --count) {
        *dest_it = *src_it;
    }
    return result;
#endif
}

template <class T>
inline constexpr STRF_HD const T&
max(const T& lhs, const T& rhs) noexcept
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return std::max(lhs, rhs);
#elif __CUDA_ARCH__
    return (lhs < rhs) ? rhs : lhs;
#endif
}

} // namespace detail

} // namespace strf


#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
