//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

#include <strf/detail/common.hpp>

#ifdef STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
#include <algorithm> // for std::fill_n
#include <cstring> // for std::memcpy
#include <string> // for std::char_traits
#endif

STRF_NAMESPACE_BEGIN

namespace detail {

template<class CharT, class Size, class T>
inline STRF_HD CharT*
str_fill_n(CharT* str, Size count, const T& value)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return std::fill_n(str, count, value);
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
inline STRF_CONSTEXPR_CHAR_TRAITS STRF_HD std::size_t
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
inline STRF_CONSTEXPR_CHAR_TRAITS STRF_HD CharT*
str_copy(CharT* destination, const CharT* source, std::size_t count)
{
#if !defined(__CUDA_ARCH__) && STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
    return static_cast<CharT*>(std::memcpy(destination, source, count));
#elif defined(__CUDA_ARCH__)
    // CUDA has a built-in device-side memcpy(); see:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
    // but it is not necessarily that fast, see:
    // https://stackoverflow.com/q/10456728/1593077
    auto result = memcpy(destination, source, count);
    return static_cast<CharT*>(result);
#else
    CharT* ret  = destination;
    for(;count != 0; ++destination, ++source) {
        *destination = *source;
    }
    return ret;
#endif
}

template <class CharT>
inline STRF_CONSTEXPR_CHAR_TRAITS STRF_HD void
char_assign(CharT& c1, const CharT& c2) noexcept
{
    c1 = c2;
}

template <class CharT>
inline STRF_CONSTEXPR_CHAR_TRAITS STRF_HD CharT*
char_assign(CharT* s, std::size_t n, CharT a)
{
    return str_fill_n<CharT, std::size_t, CharT>(s, a, n);
}


} // namespace detail

STRF_NAMESPACE_END


#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
