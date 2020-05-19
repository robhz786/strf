//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

#include <strf/outbuf.hpp>
#include <cstring>   // for std::memcpy
#include <algorithm> // for std::copy_n
//#include <cwchar>  // for std::wcslen


#ifdef STRF_PREFER_STD_LIBRARY_STRING_FUNCTIONS
#    if defined(__cpp_lib_string_view)
#        define STRF_HAS_STD_STRING_VIEW
#        define STRF_CONSTEXPR_CHAR_TRAITS constexpr
#        include <string_view>
#    else
#        include <string> // char_traits

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

// --------------------------------------------------------------------------------
// Writting functions for outbuf
// --------------------------------------------------------------------------------

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#  if (__GNUC__ >= 7)
#    pragma GCC diagnostic ignored "-Wstringop-overflow"
#  endif
#endif

template <typename Outbuf, typename CharT>
STRF_HD void outbuf_write_continuation(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto space = ob.size();
    STRF_ASSERT(space < len);

    detail::str_copy_n(ob.pointer(), str, space);
    str += space;
    len -= space;
    ob.advance_to(ob.end());
    while (ob.good()) {
        ob.recycle();
        space = ob.size();
        if (len <= space) {
            memcpy(ob.pointer(), str, len * sizeof(CharT));
            ob.advance(len);
            break;
        }
        detail::str_copy_n(ob.pointer(), str, space);
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
STRF_HD void outbuf_write(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto p = ob.pointer();
    if (p + len <= ob.end()) { // the common case
        strf::detail::str_copy_n(p, str, len);
        ob.advance(len);
    } else {
        detail::outbuf_write_continuation<Outbuf, CharT>(ob, str, len);
    }
}

} // namespace detail

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuf<CharT>& ob
    , const CharT* str
    , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuf_noexcept<CharT>& ob
    , const CharT* str
    , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , const strf::underlying_char_type<CharSize>* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuf<CharT>& ob
    , const CharT* str
    , const CharT* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuf_noexcept<CharT>& ob
    , const CharT* str
    , const CharT* str_end ) noexcept
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

inline STRF_HD void write
    ( strf::basic_outbuf<char>& ob
    , const char* str )
{
    strf::detail::outbuf_write(ob, str, detail::str_length(str));
}

inline STRF_HD void write
    ( strf::basic_outbuf_noexcept<char>& ob
    , const char* str ) noexcept
{
    strf::detail::outbuf_write(ob, str, detail::str_length(str));
}

namespace detail {

template<std::size_t CharSize>
void STRF_HD write_fill_continuation
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;

    std::size_t space = ob.size();
    STRF_ASSERT(space < count);
    strf::detail::str_fill_n<char_type>(ob.pointer(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    ob.recycle();
    while (ob.good()) {
        space = ob.size();
        if (count <= space) {
            strf::detail::str_fill_n<char_type>(ob.pointer(), count, ch);
            ob.advance(count);
            break;
        }
        strf::detail::str_fill_n(ob.pointer(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    }
}

template <std::size_t CharSize>
inline STRF_HD void write_fill
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;
    if (count <= ob.size()) { // the common case
        strf::detail::str_fill_n<char_type>(ob.pointer(), count, ch);
        ob.advance(count);
    } else {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline STRF_HD void write_fill
    ( strf::basic_outbuf<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename strf::underlying_outbuf<sizeof(CharT)>::char_type;
    write_fill(ob.as_underlying(), count, static_cast<u_char_type>(ch));
}

} // namespace detail
} // namespace strf


#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
