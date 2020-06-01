#ifndef STRF_OUTBUFF_FUNCTIONS_HPP
#define STRF_OUTBUFF_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp>

namespace strf {
namespace detail {

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#  if (__GNUC__ >= 7)
#    pragma GCC diagnostic ignored "-Wstringop-overflow"
#  endif
#endif

template <typename Outbuff, typename CharT>
STRF_HD void outbuff_write_continuation(Outbuff& ob, const CharT* str, std::size_t len)
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
            detail::str_copy_n(ob.pointer(), str, len);
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

template <typename Outbuff, typename CharT = typename Outbuff::char_type>
STRF_HD void outbuff_write(Outbuff& ob, const CharT* str, std::size_t len)
{
    auto p = ob.pointer();
    if (p + len <= ob.end()) { // the common case
        strf::detail::str_copy_n(p, str, len);
        ob.advance(len);
    } else {
        detail::outbuff_write_continuation<Outbuff, CharT>(ob, str, len);
    }
}

} // namespace detail

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuff<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , std::size_t len )
{
    strf::detail::outbuff_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuff<CharT>& ob
    , const CharT* str
    , std::size_t len )
{
    strf::detail::outbuff_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuff_noexcept<CharT>& ob
    , const CharT* str
    , std::size_t len )
{
    strf::detail::outbuff_write(ob, str, len);
}

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuff<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , const strf::underlying_char_type<CharSize>* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuff_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuff<CharT>& ob
    , const CharT* str
    , const CharT* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuff_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write
    ( strf::basic_outbuff_noexcept<CharT>& ob
    , const CharT* str
    , const CharT* str_end ) noexcept
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuff_write(ob, str, str_end - str);
}

inline STRF_HD void write
    ( strf::basic_outbuff<char>& ob
    , const char* str )
{
    strf::detail::outbuff_write(ob, str, detail::str_length(str));
}

inline STRF_HD void write
    ( strf::basic_outbuff_noexcept<char>& ob
    , const char* str ) noexcept
{
    strf::detail::outbuff_write(ob, str, detail::str_length(str));
}

namespace detail {

template<std::size_t CharSize>
void STRF_HD write_fill_continuation
    ( strf::underlying_outbuff<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuff<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuff<CharSize>::char_type;

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
    ( strf::underlying_outbuff<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuff<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuff<CharSize>::char_type;
    if (count <= ob.size()) { // the common case
        strf::detail::str_fill_n<char_type>(ob.pointer(), count, ch);
        ob.advance(count);
    } else {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline STRF_HD void write_fill
    ( strf::basic_outbuff<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename strf::underlying_outbuff<sizeof(CharT)>::char_type;
    write_fill(ob.as_underlying(), count, static_cast<u_char_type>(ch));
}

} // namespace detail
} // namespace strf

#endif  // STRF_OUTBUFF_FUNCTIONS_HPP

