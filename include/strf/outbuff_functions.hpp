#ifndef STRF_OUTBUFF_FUNCTIONS_HPP
#define STRF_OUTBUFF_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp>
#include <strf/outbuff.hpp>

namespace strf {
namespace detail {

template < typename DestCharT
         , typename SrcCharT
         , std::enable_if_t<!std::is_same<SrcCharT, DestCharT>::value, int> = 0 >
STRF_HD void outbuff_interchar_copy
    ( strf::basic_outbuff<DestCharT>& ob, const SrcCharT* str, std::size_t len )
{
    do {
        std::size_t space = ob.space();
        if (space >= len) {
            detail::copy_n(str, len, ob.pointer());
            ob.advance(len);
            return;
        }
        strf::detail::copy_n(str, space, ob.pointer());
        str += space;
        len -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    } while(ob.good());
}

template < typename CharT >
inline STRF_HD void outbuff_interchar_copy
    ( strf::basic_outbuff<CharT>& ob, const CharT* str, std::size_t len )
{
    ob.write(str, len);
}

} // namespace detail


inline STRF_HD void write
    ( strf::basic_outbuff<char>& ob
    , const char* str )
{
    ob.write(str, detail::str_length(str));
}

inline STRF_HD void write
    ( strf::basic_outbuff_noexcept<char>& ob
    , const char* str ) noexcept
{
    ob.write(str, detail::str_length(str));
}

namespace detail {

template <typename CharT>
void STRF_HD write_fill_continuation
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, CharT ch )
{
    std::size_t space = ob.space();
    STRF_ASSERT(space < count);
    strf::detail::str_fill_n<CharT>(ob.pointer(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    ob.recycle();
    while (ob.good()) {
        space = ob.space();
        if (count <= space) {
            strf::detail::str_fill_n<CharT>(ob.pointer(), count, ch);
            ob.advance(count);
            break;
        }
        strf::detail::str_fill_n(ob.pointer(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    }
}

template <typename CharT>
inline STRF_HD void write_fill
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, CharT ch )
{
    STRF_IF_LIKELY (count <= ob.space()) {
        strf::detail::str_fill_n<CharT>(ob.pointer(), count, ch);
        ob.advance(count);
    } else {
        write_fill_continuation<CharT>(ob, count, ch);
    }
}

} // namespace detail
} // namespace strf

#endif  // STRF_OUTBUFF_FUNCTIONS_HPP

