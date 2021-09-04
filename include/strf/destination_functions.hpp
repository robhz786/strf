#ifndef STRF_OUTBUFF_FUNCTIONS_HPP
#define STRF_OUTBUFF_FUNCTIONS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp>
#include <strf/destination.hpp>

namespace strf {
namespace detail {

template < typename DestCharT
         , typename SrcCharT
         , strf::detail::enable_if_t<!std::is_same<SrcCharT, DestCharT>::value, int> = 0 >
STRF_HD void destination_interchar_copy
    ( strf::destination<DestCharT>& dest, const SrcCharT* str, std::size_t len )
{
    do {
        std::size_t space = dest.space();
        if (space >= len) {
            detail::copy_n(str, len, dest.pointer());
            dest.advance(len);
            return;
        }
        strf::detail::copy_n(str, space, dest.pointer());
        str += space;
        len -= space;
        dest.advance_to(dest.end());
        dest.recycle();
    } while(dest.good());
}

template < typename CharT >
inline STRF_HD void destination_interchar_copy
    ( strf::destination<CharT>& dest, const CharT* str, std::size_t len )
{
    dest.write(str, len);
}

} // namespace detail


inline STRF_HD void write
    ( strf::destination<char>& dest
    , const char* str )
{
    dest.write(str, detail::str_length(str));
}

namespace detail {

template <typename CharT>
void STRF_HD write_fill_continuation
    ( strf::destination<CharT>& dest, std::size_t count, CharT ch )
{
    std::size_t space = dest.space();
    STRF_ASSERT(space < count);
    strf::detail::str_fill_n<CharT>(dest.pointer(), space, ch);
    count -= space;
    dest.advance_to(dest.end());
    dest.recycle();
    while (dest.good()) {
        space = dest.space();
        if (count <= space) {
            strf::detail::str_fill_n<CharT>(dest.pointer(), count, ch);
            dest.advance(count);
            break;
        }
        strf::detail::str_fill_n(dest.pointer(), space, ch);
        count -= space;
        dest.advance_to(dest.end());
        dest.recycle();
    }
}

template <typename CharT>
inline STRF_HD void write_fill
    ( strf::destination<CharT>& dest, std::size_t count, CharT ch )
{
    STRF_IF_LIKELY (count <= dest.space()) {
        strf::detail::str_fill_n<CharT>(dest.pointer(), count, ch);
        dest.advance(count);
    } else {
        write_fill_continuation<CharT>(dest, count, ch);
    }
}

} // namespace detail
} // namespace strf

#endif  // STRF_OUTBUFF_FUNCTIONS_HPP

