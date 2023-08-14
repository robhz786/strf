#ifndef STRF_OUTPUT_BUFFER_FUNCTIONS_HPP
#define STRF_OUTPUT_BUFFER_FUNCTIONS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp>
#include <strf/destination.hpp>

namespace strf {
namespace detail {

template < typename DstCharT
         , typename SrcCharT
         , strf::detail::enable_if_t<!std::is_same<SrcCharT, DstCharT>::value, int> = 0 >
STRF_HD void output_buffer_interchar_copy
    ( strf::output_buffer<DstCharT, 0>& dst, const SrcCharT* str, const SrcCharT* str_end )
{
    if (str < str_end)
    {
        auto len = static_cast<std::size_t>(str_end - str);
        do {
            const auto space = dst.buffer_space();
            if (space >= len) {
                detail::copy_n(str, len, dst.buffer_ptr());
                dst.advance(len);
                return;
            }
            strf::detail::copy_n(str, space, dst.buffer_ptr());
            str += space;
            len -= space;
            dst.advance_to(dst.buffer_end());
            dst.flush();
        } while(dst.good());
    }
}

template < typename CharT >
inline STRF_HD void output_buffer_interchar_copy
    ( strf::output_buffer<CharT, 0>& dst, const CharT* str, const CharT* str_end )
{
    dst.write(str, str_end - str);
}

} // namespace detail


inline STRF_HD void write
    ( strf::output_buffer<char, 0>& dst
    , const char* str )
{
    dst.write(str, detail::str_ssize(str));
}

namespace detail {

template <typename CharT>
void STRF_HD write_fill_continuation
    ( strf::output_buffer<CharT, 0>& dst, std::ptrdiff_t count, CharT ch )
{
    auto space = dst.buffer_sspace();
    STRF_ASSERT(space < count);
    strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), space, ch);
    count -= space;
    dst.advance_to(dst.buffer_end());
    dst.flush();
    while (dst.good()) {
        space = dst.buffer_sspace();
        if (count <= space) {
            strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), count, ch);
            dst.advance(count);
            break;
        }
        strf::detail::str_fill_n(dst.buffer_ptr(), space, ch);
        count -= space;
        dst.advance_to(dst.buffer_end());
        dst.flush();
    }
}

template <typename CharT>
inline STRF_HD void write_fill
    ( strf::output_buffer<CharT, 0>& dst, std::ptrdiff_t count, CharT ch )
{
    STRF_IF_LIKELY (count > 0) {
        STRF_IF_LIKELY (count <= dst.buffer_sspace()) {
            strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), count, ch);
            dst.advance(count);
        } else {
            write_fill_continuation<CharT>(dst, count, ch);
        }
    }
}

} // namespace detail
} // namespace strf

#endif // STRF_OUTPUT_BUFFER_FUNCTIONS_HPP

