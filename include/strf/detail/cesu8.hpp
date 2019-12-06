#ifndef STRF_DETAIL_CESU8_HPP
#define STRF_DETAIL_CESU8_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/encoding.hpp>
#include <algorithm>
#include <cstring>

STRF_NAMESPACE_BEGIN

namespace detail {

constexpr bool is_surrogate(unsigned long codepoint)
{
    return codepoint >> 11 == 0x1B;
}
constexpr bool is_high_surrogate(unsigned long codepoint) noexcept
{
    return codepoint >> 10 == 0x36;
}
constexpr bool is_low_surrogate(unsigned long codepoint) noexcept
{
    return codepoint >> 10 == 0x37;
}
constexpr bool not_surrogate(unsigned long codepoint)
{
    return codepoint >> 11 != 0x1B;
}
constexpr  bool not_high_surrogate(unsigned long codepoint)
{
    return codepoint >> 10 != 0x36;
}
constexpr  bool not_low_surrogate(unsigned long codepoint)
{
    return codepoint >> 10 != 0x37;
}
constexpr unsigned utf8_decode(unsigned ch0, unsigned ch1)
{
    return (((ch0 & 0x1F) << 6) |
            ((ch1 & 0x3F) << 0));
}
constexpr unsigned utf8_decode(unsigned ch0, unsigned ch1, unsigned ch2)
{
    return (((ch0 & 0x0F) << 12) |
            ((ch1 & 0x3F) <<  6) |
            ((ch2 & 0x3F) <<  0));
}
constexpr unsigned utf8_decode(unsigned ch0, unsigned ch1, unsigned ch2, unsigned ch3)
{
    return (((ch0 & 0x07) << 18) |
            ((ch1 & 0x3F) << 12) |
            ((ch2 & 0x3F) <<  6) |
            ((ch3 & 0x3F) <<  0));
}
constexpr bool is_utf8_continuation(std::uint8_t ch)
{
    return (ch & 0xC0) == 0x80;
}

constexpr bool valid_start_3bytes
    ( std::uint8_t ch0
    , std::uint8_t ch1
    , bool allow_surr )
{
    return ( (ch0 != 0xE0 || ch1 != 0x80)
          && (allow_surr || (0x1B != (((ch0 & 0xF) << 1) | ((ch1 >> 5) & 1)))) );
}

inline unsigned utf8_decode_first_2_of_3(std::uint8_t ch0, std::uint8_t ch1)
{
    return ((ch0 & 0x0F) << 6) | (ch1 & 0x3F);
}

inline bool first_2_of_3_are_valid(unsigned x, bool allow_surr)
{
    return /*0x1F < x && */(allow_surr || (x >> 5) != 0x1B);
}
inline bool first_2_of_3_are_valid(std::uint8_t ch0,  std::uint8_t ch1, bool allow_surr)
{
    return first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), allow_surr);
}

inline unsigned utf8_decode_first_2_of_4(std::uint8_t ch0, std::uint8_t ch1)
{
    return ((ch0 & 0x07) << 6) | (ch1 & 0x3F);
}

inline unsigned utf8_decode_last_2_of_4(unsigned long x, unsigned ch2, unsigned ch3)
{
    return (x << 12) | ((ch2 & 0x3F) <<  6) | (ch3 & 0x3F);
}

inline bool first_2_of_4_are_valid(unsigned x)
{
    return 0xF < x && x < 0x110;
}

inline bool first_2_of_4_are_valid(unsigned ch0, unsigned ch1)
{
    return first_2_of_4_are_valid(utf8_decode_first_2_of_4(ch0, ch1));
}

STRF_STATIC_LINKAGE
strf::cv_result cesu8_to_utf32_transcode
    ( const std::uint8_t** src
    , const std::uint8_t* src_end
    , char32_t** dest
    , char32_t* dest_end
    , strf::encoding_error err_hdl
    , bool allow_surr )
{
    using strf::detail::utf8_decode;
    using strf::detail::is_utf8_continuation;

    std::uint8_t ch0, ch1, ch2, ch3;
    unsigned long x;
    auto dest_it = *dest;
    auto src_it = *src;

    for(;src_it != src_end; ++dest_it)
    {
        if(dest_it == dest_end)
        {
            goto insufficient_space;
        }

        ch0 = (*src_it);
        ++src_it;
        if (ch0 < 0x80)
        {
            *dest_it = ch0;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if(ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                *dest_it = utf8_decode(ch0, ch1);
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == ch0)
        {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                *dest_it = ((ch1 & 0x3F) << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if (src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                x = utf8_decode_first_2_of_3(ch0, ch1);
                if (x >> 4 == 0x36) // high surrogate
                {
                    if ( ++src_it != src_end
                      && is_utf8_continuation(ch2 = * src_it) )
                    {
                        x = (x << 6) | (ch2 & 0x3F);
                        if ( src_it + 3 < src_end
                          && src_it[1] == 0xED && (src_it[2] >> 4) == 0xB // low surrogate
                          && is_utf8_continuation(src_it[3]) )
                        {
                            *dest = 0x10000 +
                                  ( ((x & 0x3FF) << 10)
                                  | ((src_it[2] & 0xF) << 6)
                                  |  (src_it[3] & 0x3F) );
                            ++dest;
                            src_it += 4;
                        }
                        else if (allow_surr)
                        {
                            *dest = x;
                            ++src_it;
                        } else goto invalid_sequence;
                    } else goto invalid_sequence;
                }
                else if ( first_2_of_3_are_valid(x, allow_surr)
                       && ++src_it != src_end
                       && is_utf8_continuation(ch2 = * src_it) )
                {
                    *dest_it = (x << 6) | (ch2 & 0x3F);
                    ++src_it;
                } else goto invalid_sequence;
            } else goto invalid_sequence;
        }
        else if (0xEF < ch0)
        {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                *dest_it = utf8_decode_last_2_of_4(x, ch2, ch3);
                ++src_it;
            } else goto invalid_sequence;
        }
        else
        {
            invalid_sequence:
            switch (err_hdl)
            {
                case strf::encoding_error::replace:
                    *dest_it = 0xFFFD;
                    break;
                default:
                    STRF_ASSERT(err_hdl == strf::encoding_error::stop);
                    *src = src_it;
                    *dest = dest_it;
                    return strf::cv_result::invalid_char;
            }
        }
    }
    *src = src_it;
    *dest = dest_it;
    return strf::cv_result::success;

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return strf::cv_result::insufficient_space;
}

} // namespace detail

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_CESU8_HPP

