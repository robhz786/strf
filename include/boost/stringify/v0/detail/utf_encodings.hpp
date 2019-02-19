#ifndef BOOST_STRINGIFY_V0_DETAIL_UTF_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_UTF_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/transcoding.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <cstring>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

constexpr bool is_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 11 == 0x1B;
}
constexpr bool is_high_surrogate(std::uint32_t codepoint) noexcept
{
    return codepoint >> 10 == 0x36;
}
constexpr bool is_low_surrogate(std::uint32_t codepoint) noexcept
{
    return codepoint >> 10 == 0x37;
}
constexpr bool not_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 11 != 0x1B;
}
constexpr  bool not_high_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 10 != 0x36;
}
constexpr  bool not_low_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 10 != 0x37;
}
constexpr std::uint16_t utf8_decode(std::uint16_t ch0, std::uint16_t ch1)
{
    return (((ch0 & 0x1F) << 6) |
            ((ch1 & 0x3F) << 0));
}
constexpr std::uint16_t utf8_decode(std::uint16_t ch0, std::uint16_t ch1, std::uint16_t ch2)
{
    return (((ch0 & 0x0F) << 12) |
            ((ch1 & 0x3F) <<  6) |
            ((ch2 & 0x3F) <<  0));
}
constexpr std::uint32_t utf8_decode(std::uint32_t ch0, std::uint32_t ch1, std::uint32_t ch2, std::uint32_t ch3)
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

inline bool first_2_of_4_are_valid(std::uint8_t ch0, std::uint8_t ch1)
{
    return first_2_of_4_are_valid(utf8_decode_first_2_of_4(ch0, ch1));
}

BOOST_STRINGIFY_STATIC_LINKAGE
stringify::v0::cv_result utf8_to_utf32_transcode
    ( const std::uint8_t** src
    , const std::uint8_t* src_end
    , char32_t** dest
    , char32_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;
    using stringify::v0::detail::is_utf8_continuation;

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
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                       , allow_surr )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                *dest_it = (x << 6) | (ch2 & 0x3F);
                ++src_it;
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
                case stringify::v0::error_handling::stop:
                    *src = src_it;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    --dest_it;
                    break;
            }
        }
    }
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_to_utf32_size
    ( const std::uint8_t* src
    , const std::uint8_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::uint8_t ch0, ch1, ch2;
    const std::uint8_t* src_it = src;
    std::size_t size = 0;
    while (src_it != src_end)
    {
        ch0 = (*src_it);
        ++src_it;
        if (ch0 < 0x80)
        {
            ++size;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it))
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == ch0)
        {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( ch0, ch1, allow_surr )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if(0xEF < ch0)
        {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else
        {
            invalid_sequence:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return size;
                case stringify::v0::error_handling::replace:
                    ++size;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return size;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf8_sanitize
    ( const std::uint8_t** src
    , const std::uint8_t* src_end
    , std::uint8_t** dest
    , std::uint8_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;

    std::uint8_t ch0, ch1, ch2, ch3;
    auto dest_it = *dest;
    auto src_it = *src;
    const std::uint8_t* previous_src_it;

    while(src_it != src_end)
    {
        previous_src_it = src_it;
        ch0 = (*src_it);
        ++src_it;
        if(ch0 < 0x80)
        {
            if(dest_it != dest_end)
            {
                *dest_it = ch0;
                ++dest_it;
            } else goto insufficient_space;
        }
        else if(0xC0 == (ch0 & 0xE0))
        {
            if(ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                if (dest_it + 1 < dest_end)
                {
                    ++src_it;
                    dest_it[0] = ch0;
                    dest_it[1] = ch1;
                    dest_it += 2;
                } else goto insufficient_space;
            } else goto invalid_sequence;
        }
        else if (0xE0 == ch0)
        {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                if (dest_it + 2 < dest_end)
                {
                    ++src_it;
                    dest_it[0] = ch0;
                    dest_it[1] = ch1;
                    dest_it[2] = ch2;
                    dest_it += 3;
                }
            } else goto invalid_sequence;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid(ch0, ch1, allow_surr)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                if (dest_it + 2 < dest_end)
                {
                    ++src_it;
                    dest_it[0] = ch0;
                    dest_it[1] = ch1;
                    dest_it[2] = ch2;
                    dest_it += 3;
                } else goto insufficient_space;
            } else goto invalid_sequence;
        }
        else if (0xF0 == (ch0 & 0xF8))
        {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                if (dest_it + 4 < dest_end)
                {
                    ++src_it;
                    dest_it[0] = ch0;
                    dest_it[1] = ch1;
                    dest_it[2] = ch2;
                    dest_it[3] = ch3;
                    dest_it += 4;
                } else goto insufficient_space;
            } else goto invalid_sequence;
        }
        else
        {
            invalid_sequence:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    if (dest_it + 2 < dest_end)
                    {
                        dest_it[0] = 0xEF;
                        dest_it[1] = 0xBF;
                        dest_it[2] = 0xBD;
                        dest_it += 3;
                        break;
                    } else goto insufficient_space;

                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    *dest = dest_it;
    *src = src_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *dest = dest_it;
    *src = previous_src_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_sanitize_size
    ( const std::uint8_t* src
    , const std::uint8_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;
    std::uint8_t ch0, ch1;
    const std::uint8_t* src_it = src;
    std::size_t size = 0;
    while(src_it != src_end)
    {
        ch0 = *src_it;
        ++src_it;
        if(ch0 < 0x80)
        {
            ++size;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it))
            {
                size += 2;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == ch0)
        {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                size += 3;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( ch0, ch1, allow_surr )
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                size += 3;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if(0xEF < ch0)
        {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                size += 4;
                ++src_it;
            } else goto invalid_sequence;
        }
        else
        {
            invalid_sequence:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return size;
                case stringify::v0::error_handling::replace:
                    size += 3;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return size;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_codepoints_count
        ( const std::uint8_t* begin
        , const std::uint8_t* end
        , std::size_t max_count )
{
    std::size_t count = 0;
    for(auto it = begin; it != end && count < max_count; ++it)
    {
        if (!is_utf8_continuation(*it))
        {
            ++ count;
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf8_encode_fill
    ( std::uint8_t** dest
    , std::uint8_t* end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::uint8_t* dest_it = *dest;
    const std::size_t count_ = count;
    const std::size_t available = end - dest_it;
    std::size_t minc;
    if (ch < 0x80)
    {
        minc = (std::min)(count_, available);
        std::memset(dest_it, static_cast<std::uint8_t>(ch), minc);
        dest_it += minc;
    }
    else if (ch < 0x800)
    {
        minc = (std::min)(count_, available / 2);
        std::uint8_t ch0 = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
        std::uint8_t ch1 = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
        for(std::size_t i = 0; i < minc; ++i)
        {
            dest_it[0] = ch0;
            dest_it[1] = ch1;
            dest_it += 2;
        }
    }
    else if (ch <  0x10000)
    {
        if ( ! allow_surr && detail::is_surrogate(ch))
        {
            goto invalid_char;
        }
        minc = (std::min)(count_, available / 3);
        std::uint8_t ch0 = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
        std::uint8_t ch1 = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
        std::uint8_t ch2 = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
        for(std::size_t i = 0; i < minc; ++i)
        {
            dest_it[0] = ch0;
            dest_it[1] = ch1;
            dest_it[2] = ch2;
            dest_it += 3;
        }
    }
    else if (ch < 0x110000)
    {
        minc = (std::min)(count_, available / 4);
        std::uint8_t ch0 = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
        std::uint8_t ch1 = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
        std::uint8_t ch2 = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
        std::uint8_t ch3 = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
        for(std::size_t i = 0; i < minc; ++i)
        {
            dest_it[0] = ch0;
            dest_it[1] = ch1;
            dest_it[2] = ch2;
            dest_it[3] = ch3;
            dest_it += 4;
        }
    }
    else
    {
        invalid_char:
        switch(err_hdl)
        {
            case stringify::v0::error_handling::stop:
                return stringify::v0::cv_result::invalid_char;

            case stringify::v0::error_handling::ignore:
                count = 0;
                return stringify::v0::cv_result::success;

            default:
                BOOST_ASSERT(err_hdl == stringify::v0::error_handling::replace);
                minc = (std::min)(count_, available / 3);
                for(std::size_t i = 0; i < minc; ++i)
                {
                    dest_it[0] = 0xEF;
                    dest_it[1] = 0xBF;
                    dest_it[2] = 0xBD;
                    dest_it += 3;
                }
        }
    }
    *dest = dest_it;
    if(minc == count_)
    {
        count = 0;
        return stringify::v0::cv_result::success;
    }
    count = count_ - minc;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf8_encode_char
    ( std::uint8_t** dest
    , std::uint8_t* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::uint8_t* dest_it = *dest;
    if (ch < 0x80 && dest_it != end)
    {
        *dest_it = static_cast<std::uint8_t>(ch);
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    std::size_t dest_size = end - dest_it;
    if (ch < 0x800 && 2 <= dest_size)
    {
        dest_it[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
        dest_it[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
        *dest = dest_it + 2;
        return stringify::v0::cv_result::success;
    }
    if (ch <  0x10000 && 3 <= dest_size)
    {
        if ( ! allow_surr && detail::is_surrogate(ch))
        {
            goto invalid_char;
        }
        dest_it[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
        dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
        dest_it[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
        *dest = dest_it + 3;
        return stringify::v0::cv_result::success;
    }
    if (ch < 0x110000)
    {
        if (4 <= dest_size)
        {
            dest_it[0] = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
            dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
            dest_it[2] = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
            dest_it[3] = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
            *dest = dest_it + 4;
            return stringify::v0::cv_result::success;
        }
        return stringify::v0::cv_result::insufficient_space;
    }
    invalid_char:
    switch (err_hdl)
    {
        case stringify::v0::error_handling::replace:
            if (3 <= dest_size)
            {
                dest_it[0] = 0xEF;
                dest_it[1] = 0xBF;
                dest_it[2] = 0xBD;
                *dest = dest_it + 3;
                return stringify::v0::cv_result::success;
            }
            return stringify::v0::cv_result::insufficient_space;
        case stringify::v0::error_handling::ignore:
            return stringify::v0::cv_result::success;
        default:
            BOOST_ASSERT(err_hdl == stringify::v0::error_handling::stop);
            return stringify::v0::cv_result::invalid_char;
    }
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf32_to_utf8_transcode
    ( const char32_t** src
    , const char32_t* src_end
    , std::uint8_t** dest
    , std::uint8_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto src_it = *src;
    auto dest_it = *dest;
    std::size_t available_space = dest_end - dest_it;
    for(;src_it != src_end; ++src_it)
    {
        auto ch = *src_it;
        if(ch < 0x80)
        {
            if(available_space != 0)
            {
                *dest_it = static_cast<std::uint8_t>(ch);
                ++dest_it;
                --available_space;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x800)
        {
            if(available_space >= 2)
            {
                dest_it[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
                dest_it[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
                dest_it += 2;
                available_space -= 2;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x10000)
        {
            if(allow_surr || stringify::v0::detail::not_surrogate(ch))
            {
                if (available_space >= 3)
                {
                    dest_it[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
                    dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
                    dest_it[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
                    dest_it += 3;
                    available_space -= 3;
                } else goto insufficient_space;
            }
            else goto invalid_sequence;
        }
        else if (ch < 0x110000)
        {
            if(available_space >= 4)
            {
                dest_it[0] = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
                dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
                dest_it[2] = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
                dest_it[3] = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
                dest_it += 4;
                available_space -= 4;
            }
            else goto insufficient_space;
        }
        else
        {
            invalid_sequence:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it + 1;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    if (available_space >= 3)
                    {
                        dest_it[0] = 0xEF;
                        dest_it[1] = 0xBF;
                        dest_it[2] = 0xBD;
                        dest_it += 3;
                        available_space -=3;
                        break;
                    } else goto insufficient_space;

                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    *dest = dest_it;
    *src = src_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *dest = dest_it;
    *src = src_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf32_to_utf8_size
    ( const char32_t* src
    , const char32_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto src_it = src;
    std::size_t count = 0;
    for(;src_it != src_end; ++src_it)
    {
        auto ch = *src_it;
        if(ch < 0x80)
        {
            ++count;
        }
        else if (ch < 0x800)
        {
            count += 2;
        }
        else if (ch < 0x10000)
        {
            if(allow_surr || stringify::v0::detail::not_surrogate(ch))
            {
                count += 3;
            }
            else goto invalid_char;
        }
        else if (ch < 0x110000)
        {
            count += 4;
        }
        else
        {
            invalid_char:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return count;

                case stringify::v0::error_handling::replace:
                    count += 3;
                    break;

                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE bool utf8_write_replacement_char
    ( std::uint8_t** dest
    , std::uint8_t* dest_end )
{
    auto dest_it = *dest;
    if (dest_it + 2 < dest_end)
    {
        dest_it[0] = 0xEF;
        dest_it[1] = 0xBF;
        dest_it[2] = 0xBD;
        *dest = dest_it + 3;
        return true;
    }
    return false;
}

BOOST_STRINGIFY_STATIC_LINKAGE char32_t utf8_decode_single_char(std::uint8_t ch)
{
    const std::uint8_t uch = ch;
    return uch < 0x80 ? static_cast<char32_t>(uch) : static_cast<char32_t>(-1);
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_validate(char32_t ch)
{
    return ( ch < 0x80     ? 1 :
             ch < 0x800    ? 2 :
             ch < 0x10000  ? 3 :
             ch < 0x110000 ? 4 : (std::size_t)-1 );
}

// BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_encode_char
//     ( std::uint8_t** dest
//     , std::uint8_t* end
//     , char32_t ch
//     , stringify::v0::error_handling err_hdl )
// {
//     if (ch != 0)
//     {
//         return stringify::v0::detail::utf8_encode_char(dest, end, ch, err_hdl);
//     }
//     auto dest_it = *dest;
//     if (dest_it + 1 < end)
//     {
//         dest_it[0] = '\xC0';
//         dest_it[1] = '\x80';
//         *dest = dest_it + 2;
//         return stringify::v0::cv_result::success;
//     }
//     return stringify::v0::cv_result::insufficient_space;
// }


// BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_encode_fill
//     ( std::uint8_t** dest
//     , std::uint8_t* end
//     , std::size_t& count
//     , char32_t ch
//     , stringify::v0::error_handling err_hdl )
// {
//     if (ch != 0)
//     {
//         return stringify::v0::detail::utf8_encode_fill(dest, end, count, ch, err_hdl);
//     }
//     auto dest_it = *dest;
//     std::size_t available = (end - dest_it) / 2;
//     std::size_t c = count;
//     std::size_t minc = std::min(count, available);
//     auto it = std::fill_n( reinterpret_cast<std::pair<std::uint8_t, std::uint8_t>*>(dest_it)
//                          , minc
//                          , std::pair<std::uint8_t, std::uint8_t>{'\xC0', '\x80'});
//     *dest = reinterpret_cast<std::uint8_t*>(it);
//     count = c - minc;
//     return stringify::v0::cv_result::insufficient_space;
// }

// BOOST_STRINGIFY_STATIC_LINKAGE std::size_t mutf8_validate(char32_t ch)
// {
//     return (ch ==  0 ? 2 :
//             ch < 0x80 ? 1 :
//             ch < 0x800 ? 2 :
//             ch < 0x10000 ? 3 :
//             ch < 0x110000 ? 4 : (std::size_t)-1);
// }


BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf16_to_utf32_transcode
    ( const char16_t** src
    , const char16_t* src_end
    , char32_t** dest
    , char32_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    auto dest_it = *dest;
    const char16_t* src_it_next;
    for(auto src_it = *src; src_it != src_end; src_it = src_it_next)
    {
        src_it_next = src_it + 1;
        ch = *src_it;
        src_it_next = src_it + 1;
        if (dest_it == dest_end)
        {
            *src = src_it;
            *dest = dest_it;
            return stringify::v0::cv_result::insufficient_space;
        }
        if (not_surrogate(ch))
        {
            *dest_it = ch;
            ++dest_it;
        }
        else if ( is_high_surrogate(ch)
               && src_it_next != src_end
               && is_low_surrogate(ch2 = *src_it_next))
        {
            *dest_it = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            ++dest_it;
            ++src_it_next;
        }
        else if(allow_surr)
        {
            *dest_it = ch;
            ++dest_it;
        }
        else
        {
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *src = src_it_next;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    ++dest_it;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf16_to_utf32_size
    ( const char16_t* src
    , const char16_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    std::size_t count = 0;
    const char16_t* src_it = src;
    const char16_t* src_it_next;
    for(; src_it != src_end; src_it = src_it_next)
    {
        src_it_next = src_it + 1;
        ch = *src_it;
        src_it_next = src_it + 1;

        if (not_surrogate(ch))
        {
            ++count;
        }
        else if ( is_high_surrogate(ch)
               && src_it_next != src_end
               && is_low_surrogate(ch2 = *src_it_next))
        {
            ++count;
            ++src_it_next;
        }
        else if(allow_surr)
        {
            ++count;
        }
        else
        {
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return count;
                case stringify::v0::error_handling::replace:
                    ++count;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf16_sanitize
    ( const char16_t** src
    , const char16_t* src_end
    , char16_t** dest
    , char16_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    auto src_it = *src;
    const char16_t* src_it_next;
    auto dest_it = *dest;
    for( ; src_it != src_end; src_it = src_it_next)
    {
        ch = *src_it;
        src_it_next = src_it + 1;

        if (not_surrogate(ch))
        {
            if (dest_it != dest_end)
            {
                *dest_it = static_cast<char16_t>(ch);
                ++dest_it;
            } else goto insufficient_space;
        }
        else if ( is_high_surrogate(ch)
               && src_it_next != src_end
               && is_low_surrogate(ch2 = *src_it_next))
        {
            ++src_it_next;
            if (dest_it +1 < dest_end)
            {
                dest_it[0] = static_cast<char16_t>(ch);
                dest_it[1] = static_cast<char16_t>(ch2);
                dest_it += 2;
            } else goto insufficient_space;
        }
        else if(allow_surr)
        {
            if (dest_it != dest_end)
            {
                *dest_it = static_cast<char16_t>(ch);
                ++dest_it;
            } else goto insufficient_space;
        }
        else
        {
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *src = src_it_next;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    if (dest_it != dest_end)
                    {
                        *dest_it = 0xFFFD;
                        ++dest_it;
                        break;
                    } else goto insufficient_space;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf16_sanitize_size
    ( const char16_t* src
    , const char16_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::size_t count = 0;
    const char16_t* src_it = src;
    const char16_t* src_it_next;
    unsigned long ch, ch2;
    for( ; src_it != src_end; src_it = src_it_next)
    {
        ch = *src_it;
        src_it_next = src_it + 1;

        if (not_surrogate(ch))
        {
            ++ count;
        }
        else if ( is_high_surrogate(ch)
               && src_it_next != src_end
               && is_low_surrogate(ch2 = *src_it_next))
        {
            ++ src_it_next;
            count += 2 ;
        }
        else if (allow_surr)
        {
            ++ count;
        }
        else
        {
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return count;
                case stringify::v0::error_handling::replace:
                    ++count;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf16_codepoints_count
    ( const char16_t* begin
    , const char16_t* end
    , std::size_t max_count )
{
    std::size_t count = 0;
    for(auto it = begin; it != end && count < max_count; ++it, ++count)
    {
        if(is_high_surrogate(*it))
        {
            ++it;
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf16_validate(char32_t ch)
{
    return ch < 0x10000 ? 1 : ch < 0x110000 ? 2 : (std::size_t)-1;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf16_encode_char
    ( char16_t** dest
    , char16_t* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto dest_it = *dest;
    if (ch < 0x10000 && dest_it != end)
    {
        if (!allow_surr && detail::is_surrogate(ch))
        {
            goto invalid_char;
        }
        *dest_it = static_cast<char16_t>(ch);
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    if (ch < 0x110000)
    {
        if ((end - dest_it) > 1)
        {
            char32_t sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<char16_t>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10));
            dest_it[1] = static_cast<char16_t>(0xDC00 +  (sub_codepoint &  0x3FF));
            *dest = dest_it + 2;
            return stringify::v0::cv_result::success;
        }
        return stringify::v0::cv_result::insufficient_space;
    }

    invalid_char:
    switch (err_hdl)
    {
        case stringify::v0::error_handling::replace:
            if (dest_it != end)
            {
                *dest_it = 0xFFFD;
                *dest = dest_it + 1;
                return stringify::v0::cv_result::success;
            }
            return stringify::v0::cv_result::insufficient_space;
        case stringify::v0::error_handling::ignore:
            return stringify::v0::cv_result::success;
        default:
            BOOST_ASSERT(err_hdl == stringify::v0::error_handling::stop);
            return stringify::v0::cv_result::invalid_char;
    }
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf16_encode_fill
    ( char16_t** dest
    , char16_t* dest_end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto dest_it = *dest;
    const std::size_t capacity = dest_end - dest_it;
    using traits = std::char_traits<char16_t>;
    if (ch < 0x10000)
    {
        if (!allow_surr && detail::is_surrogate(ch))
        {
            goto invalid_char;
        }

        do_write:

        if(count <= capacity)
        {
            traits::assign(dest_it, count, static_cast<char16_t>(ch));
            *dest = dest_it + count;
            count = 0;
            return stringify::v0::cv_result::success;
        }
        traits::assign(dest_it, capacity, static_cast<char16_t>(ch));
        *dest = dest_it + capacity;
        count -= capacity;
        return stringify::v0::cv_result::insufficient_space;
    }
    if (ch < 0x110000)
    {
        const std::size_t capacity_2 = capacity / 2;
        char32_t sub_codepoint = ch - 0x10000;
        std::pair<char16_t, char16_t> obj =
            { static_cast<char16_t>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10))
            , static_cast<char16_t>(0xDC00 +  (sub_codepoint &  0x3FF)) };
        auto it2 = reinterpret_cast<decltype(obj)*>(dest_it);

        if(count <= capacity_2)
        {
            *dest = reinterpret_cast<char16_t*>(std::fill_n(it2, count, obj));
            count = 0;
            return stringify::v0::cv_result::success;
        }
        *dest = reinterpret_cast<char16_t*>(std::fill_n(it2, capacity_2, obj));
        count -= capacity_2;
        return stringify::v0::cv_result::insufficient_space;
    }

    invalid_char:
    switch (err_hdl)
    {
        case stringify::v0::error_handling::replace:
            ch = 0xFFFD;
            goto do_write;
        case stringify::v0::error_handling::ignore:
            return stringify::v0::cv_result::success;
        default:
            BOOST_ASSERT(err_hdl == stringify::v0::error_handling::stop);
            return stringify::v0::cv_result::invalid_char;
    }
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf32_to_utf16_transcode
    ( const char32_t** src
    , const char32_t* src_end
    , char16_t** dest
    , char16_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto src_it = *src;
    auto dest_it = *dest;
    std::size_t available_space = dest_end - dest_it;
    for ( ; src_it != src_end; ++src_it)
    {
        auto ch = *src_it;
        if (ch < 0x10000)
        {
            if (allow_surr || stringify::v0::detail::not_surrogate(ch))
            {
                if (available_space != 0)
                {
                    *dest_it = static_cast<char16_t>(ch);
                    ++dest_it;
                    --available_space;
                } else goto insufficient_space;
            }
            else goto invalid_char;
        }
        else if (ch < 0x110000)
        {
            if(available_space >= 2)
            {
                char32_t sub_codepoint = ch - 0x10000;
                dest_it[0] = static_cast<char16_t>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10));
                dest_it[1] = static_cast<char16_t>(0xDC00 +  (sub_codepoint &  0x3FF));
                available_space -= 2;
                dest_it += 2;
            }
            else goto insufficient_space;
        }
        else
        {
            invalid_char:
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *src = src_it + 1;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    if (available_space != 0)
                    {
                        *dest_it = 0xFFFD;
                        ++dest_it;
                        --available_space;
                        break;
                    }
                    else goto insufficient_space;

                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf32_to_utf16_size
    ( const char32_t* src
    , const char32_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) allow_surr;
    std::size_t count = 0;
    const char32_t* src_it = src;
    for ( ; src_it != src_end; ++src_it)
    {
        auto ch = *src_it;
        if (ch < 0x110000)
        {
            count += ch < 0x10000 ? 1 : 2;
        }
        else
        {
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    return count;

                case stringify::v0::error_handling::replace:
                    ++count;
                    break;

                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    break;
            }
        }
    }
    return count;
}

BOOST_STRINGIFY_STATIC_LINKAGE bool utf16_write_replacement_char
    ( char16_t** dest
    , char16_t* dest_end )
{
    auto dest_it = *dest;
    if (dest_it != dest_end)
    {
        *dest_it = 0xFFFD;
        *dest = dest_it + 1;
        return true;
    }
    return false;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf32_sanitize_size
    ( const char32_t* src
    , const char32_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    (void) allow_surr;
    return src_end - src;
    // todo err_hdl
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf32_sanitize
    ( const char32_t** src
    , const char32_t* src_end
    , char32_t** dest
    , char32_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto dest_it = *dest;
    if(allow_surr)
    {
        for (auto src_it = *src; src_it < src_end; ++src_it)
        {
            if (dest_it == dest_end)
            {
                *dest = dest_end;
                *src = src_it;
                return stringify::v0::cv_result::insufficient_space;
            }
            auto ch = *src_it;
            if (ch < 0x110000)
            {
                *dest_it = ch;
                ++dest_it;
            }
            else
            {
                switch(err_hdl)
                {
                    case stringify::v0::error_handling::stop:
                        *src = src_it + 1;
                        *dest = dest_it;
                        return stringify::v0::cv_result::invalid_char;
                    case stringify::v0::error_handling::replace:
                        *dest_it = 0xFFFD;
                        ++dest_it;
                        break;
                    default:
                        BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                        break;
                }
            }
        }
    }
    else
    {
        for(auto src_it = *src; src_it < src_end; ++src_it)
        {
            char32_t ch = *src_it;
            if (dest_it == dest_end)
            {
                *dest = dest_end;
                *src = src_it;
                return stringify::v0::cv_result::insufficient_space;
            }
            if (ch < 0x110000 && stringify::v0::detail::not_surrogate(ch))
            {
                *dest_it = ch;
                ++dest_it;
            }
            else
            {
                switch(err_hdl)
                {
                    case stringify::v0::error_handling::stop:
                        *src = src_it + 1;
                        *dest = dest_it;
                        return stringify::v0::cv_result::invalid_char;
                    case stringify::v0::error_handling::replace:
                        *dest_it = 0xFFFD;
                        ++dest_it;
                        break;
                    default:
                        BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                        break;
                }
            }
        }
    }
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;
}

inline std::size_t utf32_codepoints_count
    ( const char32_t* begin
    , const char32_t* end
    , std::size_t max_count )
{
    std::size_t len = end - begin;
    return len < max_count ? len : max_count;
}

inline std::size_t utf32_validate(char32_t ch)
{
    (void)ch;
    return 1;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf32_encode_char
    ( char32_t** dest
    , char32_t* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void)err_hdl;
    auto dest_it = *dest;
    if (ch > 0x10FFFF || (!allow_surr && detail::is_surrogate(ch)))
    {
        switch(err_hdl)
        {
            case stringify::v0::error_handling::stop:
                return stringify::v0::cv_result::invalid_char;
            case stringify::v0::error_handling::ignore:
                return stringify::v0::cv_result::success;
            default:
                BOOST_ASSERT(err_hdl == stringify::v0::error_handling::replace);
                ch = 0xFFFD;
        }
    }

    if (dest_it != end)
    {
        *dest_it = ch;
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    return stringify::v0::cv_result::insufficient_space;;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf32_encode_fill
    ( char32_t** dest
    , char32_t* dest_end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    using traits = std::char_traits<char32_t>;
    if (ch > 0x10FFFF || (!allow_surr && detail::is_surrogate(ch)))
    {
        switch (err_hdl)
        {
            case stringify::v0::error_handling::stop:
                return stringify::v0::cv_result::invalid_char;
            case stringify::v0::error_handling::ignore:
                count = 0;
                return stringify::v0::cv_result::success;
            default:
                BOOST_ASSERT(err_hdl == stringify::v0::error_handling::replace);
                ch = 0xFFFD;
        }
    }

    auto dest_it = *dest;
    std::size_t available_size = dest_end - dest_it;
    if (count <= available_size)
    {
        traits::assign(dest_it, count, ch);
        *dest = dest_it + count;
        count = 0;
        return stringify::v0::cv_result::success;
    }
    traits::assign(dest_it, available_size, ch);
    *dest = dest_it + available_size;
    count -= available_size;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE bool utf32_write_replacement_char
    ( char32_t** dest
    , char32_t* dest_end )
{
    auto dest_it = *dest;
    if (dest_it != dest_end)
    {
        *dest_it = 0xFFFD;
        *dest = dest_it + 1;
        return true;
    }
    return false;
}

inline char32_t utf16_decode_single_char(char16_t ch)
{
    return ch;
}

inline char32_t utf32_decode_single_char(char32_t ch)
{
    return ch;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf8_to_utf16_transcode
    ( const std::uint8_t** src
    , const std::uint8_t* src_end
    , char16_t** dest
    , char16_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;

    std::uint8_t ch0, ch1, ch2, ch3;
    unsigned long x;
    auto dest_it = *dest;
    auto src_it = *src;
    const std::uint8_t* previous_src_it;
    for(;src_it != src_end; ++dest_it)
    {
        previous_src_it = src_it;
        if (dest_it == dest_end)
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
            if ( ch0 > 0xC1
              && src_it != src_end && is_utf8_continuation(ch1 = * src_it))
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
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                       , allow_surr )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                *dest_it = static_cast<char16_t>((x << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xEF < ch0)
        {
            if (dest_it + 1 != dest_end)
            {
                if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                  && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                  && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                  && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
                {
                    x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
                    dest_it[0] = static_cast<char16_t>(0xD800 + ((x & 0xFFC00) >> 10));
                    dest_it[1] = static_cast<char16_t>(0xDC00 +  (x & 0x3FF));
                    ++dest_it;
                    ++src_it;
                } else goto invalid_sequence;
            } else goto insufficient_space;
        }
        else
        {
            invalid_sequence:
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    --dest_it;
                    break;
            }
        }
    }
    *dest = dest_it;
    *src = src_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *dest = dest_it;
    *src = previous_src_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_to_utf16_size
    ( const std::uint8_t* src_begin
    , const std::uint8_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    using stringify::v0::detail::utf8_decode;
    using stringify::v0::detail::not_surrogate;

    std::size_t size = 0;
    std::uint8_t ch0, ch1, ch2;
    auto src_it = src_begin;
    while(src_it < src_end)
    {
        ch0 = *src_it;
        ++src_it;
        if(ch0 < 0x80)
        {
            ++size;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it))
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == ch0)
        {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( ch0, ch1, allow_surr )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ++size;
                ++src_it;
            } else goto invalid_sequence;
        }
        else if(0xEF < ch0)
        {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                size += 2;
                ++src_it;
            } else goto invalid_sequence;
        }
        else
        {
            invalid_sequence:
            if (err_hdl == stringify::v0::error_handling::stop)
            {
                return size;
            }
            if (err_hdl == stringify::v0::error_handling::replace)
            {
                ++size;
            }
        }
    }
    return size;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf16_to_utf8_transcode
    ( const char16_t** src
    , const char16_t* src_end
    , std::uint8_t** dest
    , std::uint8_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    std::uint8_t* dest_it = *dest;
    auto src_it = *src;
    for( ; src_it < src_end; ++src_it)
    {
        auto ch = *src_it;
        if (ch < 0x80)
        {
            if(dest_it != dest_end)
            {
                *dest_it = static_cast<std::uint8_t>(ch);
                ++dest_it;
            }
            else
            {
                goto insufficient_space;
            }
        }
        else if (ch < 0x800)
        {
            if(dest_it + 1 < dest_end)
            {
                dest_it[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
                dest_it[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
                dest_it += 2;
            }
            else
            {
                goto insufficient_space;
            }
        }
        else if (not_surrogate(ch))
        {
            three_bytes:
            if(dest_it + 2 < dest_end)
            {
                dest_it[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
                dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
                dest_it[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
                dest_it += 3;
                continue;
            }
            goto insufficient_space;
        }
        else if ( stringify::v0::detail::is_high_surrogate(ch)
               && src_it != src_end
               && stringify::v0::detail::is_low_surrogate(*(src_it + 1)))
        {
            if(dest_it + 3 < dest_end)
            {
                unsigned long ch2 = *++src_it;
                unsigned long codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));

                dest_it[0] = static_cast<std::uint8_t>(0xF0 | ((codepoint & 0x1C0000) >> 18));
                dest_it[1] = static_cast<std::uint8_t>(0x80 | ((codepoint &  0x3F000) >> 12));
                dest_it[2] = static_cast<std::uint8_t>(0x80 | ((codepoint &    0xFC0) >> 6));
                dest_it[3] = static_cast<std::uint8_t>(0x80 |  (codepoint &     0x3F));
                dest_it += 4;
            }
            else
            {
                goto insufficient_space;
            }
        }
        else if(allow_surr)
        {
            goto three_bytes;
        }
        else // invalid sequece
        {
             switch(err_hdl)
             {
                 case stringify::v0::error_handling::stop:
                     *src = src_it + 1;
                     *dest = dest_it;
                     return stringify::v0::cv_result::invalid_char;

                 case stringify::v0::error_handling::replace:
                 {
                     if(dest_it + 2 < dest_end)
                     {
                         dest_it[0] = 0xEF;
                         dest_it[1] = 0xBF;
                         dest_it[2] = 0xBD;
                         dest_it += 3;
                     }
                     else
                     {
                         goto insufficient_space;
                     }
                     break;
                 }

                 default:
                     BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                     break;
             }
        }
    }
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::success;

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf16_to_utf8_size
    ( const char16_t* src_begin
    , const char16_t* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    (void) allow_surr;
    std::size_t size = 0;
    for(auto it = src_begin; it < src_end; ++it)
    {
        char16_t ch = *it;
        if (ch < 0x80)
        {
            ++size;
        }
        else if (ch < 0x800)
        {
            size += 2;
        }
        else if ( stringify::v0::detail::is_high_surrogate(ch)
               && it + 1 != src_end
               && stringify::v0::detail::is_low_surrogate(*(it + 1)) )
        {
            size += 4;
            ++it;
        }
        else
        {
            size += 3;
        }
    }
    return size;
}

BOOST_STRINGIFY_STATIC_LINKAGE
const stringify::v0::detail::transcoder_impl<std::uint8_t, char16_t>* utf8_to_enc16
    ( const stringify::v0::detail::encoding_impl<char16_t>& other )
{
    if (other.id == encoding_id::eid_utf16)
    {
        static const stringify::v0::detail::transcoder_impl<std::uint8_t, char16_t> tr_obj =
            { stringify::v0::detail::utf8_to_utf16_transcode
            , stringify::v0::detail::utf8_to_utf16_size };
        return & tr_obj;
    }
    return nullptr;
}

BOOST_STRINGIFY_STATIC_LINKAGE
const stringify::v0::detail::transcoder_impl<char16_t, std::uint8_t>* utf8_from_enc16
    ( const stringify::v0::detail::encoding_impl<char16_t>& other )
{
    if (other.id == encoding_id::eid_utf16)
    {
        static const stringify::v0::detail::transcoder_impl<char16_t, std::uint8_t> tr_obj =
            { stringify::v0::detail::utf16_to_utf8_transcode
            , stringify::v0::detail::utf16_to_utf8_size };
        return & tr_obj;
    }
    return nullptr;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::detail::encoding_impl<std::uint8_t>& utf8_impl()
{
    static const stringify::v0::detail::encoding_impl<std::uint8_t> encoding_obj =
         { { stringify::v0::detail::utf32_to_utf8_transcode
           , stringify::v0::detail::utf32_to_utf8_size }
         , { stringify::v0::detail::utf8_to_utf32_transcode
           , stringify::v0::detail::utf8_to_utf32_size }
         , { stringify::v0::detail::utf8_sanitize
           , stringify::v0::detail::utf8_sanitize_size }
         , stringify::v0::detail::utf8_validate
         , stringify::v0::detail::utf8_encode_char
         , stringify::v0::detail::utf8_encode_fill
         , stringify::v0::detail::utf8_codepoints_count
         , stringify::v0::detail::utf8_write_replacement_char
         , stringify::v0::detail::utf8_decode_single_char
         , nullptr, nullptr
         , stringify::v0::detail::utf8_from_enc16
         , stringify::v0::detail::utf8_to_enc16
         , nullptr, nullptr
         , "UTF-8"
         , stringify::v0::encoding_id::eid_utf8
         , 3, 0, 0x80 };

    return encoding_obj;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::detail::encoding_impl<char16_t>& utf16_impl()
{
    static const stringify::v0::detail::encoding_impl<char16_t> encoding_obj =
        { { stringify::v0::detail::utf32_to_utf16_transcode
          , stringify::v0::detail::utf32_to_utf16_size }
        , { stringify::v0::detail::utf16_to_utf32_transcode
          , stringify::v0::detail::utf16_to_utf32_size }
        , { stringify::v0::detail::utf16_sanitize
          , stringify::v0::detail::utf16_sanitize_size }
        , stringify::v0::detail::utf16_validate
        , stringify::v0::detail::utf16_encode_char
        , stringify::v0::detail::utf16_encode_fill
        , stringify::v0::detail::utf16_codepoints_count
        , stringify::v0::detail::utf16_write_replacement_char
        , stringify::v0::detail::utf16_decode_single_char
        , nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
        , "UTF-16"
        , stringify::v0::encoding_id::eid_utf16
        , 1, 0, 0xFFFF };

    return encoding_obj;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::detail::encoding_impl<char32_t>& utf32_impl()
{
    static const stringify::v0::detail::encoding_impl<char32_t> encoding_obj =
        { { stringify::v0::detail::utf32_sanitize
          , stringify::v0::detail::utf32_sanitize_size }
        , { stringify::v0::detail::utf32_sanitize
          , stringify::v0::detail::utf32_sanitize_size }
        , { stringify::v0::detail::utf32_sanitize
          , stringify::v0::detail::utf32_sanitize_size }
        , stringify::v0::detail::utf32_validate
        , stringify::v0::detail::utf32_encode_char
        , stringify::v0::detail::utf32_encode_fill
        , stringify::v0::detail::utf32_codepoints_count
        , stringify::v0::detail::utf32_write_replacement_char
        , stringify::v0::detail::utf32_decode_single_char
        , nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
        , "UTF-32"
        , stringify::v0::encoding_id::eid_utf32
        , 1, 0, 0x10FFFF };

    return encoding_obj;
}

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_UTF_ENCODINGS_HPP

