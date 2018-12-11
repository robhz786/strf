#ifndef BOOST_STRINGIFY_V0_DETAIL_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/transcoding.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <cstring>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

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
constexpr bool not_surrogate(unsigned long codepoint) noexcept
{
    return codepoint >> 11 != 0x1B;
}
constexpr  bool not_high_surrogate(unsigned long codepoint) noexcept
{
    return codepoint >> 10 != 0x36;
}
constexpr  bool not_low_surrogate(unsigned long codepoint) noexcept
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
constexpr bool is_utf8_continuation(unsigned char ch)
{
    return (ch & 0xC0) == 0x80;
}

template <class Impl>
struct extended_ascii_encoding
{
    static std::size_t codepoints_count
        ( const char* begin
        , const char* end
        , std::size_t max_count );

    static stringify::v0::cv_result decode
        ( const char** src
        , const char* src_end
        , char32_t** dest
        , char32_t* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    static stringify::v0::cv_result encode
        ( const char32_t** src
        , const char32_t* src_end
        , char** dest
        , char* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    static stringify::v0::cv_result sanitize
        ( const char** src
        , const char* src_end
        , char** dest
        , char* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    static stringify::v0::cv_result encode_char
        ( char** dest
        , char* end
        , char32_t ch
        , stringify::v0::error_handling err_hdl );

    static stringify::v0::cv_result encode_fill
        ( char** dest
        , char* end
        , std::size_t& count
        , char32_t ch
        , stringify::v0::error_handling err_hdl );

    static std::size_t replacement_char_size();

    static bool write_replacement_char
        ( char** dest
        , char* dest_end );

    static std::size_t validate(char32_t ch);
};


template <class Impl>
std::size_t extended_ascii_encoding<Impl>::codepoints_count
    ( const char* begin
    , const char* end
    , std::size_t max_count )
{
    std::size_t len = end - begin;
    return len < max_count ? len : max_count;
}

template <class Impl>
stringify::v0::cv_result extended_ascii_encoding<Impl>::decode
    ( const char** src
    , const char* src_end
    , char32_t** dest
    , char32_t* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) allow_surr;
    auto dest_it = *dest;
    for (auto src_it = *src; src_it < src_end; ++src_it)
    {
        if(dest_it != dest_end)
        {
            *src = src_it;
            *dest = dest_end;
            return stringify::v0::cv_result::insufficient_space;
        }
        auto code = Impl::decode(*src_it);
        if(code != Impl::decode_fail)
        {
            *dest_it = static_cast<char32_t>(code);
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
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;
}

template <class Impl>
stringify::v0::cv_result extended_ascii_encoding<Impl>::sanitize
    ( const char** src
    , const char* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) allow_surr;
    auto dest_it = *dest;
    for (auto src_it = *src; src_it < src_end; ++src_it)
    {
        if(dest_it != dest_end)
        {
            *src = src_it;
            *dest = dest_end;
            return stringify::v0::cv_result::insufficient_space;
        }
        unsigned char ch = *src_it;
        if (Impl::is_valid(ch))
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
                    *dest_it = '?';
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


template <class Impl>
bool extended_ascii_encoding<Impl>::write_replacement_char
    ( char** dest
    , char* dest_end )
{
    auto dest_it = *dest;
    if (dest_it != dest_end)
    {
        *dest_it = '?';
        *dest = dest_it + 1;
        return true;
    }
    return false;
}

template <class Impl>
std::size_t extended_ascii_encoding<Impl>::replacement_char_size()
{
    return 1;
}

template <class Impl>
std::size_t extended_ascii_encoding<Impl>::validate(char32_t ch)
{
    return Impl::encode(ch) != 0x100 ? 1 : (std::size_t)-1;
}

template <class Impl>
stringify::v0::cv_result extended_ascii_encoding<Impl>::encode_char
    ( char** dest
    , char* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    char* dest_it = *dest;
    if(dest_it != end)
    {
        auto ch2 = Impl::encode(ch);
        if(ch2 != 0x100)
        {
            *dest_it = static_cast<char>(ch2);
            *dest = dest_it + 1;
            return stringify::v0::cv_result::success;
        }
        switch(err_hdl)
        {
            case stringify::v0::error_handling::stop:
                return stringify::v0::cv_result::invalid_char;
            case stringify::v0::error_handling::replace:
                *dest_it = '?';
                *dest = dest_it + 1;
                break;
            default:
                BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                break;
        }
        return stringify::v0::cv_result::success;
    }
    return stringify::v0::cv_result::insufficient_space;
}

template <class Impl>
stringify::v0::cv_result extended_ascii_encoding<Impl>::encode_fill
    ( char** dest
    , char* end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    unsigned ch2 = Impl::encode(ch);

    if (ch2 < 0x100)
    {
        do_write:

        char* dest_it = *dest;
        std::size_t count_ = count;
        std::size_t available = end - dest_it;

        if (count_ <= available)
        {
            std::memset(dest, ch2, count_);
            count = 0;
            *dest = dest_it + count_;
            return stringify::v0::cv_result::success;
        }
        std::memset(dest, ch2, available);
        count = count_ - available;
        *dest = end;
        return stringify::v0::cv_result::insufficient_space;
    }
    switch(err_hdl)
    {
        case stringify::v0::error_handling::stop:
            return stringify::v0::cv_result::invalid_char;
        case stringify::v0::error_handling::replace:
            ch2 = '?';
            goto do_write;
        default:
            BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
            break;
    }
    return stringify::v0::cv_result::success;
}

template <class Impl>
stringify::v0::cv_result extended_ascii_encoding<Impl>::encode
    ( const char32_t** src
    , const char32_t* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void)allow_surr;
    auto dest_it = *dest;
    auto src_it = *src;
    for(; src_it != src_end; ++src_it)
    {
        if(dest_it == dest_end)
        {
            *dest = dest_end;
            *src = src_it;
            return stringify::v0::cv_result::insufficient_space;
        }
        auto ch2 = Impl::encode(*src_it);
        if(ch2 != 0x100)
        {
            *dest_it = static_cast<char>(ch2);
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
                    *dest_it = '?';
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

struct strict_ascii_impl
{
    constexpr static char decode_fail = 0x80;

    static bool is_valid(unsigned char ch)
    {
        return ch < 0x80;
    }

    static char decode(unsigned char ch)
    {
        return ch < 0x80 ? ch : decode_fail;
    }

    static unsigned encode(char32_t ch)
    {
        return ch < 0x80 ? ch : 0x100;
    }
};

struct iso8859_1_impl
{
    constexpr static unsigned short decode_fail = 0xFFFF;

    static bool is_valid(unsigned char ch)
    {
        return ch < 0x80 || 0x9F < ch;
    }

    static unsigned short decode(unsigned char ch)
    {
        return (ch < 0x80 || 0x9F < ch) ? ch : decode_fail;
    }

    static unsigned encode(char32_t ch)
    {
        char32_t ch2 = ch;
        return (ch2 < 0x80 || (0x9F < ch2 && ch2 < 0x100)) ? ch2 : 0x100;
    }
};


struct iso8859_15_impl
{
    constexpr static unsigned short decode_fail = 0xFFFF;

    static bool is_valid(unsigned char ch)
    {
        return ch < 0x80 || 0x9F < ch;
    }

    static unsigned short decode(unsigned char ch)
    {
        unsigned char ch2 = ch;
        constexpr unsigned short fail = decode_fail;

        static const unsigned short ext[] = {
              fail,   fail,   fail,   fail,   fail,   fail,   fail,   fail,
              fail,   fail,   fail,   fail,   fail,   fail,   fail,   fail,
              fail,   fail,   fail,   fail,   fail,   fail,   fail,   fail,
              fail,   fail,   fail,   fail,   fail,   fail,   fail,   fail,
            0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x20AC, 0x00A5, 0x0160, 0x00A7,
            0x0161, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
            0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x017D, 0x00B5, 0x00B6, 0x00B7,
            0x017E, 0x00B9, 0x00BA, 0x00BB, 0x0152, 0x0153, 0x0178, 0x00BF,
            0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
            0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
            0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7,
            0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
            0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
            0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
            0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
            0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF
        };

        return ch2 < 0x80 ? ch2 : ext[ch2 - 0x80];
    }

    static unsigned encode(char32_t ch)
    {
        return (ch < 0x80 || (0xBE < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }

    static unsigned encode_ext(char32_t ch);
};

BOOST_STRINGIFY_INLINE unsigned iso8859_15_impl::encode_ext(char32_t ch)
{
    switch(ch)
    {
        case 0x20AC: return 0xA4;
        case 0x0160: return 0xA6;
        case 0x0161: return 0xA8;
        case 0x017D: return 0xB4;
        case 0x017E: return 0xB8;
        case 0x0152: return 0xBC;
        case 0x0153: return 0xBD;
        case 0x0178: return 0xBE;
        case 0xA4:
        case 0xA6:
        case 0xA8:
        case 0xB4:
        case 0xB8:
        case 0xBC:
        case 0xBD:
        case 0xBE:
            return 0x100;
    }
    return (0xA0 <= ch && ch <= 0xBB) ? ch : 0x100;
}

struct windows_1252_impl
{
    constexpr static unsigned short decode_fail = 0xFFFF;

    static bool is_valid(unsigned char ch)
    {
        return ( ch != 0x81 && ch != 0x8D &&
                 ch != 0xA0 && ch != 0x9D );
    }

    static unsigned short decode(unsigned char ch)
    {
        unsigned char ch2 = ch;
        constexpr unsigned short fail = decode_fail;

        static const unsigned short ext[] = {
            0x20AC,   fail, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
            0x02C6, 0x2030, 0x0160, 0x2039, 0x0152,   fail, 0x017D,   fail,
              fail, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
            0x02DC, 0x2122, 0x0161, 0x203A, 0x0153,   fail, 0x017E, 0x0178,
            0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7,
            0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
            0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
            0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
            0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
            0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
            0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7,
            0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
            0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
            0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
            0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
            0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF
        };

        return ch2 < 0x80 ? ch2 : ext[ch2 - 0x80];
    }

    static unsigned encode(char32_t ch)
    {
        return (ch < 0x80 || (0x9F < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }

    static unsigned encode_ext(char32_t ch);
};

BOOST_STRINGIFY_INLINE unsigned windows_1252_impl::encode_ext(char32_t ch)
{
    switch(ch)
    {
        case 0x20AC: return 0x80;
        case 0x201A: return 0x82;
        case 0x0192: return 0x83;
        case 0x201E: return 0x84;
        case 0x2026: return 0x85;
        case 0x2020: return 0x86;
        case 0x2021: return 0x87;
        case 0x02C6: return 0x88;
        case 0x2030: return 0x89;
        case 0x0160: return 0x8A;
        case 0x2039: return 0x8B;
        case 0x0152: return 0x8C;
        case 0x017D: return 0x8E;
        case 0x2018: return 0x91;
        case 0x2019: return 0x92;
        case 0x201C: return 0x93;
        case 0x201D: return 0x94;
        case 0x2022: return 0x95;
        case 0x2013: return 0x96;
        case 0x2014: return 0x97;
        case 0x02DC: return 0x98;
        case 0x2122: return 0x99;
        case 0x0161: return 0x9A;
        case 0x203A: return 0x9B;
        case 0x0153: return 0x9C;
        case 0x017E: return 0x9E;
        case 0x0178: return 0x9F;
    }
    return 0x100;
}

template <typename Char32>
stringify::v0::cv_result utf8_to_utf32_transcode
    ( const char** src
    , const char* src_end
    , Char32** dest
    , Char32* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;
    using stringify::v0::detail::is_utf8_continuation;

    unsigned char ch0, ch1, ch2, ch3;
    auto dest_it = *dest;
    auto src_it = *src;

    for(;src_it != src_end; ++src_it, ++dest_it)
    {
        if(dest_it == dest_end)
        {
            goto insufficient_space;
        }
        ch0 = (*src_it);
        if(ch0 < 0x80)
        {
            *dest_it = ch0;
        }
        else if(0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                *dest_it = utf8_decode(ch0, ch1);
            }
            else goto invalid_char;
        }
        else if(0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (allow_surr || stringify::v0::detail::not_surrogate(x))
                {
                    *dest_it = x;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else if(0xF0 == (ch0 & 0xF8))
        {
           if(dest_it + 1 != dest_end)
           {
               if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
               {
                   unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                   if(x <= 0x10FFFF)
                   {
                       *dest_it = x;
                       ++dest_it;
                   }
                   else goto invalid_char_with_continuations_chars;
               }
               else goto invalid_char;
           }
           else goto insufficient_space;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while(next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
            invalid_char:

            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *src = src_it;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    --src_it;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    --src_it;
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
    ( const char* src
    , const char* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned char ch0, ch1, ch2, ch3;
    const char* src_it = src;
    std::size_t count = 0;
    for(;src_it != src_end; ++src_it)
    {
        ch0 = (*src_it);
        if(ch0 < 0x80)
        {
            ++count;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                ++count;
            }
            else goto invalid_char;
        }
        else if(0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (allow_surr || stringify::v0::detail::not_surrogate(x))
                {
                    ++count;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else if(0xF0 == (ch0 & 0xF8))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                if(x <= 0x10FFFF)
                {
                    ++count;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while(next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
            invalid_char:
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


BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result utf8_sanitize
    ( const char** src
    , const char* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;

    unsigned char ch0, ch1, ch2, ch3;
    auto dest_it = *dest;
    auto src_it = *src;
    const char* previous_src_it;

    for(;src_it != src_end; ++src_it)
    {
        previous_src_it = src_it;
        ch0 = (*src_it);
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
            if(++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                auto x = utf8_decode(ch0, ch1);
                if (x >= 0x80)
                {
                    if (dest_it + 1 < dest_end)
                    {
                        dest_it[0] = ch0;
                        dest_it[1] = ch1;
                        dest_it += 2;
                    } else goto insufficient_space;
                }
                else if (dest_it != dest_end)
                {
                    *dest_it = x;
                    ++dest_it;
                } else goto insufficient_space;
            } else goto invalid_char;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (x >= 0x800)
                {
                    if( allow_surr || detail::not_surrogate(x) )
                    {
                        if (dest_it + 2 < dest_end)
                        {
                            dest_it[0] = ch0;
                            dest_it[1] = ch1;
                            dest_it[2] = ch2;
                            dest_it += 3;
                        } else goto insufficient_space;
                    } else goto invalid_char_with_continuations_chars;
                }
                else if (x >= 0x80)
                {
                    if (dest_it + 1 < dest_end)
                    {
                        dest_it[0] = 0xC0 | ((x & 0x7C0) >> 6);
                        dest_it[1] = 0x80 |  (x &  0x3F);
                        dest_it += 2;
                    } else goto insufficient_space;
                }
                else if (dest_it != dest_end)
                {
                    *dest_it = x;
                    ++dest_it;
                } else goto insufficient_space;
            } else goto invalid_char;
        }
        else if (0xF0 == (ch0 & 0xF8))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                if (x <= 0x10FFFF)
                {
                    if (x >= 0x10000)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = ch0;
                            dest_it[1] = ch1;
                            dest_it[2] = ch2;
                            dest_it[3] = ch3;
                            dest_it += 4;
                        } else goto insufficient_space;
                    }
                    else if (x >= 0x800)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = 0xE0 | ((x & 0xF000) >> 12);
                            dest_it[1] = 0x80 | ((x &  0xFC0) >> 6);
                            dest_it[2] = 0x80 |  (x &   0x3F);
                            dest_it += 3;
                        } else goto insufficient_space;
                    }
                    else if (x >= 0x80)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = 0xC0 | ((x & 0x7C0) >> 6);
                            dest_it[1] = 0x80 |  (x &  0x3F);
                            dest_it += 2;
                        } else goto insufficient_space;
                    }
                    else if (dest_it != dest_end)
                    {
                        *dest_it = ch0;
                        ++dest_it;
                    } else goto insufficient_space;
                } else goto invalid_char_with_continuations_chars;
            } else goto invalid_char;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while(next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
            invalid_char:
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
    ( const char* src
    , const char* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;
    unsigned char ch0, ch1, ch2, ch3;
    const char* src_it = src;
    std::size_t count = 0;
    for(;src_it != src_end; ++src_it)
    {
        unsigned ch0 = *src_it;
        if(ch0 < 0x80)
        {
            ++count;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                count += utf8_decode(ch0, ch1) >= 0x80 ? 2 : 1;
            } else goto invalid_char;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if( allow_surr || detail::not_surrogate(x) )
                {
                    count += x >= 0x800 ? 3 : x >= 0x80 ? 2 : 1;
                } else goto invalid_char_with_continuations_chars;
            } else goto invalid_char;
        }
        else if (0xF0 == (ch0 & 0xF8))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                if (x <= 0x10FFFF)
                {
                    count += ( x >= 0x10000 ? 4
                             : x >= 0x800   ? 3
                             : x >= 0x80    ? 2
                             : 1 );
                } else goto invalid_char_with_continuations_chars;
            } else goto invalid_char;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while(next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
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

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_codepoints_count
        ( const char* begin
        , const char* end
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
    ( char** dest
    , char* end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    char* dest_it = *dest;
    const std::size_t count_ = count;
    const std::size_t available = end - dest_it;
    std::size_t minc;
    if (ch < 0x80)
    {
        minc = (std::min)(count_, available);
        std::memset(dest_it, static_cast<char>(ch), minc);
        dest_it += minc;
    }
    else if (ch < 0x800)
    {
        minc = (std::min)(count_, available / 2);
        char ch0 = (0xC0 | ((ch & 0x7C0) >> 6));
        char ch1 = (0x80 |  (ch &  0x3F));
        for(std::size_t i = 0; i < minc; ++i)
        {
            dest_it[0] = ch0;
            dest_it[1] = ch1;
            dest_it += 2;
        }
    }
    else if (ch <  0x10000)
    {
        minc = (std::min)(count_, available / 3);
        char ch0 = (0xE0 | ((ch & 0xF000) >> 12));
        char ch1 = (0x80 | ((ch &  0xFC0) >> 6));
        char ch2 = (0x80 |  (ch &   0x3F));
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
        char ch0 = (0xF0 | ((ch & 0x1C0000) >> 18));
        char ch1 = (0x80 | ((ch &  0x3F000) >> 12));
        char ch2 = (0x80 | ((ch &    0xFC0) >> 6));
        char ch3 = (0x80 |  (ch &     0x3F));
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
                    dest_it[1] = 0xBD;
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
    ( char** dest
    , char* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    char* dest_it = *dest;
    if (ch < 0x80 && dest_it != end)
    {
        *dest_it = static_cast<char>(ch);
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    std::size_t dest_size = end - dest_it;
    if (ch < 0x800 && 2 <= dest_size)
    {
        dest_it[0] = static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6));
        dest_it[1] = static_cast<char>(0x80 |  (ch &  0x3F));
        *dest = dest_it + 2;
        return stringify::v0::cv_result::success;
    }
    if (ch <  0x10000 && 3 <= dest_size)
    {
        dest_it[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
        dest_it[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
        dest_it[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
        *dest = dest_it + 3;
        return stringify::v0::cv_result::success;
    }
    if (ch < 0x110000)
    {
        if (4 <= dest_size)
        {
            dest_it[0] = static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18));
            dest_it[1] = static_cast<char>(0x80 | ((ch &  0x3F000) >> 12));
            dest_it[2] = static_cast<char>(0x80 | ((ch &    0xFC0) >> 6));
            dest_it[3] = static_cast<char>(0x80 |  (ch &     0x3F));
            *dest = dest_it + 4;
            return stringify::v0::cv_result::success;
        }
        return stringify::v0::cv_result::insufficient_space;
    }
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

template <typename Char32>
stringify::v0::cv_result utf32_to_utf8_transcode
    ( const Char32** src
    , const Char32* src_end
    , char** dest
    , char* dest_end
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
                *dest_it = static_cast<char>(ch);
                ++dest_it;
                --available_space;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x800)
        {
            if(available_space >= 2)
            {
                dest_it[0] = static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6));
                dest_it[1] = static_cast<char>(0x80 |  (ch &  0x3F));
                dest_it += 2;
                available_space -= 2;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x10000)
        {
            if(available_space >= 3)
            {
                if(allow_surr || stringify::v0::detail::not_surrogate(ch))
                {
                    dest_it[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
                    dest_it[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
                    dest_it[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
                    dest_it += 3;
                    available_space -= 3;
                }
                else goto invalid_char;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x110000)
        {
            if(available_space >= 4)
            {
                dest_it[0] = static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18));
                dest_it[1] = static_cast<char>(0x80 | ((ch &  0x3F000) >> 12));
                dest_it[2] = static_cast<char>(0x80 | ((ch &    0xFC0) >> 6));
                dest_it[3] = static_cast<char>(0x80 |  (ch &     0x3F));
                dest_it += 4;
                available_space -= 4;
            }
            else goto insufficient_space;
        }
        else
        {
            invalid_char:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    dest_it[0] = 0xEF;
                    dest_it[1] = 0xBF;
                    dest_it[2] = 0xBD;
                    dest_it += 3;
                    available_space -=3;
                    break;

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

template <typename Char32>
std::size_t utf32_to_utf8_size
    ( const Char32* src
    , const Char32* src_end
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

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_replacement_char_size()
{
    return 3;
}

BOOST_STRINGIFY_STATIC_LINKAGE bool utf8_write_replacement_char
    ( char** dest
    , char* dest_end )
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

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf8_validate(char32_t ch)
{
    return ( ch < 0x80     ? 1 :
             ch < 0x800    ? 2 :
             ch < 0x10000  ? 3 :
             ch < 0x110000 ? 4 : (std::size_t)-1 );
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_encode_char
    ( char** dest
    , char* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    if (ch != 0)
    {
        return stringify::v0::detail::utf8_encode_char(dest, end, ch, err_hdl);
    }
    auto dest_it = *dest;
    if (dest_it + 1 < end)
    {
        dest_it[0] = '\xC0';
        dest_it[1] = '\x80';
        *dest = dest_it + 2;
        return stringify::v0::cv_result::success;
    }
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_encode
    ( const char32_t** src
    , const char32_t* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    auto src_it = *src;
    auto dest_it = *dest;
    std::size_t available_space = dest_end - dest_it;
    for(;src_it != src_end; ++src_it)
    {
        auto ch = *src_it;
        if(ch == 0)
        {
            if(available_space >= 2)
            {
                dest_it[0] = '\xC0';
                dest_it[1] = '\x80';
                dest_it += 2;
                available_space -= 2;
            }
            else goto insufficient_space;
        }
        else if(ch < 0x80)
        {
            if(available_space != 0)
            {
                *dest_it = static_cast<char>(ch);
                ++dest_it;
                --available_space;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x800)
        {
            if(available_space >= 2)
            {
                dest_it[0] = static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6));
                dest_it[1] = static_cast<char>(0x80 |  (ch &  0x3F));
                dest_it += 2;
                available_space -= 2;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x10000)
        {
            if(available_space >= 3)
            {
                if(allow_surr || stringify::v0::detail::not_surrogate(ch))
                {
                    dest_it[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
                    dest_it[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
                    dest_it[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
                    dest_it += 3;
                    available_space -= 3;
                }
                else goto invalid_char;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x110000)
        {
            if(available_space >= 4)
            {
                dest_it[0] = static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18));
                dest_it[1] = static_cast<char>(0x80 | ((ch &  0x3F000) >> 12));
                dest_it[2] = static_cast<char>(0x80 | ((ch &    0xFC0) >> 6));
                dest_it[3] = static_cast<char>(0x80 |  (ch &     0x3F));
                dest_it += 4;
                available_space -= 4;
            }
            else goto insufficient_space;
        }
        else
        {
            invalid_char:
            switch (err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    dest_it[0] = 0xEF;
                    dest_it[1] = 0xBF;
                    dest_it[2] = 0xBD;
                    dest_it += 3;
                    available_space -=3;
                    break;

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

BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_sanitize
    ( const char** src
    , const char* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;

    unsigned char ch0, ch1, ch2, ch3;
    auto dest_it = *dest;
    auto src_it = *src;
    const char* previous_src_it;

    for(;src_it != src_end; ++src_it)
    {
        previous_src_it = src_it;
        ch0 = (*src_it);
        if (ch0 == 0)
        {
            if(dest_it + 1 < dest_end)
            {
                dest_it[0] = 0xC0;
                dest_it[1] = 0x80;
                dest_it += 2;
            } else goto insufficient_space;
        }
        else if(ch0 < 0x80)
        {
            if(dest_it != dest_end)
            {
                *dest_it = ch0;
                ++dest_it;
            } else goto insufficient_space;
        }
        else if(0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                auto x = utf8_decode(ch0, ch1);
                if (x >= 0x80 || x == 0)
                {
                    if (dest_it + 1 < dest_end)
                    {
                        dest_it[0] = ch0;
                        dest_it[1] = ch1;
                        dest_it += 2;
                    } else goto insufficient_space;
                }
                else if (dest_it != dest_end)
                {
                    *dest_it = x;
                    ++dest_it;
                } else goto insufficient_space;
            } else goto invalid_char;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (x >= 0x800)
                {
                    if( allow_surr || detail::not_surrogate(x) )
                    {
                        if (dest_it + 2 < dest_end)
                        {
                            dest_it[0] = ch0;
                            dest_it[1] = ch1;
                            dest_it[2] = ch2;
                            dest_it += 3;
                        } else goto insufficient_space;
                    } else goto invalid_char_with_continuations_chars;
                }
                else if (x >= 0x80 || x == 0)
                {
                    if (dest_it + 1 < dest_end)
                    {
                        dest_it[0] = 0xC0 | ((x & 0x7C0) >> 6);
                        dest_it[1] = 0x80 |  (x &  0x3F);
                        dest_it += 2;
                    } else goto insufficient_space;
                }
                else if (dest_it != dest_end)
                {
                    *dest_it = x;
                    ++dest_it;
                } else goto insufficient_space;
            } else goto invalid_char;
        }
        else if (0xF0 == (ch0 & 0xF8))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                if (x <= 0x10FFFF)
                {
                    if (x > 0x10000)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = ch0;
                            dest_it[1] = ch1;
                            dest_it[2] = ch2;
                            dest_it[3] = ch3;
                            dest_it += 4;
                        } else goto insufficient_space;
                    }
                    else if (x > 0x800)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = 0xE0 | ((x & 0xF000) >> 12);
                            dest_it[1] = 0x80 | ((x &  0xFC0) >> 6);
                            dest_it[2] = 0x80 |  (x &   0x3F);
                            dest_it += 3;
                        } else goto insufficient_space;
                    }
                    else if (x > 0x80 || x == 0)
                    {
                        if (dest_it + 4 < dest_end)
                        {
                            dest_it[0] = 0xC0 | ((x & 0x7C0) >> 6);
                            dest_it[1] = 0x80 |  (x &  0x3F);
                            dest_it += 2;
                        } else goto insufficient_space;
                    }
                    else if (dest_it != dest_end)
                    {
                        *dest_it = ch0;
                        ++dest_it;
                    } else goto insufficient_space;
                } else goto invalid_char_with_continuations_chars;
            } else goto invalid_char;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while(next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
            invalid_char:
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


BOOST_STRINGIFY_STATIC_LINKAGE stringify::v0::cv_result mutf8_encode_fill
    ( char** dest
    , char* end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    if (ch != 0)
    {
        return stringify::v0::detail::utf8_encode_fill(dest, end, count, ch, err_hdl);
    }
    auto dest_it = *dest;
    std::size_t available = (end - dest_it) / 2;
    std::size_t c = count;
    std::size_t minc = std::min(count, available);
    auto it = std::fill_n( reinterpret_cast<std::pair<char, char>*>(dest_it)
                         , minc
                         , std::pair<char, char>{'\xC0', '\x80'});
    *dest = reinterpret_cast<char*>(it);
    count = c - minc;
    return stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t mutf8_validate(char32_t ch)
{
    return (ch ==  0 ? 2 :
            ch < 0x80 ? 1 :
            ch < 0x800 ? 2 :
            ch < 0x10000 ? 3 :
            ch < 0x110000 ? 4 : (std::size_t)-1);
}


template <typename CharIn, typename CharOut>
stringify::v0::cv_result utf16_to_utf32_transcode
    ( const CharIn** src
    , const CharIn* src_end
    , CharOut** dest
    , CharOut* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    auto dest_it = *dest;
    const CharIn* src_it_next;
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

template <typename CharIn>
std::size_t utf16_to_utf32_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    std::size_t count = 0;
    const CharIn* src_it = src;
    const CharIn* src_it_next;
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

template <typename CharT>
stringify::v0::cv_result utf16_sanitize
    ( const CharT** src
    , const CharT* src_end
    , CharT** dest
    , CharT* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    unsigned long ch, ch2;
    auto src_it = *src;
    const CharT* src_it_next;
    auto dest_it = *dest;
    for( ; src_it != src_end; src_it = src_it_next)
    {
        ch = *src_it;
        src_it_next = src_it + 1;

        if (not_surrogate(ch))
        {
            if (dest_it != dest_end)
            {
                *dest_it = ch;
                ++dest_it;
            } else goto insufficient_space;
        }
        else if ( is_high_surrogate(ch)
               && src_it_next != src_end
               && is_low_surrogate(ch2 = static_cast<CharT>(*src_it_next)))
        {
            ++src_it_next;
            if (dest_it +1 < dest_end)
            {
                dest_it[0] = ch;
                dest_it[1] = ch2;
                dest_it += 2;
            } else goto insufficient_space;
        }
        else if(allow_surr)
        {
            if (dest_it != dest_end)
            {
                *dest_it = ch;
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

template <typename CharT>
std::size_t utf16_sanitize_size
    ( const CharT* src
    , const CharT* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::size_t count = 0;
    const CharT* src_it = src;
    const CharT* src_it_next;
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
               && is_low_surrogate(ch2 = static_cast<CharT>(*src_it_next)))
        {
            ++ src_it_next;
            ++ count;
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

template <typename CharT>
std::size_t utf16_codepoints_count
    ( const CharT* begin
    , const CharT* end
    , std::size_t max_count )
{
    std::size_t count = 0;
    for(auto it = begin; it != end && count < max_count; ++it, ++count)
    {
        if(is_high_surrogate(static_cast<std::make_unsigned_t<CharT>>(*it)))
        {
            ++it;
        }
    }
    return count;
}

template <typename CharT>
std::size_t utf16_validate(char32_t ch)
{
    return ch < 0x10000 ? 1 : ch < 0x110000 ? 2 : (std::size_t)-1;
}

template <typename CharT>
stringify::v0::cv_result utf16_encode_char
    ( CharT** dest
    , CharT* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    auto dest_it = *dest;
    if (ch < 0x10000 && dest_it != end)
    {
        *dest_it = static_cast<CharT>(ch);
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    if (ch < 0x110000)
    {
        if ((end - dest_it) > 1)
        {
            char32_t sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<CharT>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10));
            dest_it[1] = static_cast<CharT>(0xDC00 +  (sub_codepoint &  0x3FF));
            *dest = dest_it + 2;
            return stringify::v0::cv_result::success;
        }
        return stringify::v0::cv_result::insufficient_space;
    }
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

template <typename CharT>
stringify::v0::cv_result utf16_encode_fill
    ( CharT** dest
    , CharT* dest_end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    auto dest_it = *dest;
    const std::size_t capacity = dest_end - dest_it;
    if (ch < 0x10000)
    {
        do_write:

        if(count >= capacity)
        {
            *dest = std::fill_n(dest_it, count, static_cast<CharT>(ch));
            count = 0;
            return stringify::v0::cv_result::success;
        }
        *dest = std::fill_n(dest_it, capacity, static_cast<CharT>(ch));
        count -= capacity;
        return stringify::v0::cv_result::insufficient_space;
    }
    if (ch < 0x110000)
    {
        const std::size_t capacity_2 = capacity / 2;
        char32_t sub_codepoint = ch - 0x10000;
        std::pair<CharT, CharT> obj =
            { static_cast<CharT>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10))
            , static_cast<CharT>(0xDC00 +  (sub_codepoint &  0x3FF)) };
        auto it2 = reinterpret_cast<decltype(obj)*>(dest_it);

        if(count >= capacity_2)
        {
            *dest = reinterpret_cast<CharT*>(std::fill_n(it2, count, obj));
            count = 0;
            return stringify::v0::cv_result::success;
        }
        *dest = reinterpret_cast<CharT*>(std::fill_n(it2, capacity_2, obj));
        count -= capacity_2;
        return stringify::v0::cv_result::insufficient_space;
    }


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

template <typename CharIn, typename CharOut>
stringify::v0::cv_result utf32_to_utf16_transcode
    ( const CharIn** src
    , const CharIn* src_end
    , CharOut** dest
    , CharOut* dest_end
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
            if (available_space != 0)
            {
                if (allow_surr || stringify::v0::detail::not_surrogate(ch))
                {
                    *dest_it = ch;
                    ++dest_it;
                    --available_space;
                }
                else goto invalid_char;
            }
            else goto insufficient_space;
        }
        else if (ch < 0x110000)
        {
            if(available_space >= 2)
            {
                CharIn sub_codepoint = ch - 0x10000;
                dest_it[0] = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
                dest_it[1] = 0xDC00 +  (sub_codepoint &  0x3FF);
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
                    *src = src_it;
                    *dest = dest_it;
                    return stringify::v0::cv_result::invalid_char;

                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    ++dest_it;
                    --available_space;
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

    insufficient_space:
    *src = src_it;
    *dest = dest_it;
    return stringify::v0::cv_result::insufficient_space;
}

template <typename CharIn>
std::size_t utf32_to_utf16_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    std::size_t count = 0;
    const CharIn* src_it = src;
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

template <typename CharT>
bool utf16_write_replacement_char
    ( CharT** dest
    , CharT* dest_end )
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

template <typename CharIn>
std::size_t utf32_sanitize_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    return src_end - src;
    // todo err_hdl
}

template <typename CharIn, typename CharOut>
stringify::v0::cv_result utf32_sanitize
    ( const CharIn** src
    , const CharIn* src_end
    , CharOut** dest
    , CharOut* dest_end
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
            std::make_unsigned_t<char32_t> ch = *src_it;
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

template <typename CharOut>
std::size_t utf32_codepoints_count
    ( const CharOut* begin
    , const CharOut* end
    , std::size_t max_count )
{
    std::size_t len = end - begin;
    return len < max_count ? len : max_count;
}

BOOST_STRINGIFY_STATIC_LINKAGE std::size_t utf32_validate(char32_t ch)
{
    (void)ch;
    return 1;
}

template <typename CharOut>
stringify::v0::cv_result utf32_encode_char
    ( CharOut** dest
    , CharOut* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    (void)err_hdl;
    auto dest_it = *dest;
    if (dest_it != end)
    {
        *dest_it = ch;
        *dest = dest_it + 1;
        return stringify::v0::cv_result::success;
    }
    return stringify::v0::cv_result::insufficient_space;;
}

template <typename CharOut>
stringify::v0::cv_result utf32_encode_fill
    ( CharOut** dest
    , CharOut* dest_end
    , std::size_t& count
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    (void) err_hdl;
    if (ch >= 0x110000)
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
    if (count >= available_size)
    {
        std::fill_n(dest_it, count, ch);
        dest += count;
        count = 0;
        return stringify::v0::cv_result::success;
    }
    std::fill_n(dest_it, available_size, ch);
    dest += available_size;
    count -= available_size;
    return stringify::v0::cv_result::insufficient_space;
}

template <typename CharOut>
bool utf32_write_replacement_char
    ( CharOut** dest
    , CharOut* dest_end )
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

template <typename CharT>
stringify::v0::cv_result utf8_to_utf16_transcode
    ( const char** src
    , const char* src_end
    , CharT** dest
    , CharT* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    using stringify::v0::detail::utf8_decode;

    unsigned char ch0, ch1, ch2, ch3;
    auto dest_it = *dest;
    auto src_it = *src;
    const char* previous_src_it;
    for(;src_it != src_end; ++src_it, ++dest_it)
    {
        previous_src_it = src_it;
        if (dest_it == dest_end)
        {
            goto insufficient_space;
        }
        ch0 = (*src_it);
        if (ch0 < 0x80)
        {
            *dest_it = ch0;
        }
        else if (0xC0 == (ch0 & 0xE0))
        {
            if (++src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                *dest_it = utf8_decode(ch0, ch1);
            }
            else goto invalid_char;
        }
        else if (0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (allow_surr || stringify::v0::detail::not_surrogate(x))
                {
                    *dest_it = x;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else if (0xF0 == (ch0 & 0xF8))
        {
           if (dest_it + 1 != dest_end)
           {
               if ( ++src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
               {
                   unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                   if(x <= 0x10FFFF)
                   {
                       dest_it[0] = 0xD800 + ((x & 0xFFC00) >> 10);
                       dest_it[1] = 0xDC00 + (x & 0x3FF);
                       ++dest_it;
                   }
                   else goto invalid_char_with_continuations_chars;
               }
               else goto invalid_char;
           }
           else goto insufficient_space;
        }
        else
        {
            invalid_char_with_continuations_chars:
            {
                auto next_src_it = src_it + 1;
                while (next_src_it != src_end && is_utf8_continuation(*next_src_it))
                {
                    ++next_src_it;
                }
                src_it = next_src_it;
            }
            invalid_char:
            switch(err_hdl)
            {
                case stringify::v0::error_handling::stop:
                    *dest = dest_it;
                    *src = src_it;
                    return stringify::v0::cv_result::invalid_char;
                case stringify::v0::error_handling::replace:
                    *dest_it = 0xFFFD;
                    --src_it;
                    break;
                default:
                    BOOST_ASSERT(err_hdl == stringify::v0::error_handling::ignore);
                    --src_it;
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
    ( const char* src_begin
    , const char* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    using stringify::v0::detail::utf8_decode;
    using stringify::v0::detail::not_surrogate;

    std::size_t size = 0;
    unsigned char ch0, ch1, ch2;
    auto src_it = src_begin;
    while(src_it < src_end)
    {
        ch0 = *src_it;
        if(ch0 < 0x80)
        {
            ++size;
            ++src_it;
        }
        else if(0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_utf8_continuation(*src_it))
            {
                ++size;
                ++src_it;
            }
            else goto invalid_char;
        }
        else if(0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = *src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = *src_it)  )
            {
                if (allow_surr || ! is_surrogate(utf8_decode(ch0, ch1, ch2)))
                {
                    ++src_it;
                    ++size;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else if(0xF0 == (ch0 & 0xF8))
        {
            unsigned ch1, ch2, ch3;
            if ( ++src_it != src_end && is_utf8_continuation(ch1 = *src_it)
              && ++src_it != src_end && is_utf8_continuation(ch2 = *src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = *src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                if(x <= 0x10FFFF)
                {
                    ++src_it;
                    size += 2;
                }
                else goto invalid_char_with_continuations_chars;
            }
            else goto invalid_char;
        }
        else
        {
            invalid_char_with_continuations_chars:
            do
            {
                ++src_it;
            } while(src_it != src_end && is_utf8_continuation(*src_it));
            invalid_char:
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

template <typename CharT>
stringify::v0::cv_result utf16_to_utf8_transcode
    ( const CharT** src
    , const CharT* src_end
    , char** dest
    , char* dest_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    char* dest_it = *dest;
    auto src_it = *src;
    for( ; src_it < src_end; ++src_it)
    {
        unsigned long ch = *src_it;
        if (ch < 0x80)
        {
            if(dest_it != dest_end)
            {
                *dest_it = static_cast<char>(ch);
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
                dest_it[0] = static_cast<char>((0xC0 | ((ch & 0x7C0) >> 6)));
                dest_it[1] = static_cast<char>((0x80 |  (ch &  0x3F)));
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
                dest_it[0] = static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
                dest_it[1] = static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
                dest_it[2] = static_cast<char>(0x80 |  (ch &   0x3F));
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

                dest_it[0] = static_cast<char>((0xF0 | ((codepoint & 0x1C0000) >> 18)));
                dest_it[1] = static_cast<char>((0x80 | ((codepoint &  0x3F000) >> 12)));
                dest_it[2] = static_cast<char>((0x80 | ((codepoint &    0xFC0) >> 6)));
                dest_it[3] = static_cast<char>((0x80 |  (codepoint &     0x3F)));
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

template <typename CharT>
std::size_t utf16_to_utf8_size
    ( const CharT* src_begin
    , const CharT* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) err_hdl;
    (void) allow_surr;
    std::size_t size = 0;
    for(auto it = src_begin; it < src_end; ++it)
    {
        CharT ch = *it;
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
               && stringify::v0::detail::is_low_surrogate(*++it) )
        {
            size += 4;
        }
        else
        {
            size += 3;
        }
    }
    return size;
}

template <typename CharX, std::size_t CharXSize>
struct utf8_utfx_impl;

template <typename CharX32>
struct utf8_utfx_impl<CharX32, 4>
{
    static const stringify::v0::transcoder<char, CharX32>* utf8_to_utfx
        ( const stringify::v0::encoding<CharX32>& other )
    {
        if (other.id == encoding_id::eid_utf32)
        {
            static const stringify::v0::transcoder<char, CharX32> tr_obj =
                { stringify::v0::detail::utf8_to_utf32_transcode<CharX32>
                , stringify::v0::detail::utf8_to_utf32_size };
            return & tr_obj;
        }
        return nullptr;
    }

    static const stringify::v0::transcoder<CharX32, char>* utf8_from_utfx
        ( const stringify::v0::encoding<CharX32>& other )
    {
        if (other.id == encoding_id::eid_utf32)
        {
            static const stringify::v0::transcoder<CharX32, char> tr_obj =
                { stringify::v0::detail::utf32_to_utf8_transcode<CharX32>
                , stringify::v0::detail::utf32_to_utf8_size<CharX32> };
            return & tr_obj;
        }
        return nullptr;
    }
};

template <typename CharX16>
struct utf8_utfx_impl<CharX16, 2>
{
    static const stringify::v0::transcoder<char, CharX16>* utf8_to_utfx
        ( const stringify::v0::encoding<CharX16>& other )
    {
        if (other.id == encoding_id::eid_utf16)
        {
            static const stringify::v0::transcoder<char, CharX16> tr_obj =
                { stringify::v0::detail::utf8_to_utf16_transcode<CharX16>
                , stringify::v0::detail::utf8_to_utf16_size };
            return & tr_obj;
        }
        return nullptr;
    }

    static const stringify::v0::transcoder<CharX16, char>* utf8_from_utfx
        ( const stringify::v0::encoding<CharX16>& other )
    {
        if (other.id == encoding_id::eid_utf16)
        {
            static const stringify::v0::transcoder<CharX16, char> tr_obj =
                { stringify::v0::detail::utf16_to_utf8_transcode<CharX16>
                , stringify::v0::detail::utf16_to_utf8_size<CharX16> };
            return & tr_obj;
        }
        return nullptr;
    }
};

template <typename CharX>
using utf8_utfx = stringify::v0::detail::utf8_utfx_impl<CharX, sizeof(CharX)>;

template <typename CharT, typename CharX, std::size_t CharXSize>
struct utf16_utfx_impl;

template <typename CharT, typename CharX32>
struct utf16_utfx_impl<CharT, CharX32, 4>
{
    static const stringify::v0::transcoder<CharT, CharX32>* utf16_to_utfx
        ( const stringify::v0::encoding<CharX32>& dest_enc )
    {
        if (dest_enc.id == encoding_id::eid_utf32)
        {
            static const stringify::v0::transcoder<CharT, CharX32> tr_obj =
                { stringify::v0::detail::utf16_to_utf32_transcode<CharT, CharX32>
                , stringify::v0::detail::utf16_to_utf32_size<CharT> };
            return & tr_obj;
        }
        return nullptr;
    }

    static const stringify::v0::transcoder<CharX32, CharT>* utf16_from_utfx
        ( const stringify::v0::encoding<CharX32>& src_enc )
    {
        if (src_enc.id == encoding_id::eid_utf32)
        {
            static const stringify::v0::transcoder<CharX32, CharT> tr_obj =
                { stringify::v0::detail::utf32_to_utf16_transcode<CharX32, CharT>
                , stringify::v0::detail::utf32_to_utf16_size<CharX32> };
            return & tr_obj;
        }
        return nullptr;
    }
};

template <typename CharT, typename CharX16>
struct utf16_utfx_impl<CharT, CharX16, 2>
{
    static const stringify::v0::transcoder<CharT, CharX16>* utf16_to_utfx
        ( const stringify::v0::encoding<CharX16>& other )
    {
        if (other.id == encoding_id::eid_utf16)
        {
            static const stringify::v0::transcoder<CharT, CharX16> tr_obj =
                { stringify::v0::detail::utf16_sanitize<CharT, CharX16>
                , stringify::v0::detail::utf16_sanitize_size<CharT> };
            return & tr_obj;
        }
        return nullptr;
    }

    static const stringify::v0::transcoder<CharX16, CharT>* utf16_from_utfx
        ( const stringify::v0::encoding<CharX16>& other )
    {
        if (other.id == encoding_id::eid_utf16)
        {
            static const stringify::v0::transcoder<CharT, CharX16> tr_obj =
                { stringify::v0::detail::utf16_sanitize<CharX16, CharT>
                , stringify::v0::detail::utf16_sanitize_size<CharX16> };
            return & tr_obj;
        }
        return nullptr;
    }
};

template <typename CharT, typename CharX>
using utf16_utfx = stringify::v0::detail::utf16_utfx_impl<CharT, CharX, sizeof(CharX)>;

} // namespace detail

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char>& utf8()
{
    static const stringify::v0::encoding<char> encoding_obj =
         { { stringify::v0::detail::utf32_to_utf8_transcode<char32_t>
           , stringify::v0::detail::utf32_to_utf8_size<char32_t> }
         , { stringify::v0::detail::utf8_to_utf32_transcode<char32_t>
           , stringify::v0::detail::utf8_to_utf32_size }
         , { stringify::v0::detail::utf8_sanitize
           , stringify::v0::detail::utf8_sanitize_size }
         , stringify::v0::detail::utf8_validate
         , stringify::v0::detail::utf8_encode_char
         , stringify::v0::detail::utf8_encode_fill
         , stringify::v0::detail::utf8_codepoints_count
         , stringify::v0::detail::utf8_write_replacement_char
         , nullptr
         , nullptr
         , stringify::v0::detail::utf8_utfx<char16_t>::utf8_from_utfx
         , stringify::v0::detail::utf8_utfx<char16_t>::utf8_to_utfx
         , nullptr
         , nullptr
         , stringify::v0::detail::utf8_utfx<wchar_t>::utf8_from_utfx
         , stringify::v0::detail::utf8_utfx<wchar_t>::utf8_to_utfx
         , "UTF-8"
         , stringify::v0::encoding_id::eid_utf8
         , 3 };

    return encoding_obj;
}

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char16_t>& utf16()
{
    static const stringify::v0::encoding<char16_t> encoding_obj =
        { { stringify::v0::detail::utf32_to_utf16_transcode<char32_t, char16_t>
          , stringify::v0::detail::utf32_to_utf16_size<char32_t> }
        , { stringify::v0::detail::utf16_to_utf32_transcode<char16_t, char32_t>
          , stringify::v0::detail::utf16_to_utf32_size<char16_t> }
        , { stringify::v0::detail::utf16_sanitize<char16_t>
          , stringify::v0::detail::utf16_sanitize_size<char16_t> }
        , stringify::v0::detail::utf16_validate<char16_t>
        , stringify::v0::detail::utf16_encode_char<char16_t>
        , stringify::v0::detail::utf16_encode_fill<char16_t>
        , stringify::v0::detail::utf16_codepoints_count<char16_t>
        , stringify::v0::detail::utf16_write_replacement_char<char16_t>
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , stringify::v0::detail::utf16_utfx<char16_t, wchar_t>::utf16_from_utfx
        , stringify::v0::detail::utf16_utfx<char16_t, wchar_t>::utf16_to_utfx
        , "UTF-16"
        , stringify::v0::encoding_id::eid_utf16
        , 1 };

    return encoding_obj;
}

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char32_t>& utf32()
{
    static const stringify::v0::encoding<char32_t> encoding_obj =
        { { stringify::v0::detail::utf32_sanitize<char32_t, char32_t>
          , stringify::v0::detail::utf32_sanitize_size<char32_t> }
        , { stringify::v0::detail::utf32_sanitize<char32_t, char32_t>
          , stringify::v0::detail::utf32_sanitize_size<char32_t> }
        , { stringify::v0::detail::utf32_sanitize<char32_t, char32_t>
          , stringify::v0::detail::utf32_sanitize_size<char32_t> }
        , stringify::v0::detail::utf32_validate
        , stringify::v0::detail::utf32_encode_char<char32_t>
        , stringify::v0::detail::utf32_encode_fill<char32_t>
        , stringify::v0::detail::utf32_codepoints_count<char32_t>
        , stringify::v0::detail::utf32_write_replacement_char<char32_t>
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , "UTF-32"
        , stringify::v0::encoding_id::eid_utf32
        , 1 };

    return encoding_obj;
}

namespace detail
{

template <typename WChar16>
const stringify::v0::encoding<WChar16>& utfw_impl(std::integral_constant<std::size_t, 2>)
{
    static const stringify::v0::encoding<WChar16> encoding_obj =
        { { stringify::v0::detail::utf32_to_utf16_transcode<char32_t, WChar16>
          , stringify::v0::detail::utf32_to_utf16_size<char32_t> }
        , { stringify::v0::detail::utf16_to_utf32_transcode<WChar16, char32_t>
          , stringify::v0::detail::utf16_to_utf32_size<WChar16> }
        , { stringify::v0::detail::utf16_sanitize<WChar16>
          , stringify::v0::detail::utf16_sanitize_size<WChar16> }
        , stringify::v0::detail::utf16_validate<WChar16>
        , stringify::v0::detail::utf16_encode_char<WChar16>
        , stringify::v0::detail::utf16_encode_fill<WChar16>
        , stringify::v0::detail::utf16_codepoints_count<WChar16>
        , stringify::v0::detail::utf16_write_replacement_char<WChar16>
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , "UTF-16"
        , stringify::v0::encoding_id::eid_utf16
        , 1 };

    return encoding_obj;
}

template <typename WChar32>
inline const stringify::v0::encoding<WChar32>& utfw_impl
    ( std::integral_constant<std::size_t, 4> )
{
    static const stringify::v0::encoding<WChar32> encoding_obj =
        { { stringify::v0::detail::utf32_sanitize<char32_t, WChar32>
          , stringify::v0::detail::utf32_sanitize_size<char32_t> }
        , { stringify::v0::detail::utf32_sanitize<WChar32, char32_t>
          , stringify::v0::detail::utf32_sanitize_size<WChar32> }
        , { stringify::v0::detail::utf32_sanitize<WChar32, WChar32>
          , stringify::v0::detail::utf32_sanitize_size<WChar32> }
        , stringify::v0::detail::utf32_validate
        , stringify::v0::detail::utf32_encode_char<WChar32>
        , stringify::v0::detail::utf32_encode_fill<WChar32>
        , stringify::v0::detail::utf32_codepoints_count<WChar32>
        , stringify::v0::detail::utf32_write_replacement_char<WChar32>
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , nullptr
        , "UTF-32"
        , stringify::v0::encoding_id::eid_utf32
        , 1 };

    return encoding_obj;
}

} // namespace detail

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<wchar_t>& wchar_encoding()
{
    return stringify::v0::detail::utfw_impl<wchar_t>
        ( std::integral_constant<std::size_t, sizeof(wchar_t)>{} );
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_ENCODINGS_HPP

