#ifndef BOOST_STRINGIFY_V0_DETAIL_SINGLE_BYTE_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_SINGLE_BYTE_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharIn>
static std::size_t same_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::error_handling err_hdl
    , bool allow_surr )
{
    (void) allow_surr;
    (void) err_hdl;
    return src_end - src;
}



template <class Impl>
struct single_byte_encoding
{
    static stringify::v0::cv_result to_utf32
        ( const char** src
        , const char* src_end
        , char32_t** dest
        , char32_t* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    static stringify::v0::cv_result from_utf32
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

    static char32_t decode_single_char(char ch)
    {
        return Impl::encode(ch);
    }

    static std::size_t codepoints_count
        ( const char* begin
        , const char* end
        , std::size_t max_count );

    static std::size_t replacement_char_size();

    static bool write_replacement_char
        ( char** dest
        , char* dest_end );

    static std::size_t validate(char32_t ch);
};


template <class Impl>
std::size_t single_byte_encoding<Impl>::codepoints_count
    ( const char* begin
    , const char* end
    , std::size_t max_count )
{
    std::size_t len = end - begin;
    return len < max_count ? len : max_count;
}

template <class Impl>
stringify::v0::cv_result single_byte_encoding<Impl>::to_utf32
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
            char32_t ch32 = Impl::decode(*src_it);
            if(ch32 != (char32_t)-1)
            {
                *dest_it = ch32;
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
        else
        {
            *src = src_it;
            *dest = dest_end;
            return stringify::v0::cv_result::insufficient_space;
        }
    }
    *src = src_end;
    *dest = dest_it;
    return stringify::v0::cv_result::success;
}

template <class Impl>
stringify::v0::cv_result single_byte_encoding<Impl>::sanitize
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
bool single_byte_encoding<Impl>::write_replacement_char
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
std::size_t single_byte_encoding<Impl>::replacement_char_size()
{
    return 1;
}

template <class Impl>
std::size_t single_byte_encoding<Impl>::validate(char32_t ch)
{
    return Impl::encode(ch) < 0x100 ? 1 : (std::size_t)-1;
}

template <class Impl>
stringify::v0::cv_result single_byte_encoding<Impl>::encode_char
    ( char** dest
    , char* end
    , char32_t ch
    , stringify::v0::error_handling err_hdl )
{
    char* dest_it = *dest;
    if(dest_it != end)
    {
        auto ch2 = Impl::encode(ch);
        if(ch2 < 0x100)
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
stringify::v0::cv_result single_byte_encoding<Impl>::encode_fill
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
stringify::v0::cv_result single_byte_encoding<Impl>::from_utf32
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
        if(ch2 < 0x100)
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
    static bool is_valid(unsigned char ch)
    {
        return ch < 0x80;
    }

    static char32_t decode(char ch)
    {
        unsigned char uch = ch;
        if (uch < 0x80)
            return uch;
        return -1;
    }

    static unsigned encode(char32_t ch)
    {
        return ch < 0x80 ? ch : 0x100;
    }
};

struct iso8859_1_impl
{
    static bool is_valid(unsigned char ch)
    {
        return true; //ch < 0x80 || 0x9F < ch;
    }

    static char32_t decode(char ch)
    {
        return (unsigned char)ch;
    }

    static unsigned encode(char32_t ch)
    {
        return ch;
        //return (ch < 0x80 || (0x9F < ch && ch < 0x100)) ? ch : 0x100;
    }
};


class iso8859_15_impl
{
public:

    static bool is_valid(unsigned char ch)
    {
        return true; //ch < 0x80 || 0x9F < ch;
    }

    static char32_t decode(char ch)
    {
        static const unsigned short ext[] = {
            /*                           */ 0x20AC, 0x00A5, 0x0160, 0x00A7,
            0x0161, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
            0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x017D, 0x00B5, 0x00B6, 0x00B7,
            0x017E, 0x00B9, 0x00BA, 0x00BB, 0x0152, 0x0153, 0x0178
        };

        unsigned char ch2 = ch;
        if (ch2 <= 0xA3 || 0xBF <= ch2)
            return ch2;

        return ext[ch2 - 0xA3];
    }

    static unsigned encode(char32_t ch)
    {
        return (ch < 0xA0 || (0xBE < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }

private:

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
    return ch;
}

class windows_1252_impl
{
public:

    constexpr static unsigned short decode_fail = 0xFFFF;

    static bool is_valid(unsigned char ch)
    {
        return ( ch != 0x81 && ch != 0x8D &&
                 ch != 0xA0 && ch != 0x9D );
    }

    static char32_t decode(char ch)
    {
        unsigned char ch2 = ch;
        constexpr short undef = -1;

        static const short ext[] = {
            0x20AC,  undef, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
            0x02C6, 0x2030, 0x0160, 0x2039, 0x0152,  undef, 0x017D,  undef,
             undef, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
            0x02DC, 0x2122, 0x0161, 0x203A, 0x0153,  undef, 0x017E, 0x0178,
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

        if (ch2 < 0x80)
            return ch2;

        return (int) ext[ch2 - 0x80];
    }

    static unsigned encode(char32_t ch)
    {
        return (ch < 0x80 || (0x9F < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }
private:

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

} // namespace detail

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char>& windows_1252()
{
    using impl = detail::single_byte_encoding<detail::windows_1252_impl>;
    static const stringify::v0::encoding<char> info =
         { { impl::from_utf32, detail::same_size<char32_t> }
         , { impl::to_utf32,   detail::same_size<char> }
         , { impl::sanitize,   detail::same_size<char> }
         , impl::validate
         , impl::encode_char
         , impl::encode_fill
         , impl::codepoints_count
         , impl::write_replacement_char
         , impl::decode_single_char
         , nullptr, nullptr , nullptr, nullptr
         , nullptr, nullptr , nullptr, nullptr
         , "windows-1252"
         , stringify::v0::encoding_id::eid_windows_1252
         , 1, 0x7F };
    return info;
}

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char>& iso_8859_1()
{
    using impl = detail::single_byte_encoding<detail::iso8859_1_impl>;
    static const stringify::v0::encoding<char> info =
         { { impl::from_utf32         , detail::same_size<char32_t> }
         , { impl::to_utf32 , detail::same_size<char> }
         , { impl::sanitize           , detail::same_size<char> }
         , impl::validate
         , impl::encode_char
         , impl::encode_fill
         , impl::codepoints_count
         , impl::write_replacement_char
         , impl::decode_single_char
         , nullptr, nullptr, nullptr, nullptr
         , nullptr, nullptr, nullptr, nullptr
         , "ISO-8859-1"
         , stringify::v0::encoding_id::eid_windows_1252
         , 1, 0x7F };
    return info;
}

BOOST_STRINGIFY_INLINE const stringify::v0::encoding<char>& iso_8859_15()
{
    using impl = detail::single_byte_encoding<detail::iso8859_15_impl>;
    static const stringify::v0::encoding<char> info =
         { { impl::from_utf32         , detail::same_size<char32_t> }
         , { impl::to_utf32 , detail::same_size<char> }
         , { impl::sanitize           , detail::same_size<char> }
         , impl::validate
         , impl::encode_char
         , impl::encode_fill
         , impl::codepoints_count
         , impl::write_replacement_char
         , impl::decode_single_char
         , nullptr, nullptr, nullptr, nullptr
         , nullptr, nullptr, nullptr, nullptr
         , "ISO-8859-15"
         , stringify::v0::encoding_id::eid_windows_1252
         , 1, 0x7F };
    return info;
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_SINGLE_BYTE_ENCODINGS_HPP

