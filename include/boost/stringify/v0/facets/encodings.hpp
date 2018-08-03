#ifndef BOOST_STRINGIFY_V0_FACETS_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_FACETS_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
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

class ascii_variant_decoder: public stringify::v0::decoder<char>
{
public:

    std::size_t remaining_codepoints_count
        ( std::size_t minuend
        , const char* begin
        , const char* end
        ) const override;
};

class ascii_variant_encoder: public stringify::v0::encoder<char>
{
public:

    std::size_t length(char32_t ch, bool allow_surrogates) const override;
};

class ascii_decoder: public stringify::v0::detail::ascii_variant_decoder
{
public:

    stringify::v0::decoder_result<char> decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const override;
};

class ascii_encoder: public stringify::v0::detail::ascii_variant_encoder
{
public:

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;
};


class iso_8859_1_decoder: public stringify::v0::detail::ascii_variant_decoder
{
public:

    stringify::v0::decoder_result<char> decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const override;
};

class iso_8859_1_encoder: public stringify::v0::detail::ascii_variant_encoder
{
public:

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;
};

class iso_8859_15_decoder: public stringify::v0::detail::ascii_variant_decoder
{
public:

    stringify::v0::decoder_result<char> decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const override;
};

class iso_8859_15_encoder: public stringify::v0::detail::ascii_variant_encoder
{
public:

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

private:

    unsigned convert_from_char32(char32_t ch) const;
};


class windows_1252_decoder: public stringify::v0::detail::ascii_variant_decoder
{
public:

    stringify::v0::decoder_result<char> decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const override;
};


class windows_1252_encoder: public stringify::v0::detail::ascii_variant_encoder
{
public:

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

private:

    unsigned convert_from_char32(char32_t ch) const;
};


class utf8_decoder: public stringify::v0::decoder<char>
{
public:

    utf8_decoder() = default;
    ~utf8_decoder() = default;

    stringify::v0::decoder_result<char> decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const override;

    std::size_t remaining_codepoints_count
        ( std::size_t minuend
        , const char* begin
        , const char* end
        ) const override;

private:

    static bool is_continuation(unsigned ch)
    {
        return (ch & 0xC0) == 0x80;
    }
};

class utf8_encoder: public stringify::v0::encoder<char>
{
public:

    utf8_encoder() = default;
    ~utf8_encoder() = default;

    std::size_t length(char32_t ch, bool allow_surrogates) const override;

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates
        ) const override;
};

using mutf8_decoder = stringify::v0::detail::utf8_decoder;

class mutf8_encoder: public stringify::v0::detail::utf8_encoder
{
public:

    mutf8_encoder() = default;
    ~mutf8_encoder() = default;

    std::size_t length
        ( char32_t ch
        , bool allow_surrogates )
        const override;

    char* encode
        ( char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates )
        const override;

    stringify::v0::char_cv_result<char> encode
        ( std::size_t count
        , char32_t ch
        , char* dest
        , char* end
        , bool allow_surrogates )
        const override;
};

template <typename CharIn>
class utf16_decoder: public stringify::v0::decoder<CharIn>
{
    static_assert(sizeof(CharIn) == 2, "incompatible character type for UTF-16");

public:

    utf16_decoder() = default;
    ~utf16_decoder() = default;

    stringify::v0::decoder_result<CharIn> decode
        ( stringify::v0::u32output& dest
        , const CharIn* begin
        , const CharIn* end
        , bool allow_surrogates )
        const override;

    std::size_t remaining_codepoints_count
        ( std::size_t minuend
        , const CharIn* begin
        , const CharIn* end )
        const override;
};


template <typename CharOut>
class utf16_encoder: public stringify::v0::encoder<CharOut>
{
    static_assert(sizeof(CharOut) == 2, "incompatible character type for UTF-16");

public:

    utf16_encoder() = default;
    ~utf16_encoder() = default;

    std::size_t length(char32_t ch, bool allow_surrogates) const override;

    CharOut* encode
        ( char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool allow_surrogates )
        const override;

   stringify::v0::char_cv_result<CharOut> encode
        ( std::size_t count
        , char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool allow_surr )
        const override;
};


template <typename CharIn>
class utf32_decoder: public stringify::v0::decoder<CharIn>
{
    static_assert(sizeof(CharIn) == 4, "incompatible character type for UTF-32");

public:

    utf32_decoder()
    {
    }
    ~utf32_decoder()
    {
    }

    stringify::v0::decoder_result<CharIn> decode
        ( stringify::v0::u32output& dest
        , const CharIn* begin
        , const CharIn* end
        , bool allow_surrogates )
        const override;

    std::size_t remaining_codepoints_count
        ( std::size_t minuend
        , const CharIn* begin
        , const CharIn* end )
        const override;
};

template <typename CharOut>
class utf32_encoder: public stringify::v0::encoder<CharOut>
{

    static_assert(sizeof(CharOut) == 4, "incompatible character type for UTF-32");

public:

    utf32_encoder()
    {
    }
    ~utf32_encoder()
    {
    }

    std::size_t length(char32_t ch, bool allow_surrogates) const override;

    CharOut* encode
        ( char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool allow_surrogates)
        const override;

    stringify::v0::char_cv_result<CharOut> encode
        ( std::size_t count
        , char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool allow_surr )
        const override;
};

template <typename CharIn, typename CharOut>
class bypass_transcoder: public stringify::v0::transcoder<CharIn, CharOut>
{
public:

    bypass_transcoder()
    {
    }

    virtual ~bypass_transcoder();

    using char_cv_result = stringify::v0::char_cv_result<CharOut>;
    using str_cv_result = stringify::v0::str_cv_result<CharIn, CharOut>;

    virtual str_cv_result convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;

    virtual std::size_t required_size
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;

    virtual char_cv_result convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;

    virtual std::size_t required_size
        ( CharIn ch
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;
};

template <typename CharOut>
class utf8_to_utf16
    : public stringify::v0::transcoder<char, CharOut>
    , private stringify::v0::detail::utf16_encoder<CharOut>
{
    using encoder = stringify::v0::detail::utf16_encoder<CharOut>;
    static_assert(sizeof(CharOut) == 2, "incompatible character type for UTF-16");

public:

    utf8_to_utf16()
    {
    }

    using CharIn = char;
    virtual stringify::v0::str_cv_result<char, CharOut> convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual std::size_t required_size
        ( const CharIn* src_begin
        , const CharIn* src_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual CharOut* convert
        ( char ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual stringify::v0::char_cv_result<CharOut> convert
        ( std::size_t count
        , char ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual std::size_t required_size
        ( char ch
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

private:

    template <typename T>
    static bool is_continuation(T ch)
    {
        return (static_cast<unsigned char>(ch) & 0xC0) == 0x80;
    }
};

template <typename CharIn>
class utf16_to_utf8
    : public stringify::v0::transcoder<CharIn, char>
    , private stringify::v0::detail::utf8_encoder
{
    using encoder = stringify::v0::detail::utf8_encoder;
    static_assert(sizeof(CharIn) == 2, "incompatible character type for UTF-16");

public:

    using CharOut = char;

    utf16_to_utf8()
    {
    }

    virtual stringify::v0::str_cv_result<CharIn, char> convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual std::size_t required_size
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual stringify::v0::char_cv_result<CharOut> convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;

    virtual std::size_t required_size
        ( CharIn ch
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates )
        const override;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE std::size_t ascii_variant_decoder::remaining_codepoints_count
    ( std::size_t minuend
    , const char* begin
    , const char* end
    ) const
{
    std::size_t len = end - begin;
    return len > minuend ? len - minuend : 0;
}

BOOST_STRINGIFY_INLINE std::size_t ascii_variant_encoder::length
    ( char32_t ch
    , bool allow_surrogates
    ) const
{
    (void)ch;
    (void)allow_surrogates;
    return 1;
}

BOOST_STRINGIFY_INLINE stringify::v0::decoder_result<char> ascii_decoder::decode
    ( stringify::v0::u32output& dest
    , const char* begin
    , const char* end
    , bool allow_surrogates
    ) const
{
    (void) allow_surrogates;

    for(auto it = begin; it < end; ++it)
    {
        unsigned char ch = *it;
        if(ch < 0x80)
        {
            auto r = dest.put32(static_cast<char32_t>(ch));
            if(r == stringify::v0::cv_result::success)
            {
                continue;
            }
            if (r == stringify::v0::cv_result::insufficient_space)
            {
                return { it, r };
            }
            BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
        }
        if( !dest.signalize_error())
        {
            // TODO: the line below is wrong. If signalize_error() returns false,
            // it is not necessarily because of insufficient space.
            return { it, stringify::v0::cv_result::insufficient_space };
        }
    }
    return { end, stringify::v0::cv_result::success };
}

BOOST_STRINGIFY_INLINE char* ascii_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    if(ch >= 0x80)
    {
        return nullptr;
    }
    if(dest != end)
    {
        *dest = static_cast<char>(ch);
        return dest + 1;
    }
    return end + 1;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> ascii_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;

    if(ch < 0x80)
    {
        std::size_t count2 = (std::min)(count, std::size_t(end - dest));
        std::memset(dest, ch, count2);
        return {count2, dest + count2};
    }
    return {0, nullptr};
}


BOOST_STRINGIFY_INLINE stringify::v0::decoder_result<char> iso_8859_1_decoder::decode
    ( stringify::v0::u32output& dest
    , const char* begin
    , const char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    for (auto it = begin; it < end; ++it)
    {
        unsigned char ch = *it;
        if(ch < 0x80 || 0x9F < ch)
        {
            auto r = dest.put32(static_cast<char32_t>(ch));
            if(r == stringify::v0::cv_result::success)
            {
                continue;
            }
            if(r == stringify::v0::cv_result::insufficient_space)
            {
                return { it, r };
            }
            BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
        }
        if( !dest.signalize_error())
        {
            return { it, stringify::v0::cv_result::insufficient_space };
        }
    }
    return { end , stringify::v0::cv_result::success };
}

BOOST_STRINGIFY_INLINE char* iso_8859_1_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void) allow_surrogates;
    if(ch < 0x80 || (0x9F < ch && ch < 0x100))
    {
        if(dest != end)
        {
            *dest = static_cast<char>(ch);
            return dest + 1;
        }
        return end + 1;
    }
    return nullptr;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> iso_8859_1_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    if(ch < 0x80 || (0x9F < ch && ch < 0x100))
    {
        std::size_t count2 = (std::min)(count, std::size_t(end - dest));
        std::memset(dest, ch, count2);
        return {count2, dest + count2};
    }
    return {0, nullptr};
}

BOOST_STRINGIFY_INLINE stringify::v0::decoder_result<char> iso_8859_15_decoder::decode
    ( stringify::v0::u32output& dest
    , const char* begin
    , const char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    constexpr unsigned short invalid = 0xFFFF;
    static const unsigned short table[] = {
        0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,
        0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x000F,
        0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017,
        0x0018, 0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F,
        0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
        0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
        0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
        0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
        0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
        0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
        0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
        0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
        0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
        0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
        0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
        0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x007F,
        invalid, invalid, invalid, invalid, invalid, invalid, invalid, invalid,
        invalid, invalid, invalid, invalid, invalid, invalid, invalid, invalid,
        invalid, invalid, invalid, invalid, invalid, invalid, invalid, invalid,
        invalid, invalid, invalid, invalid, invalid, invalid, invalid, invalid,
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

    for (auto it = begin; it < end; ++it)
    {
        auto code = table[static_cast<unsigned char>(*it)];
        if(code != invalid)
        {
            auto r = dest.put32(static_cast<char32_t>(code));
            if(r == stringify::v0::cv_result::success)
            {
                continue;
            }
            if(r == stringify::v0::cv_result::insufficient_space)
            {
                return { it, stringify::v0::cv_result::insufficient_space };
            }
        }
        if( !dest.signalize_error())
        {
            return { it, stringify::v0::cv_result::insufficient_space };
        }
    }
    return { end, stringify::v0::cv_result::success};
}

BOOST_STRINGIFY_INLINE char* iso_8859_15_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;

    unsigned ch2 = convert_from_char32(ch);
    if (ch2 < 0x100)
    {
        if(dest != end)
        {
            *dest = static_cast<char>(ch2);
            return dest + 1;
        }
        return end + 1;
    }
    return nullptr;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> iso_8859_15_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    unsigned ch2 = convert_from_char32(ch);

    if (ch2 < 0x100)
    {
        std::size_t count2 = (std::min)(count, std::size_t(end - dest));
        std::memset(dest, ch2, count2);
        return {count2, dest + count2};
    }
    return {0, nullptr};
}


BOOST_STRINGIFY_INLINE unsigned iso_8859_15_encoder::convert_from_char32(char32_t ch) const
{
    if(ch < 0x80 || (0xBE < ch && ch < 0x100))
    {
        return ch;
    }
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


BOOST_STRINGIFY_INLINE stringify::v0::decoder_result<char> windows_1252_decoder::decode
    ( stringify::v0::u32output& dest
    , const char* begin
    , const char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    constexpr unsigned short invalid = 0xFFFF;
    static const unsigned short table[] = {
        0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,
        0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x000F,
        0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017,
        0x0018, 0x0019, 0x001A, 0x001B, 0x001C, 0x001D, 0x001E, 0x001F,
        0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
        0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
        0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
        0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
        0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
        0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
        0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
        0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
        0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
        0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
        0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
        0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x007F,
        0x20AC, invalid, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
        0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, invalid, 0x017D, invalid,
        invalid, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
        0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, invalid, 0x017E, 0x0178,
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

    for (auto it = begin; it < end; ++it)
    {
        unsigned char index = *it;
        auto code = table[index];
        if(code != invalid)
        {
            auto r = dest.put32(code);
            if(r == stringify::v0::cv_result::success)
            {
                continue;
            }
            if(r == stringify::v0::cv_result::insufficient_space)
            {
                return { it, r };
            }
            BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
        }
        if( !dest.signalize_error())
        {
            return { it, stringify::v0::cv_result::insufficient_space };
        }
    }
    return { end, stringify::v0::cv_result::success };
}

BOOST_STRINGIFY_INLINE char* windows_1252_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;

    unsigned ch2 = (ch < 0x80 || (0x9F < ch && ch < 0x100))
        ? static_cast<unsigned>(ch)
        : convert_from_char32(ch);
    if (ch2 < 0x100)
    {
        if(dest != end)
        {
            *dest = static_cast<char>(ch2);
            return dest + 1;
        }
        return end + 1;
    }
    return nullptr;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> windows_1252_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    (void)allow_surrogates;
    unsigned ch2 = (ch < 0x80 || (0x9F < ch && ch < 0x100))
        ? static_cast<unsigned>(ch)
        : convert_from_char32(ch);

    if (ch2 < 0x100)
    {
        std::size_t count2 = (std::min)(count, std::size_t(end - dest));
        std::memset(dest, ch2, count2);
        return {count2, dest + count2};
    }
    return {0, nullptr};
}

BOOST_STRINGIFY_INLINE unsigned windows_1252_encoder::convert_from_char32(char32_t ch) const
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

BOOST_STRINGIFY_INLINE stringify::v0::decoder_result<char> utf8_decoder::decode
        ( stringify::v0::u32output& dest
        , const char* begin
        , const char* end
        , bool allow_surrogates
        ) const
{
    unsigned long ch1, ch2, ch3, x;
    bool failed_previous = false;
    auto it = begin;
    (void)allow_surrogates;
    while(it != end)
    {
        unsigned ch0 = (unsigned char)*it;
        ++it;
        if (0x80 > (x = ch0) ||

            (0xC0 == (ch0 & 0xE0)
             ? (it != end && is_continuation(ch1 = *it) &&
                (++it, x = utf8_decode(ch0, ch1), true))

             : 0xE0 == (ch0 & 0xF0)
             ? (it   != end && is_continuation(ch1 = *it) &&
                ++it != end && is_continuation(ch2 = *it) &&
                ( ++it
                , x = utf8_decode(ch0, ch1, ch2)
                , allow_surrogates || not_surrogate(x)))

            : (0xF0 == (ch0 & 0xF8) &&
               it   != end && is_continuation(ch1 = *it) &&
               ++it != end && is_continuation(ch2 = *it) &&
               ++it != end && is_continuation(ch3 = *it) &&
               (++it, (x = utf8_decode(ch0, ch1, ch2, ch3)) < 0x110000))))
        {
            auto r = dest.put32(x);
            if(r == stringify::v0::cv_result::success)
            {
                failed_previous = false;
                continue;
            }
            if(r == stringify::v0::cv_result::insufficient_space)
            {
                return { it - 1, stringify::v0::cv_result::insufficient_space };
            }
            BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
        }
        if(failed_previous && is_continuation(ch0))
        {
            continue;
        }
        if( ! dest.signalize_error())
        {
            return {it - 1, stringify::v0::cv_result::insufficient_space };
        }
        failed_previous = true;
    }
    return { it, stringify::v0::cv_result::success };
}

BOOST_STRINGIFY_INLINE std::size_t utf8_decoder::remaining_codepoints_count
        ( std::size_t minuend
        , const char* begin
        , const char* end
        ) const
{
    for(auto it = begin; it != end && minuend != 0; ++it)
    {
        if (!is_continuation(*it))
        {
            --minuend;
        }
    }
    return minuend;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> utf8_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    std::size_t c=0;
    if (ch < 0x80)
    {
        c = (std::min)(count, std::size_t(end - dest));
        std::memset(dest, static_cast<char>(ch), c);
        dest += c;
    }
    else if (ch < 0x800)
    {
        while(c < count && dest + 1 < end)
        {
            dest[0] = static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6));
            dest[1] = static_cast<char>(0x80 |  (ch &  0x3F));
            dest += 2;
            ++c;
        }
    }
    else if (ch <  0x10000)
    {
        if (!allow_surrogates && is_surrogate(ch))
        {
            return {0, nullptr};
        }
        while(c < count && dest + 2 < end)
        {
            dest[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
            dest[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
            dest[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
            dest += 3;
            ++c;
        }
    }
    else if (ch < 0x110000)
    {
        while(c < count && dest + 3 < end)
        {
            dest[0] = static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18));
            dest[1] = static_cast<char>(0x80 | ((ch &  0x3F000) >> 12));
            dest[2] = static_cast<char>(0x80 | ((ch &    0xFC0) >> 6));
            dest[3] = static_cast<char>(0x80 |  (ch &     0x3F));
            dest += 4;
            ++c;
        }
    }
    else
    {
        return {0, nullptr};
    }
    return {c, dest};
}

BOOST_STRINGIFY_INLINE char* utf8_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    if (ch < 0x80 && dest != end)
    {
        *dest = static_cast<char>(ch);
        return dest + 1;
    }
    std::size_t dest_size = end - dest;
    if (ch < 0x800 && 2 <= dest_size)
    {

        dest[0] = static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6));
        dest[1] = static_cast<char>(0x80 |  (ch &  0x3F));
        return dest + 2;
    }
    if (ch <  0x10000 && 3 <= dest_size)
    {
        if (allow_surrogates || stringify::v0::detail::not_surrogate(ch))
        {
            dest[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
            dest[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
            dest[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
            return dest + 3;
        }
        return nullptr;
    }
    if (ch < 0x110000 && 4 <= dest_size)
    {
        dest[0] = static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18));
        dest[1] = static_cast<char>(0x80 | ((ch &  0x3F000) >> 12));
        dest[2] = static_cast<char>(0x80 | ((ch &    0xFC0) >> 6));
        dest[3] = static_cast<char>(0x80 |  (ch &     0x3F));
        return dest + 4;
    }
    return ch >= 0x110000 ? nullptr : end + 1;
}

BOOST_STRINGIFY_INLINE std::size_t utf8_encoder::length(char32_t ch, bool allow_surrogates) const
{
    return (ch < 0x80 ? 1 :
            ch < 0x800 ? 2 :
            ! allow_surrogates && is_surrogate(ch) ? 4 :
            ch < 0x10000 ? 3 : 4);
}

BOOST_STRINGIFY_INLINE char* mutf8_encoder::encode
    ( char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    if (ch != 0)
    {
        return stringify::v0::detail::utf8_encoder
            ::encode(ch, dest, end, allow_surrogates);
    }
    if (dest + 1 < end)
    {
        dest[0] = '\xC0';
        dest[1] = '\x80';
        return dest + 2;
    }
    return end + 1;
}

BOOST_STRINGIFY_INLINE stringify::v0::char_cv_result<char> mutf8_encoder::encode
    ( std::size_t count
    , char32_t ch
    , char* dest
    , char* end
    , bool allow_surrogates
    ) const
{
    if (ch != 0)
    {
        return stringify::v0::detail::utf8_encoder
            ::encode(count, ch, dest, end, allow_surrogates);
    }
    std::size_t c = 0;
    for(; c != count; ++c)
    {
        dest[0] = '\xC0';
        dest[1] = '\x80';
        dest += 2;
    }
    return {c, dest};
}

BOOST_STRINGIFY_INLINE std::size_t mutf8_encoder::length(char32_t ch, bool allow_surrogates) const
{
    return (ch ==  0 ? 2 :
            ch < 0x80 ? 1 :
            ch < 0x800 ? 2 :
            ! allow_surrogates && is_surrogate(ch) ? 4 :
            ch < 0x10000 ? 3 : 4);
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

template <typename CharIn>
stringify::v0::decoder_result<CharIn> utf16_decoder<CharIn>::decode
    ( stringify::v0::u32output& dest
    , const CharIn* begin
    , const CharIn* end
    , bool allow_surrogates
    ) const
{
    (void) allow_surrogates;

    unsigned long ch, ch2, codepoint;
    auto it_next = begin;
    for(auto it = begin; it != end; it = it_next)
    {
        ch = *it;
        it_next = it + 1;

        if (not_surrogate(ch))
        {
            codepoint = ch;
        }
        else if (is_high_surrogate(ch)
                 && it_next != end
                 && is_low_surrogate(ch2 = static_cast<CharIn>(*it_next)))
        {
            codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            ++it_next;
        }
        else if(allow_surrogates && ch < 0x10FFFF)
        {
            codepoint = ch;
        }
        else
        {
            goto sinalize_error;
        }

        {
            auto r = dest.put32(codepoint);
            if(r == stringify::v0::cv_result::success)
            {
                continue;
            }
            if(r == stringify::v0::cv_result::insufficient_space)
            {
                return { it, stringify::v0::cv_result::insufficient_space };
            }
            BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
        }

        sinalize_error:

        if( ! dest.signalize_error())
        {
            return {it, stringify::v0::cv_result::insufficient_space };
        }
    }
    return { it_next, stringify::v0::cv_result::success };
}

template <typename CharIn>
std::size_t utf16_decoder<CharIn>::remaining_codepoints_count
    ( std::size_t minuend
    , const CharIn* begin
    , const CharIn* end
    ) const
{
    for(auto it = begin; it != end && minuend != 0; ++it, --minuend)
    {
        if(is_high_surrogate(static_cast<std::make_unsigned_t<CharIn>>(*it)))
        {
            ++it;
        }
    }
    return minuend;
}

template <typename CharOut>
std::size_t utf16_encoder<CharOut>::length(char32_t ch, bool allow_surrogates) const
{
    return (ch < 0x10000 && (allow_surrogates || ! is_surrogate(ch))) ? 1 : 2;
}

template <typename CharOut>
CharOut* utf16_encoder<CharOut>::encode
    ( char32_t ch
    , CharOut* dest
    , CharOut* end
    , bool allow_surrogates
    ) const
{
    if (ch < 0x10000 && dest != end)
    {
        if(allow_surrogates || ! is_surrogate(ch))
        {
            *dest = static_cast<CharOut>(ch);
            return dest + 1;
        }
        return nullptr;
    }
    if (ch < 0x110000 && (end - dest) > 1)
    {
        char32_t sub_codepoint = ch - 0x10000;
        dest[0] = static_cast<CharOut>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10));
        dest[1] = static_cast<CharOut>(0xDC00 +  (sub_codepoint &  0x3FF));
        return dest + 2;
    }
    return ch >= 0x110000 ? nullptr : end + 1;
}

template <typename CharOut>
stringify::v0::char_cv_result<CharOut> utf16_encoder<CharOut>::encode
    ( std::size_t count
    , char32_t ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , bool allow_surr )
    const
{
    std::size_t capacity = dest_end - dest_begin;
    if (ch < 0x10000)
    {
        if(allow_surr || ! is_surrogate(ch))
        {
            count = std::min(count, capacity);
            auto dest_it = std::fill_n(dest_begin, count, static_cast<CharOut>(ch));
            return {count, dest_it};
        }
        return {0, nullptr};
    }
    if (ch < 0x110000)
    {
        char32_t sub_codepoint = ch - 0x10000;
        std::pair<CharOut, CharOut> obj =
            { static_cast<CharOut>(0xD800 + ((sub_codepoint & 0xFFC00) >> 10))
            , static_cast<CharOut>(0xDC00 +  (sub_codepoint &  0x3FF)) };
        count = std::min(count, capacity / 2);
        std::fill_n(reinterpret_cast<decltype(obj)*>(dest_begin), count, obj);
        return {count, dest_begin + count * 2};
    }
    return {0, nullptr};
}

template <typename CharIn>
stringify::v0::decoder_result<CharIn> utf32_decoder<CharIn>::decode
    ( stringify::v0::u32output& dest
    , const CharIn* begin
    , const CharIn* end
    , bool allow_surrogates
    ) const
{
    auto it =begin;
    if(allow_surrogates)
    {
        for( ; it < end; ++it)
        {
            std::make_unsigned_t<CharIn> ch = *it;
            if(ch < 0x110000)
            {
                auto r = dest.put32(ch);
                if(r == stringify::v0::cv_result::success)
                {
                    continue;
                }
                if(r == stringify::v0::cv_result::insufficient_space)
                {
                    return { it, stringify::v0::cv_result::insufficient_space };
                }
                BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
            }
            if( ! dest.signalize_error())
            {
                return { it, stringify::v0::cv_result::insufficient_space };
            }
        }
    }
    else
    {
        for( ; it < end; ++it)
        {
            auto ch = *it;
            if(ch < 0x110000 && stringify::v0::detail::not_surrogate(ch))
            {
                auto r = dest.put32(ch);
                if(r == stringify::v0::cv_result::success)
                {
                    continue;
                }
                if(r == stringify::v0::cv_result::insufficient_space)
                {
                    return { it, stringify::v0::cv_result::insufficient_space };
                }
                BOOST_ASSERT(r == stringify::v0::cv_result::invalid_char);
            }
            if( ! dest.signalize_error())
            {
                return { it, stringify::v0::cv_result::insufficient_space };
            }
        }
    }

    return { end, stringify::v0::cv_result::success };
}

template <typename CharIn>
std::size_t utf32_decoder<CharIn>::remaining_codepoints_count
    ( std::size_t minuend
    , const CharIn* begin
    , const CharIn* end
    ) const
{
    std::size_t len = end - begin;
    return len < minuend ? minuend - len : 0;
}

template <typename CharOut>
std::size_t utf32_encoder<CharOut>::length(char32_t ch, bool allow_surrogates) const
{
    (void)ch;
    (void)allow_surrogates;
    return 1;
}

template <typename CharOut>
CharOut* utf32_encoder<CharOut>::encode
    ( char32_t ch
    , CharOut* dest
    , CharOut * end
    , bool allow_surrogates
    ) const
{
    if (dest != end && ch < 0x110000 && (allow_surrogates || ! is_surrogate(ch)))
    {
        *dest = ch;
        return dest + 1;
    }
    return dest == end ? (end + 1) : nullptr;
}

template <typename CharOut>
stringify::v0::char_cv_result<CharOut> utf32_encoder<CharOut>::encode
    ( std::size_t count
    , char32_t ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , bool allow_surr )
    const
{
    if(ch < 0x110000 && (allow_surr || ! is_surrogate(ch)))
    {
        count = std::min(count, std::size_t(dest_end - dest_begin));
        auto dest_it = std::fill_n(dest_begin, count, ch);
        return {count, dest_it};
    }
    return {0, nullptr};
}

template <typename CharIn, typename CharOut>
bypass_transcoder<CharIn, CharOut>::~bypass_transcoder()
{
}

template <typename CharIn, typename CharOut>
stringify::v0::str_cv_result<CharIn, CharOut>
bypass_transcoder<CharIn, CharOut>::convert
    ( const CharIn* src_begin
    , const CharIn* src_end
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) allow_surrogates;
    (void) err_sig;
    std::size_t src_size = src_end - src_begin;
    std::size_t dest_size = dest_end - dest_begin;
    std::size_t min_size = std::min(src_size, dest_size);

#if defined(_MSC_VER)

    for(std::size_t i = 0; i < min_size; ++i)
    {
        dest_begin[i] = static_cast<CharOut>(src_begin[i]);
    }
    CharOut *it = dest_begin + min_size;

#else

    // this emits a warning in Visual Studio
    CharOut* it = std::copy_n(src_begin, min_size, dest_begin);

#endif

    stringify::v0::cv_result status =
        ( src_size <= dest_size
        ? stringify::v0::cv_result::success
        : stringify::v0::cv_result::insufficient_space );
    return { src_begin + min_size, it, status };
}

template <typename CharIn, typename CharOut>
std::size_t bypass_transcoder<CharIn, CharOut>::required_size
    ( const CharIn* begin
    , const CharIn* end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    return end - begin;
}

template <typename CharIn, typename CharOut>
CharOut* bypass_transcoder<CharIn, CharOut>::convert
    ( CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    if(dest_end != dest_begin)
    {
        * dest_begin = ch;
        return dest_begin + 1;
    }
    return dest_begin;
}

template <typename CharIn, typename CharOut>
stringify::v0::char_cv_result<CharOut> bypass_transcoder<CharIn, CharOut>::convert
    ( std::size_t count
    , CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    std::size_t dest_size = dest_end - dest_begin;
    if(count > dest_size)
    {
        count = dest_size;
    }
    auto dest_it = std::fill_n(dest_begin, count, ch);
    return {count, dest_it};
}

template <typename CharIn, typename CharOut>
std::size_t bypass_transcoder<CharIn, CharOut>::required_size
    ( CharIn ch
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) ch;
    (void) err_sig;
    (void) allow_surrogates;
    return 1;
}

template <typename CharOut>
stringify::v0::str_cv_result<char, CharOut> utf8_to_utf16<CharOut>::convert
    ( const char* src_begin
    , const char* src_end
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    using stringify::v0::detail::utf8_decode;

    unsigned char ch0, ch1, ch2, ch3;
    auto dest_it = dest_begin;
    auto src_it = src_begin;
    auto previous_src_it = src_it;
    for(;src_it != src_end; ++src_it)
    {
        previous_src_it = src_it;
        if(dest_it == dest_end)
        {
            goto insufficient_space;
        }
        ch0 = (*src_it);
        if(ch0 < 0x80)
        {
            *dest_it = ch0;
            ++dest_it;
        }
        else if(0xC0 == (ch0 & 0xE0))
        {
            if(++src_it != src_end && is_continuation(ch1 = * src_it))
            {
                *dest_it = utf8_decode(ch0, ch1);
                ++dest_it;
            }
            else goto invalid_char;
        }
        else if(0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_continuation(ch1 = * src_it)
              && ++src_it != src_end && is_continuation(ch2 = * src_it) )
            {
                unsigned x = utf8_decode(ch0, ch1, ch2);
                if (allow_surrogates || stringify::v0::detail::not_surrogate(x))
                {
                    *dest_it = x;
                    ++dest_it;
                }
                else goto invalid_char;
            }
            else goto invalid_char;
        }
        else if(0xF0 == (ch0 & 0xF8))
        {
           if(dest_it + 1 != dest_end)
           {
               if ( ++src_it != src_end && is_continuation(ch1 = * src_it)
                 && ++src_it != src_end && is_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_continuation(ch3 = * src_it) )
               {
                   unsigned x = utf8_decode(ch0, ch1, ch2, ch3) - 0x10000;
                   if(x <= 0x10FFFF)
                   {
                       dest_it[0] = 0xD800 + ((x & 0xFFC00) >> 10);
                       dest_it[1] = 0xDC00 + (x & 0x3FF);
                       dest_it += 2;
                       continue;
                   }
               }
               goto invalid_char;
           }
           else goto insufficient_space;
        }
        else
        {
             goto invalid_char;
        }

        continue;

    invalid_char:
        auto it = emit_error(err_sig, *this, dest_it, dest_end, allow_surrogates);
        if(it == dest_end + 1)
        {
            return { previous_src_it, dest_it, stringify::v0::cv_result::insufficient_space };
        }
        while(src_it + 1 != src_end && is_continuation(*(src_it + 1)))
        {
            ++src_it;
        }
        if(it == nullptr)
        {
            return { src_it + 1, dest_it, stringify::v0::cv_result::invalid_char };
        }
        dest_it = it;
    }

    return {src_it, dest_it, stringify::v0::cv_result::success};

    insufficient_space:
    return { previous_src_it, dest_it, stringify::v0::cv_result::insufficient_space};
}

template <typename CharOut>
std::size_t utf8_to_utf16<CharOut>::required_size
        ( const char* src_begin
        , const char* src_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const
{
    (void) err_sig;
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
            if(++src_it != src_end && is_continuation(*src_it))
            {
                ++size;
                ++src_it;
            }
            else
            {
                size += 2;
            }
        }
        else if(0xE0 == (ch0 & 0xF0))
        {
            if ( ++src_it != src_end && is_continuation(ch1 = *src_it)
              && ++src_it != src_end && is_continuation(ch2 = *src_it) )
            {
                if(!allow_surrogates && is_surrogate(utf8_decode(ch0, ch1, ch2)))
                {
                    ++size;
                }
                ++src_it;
            }
            ++size;
        }
        else
        {
            if(0xF0 == (ch0 & 0xF8))
            {
                if ( ++src_it != src_end && is_continuation(*src_it)
                  && ++src_it != src_end && is_continuation(*src_it)
                  && ++src_it != src_end && is_continuation(*src_it) )
                {
                    ++src_it;
                }
            }
            else
            {
                while(++src_it != src_end && is_continuation(*src_it))
                {
                }
            }
            size += 2;
        }
    }
    return size;

}

template <typename CharOut>
CharOut* utf8_to_utf16<CharOut>::convert
    ( char ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    if(dest_begin != dest_end)
    {
        if(ch < 0x80)
        {
            *dest_begin = ch;
            return dest_begin + 1;
        }
        return nullptr;
    }
    return dest_end + 1;
}

template <typename CharOut>
stringify::v0::char_cv_result<CharOut> utf8_to_utf16<CharOut>::convert
    ( std::size_t count
    , char ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    if(ch < 0x80)
    {
        std::size_t dest_size = dest_end - dest_begin;
        if (count > dest_size)
        {
            count = dest_size;
        }
        std::fill_n(dest_begin, count, ch);
    }
    return {0, nullptr};
}

template <typename CharOut>
std::size_t utf8_to_utf16<CharOut>::required_size
    ( char ch
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    return ch < 0x80 ? 1 : 2;
}

template <typename CharIn>
stringify::v0::str_cv_result<CharIn, char> utf16_to_utf8<CharIn>::convert
    ( const CharIn* src_begin
    , const CharIn* src_end
    , char* dest_begin
    , char* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    using result = stringify::v0::str_cv_result<CharIn, char>;
    char* dest_it = dest_begin;
    auto src_it = src_begin;
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
                dest_it[0] =  static_cast<char>(0xE0 | ((ch & 0xF000) >> 12));
                dest_it[1] =  static_cast<char>(0x80 | ((ch &  0xFC0) >> 6));
                dest_it[2] =  static_cast<char>(0x80 |  (ch &   0x3F));
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
        else if(allow_surrogates)
        {
            goto three_bytes;
        }
        else if(err_sig.has_char())
        {
            auto it = encoder::encode
                ( err_sig.get_char(), dest_it, dest_end, allow_surrogates );
            if (it == dest_end + 1)
            {
                goto insufficient_space;
            }
            if (it == nullptr)
            {
                ch = 0xFFFE;
                goto three_bytes;
            }
            dest_it = it;
        }
        else if(! err_sig.skip())
        {
            return result{src_it, dest_it, stringify::v0::cv_result::invalid_char};
        }
    }

    return result{src_it, dest_it, stringify::v0::cv_result::success};

    insufficient_space:
    return result{src_it, dest_it, stringify::v0::cv_result::insufficient_space};
}

template <typename CharIn>
std::size_t utf16_to_utf8<CharIn>::required_size
    ( const CharIn* src_begin
    , const CharIn* src_end
    , const stringify::v0::error_signal & err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    (void) allow_surrogates;
    std::size_t size = 0;
    for(auto it = src_begin; it < src_end; ++it)
    {
        CharIn ch = *it;
        if (ch < 0x80)
        {
            ++size;
        }
        else if (ch < 0x800)
        {
            size += 2;
        }
        else if (stringify::v0::detail::not_surrogate(ch))
        {
            size+= 3;
        }
        else if (stringify::v0::detail::is_high_surrogate(ch)
              && it != src_end
              && stringify::v0::detail::is_low_surrogate(*(it + 1)))
        {
            ++it;
            size += 4;
        }
        else
        {
            size += allow_surrogates ? 3 : 4;
        }
    }
    return size;
}

// from single char

template <typename CharIn>
char* utf16_to_utf8<CharIn>::convert
    ( CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    return encoder::encode(ch, dest_begin, dest_end, allow_surrogates);
}

template <typename CharIn>
stringify::v0::char_cv_result<char> utf16_to_utf8<CharIn>::convert
    ( std::size_t count
    , CharIn ch
    , char* dest_begin
    , char* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    return encoder::encode(count, ch, dest_begin, dest_end, allow_surrogates);
}

template <typename CharIn>
std::size_t utf16_to_utf8<CharIn>::required_size
    ( CharIn ch
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    (void) err_sig;
    return encoder::length(ch, allow_surrogates);
}

/*
template <typename CharOut>
class from_ascii : public transcoder<char, CharOut>
{
    using CharIn = char;

public:

    from_ascii(bool clean)
        : m_clean(clean)
    {
    }

    virtual void write
        ( stringify::v0::output_writer<CharOut>& dest
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override
    {
        for(auto it = begin; it != end; ++it)
        {
            CharOut ch = static_cast<CharOut>(*it);
            if(ch < 0x80)
            {
                dest.put32(static_cast<CharOut>(ch));
            }
            else //if( ! boost::stringify::apply_error(dest, err_sig))
            {
                // todo appply error
                return;
            }
        }
    }

    virtual std::size_t length
        ( const CharIn* begin
        , const CharIn* end
        , bool allow_surrogates
        ) const override
    {
        return begin - end;
    }

private:

    bool m_clean = false;
};

template <>
class from_ascii<char> : public transcoder<char, char>
{
    using CharIn = char;
    using CharOut = char;

public:

    from_ascii(bool clean)
        : m_clean(clean)
    {
    }

    virtual void write
        ( stringify::v0::output_writer<CharOut>& dest
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override
    {
        if (m_clean)
        {
            dest.put32(begin, end - begin);
        }
        else
        {
            const char* prev = begin;
            for(const char* it = begin; it != end; ++it)
            {
                todo;
            }

        }
    }

    virtual std::size_t length
        ( const CharIn* begin
        , const CharIn* end
        , bool allow_surrogates
        ) const override
    {
        return begin - end;
    }

private:

    bool m_clean = false;
};


struct not_ascii_char
{
    bool operator()(char ch) const
    {
        return ch > 0x7f;
    }
};


class narrow_to_narrow: public transcoder<char, char>
{
    using CharIn = char;
    using CharOut = char;

public:

    virtual void write
        ( stringify::v0::output_writer<CharOut>& dest
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        ) const override
    {
        const char* prev = begin;
        const char* it = begin;
        do
        {
            it = std::find_if(it, end, stringify::v0::detail::not_ascii_char{});
            if (it > prev)
            {
                dest.put32(prev, it - prev);
            }
            it = write_non_ascii(dest, it, end, err_sig);
            prev = it;
        }
        while(it != end);
     }

protected:

     const char* write_non_ascii
        ( stringify::v0::output_writer<char>& dest
        , const char* begin
        , const char* end
        , const stringify::v0::error_signal& err_sig
        ) const = 0;
};

class ascii_to_narrow: public from_narrow_to_narrow
{
};

class sani_utf8: public from_narrow_to_narrow
{
};

class utf8_to_sesu8: public from_narrow_to_narrow
{
};

class utf8_to_sesu8: public from_narrow_to_narrow
{
};

class ascii_decoder: public stringify::v0::decoder<char>
{
public:

    ascii_decoder(bool clean)
        : m_clean(clean)
    {
    }

    ~ascii_decoder()
    {
    }


   virtual const stringify::v0::transcoder<CharIn, char>* converter
        ( stringify::v0::encoding<char>& output_encoding
        ) const override
    {
        return do_get_converter(output_encoding);
    }

    virtual const stringify::v0::transcoder<CharIn, char16_t>* converter
        ( stringify::v0::encoding<char16_t>& output_encoding
        ) const override
    {
        return do_get_converter(output_encoding);
    }

    virtual const stringify::v0::transcoder<CharIn, char32_t>* converter
        ( stringify::v0::encoding<char32_t>& output_encoding
        ) const override
    {
        return do_get_converter(output_encoding);
    }

    virtual const stringify::v0::transcoder<CharIn, wchar_t>* converter
        ( stringify::v0::encoding<wchar_t>& output_encoding
        ) const override
    {
        return do_get_converter(output_encoding);
    }

private:

    template <typename CharOut>
    const stringify::v0::transcoder<CharIn, CharOut>* converter_impl
        ( stringify::v0::encoding<CharOut>& output_encoding
        ) const override
    {
        if (m_clean)
        {
            static const stringify::v0::detail::from_ascii<CharOut> cv{true};
            return & cv;
        }
        else
        {
            static const stringify::v0::detail::from_ascii<CharOut> cv{false};
            return & cv;
        }
    }

private:

    bool m_clean = false;
};

*/


template <typename CharT> class utf16_info;
template <typename CharT> class utf32_info;

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf16_decoder<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf16_encoder<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf16_to_utf8<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf8_to_utf16<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf32_decoder<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf32_encoder<char32_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

class ascii_info: public stringify::v0::encoding_info<char>
{

public:

    ascii_info() noexcept;

    const stringify::v0::transcoder<char, char>*
    to(stringify::v0::encoding<char> enc) const override;

    const stringify::v0::transcoder<char, char16_t>*
    to(stringify::v0::encoding<char16_t> enc) const override;

    const stringify::v0::transcoder<char, char32_t>*
    to(stringify::v0::encoding<char32_t> enc) const override;

    const stringify::v0::transcoder<char, wchar_t>*
    to(stringify::v0::encoding<wchar_t> enc) const override;

private:

    stringify::v0::detail::ascii_decoder m_decoder;
    stringify::v0::detail::ascii_encoder m_encoder;
};


template <typename CharT>
class utf16_info: public stringify::v0::encoding_info<CharT>
{
    using base = stringify::v0::encoding_info<CharT>;

public:

    utf16_info() noexcept
        : base(m_decoder, m_encoder, stringify::v0::encoding_id::eid_utf16)
    {
    }

    const stringify::v0::transcoder<char, CharT>*
    sani_from(stringify::v0::encoding<char> enc) const override;

    const stringify::v0::transcoder<CharT, char>*
    sani_to(stringify::v0::encoding<char> enc) const override;

private:

    stringify::v0::detail::utf16_decoder<CharT> m_decoder;
    stringify::v0::detail::utf16_encoder<CharT> m_encoder;
};

} // namespace detail


inline stringify::v0::encoding<char> ascii()
{
    static const stringify::v0::detail::ascii_info info{};
    return info;
}

inline stringify::v0::encoding<char> iso_8859_1()
{
    static const stringify::v0::detail::iso_8859_1_decoder dec{};
    static const stringify::v0::detail::iso_8859_1_encoder enc{};
    static const stringify::v0::encoding_info<char> info
        {dec, enc, stringify::v0::encoding_id::eid_iso_8859_1};
    return info;
}

inline stringify::v0::encoding<char> iso_8859_15()
{
    static const stringify::v0::detail::iso_8859_15_decoder dec{};
    static const stringify::v0::detail::iso_8859_15_encoder enc{};
    static const stringify::v0::encoding_info<char> info
        {dec, enc, stringify::v0::encoding_id::eid_iso_8859_15};
    return info;
}

inline stringify::v0::encoding<char> windows_1252()
{
    static const stringify::v0::detail::windows_1252_decoder dec{};
    static const stringify::v0::detail::windows_1252_encoder enc{};
    static const stringify::v0::encoding_info<char> info
        {dec, enc, stringify::v0::encoding_id::eid_windows_1252};
    return info;
}

inline stringify::v0::encoding<char> utf8()
{
    static const stringify::v0::detail::utf8_decoder dec{};
    static const stringify::v0::detail::utf8_encoder enc{};
    static const stringify::v0::encoding_info<char> info
        { dec, enc, stringify::v0::encoding_id::eid_utf8 };
    return info;
}

inline stringify::v0::encoding<char> mutf8()
{
    static const stringify::v0::detail::mutf8_decoder dec{};
    static const stringify::v0::detail::mutf8_encoder enc{};
    static const stringify::v0::encoding_info<char> info
        { dec, enc, stringify::v0::encoding_id::eid_mutf8 };
    return info;
}

template <typename CharT>
inline stringify::v0::encoding<CharT> utf16()
{
    static_assert(sizeof(CharT) == 2, "incompatible character type for UTF-16");
    static const stringify::v0::detail::utf16_info<CharT> info;
    return info;
}

template <typename CharT>
inline stringify::v0::encoding<CharT> utf32()
{
    static_assert(sizeof(CharT) == 4, "incompatible character type for UTF-32");

    static const stringify::v0::detail::utf32_decoder<CharT> dec;
    static const stringify::v0::detail::utf32_encoder<CharT> enc;
    static const stringify::v0::encoding_info<CharT> info
        { dec, enc, stringify::v0::encoding_id::eid_utf32 };

    return info;
}

namespace detail
{

template <typename CharT>
inline stringify::v0::encoding<CharT> utfw_impl
    ( std::integral_constant<std::size_t, 2> )
{
    return stringify::v0::utf16<CharT>();
}

template <typename CharT>
inline stringify::v0::encoding<CharT> utfw_impl
    ( std::integral_constant<std::size_t, 4> )
{
    return stringify::v0::utf32<CharT>();
}

} // namespace detail

inline stringify::v0::encoding<wchar_t> utfw()
{
    return stringify::v0::detail::utfw_impl<wchar_t>
        ( std::integral_constant<std::size_t, sizeof(wchar_t)>{} );
}



namespace detail
{

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE ascii_info::ascii_info() noexcept
    : stringify::v0::encoding_info<char>
        (m_decoder, m_encoder, stringify::v0::encoding_id::eid_pure_ascii)
{
}

BOOST_STRINGIFY_INLINE const stringify::v0::transcoder<char, char>*
ascii_info::to(stringify::v0::encoding<char> enc) const
{
    (void)enc;
    static const stringify::v0::detail::bypass_transcoder<char, char> trans;
    return &trans;
}

BOOST_STRINGIFY_INLINE const stringify::v0::transcoder<char, char16_t>*
ascii_info::to(stringify::v0::encoding<char16_t> enc) const
{
    (void)enc;
    static const stringify::v0::detail::bypass_transcoder<char, char16_t> trans;
    return &trans;
}

BOOST_STRINGIFY_INLINE const stringify::v0::transcoder<char, char32_t>*
ascii_info::to(stringify::v0::encoding<char32_t> enc) const
{
    (void)enc;
    static const stringify::v0::detail::bypass_transcoder<char, char32_t> trans;
    return &trans;
}

BOOST_STRINGIFY_INLINE const stringify::v0::transcoder<char, wchar_t>*
ascii_info::to(stringify::v0::encoding<wchar_t> enc) const
{
    (void)enc;
    static const stringify::v0::detail::bypass_transcoder<char, wchar_t> trans;
    return &trans;
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)


template <typename CharT>
const stringify::v0::transcoder<char, CharT>*
utf16_info<CharT>::sani_from(stringify::v0::encoding<char> enc) const
{
    if (enc == stringify::v0::utf8())
    {
        static const stringify::v0::detail::utf8_to_utf16<CharT> trans{};
        return &trans;
    }
    return nullptr;
}

template <typename CharT>
const stringify::v0::transcoder<CharT, char>*
utf16_info<CharT>::sani_to(stringify::v0::encoding<char> enc) const
{
    if (enc == stringify::v0::utf8())
    {
        static const stringify::v0::detail::utf16_to_utf8<CharT> trans{};
        return &trans;
    }
    return nullptr;
}


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class utf16_info<char16_t>;

#endif //defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

} // namespace detail;


template <>
struct encoding_category<char>
{
    static constexpr bool constrainable = false;

    static const encoding<char>& get_default()
    {
        static const encoding<char> obj{ stringify::v0::utf8() };
        return obj;
    }
};

template <>
struct encoding_category<char16_t>
{
    static constexpr bool constrainable = false;

    static const encoding<char16_t>& get_default()
    {
        static const encoding<char16_t> obj{ stringify::v0::utf16<char16_t>() };
        return obj;
    }
};

template <>
struct encoding_category<char32_t>
{
    static constexpr bool constrainable = false;

    static const encoding<char32_t>& get_default()
    {
        static const encoding<char32_t> obj{ stringify::v0::utf32<char32_t>() };
        return obj;
    }
};

template <>
struct encoding_category<wchar_t>
{
    static constexpr bool constrainable = false;

    static const encoding<wchar_t>& get_default()
    {
        static const encoding<wchar_t> obj{ stringify::v0::utfw() };
        return obj;
    }
};

struct encoding_error_category
{
    static constexpr bool constrainable = false;

    static const stringify::v0::encoding_error& get_default()
    {
        static stringify::v0::encoding_error obj{U'\uFFFD'};
        return obj;
    }
};

struct allow_surrogates_category
{
    static constexpr bool constrainable = false;

    static const stringify::v0::allow_surrogates& get_default()
    {
        static stringify::v0::allow_surrogates obj{false};
        return obj;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_ENCODINGS_HPP

