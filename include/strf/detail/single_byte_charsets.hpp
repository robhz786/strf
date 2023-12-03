#ifndef STRF_DETAIL_SINGLE_BYTE_CHARSETS_HPP
#define STRF_DETAIL_SINGLE_BYTE_CHARSETS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>

// References
// https://www.compart.com/en/unicode/charsets/
// http://www.unicode.org/Public/MAPPINGS/ISO8859/
// https://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WindowsBestFit/
// https://www.fileformat.info/info/charset/ISO-8859-8/

#define STRF_SBC_CHECK_DST                                    \
    STRF_IF_UNLIKELY (dst_it == dst_end) {                   \
        return {seq_begin, dst_it, reason::insufficient_output_space};    \
    }

#define STRF_SBC_CHECK_DST_SIZE(SIZE)                         \
    STRF_IF_UNLIKELY (dst_it + (SIZE) > dst_end) {           \
        return {seq_begin, dst_it, reason::insufficient_output_space};    \
    }

#define STRF_DEF_SINGLE_BYTE_CHARSET_(CHARSET)                                \
    template <typename CharT>                                                 \
    class static_charset<CharT, strf::csid_ ## CHARSET>                       \
        : public strf::detail::single_byte_charset                            \
            < CharT, strf::detail::impl_ ## CHARSET >                         \
    { };                                                                      \
                                                                              \
    template <typename SrcCharT, typename DstCharT>                          \
    class static_transcoder                                                   \
        < SrcCharT, DstCharT, strf::csid_ ## CHARSET, strf::csid_ ## CHARSET > \
        : public strf::detail::single_byte_charset_to_itself                  \
            < SrcCharT, DstCharT, strf::detail::impl_ ## CHARSET >           \
    {};                                                                       \
                                                                              \
    template <typename SrcCharT, typename DstCharT>                          \
    class static_transcoder                                                   \
        < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_ ## CHARSET >     \
        : public strf::detail::utf32_to_single_byte_charset                   \
            < SrcCharT, DstCharT, strf::detail::impl_ ## CHARSET >           \
    {};                                                                       \
                                                                              \
    template <typename SrcCharT, typename DstCharT>                          \
    class static_transcoder                                                   \
        < SrcCharT, DstCharT, strf::csid_ ## CHARSET, strf::csid_utf32 >     \
        : public strf::detail::single_byte_charset_to_utf32                   \
            < SrcCharT, DstCharT, strf::detail::impl_ ## CHARSET >           \
    {};                                                                       \
                                                                              \
    template <typename CharT>                                                 \
    using CHARSET ## _t =                                                     \
        strf::static_charset<CharT, strf::csid_ ## CHARSET>;

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

#if defined(__CUDACC__)

#define STRF_DEF_SINGLE_BYTE_CHARSET(CHARSET)                           \
    STRF_DEF_SINGLE_BYTE_CHARSET_(CHARSET)                              \
    template <typename CharT>                                           \
    STRF_DEVICE CHARSET ## _t<CharT> CHARSET = {} // NOLINT(bugprone-macro-parentheses)

#else

#define STRF_DEF_SINGLE_BYTE_CHARSET(CHARSET)                           \
    STRF_DEF_SINGLE_BYTE_CHARSET_(CHARSET)                              \
    template <typename CharT>                                           \
    STRF_DEVICE constexpr CHARSET ## _t<CharT> CHARSET = {} // NOLINT(bugprone-macro-parentheses)

#endif // defined(__CUDACC__)

#else

#define STRF_DEF_SINGLE_BYTE_CHARSET(CHARSET) STRF_DEF_SINGLE_BYTE_CHARSET_(CHARSET)

#endif // defined(STRF_HAS_VARIABLE_TEMPLATE)

namespace strf {

namespace detail {

template <typename T>
struct sbcs_find_invalid_seq_crtp
{
    template <typename CharT>
    static STRF_HD strf::transcode_size_result<CharT> find_invalid_sequence
        ( const CharT* src, const CharT* src_end, std::ptrdiff_t limit ) noexcept
    {
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            for(const auto* it = src; it < src_end; ++it) {
                if (!T::is_valid(static_cast<std::uint8_t>(*it))) {
                    return {it - src, it, stop_reason::invalid_sequence};;
                }
            }
            return {src_end - src, src_end, stop_reason::completed};
        }
        const auto* const src_limit = src + limit;
        for(const auto* it = src; ; ++it) {
            if (!T::is_valid(static_cast<std::uint8_t>(*it))) {
                return {it - src, it, stop_reason::invalid_sequence};
            }
            if (it == src_limit) {
                return {it - src, it, stop_reason::insufficient_output_space};
            }
        }
    }
};

template <typename T>
struct sbcs_never_has_invalid_seq_crtp
{
    template <typename CharT>
    static STRF_HD strf::transcode_size_result<CharT> find_invalid_sequence
        ( const CharT* src, const CharT* src_end, std::ptrdiff_t limit ) noexcept
    {
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }
};

template <typename T>
struct sbcs_find_unsupported_codepoint_crtp
{
    template <typename CharT>
    static STRF_HD strf::transcode_size_result<CharT> find_first_invalid_codepoint
        ( const CharT* src
        , const CharT* src_end
        , std::ptrdiff_t limit
        , bool strict_surrogates ) noexcept
    {
        using stop_reason = strf::transcode_stop_reason;
        static_assert(sizeof(CharT) == 4, "");

        if (src_end - src <= limit) {
            if (strict_surrogates) {
                for(auto it = src; it < src_end; ++it) {
                    if (*it >= 0x110000 || (0xD800 <= *it && *it <= 0xFFFF)) {
                        return {it - src, it, stop_reason::invalid_sequence};
                    }
                }
            } else {
                for(auto it = src; it < src_end; ++it) {
                    if (*it >= 0x110000) {
                        return {it - src, it, stop_reason::invalid_sequence};
                    }
                }
            }
            return {src_end - src, src_end, stop_reason::completed};
        }
        const auto* const src_limit = src + limit;
        if (strict_surrogates) {
            for(auto it = src; ; ++it) {
                if (*it >= 0x110000 || (0xD800 <= *it && *it <= 0xFFFF)) {
                    return {it - src, it, stop_reason::invalid_sequence};
                }
                if (it == src_limit) {
                    return {it - src, it, stop_reason::insufficient_output_space};
                }
            }
        }
        for(auto it = src; ; ++it) {
            if (*it >= 0x110000) {
                return {it - src, it, stop_reason::invalid_sequence};
            }
            if (it == src_limit) {
                return {it - src, it, stop_reason::insufficient_output_space};
            }
        }
    }

    template <typename CharT>
    static STRF_HD strf::transcode_size_result<CharT> find_first_unsupported_or_invalid_codepoint
        ( const CharT* src
        , const CharT* src_end
        , std::ptrdiff_t limit
        , bool strict_surrogates ) noexcept
    {
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            if (strict_surrogates) {
                for(auto it = src; it < src_end; ++it) {
                    if (*it >= 0x110000 || (0xD800 <= *it && *it <= 0xFFFF)) {
                        return {it - src, it, stop_reason::invalid_sequence};
                    }
                    if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                        return {it - src, it, stop_reason::unsupported_codepoint};
                    }
                }
            } else {
                for(auto it = src; it < src_end; ++it) {
                    if (*it >= 0x110000) {
                        return {it - src, it, stop_reason::invalid_sequence};
                    }
                    if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                        return {it - src, it, stop_reason::unsupported_codepoint};
                    }
                }
            }
            return {src_end - src, src_end, stop_reason::completed};
        }
        const auto* const src_limit = src + limit;
        if (strict_surrogates) {
            for(auto it = src; ; ++it) {
                if (*it >= 0x110000 || (0xD800 <= *it && *it <= 0xFFFF)) {
                    return {it - src, it, stop_reason::invalid_sequence};
                }
                if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                    return {it - src, it, stop_reason::unsupported_codepoint};
                }
                if (it == src_limit) {
                    return {it - src, it, stop_reason::insufficient_output_space};
                }
            }
        }
        for(auto it = src; ; ++it) {
            if (*it >= 0x110000) {
                return {it - src, it, stop_reason::invalid_sequence};
            }
            if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                return {it - src, it, stop_reason::unsupported_codepoint};
            }
            if (it == src_limit) {
                return {it - src, it, stop_reason::insufficient_output_space};
            }
        }
    }

    template <typename CharT>
    static STRF_HD strf::transcode_size_result<CharT> find_first_valid_unsupported_codepoint
        ( const CharT* src
        , const CharT* src_end
        , std::ptrdiff_t limit
        , bool strict_surrogates ) noexcept
    {
        static_assert(sizeof(CharT) == 4, "");
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            if (strict_surrogates) {
                for(auto it = src; it < src_end; ++it) {
                    if ( (*it <= 0xD800 || (0xFFFF <= *it && *it <= 0x10FFFF))
                         && T::encode(static_cast<char32_t>(*it)) >= 0x100)
                    {
                        return {it - src, it, stop_reason::unsupported_codepoint};
                    }
                }
            } else {
                for(auto it = src; it < src_end; ++it) {
                    if (*it <= 0x110000 && T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                        return {it - src, it, stop_reason::unsupported_codepoint};
                    }
                }
            }
            return {src_end - src, src_end, stop_reason::completed};
        }

        const auto* const src_limit = src + limit;
        if (strict_surrogates) {
            for(auto it = src; ; ++it) {
                if ( (*it <= 0xD800 || (0xFFFF <= *it && *it <= 0x10FFFF))
                     && T::encode(static_cast<char32_t>(*it)) >= 0x100)
                {
                    return {it - src, it, stop_reason::unsupported_codepoint};
                }
                if (it == src_limit) {
                    return {it - src, it, stop_reason::insufficient_output_space};
                }
            }
        }
        for(auto it = src; ; ++it) {
            if (*it <= 0x110000 && T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                return {it - src, it, stop_reason::unsupported_codepoint};
            }
            if (it == src_limit) {
                return {src_end - src, src_end, stop_reason::insufficient_output_space};
            }
        }
    }


   template <typename CharT>
   static STRF_HD strf::transcode_size_result<CharT> find_first_unsupported_codepoint
        ( const CharT* src
        , const CharT* src_end
        , std::ptrdiff_t limit ) noexcept
    {
        static_assert(sizeof(CharT) == 4, "");
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            for(auto it = src; it < src_end; ++it) {
                if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                    return {it - src, it, stop_reason::unsupported_codepoint};
                }
            }
            return {src_end - src, src_end, stop_reason::completed};
        }
        const auto* const src_limit = src + limit;
        for(auto it = src; ; ++it) {
            if (T::encode(static_cast<char32_t>(*it)) >= 0x100) {
                return {it - src, it, stop_reason::unsupported_codepoint};
            }
            if (it == src_limit) {
                return {it - src, it, stop_reason::insufficient_output_space};
            }
        }
    }

};

template<size_t SIZE, class T>
constexpr STRF_HD size_t array_size(T (&)[SIZE]) {
    return SIZE;
}

struct ch32_to_char
{
    char32_t key;
    unsigned value;
};

struct cmp_ch32_to_char
{
    STRF_HD bool operator()(ch32_to_char a, ch32_to_char b) const
    {
        return a.key < b.key;
    }
};

struct impl_ascii
    : sbcs_find_invalid_seq_crtp<impl_ascii>
    , sbcs_find_unsupported_codepoint_crtp<impl_ascii>
{
    static STRF_HD const char* name() noexcept
    {
        return "ASCII";
    };
    static constexpr strf::charset_id id = strf::csid_ascii;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch < 0x80;
    }
    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch < 0x80)
            return ch;
        return 0xFFFD;
    }
    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return ch < 0x80 ? ch : (ch == 0xFFFD ? '?' : 0x100);
    }
};

struct impl_iso_8859_1
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_1>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_1>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-1";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_1;

    static STRF_HD bool is_valid(std::uint8_t)
    {
        return true;
    }
    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        return ch;
    }
    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return ch == 0xFFFD ? '?' : ch;
    }
};

struct impl_iso_8859_2
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_2>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_2>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-2";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_2;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0104, 0x02D8, 0x0141, 0x00A4, 0x013D, 0x015A, 0x00A7
            , 0x00A8, 0x0160, 0x015E, 0x0164, 0x0179, 0x00AD, 0x017D, 0x017B
            , 0x00B0, 0x0105, 0x02DB, 0x0142, 0x00B4, 0x013E, 0x015B, 0x02C7
            , 0x00B8, 0x0161, 0x015F, 0x0165, 0x017A, 0x02DD, 0x017E, 0x017C
            , 0x0154, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7
            , 0x010C, 0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E
            , 0x0110, 0x0143, 0x0147, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x00D7
            , 0x0158, 0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF
            , 0x0155, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7
            , 0x010D, 0x00E9, 0x0119, 0x00EB, 0x011B, 0x00ED, 0x00EE, 0x010F
            , 0x0111, 0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7
            , 0x0159, 0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9 };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_2::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A4, 0xA4}, {0x00A7, 0xA7}, {0x00A8, 0xA8}, {0x00AD, 0xAD}
        , {0x00B0, 0xB0}, {0x00B4, 0xB4}, {0x00B8, 0xB8}, {0x00C1, 0xC1}
        , {0x00C2, 0xC2}, {0x00C4, 0xC4}, {0x00C7, 0xC7}, {0x00C9, 0xC9}
        , {0x00CB, 0xCB}, {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00D3, 0xD3}
        , {0x00D4, 0xD4}, {0x00D6, 0xD6}, {0x00D7, 0xD7}, {0x00DA, 0xDA}
        , {0x00DC, 0xDC}, {0x00DD, 0xDD}, {0x00DF, 0xDF}, {0x00E1, 0xE1}
        , {0x00E2, 0xE2}, {0x00E4, 0xE4}, {0x00E7, 0xE7}, {0x00E9, 0xE9}
        , {0x00EB, 0xEB}, {0x00ED, 0xED}, {0x00EE, 0xEE}, {0x00F3, 0xF3}
        , {0x00F4, 0xF4}, {0x00F6, 0xF6}, {0x00F7, 0xF7}, {0x00FA, 0xFA}
        , {0x00FC, 0xFC}, {0x00FD, 0xFD}, {0x0102, 0xC3}, {0x0103, 0xE3}
        , {0x0104, 0xA1}, {0x0105, 0xB1}, {0x0106, 0xC6}, {0x0107, 0xE6}
        , {0x010C, 0xC8}, {0x010D, 0xE8}, {0x010E, 0xCF}, {0x010F, 0xEF}
        , {0x0110, 0xD0}, {0x0111, 0xF0}, {0x0118, 0xCA}, {0x0119, 0xEA}
        , {0x011A, 0xCC}, {0x011B, 0xEC}, {0x0139, 0xC5}, {0x013A, 0xE5}
        , {0x013D, 0xA5}, {0x013E, 0xB5}, {0x0141, 0xA3}, {0x0142, 0xB3}
        , {0x0143, 0xD1}, {0x0144, 0xF1}, {0x0147, 0xD2}, {0x0148, 0xF2}
        , {0x0150, 0xD5}, {0x0151, 0xF5}, {0x0154, 0xC0}, {0x0155, 0xE0}
        , {0x0158, 0xD8}, {0x0159, 0xF8}, {0x015A, 0xA6}, {0x015B, 0xB6}
        , {0x015E, 0xAA}, {0x015F, 0xBA}, {0x0160, 0xA9}, {0x0161, 0xB9}
        , {0x0162, 0xDE}, {0x0163, 0xFE}, {0x0164, 0xAB}, {0x0165, 0xBB}
        , {0x016E, 0xD9}, {0x016F, 0xF9}, {0x0170, 0xDB}, {0x0171, 0xFB}
        , {0x0179, 0xAC}, {0x017A, 0xBC}, {0x017B, 0xAF}, {0x017C, 0xBF}
        , {0x017D, 0xAE}, {0x017E, 0xBE}, {0x02C7, 0xB7}, {0x02D8, 0xA2}
        , {0x02D9, 0xFF}, {0x02DB, 0xB2}, {0x02DD, 0xBD}, {0xFFFD, '?' } };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_3
    : sbcs_find_invalid_seq_crtp<impl_iso_8859_3>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_3>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-3";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_3;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch < 0xA5 || ( ch != 0xA5 && ch != 0xAE && ch != 0xBE && ch != 0xC3 &&
                              ch != 0xD0 && ch != 0xE3 && ch != 0xF0 );
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        constexpr std::uint16_t undef = 0xFFFD;
        static const std::uint16_t ext[] =
            { /* A0*/ 0x0126, 0x02D8, 0x00A3, 0x00A4,  undef, 0x0124, 0x00A7
            , 0x00A8, 0x0130, 0x015E, 0x011E, 0x0134, 0x00AD,  undef, 0x017B
            , 0x00B0, 0x0127, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x0125, 0x00B7
            , 0x00B8, 0x0131, 0x015F, 0x011F, 0x0135, 0x00BD,  undef, 0x017C
            , 0x00C0, 0x00C1, 0x00C2,  undef, 0x00C4, 0x010A, 0x0108, 0x00C7
            , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF
            ,  undef, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x0120, 0x00D6, 0x00D7
            , 0x011C, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x016C, 0x015C, 0x00DF
            , 0x00E0, 0x00E1, 0x00E2,  undef, 0x00E4, 0x010B, 0x0109, 0x00E7
            , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF
            ,  undef, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x0121, 0x00F6, 0x00F7
            , 0x011D, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x016D, 0x015D, 0x02D9 };

        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_3::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A3, 0xA3}, {0x00A4, 0xA4}, {0x00A7, 0xA7}, {0x00A8, 0xA8}
        , {0x00AD, 0xAD}, {0x00B0, 0xB0}, {0x00B2, 0xB2}, {0x00B3, 0xB3}
        , {0x00B4, 0xB4}, {0x00B5, 0xB5}, {0x00B7, 0xB7}, {0x00B8, 0xB8}
        , {0x00BD, 0xBD}, {0x00C0, 0xC0}, {0x00C1, 0xC1}, {0x00C2, 0xC2}
        , {0x00C4, 0xC4}, {0x00C7, 0xC7}, {0x00C8, 0xC8}, {0x00C9, 0xC9}
        , {0x00CA, 0xCA}, {0x00CB, 0xCB}, {0x00CC, 0xCC}, {0x00CD, 0xCD}
        , {0x00CE, 0xCE}, {0x00CF, 0xCF}, {0x00D1, 0xD1}, {0x00D2, 0xD2}
        , {0x00D3, 0xD3}, {0x00D4, 0xD4}, {0x00D6, 0xD6}, {0x00D7, 0xD7}
        , {0x00D9, 0xD9}, {0x00DA, 0xDA}, {0x00DB, 0xDB}, {0x00DC, 0xDC}
        , {0x00DF, 0xDF}, {0x00E0, 0xE0}, {0x00E1, 0xE1}, {0x00E2, 0xE2}
        , {0x00E4, 0xE4}, {0x00E7, 0xE7}, {0x00E8, 0xE8}, {0x00E9, 0xE9}
        , {0x00EA, 0xEA}, {0x00EB, 0xEB}, {0x00EC, 0xEC}, {0x00ED, 0xED}
        , {0x00EE, 0xEE}, {0x00EF, 0xEF}, {0x00F1, 0xF1}, {0x00F2, 0xF2}
        , {0x00F3, 0xF3}, {0x00F4, 0xF4}, {0x00F6, 0xF6}, {0x00F7, 0xF7}
        , {0x00F9, 0xF9}, {0x00FA, 0xFA}, {0x00FB, 0xFB}, {0x00FC, 0xFC}
        , {0x0108, 0xC6}, {0x0109, 0xE6}, {0x010A, 0xC5}, {0x010B, 0xE5}
        , {0x011C, 0xD8}, {0x011D, 0xF8}, {0x011E, 0xAB}, {0x011F, 0xBB}
        , {0x0120, 0xD5}, {0x0121, 0xF5}, {0x0124, 0xA6}, {0x0125, 0xB6}
        , {0x0126, 0xA1}, {0x0127, 0xB1}, {0x0130, 0xA9}, {0x0131, 0xB9}
        , {0x0134, 0xAC}, {0x0135, 0xBC}, {0x015C, 0xDE}, {0x015D, 0xFE}
        , {0x015E, 0xAA}, {0x015F, 0xBA}, {0x016C, 0xDD}, {0x016D, 0xFD}
        , {0x017B, 0xAF}, {0x017C, 0xBF}, {0x02D8, 0xA2}, {0x02D9, 0xFF}
        , {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_4
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_4>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_4>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-4";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_4;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0104, 0x0138, 0x0156, 0x00A4, 0x0128, 0x013B, 0x00A7
            , 0x00A8, 0x0160, 0x0112, 0x0122, 0x0166, 0x00AD, 0x017D, 0x00AF
            , 0x00B0, 0x0105, 0x02DB, 0x0157, 0x00B4, 0x0129, 0x013C, 0x02C7
            , 0x00B8, 0x0161, 0x0113, 0x0123, 0x0167, 0x014A, 0x017E, 0x014B
            , 0x0100, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x012E
            , 0x010C, 0x00C9, 0x0118, 0x00CB, 0x0116, 0x00CD, 0x00CE, 0x012A
            , 0x0110, 0x0145, 0x014C, 0x0136, 0x00D4, 0x00D5, 0x00D6, 0x00D7
            , 0x00D8, 0x0172, 0x00DA, 0x00DB, 0x00DC, 0x0168, 0x016A, 0x00DF
            , 0x0101, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x012F
            , 0x010D, 0x00E9, 0x0119, 0x00EB, 0x0117, 0x00ED, 0x00EE, 0x012B
            , 0x0111, 0x0146, 0x014D, 0x0137, 0x00F4, 0x00F5, 0x00F6, 0x00F7
            , 0x00F8, 0x0173, 0x00FA, 0x00FB, 0x00FC, 0x0169, 0x016B, 0x02D9 };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_4::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A4, 0xA4}, {0x00A7, 0xA7}, {0x00A8, 0xA8}, {0x00AD, 0xAD}
        , {0x00AF, 0xAF}, {0x00B0, 0xB0}, {0x00B4, 0xB4}, {0x00B8, 0xB8}
        , {0x00C1, 0xC1}, {0x00C2, 0xC2}, {0x00C3, 0xC3}, {0x00C4, 0xC4}
        , {0x00C5, 0xC5}, {0x00C6, 0xC6}, {0x00C9, 0xC9}, {0x00CB, 0xCB}
        , {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00D4, 0xD4}, {0x00D5, 0xD5}
        , {0x00D6, 0xD6}, {0x00D7, 0xD7}, {0x00D8, 0xD8}, {0x00DA, 0xDA}
        , {0x00DB, 0xDB}, {0x00DC, 0xDC}, {0x00DF, 0xDF}, {0x00E1, 0xE1}
        , {0x00E2, 0xE2}, {0x00E3, 0xE3}, {0x00E4, 0xE4}, {0x00E5, 0xE5}
        , {0x00E6, 0xE6}, {0x00E9, 0xE9}, {0x00EB, 0xEB}, {0x00ED, 0xED}
        , {0x00EE, 0xEE}, {0x00F4, 0xF4}, {0x00F5, 0xF5}, {0x00F6, 0xF6}
        , {0x00F7, 0xF7}, {0x00F8, 0xF8}, {0x00FA, 0xFA}, {0x00FB, 0xFB}
        , {0x00FC, 0xFC}, {0x0100, 0xC0}, {0x0101, 0xE0}, {0x0104, 0xA1}
        , {0x0105, 0xB1}, {0x010C, 0xC8}, {0x010D, 0xE8}, {0x0110, 0xD0}
        , {0x0111, 0xF0}, {0x0112, 0xAA}, {0x0113, 0xBA}, {0x0116, 0xCC}
        , {0x0117, 0xEC}, {0x0118, 0xCA}, {0x0119, 0xEA}, {0x0122, 0xAB}
        , {0x0123, 0xBB}, {0x0128, 0xA5}, {0x0129, 0xB5}, {0x012A, 0xCF}
        , {0x012B, 0xEF}, {0x012E, 0xC7}, {0x012F, 0xE7}, {0x0136, 0xD3}
        , {0x0137, 0xF3}, {0x0138, 0xA2}, {0x013B, 0xA6}, {0x013C, 0xB6}
        , {0x0145, 0xD1}, {0x0146, 0xF1}, {0x014A, 0xBD}, {0x014B, 0xBF}
        , {0x014C, 0xD2}, {0x014D, 0xF2}, {0x0156, 0xA3}, {0x0157, 0xB3}
        , {0x0160, 0xA9}, {0x0161, 0xB9}, {0x0166, 0xAC}, {0x0167, 0xBC}
        , {0x0168, 0xDD}, {0x0169, 0xFD}, {0x016A, 0xDE}, {0x016B, 0xFE}
        , {0x0172, 0xD9}, {0x0173, 0xF9}, {0x017D, 0xAE}, {0x017E, 0xBE}
        , {0x02C7, 0xB7}, {0x02D9, 0xFF}, {0x02DB, 0xB2}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_5
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_5>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_5>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-5";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_5;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407
            , 0x0408, 0x0409, 0x040A, 0x040B, 0x040C, 0x00AD, 0x040E, 0x040F
            , 0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417
            , 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F
            , 0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427
            , 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F
            , 0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437
            , 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F
            , 0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447
            , 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F
            , 0x2116, 0x0451, 0x0452, 0x0453, 0x0454, 0x0455, 0x0456, 0x0457
            , 0x0458, 0x0459, 0x045A, 0x045B, 0x045C, 0x00A7, 0x045E, 0x045F };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_5::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A7, 0xFD}, {0x00AD, 0xAD}, {0x0401, 0xA1}, {0x0402, 0xA2}
        , {0x0403, 0xA3}, {0x0404, 0xA4}, {0x0405, 0xA5}, {0x0406, 0xA6}
        , {0x0407, 0xA7}, {0x0408, 0xA8}, {0x0409, 0xA9}, {0x040A, 0xAA}
        , {0x040B, 0xAB}, {0x040C, 0xAC}, {0x040E, 0xAE}, {0x040F, 0xAF}
        , {0x0410, 0xB0}, {0x0411, 0xB1}, {0x0412, 0xB2}, {0x0413, 0xB3}
        , {0x0414, 0xB4}, {0x0415, 0xB5}, {0x0416, 0xB6}, {0x0417, 0xB7}
        , {0x0418, 0xB8}, {0x0419, 0xB9}, {0x041A, 0xBA}, {0x041B, 0xBB}
        , {0x041C, 0xBC}, {0x041D, 0xBD}, {0x041E, 0xBE}, {0x041F, 0xBF}
        , {0x0420, 0xC0}, {0x0421, 0xC1}, {0x0422, 0xC2}, {0x0423, 0xC3}
        , {0x0424, 0xC4}, {0x0425, 0xC5}, {0x0426, 0xC6}, {0x0427, 0xC7}
        , {0x0428, 0xC8}, {0x0429, 0xC9}, {0x042A, 0xCA}, {0x042B, 0xCB}
        , {0x042C, 0xCC}, {0x042D, 0xCD}, {0x042E, 0xCE}, {0x042F, 0xCF}
        , {0x0430, 0xD0}, {0x0431, 0xD1}, {0x0432, 0xD2}, {0x0433, 0xD3}
        , {0x0434, 0xD4}, {0x0435, 0xD5}, {0x0436, 0xD6}, {0x0437, 0xD7}
        , {0x0438, 0xD8}, {0x0439, 0xD9}, {0x043A, 0xDA}, {0x043B, 0xDB}
        , {0x043C, 0xDC}, {0x043D, 0xDD}, {0x043E, 0xDE}, {0x043F, 0xDF}
        , {0x0440, 0xE0}, {0x0441, 0xE1}, {0x0442, 0xE2}, {0x0443, 0xE3}
        , {0x0444, 0xE4}, {0x0445, 0xE5}, {0x0446, 0xE6}, {0x0447, 0xE7}
        , {0x0448, 0xE8}, {0x0449, 0xE9}, {0x044A, 0xEA}, {0x044B, 0xEB}
        , {0x044C, 0xEC}, {0x044D, 0xED}, {0x044E, 0xEE}, {0x044F, 0xEF}
        , {0x0451, 0xF1}, {0x0452, 0xF2}, {0x0453, 0xF3}, {0x0454, 0xF4}
        , {0x0455, 0xF5}, {0x0456, 0xF6}, {0x0457, 0xF7}, {0x0458, 0xF8}
        , {0x0459, 0xF9}, {0x045A, 0xFA}, {0x045B, 0xFB}, {0x045C, 0xFC}
        , {0x045E, 0xFE}, {0x045F, 0xFF}, {0x2116, 0xF0}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_6
    : sbcs_find_invalid_seq_crtp<impl_iso_8859_6>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_6>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-6";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_6;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch <= 0xA0 || ch == 0xA4 || ch == 0xAC || ch == 0xAD
            || ch == 0xBB || ch == 0xBF || (0xC1 <= ch && ch <= 0xDA)
            || (0xE0 <= ch && ch <= 0xF2);
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0xFFFD, 0xFFFD, 0xFFFD, 0x00A4, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x060C, 0x00AD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0x061B, 0xFFFD, 0xFFFD, 0xFFFD, 0x061F
            , 0xFFFD, 0x0621, 0x0622, 0x0623, 0x0624, 0x0625, 0x0626, 0x0627
            , 0x0628, 0x0629, 0x062A, 0x062B, 0x062C, 0x062D, 0x062E, 0x062F
            , 0x0630, 0x0631, 0x0632, 0x0633, 0x0634, 0x0635, 0x0636, 0x0637
            , 0x0638, 0x0639, 0x063A, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0x0640, 0x0641, 0x0642, 0x0643, 0x0644, 0x0645, 0x0646, 0x0647
            , 0x0648, 0x0649, 0x064A, 0x064B, 0x064C, 0x064D, 0x064E, 0x064F
            , 0x0650, 0x0651, 0x0652, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_6::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A4, 0xA4}, {0x00AD, 0xAD}, {0x060C, 0xAC}, {0x061B, 0xBB}
        , {0x061F, 0xBF}, {0x0621, 0xC1}, {0x0622, 0xC2}, {0x0623, 0xC3}
        , {0x0624, 0xC4}, {0x0625, 0xC5}, {0x0626, 0xC6}, {0x0627, 0xC7}
        , {0x0628, 0xC8}, {0x0629, 0xC9}, {0x062A, 0xCA}, {0x062B, 0xCB}
        , {0x062C, 0xCC}, {0x062D, 0xCD}, {0x062E, 0xCE}, {0x062F, 0xCF}
        , {0x0630, 0xD0}, {0x0631, 0xD1}, {0x0632, 0xD2}, {0x0633, 0xD3}
        , {0x0634, 0xD4}, {0x0635, 0xD5}, {0x0636, 0xD6}, {0x0637, 0xD7}
        , {0x0638, 0xD8}, {0x0639, 0xD9}, {0x063A, 0xDA}, {0x0640, 0xE0}
        , {0x0641, 0xE1}, {0x0642, 0xE2}, {0x0643, 0xE3}, {0x0644, 0xE4}
        , {0x0645, 0xE5}, {0x0646, 0xE6}, {0x0647, 0xE7}, {0x0648, 0xE8}
        , {0x0649, 0xE9}, {0x064A, 0xEA}, {0x064B, 0xEB}, {0x064C, 0xEC}
        , {0x064D, 0xED}, {0x064E, 0xEE}, {0x064F, 0xEF}, {0x0650, 0xF0}
        , {0x0651, 0xF1}, {0x0652, 0xF2}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_7
    : sbcs_find_invalid_seq_crtp<impl_iso_8859_7>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_7>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-7";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_7;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch < 0xAE || (ch != 0xAE && ch != 0xD2 && ch != 0xFF);
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x2018, 0x2019, 0x00A3, 0x20AC, 0x20AF, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x037A, 0x00AB, 0x00AC, 0x00AD, 0xFFFD, 0x2015
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x0384, 0x0385, 0x0386, 0x00B7
            , 0x0388, 0x0389, 0x038A, 0x00BB, 0x038C, 0x00BD, 0x038E, 0x038F
            , 0x0390, 0x0391, 0x0392, 0x0393, 0x0394, 0x0395, 0x0396, 0x0397
            , 0x0398, 0x0399, 0x039A, 0x039B, 0x039C, 0x039D, 0x039E, 0x039F
            , 0x03A0, 0x03A1, 0xFFFD, 0x03A3, 0x03A4, 0x03A5, 0x03A6, 0x03A7
            , 0x03A8, 0x03A9, 0x03AA, 0x03AB, 0x03AC, 0x03AD, 0x03AE, 0x03AF
            , 0x03B0, 0x03B1, 0x03B2, 0x03B3, 0x03B4, 0x03B5, 0x03B6, 0x03B7
            , 0x03B8, 0x03B9, 0x03BA, 0x03BB, 0x03BC, 0x03BD, 0x03BE, 0x03BF
            , 0x03C0, 0x03C1, 0x03C2, 0x03C3, 0x03C4, 0x03C5, 0x03C6, 0x03C7
            , 0x03C8, 0x03C9, 0x03CA, 0x03CB, 0x03CC, 0x03CD, 0x03CE, 0xFFFD };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_7::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A3, 0xA3}, {0x00A6, 0xA6}, {0x00A7, 0xA7}, {0x00A8, 0xA8}
        , {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}, {0x00AD, 0xAD}
        , {0x00B0, 0xB0}, {0x00B1, 0xB1}, {0x00B2, 0xB2}, {0x00B3, 0xB3}
        , {0x00B7, 0xB7}, {0x00BB, 0xBB}, {0x00BD, 0xBD}, {0x037A, 0xAA}
        , {0x0384, 0xB4}, {0x0385, 0xB5}, {0x0386, 0xB6}, {0x0388, 0xB8}
        , {0x0389, 0xB9}, {0x038A, 0xBA}, {0x038C, 0xBC}, {0x038E, 0xBE}
        , {0x038F, 0xBF}, {0x0390, 0xC0}, {0x0391, 0xC1}, {0x0392, 0xC2}
        , {0x0393, 0xC3}, {0x0394, 0xC4}, {0x0395, 0xC5}, {0x0396, 0xC6}
        , {0x0397, 0xC7}, {0x0398, 0xC8}, {0x0399, 0xC9}, {0x039A, 0xCA}
        , {0x039B, 0xCB}, {0x039C, 0xCC}, {0x039D, 0xCD}, {0x039E, 0xCE}
        , {0x039F, 0xCF}, {0x03A0, 0xD0}, {0x03A1, 0xD1}, {0x03A3, 0xD3}
        , {0x03A4, 0xD4}, {0x03A5, 0xD5}, {0x03A6, 0xD6}, {0x03A7, 0xD7}
        , {0x03A8, 0xD8}, {0x03A9, 0xD9}, {0x03AA, 0xDA}, {0x03AB, 0xDB}
        , {0x03AC, 0xDC}, {0x03AD, 0xDD}, {0x03AE, 0xDE}, {0x03AF, 0xDF}
        , {0x03B0, 0xE0}, {0x03B1, 0xE1}, {0x03B2, 0xE2}, {0x03B3, 0xE3}
        , {0x03B4, 0xE4}, {0x03B5, 0xE5}, {0x03B6, 0xE6}, {0x03B7, 0xE7}
        , {0x03B8, 0xE8}, {0x03B9, 0xE9}, {0x03BA, 0xEA}, {0x03BB, 0xEB}
        , {0x03BC, 0xEC}, {0x03BD, 0xED}, {0x03BE, 0xEE}, {0x03BF, 0xEF}
        , {0x03C0, 0xF0}, {0x03C1, 0xF1}, {0x03C2, 0xF2}, {0x03C3, 0xF3}
        , {0x03C4, 0xF4}, {0x03C5, 0xF5}, {0x03C6, 0xF6}, {0x03C7, 0xF7}
        , {0x03C8, 0xF8}, {0x03C9, 0xF9}, {0x03CA, 0xFA}, {0x03CB, 0xFB}
        , {0x03CC, 0xFC}, {0x03CD, 0xFD}, {0x03CE, 0xFE}, {0x2015, 0xAF}
        , {0x2018, 0xA1}, {0x2019, 0xA2}, {0x20AC, 0xA4}, {0x20AF, 0xA5}
        , {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_8
    : sbcs_find_invalid_seq_crtp<impl_iso_8859_8>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_8>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-8";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_8;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return (ch <= 0xBE && ch != 0xA1) || (0xDF <= ch && ch <= 0xFA)
            || ch == 0xFD || ch == 0xFE;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0xFFFD, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x00D7, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00B8, 0x00B9, 0x00F7, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x2017
            , 0x05D0, 0x05D1, 0x05D2, 0x05D3, 0x05D4, 0x05D5, 0x05D6, 0x05D7
            , 0x05D8, 0x05D9, 0x05DA, 0x05DB, 0x05DC, 0x05DD, 0x05DE, 0x05DF
            , 0x05E0, 0x05E1, 0x05E2, 0x05E3, 0x05E4, 0x05E5, 0x05E6, 0x05E7
            , 0x05E8, 0x05E9, 0x05EA, 0xFFFD, 0xFFFD, 0x200E, 0x200F, 0xFFFD };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_8::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A2, 0xA2}, {0x00A3, 0xA3}, {0x00A4, 0xA4}, {0x00A5, 0xA5}
        , {0x00A6, 0xA6}, {0x00A7, 0xA7}, {0x00A8, 0xA8}, {0x00A9, 0xA9}
        , {0x00AB, 0xAB}, {0x00AC, 0xAC}, {0x00AD, 0xAD}, {0x00AE, 0xAE}
        , {0x00AF, 0xAF}, {0x00B0, 0xB0}, {0x00B1, 0xB1}, {0x00B2, 0xB2}
        , {0x00B3, 0xB3}, {0x00B4, 0xB4}, {0x00B5, 0xB5}, {0x00B6, 0xB6}
        , {0x00B7, 0xB7}, {0x00B8, 0xB8}, {0x00B9, 0xB9}, {0x00BB, 0xBB}
        , {0x00BC, 0xBC}, {0x00BD, 0xBD}, {0x00BE, 0xBE}, {0x00D7, 0xAA}
        , {0x00F7, 0xBA}, {0x05D0, 0xE0}, {0x05D1, 0xE1}, {0x05D2, 0xE2}
        , {0x05D3, 0xE3}, {0x05D4, 0xE4}, {0x05D5, 0xE5}, {0x05D6, 0xE6}
        , {0x05D7, 0xE7}, {0x05D8, 0xE8}, {0x05D9, 0xE9}, {0x05DA, 0xEA}
        , {0x05DB, 0xEB}, {0x05DC, 0xEC}, {0x05DD, 0xED}, {0x05DE, 0xEE}
        , {0x05DF, 0xEF}, {0x05E0, 0xF0}, {0x05E1, 0xF1}, {0x05E2, 0xF2}
        , {0x05E3, 0xF3}, {0x05E4, 0xF4}, {0x05E5, 0xF5}, {0x05E6, 0xF6}
        , {0x05E7, 0xF7}, {0x05E8, 0xF8}, {0x05E9, 0xF9}, {0x05EA, 0xFA}
        , {0x200E, 0xFD}, {0x200F, 0xFE}, {0x2017, 0xDF}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_9
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_9>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_9>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-9";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_9;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        STRF_IF_LIKELY (ch <= 0xCF) {
            return ch;
        }
        switch (ch) {
            case 0x011E: return 0xD0;
            case 0x0130: return 0xDD;
            case 0x015E: return 0xDE;
            case 0x011F: return 0xF0;
            case 0x0131: return 0xFD;
            case 0x015F: return 0xFE;
            case 0xD0:
            case 0xDD:
            case 0xDE:
            case 0xF0:
            case 0xFD:
            case 0xFE:
                return 0x100;
            default:
                return ch;
        }
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch <= 0xCF) {
            return ch;
        }
        switch (ch) {
            case 0xD0: return 0x011E;
            case 0xDD: return 0x0130;
            case 0xDE: return 0x015E;
            case 0xF0: return 0x011F;
            case 0xFD: return 0x0131;
            case 0xFE: return 0x015F;
            default:
                return ch;
        }
    }
};

struct impl_iso_8859_10
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_10>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_10>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-10";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_10;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0104, 0x0112, 0x0122, 0x012A, 0x0128, 0x0136, 0x00A7
            , 0x013B, 0x0110, 0x0160, 0x0166, 0x017D, 0x00AD, 0x016A, 0x014A
            , 0x00B0, 0x0105, 0x0113, 0x0123, 0x012B, 0x0129, 0x0137, 0x00B7
            , 0x013C, 0x0111, 0x0161, 0x0167, 0x017E, 0x2015, 0x016B, 0x014B
            , 0x0100, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x012E
            , 0x010C, 0x00C9, 0x0118, 0x00CB, 0x0116, 0x00CD, 0x00CE, 0x00CF
            , 0x00D0, 0x0145, 0x014C, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x0168
            , 0x00D8, 0x0172, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF
            , 0x0101, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x012F
            , 0x010D, 0x00E9, 0x0119, 0x00EB, 0x0117, 0x00ED, 0x00EE, 0x00EF
            , 0x00F0, 0x0146, 0x014D, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x0169
            , 0x00F8, 0x0173, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x0138 };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_10::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A7, 0xA7}, {0x00AD, 0xAD}, {0x00B0, 0xB0}, {0x00B7, 0xB7}
        , {0x00C1, 0xC1}, {0x00C2, 0xC2}, {0x00C3, 0xC3}, {0x00C4, 0xC4}
        , {0x00C5, 0xC5}, {0x00C6, 0xC6}, {0x00C9, 0xC9}, {0x00CB, 0xCB}
        , {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00CF, 0xCF}, {0x00D0, 0xD0}
        , {0x00D3, 0xD3}, {0x00D4, 0xD4}, {0x00D5, 0xD5}, {0x00D6, 0xD6}
        , {0x00D8, 0xD8}, {0x00DA, 0xDA}, {0x00DB, 0xDB}, {0x00DC, 0xDC}
        , {0x00DD, 0xDD}, {0x00DE, 0xDE}, {0x00DF, 0xDF}, {0x00E1, 0xE1}
        , {0x00E2, 0xE2}, {0x00E3, 0xE3}, {0x00E4, 0xE4}, {0x00E5, 0xE5}
        , {0x00E6, 0xE6}, {0x00E9, 0xE9}, {0x00EB, 0xEB}, {0x00ED, 0xED}
        , {0x00EE, 0xEE}, {0x00EF, 0xEF}, {0x00F0, 0xF0}, {0x00F3, 0xF3}
        , {0x00F4, 0xF4}, {0x00F5, 0xF5}, {0x00F6, 0xF6}, {0x00F8, 0xF8}
        , {0x00FA, 0xFA}, {0x00FB, 0xFB}, {0x00FC, 0xFC}, {0x00FD, 0xFD}
        , {0x00FE, 0xFE}, {0x0100, 0xC0}, {0x0101, 0xE0}, {0x0104, 0xA1}
        , {0x0105, 0xB1}, {0x010C, 0xC8}, {0x010D, 0xE8}, {0x0110, 0xA9}
        , {0x0111, 0xB9}, {0x0112, 0xA2}, {0x0113, 0xB2}, {0x0116, 0xCC}
        , {0x0117, 0xEC}, {0x0118, 0xCA}, {0x0119, 0xEA}, {0x0122, 0xA3}
        , {0x0123, 0xB3}, {0x0128, 0xA5}, {0x0129, 0xB5}, {0x012A, 0xA4}
        , {0x012B, 0xB4}, {0x012E, 0xC7}, {0x012F, 0xE7}, {0x0136, 0xA6}
        , {0x0137, 0xB6}, {0x0138, 0xFF}, {0x013B, 0xA8}, {0x013C, 0xB8}
        , {0x0145, 0xD1}, {0x0146, 0xF1}, {0x014A, 0xAF}, {0x014B, 0xBF}
        , {0x014C, 0xD2}, {0x014D, 0xF2}, {0x0160, 0xAA}, {0x0161, 0xBA}
        , {0x0166, 0xAB}, {0x0167, 0xBB}, {0x0168, 0xD7}, {0x0169, 0xF7}
        , {0x016A, 0xAE}, {0x016B, 0xBE}, {0x0172, 0xD9}, {0x0173, 0xF9}
        , {0x017D, 0xAC}, {0x017E, 0xBC}, {0x2015, 0xBD}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_11
    : sbcs_find_invalid_seq_crtp<impl_iso_8859_11>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_11>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-11";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_11;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch <= 0xDA || (ch >= 0xDF && ch <= 0xFB);
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0E01, 0x0E02, 0x0E03, 0x0E04, 0x0E05, 0x0E06, 0x0E07
            , 0x0E08, 0x0E09, 0x0E0A, 0x0E0B, 0x0E0C, 0x0E0D, 0x0E0E, 0x0E0F
            , 0x0E10, 0x0E11, 0x0E12, 0x0E13, 0x0E14, 0x0E15, 0x0E16, 0x0E17
            , 0x0E18, 0x0E19, 0x0E1A, 0x0E1B, 0x0E1C, 0x0E1D, 0x0E1E, 0x0E1F
            , 0x0E20, 0x0E21, 0x0E22, 0x0E23, 0x0E24, 0x0E25, 0x0E26, 0x0E27
            , 0x0E28, 0x0E29, 0x0E2A, 0x0E2B, 0x0E2C, 0x0E2D, 0x0E2E, 0x0E2F
            , 0x0E30, 0x0E31, 0x0E32, 0x0E33, 0x0E34, 0x0E35, 0x0E36, 0x0E37
            , 0x0E38, 0x0E39, 0x0E3A, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x0E3F
            , 0x0E40, 0x0E41, 0x0E42, 0x0E43, 0x0E44, 0x0E45, 0x0E46, 0x0E47
            , 0x0E48, 0x0E49, 0x0E4A, 0x0E4B, 0x0E4C, 0x0E4D, 0x0E4E, 0x0E4F
            , 0x0E50, 0x0E51, 0x0E52, 0x0E53, 0x0E54, 0x0E55, 0x0E56, 0x0E57
            , 0x0E58, 0x0E59, 0x0E5A, 0x0E5B, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_11::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0E01, 0xA1}, {0x0E02, 0xA2}, {0x0E03, 0xA3}, {0x0E04, 0xA4}
        , {0x0E05, 0xA5}, {0x0E06, 0xA6}, {0x0E07, 0xA7}, {0x0E08, 0xA8}
        , {0x0E09, 0xA9}, {0x0E0A, 0xAA}, {0x0E0B, 0xAB}, {0x0E0C, 0xAC}
        , {0x0E0D, 0xAD}, {0x0E0E, 0xAE}, {0x0E0F, 0xAF}, {0x0E10, 0xB0}
        , {0x0E11, 0xB1}, {0x0E12, 0xB2}, {0x0E13, 0xB3}, {0x0E14, 0xB4}
        , {0x0E15, 0xB5}, {0x0E16, 0xB6}, {0x0E17, 0xB7}, {0x0E18, 0xB8}
        , {0x0E19, 0xB9}, {0x0E1A, 0xBA}, {0x0E1B, 0xBB}, {0x0E1C, 0xBC}
        , {0x0E1D, 0xBD}, {0x0E1E, 0xBE}, {0x0E1F, 0xBF}, {0x0E20, 0xC0}
        , {0x0E21, 0xC1}, {0x0E22, 0xC2}, {0x0E23, 0xC3}, {0x0E24, 0xC4}
        , {0x0E25, 0xC5}, {0x0E26, 0xC6}, {0x0E27, 0xC7}, {0x0E28, 0xC8}
        , {0x0E29, 0xC9}, {0x0E2A, 0xCA}, {0x0E2B, 0xCB}, {0x0E2C, 0xCC}
        , {0x0E2D, 0xCD}, {0x0E2E, 0xCE}, {0x0E2F, 0xCF}, {0x0E30, 0xD0}
        , {0x0E31, 0xD1}, {0x0E32, 0xD2}, {0x0E33, 0xD3}, {0x0E34, 0xD4}
        , {0x0E35, 0xD5}, {0x0E36, 0xD6}, {0x0E37, 0xD7}, {0x0E38, 0xD8}
        , {0x0E39, 0xD9}, {0x0E3A, 0xDA}, {0x0E3F, 0xDF}, {0x0E40, 0xE0}
        , {0x0E41, 0xE1}, {0x0E42, 0xE2}, {0x0E43, 0xE3}, {0x0E44, 0xE4}
        , {0x0E45, 0xE5}, {0x0E46, 0xE6}, {0x0E47, 0xE7}, {0x0E48, 0xE8}
        , {0x0E49, 0xE9}, {0x0E4A, 0xEA}, {0x0E4B, 0xEB}, {0x0E4C, 0xEC}
        , {0x0E4D, 0xED}, {0x0E4E, 0xEE}, {0x0E4F, 0xEF}, {0x0E50, 0xF0}
        , {0x0E51, 0xF1}, {0x0E52, 0xF2}, {0x0E53, 0xF3}, {0x0E54, 0xF4}
        , {0x0E55, 0xF5}, {0x0E56, 0xF6}, {0x0E57, 0xF7}, {0x0E58, 0xF8}
        , {0x0E59, 0xF9}, {0x0E5A, 0xFA}, {0x0E5B, 0xFB}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_13
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_13>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_13>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-13";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_13;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x201D, 0x00A2, 0x00A3, 0x00A4, 0x201E, 0x00A6, 0x00A7
            , 0x00D8, 0x00A9, 0x0156, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00C6
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x201C, 0x00B5, 0x00B6, 0x00B7
            , 0x00F8, 0x00B9, 0x0157, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00E6
            , 0x0104, 0x012E, 0x0100, 0x0106, 0x00C4, 0x00C5, 0x0118, 0x0112
            , 0x010C, 0x00C9, 0x0179, 0x0116, 0x0122, 0x0136, 0x012A, 0x013B
            , 0x0160, 0x0143, 0x0145, 0x00D3, 0x014C, 0x00D5, 0x00D6, 0x00D7
            , 0x0172, 0x0141, 0x015A, 0x016A, 0x00DC, 0x017B, 0x017D, 0x00DF
            , 0x0105, 0x012F, 0x0101, 0x0107, 0x00E4, 0x00E5, 0x0119, 0x0113
            , 0x010D, 0x00E9, 0x017A, 0x0117, 0x0123, 0x0137, 0x012B, 0x013C
            , 0x0161, 0x0144, 0x0146, 0x00F3, 0x014D, 0x00F5, 0x00F6, 0x00F7
            , 0x0173, 0x0142, 0x015B, 0x016B, 0x00FC, 0x017C, 0x017E, 0x2019 };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_13::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A2, 0xA2}, {0x00A3, 0xA3}, {0x00A4, 0xA4}, {0x00A6, 0xA6}
        , {0x00A7, 0xA7}, {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}
        , {0x00AD, 0xAD}, {0x00AE, 0xAE}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B5, 0xB5}, {0x00B6, 0xB6}
        , {0x00B7, 0xB7}, {0x00B9, 0xB9}, {0x00BB, 0xBB}, {0x00BC, 0xBC}
        , {0x00BD, 0xBD}, {0x00BE, 0xBE}, {0x00C4, 0xC4}, {0x00C5, 0xC5}
        , {0x00C6, 0xAF}, {0x00C9, 0xC9}, {0x00D3, 0xD3}, {0x00D5, 0xD5}
        , {0x00D6, 0xD6}, {0x00D7, 0xD7}, {0x00D8, 0xA8}, {0x00DC, 0xDC}
        , {0x00DF, 0xDF}, {0x00E4, 0xE4}, {0x00E5, 0xE5}, {0x00E6, 0xBF}
        , {0x00E9, 0xE9}, {0x00F3, 0xF3}, {0x00F5, 0xF5}, {0x00F6, 0xF6}
        , {0x00F7, 0xF7}, {0x00F8, 0xB8}, {0x00FC, 0xFC}, {0x0100, 0xC2}
        , {0x0101, 0xE2}, {0x0104, 0xC0}, {0x0105, 0xE0}, {0x0106, 0xC3}
        , {0x0107, 0xE3}, {0x010C, 0xC8}, {0x010D, 0xE8}, {0x0112, 0xC7}
        , {0x0113, 0xE7}, {0x0116, 0xCB}, {0x0117, 0xEB}, {0x0118, 0xC6}
        , {0x0119, 0xE6}, {0x0122, 0xCC}, {0x0123, 0xEC}, {0x012A, 0xCE}
        , {0x012B, 0xEE}, {0x012E, 0xC1}, {0x012F, 0xE1}, {0x0136, 0xCD}
        , {0x0137, 0xED}, {0x013B, 0xCF}, {0x013C, 0xEF}, {0x0141, 0xD9}
        , {0x0142, 0xF9}, {0x0143, 0xD1}, {0x0144, 0xF1}, {0x0145, 0xD2}
        , {0x0146, 0xF2}, {0x014C, 0xD4}, {0x014D, 0xF4}, {0x0156, 0xAA}
        , {0x0157, 0xBA}, {0x015A, 0xDA}, {0x015B, 0xFA}, {0x0160, 0xD0}
        , {0x0161, 0xF0}, {0x016A, 0xDB}, {0x016B, 0xFB}, {0x0172, 0xD8}
        , {0x0173, 0xF8}, {0x0179, 0xCA}, {0x017A, 0xEA}, {0x017B, 0xDD}
        , {0x017C, 0xFD}, {0x017D, 0xDE}, {0x017E, 0xFE}, {0x2019, 0xFF}
        , {0x201C, 0xB4}, {0x201D, 0xA1}, {0x201E, 0xA5}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_14
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_14>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_14>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-14";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_14;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x1E02, 0x1E03, 0x00A3, 0x010A, 0x010B, 0x1E0A, 0x00A7
            , 0x1E80, 0x00A9, 0x1E82, 0x1E0B, 0x1EF2, 0x00AD, 0x00AE, 0x0178
            , 0x1E1E, 0x1E1F, 0x0120, 0x0121, 0x1E40, 0x1E41, 0x00B6, 0x1E56
            , 0x1E81, 0x1E57, 0x1E83, 0x1E60, 0x1EF3, 0x1E84, 0x1E85, 0x1E61
            , 0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7
            , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF
            , 0x0174, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x1E6A
            , 0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x0176, 0x00DF
            , 0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7
            , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF
            , 0x0175, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x1E6B
            , 0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x0177, 0x00FF };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_14::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A3, 0xA3}, {0x00A7, 0xA7}, {0x00A9, 0xA9}, {0x00AD, 0xAD}
        , {0x00AE, 0xAE}, {0x00B6, 0xB6}, {0x00C0, 0xC0}, {0x00C1, 0xC1}
        , {0x00C2, 0xC2}, {0x00C3, 0xC3}, {0x00C4, 0xC4}, {0x00C5, 0xC5}
        , {0x00C6, 0xC6}, {0x00C7, 0xC7}, {0x00C8, 0xC8}, {0x00C9, 0xC9}
        , {0x00CA, 0xCA}, {0x00CB, 0xCB}, {0x00CC, 0xCC}, {0x00CD, 0xCD}
        , {0x00CE, 0xCE}, {0x00CF, 0xCF}, {0x00D1, 0xD1}, {0x00D2, 0xD2}
        , {0x00D3, 0xD3}, {0x00D4, 0xD4}, {0x00D5, 0xD5}, {0x00D6, 0xD6}
        , {0x00D8, 0xD8}, {0x00D9, 0xD9}, {0x00DA, 0xDA}, {0x00DB, 0xDB}
        , {0x00DC, 0xDC}, {0x00DD, 0xDD}, {0x00DF, 0xDF}, {0x00E0, 0xE0}
        , {0x00E1, 0xE1}, {0x00E2, 0xE2}, {0x00E3, 0xE3}, {0x00E4, 0xE4}
        , {0x00E5, 0xE5}, {0x00E6, 0xE6}, {0x00E7, 0xE7}, {0x00E8, 0xE8}
        , {0x00E9, 0xE9}, {0x00EA, 0xEA}, {0x00EB, 0xEB}, {0x00EC, 0xEC}
        , {0x00ED, 0xED}, {0x00EE, 0xEE}, {0x00EF, 0xEF}, {0x00F1, 0xF1}
        , {0x00F2, 0xF2}, {0x00F3, 0xF3}, {0x00F4, 0xF4}, {0x00F5, 0xF5}
        , {0x00F6, 0xF6}, {0x00F8, 0xF8}, {0x00F9, 0xF9}, {0x00FA, 0xFA}
        , {0x00FB, 0xFB}, {0x00FC, 0xFC}, {0x00FD, 0xFD}, {0x00FF, 0xFF}
        , {0x010A, 0xA4}, {0x010B, 0xA5}, {0x0120, 0xB2}, {0x0121, 0xB3}
        , {0x0174, 0xD0}, {0x0175, 0xF0}, {0x0176, 0xDE}, {0x0177, 0xFE}
        , {0x0178, 0xAF}, {0x1E02, 0xA1}, {0x1E03, 0xA2}, {0x1E0A, 0xA6}
        , {0x1E0B, 0xAB}, {0x1E1E, 0xB0}, {0x1E1F, 0xB1}, {0x1E40, 0xB4}
        , {0x1E41, 0xB5}, {0x1E56, 0xB7}, {0x1E57, 0xB9}, {0x1E60, 0xBB}
        , {0x1E61, 0xBF}, {0x1E6A, 0xD7}, {0x1E6B, 0xF7}, {0x1E80, 0xA8}
        , {0x1E81, 0xB8}, {0x1E82, 0xAA}, {0x1E83, 0xBA}, {0x1E84, 0xBD}
        , {0x1E85, 0xBE}, {0x1EF2, 0xAC}, {0x1EF3, 0xBC}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)


class impl_iso_8859_15
    : public sbcs_never_has_invalid_seq_crtp<impl_iso_8859_15>
    , public sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_15>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-15";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_15;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }
    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        static const std::uint16_t ext[] = {
            /*                           */ 0x20AC, 0x00A5, 0x0160, 0x00A7,
            0x0161, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
            0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x017D, 0x00B5, 0x00B6, 0x00B7,
            0x017E, 0x00B9, 0x00BA, 0x00BB, 0x0152, 0x0153, 0x0178
        };

        STRF_IF_LIKELY (ch <= 0xA3 || 0xBF <= ch)
            return ch;

        return ext[ch - 0xA4];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0xA0 || (0xBE < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_15::encode_ext(char32_t ch) noexcept
{
    switch(ch) {
        case 0x20AC: return 0xA4;
        case 0x0160: return 0xA6;
        case 0x0161: return 0xA8;
        case 0x017D: return 0xB4;
        case 0x017E: return 0xB8;
        case 0x0152: return 0xBC;
        case 0x0153: return 0xBD;
        case 0x0178: return 0xBE;
        case 0xFFFD: return '?';
        case 0xA4:
        case 0xA6:
        case 0xA8:
        case 0xB4:
        case 0xB8:
        case 0xBC:
        case 0xBD:
        case 0xBE:
            return 0x100;
        default:
            return ch;
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

struct impl_iso_8859_16
    : sbcs_never_has_invalid_seq_crtp<impl_iso_8859_16>
    , sbcs_find_unsupported_codepoint_crtp<impl_iso_8859_16>
{
    static STRF_HD const char* name() noexcept
    {
        return "ISO-8859-16";
    };
    static constexpr strf::charset_id id = strf::csid_iso_8859_16;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        return encode_ext(ch);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0xA0) {
            return ch;
        }
        static const char32_t ext[] =
            { /*A0 */ 0x0104, 0x0105, 0x0141, 0x20AC, 0x201E, 0x0160, 0x00A7
            , 0x0161, 0x00A9, 0x0218, 0x00AB, 0x0179, 0x00AD, 0x017A, 0x017B
            , 0x00B0, 0x00B1, 0x010C, 0x0142, 0x017D, 0x201D, 0x00B6, 0x00B7
            , 0x017E, 0x010D, 0x0219, 0x00BB, 0x0152, 0x0153, 0x0178, 0x017C
            , 0x00C0, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0106, 0x00C6, 0x00C7
            , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF
            , 0x0110, 0x0143, 0x00D2, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x015A
            , 0x0170, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x0118, 0x021A, 0x00DF
            , 0x00E0, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x0107, 0x00E6, 0x00E7
            , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF
            , 0x0111, 0x0144, 0x00F2, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x015B
            , 0x0171, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x0119, 0x021B, 0x00FF };
        return ext[ch - 0xA1];
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_iso_8859_16::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A7, 0xA7}, {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AD, 0xAD}
        , {0x00B0, 0xB0}, {0x00B1, 0xB1}, {0x00B6, 0xB6}, {0x00B7, 0xB7}
        , {0x00BB, 0xBB}, {0x00C0, 0xC0}, {0x00C1, 0xC1}, {0x00C2, 0xC2}
        , {0x00C4, 0xC4}, {0x00C6, 0xC6}, {0x00C7, 0xC7}, {0x00C8, 0xC8}
        , {0x00C9, 0xC9}, {0x00CA, 0xCA}, {0x00CB, 0xCB}, {0x00CC, 0xCC}
        , {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00CF, 0xCF}, {0x00D2, 0xD2}
        , {0x00D3, 0xD3}, {0x00D4, 0xD4}, {0x00D6, 0xD6}, {0x00D9, 0xD9}
        , {0x00DA, 0xDA}, {0x00DB, 0xDB}, {0x00DC, 0xDC}, {0x00DF, 0xDF}
        , {0x00E0, 0xE0}, {0x00E1, 0xE1}, {0x00E2, 0xE2}, {0x00E4, 0xE4}
        , {0x00E6, 0xE6}, {0x00E7, 0xE7}, {0x00E8, 0xE8}, {0x00E9, 0xE9}
        , {0x00EA, 0xEA}, {0x00EB, 0xEB}, {0x00EC, 0xEC}, {0x00ED, 0xED}
        , {0x00EE, 0xEE}, {0x00EF, 0xEF}, {0x00F2, 0xF2}, {0x00F3, 0xF3}
        , {0x00F4, 0xF4}, {0x00F6, 0xF6}, {0x00F9, 0xF9}, {0x00FA, 0xFA}
        , {0x00FB, 0xFB}, {0x00FC, 0xFC}, {0x00FF, 0xFF}, {0x0102, 0xC3}
        , {0x0103, 0xE3}, {0x0104, 0xA1}, {0x0105, 0xA2}, {0x0106, 0xC5}
        , {0x0107, 0xE5}, {0x010C, 0xB2}, {0x010D, 0xB9}, {0x0110, 0xD0}
        , {0x0111, 0xF0}, {0x0118, 0xDD}, {0x0119, 0xFD}, {0x0141, 0xA3}
        , {0x0142, 0xB3}, {0x0143, 0xD1}, {0x0144, 0xF1}, {0x0150, 0xD5}
        , {0x0151, 0xF5}, {0x0152, 0xBC}, {0x0153, 0xBD}, {0x015A, 0xD7}
        , {0x015B, 0xF7}, {0x0160, 0xA6}, {0x0161, 0xA8}, {0x0170, 0xD8}
        , {0x0171, 0xF8}, {0x0178, 0xBE}, {0x0179, 0xAC}, {0x017A, 0xAE}
        , {0x017B, 0xAF}, {0x017C, 0xBF}, {0x017D, 0xB4}, {0x017E, 0xB8}
        , {0x0218, 0xAA}, {0x0219, 0xBA}, {0x021A, 0xDE}, {0x021B, 0xFE}
        , {0x201D, 0xB5}, {0x201E, 0xA5}, {0x20AC, 0xA4}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

class impl_windows_1250
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1250>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1250>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1250";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1250;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }
    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const char32_t ext[] =
            { 0x20AC, 0x0081, 0x201A, 0x0083, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x0088, 0x2030, 0x0160, 0x2039, 0x015A, 0x0164, 0x017D, 0x0179
            , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x0098, 0x2122, 0x0161, 0x203A, 0x015B, 0x0165, 0x017E, 0x017A
            , 0x00A0, 0x02C7, 0x02D8, 0x0141, 0x00A4, 0x0104, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x015E, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x017B
            , 0x00B0, 0x00B1, 0x02DB, 0x0142, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00B8, 0x0105, 0x015F, 0x00BB, 0x013D, 0x02DD, 0x013E, 0x017C
            , 0x0154, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7
            , 0x010C, 0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E
            , 0x0110, 0x0143, 0x0147, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x00D7
            , 0x0158, 0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF
            , 0x0155, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7
            , 0x010D, 0x00E9, 0x0119, 0x00EB, 0x011B, 0x00ED, 0x00EE, 0x010F
            , 0x0111, 0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7
            , 0x0159, 0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9 };
        return ext[ch - 0x80];
    }
    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return ch < 0x80 ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1250::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0081, 0x81}, {0x0083, 0x83}, {0x0088, 0x88}, {0x0090, 0x90}
        , {0x0098, 0x98}, {0x00A0, 0xA0}, {0x00A4, 0xA4}, {0x00A6, 0xA6}
        , {0x00A7, 0xA7}, {0x00A8, 0xA8}, {0x00A9, 0xA9}, {0x00AB, 0xAB}
        , {0x00AC, 0xAC}, {0x00AD, 0xAD}, {0x00AE, 0xAE}, {0x00B0, 0xB0}
        , {0x00B1, 0xB1}, {0x00B4, 0xB4}, {0x00B5, 0xB5}, {0x00B6, 0xB6}
        , {0x00B7, 0xB7}, {0x00B8, 0xB8}, {0x00BB, 0xBB}, {0x00C1, 0xC1}
        , {0x00C2, 0xC2}, {0x00C4, 0xC4}, {0x00C7, 0xC7}, {0x00C9, 0xC9}
        , {0x00CB, 0xCB}, {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00D3, 0xD3}
        , {0x00D4, 0xD4}, {0x00D6, 0xD6}, {0x00D7, 0xD7}, {0x00DA, 0xDA}
        , {0x00DC, 0xDC}, {0x00DD, 0xDD}, {0x00DF, 0xDF}, {0x00E1, 0xE1}
        , {0x00E2, 0xE2}, {0x00E4, 0xE4}, {0x00E7, 0xE7}, {0x00E9, 0xE9}
        , {0x00EB, 0xEB}, {0x00ED, 0xED}, {0x00EE, 0xEE}, {0x00F3, 0xF3}
        , {0x00F4, 0xF4}, {0x00F6, 0xF6}, {0x00F7, 0xF7}, {0x00FA, 0xFA}
        , {0x00FC, 0xFC}, {0x00FD, 0xFD}, {0x0102, 0xC3}, {0x0103, 0xE3}
        , {0x0104, 0xA5}, {0x0105, 0xB9}, {0x0106, 0xC6}, {0x0107, 0xE6}
        , {0x010C, 0xC8}, {0x010D, 0xE8}, {0x010E, 0xCF}, {0x010F, 0xEF}
        , {0x0110, 0xD0}, {0x0111, 0xF0}, {0x0118, 0xCA}, {0x0119, 0xEA}
        , {0x011A, 0xCC}, {0x011B, 0xEC}, {0x0139, 0xC5}, {0x013A, 0xE5}
        , {0x013D, 0xBC}, {0x013E, 0xBE}, {0x0141, 0xA3}, {0x0142, 0xB3}
        , {0x0143, 0xD1}, {0x0144, 0xF1}, {0x0147, 0xD2}, {0x0148, 0xF2}
        , {0x0150, 0xD5}, {0x0151, 0xF5}, {0x0154, 0xC0}, {0x0155, 0xE0}
        , {0x0158, 0xD8}, {0x0159, 0xF8}, {0x015A, 0x8C}, {0x015B, 0x9C}
        , {0x015E, 0xAA}, {0x015F, 0xBA}, {0x0160, 0x8A}, {0x0161, 0x9A}
        , {0x0162, 0xDE}, {0x0163, 0xFE}, {0x0164, 0x8D}, {0x0165, 0x9D}
        , {0x016E, 0xD9}, {0x016F, 0xF9}, {0x0170, 0xDB}, {0x0171, 0xFB}
        , {0x0179, 0x8F}, {0x017A, 0x9F}, {0x017B, 0xAF}, {0x017C, 0xBF}
        , {0x017D, 0x8E}, {0x017E, 0x9E}, {0x02C7, 0xA1}, {0x02D8, 0xA2}
        , {0x02D9, 0xFF}, {0x02DB, 0xB2}, {0x02DD, 0xBD}, {0x2013, 0x96}
        , {0x2014, 0x97}, {0x2018, 0x91}, {0x2019, 0x92}, {0x201A, 0x82}
        , {0x201C, 0x93}, {0x201D, 0x94}, {0x201E, 0x84}, {0x2020, 0x86}
        , {0x2021, 0x87}, {0x2022, 0x95}, {0x2026, 0x85}, {0x2030, 0x89}
        , {0x2039, 0x8B}, {0x203A, 0x9B}, {0x20AC, 0x80}, {0x2122, 0x99}
        , {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif // ! defined(STRF_OMIT_IMPL)

class impl_windows_1251
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1251>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1251>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1251";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1251;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x20AC, 0x2030, 0x0409, 0x2039, 0x040A, 0x040C, 0x040B, 0x040F
            , 0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x0098, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F
            , 0x00A0, 0x040E, 0x045E, 0x0408, 0x00A4, 0x0490, 0x00A6, 0x00A7
            , 0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407
            , 0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7
            , 0x0451, 0x2116, 0x0454, 0x00BB, 0x0458, 0x0405, 0x0455, 0x0457
            , 0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417
            , 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F
            , 0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427
            , 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F
            , 0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437
            , 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F
            , 0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447
            , 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return ch < 0x80 ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1251::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0098, 0x98}, {0x00A0, 0xA0}, {0x00A4, 0xA4}, {0x00A6, 0xA6}
        , {0x00A7, 0xA7}, {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}
        , {0x00AD, 0xAD}, {0x00AE, 0xAE}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B5, 0xB5}, {0x00B6, 0xB6}, {0x00B7, 0xB7}, {0x00BB, 0xBB}
        , {0x0401, 0xA8}, {0x0402, 0x80}, {0x0403, 0x81}, {0x0404, 0xAA}
        , {0x0405, 0xBD}, {0x0406, 0xB2}, {0x0407, 0xAF}, {0x0408, 0xA3}
        , {0x0409, 0x8A}, {0x040A, 0x8C}, {0x040B, 0x8E}, {0x040C, 0x8D}
        , {0x040E, 0xA1}, {0x040F, 0x8F}, {0x0410, 0xC0}, {0x0411, 0xC1}
        , {0x0412, 0xC2}, {0x0413, 0xC3}, {0x0414, 0xC4}, {0x0415, 0xC5}
        , {0x0416, 0xC6}, {0x0417, 0xC7}, {0x0418, 0xC8}, {0x0419, 0xC9}
        , {0x041A, 0xCA}, {0x041B, 0xCB}, {0x041C, 0xCC}, {0x041D, 0xCD}
        , {0x041E, 0xCE}, {0x041F, 0xCF}, {0x0420, 0xD0}, {0x0421, 0xD1}
        , {0x0422, 0xD2}, {0x0423, 0xD3}, {0x0424, 0xD4}, {0x0425, 0xD5}
        , {0x0426, 0xD6}, {0x0427, 0xD7}, {0x0428, 0xD8}, {0x0429, 0xD9}
        , {0x042A, 0xDA}, {0x042B, 0xDB}, {0x042C, 0xDC}, {0x042D, 0xDD}
        , {0x042E, 0xDE}, {0x042F, 0xDF}, {0x0430, 0xE0}, {0x0431, 0xE1}
        , {0x0432, 0xE2}, {0x0433, 0xE3}, {0x0434, 0xE4}, {0x0435, 0xE5}
        , {0x0436, 0xE6}, {0x0437, 0xE7}, {0x0438, 0xE8}, {0x0439, 0xE9}
        , {0x043A, 0xEA}, {0x043B, 0xEB}, {0x043C, 0xEC}, {0x043D, 0xED}
        , {0x043E, 0xEE}, {0x043F, 0xEF}, {0x0440, 0xF0}, {0x0441, 0xF1}
        , {0x0442, 0xF2}, {0x0443, 0xF3}, {0x0444, 0xF4}, {0x0445, 0xF5}
        , {0x0446, 0xF6}, {0x0447, 0xF7}, {0x0448, 0xF8}, {0x0449, 0xF9}
        , {0x044A, 0xFA}, {0x044B, 0xFB}, {0x044C, 0xFC}, {0x044D, 0xFD}
        , {0x044E, 0xFE}, {0x044F, 0xFF}, {0x0451, 0xB8}, {0x0452, 0x90}
        , {0x0453, 0x83}, {0x0454, 0xBA}, {0x0455, 0xBE}, {0x0456, 0xB3}
        , {0x0457, 0xBF}, {0x0458, 0xBC}, {0x0459, 0x9A}, {0x045A, 0x9C}
        , {0x045B, 0x9E}, {0x045C, 0x9D}, {0x045E, 0xA2}, {0x045F, 0x9F}
        , {0x0490, 0xA5}, {0x0491, 0xB4}, {0x2013, 0x96}, {0x2014, 0x97}
        , {0x2018, 0x91}, {0x2019, 0x92}, {0x201A, 0x82}, {0x201C, 0x93}
        , {0x201D, 0x94}, {0x201E, 0x84}, {0x2020, 0x86}, {0x2021, 0x87}
        , {0x2022, 0x95}, {0x2026, 0x85}, {0x2030, 0x89}, {0x2039, 0x8B}
        , {0x203A, 0x9B}, {0x20AC, 0x88}, {0x2116, 0xB9}, {0x2122, 0x99}
        , {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1252
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1252>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1252>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1252";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1252;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80 || 0x9F < ch) {
            return ch;
        }
        static const std::uint16_t ext[] = {
            0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
            0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
            0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
            0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
        };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80 || (0x9F < ch && ch < 0x100)) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1252::encode_ext(char32_t ch) noexcept
{
    switch(ch) {
        case 0x81: return 0x81;
        case 0x8D: return 0x8D;
        case 0x8F: return 0x8F;
        case 0x90: return 0x90;
        case 0x9D: return 0x9D;
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
        case 0xFFFD: return '?';
        default: return 0x100;
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

class impl_windows_1253
    : public sbcs_find_invalid_seq_crtp<impl_windows_1253>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1253>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1253";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1253;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch < 0xAA || (ch != 0xAA && ch != 0xD2 && ch != 0xFF);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x0088, 0x2030, 0x008A, 0x2039, 0x008C, 0x008D, 0x008E, 0x008F
            , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x0098, 0x2122, 0x009A, 0x203A, 0x009C, 0x009D, 0x009E, 0x009F
            , 0x00A0, 0x0385, 0x0386, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0xFFFD, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x2015
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x0384, 0x00B5, 0x00B6, 0x00B7
            , 0x0388, 0x0389, 0x038A, 0x00BB, 0x038C, 0x00BD, 0x038E, 0x038F
            , 0x0390, 0x0391, 0x0392, 0x0393, 0x0394, 0x0395, 0x0396, 0x0397
            , 0x0398, 0x0399, 0x039A, 0x039B, 0x039C, 0x039D, 0x039E, 0x039F
            , 0x03A0, 0x03A1, 0xFFFD, 0x03A3, 0x03A4, 0x03A5, 0x03A6, 0x03A7
            , 0x03A8, 0x03A9, 0x03AA, 0x03AB, 0x03AC, 0x03AD, 0x03AE, 0x03AF
            , 0x03B0, 0x03B1, 0x03B2, 0x03B3, 0x03B4, 0x03B5, 0x03B6, 0x03B7
            , 0x03B8, 0x03B9, 0x03BA, 0x03BB, 0x03BC, 0x03BD, 0x03BE, 0x03BF
            , 0x03C0, 0x03C1, 0x03C2, 0x03C3, 0x03C4, 0x03C5, 0x03C6, 0x03C7
            , 0x03C8, 0x03C9, 0x03CA, 0x03CB, 0x03CC, 0x03CD, 0x03CE, 0xFFFD };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1253::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0081, 0x81}, {0x0088, 0x88}, {0x008A, 0x8A}, {0x008C, 0x8C}
        , {0x008D, 0x8D}, {0x008E, 0x8E}, {0x008F, 0x8F}, {0x0090, 0x90}
        , {0x0098, 0x98}, {0x009A, 0x9A}, {0x009C, 0x9C}, {0x009D, 0x9D}
        , {0x009E, 0x9E}, {0x009F, 0x9F}, {0x00A0, 0xA0}, {0x00A3, 0xA3}
        , {0x00A4, 0xA4}, {0x00A5, 0xA5}, {0x00A6, 0xA6}, {0x00A7, 0xA7}
        , {0x00A8, 0xA8}, {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}
        , {0x00AD, 0xAD}, {0x00AE, 0xAE}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B5, 0xB5}, {0x00B6, 0xB6}
        , {0x00B7, 0xB7}, {0x00BB, 0xBB}, {0x00BD, 0xBD}, {0x0192, 0x83}
        , {0x0384, 0xB4}, {0x0385, 0xA1}, {0x0386, 0xA2}, {0x0388, 0xB8}
        , {0x0389, 0xB9}, {0x038A, 0xBA}, {0x038C, 0xBC}, {0x038E, 0xBE}
        , {0x038F, 0xBF}, {0x0390, 0xC0}, {0x0391, 0xC1}, {0x0392, 0xC2}
        , {0x0393, 0xC3}, {0x0394, 0xC4}, {0x0395, 0xC5}, {0x0396, 0xC6}
        , {0x0397, 0xC7}, {0x0398, 0xC8}, {0x0399, 0xC9}, {0x039A, 0xCA}
        , {0x039B, 0xCB}, {0x039C, 0xCC}, {0x039D, 0xCD}, {0x039E, 0xCE}
        , {0x039F, 0xCF}, {0x03A0, 0xD0}, {0x03A1, 0xD1}, {0x03A3, 0xD3}
        , {0x03A4, 0xD4}, {0x03A5, 0xD5}, {0x03A6, 0xD6}, {0x03A7, 0xD7}
        , {0x03A8, 0xD8}, {0x03A9, 0xD9}, {0x03AA, 0xDA}, {0x03AB, 0xDB}
        , {0x03AC, 0xDC}, {0x03AD, 0xDD}, {0x03AE, 0xDE}, {0x03AF, 0xDF}
        , {0x03B0, 0xE0}, {0x03B1, 0xE1}, {0x03B2, 0xE2}, {0x03B3, 0xE3}
        , {0x03B4, 0xE4}, {0x03B5, 0xE5}, {0x03B6, 0xE6}, {0x03B7, 0xE7}
        , {0x03B8, 0xE8}, {0x03B9, 0xE9}, {0x03BA, 0xEA}, {0x03BB, 0xEB}
        , {0x03BC, 0xEC}, {0x03BD, 0xED}, {0x03BE, 0xEE}, {0x03BF, 0xEF}
        , {0x03C0, 0xF0}, {0x03C1, 0xF1}, {0x03C2, 0xF2}, {0x03C3, 0xF3}
        , {0x03C4, 0xF4}, {0x03C5, 0xF5}, {0x03C6, 0xF6}, {0x03C7, 0xF7}
        , {0x03C8, 0xF8}, {0x03C9, 0xF9}, {0x03CA, 0xFA}, {0x03CB, 0xFB}
        , {0x03CC, 0xFC}, {0x03CD, 0xFD}, {0x03CE, 0xFE}, {0x2013, 0x96}
        , {0x2014, 0x97}, {0x2015, 0xAF}, {0x2018, 0x91}, {0x2019, 0x92}
        , {0x201A, 0x82}, {0x201C, 0x93}, {0x201D, 0x94}, {0x201E, 0x84}
        , {0x2020, 0x86}, {0x2021, 0x87}, {0x2022, 0x95}, {0x2026, 0x85}
        , {0x2030, 0x89}, {0x2039, 0x8B}, {0x203A, 0x9B}, {0x20AC, 0x80}
        , {0x2122, 0x99}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1254
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1254>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1254>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1254";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1254;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        if (ch <= 0x7F) {
            return ch;
        }
        if (ch <= 0x9F) {
            static const std::uint16_t ext[] =
                { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
                , 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x008E, 0x008F
                , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
                , 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x009E, 0x0178 };
            return ext[ch - 0x80];
        }
        switch (ch) {
            case 0xD0: return 0x011E;
            case 0xDD: return 0x0130;
            case 0xDE: return 0x015E;
            case 0xF0: return 0x011F;
            case 0xFD: return 0x0131;
            case 0xFE: return 0x015F;
            default: return ch;
        }
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return ch < 0x80 ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1254::encode_ext(char32_t ch) noexcept
{
    switch(ch) {
        case 0x81: return 0x81;
        case 0x8D: return 0x8D;
        case 0x8E: return 0x8E;
        case 0x8F: return 0x8F;
        case 0x90: return 0x90;
        case 0x9D: return 0x9D;
        case 0x9E: return 0x9E;
        case 0x011E: return 0xD0;
        case 0x011F: return 0xF0;
        case 0x0130: return 0xDD;
        case 0x0131: return 0xFD;
        case 0x0152: return 0x8C;
        case 0x0153: return 0x9C;
        case 0x015E: return 0xDE;
        case 0x015F: return 0xFE;
        case 0x0160: return 0x8A;
        case 0x0161: return 0x9A;
        case 0x0178: return 0x9F;
        case 0x0192: return 0x83;
        case 0x02C6: return 0x88;
        case 0x02DC: return 0x98;
        case 0x2013: return 0x96;
        case 0x2014: return 0x97;
        case 0x2018: return 0x91;
        case 0x2019: return 0x92;
        case 0x201A: return 0x82;
        case 0x201C: return 0x93;
        case 0x201D: return 0x94;
        case 0x201E: return 0x84;
        case 0x2020: return 0x86;
        case 0x2021: return 0x87;
        case 0x2022: return 0x95;
        case 0x2026: return 0x85;
        case 0x2030: return 0x89;
        case 0x2039: return 0x8B;
        case 0x203A: return 0x9B;
        case 0x20AC: return 0x80;
        case 0x2122: return 0x99;
        case 0xFFFD: return '?';
        default: return 0xA0 <= ch && ch <= 0xFF ? ch : 0x100;
    }
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1255
    : public sbcs_find_invalid_seq_crtp<impl_windows_1255>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1255>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1255";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1255;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch < 0xD9 || (ch != 0xD9 && ch != 0xDA && ch != 0xDB &&
                             ch != 0xDC && ch != 0xDD && ch != 0xDE &&
                             ch != 0xDF && ch != 0xFB && ch != 0xFC &&
                             ch != 0xFF);
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x02C6, 0x2030, 0x008A, 0x2039, 0x008C, 0x008D, 0x008E, 0x008F
            , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x02DC, 0x2122, 0x009A, 0x203A, 0x009C, 0x009D, 0x009E, 0x009F
            , 0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x20AA, 0x00A5, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x00D7, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00B8, 0x00B9, 0x00F7, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF
            , 0x05B0, 0x05B1, 0x05B2, 0x05B3, 0x05B4, 0x05B5, 0x05B6, 0x05B7
            , 0x05B8, 0x05B9, 0x05BA, 0x05BB, 0x05BC, 0x05BD, 0x05BE, 0x05BF
            , 0x05C0, 0x05C1, 0x05C2, 0x05C3, 0x05F0, 0x05F1, 0x05F2, 0x05F3
            , 0x05F4, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD
            , 0x05D0, 0x05D1, 0x05D2, 0x05D3, 0x05D4, 0x05D5, 0x05D6, 0x05D7
            , 0x05D8, 0x05D9, 0x05DA, 0x05DB, 0x05DC, 0x05DD, 0x05DE, 0x05DF
            , 0x05E0, 0x05E1, 0x05E2, 0x05E3, 0x05E4, 0x05E5, 0x05E6, 0x05E7
            , 0x05E8, 0x05E9, 0x05EA, 0xFFFD, 0xFFFD, 0x200E, 0x200F, 0xFFFD
        };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1255::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0081, 0x81}, {0x008A, 0x8A}, {0x008C, 0x8C}, {0x008D, 0x8D}
        , {0x008E, 0x8E}, {0x008F, 0x8F}, {0x0090, 0x90}, {0x009A, 0x9A}
        , {0x009C, 0x9C}, {0x009D, 0x9D}, {0x009E, 0x9E}, {0x009F, 0x9F}
        , {0x00A0, 0xA0}, {0x00A1, 0xA1}, {0x00A2, 0xA2}, {0x00A3, 0xA3}
        , {0x00A5, 0xA5}, {0x00A6, 0xA6}, {0x00A7, 0xA7}, {0x00A8, 0xA8}
        , {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}, {0x00AD, 0xAD}
        , {0x00AE, 0xAE}, {0x00AF, 0xAF}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B4, 0xB4}, {0x00B5, 0xB5}
        , {0x00B6, 0xB6}, {0x00B7, 0xB7}, {0x00B8, 0xB8}, {0x00B9, 0xB9}
        , {0x00BB, 0xBB}, {0x00BC, 0xBC}, {0x00BD, 0xBD}, {0x00BE, 0xBE}
        , {0x00BF, 0xBF}, {0x00D7, 0xAA}, {0x00F7, 0xBA}, {0x0192, 0x83}
        , {0x02C6, 0x88}, {0x02DC, 0x98}, {0x05B0, 0xC0}, {0x05B1, 0xC1}
        , {0x05B2, 0xC2}, {0x05B3, 0xC3}, {0x05B4, 0xC4}, {0x05B5, 0xC5}
        , {0x05B6, 0xC6}, {0x05B7, 0xC7}, {0x05B8, 0xC8}, {0x05B9, 0xC9}
        , {0x05BA, 0xCA}, {0x05BB, 0xCB}, {0x05BC, 0xCC}, {0x05BD, 0xCD}
        , {0x05BE, 0xCE}, {0x05BF, 0xCF}, {0x05C0, 0xD0}, {0x05C1, 0xD1}
        , {0x05C2, 0xD2}, {0x05C3, 0xD3}, {0x05D0, 0xE0}, {0x05D1, 0xE1}
        , {0x05D2, 0xE2}, {0x05D3, 0xE3}, {0x05D4, 0xE4}, {0x05D5, 0xE5}
        , {0x05D6, 0xE6}, {0x05D7, 0xE7}, {0x05D8, 0xE8}, {0x05D9, 0xE9}
        , {0x05DA, 0xEA}, {0x05DB, 0xEB}, {0x05DC, 0xEC}, {0x05DD, 0xED}
        , {0x05DE, 0xEE}, {0x05DF, 0xEF}, {0x05E0, 0xF0}, {0x05E1, 0xF1}
        , {0x05E2, 0xF2}, {0x05E3, 0xF3}, {0x05E4, 0xF4}, {0x05E5, 0xF5}
        , {0x05E6, 0xF6}, {0x05E7, 0xF7}, {0x05E8, 0xF8}, {0x05E9, 0xF9}
        , {0x05EA, 0xFA}, {0x05F0, 0xD4}, {0x05F1, 0xD5}, {0x05F2, 0xD6}
        , {0x05F3, 0xD7}, {0x05F4, 0xD8}, {0x200E, 0xFD}, {0x200F, 0xFE}
        , {0x2013, 0x96}, {0x2014, 0x97}, {0x2018, 0x91}, {0x2019, 0x92}
        , {0x201A, 0x82}, {0x201C, 0x93}, {0x201D, 0x94}, {0x201E, 0x84}
        , {0x2020, 0x86}, {0x2021, 0x87}, {0x2022, 0x95}, {0x2026, 0x85}
        , {0x2030, 0x89}, {0x2039, 0x8B}, {0x203A, 0x9B}, {0x20AA, 0xA4}
        , {0x20AC, 0x80}, {0x2122, 0x99}, {0xFFFD, '?'}};

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1256
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1256>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1256>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1256";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1256;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x20AC, 0x067E, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x02C6, 0x2030, 0x0679, 0x2039, 0x0152, 0x0686, 0x0698, 0x0688
            , 0x06AF, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x06A9, 0x2122, 0x0691, 0x203A, 0x0153, 0x200C, 0x200D, 0x06BA
            , 0x00A0, 0x060C, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x06BE, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00B8, 0x00B9, 0x061B, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x061F
            , 0x06C1, 0x0621, 0x0622, 0x0623, 0x0624, 0x0625, 0x0626, 0x0627
            , 0x0628, 0x0629, 0x062A, 0x062B, 0x062C, 0x062D, 0x062E, 0x062F
            , 0x0630, 0x0631, 0x0632, 0x0633, 0x0634, 0x0635, 0x0636, 0x00D7
            , 0x0637, 0x0638, 0x0639, 0x063A, 0x0640, 0x0641, 0x0642, 0x0643
            , 0x00E0, 0x0644, 0x00E2, 0x0645, 0x0646, 0x0647, 0x0648, 0x00E7
            , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x0649, 0x064A, 0x00EE, 0x00EF
            , 0x064B, 0x064C, 0x064D, 0x064E, 0x00F4, 0x064F, 0x0650, 0x00F7
            , 0x0651, 0x00F9, 0x0652, 0x00FB, 0x00FC, 0x200E, 0x200F, 0x06D2 };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1256::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x00A0, 0xA0}, {0x00A2, 0xA2}, {0x00A3, 0xA3}, {0x00A4, 0xA4}
        , {0x00A5, 0xA5}, {0x00A6, 0xA6}, {0x00A7, 0xA7}, {0x00A8, 0xA8}
        , {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}, {0x00AD, 0xAD}
        , {0x00AE, 0xAE}, {0x00AF, 0xAF}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B4, 0xB4}, {0x00B5, 0xB5}
        , {0x00B6, 0xB6}, {0x00B7, 0xB7}, {0x00B8, 0xB8}, {0x00B9, 0xB9}
        , {0x00BB, 0xBB}, {0x00BC, 0xBC}, {0x00BD, 0xBD}, {0x00BE, 0xBE}
        , {0x00D7, 0xD7}, {0x00E0, 0xE0}, {0x00E2, 0xE2}, {0x00E7, 0xE7}
        , {0x00E8, 0xE8}, {0x00E9, 0xE9}, {0x00EA, 0xEA}, {0x00EB, 0xEB}
        , {0x00EE, 0xEE}, {0x00EF, 0xEF}, {0x00F4, 0xF4}, {0x00F7, 0xF7}
        , {0x00F9, 0xF9}, {0x00FB, 0xFB}, {0x00FC, 0xFC}, {0x0152, 0x8C}
        , {0x0153, 0x9C}, {0x0192, 0x83}, {0x02C6, 0x88}, {0x060C, 0xA1}
        , {0x061B, 0xBA}, {0x061F, 0xBF}, {0x0621, 0xC1}, {0x0622, 0xC2}
        , {0x0623, 0xC3}, {0x0624, 0xC4}, {0x0625, 0xC5}, {0x0626, 0xC6}
        , {0x0627, 0xC7}, {0x0628, 0xC8}, {0x0629, 0xC9}, {0x062A, 0xCA}
        , {0x062B, 0xCB}, {0x062C, 0xCC}, {0x062D, 0xCD}, {0x062E, 0xCE}
        , {0x062F, 0xCF}, {0x0630, 0xD0}, {0x0631, 0xD1}, {0x0632, 0xD2}
        , {0x0633, 0xD3}, {0x0634, 0xD4}, {0x0635, 0xD5}, {0x0636, 0xD6}
        , {0x0637, 0xD8}, {0x0638, 0xD9}, {0x0639, 0xDA}, {0x063A, 0xDB}
        , {0x0640, 0xDC}, {0x0641, 0xDD}, {0x0642, 0xDE}, {0x0643, 0xDF}
        , {0x0644, 0xE1}, {0x0645, 0xE3}, {0x0646, 0xE4}, {0x0647, 0xE5}
        , {0x0648, 0xE6}, {0x0649, 0xEC}, {0x064A, 0xED}, {0x064B, 0xF0}
        , {0x064C, 0xF1}, {0x064D, 0xF2}, {0x064E, 0xF3}, {0x064F, 0xF5}
        , {0x0650, 0xF6}, {0x0651, 0xF8}, {0x0652, 0xFA}, {0x0679, 0x8A}
        , {0x067E, 0x81}, {0x0686, 0x8D}, {0x0688, 0x8F}, {0x0691, 0x9A}
        , {0x0698, 0x8E}, {0x06A9, 0x98}, {0x06AF, 0x90}, {0x06BA, 0x9F}
        , {0x06BE, 0xAA}, {0x06C1, 0xC0}, {0x06D2, 0xFF}, {0x200C, 0x9D}
        , {0x200D, 0x9E}, {0x200E, 0xFD}, {0x200F, 0xFE}, {0x2013, 0x96}
        , {0x2014, 0x97}, {0x2018, 0x91}, {0x2019, 0x92}, {0x201A, 0x82}
        , {0x201C, 0x93}, {0x201D, 0x94}, {0x201E, 0x84}, {0x2020, 0x86}
        , {0x2021, 0x87}, {0x2022, 0x95}, {0x2026, 0x85}, {0x2030, 0x89}
        , {0x2039, 0x8B}, {0x203A, 0x9B}, {0x20AC, 0x80}, {0x2122, 0x99}
        , {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1257
    : public sbcs_find_invalid_seq_crtp<impl_windows_1257>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1257>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1257";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1257;

    static STRF_HD bool is_valid(std::uint8_t ch) noexcept
    {
        return ch != 0xA1 && ch != 0xA5;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x20AC, 0x0081, 0x201A, 0x0083, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x0088, 0x2030, 0x008A, 0x2039, 0x008C, 0x00A8, 0x02C7, 0x00B8
            , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x0098, 0x2122, 0x009A, 0x203A, 0x009C, 0x00AF, 0x02DB, 0x009F
            , 0x00A0, 0xFFFD, 0x00A2, 0x00A3, 0x00A4, 0xFFFD, 0x00A6, 0x00A7
            , 0x00D8, 0x00A9, 0x0156, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00C6
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00F8, 0x00B9, 0x0157, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00E6
            , 0x0104, 0x012E, 0x0100, 0x0106, 0x00C4, 0x00C5, 0x0118, 0x0112
            , 0x010C, 0x00C9, 0x0179, 0x0116, 0x0122, 0x0136, 0x012A, 0x013B
            , 0x0160, 0x0143, 0x0145, 0x00D3, 0x014C, 0x00D5, 0x00D6, 0x00D7
            , 0x0172, 0x0141, 0x015A, 0x016A, 0x00DC, 0x017B, 0x017D, 0x00DF
            , 0x0105, 0x012F, 0x0101, 0x0107, 0x00E4, 0x00E5, 0x0119, 0x0113
            , 0x010D, 0x00E9, 0x017A, 0x0117, 0x0123, 0x0137, 0x012B, 0x013C
            , 0x0161, 0x0144, 0x0146, 0x00F3, 0x014D, 0x00F5, 0x00F6, 0x00F7
            , 0x0173, 0x0142, 0x015B, 0x016B, 0x00FC, 0x017C, 0x017E, 0x02D9 };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1257::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0081, 0x81}, {0x0083, 0x83}, {0x0088, 0x88}, {0x008A, 0x8A}
        , {0x008C, 0x8C}, {0x0090, 0x90}, {0x0098, 0x98}, {0x009A, 0x9A}
        , {0x009C, 0x9C}, {0x009F, 0x9F}, {0x00A0, 0xA0}, {0x00A2, 0xA2}
        , {0x00A3, 0xA3}, {0x00A4, 0xA4}, {0x00A6, 0xA6}, {0x00A7, 0xA7}
        , {0x00A8, 0x8D}, {0x00A9, 0xA9}, {0x00AB, 0xAB}, {0x00AC, 0xAC}
        , {0x00AD, 0xAD}, {0x00AE, 0xAE}, {0x00AF, 0x9D}, {0x00B0, 0xB0}
        , {0x00B1, 0xB1}, {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B4, 0xB4}
        , {0x00B5, 0xB5}, {0x00B6, 0xB6}, {0x00B7, 0xB7}, {0x00B8, 0x8F}
        , {0x00B9, 0xB9}, {0x00BB, 0xBB}, {0x00BC, 0xBC}, {0x00BD, 0xBD}
        , {0x00BE, 0xBE}, {0x00C4, 0xC4}, {0x00C5, 0xC5}, {0x00C6, 0xAF}
        , {0x00C9, 0xC9}, {0x00D3, 0xD3}, {0x00D5, 0xD5}, {0x00D6, 0xD6}
        , {0x00D7, 0xD7}, {0x00D8, 0xA8}, {0x00DC, 0xDC}, {0x00DF, 0xDF}
        , {0x00E4, 0xE4}, {0x00E5, 0xE5}, {0x00E6, 0xBF}, {0x00E9, 0xE9}
        , {0x00F3, 0xF3}, {0x00F5, 0xF5}, {0x00F6, 0xF6}, {0x00F7, 0xF7}
        , {0x00F8, 0xB8}, {0x00FC, 0xFC}, {0x0100, 0xC2}, {0x0101, 0xE2}
        , {0x0104, 0xC0}, {0x0105, 0xE0}, {0x0106, 0xC3}, {0x0107, 0xE3}
        , {0x010C, 0xC8}, {0x010D, 0xE8}, {0x0112, 0xC7}, {0x0113, 0xE7}
        , {0x0116, 0xCB}, {0x0117, 0xEB}, {0x0118, 0xC6}, {0x0119, 0xE6}
        , {0x0122, 0xCC}, {0x0123, 0xEC}, {0x012A, 0xCE}, {0x012B, 0xEE}
        , {0x012E, 0xC1}, {0x012F, 0xE1}, {0x0136, 0xCD}, {0x0137, 0xED}
        , {0x013B, 0xCF}, {0x013C, 0xEF}, {0x0141, 0xD9}, {0x0142, 0xF9}
        , {0x0143, 0xD1}, {0x0144, 0xF1}, {0x0145, 0xD2}, {0x0146, 0xF2}
        , {0x014C, 0xD4}, {0x014D, 0xF4}, {0x0156, 0xAA}, {0x0157, 0xBA}
        , {0x015A, 0xDA}, {0x015B, 0xFA}, {0x0160, 0xD0}, {0x0161, 0xF0}
        , {0x016A, 0xDB}, {0x016B, 0xFB}, {0x0172, 0xD8}, {0x0173, 0xF8}
        , {0x0179, 0xCA}, {0x017A, 0xEA}, {0x017B, 0xDD}, {0x017C, 0xFD}
        , {0x017D, 0xDE}, {0x017E, 0xFE}, {0x02C7, 0x8E}, {0x02D9, 0xFF}
        , {0x02DB, 0x9E}, {0x2013, 0x96}, {0x2014, 0x97}, {0x2018, 0x91}
        , {0x2019, 0x92}, {0x201A, 0x82}, {0x201C, 0x93}, {0x201D, 0x94}
        , {0x201E, 0x84}, {0x2020, 0x86}, {0x2021, 0x87}, {0x2022, 0x95}
        , {0x2026, 0x85}, {0x2030, 0x89}, {0x2039, 0x8B}, {0x203A, 0x9B}
        , {0x20AC, 0x80}, {0x2122, 0x99}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

class impl_windows_1258
    : public sbcs_never_has_invalid_seq_crtp<impl_windows_1258>
    , public sbcs_find_unsupported_codepoint_crtp<impl_windows_1258>
{
public:

    static STRF_HD const char* name() noexcept
    {
        return "windows-1258";
    };
    static constexpr strf::charset_id id = strf::csid_windows_1258;

    static STRF_HD bool is_valid(std::uint8_t) noexcept
    {
        return true;
    }

    static STRF_HD char32_t decode(std::uint8_t ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80) {
            return ch;
        }
        static const std::uint16_t ext[] =
            { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021
            , 0x02C6, 0x2030, 0x008A, 0x2039, 0x0152, 0x008D, 0x008E, 0x008F
            , 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014
            , 0x02DC, 0x2122, 0x009A, 0x203A, 0x0153, 0x009D, 0x009E, 0x0178
            , 0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7
            , 0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF
            , 0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7
            , 0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF
            , 0x00C0, 0x00C1, 0x00C2, 0x0102, 0x00C4, 0x00C5, 0x00C6, 0x00C7
            , 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x0300, 0x00CD, 0x00CE, 0x00CF
            , 0x0110, 0x00D1, 0x0309, 0x00D3, 0x00D4, 0x01A0, 0x00D6, 0x00D7
            , 0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x01AF, 0x0303, 0x00DF
            , 0x00E0, 0x00E1, 0x00E2, 0x0103, 0x00E4, 0x00E5, 0x00E6, 0x00E7
            , 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x0301, 0x00ED, 0x00EE, 0x00EF
            , 0x0111, 0x00F1, 0x0323, 0x00F3, 0x00F4, 0x01A1, 0x00F6, 0x00F7
            , 0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x01B0, 0x20AB, 0x00FF };
        return ext[ch - 0x80];
    }

    static STRF_HD unsigned encode(char32_t ch) noexcept
    {
        return (ch < 0x80) ? ch : encode_ext(ch);
    }

private:

    static STRF_HD unsigned encode_ext(char32_t ch) noexcept;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD unsigned impl_windows_1258::encode_ext(char32_t ch) noexcept
{
    static const ch32_to_char char_map[] =
        { {0x0081, 0x81}, {0x008A, 0x8A}, {0x008D, 0x8D}, {0x008E, 0x8E}
        , {0x008F, 0x8F}, {0x0090, 0x90}, {0x009A, 0x9A}, {0x009D, 0x9D}
        , {0x009E, 0x9E}, {0x009F, 0x9F}, {0x00A0, 0xA0}, {0x00A1, 0xA1}
        , {0x00A2, 0xA2}, {0x00A3, 0xA3}, {0x00A4, 0xA4}, {0x00A5, 0xA5}
        , {0x00A6, 0xA6}, {0x00A7, 0xA7}, {0x00A8, 0xA8}, {0x00A9, 0xA9}
        , {0x00AA, 0xAA}, {0x00AB, 0xAB}, {0x00AC, 0xAC}, {0x00AD, 0xAD}
        , {0x00AE, 0xAE}, {0x00AF, 0xAF}, {0x00B0, 0xB0}, {0x00B1, 0xB1}
        , {0x00B2, 0xB2}, {0x00B3, 0xB3}, {0x00B4, 0xB4}, {0x00B5, 0xB5}
        , {0x00B6, 0xB6}, {0x00B7, 0xB7}, {0x00B8, 0xB8}, {0x00B9, 0xB9}
        , {0x00BA, 0xBA}, {0x00BB, 0xBB}, {0x00BC, 0xBC}, {0x00BD, 0xBD}
        , {0x00BE, 0xBE}, {0x00BF, 0xBF}, {0x00C0, 0xC0}, {0x00C1, 0xC1}
        , {0x00C2, 0xC2}, {0x00C4, 0xC4}, {0x00C5, 0xC5}, {0x00C6, 0xC6}
        , {0x00C7, 0xC7}, {0x00C8, 0xC8}, {0x00C9, 0xC9}, {0x00CA, 0xCA}
        , {0x00CB, 0xCB}, {0x00CD, 0xCD}, {0x00CE, 0xCE}, {0x00CF, 0xCF}
        , {0x00D1, 0xD1}, {0x00D3, 0xD3}, {0x00D4, 0xD4}, {0x00D6, 0xD6}
        , {0x00D7, 0xD7}, {0x00D8, 0xD8}, {0x00D9, 0xD9}, {0x00DA, 0xDA}
        , {0x00DB, 0xDB}, {0x00DC, 0xDC}, {0x00DF, 0xDF}, {0x00E0, 0xE0}
        , {0x00E1, 0xE1}, {0x00E2, 0xE2}, {0x00E4, 0xE4}, {0x00E5, 0xE5}
        , {0x00E6, 0xE6}, {0x00E7, 0xE7}, {0x00E8, 0xE8}, {0x00E9, 0xE9}
        , {0x00EA, 0xEA}, {0x00EB, 0xEB}, {0x00ED, 0xED}, {0x00EE, 0xEE}
        , {0x00EF, 0xEF}, {0x00F1, 0xF1}, {0x00F3, 0xF3}, {0x00F4, 0xF4}
        , {0x00F6, 0xF6}, {0x00F7, 0xF7}, {0x00F8, 0xF8}, {0x00F9, 0xF9}
        , {0x00FA, 0xFA}, {0x00FB, 0xFB}, {0x00FC, 0xFC}, {0x00FF, 0xFF}
        , {0x0102, 0xC3}, {0x0103, 0xE3}, {0x0110, 0xD0}, {0x0111, 0xF0}
        , {0x0152, 0x8C}, {0x0153, 0x9C}, {0x0178, 0x9F}, {0x0192, 0x83}
        , {0x01A0, 0xD5}, {0x01A1, 0xF5}, {0x01AF, 0xDD}, {0x01B0, 0xFD}
        , {0x02C6, 0x88}, {0x02DC, 0x98}, {0x0300, 0xCC}, {0x0301, 0xEC}
        , {0x0303, 0xDE}, {0x0309, 0xD2}, {0x0323, 0xF2}, {0x2013, 0x96}
        , {0x2014, 0x97}, {0x2018, 0x91}, {0x2019, 0x92}, {0x201A, 0x82}
        , {0x201C, 0x93}, {0x201D, 0x94}, {0x201E, 0x84}, {0x2020, 0x86}
        , {0x2021, 0x87}, {0x2022, 0x95}, {0x2026, 0x85}, {0x2030, 0x89}
        , {0x2039, 0x8B}, {0x203A, 0x9B}, {0x20AB, 0xFE}, {0x20AC, 0x80}
        , {0x2122, 0x99}, {0xFFFD, '?'} };

    const ch32_to_char* char_map_end = char_map + detail::array_size(char_map);
    const auto *it = strf::detail::lower_bound
        ( char_map, char_map_end, ch32_to_char{ch, 0}, cmp_ch32_to_char{} );
    return it != char_map_end && it->key == ch ? it->value : 0x100;
}

#endif //! defined(STRF_OMIT_IMPL)

template <typename SrcCharT, typename DstCharT, class Impl>
struct single_byte_charset_to_utf32
{
    using src_code_unit = SrcCharT;
    using dst_code_unit = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static constexpr STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept
    {
        STRF_ASSERT(src <= src_end);
        if (strf::with_stop_on_invalid_sequence(flags)) {
            return Impl::find_invalid_sequence(src, src_end, limit);
        }
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags )
    {
        STRF_ASSERT(src <= src_end);
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }

    static STRF_HD strf::transcode_f<SrcCharT, DstCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DstCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DstCharT, class Impl>
STRF_HD strf::transcode_result<SrcCharT, DstCharT>
single_byte_charset_to_utf32<SrcCharT, DstCharT, Impl>::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    (void) flags;
    auto *dst_it = dst;
    for (; src < src_end; ++src, ++dst_it) {
        char32_t ch32 = Impl::decode(static_cast<std::uint8_t>(*src));
        STRF_IF_UNLIKELY (ch32 == 0xFFFD) {
            if (err_notifier) {
                err_notifier->invalid_sequence(sizeof(SrcCharT), Impl::name(), src, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src, dst_it, reason::invalid_sequence};
            }
            ch32 = 0xFFFD;
        }
        STRF_IF_UNLIKELY (dst_it == dst_end) {
            return {src, dst_it, reason::insufficient_output_space};
        }
        *dst_it = static_cast<DstCharT>(ch32);
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT, class Impl>
STRF_HD strf::transcode_result<SrcCharT, DstCharT>
single_byte_charset_to_utf32<SrcCharT, DstCharT, Impl>::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    for (; src < src_end; ++src, ++dst_it) {
        STRF_IF_UNLIKELY (dst_it == dst_end) {
            return {src, dst_it, reason::insufficient_output_space};
        }
        const char32_t ch32 = Impl::decode(static_cast<std::uint8_t>(*src));
        *dst_it = static_cast<DstCharT>(ch32);
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT, class Impl>
struct utf32_to_single_byte_charset
{
    using src_code_unit = SrcCharT;
    using dst_code_unit = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static constexpr STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags) noexcept
    {
        STRF_ASSERT(src <= src_end);
        const bool strict_surrogates = strf::with_strict_surrogate_policy(flags);
        if ( strf::with_stop_on_unsupported_codepoint(flags) &&
             strf::with_stop_on_invalid_sequence(flags) ) {
            return Impl::find_first_unsupported_or_invalid_codepoint
                (src, src_end, limit, strict_surrogates);
        }
        if (strf::with_stop_on_invalid_sequence(flags)) {
            return Impl::find_first_invalid_codepoint
                (src, src_end, limit, strict_surrogates);
        }
        if (strf::with_stop_on_unsupported_codepoint(flags)) {
            return Impl::find_first_valid_unsupported_codepoint
                (src, src_end, limit, strict_surrogates);
        }
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static constexpr STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags) noexcept
    {
        STRF_ASSERT(src <= src_end);
        using stop_reason = strf::transcode_stop_reason;
        if (! strf::with_stop_on_unsupported_codepoint(flags)) {
            if (src_end - src <= limit) {
                return {src_end - src, src_end, stop_reason::completed};
            }
            return {limit, src + limit, stop_reason::insufficient_output_space};
        }
        return Impl::find_first_unsupported_codepoint(src, src_end, limit);
    }
    static STRF_HD strf::transcode_f<SrcCharT, DstCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DstCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DstCharT, class Impl>
STRF_HD strf::transcode_result<SrcCharT, DstCharT>
utf32_to_single_byte_charset<SrcCharT, DstCharT, Impl>::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags)
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    for (; src != src_end; ++src, ++dst_it) {
        unsigned ch = Impl::encode(detail::cast_u32(*src));
        STRF_IF_UNLIKELY (ch >= 0x100) {
            const bool is_invalid =
                ( *src >= 0x110000
               || ( strf::with_strict_surrogate_policy(flags)
                 && 0xD800 <= *src
                 && *src <= 0xFFFF));

            if (!is_invalid) {
                if (err_notifier) {
                    auto codepoint = detail::cast_unsigned(*src);
                    err_notifier->unsupported_codepoint(Impl::name(), codepoint);
                }
                if (strf::with_stop_on_unsupported_codepoint(flags)) {
                    return {src, dst_it, reason::unsupported_codepoint};
                }
            } else {
                if (err_notifier) {
                    err_notifier->invalid_sequence(4, "UTF-32", src, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {src, dst_it, reason::invalid_sequence};
                }
            }
            ch = static_cast<unsigned char>('?');
        }
        STRF_IF_UNLIKELY (dst_it == dst_end) {
            return {src, dst_it, reason::insufficient_output_space};
        }
        *dst_it = static_cast<DstCharT>(ch);
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT, class Impl>
STRF_HD strf::transcode_result<SrcCharT, DstCharT>
utf32_to_single_byte_charset<SrcCharT, DstCharT, Impl>::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    for (; src != src_end; ++src, ++dst_it) {
        unsigned ch = Impl::encode(detail::cast_u32(*src));
        STRF_IF_UNLIKELY (ch >= 0x100) {
            if (err_notifier) {
                auto codepoint = detail::cast_unsigned(*src);
                err_notifier->unsupported_codepoint(Impl::name(), codepoint);
            }
            if (strf::with_stop_on_unsupported_codepoint(flags)) {
                return {src, dst_it, reason::unsupported_codepoint};
            }
            ch = U'?';
        }
        STRF_IF_UNLIKELY (dst_it == dst_end) {
            return {src, dst_it, reason::insufficient_output_space};
        }
        * dst_it = static_cast<DstCharT>(ch);
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT, class Impl>
struct single_byte_charset_to_itself
{
    using src_code_unit = SrcCharT;
    using dst_code_unit = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static constexpr STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags) noexcept
    {
        STRF_ASSERT(src <= src_end);
        if (strf::with_stop_on_invalid_sequence(flags)) {
            return Impl::find_invalid_sequence(src, src_end, limit);
        }
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags )
    {
        return detail::bypass_unsafe_transcode(src, src_end, dst, dst_end, err_notifier, flags);
    }

    static STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags )
    {
        STRF_ASSERT(src <= src_end);
        using stop_reason = strf::transcode_stop_reason;
        if (src_end - src <= limit) {
            return {src_end - src, src_end, stop_reason::completed};
        }
        return {limit, src + limit, stop_reason::insufficient_output_space};
    }

    static STRF_HD strf::transcode_f<SrcCharT, DstCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DstCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DstCharT, class Impl>
STRF_HD strf::transcode_result<SrcCharT, DstCharT>
single_byte_charset_to_itself<SrcCharT, DstCharT, Impl>::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    for (; src < src_end; ++src, ++dst_it) {
        auto ch = static_cast<std::uint8_t>(*src);
        STRF_IF_UNLIKELY (!Impl::is_valid(ch)) {
            STRF_IF_UNLIKELY (err_notifier) {
                err_notifier->invalid_sequence(sizeof(SrcCharT), Impl::name(), src, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src, dst_it, reason::invalid_sequence};
            }
            ch = static_cast<std::uint8_t>('?');
        }
        STRF_IF_UNLIKELY (dst_it == dst_end) {
            return {src, dst_it, reason::insufficient_output_space};
        }
        *dst_it = static_cast<DstCharT>(ch);
    }
    return {src, dst_it, reason::completed};
}

template <std::ptrdiff_t wchar_size, typename CharT, strf::charset_id>
class single_byte_charset_tofrom_wchar
{
public:

    static STRF_HD strf::dynamic_transcoder<wchar_t, CharT>
    find_transcoder_from(strf::tag<wchar_t>, strf::charset_id) noexcept
    {
        return {};
    }
    static STRF_HD strf::dynamic_transcoder<CharT, wchar_t>
    find_transcoder_to(strf::tag<wchar_t>, strf::charset_id) noexcept
    {
        return {};
    }

protected:

    constexpr static std::nullptr_t find_transcoder_to_wchar = nullptr;
    constexpr static std::nullptr_t  find_transcoder_from_wchar = nullptr;
};

template <typename CharT, strf::charset_id Id>
class single_byte_charset_tofrom_wchar<4, CharT, Id>
{
public:

    static STRF_HD strf::dynamic_transcoder<wchar_t, CharT>
    find_transcoder_from(strf::tag<wchar_t>, strf::charset_id id) noexcept
    {
        return find_transcoder_from_wchar(id);
    }
    static STRF_HD strf::dynamic_transcoder<CharT, wchar_t>
    find_transcoder_to(strf::tag<wchar_t>, strf::charset_id id) noexcept
    {
        return find_transcoder_to_wchar(id);
    }

protected:

    static STRF_HD strf::dynamic_transcoder<wchar_t, CharT>
    find_transcoder_from_wchar(strf::charset_id id) noexcept
    {
        using return_type = strf::dynamic_transcoder<wchar_t, CharT>;
        if (id == strf::csid_utf32) {
            const strf::static_transcoder<wchar_t, CharT, strf::csid_utf32, Id> t;
            return return_type{t};
        }
        return {};
    }
    static STRF_HD strf::dynamic_transcoder<CharT, wchar_t>
    find_transcoder_to_wchar(strf::charset_id id) noexcept
    {
        using return_type = strf::dynamic_transcoder<CharT, wchar_t>;
        if (id == strf::csid_utf32) {
            const strf::static_transcoder<CharT, wchar_t, Id, strf::csid_utf32> t;
            return return_type{t};
        }
        return {};
    }
};


template <typename CharT, class Impl>
class single_byte_charset
    : public strf::detail::single_byte_charset_tofrom_wchar
        < sizeof(wchar_t), CharT, Impl::id >
{
    static_assert(sizeof(CharT) == 1, "Character type with this encoding");

    using wchar_stuff_ =
        strf::detail::single_byte_charset_tofrom_wchar<sizeof(wchar_t), CharT, Impl::id>;
public:

    using code_unit = CharT;

    static STRF_HD const char* name() noexcept
    {
        return Impl::name();
    };
    static constexpr STRF_HD strf::charset_id id() noexcept
    {
        return Impl::id;
    }
    static constexpr STRF_HD char32_t replacement_char() noexcept
    {
        return U'?';
    }
    static constexpr STRF_HD int replacement_char_size() noexcept
    {
        return 1;
    }
    static STRF_HD void write_replacement_char(strf::transcode_dst<CharT>& dst) noexcept
    {
        strf::put(dst, static_cast<CharT>('?'));
    }
    static STRF_HD int validate(char32_t ch32) noexcept
    {
        return Impl::encode(ch32) < 0x100 ? 1 : -1;
    }
    static constexpr STRF_HD int encoded_char_size(char32_t) noexcept
    {
        return 1;
    }
    static STRF_HD CharT* encode_char(CharT* dst, char32_t ch) noexcept;

    static STRF_HD void encode_fill
        ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept
    {
        (void) src;
        STRF_ASSERT(src <= src_end);
        const auto src_size = src_end - src;
        if (max_count < src_size) {
            return {max_count, src + max_count};
        }
        return {src_size, src_end};
    }
    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept
    {
        STRF_ASSERT(src <= src_end);
        const auto src_size = src_end - src;
        if (max_count < src_size) {
            return {max_count, src + max_count};
        }
        return {src_size, src_end};
    }
    static STRF_HD char32_t decode_unit(CharT ch) noexcept
    {
        return Impl::decode(static_cast<std::uint8_t>(ch));
    }
    static STRF_HD strf::encode_char_f<CharT> encode_char_func() noexcept
    {
        return encode_char;
    }
    static STRF_HD strf::encode_fill_f<CharT> encode_fill_func() noexcept
    {
        return encode_fill;
    }
    static STRF_HD strf::validate_f validate_func() noexcept
    {
        return validate;
    }
    static STRF_HD
    strf::write_replacement_char_f<CharT> write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr
    STRF_HD strf::static_transcoder<CharT, CharT, Impl::id, Impl::id>
    sanitizer() noexcept
    {
        return {};
    }
    static constexpr STRF_HD
    strf::static_transcoder<char32_t, CharT, strf::csid_utf32, Impl::id>
    from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD
    strf::static_transcoder<CharT, char32_t, Impl::id, strf::csid_utf32>
    to_u32() noexcept
    {
        return {};
    }

    using wchar_stuff_::find_transcoder_from;
    using wchar_stuff_::find_transcoder_to;

    static STRF_HD strf::dynamic_transcoder<char, CharT>
    find_transcoder_from(strf::tag<char>, strf::charset_id id) noexcept
    {
        return find_transcoder_from_narrow<char>(id);
    }
    static STRF_HD strf::dynamic_transcoder<CharT, char>
    find_transcoder_to(strf::tag<char>, strf::charset_id id) noexcept
    {
        return find_transcoder_to_narrow<char>(id);
    }

#if defined (__cpp_char8_t)

    static STRF_HD strf::dynamic_transcoder<char8_t, CharT>
    find_transcoder_from(strf::tag<char8_t>, strf::charset_id id) noexcept
    {
        return find_transcoder_from_narrow<char8_t>(id);
    }
    static STRF_HD strf::dynamic_transcoder<CharT, char8_t>
    find_transcoder_to(strf::tag<char8_t>, strf::charset_id id) noexcept
    {
        return find_transcoder_to_narrow<char8_t>(id);
    }

#endif

    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, count_codepoints_fast,
            count_codepoints, write_replacement_char, decode_unit,
            strf::dynamic_transcoder<CharT, CharT>{sanitizer()},
            strf::dynamic_transcoder<char32_t, CharT>{from_u32()},
            strf::dynamic_transcoder<CharT, char32_t>{to_u32()},
            wchar_stuff_::find_transcoder_from_wchar,
            wchar_stuff_::find_transcoder_to_wchar,
            nullptr,
            nullptr,
            find_transcoder_from_narrow<char>,
            find_transcoder_to_narrow<char>,
#if defined (__cpp_char8_t)
            find_transcoder_from_narrow<char8_t>,
            find_transcoder_to_narrow<char8_t>,
#else
            nullptr,
            nullptr,
#endif // defined (__cpp_char8_t)
        };
        return strf::dynamic_charset<CharT>{data};
    }

    static STRF_HD strf::dynamic_charset_data<CharT> make_data() noexcept
    {
        return {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, count_codepoints_fast,
            count_codepoints, write_replacement_char, decode_unit,
            strf::dynamic_transcoder<CharT, CharT>{sanitizer()},
            strf::dynamic_transcoder<char32_t, CharT>{from_u32()},
            strf::dynamic_transcoder<CharT, char32_t>{to_u32()},
            wchar_stuff_::find_transcoder_from_wchar,
            wchar_stuff_::find_transcoder_to_wchar,
            nullptr,
            nullptr,
            find_transcoder_from_narrow<char>,
            find_transcoder_to_narrow<char>,
#if defined (__cpp_char8_t)
            find_transcoder_from_narrow<char8_t>,
            find_transcoder_to_narrow<char8_t>,
#else
            nullptr,
            nullptr,
#endif // defined (__cpp_char8_t)
        };
    }

private:

    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from_narrow(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == Impl::id) {
            const static_transcoder<SrcCharT, CharT, Impl::id, Impl::id> t;
            return transcoder_type{ t };
        }
        return {};
    }
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to_narrow(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DstCharT>;
        if (id == Impl::id) {
            const static_transcoder<CharT, DstCharT, Impl::id, Impl::id> t;
            return transcoder_type{ t };
        }
        return {};
    }

};

template <typename CharT, class Impl>
STRF_HD CharT* single_byte_charset<CharT, Impl>::encode_char
    ( CharT* dst
    , char32_t ch ) noexcept
{
    auto ch2 = Impl::encode(ch);
    const bool valid = (ch2 < 0x100);
    *dst = static_cast<CharT>(valid * ch2 + (!valid) * '?');
    return dst + 1;
}

template <typename CharT, class Impl>
STRF_HD void single_byte_charset<CharT, Impl>::encode_fill
    ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch )
{
    unsigned ch2 = Impl::encode(ch);
    STRF_IF_UNLIKELY (ch2 >= 0x100) {
        ch2 = '?';
    }
    auto ch3 = static_cast<CharT>(ch2);
    while(true) {
        const std::ptrdiff_t available = dst.buffer_sspace();
        STRF_IF_LIKELY (count <= available) {
            strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), count, ch3);
            dst.advance(count);
            return;
        }
        strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), available, ch3);
        dst.advance_to(dst.buffer_end());
        count -= available;
        dst.flush();
    }
}

} // namespace detail

STRF_DEF_SINGLE_BYTE_CHARSET(ascii);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_1);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_2);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_3);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_4);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_5);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_6);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_7);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_8);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_9);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_10);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_11);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_13);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_14);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_15);
STRF_DEF_SINGLE_BYTE_CHARSET(iso_8859_16);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1250);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1251);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1252);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1253);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1254);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1255);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1256);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1257);
STRF_DEF_SINGLE_BYTE_CHARSET(windows_1258);

} // namespace strf

#undef STRF_SBC_CHECK_DST
#undef STRF_SBC_CHECK_DST_SIZE

#endif  // STRF_DETAIL_SINGLE_BYTE_CHARSETS_HPP
