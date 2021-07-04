#ifndef STRF_DETAIL_UTF_HPP
#define STRF_DETAIL_UTF_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>
#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

#if ! defined(STRF_CHECK_DEST)

#define STRF_CHECK_DEST                         \
    STRF_IF_UNLIKELY (dest_it == dest_end) {    \
        ob.advance_to(dest_it);                 \
        ob.recycle();                           \
        STRF_IF_UNLIKELY (!ob.good()) {         \
            return;                             \
        }                                       \
        dest_it = ob.pointer();                 \
        dest_end = ob.end();                    \
    }

#define STRF_CHECK_DEST_SIZE(SIZE)                  \
    STRF_IF_UNLIKELY (dest_it + SIZE > dest_end) {  \
        ob.advance_to(dest_it);                     \
        ob.recycle();                               \
        STRF_IF_UNLIKELY (!ob.good()) {             \
            return;                                 \
        }                                           \
        dest_it = ob.pointer();                     \
        dest_end = ob.end();                        \
    }

#endif // ! defined(STRF_CHECK_DEST)

namespace detail {

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::basic_outbuff<CharT>& ob
    , std::size_t count
    , CharT ch0
    , CharT ch1 ) noexcept
{
    auto p = ob.pointer();
    constexpr std::size_t seq_size = 2;
    std::size_t space;
    std::size_t inner_count;
    while (1) {
        space = (ob.end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p += seq_size;
        }
        ob.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        ob.recycle();
        STRF_IF_UNLIKELY (!ob.good()) {
            return;
        }
        p = ob.pointer();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::basic_outbuff<CharT>& ob
    , std::size_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2 ) noexcept
{
    auto p = ob.pointer();
    constexpr std::size_t seq_size = 3;
    std::size_t space;
    std::size_t inner_count;
    while (1) {
        space = (ob.end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p += seq_size;
        }
        ob.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        ob.recycle();
        STRF_IF_UNLIKELY (!ob.good()) {
            return;
        }
        p = ob.pointer();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::basic_outbuff<CharT>& ob
    , std::size_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2
    , CharT ch3 ) noexcept
{
    auto p = ob.pointer();
    constexpr std::size_t seq_size = 4;
    std::size_t space;
    std::size_t inner_count;
    while (1) {
        space = (ob.end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p[3] = ch3;
            p += seq_size;
        }
        ob.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        ob.recycle();
        STRF_IF_UNLIKELY (!ob.good()) {
            return;
        }
        p = ob.pointer();
        count -= space;
    }
}

constexpr STRF_HD bool is_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 11 == 0x1B;
}
constexpr STRF_HD bool is_high_surrogate(std::uint32_t codepoint) noexcept
{
    return codepoint >> 10 == 0x36;
}
constexpr STRF_HD bool is_low_surrogate(std::uint32_t codepoint) noexcept
{
    return codepoint >> 10 == 0x37;
}
constexpr STRF_HD bool not_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 11 != 0x1B;
}
constexpr STRF_HD  bool not_high_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 10 != 0x36;
}
constexpr STRF_HD  bool not_low_surrogate(std::uint32_t codepoint)
{
    return codepoint >> 10 != 0x37;
}
constexpr STRF_HD std::uint16_t utf8_decode(std::uint16_t ch0, std::uint16_t ch1)
{
    return (((ch0 & 0x1F) << 6) |
            ((ch1 & 0x3F) << 0));
}
constexpr STRF_HD std::uint16_t utf8_decode(std::uint16_t ch0, std::uint16_t ch1, std::uint16_t ch2)
{
    return (((ch0 & 0x0F) << 12) |
            ((ch1 & 0x3F) <<  6) |
            ((ch2 & 0x3F) <<  0));
}
constexpr STRF_HD std::uint32_t utf8_decode(std::uint32_t ch0, std::uint32_t ch1, std::uint32_t ch2, std::uint32_t ch3)
{
    return (((ch0 & 0x07) << 18) |
            ((ch1 & 0x3F) << 12) |
            ((ch2 & 0x3F) <<  6) |
            ((ch3 & 0x3F) <<  0));
}
constexpr STRF_HD bool is_utf8_continuation(std::uint8_t ch)
{
    return (ch & 0xC0) == 0x80;
}

constexpr STRF_HD bool valid_start_3bytes
    ( std::uint8_t ch0
    , std::uint8_t ch1
    , strf::surrogate_policy surr_poli )
{
    return ( (ch0 != 0xE0 || ch1 != 0x80)
          && ( surr_poli == strf::surrogate_policy::lax
            || (0x1B != (((ch0 & 0xF) << 1) | ((ch1 >> 5) & 1)))) );
}

inline STRF_HD unsigned utf8_decode_first_2_of_3(std::uint8_t ch0, std::uint8_t ch1)
{
    return ((ch0 & 0x0F) << 6) | (ch1 & 0x3F);
}

inline STRF_HD bool first_2_of_3_are_valid(unsigned x, strf::surrogate_policy surr_poli)
{
    return ( surr_poli == strf::surrogate_policy::lax
          || (x >> 5) != 0x1B );
}
inline STRF_HD bool first_2_of_3_are_valid
    ( std::uint8_t ch0
    , std::uint8_t ch1
    , strf::surrogate_policy surr_poli )
{
    return first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), surr_poli);
}

inline STRF_HD unsigned utf8_decode_first_2_of_4(std::uint8_t ch0, std::uint8_t ch1)
{
    return ((ch0 ^ 0xF0) << 6) | (ch1 & 0x3F);
}

inline STRF_HD unsigned utf8_decode_last_2_of_4(unsigned long x, unsigned ch2, unsigned ch3)
{
    return (x << 12) | ((ch2 & 0x3F) <<  6) | (ch3 & 0x3F);
}

inline STRF_HD bool first_2_of_4_are_valid(unsigned x)
{
    return 0xF < x && x < 0x110;
}

inline STRF_HD bool first_2_of_4_are_valid(std::uint8_t ch0, std::uint8_t ch1)
{
    return first_2_of_4_are_valid(utf8_decode_first_2_of_4(ch0, ch1));
}

} // namespace detail

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-8");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-8");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-1");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy )
    {
        (void) src;
        return src_size;
    }
    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
using utf8_to_utf8 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >;
template <typename SrcCharT, typename DestCharT>
using utf8_to_utf16 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >;
template <typename SrcCharT, typename DestCharT>
using utf8_to_utf32 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >;

template <typename SrcCharT, typename DestCharT>
using utf16_to_utf8 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >;
template <typename SrcCharT, typename DestCharT>
using utf16_to_utf16 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >;
template <typename SrcCharT, typename DestCharT>
using utf16_to_utf32 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >;

template <typename SrcCharT, typename DestCharT>
using utf32_to_utf8 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >;
template <typename SrcCharT, typename DestCharT>
using utf32_to_utf16 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >;
template <typename SrcCharT, typename DestCharT>
using utf32_to_utf32 = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >;

template <typename SrcCharT, typename DestCharT>
using utf_to_utf = strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf<SrcCharT>, strf::csid_utf<DestCharT> >;

template <typename CharT>
class static_charset<CharT, strf::csid_utf8>
{
public:
    static_assert(sizeof(CharT) == 1, "Incompatible character type for UTF-8");
    using code_unit = CharT;
    using char_type STRF_DEPRECATED = CharT;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-8";
    };
    static constexpr STRF_HD strf::charset_id id() noexcept
    {
        return strf::csid_utf8;
    }
    static constexpr STRF_HD char32_t replacement_char() noexcept
    {
        return 0xFFFD;
    }
    static constexpr STRF_HD std::size_t replacement_char_size() noexcept
    {
        return 3;
    }
    static constexpr STRF_HD std::size_t validate(char32_t ch) noexcept
    {
        return ( ch < 0x80     ? 1 :
                 ch < 0x800    ? 2 :
                 ch < 0x10000  ? 3 :
                 ch < 0x110000 ? 4 : (std::size_t)-1 );
    }
    static constexpr STRF_HD std::size_t encoded_char_size(char32_t ch) noexcept
    {
        return ( ch < 0x80     ? 1 :
                 ch < 0x800    ? 2 :
                 ch < 0x10000  ? 3 :
                 ch < 0x110000 ? 4 : 3 );
    }
    static STRF_HD CharT* encode_char
        ( CharT* dest, char32_t ch ) noexcept;

    static STRF_HD void encode_fill
        ( strf::basic_outbuff<CharT>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count ) noexcept;

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::basic_outbuff<CharT>& );

    static STRF_HD char32_t decode_unit(CharT ch) noexcept
    {
        STRF_IF_LIKELY (ch < 0x80)
            return static_cast<char32_t>(ch);
        return 0xFFFD;
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
    static STRF_HD strf::write_replacement_char_f<CharT>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf8_to_utf8<CharT, CharT> sanitizer() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf8<char32_t, CharT> from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf8_to_utf32<CharT, char32_t> to_u32() noexcept
    {
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::tag<SrcCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_from<SrcCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::tag<DestCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DestCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DestCharT>;
        if (id == strf::csid_utf<DestCharT>) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::csid_utf<SrcCharT>) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 3, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_unit,
            strf::dynamic_transcoder<CharT,    CharT>{sanitizer()},
            strf::dynamic_transcoder<char32_t, CharT>{from_u32()},
            strf::dynamic_transcoder<CharT, char32_t>{to_u32()},
            find_transcoder_from<wchar_t>,
            find_transcoder_to<wchar_t>,
            find_transcoder_from<char16_t>,
            find_transcoder_to<char16_t>,
            find_transcoder_from<char>,
            find_transcoder_to<char>,
#if defined (__cpp_char8_t)
            find_transcoder_from<char8_t>,
            find_transcoder_to<char8_t>
#else
            nullptr,
            nullptr
#endif // defined (__cpp_char8_t)
        };
        return strf::dynamic_charset<CharT>{data};
    }
    explicit operator strf::dynamic_charset<CharT> () const
    {
        return to_dynamic();
    }
};

template <typename CharT>
class static_charset<CharT, strf::csid_utf16>
{
public:
    static_assert(sizeof(CharT) == 2, "Incompatible character type for UTF-16");
    using code_unit = CharT;
    using char_type STRF_DEPRECATED = CharT;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-16";
    };
    static constexpr STRF_HD strf::charset_id id() noexcept
    {
        return strf::csid_utf16;
    }
    static constexpr STRF_HD char32_t replacement_char() noexcept
    {
        return 0xFFFD;
    }
    static constexpr STRF_HD std::size_t replacement_char_size() noexcept
    {
        return 1;
    }
    static constexpr STRF_HD std::size_t validate(char32_t ch) noexcept
    {
        return ch < 0x10000 ? 1 : ch < 0x110000 ? 2 : (std::size_t)-1;
    }
    static constexpr STRF_HD std::size_t encoded_char_size(char32_t ch) noexcept
    {
        return (std::size_t)1 + (0x10000 <= ch && ch < 0x110000);
    }

    static STRF_HD CharT* encode_char
        (CharT* dest, char32_t ch) noexcept;

    static STRF_HD void encode_fill
        ( strf::basic_outbuff<CharT>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count ) noexcept;

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::basic_outbuff<CharT>& );

    static STRF_HD char32_t decode_unit(CharT ch) noexcept
    {
        return ch;
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
    static STRF_HD strf::write_replacement_char_f<CharT>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf16_to_utf16<CharT, CharT> sanitizer() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf16<char32_t, CharT> from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf16_to_utf32<CharT, char32_t> to_u32() noexcept
    {
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::tag<SrcCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_from<SrcCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::tag<DestCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DestCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DestCharT>;
        if (id == strf::csid_utf<DestCharT>) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::csid_utf<SrcCharT>) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_unit,
            strf::dynamic_transcoder<CharT,    CharT>{sanitizer()},
            strf::dynamic_transcoder<char32_t, CharT>{from_u32()},
            strf::dynamic_transcoder<CharT, char32_t>{to_u32()},
            find_transcoder_from<wchar_t>,
            find_transcoder_to<wchar_t>,
            find_transcoder_from<char16_t>,
            find_transcoder_to<char16_t>,
            find_transcoder_from<char>,
            find_transcoder_to<char>,
#if defined (__cpp_char8_t)
            find_transcoder_from<char8_t>,
            find_transcoder_to<char8_t>
#else
            nullptr,
            nullptr
#endif // defined (__cpp_char8_t)
        };
        return strf::dynamic_charset<CharT>{data};
    }
    explicit operator strf::dynamic_charset<CharT> () const
    {
        return to_dynamic();
    }
};

template <typename CharT>
class static_charset<CharT, strf::csid_utf32>
{
public:
    static_assert(sizeof(CharT) == 4, "Incompatible character type for UTF-32");
    using code_unit = CharT;
    using char_type STRF_DEPRECATED = CharT;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-32";
    };
    static constexpr STRF_HD strf::charset_id id() noexcept
    {
        return strf::csid_utf32;
    }
    static constexpr STRF_HD char32_t replacement_char() noexcept
    {
        return 0xFFFD;
    }
    static constexpr STRF_HD std::size_t replacement_char_size() noexcept
    {
        return 1;
    }
    static constexpr STRF_HD char32_t u32equivalence_begin() noexcept
    {
        return 0;
    }
    static constexpr STRF_HD char32_t u32equivalence_end() noexcept
    {
        return 0x10FFFF;
    }
    static constexpr STRF_HD std::size_t validate(char32_t) noexcept
    {
        return 1;
    }
    static constexpr STRF_HD std::size_t encoded_char_size(char32_t) noexcept
    {
        return 1;
    }
    static STRF_HD CharT* encode_char
        (CharT* dest, char32_t ch) noexcept
    {
        *dest = ch;
        return dest + 1;
    }
    static STRF_HD void encode_fill
        ( strf::basic_outbuff<CharT>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count ) noexcept
    {
        (void) src;
        if (max_count <= src_size) {
            return {max_count, max_count};
        }
        return {src_size, src_size};
    }

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const CharT* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept
    {
        (void)surr_poli;
        (void) src;
        if (max_count <= src_size) {
            return {max_count, max_count};
        }
        return {src_size, src_size};
    }

    static STRF_HD void write_replacement_char
        ( strf::basic_outbuff<CharT>& );

    static STRF_HD char32_t decode_unit(CharT ch) noexcept
    {
        return ch;
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
    static STRF_HD strf::write_replacement_char_f<CharT>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf32_to_utf32<CharT, CharT> sanitizer() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf32<char32_t, CharT> from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf32<CharT, char32_t> to_u32() noexcept
    {
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::tag<SrcCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_from<SrcCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::tag<DestCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DestCharT>(id);
    }
    template <typename DestCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DestCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DestCharT>;
        if (id == strf::csid_utf<DestCharT>) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::csid_utf<SrcCharT>) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_unit,
            strf::dynamic_transcoder<CharT,    CharT>{sanitizer()},
            strf::dynamic_transcoder<char32_t, CharT>{from_u32()},
            strf::dynamic_transcoder<CharT, char32_t>{to_u32()},
            find_transcoder_from<wchar_t>,
            find_transcoder_to<wchar_t>,
            find_transcoder_from<char16_t>,
            find_transcoder_to<char16_t>,
            find_transcoder_from<char>,
            find_transcoder_to<char>,
#if defined (__cpp_char8_t)
            find_transcoder_from<char8_t>,
            find_transcoder_to<char8_t>,
#else
            nullptr,
            nullptr
#endif // defined (__cpp_char8_t)
        };
        return strf::dynamic_charset<CharT>{data};
    }
    explicit operator strf::dynamic_charset<CharT> () const
    {
        return to_dynamic();
    }
};

template <typename CharT>
using utf8_impl = static_charset<CharT, strf::csid_utf8>;

template <typename CharT>
using utf16_impl = static_charset<CharT, strf::csid_utf16>;

template <typename CharT>
using utf32_impl = static_charset<CharT, strf::csid_utf32>;

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    <SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;
    using strf::detail::utf8_decode_last_2_of_4;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::is_utf8_continuation;

    std::uint8_t ch0, ch1, ch2, ch3;
    unsigned long x;
    auto src_it = src;
    auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    DestCharT ch32;

    while(src_it != src_end) {
        ch0 = (*src_it);
        ++src_it;
        if (ch0 < 0x80) {
            ch32 = ch0;
        } else if (0xC0 == (ch0 & 0xE0)) {
            if(ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(ch1 = * src_it)) {
                ch32 = utf8_decode(ch0, ch1);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ch32 = ((ch1 & 0x3F) << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                       , surr_poli )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                ch32 = (x << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
        {
            ch32 = utf8_decode_last_2_of_4(x, ch2, ch3);
            ++src_it;
        } else {
            invalid_sequence:
            ch32 = 0xFFFD;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }

        STRF_CHECK_DEST;
        *dest_it = ch32;
        ++dest_it;
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1;
    auto src_it = src;
    auto src_end = src + src_size;
    std::size_t size = 0;
    while (src_it != src_end) {
        ch0 = (*src_it);
        ++src_it;
        ++size;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it)) {
                ++src_it;
            }
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && ((*src_it & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++src_it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( src_it != src_end && is_utf8_continuation(ch1 = *src_it)
              && first_2_of_3_are_valid( ch0, ch1, surr_poli )
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++src_it;
            }
        } else if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src_it != src_end && is_utf8_continuation(*src_it)
                 && ++src_it != src_end && is_utf8_continuation(*src_it) )
        {
                ++src_it;
        }
    }
    return size;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1, ch2, ch3;
    auto src_it = src;
    auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    while(src_it != src_end) {
        ch0 = (*src_it);
        ++src_it;
        if(ch0 < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = ch0;
            ++dest_it;
        } else if(0xC0 == (ch0 & 0xE0)) {
            if(ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(ch1 = * src_it)) {
                STRF_CHECK_DEST_SIZE(2);
                ++src_it;
                dest_it[0] = ch0;
                dest_it[1] = ch1;
                dest_it += 2;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                STRF_CHECK_DEST_SIZE(3);
                ++src_it;
                dest_it[0] = ch0;
                dest_it[1] = ch1;
                dest_it[2] = ch2;
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                STRF_CHECK_DEST_SIZE(3);
                ++src_it;
                dest_it[0] = ch0;
                dest_it[1] = ch1;
                dest_it[2] = ch2;
                dest_it += 3;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
        {
            STRF_CHECK_DEST_SIZE(4);
            ++src_it;
            dest_it[0] = ch0;
            dest_it[1] = ch1;
            dest_it[2] = ch2;
            dest_it[3] = ch3;
            dest_it += 4;
        } else {
            invalid_sequence:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1;
    const SrcCharT* src_it = src;
    auto src_end = src + src_size;
    std::size_t size = 0;
    while(src_it != src_end) {
        ch0 = *src_it;
        ++src_it;
        if(ch0 < 0x80) {
            ++size;
        } else if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it)) {
                size += 2;
                ++src_it;
            } else {
                size += 3;
            }
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                size += 3;
                ++src_it;
            } else {
                size += 3;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            size += 3;
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( ch0, ch1, surr_poli )
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if( src_it != src_end
              && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
        {
            size += 4;
            ++src_it;
        } else {
            size += 3;
        }
    }
    return size;
}

template <typename CharT>
STRF_HD strf::codepoints_count_result
static_charset<CharT, strf::csid_utf8>::codepoints_fast_count
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count ) noexcept
{
    std::size_t count = 0;
    auto it = src;
    auto end = src + src_size;
    while (it != end && count < max_count) {
        if (!strf::detail::is_utf8_continuation(*it)) {
            ++ count;
        }
        ++it;
    }
    while(it != end && strf::detail::is_utf8_continuation(*it)) {
        ++it;
    }
    return {count, static_cast<std::size_t>(it - src)};
}

template <typename CharT>
STRF_HD strf::codepoints_count_result
static_charset<CharT, strf::csid_utf8>::codepoints_robust_count
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count
    , strf::surrogate_policy surr_poli ) noexcept
{
    using strf::detail::utf8_decode;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1;
    std::size_t count = 0;
    auto it = src;
    auto end = src + src_size;
    while (it != end && count != max_count) {
        ch0 = (*it);
        ++it;
        ++count;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && it != end && is_utf8_continuation(*it)) {
                ++it;
            }
        } else if (0xE0 == ch0) {
            if (   it != end && ((*it & 0xE0) == 0xA0)
              && ++it != end && is_utf8_continuation(*it) )
            {
                ++it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( it != end && is_utf8_continuation(ch1 = *it)
              && first_2_of_3_are_valid( ch0, ch1, surr_poli )
              && ++it != end && is_utf8_continuation(*it) )
            {
                ++it;
            }
        } else if (   it != end && is_utf8_continuation(ch1 = * it)
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++it != end && is_utf8_continuation(*it)
                 && ++it != end && is_utf8_continuation(*it) )
        {
            ++it;
        }
    }
    return {count, static_cast<std::size_t>(it - src)};
}


template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::encode_fill
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x80) {
        strf::detail::write_fill(ob, count, static_cast<CharT>(ch));
    } else if (ch < 0x800) {
        CharT ch0 = static_cast<CharT>(0xC0 | ((ch & 0x7C0) >> 6));
        CharT ch1 = static_cast<CharT>(0x80 |  (ch &  0x3F));
        strf::detail::repeat_sequence<CharT>(ob, count, ch0, ch1);
    } else if (ch <  0x10000) {
        CharT ch0 = static_cast<CharT>(0xE0 | ((ch & 0xF000) >> 12));
        CharT ch1 = static_cast<CharT>(0x80 | ((ch &  0xFC0) >> 6));
        CharT ch2 = static_cast<CharT>(0x80 |  (ch &   0x3F));
        strf::detail::repeat_sequence<CharT>(ob, count, ch0, ch1, ch2);
    } else if (ch < 0x110000) {
        CharT ch0 = static_cast<CharT>(0xF0 | ((ch & 0x1C0000) >> 18));
        CharT ch1 = static_cast<CharT>(0x80 | ((ch &  0x3F000) >> 12));
        CharT ch2 = static_cast<CharT>(0x80 | ((ch &    0xFC0) >> 6));
        CharT ch3 = static_cast<CharT>(0x80 |  (ch &     0x3F));
        strf::detail::repeat_sequence<CharT>(ob, count, ch0, ch1, ch2, ch3);
    } else {
        CharT ch0 = static_cast<CharT>('\xEF');
        CharT ch1 = static_cast<CharT>('\xBF');
        CharT ch2 = static_cast<CharT>('\xBD');
        strf::detail::repeat_sequence<CharT>(ob, count, ch0, ch1, ch2);
    }
}

template <typename CharT>
STRF_HD CharT*
static_charset<CharT, strf::csid_utf8>::encode_char
    ( CharT* dest
    , char32_t ch ) noexcept
{
    STRF_IF_LIKELY (ch < 0x80) {
        *dest = static_cast<CharT>(ch);
        return dest + 1;
    }
    if (ch < 0x800) {
        dest[0] = static_cast<CharT>(0xC0 | ((ch & 0x7C0) >> 6));
        dest[1] = static_cast<CharT>(0x80 |  (ch &  0x3F));
        return dest + 2;
    }
    if (ch <  0x10000) {
        dest[0] = static_cast<CharT>(0xE0 | ((ch & 0xF000) >> 12));
        dest[1] = static_cast<CharT>(0x80 | ((ch &  0xFC0) >> 6));
        dest[2] = static_cast<CharT>(0x80 |  (ch &   0x3F));
        return dest + 3;
    }
    if (ch < 0x110000) {
        dest[0] = static_cast<CharT>(0xF0 | ((ch & 0x1C0000) >> 18));
        dest[1] = static_cast<CharT>(0x80 | ((ch &  0x3F000) >> 12));
        dest[2] = static_cast<CharT>(0x80 | ((ch &    0xFC0) >> 6));
        dest[3] = static_cast<CharT>(0x80 |  (ch &     0x3F));
        return dest + 4;
    }
    dest[0] = static_cast<CharT>('\xEF');
    dest[1] = static_cast<CharT>('\xBF');
    dest[2] = static_cast<CharT>('\xBD');
    return dest + 3;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    auto src_it = src;
    auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for(;src_it != src_end; ++src_it) {
        auto ch = *src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | ((ch & 0x7C0) >> 6));
            dest_it[1] = static_cast<DestCharT>(0x80 |  (ch &  0x3F));
            dest_it += 2;
        } else if (ch < 0x10000) {
            STRF_IF_LIKELY ( surr_poli == strf::surrogate_policy::lax
                          || strf::detail::not_surrogate(ch))
            {
                STRF_CHECK_DEST_SIZE(3);
                dest_it[0] = static_cast<DestCharT>(0xE0 | ((ch & 0xF000) >> 12));
                dest_it[1] = static_cast<DestCharT>(0x80 | ((ch &  0xFC0) >> 6));
                dest_it[2] = static_cast<DestCharT>(0x80 |  (ch &   0x3F));
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (ch < 0x110000) {
            STRF_CHECK_DEST_SIZE(4);
            dest_it[0] = static_cast<DestCharT>(0xF0 | ((ch & 0x1C0000) >> 18));
            dest_it[1] = static_cast<DestCharT>(0x80 | ((ch &  0x3F000) >> 12));
            dest_it[2] = static_cast<DestCharT>(0x80 | ((ch &    0xFC0) >> 6));
            dest_it[3] = static_cast<DestCharT>(0x80 |  (ch &     0x3F));
            dest_it += 4;
        } else {
            invalid_sequence:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
            STRF_IF_UNLIKELY (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    auto src_it = src;
    auto src_end = src + src_size;
    std::size_t count = 0;
    for(;src_it != src_end; ++src_it) {
        auto ch = *src_it;
        STRF_IF_LIKELY (ch < 0x110000) {
            count += 1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000);
        } else {
            count += 3;
        }
    }
    return count;
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::write_replacement_char
    ( strf::basic_outbuff<CharT>& ob )
{
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    STRF_CHECK_DEST_SIZE(3);
    dest_it[0] = static_cast<CharT>('\xEF');
    dest_it[1] = static_cast<CharT>('\xBF');
    dest_it[2] = static_cast<CharT>('\xBD');
    dest_it += 3;
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    unsigned long ch, ch2;
    DestCharT ch32;
    const SrcCharT* src_it_next;
    auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for(auto src_it = src; src_it != src_end; src_it = src_it_next) {
        src_it_next = src_it + 1;
        ch = *src_it;
        src_it_next = src_it + 1;

        STRF_IF_LIKELY (strf::detail::not_surrogate(ch)) {
            ch32 = ch;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src_it_next != src_end
               && strf::detail::is_low_surrogate(ch2 = *src_it_next)) {
            ch32 = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            ++src_it_next;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            ch32 = ch;
        } else {
            ch32 = 0xFFFD;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }

        STRF_CHECK_DEST;
        *dest_it = ch32;
        ++dest_it;
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    unsigned long ch;
    std::size_t count = 0;
    auto src_it = src;
    const auto src_end = src + src_size;
    const SrcCharT* src_it_next;
    for(; src_it != src_end; src_it = src_it_next) {
        src_it_next = src_it + 1;
        ch = *src_it;
        src_it_next = src_it + 1;

        ++count;
        if ( strf::detail::is_high_surrogate(ch)
          && src_it_next != src_end
          && strf::detail::is_low_surrogate(*src_it_next))
        {
            ++src_it_next;
        }
    }
    return count;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    unsigned long ch, ch2;
    auto src_it = src;
    const auto src_end = src + src_size;
    const SrcCharT* src_it_next;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for( ; src_it != src_end; src_it = src_it_next) {
        ch = *src_it;
        src_it_next = src_it + 1;

        STRF_IF_LIKELY (strf::detail::not_surrogate(ch)) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if ( strf::detail::is_high_surrogate(ch)
                 && src_it_next != src_end
                 && strf::detail::is_low_surrogate(ch2 = *src_it_next))
        {
            ++src_it_next;
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(ch);
            dest_it[1] = static_cast<DestCharT>(ch2);
            dest_it += 2;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else {
            STRF_CHECK_DEST;
            *dest_it = 0xFFFD;
            ++dest_it;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::size_t count = 0;
    const SrcCharT* src_it = src;
    const auto src_end = src + src_size;
    unsigned long ch;
    while (src_it != src_end) {
        ch = *src_it;
        ++ src_it;
        ++ count;
        if ( strf::detail::is_high_surrogate(ch)
          && src_it != src_end
          && strf::detail::is_low_surrogate(*src_it) )
        {
            ++ src_it;
            ++ count;
        }
    }
    return count;
}

template <typename CharT>
STRF_HD strf::codepoints_count_result
static_charset<CharT, strf::csid_utf16>::codepoints_fast_count
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count ) noexcept
{
    if (src_size == 0) {
        return {0, 0};
    }
    std::size_t count = 0;
    auto it = src;
    const auto end = src + src_size;
    while (count < max_count) {
        if (strf::detail::is_high_surrogate(*it)) {
            ++it;
        }
        ++it;
        ++count;
        if (it >= end) {
            return {count, src_size};
        }
    }
    return {count, static_cast<std::size_t>(it - src)};
}

template <typename CharT>
STRF_HD strf::codepoints_count_result
static_charset<CharT, strf::csid_utf16>::codepoints_robust_count
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count
    , strf::surrogate_policy surr_poli ) noexcept
{
    (void) surr_poli;
    std::size_t count = 0;
    const CharT* it = src;
    const auto end = src + src_size;
    unsigned long ch;
    while (it != end && count < max_count) {
        ch = *it;
        ++ it;
        ++ count;
        if ( strf::detail::is_high_surrogate(ch) && it != end
          && strf::detail::is_low_surrogate(*it)) {
            ++ it;
        }
    }
    return {count, static_cast<std::size_t>(it - src)};
}

template <typename CharT>
STRF_HD CharT*
static_charset<CharT, strf::csid_utf16>::encode_char
    ( CharT* dest
    , char32_t ch ) noexcept
{
    STRF_IF_LIKELY (ch < 0x10000) {
        *dest = static_cast<CharT>(ch);
        return dest + 1;
    }
    if (ch < 0x110000) {
        char32_t sub_codepoint = ch - 0x10000;
        dest[0] = static_cast<CharT>(0xD800 + (sub_codepoint >> 10));
        dest[1] = static_cast<CharT>(0xDC00 + (sub_codepoint &  0x3FF));
        return dest + 2;
    }
    *dest = 0xFFFD;
    return dest + 1;
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf16>::encode_fill
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x10000) {
        strf::detail::write_fill<CharT>(ob, count, static_cast<CharT>(ch));
    } else if (ch < 0x110000) {
        char32_t sub_codepoint = ch - 0x10000;
        CharT ch0 = static_cast<CharT>(0xD800 + (sub_codepoint >> 10));
        CharT ch1 = static_cast<CharT>(0xDC00 + (sub_codepoint &  0x3FF));
        strf::detail::repeat_sequence<CharT>(ob, count, ch0, ch1);
    } else {
        strf::detail::write_fill<CharT>(ob, count, 0xFFFD);
    }
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    auto src_it = src;
    const auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for ( ; src_it != src_end; ++src_it) {
        auto ch = *src_it;
        STRF_IF_LIKELY (ch < 0x10000) {
            STRF_IF_LIKELY ( surr_poli == strf::surrogate_policy::lax
                          || strf::detail::not_surrogate(ch) )
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(ch);
                ++dest_it;
            } else goto invalid_char;
        } else if (ch < 0x110000) {
            STRF_CHECK_DEST_SIZE(2);
            char32_t sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 | (sub_codepoint >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dest_it += 2;
        } else {
            invalid_char:
            STRF_CHECK_DEST;
            *dest_it = 0xFFFD;
            ++dest_it;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::size_t count = 0;
    const SrcCharT* src_it = src;
    const auto src_end = src + src_size;
    for ( ; src_it != src_end; ++src_it) {
        auto ch = *src_it;
        count += 1 + (0x10000 <= ch && ch < 0x110000);
    }
    return count;
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf16>::write_replacement_char
    ( strf::basic_outbuff<CharT>& ob )
{
    ob.ensure(1);
    *ob.pointer() = 0xFFFD;
    ob.advance();
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    const auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    if (surr_poli == strf::surrogate_policy::lax) {
        for (auto src_it = src; src_it < src_end; ++src_it) {
            auto ch = *src_it;
            STRF_IF_UNLIKELY (ch >= 0x110000) {
                ch = 0xFFFD;
                if (inv_seq_notifier) {
                    ob.advance_to(dest_it);
                    inv_seq_notifier.notify();
                }
            }
            STRF_CHECK_DEST;
            *dest_it = ch;
            ++dest_it;
        }
    } else {
        for(auto src_it = src; src_it < src_end; ++src_it) {
            char32_t ch = *src_it;
            STRF_IF_UNLIKELY (ch >= 0x110000 || strf::detail::is_surrogate(ch)) {
                ch = 0xFFFD;
                if (inv_seq_notifier) {
                    ob.advance_to(dest_it);
                    inv_seq_notifier.notify();
                }
            }
            STRF_CHECK_DEST;
            *dest_it = ch;
            ++dest_it;
        }
    }
    ob.advance_to(dest_it);
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf32>::encode_fill
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, char32_t ch )
{
    strf::detail::write_fill(ob, count, static_cast<CharT>(ch));
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf32>::write_replacement_char
    ( strf::basic_outbuff<CharT>& ob )
{
    ob.ensure(1);
    *ob.pointer() = 0xFFFD;
    ob.advance();
}


template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;
    using strf::detail::utf8_decode_last_2_of_4;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1, ch2, ch3;
    unsigned long x;
    auto src_it = src;
    const auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();

    for (;src_it != src_end; ++dest_it) {
        ch0 = (*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch0 < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = ch0;
        } else if (0xC0 == (ch0 & 0xE0)) {
            STRF_IF_LIKELY ( ch0 > 0xC1
                          && src_it != src_end
                          && is_utf8_continuation(ch1 = * src_it))
            {
                STRF_CHECK_DEST;
                *dest_it = utf8_decode(ch0, ch1);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            STRF_IF_LIKELY ( src_it != src_end
                          && (((ch1 = * src_it) & 0xE0) == 0xA0)
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = * src_it) )
            {
                STRF_CHECK_DEST;
                *dest_it = ((ch1 & 0x3F) << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            STRF_IF_LIKELY (( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                          && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                                  , surr_poli )
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = * src_it) ))
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>((x << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
        {
            STRF_CHECK_DEST_SIZE(2);
            x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 +  (x >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 +  (x & 0x3FF));
            ++dest_it;
            ++src_it;
        } else {
            invalid_sequence:
            STRF_CHECK_DEST;
            *dest_it = 0xFFFD;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::size_t size = 0;
    std::uint8_t ch0, ch1;
    auto src_it = src;
    const auto src_end = src + src_size;
    while(src_it < src_end) {
        ch0 = *src_it;
        ++src_it;
        ++size;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it)) {
                ++src_it;
            }
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( ch0, ch1, surr_poli )
              && ++src_it != src_end && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src_it != src_end && is_utf8_continuation(*src_it)
                 && ++src_it != src_end && is_utf8_continuation(*src_it) )
        {
            ++src_it;
            ++size;
        }
    }
    return size;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    (void) inv_seq_notifier;
    auto src_it = src;
    const auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();

    for( ; src_it < src_end; ++src_it) {
        auto ch = *src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | ((ch & 0x7C0) >> 6));
            dest_it[1] = static_cast<DestCharT>(0x80 |  (ch &  0x3F));
            dest_it += 2;
        } else if (strf::detail::not_surrogate(ch)) {
            three_bytes:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>(0xE0 | ((ch & 0xF000) >> 12));
            dest_it[1] = static_cast<DestCharT>(0x80 | ((ch &  0xFC0) >> 6));
            dest_it[2] = static_cast<DestCharT>(0x80 |  (ch &   0x3F));
            dest_it += 3;
        } else if ( strf::detail::is_high_surrogate(ch)
               && (src_it + 1) != src_end
               && strf::detail::is_low_surrogate(*(src_it + 1)))
        {
            STRF_CHECK_DEST_SIZE(4);
            unsigned long ch2 = *++src_it;
            unsigned long codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | ((codepoint & 0x1C0000) >> 18));
            dest_it[1] = static_cast<DestCharT>(0x80 | ((codepoint &  0x3F000) >> 12));
            dest_it[2] = static_cast<DestCharT>(0x80 | ((codepoint &    0xFC0) >> 6));
            dest_it[3] = static_cast<DestCharT>(0x80 |  (codepoint &     0x3F));
            dest_it += 4;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            goto three_bytes;
        } else { // invalid sequece
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::size_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    const auto src_end = src + src_size;
    std::size_t size = 0;
    for(auto it = src; it < src_end; ++it) {
        SrcCharT ch = *it;
        STRF_IF_LIKELY (ch < 0x80) {
            ++size;
        } else if (ch < 0x800) {
            size += 2;
        } else if ( strf::detail::is_high_surrogate(ch)
               && it + 1 != src_end
               && strf::detail::is_low_surrogate(*(it + 1)) )
        {
            size += 4;
            ++it;
        } else {
            size += 3;
        }
    }
    return size;
}

template <typename CharT>
using utf8_t = strf::static_charset<CharT, strf::csid_utf8>;

template <typename CharT>
using utf16_t = strf::static_charset<CharT, strf::csid_utf16>;

template <typename CharT>
using utf32_t = strf::static_charset<CharT, strf::csid_utf32>;

template <typename CharT>
using utf_t = strf::static_charset<CharT, strf::csid_utf<CharT>>;

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename CharT>
STRF_DEVICE constexpr utf8_t<CharT> utf8 {};

template <typename CharT>
STRF_DEVICE constexpr utf16_t<CharT> utf16 {};

template <typename CharT>
STRF_DEVICE constexpr utf32_t<CharT> utf32 {};

template <typename CharT>
STRF_DEVICE constexpr utf_t<CharT> utf {};

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)

} // namespace strf

#endif  // STRF_DETAIL_UTF_HPP

