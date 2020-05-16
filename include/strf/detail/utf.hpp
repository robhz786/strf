#ifndef STRF_DETAIL_UTF_HPP
#define STRF_DETAIL_UTF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/standard_lib_functions.hpp>
#include <cstdint>
#include <cstddef> // for std::size_t

namespace strf {

#if ! defined(STRF_CHECK_DEST)

#define STRF_CHECK_DEST     \
    if (dest_it == dest_end) {         \
        ob.advance_to(dest_it);        \
        ob.recycle();                  \
        if (!ob.good()) {              \
            return;                    \
        }                              \
        dest_it = ob.pointer();        \
        dest_end = ob.end();           \
    }

#define STRF_CHECK_DEST_SIZE(SIZE)     \
    if (dest_it + SIZE > dest_end) {              \
        ob.advance_to(dest_it);                   \
        ob.recycle();                             \
        if (!ob.good()) {                         \
            return;                               \
        }                                         \
        dest_it = ob.pointer();                   \
        dest_end = ob.end();                      \
    }

#endif // ! defined(STRF_CHECK_DEST)

namespace detail {

template <typename T, std::size_t N>
struct simple_array;

template <typename T> struct simple_array<T,1> { T obj0; };
template <typename T> struct simple_array<T,2> { T obj0;  T obj1; };
template <typename T> struct simple_array<T,3> { T obj0;  T obj1; T obj2; };
template <typename T> struct simple_array<T,4> { T obj0;  T obj1; T obj2; T obj3; };

template <typename CharT, std::size_t N>
inline STRF_HD void do_repeat_sequence
    ( CharT* dest
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    auto* reinterpreted_dest = reinterpret_cast<simple_array<CharT, N>*>(dest);
    strf::detail::str_fill_n(reinterpreted_dest, count, seq);
}

template <typename CharT, std::size_t N>
STRF_HD void repeat_sequence_continuation
    ( strf::underlying_outbuf<sizeof(CharT)>& ob
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    std::size_t space = ob.size() / N;
    STRF_ASSERT(space < count);

    strf::detail::do_repeat_sequence(ob.pointer(), space, seq);
    count -= space;
    ob.advance(space * N);
    ob.recycle();
    while (true) { //ob.good())
        space = ob.size() / N;
        if (count <= space) {
            strf::detail::do_repeat_sequence(ob.pointer(), count, seq);
            ob.advance(count * N);
            return;
        }
        strf::detail::do_repeat_sequence(ob.pointer(), space, seq);
        count -= space;
        ob.advance(space * N);
        ob.recycle();
    }
}

template <typename CharT, std::size_t N>
inline STRF_HD void repeat_sequence
    ( strf::underlying_outbuf<sizeof(CharT)>& ob
    , std::size_t count
    , simple_array<CharT, N> seq )
{
    if (count * N <= ob.size()) {
        strf::detail::do_repeat_sequence(ob.pointer(), count, seq);
        ob.advance(count * N);
    } else {
        strf::detail::repeat_sequence_continuation(ob, count, seq);
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

inline STRF_HD bool first_2_of_3_are_valid( unsigned x
                                  , strf::surrogate_policy surr_poli )
{
    return ( surr_poli == strf::surrogate_policy::lax
          || (x >> 5) != 0x1B );
}
inline STRF_HD bool first_2_of_3_are_valid( std::uint8_t ch0
                                  , std::uint8_t ch1
                                  , strf::surrogate_policy surr_poli )
{
    return first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), surr_poli);
}

inline STRF_HD unsigned utf8_decode_first_2_of_4(std::uint8_t ch0, std::uint8_t ch1)
{
    return ((ch0 & 0x07) << 6) | (ch1 & 0x3F);
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

template <>
class static_underlying_transcoder<strf::char_encoding_id::utf8, strf::char_encoding_id::utf8>
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<1>& ob
        , const std::uint8_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const std::uint8_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<1, 1> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<1> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf8_to_utf16
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf8, strf::char_encoding_id::utf16 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<2>& ob
        , const std::uint8_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const std::uint8_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<1, 2> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<1> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf8, strf::char_encoding_id::utf32 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<4>& ob
        , const std::uint8_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const std::uint8_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<1, 4> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<1> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf16_to_utf8
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf8 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<1>& ob
        , const char16_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char16_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<2, 1> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<2> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf16_to_utf16
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf16 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<2>& ob
        , const char16_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char16_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<2, 2> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<2> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf16_to_utf32
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf32 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<4>& ob
        , const char16_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char16_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<2, 4> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<2> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf32_to_utf8
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf8 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<1>& ob
        , const char32_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char32_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<4, 1> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<4> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf32_to_utf16
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf16 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<2>& ob
        , const char32_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char32_t* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli );

    static STRF_HD strf::transcode_f<4, 2> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<4> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

//struct utf32_to_utf32
template <>
class static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf32 >
{
public:
    static STRF_HD void transcode
        ( strf::underlying_outbuf<4>& ob
        , const char32_t* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::size_t transcode_size
        ( const char32_t* src
        , std::size_t src_size
        , strf::surrogate_policy )
    {
        (void) src;
        return src_size;
    }
    static STRF_HD strf::transcode_f<4, 4> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<4> transcode_size_func() noexcept
    {
        return transcode_size;
    }
};

using utf8_to_utf8 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf8, strf::char_encoding_id::utf8 >;
using utf8_to_utf16 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf8, strf::char_encoding_id::utf16 >;
using utf8_to_utf32 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf8, strf::char_encoding_id::utf32 >;

using utf16_to_utf8 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf8 >;
using utf16_to_utf16 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf16 >;
using utf16_to_utf32 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf16, strf::char_encoding_id::utf32 >;

using utf32_to_utf8 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf8 >;
using utf32_to_utf16 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf16 >;
using utf32_to_utf32 = strf::static_underlying_transcoder
    < strf::char_encoding_id::utf32, strf::char_encoding_id::utf32 >;


template <>
class static_underlying_char_encoding<strf::char_encoding_id::utf8>
{
    using char_type_ = std::uint8_t;
public:
    static constexpr std::size_t char_size = 1;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-8";
    };
    static constexpr STRF_HD strf::char_encoding_id id() noexcept
    {
        return strf::char_encoding_id::utf8;
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
    static STRF_HD char_type_* encode_char
        ( char_type_* dest, char32_t ch ) noexcept;

    static STRF_HD void encode_fill
        ( strf::underlying_outbuf<char_size>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count ) noexcept;

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::underlying_outbuf<char_size>& );

    static STRF_HD char32_t decode_char(char_type_ ch) noexcept
    {
        if (ch < 0x80)
            return static_cast<char32_t>(ch);
        return 0xFFFD;
    }
    static STRF_HD strf::encode_char_f<char_size> encode_char_func() noexcept
    {
        return encode_char;
    }
    static STRF_HD strf::encode_fill_f<char_size> encode_fill_func() noexcept
    {
        return encode_fill;
    }
    static STRF_HD strf::write_replacement_char_f<char_size>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf32_to_utf8 from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf8_to_utf32 to_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf8_to_utf8 sanitizer() noexcept
    {
        return {};
    }
    static STRF_HD strf::dynamic_underlying_transcoder<2, 1>
    find_transcoder_from
        ( std::integral_constant<std::size_t, 2>
        , strf::char_encoding_id id ) noexcept
    {
        return find_transcoder_from_2bytes_encoding(id);
    }
    static STRF_HD strf::dynamic_underlying_transcoder<2, 1>
    find_transcoder_from_2bytes_encoding(strf::char_encoding_id id) noexcept
    {
        if (id == strf::char_encoding_id::utf16) {
            strf::dynamic_underlying_transcoder<2, 1>{strf::utf16_to_utf8{}};
        }
        return {};
    }
    static STRF_HD strf::dynamic_underlying_transcoder<1, 2>
    find_transcoder_to
        ( std::integral_constant<std::size_t, 2>
        , strf::char_encoding_id id) noexcept
    {
        return find_transcoder_to_2bytes_encoding(id);
    }
    static STRF_HD strf::dynamic_underlying_transcoder<1, 2>
    find_transcoder_to_2bytes_encoding(strf::char_encoding_id id) noexcept
    {
        if (id == strf::char_encoding_id::utf16) {
            strf::dynamic_underlying_transcoder<1, 2>{strf::utf8_to_utf16{}};
        }
        return {};
    }
    static strf::dynamic_underlying_char_encoding<1> to_dynamic() noexcept
    {
        static const strf::dynamic_underlying_char_encoding_data<1> data = {
            name(), id(), replacement_char(), 3, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_char,
            strf::dynamic_underlying_transcoder<4, 1>{from_u32()},
            strf::dynamic_underlying_transcoder<1, 4>{to_u32()},
            strf::dynamic_underlying_transcoder<1, 1>{sanitizer()},
            strf::find_transcoder_f<1, 1>{nullptr},
            find_transcoder_from_2bytes_encoding,
            strf::find_transcoder_f<1, 1>{nullptr},
            find_transcoder_to_2bytes_encoding
        };
        return strf::dynamic_underlying_char_encoding<1>{data};
    }
    explicit operator strf::dynamic_underlying_char_encoding<1> () const
    {
        return to_dynamic();
    }
};

template <>
class static_underlying_char_encoding<strf::char_encoding_id::utf16>
{
    using char_type_ = char16_t;

public:

    static constexpr std::size_t char_size = 2;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-16";
    };
    static constexpr STRF_HD strf::char_encoding_id id() noexcept
    {
        return strf::char_encoding_id::utf16;
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

    static STRF_HD char_type_* encode_char
        (char_type_* dest, char32_t ch) noexcept;

    static STRF_HD void encode_fill
        ( strf::underlying_outbuf<char_size>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count ) noexcept;

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::underlying_outbuf<char_size>& );

    static STRF_HD char32_t decode_char(char_type_ ch) noexcept
    {
        return ch;
    }
    static STRF_HD strf::encode_char_f<char_size> encode_char_func() noexcept
    {
        return encode_char;
    }
    static STRF_HD strf::encode_fill_f<char_size> encode_fill_func() noexcept
    {
        return encode_fill;
    }
    static STRF_HD strf::write_replacement_char_f<char_size>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf32_to_utf16 from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf16_to_utf32 to_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf16_to_utf16 sanitizer() noexcept
    {
        return {};
    }
    static STRF_HD strf::dynamic_underlying_transcoder<1, 2>
    find_transcoder_from
        ( std::integral_constant<std::size_t, 1>
        , strf::char_encoding_id id ) noexcept
    {
        return find_transcoder_from_1byte_encoding(id);
    }
    static STRF_HD strf::dynamic_underlying_transcoder<2, 1>
    find_transcoder_to
        ( std::integral_constant<std::size_t, 1>
        , strf::char_encoding_id id) noexcept
    {
        return find_transcoder_to_1byte_encoding(id);
    }
    static STRF_HD strf::dynamic_underlying_transcoder<1, 2>
    find_transcoder_from_1byte_encoding(strf::char_encoding_id id) noexcept
    {
        if (id == strf::char_encoding_id::utf8) {
            return strf::dynamic_underlying_transcoder<1, 2>{strf::utf8_to_utf16{}};
        }
        return {};
    }
    static STRF_HD strf::dynamic_underlying_transcoder<2, 1>
    find_transcoder_to_1byte_encoding(strf::char_encoding_id id) noexcept
    {
        if (id == strf::char_encoding_id::utf8) {
            return strf::dynamic_underlying_transcoder<2, 1>{strf::utf16_to_utf8{}};
        }
        return {};
    }
    static strf::dynamic_underlying_char_encoding<2> to_dynamic() noexcept
    {
        static const strf::dynamic_underlying_char_encoding_data<2> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_char,
            strf::dynamic_underlying_transcoder<4, 2>{from_u32()},
            strf::dynamic_underlying_transcoder<2, 4>{to_u32()},
            strf::dynamic_underlying_transcoder<2, 2>{sanitizer()},
            find_transcoder_from_1byte_encoding,
            strf::find_transcoder_f<2, 2>{nullptr},
            find_transcoder_to_1byte_encoding,
            strf::find_transcoder_f<2, 2>{nullptr}
        };
        return strf::dynamic_underlying_char_encoding<2>{data};
    }
    explicit operator strf::dynamic_underlying_char_encoding<2> () const
    {
        return to_dynamic();
    }
};

template <>
class static_underlying_char_encoding<strf::char_encoding_id::utf32>
{
    using char_type_ = char32_t;
public:
    static constexpr std::size_t char_size = 4;

    static STRF_HD const char* name() noexcept
    {
        return "UTF-32";
    };
    static constexpr STRF_HD strf::char_encoding_id id() noexcept
    {
        return strf::char_encoding_id::utf32;
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
    static STRF_HD char_type_* encode_char
        (char_type_* dest, char32_t ch) noexcept
    {
        *dest = ch;
        return dest + 1;
    }
    static STRF_HD void encode_fill
        ( strf::underlying_outbuf<char_size>&, std::size_t count, char32_t ch );

    static STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count ) noexcept
    {
        (void) src;
        if (max_count <= src_size) {
            return {max_count, max_count};
        }
        return {src_size, src_size};
    }

    static STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const char_type_* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) noexcept
    {
        (void)surr_poli;
        return codepoints_fast_count(src, src_size, max_count);
    }

    static STRF_HD void write_replacement_char
        ( strf::underlying_outbuf<char_size>& );

    static STRF_HD char32_t decode_char(char_type_ ch) noexcept
    {
        return ch;
    }
    static STRF_HD strf::encode_char_f<char_size> encode_char_func() noexcept
    {
        return encode_char;
    }
    static STRF_HD strf::encode_fill_f<char_size> encode_fill_func() noexcept
    {
        return encode_fill;
    }
    static STRF_HD strf::write_replacement_char_f<char_size>
    write_replacement_char_func() noexcept
    {
        return write_replacement_char;
    }
    static constexpr STRF_HD strf::utf32_to_utf32 from_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf32 to_u32() noexcept
    {
        return {};
    }
    static constexpr STRF_HD strf::utf32_to_utf32 sanitizer() noexcept
    {
        return {};
    }
    static strf::dynamic_underlying_char_encoding<4> to_dynamic() noexcept
    {
        static const strf::dynamic_underlying_char_encoding_data<4> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, codepoints_fast_count,
            codepoints_robust_count, write_replacement_char, decode_char,
            strf::dynamic_underlying_transcoder<4, 4>{from_u32()},
            strf::dynamic_underlying_transcoder<4, 4>{to_u32()},
            strf::dynamic_underlying_transcoder<4, 4>{sanitizer()},
            nullptr, nullptr, nullptr, nullptr
        };
        return strf::dynamic_underlying_char_encoding<4>{data};
    }
    explicit operator strf::dynamic_underlying_char_encoding<4> () const
    {
        return to_dynamic();
    }
};

using utf8_impl = static_underlying_char_encoding<strf::char_encoding_id::utf8>;
using utf16_impl = static_underlying_char_encoding<strf::char_encoding_id::utf16>;
using utf32_impl = static_underlying_char_encoding<strf::char_encoding_id::utf32>;

#if ! defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD void utf8_to_utf32::transcode
    ( strf::underlying_outbuf<4>& ob
    , const std::uint8_t* src
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
    char32_t ch32;

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
        } else if (0xEF < ch0) {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
              && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                ch32 = utf8_decode_last_2_of_4(x, ch2, ch3);
                ++src_it;
            } else goto invalid_sequence;
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

STRF_INLINE STRF_HD std::size_t utf8_to_utf32::transcode_size
    ( const std::uint8_t* src
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
        } else if(0xEF < ch0) {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++src_it;
            }
        }
    }
    return size;
}

STRF_INLINE STRF_HD void utf8_to_utf8::transcode
    ( strf::underlying_outbuf<1>& ob
    , const std::uint8_t* src
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
        } else if (0xF0 == (ch0 & 0xF8)) {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
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
            } else goto invalid_sequence;
        } else {
            invalid_sequence:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = 0xEF;
            dest_it[1] = 0xBF;
            dest_it[2] = 0xBD;
            dest_it += 3;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

STRF_INLINE STRF_HD std::size_t utf8_to_utf8::transcode_size
    ( const std::uint8_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::uint8_t ch0, ch1;
    const std::uint8_t* src_it = src;
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
        } else if( 0xEF < ch0
              &&   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
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

STRF_INLINE STRF_HD strf::codepoints_count_result utf8_impl::codepoints_fast_count
    ( const std::uint8_t* src
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
    return {count, static_cast<std::size_t>(it - src)};
}

STRF_INLINE STRF_HD strf::codepoints_count_result utf8_impl::codepoints_robust_count
    ( const std::uint8_t* src
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
        } else if(0xEF < ch0) {
            if (   it != end && is_utf8_continuation(ch1 = * it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++it != end && is_utf8_continuation(*it)
              && ++it != end && is_utf8_continuation(*it) )
            {
                ++it;
            }
        }
    }
    return {count, static_cast<std::size_t>(it - src)};
}


STRF_INLINE STRF_HD void utf8_impl::encode_fill
    ( strf::underlying_outbuf<1>& ob, std::size_t count, char32_t ch )
{
    if (ch < 0x80) {
        strf::detail::write_fill(ob, count, static_cast<std::uint8_t>(ch));
    } else if (ch < 0x800) {
        strf::detail::simple_array<std::uint8_t, 2> seq = {
            static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6)),
            static_cast<std::uint8_t>(0x80 |  (ch &  0x3F))
        };
        strf::detail::repeat_sequence(ob, count, seq);
    } else if (ch <  0x10000) {
        strf::detail::simple_array<std::uint8_t, 3> seq = {
            static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12)),
            static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6)),
            static_cast<std::uint8_t>(0x80 |  (ch &   0x3F)),
        };
        strf::detail::repeat_sequence(ob, count, seq);
    } else if (ch < 0x110000) {
        strf::detail::simple_array<std::uint8_t, 4> seq = {
            static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18)),
            static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12)),
            static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6)),
            static_cast<std::uint8_t>(0x80 |  (ch &     0x3F))
        };
        strf::detail::repeat_sequence(ob, count, seq);
    } else {
        strf::detail::simple_array<std::uint8_t, 3> seq{ 0xEF, 0xBF, 0xBD };
        strf::detail::repeat_sequence(ob, count, seq);
    }
}

STRF_INLINE STRF_HD std::uint8_t* utf8_impl::encode_char
    ( std::uint8_t* dest
    , char32_t ch ) noexcept
{
    if (ch < 0x80) {
        *dest = static_cast<std::uint8_t>(ch);
        return dest + 1;
    }
    if (ch < 0x800) {
        dest[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
        dest[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
        return dest + 2;
    }
    if (ch <  0x10000) {
        dest[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
        dest[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
        dest[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
        return dest + 3;
    }
    if (ch < 0x110000) {
        dest[0] = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
        dest[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
        dest[2] = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
        dest[3] = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
        return dest + 4;
    }
    dest[0] = 0xEF;
    dest[1] = 0xBF;
    dest[2] = 0xBD;
    return dest + 3;
}


// STRF_INLINE STRF_HD void utf8_impl::encode_char
//     ( strf::underlying_outbuf<char_size>& ob
//     , char32_t ch )
// {
//     if (ch < 0x80) {
//         strf::put(ob, static_cast<std::uint8_t>(ch));
//     }
//     else if (ch < 0x800) {
//         ob.ensure(2);
//         auto dest = ob.pointer();
//         dest[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
//         dest[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
//         ob.advance_to(dest + 2);
//     }
//     else if (ch <  0x10000) {
//         ob.ensure(3);
//         auto dest = ob.pointer();
//         dest[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
//         dest[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
//         dest[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
//         ob.advance_to(dest + 3);
//     }
//     else if (ch < 0x110000) {
//         ob.ensure(4);
//         auto dest = ob.pointer();
//         dest[0] = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
//         dest[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
//         dest[2] = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
//         dest[3] = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
//         ob.advance_to(dest + 4);
//     } else {
//         ob.ensure(3);
//         auto dest = ob.pointer();
//         dest[0] = 0xEF;
//         dest[1] = 0xBF;
//         dest[2] = 0xBD;
//         ob.advance_to(dest + 3);
//     }
// }

STRF_INLINE STRF_HD void utf32_to_utf8::transcode
    ( strf::underlying_outbuf<1>& ob
    , const char32_t* src
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
        if(ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<std::uint8_t>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
            dest_it[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
            dest_it += 2;
        } else if (ch < 0x10000) {
            if ( surr_poli == strf::surrogate_policy::lax
              || strf::detail::not_surrogate(ch))
            {
                STRF_CHECK_DEST_SIZE(3);
                dest_it[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
                dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
                dest_it[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (ch < 0x110000) {
            STRF_CHECK_DEST_SIZE(4);
            dest_it[0] = static_cast<std::uint8_t>(0xF0 | ((ch & 0x1C0000) >> 18));
            dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0x3F000) >> 12));
            dest_it[2] = static_cast<std::uint8_t>(0x80 | ((ch &    0xFC0) >> 6));
            dest_it[3] = static_cast<std::uint8_t>(0x80 |  (ch &     0x3F));
            dest_it += 4;
        } else {
            invalid_sequence:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = 0xEF;
            dest_it[1] = 0xBF;
            dest_it[2] = 0xBD;
            dest_it += 3;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

STRF_INLINE STRF_HD std::size_t utf32_to_utf8::transcode_size
    ( const char32_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    auto src_it = src;
    auto src_end = src + src_size;
    std::size_t count = 0;
    for(;src_it != src_end; ++src_it) {
        auto ch = *src_it;
        if (ch < 0x110000) {
            count += 1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000);
        } else {
            count += 3;
        }
    }
    return count;
}

STRF_INLINE STRF_HD void utf8_impl::write_replacement_char
    ( strf::underlying_outbuf<1>& ob )
{
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    STRF_CHECK_DEST_SIZE(3);
    dest_it[0] = 0xEF;
    dest_it[1] = 0xBF;
    dest_it[2] = 0xBD;
    dest_it += 3;
    ob.advance_to(dest_it);
}

STRF_INLINE STRF_HD void utf16_to_utf32::transcode
    ( strf::underlying_outbuf<4>& ob
    , const char16_t* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    unsigned long ch, ch2;
    char32_t ch32;
    const char16_t* src_it_next;
    auto src_end = src + src_size;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for(auto src_it = src; src_it != src_end; src_it = src_it_next) {
        src_it_next = src_it + 1;
        ch = *src_it;
        src_it_next = src_it + 1;

        if (strf::detail::not_surrogate(ch)) {
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

STRF_INLINE STRF_HD std::size_t utf16_to_utf32::transcode_size
    ( const char16_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    unsigned long ch;
    std::size_t count = 0;
    auto src_it = src;
    const auto src_end = src + src_size;
    const char16_t* src_it_next;
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

STRF_INLINE STRF_HD void utf16_to_utf16::transcode
    ( strf::underlying_outbuf<2>& ob
    , const char16_t* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    unsigned long ch, ch2;
    auto src_it = src;
    const auto src_end = src + src_size;
    const char16_t* src_it_next;
    auto dest_it = ob.pointer();
    auto dest_end = ob.end();
    for( ; src_it != src_end; src_it = src_it_next) {
        ch = *src_it;
        src_it_next = src_it + 1;

        if (strf::detail::not_surrogate(ch)) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<char16_t>(ch);
            ++dest_it;
        } else if ( strf::detail::is_high_surrogate(ch)
                 && src_it_next != src_end
                 && strf::detail::is_low_surrogate(ch2 = *src_it_next))
        {
            ++src_it_next;
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<char16_t>(ch);
            dest_it[1] = static_cast<char16_t>(ch2);
            dest_it += 2;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<char16_t>(ch);
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

STRF_INLINE STRF_HD std::size_t utf16_to_utf16::transcode_size
    ( const char16_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::size_t count = 0;
    const char16_t* src_it = src;
    const auto src_end = src + src_size;
    unsigned long ch;
    while (src_it != src_end) {
        ch = *src_it;
        ++ src_it;
        ++ count;
        if ( strf::detail::is_high_surrogate(ch)
          && src_it != src_end
          && strf::detail::is_low_surrogate(*src_it))
        {
            ++ src_it;
            ++ count;
        }
    }
    return count;
}

STRF_INLINE STRF_HD strf::codepoints_count_result utf16_impl::codepoints_fast_count
    ( const char16_t* src
    , std::size_t src_size
    , std::size_t max_count ) noexcept
{
    std::size_t count = 0;
    auto it = src;
    const auto end = src + src_size;
    while(it != end && count < max_count) {
        if(strf::detail::is_high_surrogate(*it)) {
            ++it;
        }
        ++it;
        ++count;
    }
    return {count, static_cast<std::size_t>(it - src)};
}

STRF_INLINE STRF_HD strf::codepoints_count_result utf16_impl::codepoints_robust_count
    ( const char16_t* src
    , std::size_t src_size
    , std::size_t max_count
    , strf::surrogate_policy surr_poli ) noexcept
{
    (void) surr_poli;
    std::size_t count = 0;
    const char16_t* it = src;
    const auto end = src + src_size;
    unsigned long ch;
    while (it != end && count < max_count) {
        ch = *it;
        ++ it;
        ++ count;
        if ( strf::detail::is_high_surrogate(ch) && it != end
          && strf::detail::is_low_surrogate(*it)) {
            ++ it;
            ++ count;
        }
    }
    return {count, static_cast<std::size_t>(it - src)};
}

STRF_INLINE STRF_HD char16_t* utf16_impl::encode_char
    ( char16_t* dest
    , char32_t ch ) noexcept
{
    if (ch < 0x10000) {
        *dest = static_cast<char16_t>(ch);
        return dest + 1;
    }
    if (ch < 0x110000) {
        char32_t sub_codepoint = ch - 0x10000;
        dest[0] = static_cast<char16_t>(0xD800 + (sub_codepoint >> 10));
        dest[1] = static_cast<char16_t>(0xDC00 + (sub_codepoint &  0x3FF));
        return dest + 2;
    }
    *dest = 0xFFFD;
    return dest + 1;
}


// STRF_INLINE STRF_HD void utf16_impl::encode_char
//     ( strf::underlying_outbuf<char_size>& ob
//     , char32_t ch )
// {
//     if (ch < 0x10000) {
//         strf::put(ob, static_cast<char16_t>(ch));
//     }
//     else if (ch < 0x110000) {
//         ob.ensure(2);
//         auto dest = ob.pointer();
//         char32_t sub_codepoint = ch - 0x10000;
//         dest[0] = static_cast<char16_t>(0xD800 + (sub_codepoint >> 10));
//         dest[1] = static_cast<char16_t>(0xDC00 + (sub_codepoint &  0x3FF));
//         ob.advance_to(dest + 2);
//     }
//     else {
//         strf::put(ob, u'\uFFFD');
//     }
// }

STRF_INLINE STRF_HD void utf16_impl::encode_fill
    ( strf::underlying_outbuf<2>& ob, std::size_t count, char32_t ch )
{
    if (ch < 0x10000) {
        strf::detail::write_fill(ob, count, static_cast<char16_t>(ch));
    } else if (ch < 0x110000) {
        char32_t sub_codepoint = ch - 0x10000;
        strf::detail::simple_array<char16_t, 2> seq = {
            static_cast<char16_t>(0xD800 + (sub_codepoint >> 10)),
            static_cast<char16_t>(0xDC00 + (sub_codepoint &  0x3FF))
        };
        strf::detail::repeat_sequence(ob, count, seq);
    } else {
        strf::detail::write_fill(ob, count, u'\uFFFD');
    }
}

STRF_INLINE STRF_HD void utf32_to_utf16::transcode
    ( strf::underlying_outbuf<2>& ob
    , const char32_t* src
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
        if (ch < 0x10000) {
            if ( surr_poli == strf::surrogate_policy::lax
              || strf::detail::not_surrogate(ch) )
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<char16_t>(ch);
                ++dest_it;
            } else goto invalid_char;
        } else if (ch < 0x110000) {
            STRF_CHECK_DEST_SIZE(2);
            char32_t sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<char16_t>(0xD800 | (sub_codepoint >> 10));
            dest_it[1] = static_cast<char16_t>(0xDC00 | (sub_codepoint &  0x3FF));
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

STRF_INLINE STRF_HD std::size_t utf32_to_utf16::transcode_size
    ( const char32_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::size_t count = 0;
    const char32_t* src_it = src;
    const auto src_end = src + src_size;
    for ( ; src_it != src_end; ++src_it) {
        auto ch = *src_it;
        count += 1 + (0x10000 <= ch && ch < 0x110000);
    }
    return count;
}

STRF_INLINE STRF_HD void utf16_impl::write_replacement_char
    ( strf::underlying_outbuf<2>& ob )
{
    ob.ensure(1);
    *ob.pointer() = 0xFFFD;
    ob.advance();
}

STRF_INLINE STRF_HD void utf32_to_utf32::transcode
    ( strf::underlying_outbuf<4>& ob
    , const char32_t* src
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
            if (ch >= 0x110000) {
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
            if (ch >= 0x110000 || strf::detail::is_surrogate(ch)) {
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

STRF_INLINE STRF_HD void utf32_impl::encode_fill
    ( strf::underlying_outbuf<4>& ob, std::size_t count, char32_t ch )
{
    strf::detail::write_fill(ob, count, ch);
}

STRF_INLINE STRF_HD void utf32_impl::write_replacement_char
    ( strf::underlying_outbuf<4>& ob )
{
    ob.ensure(1);
    *ob.pointer() = 0xFFFD;
    ob.advance();
}


STRF_INLINE STRF_HD void utf8_to_utf16::transcode
    ( strf::underlying_outbuf<2>& ob
    , const std::uint8_t* src
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
        if (ch0 < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = ch0;
        } else if (0xC0 == (ch0 & 0xE0)) {
            if ( ch0 > 0xC1
              && src_it != src_end && is_utf8_continuation(ch1 = * src_it))
            {
                STRF_CHECK_DEST;
                *dest_it = utf8_decode(ch0, ch1);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end && (((ch1 = * src_it) & 0xE0) == 0xA0)
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                STRF_CHECK_DEST;
                *dest_it = ((ch1 & 0x3F) << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                       , surr_poli )
              && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it) )
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<char16_t>((x << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xEF < ch0) {
            if ( src_it != src_end && is_utf8_continuation(ch1 = * src_it)
                 && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                 && ++src_it != src_end && is_utf8_continuation(ch2 = * src_it)
                 && ++src_it != src_end && is_utf8_continuation(ch3 = * src_it) )
            {
                STRF_CHECK_DEST_SIZE(2);
                x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
                dest_it[0] = static_cast<char16_t>(0xD800 +  (x >> 10));
                dest_it[1] = static_cast<char16_t>(0xDC00 +  (x & 0x3FF));
                ++dest_it;
                ++src_it;
            } else goto invalid_sequence;
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

STRF_INLINE STRF_HD std::size_t utf8_to_utf16::transcode_size
    ( const std::uint8_t* src
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
        } else if(0xEF < ch0) {
            if (   src_it != src_end && is_utf8_continuation(ch1 = * src_it)
              && first_2_of_4_are_valid(ch0, ch1)
              && ++src_it != src_end && is_utf8_continuation(*src_it)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++src_it;
                ++size;
            }
        }

    }
    return size;
}

STRF_INLINE STRF_HD void utf16_to_utf8::transcode
    ( strf::underlying_outbuf<1>& ob
    , const char16_t* src
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
        if (ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<std::uint8_t>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<std::uint8_t>(0xC0 | ((ch & 0x7C0) >> 6));
            dest_it[1] = static_cast<std::uint8_t>(0x80 |  (ch &  0x3F));
            dest_it += 2;
        } else if (strf::detail::not_surrogate(ch)) {
            three_bytes:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<std::uint8_t>(0xE0 | ((ch & 0xF000) >> 12));
            dest_it[1] = static_cast<std::uint8_t>(0x80 | ((ch &  0xFC0) >> 6));
            dest_it[2] = static_cast<std::uint8_t>(0x80 |  (ch &   0x3F));
            dest_it += 3;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src_it != src_end
               && strf::detail::is_low_surrogate(*(src_it + 1)))
        {
            STRF_CHECK_DEST_SIZE(4);
            unsigned long ch2 = *++src_it;
            unsigned long codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<std::uint8_t>(0xF0 | ((codepoint & 0x1C0000) >> 18));
            dest_it[1] = static_cast<std::uint8_t>(0x80 | ((codepoint &  0x3F000) >> 12));
            dest_it[2] = static_cast<std::uint8_t>(0x80 | ((codepoint &    0xFC0) >> 6));
            dest_it[3] = static_cast<std::uint8_t>(0x80 |  (codepoint &     0x3F));
            dest_it += 4;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            goto three_bytes;
        } else { // invalid sequece
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = 0xEF;
            dest_it[1] = 0xBF;
            dest_it[2] = 0xBD;
            dest_it += 3;
            if (inv_seq_notifier) {
                ob.advance_to(dest_it);
                inv_seq_notifier.notify();
            }
        }
    }
    ob.advance_to(dest_it);
}

STRF_INLINE STRF_HD std::size_t utf16_to_utf8::transcode_size
    ( const char16_t* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    const auto src_end = src + src_size;
    std::size_t size = 0;
    for(auto it = src; it < src_end; ++it) {
        char16_t ch = *it;
        if (ch < 0x80) {
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

#endif // ! defined(STRF_OMIT_IMPL)

template <typename CharT>
using utf8 = strf::static_char_encoding<CharT, strf::char_encoding_id::utf8>;

template <typename CharT>
using utf16 = strf::static_char_encoding<CharT, strf::char_encoding_id::utf16>;

template <typename CharT>
using utf32 = strf::static_char_encoding<CharT, strf::char_encoding_id::utf32>;

namespace detail {

template <std::size_t CharSize>
struct mp_utf_encoding;

template <>
struct mp_utf_encoding<1>
{
    template <typename CharT>
    using fn = strf::utf8<CharT>;
};
template <>
struct mp_utf_encoding<2>
{
    template <typename CharT>
    using fn = strf::utf16<CharT>;
};
template <>
struct mp_utf_encoding<4>
{
    template <typename CharT>
    using fn = strf::utf32<CharT>;
};

} // namespace detail

template <typename CharT>
using utf = typename strf::detail::mp_utf_encoding<sizeof(CharT)>
    :: template fn<CharT>;

// using utfw = strf::utf<wchar_t>;
// using wchar_encoding = strf::utf<wchar_t>;

} // namespace strf

#endif  // STRF_DETAIL_UTF_HPP

