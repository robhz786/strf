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

#define STRF_RECYCLE                         \
    dest.advance_to(dest_it);               \
    dest.flush();                           \
    STRF_IF_UNLIKELY (!dest.good()) {       \
        return;                             \
    }                                       \
    dest_it = dest.buffer_ptr();            \
    dest_end = dest.buffer_end();           \

#define STRF_CHECK_DEST                         \
    STRF_IF_UNLIKELY (dest_it >= dest_end) {    \
        STRF_RECYCLE;                           \
    }

#define STRF_CHECK_DEST_SIZE(SIZE)                      \
    STRF_IF_UNLIKELY (dest_it + (SIZE) > dest_end) {    \
        STRF_RECYCLE;                                   \
    }

#endif // ! defined(STRF_CHECK_DEST)

namespace detail {

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dest<CharT>& dest
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1 ) noexcept
{
    auto *p = dest.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 2;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dest.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p += seq_size;
        }
        dest.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dest.flush();
        STRF_IF_UNLIKELY (!dest.good()) {
            return;
        }
        p = dest.buffer_ptr();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dest<CharT>& dest
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2 ) noexcept
{
    auto *p = dest.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 3;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dest.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p += seq_size;
        }
        dest.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dest.flush();
        STRF_IF_UNLIKELY (!dest.good()) {
            return;
        }
        p = dest.buffer_ptr();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dest<CharT>& dest
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2
    , CharT ch3 ) noexcept
{
    auto *p = dest.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 4;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dest.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p[3] = ch3;
            p += seq_size;
        }
        dest.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dest.flush();
        STRF_IF_UNLIKELY (!dest.good()) {
            return;
        }
        p = dest.buffer_ptr();
        count -= space;
    }
}

constexpr STRF_HD bool is_surrogate(unsigned codepoint)
{
    return codepoint >> 11 == 0x1B;
}
constexpr STRF_HD bool is_high_surrogate(unsigned codepoint) noexcept
{
    return codepoint >> 10 == 0x36;
}
constexpr STRF_HD bool is_low_surrogate(unsigned codepoint) noexcept
{
    return codepoint >> 10 == 0x37;
}
constexpr STRF_HD bool not_surrogate(unsigned codepoint)
{
    return codepoint >> 11 != 0x1B;
}
constexpr STRF_HD bool not_high_surrogate(unsigned codepoint)
{
    return codepoint >> 10 != 0x36;
}
constexpr STRF_HD bool not_low_surrogate(unsigned codepoint)
{
    return codepoint >> 10 != 0x37;
}
constexpr STRF_HD unsigned utf8_decode(unsigned ch0, unsigned ch1)
{
    return (((ch0 & 0x1F) << 6) |
            ((ch1 & 0x3F) << 0));
}
constexpr STRF_HD unsigned utf8_decode(unsigned ch0, unsigned ch1, unsigned ch2)
{
    return (((ch0 & 0x0F) << 12) |
            ((ch1 & 0x3F) <<  6) |
            ((ch2 & 0x3F) <<  0));
}
constexpr STRF_HD unsigned utf8_decode(unsigned ch0, unsigned ch1, unsigned ch2, unsigned ch3)
{
    return (((ch0 & 0x07) << 18) |
            ((ch1 & 0x3F) << 12) |
            ((ch2 & 0x3F) <<  6) |
            ((ch3 & 0x3F) <<  0));
}

template <typename IntT, strf::detail::enable_if_t<std::is_same<unsigned, IntT>::value, int> =0>
constexpr STRF_HD bool is_utf8_continuation(IntT ch)
{
    return (ch & 0xC0) == 0x80;
}

template <typename IntT, strf::detail::enable_if_t<sizeof(IntT) == 1, int> =0>
constexpr STRF_HD bool is_utf8_continuation(IntT ch)
{
    return is_utf8_continuation(static_cast<unsigned>(static_cast<unsigned char>(ch)));
}

template <typename IntT, strf::detail::enable_if_t<std::is_same<unsigned, IntT>::value, int> =0>
constexpr STRF_HD bool not_utf8_continuation(IntT ch)
{
    return (ch & 0xC0) != 0x80;
}

template <typename IntT, strf::detail::enable_if_t<sizeof(IntT) == 1, int> =0>
constexpr STRF_HD bool not_utf8_continuation(IntT ch)
{
    return not_utf8_continuation(static_cast<unsigned>(static_cast<unsigned char>(ch)));
}

// constexpr STRF_HD bool valid_start_3bytes
//     ( std::uint8_t ch0
//     , std::uint8_t ch1
//     , strf::surrogate_policy surr_poli )
// {
//     return ( (ch0 != 0xE0 || ch1 != 0x80)
//           && ( surr_poli == strf::surrogate_policy::lax
//             || (0x1B != (((ch0 & 0xF) << 1) | ((ch1 >> 5) & 1)))) );
// }

inline STRF_HD unsigned utf8_decode_first_2_of_3(unsigned ch0, unsigned ch1)
{
    return static_cast<unsigned>(((ch0 & 0x0F) << 6) | (ch1 & 0x3F));
}

inline STRF_HD bool first_2_of_3_are_valid(unsigned x, strf::surrogate_policy surr_poli)
{
    return ( surr_poli == strf::surrogate_policy::lax
          || (x >> 5) != 0x1B );
}
inline STRF_HD bool first_2_of_3_are_valid
    ( unsigned ch0
    , unsigned ch1
    , strf::surrogate_policy surr_poli )
{
    return first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), surr_poli);
}

inline STRF_HD unsigned utf8_decode_first_2_of_4(unsigned ch0, unsigned ch1)
{
    return ((ch0 ^ 0xF0) << 6) | (ch1 & 0x3F);
}

inline STRF_HD unsigned utf8_decode_last_2_of_4(unsigned x, unsigned ch2, unsigned ch3)
{
    return (x << 12) | ((ch2 & 0x3F) <<  6) | (ch3 & 0x3F);
}

inline STRF_HD bool first_2_of_4_are_valid(unsigned x)
{
    return 0xF < x && x < 0x110;
}

inline STRF_HD bool first_2_of_4_are_valid(unsigned ch0, unsigned ch1)
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
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end )
    {
        return src_end - src;
    }

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-8");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end )
    {
        return src_end - src;
    }

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-1");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy surr_poli );

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end );

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    static STRF_HD void transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli );

    static STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::surrogate_policy )
    {
        return src_end - src;
    }

    static STRF_HD void unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier );

    static STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end )
    {
        return src_end - src;
    }

    static STRF_HD strf::transcode_f<SrcCharT, DestCharT> transcode_func() noexcept
    {
        return transcode;
    }
    static STRF_HD strf::transcode_size_f<SrcCharT> transcode_size_func() noexcept
    {
        return transcode_size;
    }
    static STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func() noexcept
    {
        return unsafe_transcode;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
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
    < SrcCharT, DestCharT, strf::get_csid_utf<SrcCharT>(), strf::get_csid_utf<DestCharT>() >;

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
    static constexpr STRF_HD int replacement_char_size() noexcept
    {
        return 3;
    }
    static constexpr STRF_HD int validate(char32_t ch) noexcept
    {
        return ( ch < 0x80     ? 1 :
                 ch < 0x800    ? 2 :
                 ch < 0x10000  ? 3 :
                 ch < 0x110000 ? 4 : -1 );
    }
    static constexpr STRF_HD int encoded_char_size(char32_t ch) noexcept
    {
        return ( ch < 0x80     ? 1 :
                 ch < 0x800    ? 2 :
                 ch < 0x10000  ? 3 :
                 ch < 0x110000 ? 4 : 3 );
    }
    static STRF_HD CharT* encode_char
        ( CharT* dest, char32_t ch ) noexcept;

    static STRF_HD void encode_fill
        ( strf::transcode_dest<CharT>&, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::transcode_dest<CharT>& );

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
        if (id == strf::get_csid_utf<DestCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::get_csid_utf<SrcCharT>()) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 3, validate, encoded_char_size,
            encode_char, encode_fill, count_codepoints_fast,
            count_codepoints, write_replacement_char, decode_unit,
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
#endif // defined (__cpp_char8_t)
        };
        return strf::dynamic_charset<CharT>{data};
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
    static constexpr STRF_HD int replacement_char_size() noexcept
    {
        return 1;
    }
    static constexpr STRF_HD int validate(char32_t ch) noexcept
    {
        return ch < 0x10000 ? 1 : ch < 0x110000 ? 2 : -1;
    }
    static constexpr STRF_HD int encoded_char_size(char32_t ch) noexcept
    {
        return 1 + (0x10000 <= ch && ch < 0x110000);
    }

    static STRF_HD CharT* encode_char
        (CharT* dest, char32_t ch) noexcept;

    static STRF_HD void encode_fill
        ( strf::transcode_dest<CharT>&, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count, strf::surrogate_policy surr_poli ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::transcode_dest<CharT>& );

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
        if (id == strf::get_csid_utf<DestCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::get_csid_utf<SrcCharT>()) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, count_codepoints_fast,
            count_codepoints, write_replacement_char, decode_unit,
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
    static constexpr STRF_HD std::ptrdiff_t replacement_char_size() noexcept
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
    static constexpr STRF_HD int validate(char32_t) noexcept
    {
        return 1;
    }
    static constexpr STRF_HD int encoded_char_size(char32_t) noexcept
    {
        return 1;
    }
    static STRF_HD CharT* encode_char
        (CharT* dest, char32_t ch) noexcept
    {
        *dest = static_cast<CharT>(ch);
        return dest + 1;
    }
    static STRF_HD void encode_fill
        ( strf::transcode_dest<CharT>&, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept
    {
        STRF_ASSERT(src <= src_end);
        auto src_size = src_end - src;
        if (max_count <= src_size) {
            return {max_count, src + max_count};
        }
        return {src_size, src_end};

    }

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count, strf::surrogate_policy surr_poli ) noexcept
    {
        (void)surr_poli;
        STRF_ASSERT(src <= src_end);
        auto src_size = src_end - src;
        if (max_count <= src_size) {
            return {max_count, src + max_count};
        }
        return {src_size, src_end};
    }

    static STRF_HD void write_replacement_char
        ( strf::transcode_dest<CharT>& );

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
        if (id == strf::get_csid_utf<DestCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DestCharT>{}};
        }
        return {};
    }
    template <typename SrcCharT>
    static STRF_HD strf::dynamic_transcoder<SrcCharT, CharT>
    find_transcoder_from(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<SrcCharT, CharT>;
        if (id == strf::get_csid_utf<SrcCharT>()) {
            return transcoder_type{strf::utf_to_utf<SrcCharT, CharT>{}};
        }
        return {};
    }
    static strf::dynamic_charset<CharT> to_dynamic() noexcept
    {
        static const strf::dynamic_charset_data<CharT> data = {
            name(), id(), replacement_char(), 1, validate, encoded_char_size,
            encode_char, encode_fill, count_codepoints_fast,
            count_codepoints, write_replacement_char, decode_unit,
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
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;
    using strf::detail::utf8_decode_last_2_of_4;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::is_utf8_continuation;

    unsigned ch0 = 0, ch1 = 0, ch2 = 0, ch3 = 0;
    unsigned ch32 = 0;
    unsigned x = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();

    while (src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* seq_begin = src_it;
        ++src_it;
        if (ch0 < 0x80) {
            ch32 = ch0;
        } else if (0xC0 == (ch0 & 0xE0)) {
            if( ch0 > 0xC1 && src_it != src_end
             && is_utf8_continuation(ch1 = detail::cast_u8(*src_it)) ) {
                ch32 = utf8_decode(ch0, ch1);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end
              && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                ch32 = ((ch1 & 0x3F) << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                       , surr_poli )
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                ch32 = (x << 6) | (ch2 & 0x3F);
                ++src_it;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                 && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch2 = detail::cast_u8(*src_it))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch3 = detail::cast_u8(*src_it)) )
        {
            ch32 = utf8_decode_last_2_of_4(x, ch2, ch3);
            ++src_it;
        } else {
            invalid_sequence:
            ch32 = 0xFFFD;
            if (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
        }

        STRF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    while (src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
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
            if ( src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++src_it != src_end && is_utf8_continuation(*src_it) )
            {
                ++src_it;
            }
        } else if (   src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
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
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();

    while (src_it < src_end) {
        unsigned ch32 = 0;
        const unsigned ch0 = detail::cast_u8(*src_it);
        ++src_it;
        if (ch0 < 0x80) {
            ch32 = ch0;
        } else if (ch0 < 0xE0) {
            ch32 = strf::detail::utf8_decode(ch0, detail::cast_u8(src_it[0]));
            ++ src_it;
        } else if (ch0 < 0xF0) {
            ch32 = strf::detail::utf8_decode
                (ch0
                , detail::cast_u8(src_it[0])
                , detail::cast_u8(src_it[1]) );
            src_it += 2;
        } else {
            ch32 = strf::detail::utf8_decode
                ( ch0
                , detail::cast_u8(src_it[0])
                , detail::cast_u8(src_it[1])
                , detail::cast_u8(src_it[2]));
            src_it += 3;
        }
        STRF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    for (;src_it < src_end; ++src_it) {
        const unsigned ch = detail::cast_u8(*src_it);
        size += detail::not_utf8_continuation(ch);
    }
    //     unsigned ch0 = detail::cast_u8(*src_it);
    //     src_it += ( ch0 < 0x80 ? 1
    //               : ch0 < 0xE0 ? 2
    //               : ch0 < 0xF0 ? 3
    //               :              4);
    // }
    return size;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0, ch2 = 0, ch3 = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    while(src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* seq_begin = src_it;
        ++src_it;
        if(ch0 < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch0);
            ++dest_it;
        } else if(0xC0 == (ch0 & 0xE0)) {
            if ( ch0 > 0xC1
              && src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))) {
                STRF_CHECK_DEST_SIZE(2);
                ++src_it;
                dest_it[0] = static_cast<DestCharT>(ch0);
                dest_it[1] = static_cast<DestCharT>(ch1);
                dest_it += 2;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end
              && (0xA0 == ((ch1 = detail::cast_u8(*src_it)) & 0xE0))
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_CHECK_DEST_SIZE(3);
                ++src_it;
                dest_it[0] = static_cast<DestCharT>(ch0);
                dest_it[1] = static_cast<DestCharT>(ch1);
                dest_it[2] = static_cast<DestCharT>(ch2);
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_CHECK_DEST_SIZE(3);
                ++src_it;
                dest_it[0] = static_cast<DestCharT>(ch0);
                dest_it[1] = static_cast<DestCharT>(ch1);
                dest_it[2] = static_cast<DestCharT>(ch2);
                dest_it += 3;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src_it != src_end
                 && is_utf8_continuation(ch2 = detail::cast_u8(*src_it))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch3 = detail::cast_u8(*src_it)) )
        {
            STRF_CHECK_DEST_SIZE(4);
            ++src_it;
            dest_it[0] = static_cast<DestCharT>(ch0);
            dest_it[1] = static_cast<DestCharT>(ch1);
            dest_it[2] = static_cast<DestCharT>(ch2);
            dest_it[3] = static_cast<DestCharT>(ch3);
            dest_it += 4;
        } else {
            invalid_sequence:
            if (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    const SrcCharT* src_it = src;
    std::ptrdiff_t size = 0;
    while(src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
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
            if (   src_it != src_end
              && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
              && ++src_it != src_end
              && is_utf8_continuation(* src_it) )
            {
                size += 3;
                ++src_it;
            } else {
                size += 3;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            size += 3;
            if ( src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++src_it != src_end
              && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if( src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
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

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    STRF_ASSERT(src <= src_end);
    (void) err_notifier;
    detail::output_buffer_interchar_copy<DestCharT>(dest, src, src_end);
}

template <typename CharT>
STRF_HD strf::count_codepoints_result<CharT>
static_charset<CharT, strf::csid_utf8>::count_codepoints_fast
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count ) noexcept
{
    STRF_ASSERT(src <= src_end);
    std::ptrdiff_t count = 0;
    const auto *it = src;
    while (it < src_end && count < max_count) {
        if (!strf::detail::is_utf8_continuation(*it)) {
            ++ count;
        }
        ++it;
    }
    while(it < src_end && strf::detail::is_utf8_continuation(*it)) {
        ++it;
    }
    return {count, it};
}

template <typename CharT>
STRF_HD strf::count_codepoints_result<CharT>
static_charset<CharT, strf::csid_utf8>::count_codepoints
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count
    , strf::surrogate_policy surr_poli ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    std::ptrdiff_t count = 0;
    const auto *it = src;
    while (it < src_end && count < max_count) {
        ch0 = detail::cast_u8(*it);
        ++it;
        ++count;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && it != src_end && is_utf8_continuation(*it)) {
                ++it;
            }
        } else if (0xE0 == ch0) {
            if (   it != src_end && ((*it & 0xE0) == 0xA0)
              && ++it != src_end && is_utf8_continuation(*it) )
            {
                ++it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( it != src_end && is_utf8_continuation(ch1 = detail::cast_u8(*it))
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++it != src_end && is_utf8_continuation(*it) )
            {
                ++it;
            }
        } else if (   it != src_end && is_utf8_continuation(ch1 = detail::cast_u8(*it))
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++it != src_end && is_utf8_continuation(*it)
                 && ++it != src_end && is_utf8_continuation(*it) )
        {
            ++it;
        }
    }
    return {count, it};
}


template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::encode_fill
    ( strf::transcode_dest<CharT>& dest, std::ptrdiff_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x80) {
        strf::detail::write_fill(dest, count, static_cast<CharT>(ch));
    } else if (ch < 0x800) {
        auto ch0 = static_cast<CharT>(0xC0 | ((ch & 0x7C0) >> 6));
        auto ch1 = static_cast<CharT>(0x80 |  (ch &  0x3F));
        strf::detail::repeat_sequence<CharT>(dest, count, ch0, ch1);
    } else if (ch <  0x10000) {
        auto ch0 = static_cast<CharT>(0xE0 | ((ch & 0xF000) >> 12));
        auto ch1 = static_cast<CharT>(0x80 | ((ch &  0xFC0) >> 6));
        auto ch2 = static_cast<CharT>(0x80 |  (ch &   0x3F));
        strf::detail::repeat_sequence<CharT>(dest, count, ch0, ch1, ch2);
    } else if (ch < 0x110000) {
        auto ch0 = static_cast<CharT>(0xF0 | ((ch & 0x1C0000) >> 18));
        auto ch1 = static_cast<CharT>(0x80 | ((ch &  0x3F000) >> 12));
        auto ch2 = static_cast<CharT>(0x80 | ((ch &    0xFC0) >> 6));
        auto ch3 = static_cast<CharT>(0x80 |  (ch &     0x3F));
        strf::detail::repeat_sequence<CharT>(dest, count, ch0, ch1, ch2, ch3);
    } else {
        auto ch0 = static_cast<CharT>('\xEF');
        auto ch1 = static_cast<CharT>('\xBF');
        auto ch2 = static_cast<CharT>('\xBD');
        strf::detail::repeat_sequence<CharT>(dest, count, ch0, ch1, ch2);
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
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    for(;src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (ch < 0x10000) {
            STRF_IF_LIKELY ( surr_poli == strf::surrogate_policy::lax
                          || strf::detail::not_surrogate(ch))
            {
                STRF_CHECK_DEST_SIZE(3);
                dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
                dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
                dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (ch < 0x110000) {
            STRF_CHECK_DEST_SIZE(4);
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (ch >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 4;
        } else {
            invalid_sequence:
            STRF_IF_UNLIKELY (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence(4, "UTF-32", (const void*)src_it, 1);
            }
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    const auto *src_it = src;
    std::ptrdiff_t count = 0;
    for(;src_it < src_end; ++src_it) {
        auto ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x110000) {
            count += 1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000);
        } else {
            count += 3;
        }
    }
    return count;
}


template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    for(;src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);

        const auto dest_space = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : ch < 0x10000 ? 3 : 4);
            if (required_space > dest_space) {
                STRF_RECYCLE;
            }
        }
        STRF_IF_LIKELY (ch < 0x80) {
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (ch < 0x10000) {
            dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
            dest_it += 3;
        } else if (ch < 0x110000) {
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (ch >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 4;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    for(;src_it < src_end; ++src_it) {
        auto ch = detail::cast_u32(*src_it);
        size += ( ch < 0x80 ? 1
                : ch < 0x800 ? 2
                : ch < 0x10000 ? 3 : 4 );
    }
    return size;
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::write_replacement_char
    ( strf::transcode_dest<CharT>& dest )
{
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    STRF_CHECK_DEST_SIZE(3);
    dest_it[0] = static_cast<CharT>('\xEF');
    dest_it[1] = static_cast<CharT>('\xBF');
    dest_it[2] = static_cast<CharT>('\xBD');
    dest_it += 3;
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    const auto *src_it = src;
    while (src_it < src_end) {
        const unsigned ch = detail::cast_u16(*src_it);
        unsigned ch32 = ch;
        ++src_it;
        STRF_IF_UNLIKELY (strf::detail::is_surrogate(ch)) {
            unsigned ch2 = 0;
            if ( strf::detail::is_high_surrogate(ch)
              && src_it != src_end
              && strf::detail::is_low_surrogate(ch2 = detail::cast_u16(*src_it))) {
                ch32 = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
                ++src_it;
            } else if (surr_poli != strf::surrogate_policy::lax) {
                ch32 = 0xFFFD;
                if (err_notifier) {
                    dest.advance_to(dest_it);
                    err_notifier->invalid_sequence(2, "UTF-16", src_it - 1, 1);
                }
            }
        }
        STRF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    unsigned ch = 0;
    std::ptrdiff_t count = 0;
    const auto *src_it = src;
    const SrcCharT* src_it_next = nullptr;
    for(; src_it < src_end; src_it = src_it_next) {
        src_it_next = src_it + 1;
        ch = detail::cast_u16(*src_it);

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
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;

    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    const auto *src_it = src;
    while (src_it < src_end) {
        const unsigned ch = detail::cast_u16(*src_it);
        unsigned ch32 = ch;
        ++src_it;
        STRF_IF_UNLIKELY (strf::detail::is_high_surrogate(ch)) {
            const unsigned ch2 = detail::cast_u16(*src_it);
            ch32 = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            ++src_it;
        }
        STRF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    std::ptrdiff_t size = 0;
    const auto *src_it = src;
    while (src_it < src_end) {
        const unsigned ch = detail::cast_u16(*src_it);
        src_it += strf::detail::is_high_surrogate(ch) ? 2 : 1;
        ++ size;
    }
    return size;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    while (src_it < src_end) {
        unsigned ch = detail::cast_u16(*src_it);
        ++src_it;
        STRF_IF_UNLIKELY (strf::detail::is_surrogate(ch)) {
            unsigned ch2 = 0;
            if ( strf::detail::is_high_surrogate(ch)
              && src_it != src_end
              && strf::detail::is_low_surrogate(ch2 = detail::cast_u16(*src_it)) )
            {
                ++src_it;
                STRF_CHECK_DEST_SIZE(2);
                dest_it[0] = static_cast<DestCharT>(ch);
                dest_it[1] = static_cast<DestCharT>(ch2);
                dest_it += 2;
                continue;
            }
            if (surr_poli != strf::surrogate_policy::lax) {
                ch = 0xFFFD;
                if (err_notifier) {
                    dest.advance_to(dest_it);
                    err_notifier->invalid_sequence(2, "UTF-16", src_it - 1, 1);
                }
            }
        }
        STRF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch);
        ++dest_it;
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::ptrdiff_t count = 0;
    const SrcCharT* src_it = src;
    unsigned ch = 0;
    while (src_it < src_end) {
        ch = detail::cast_u16(*src_it);
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

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    STRF_ASSERT(src <= src_end);
    detail::output_buffer_interchar_copy<DestCharT>(dest, src, src_end);
}

template <typename CharT>
STRF_HD strf::count_codepoints_result<CharT>
static_charset<CharT, strf::csid_utf16>::count_codepoints_fast
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count ) noexcept
{
    if (src >= src_end) {
        return {0, src};
    }
    std::ptrdiff_t count = 0;
    const auto *it = src;
    while (count < max_count) {
        if (strf::detail::is_high_surrogate(*it)) {
            ++it;
        }
        ++it;
        ++count;
        if (it >= src_end) {
            return {count, src_end};
        }
    }
    return {count, it};
}

template <typename CharT>
STRF_HD strf::count_codepoints_result<CharT>
static_charset<CharT, strf::csid_utf16>::count_codepoints
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count
    , strf::surrogate_policy surr_poli ) noexcept
{
    (void) surr_poli;
    std::ptrdiff_t count = 0;
    const CharT* it = src;
    unsigned ch = 0;
    while (it < src_end && count < max_count) {
        ch = detail::cast_u16(*it);
        ++ it;
        ++ count;
        if ( strf::detail::is_high_surrogate(ch) && it != src_end
          && strf::detail::is_low_surrogate(*it)) {
            ++ it;
        }
    }
    return {count, it};
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
        const char32_t sub_codepoint = ch - 0x10000;
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
    ( strf::transcode_dest<CharT>& dest, std::ptrdiff_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x10000) {
        strf::detail::write_fill<CharT>(dest, count, static_cast<CharT>(ch));
    } else if (ch < 0x110000) {
        const char32_t sub_codepoint = ch - 0x10000;
        auto ch0 = static_cast<CharT>(0xD800 + (sub_codepoint >> 10));
        auto ch1 = static_cast<CharT>(0xDC00 + (sub_codepoint &  0x3FF));
        strf::detail::repeat_sequence<CharT>(dest, count, ch0, ch1);
    } else {
        strf::detail::write_fill<CharT>(dest, count, 0xFFFD);
    }
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    for ( ; src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);
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
            const auto sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 | (sub_codepoint >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dest_it += 2;
        } else {
            invalid_char:
            if (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence(4, "UTF-32", src_it, 1);
            }
            STRF_CHECK_DEST;
            *dest_it = 0xFFFD;
            ++dest_it;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::ptrdiff_t count = 0;
    const SrcCharT* src_it = src;
    for ( ; src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);
        count += 1 + (0x10000 <= ch && ch < 0x110000);
    }
    return count;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    for ( ; src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);

        const auto dest_size = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_size < 2) {
            const int required_size = ch < 0x10000 ? 1 : 2;
            if (dest_size < required_size) {
                STRF_RECYCLE;
            }
        }
        STRF_IF_LIKELY (ch < 0x10000) {
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else  {
            const auto sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 | (sub_codepoint >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dest_it += 2;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    return transcode_size(src, src_end, strf::surrogate_policy::strict);
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf16>::write_replacement_char
    ( strf::transcode_dest<CharT>& dest )
{
    dest.ensure(1);
    *dest.buffer_ptr() = 0xFFFD;
    dest.advance();
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    if (surr_poli == strf::surrogate_policy::lax) {
        for (const auto *src_it = src; src_it < src_end; ++src_it) {
            unsigned ch = detail::cast_u32(*src_it);
            STRF_IF_UNLIKELY (ch >= 0x110000) {
                ch = 0xFFFD;
                if (err_notifier) {
                    dest.advance_to(dest_it);
                    err_notifier->invalid_sequence(4, "UTF-32", src_it, 1);
                }
            }
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        }
    } else {
        for(const auto *src_it = src; src_it < src_end; ++src_it) {
            unsigned ch = detail::cast_u32(*src_it);
            STRF_IF_UNLIKELY (ch >= 0x110000 || strf::detail::is_surrogate(ch)) {
                ch = 0xFFFD;
                if (err_notifier) {
                    dest.advance_to(dest_it);
                    err_notifier->invalid_sequence(4, "UTF-32", src_it, 1);
                }
            }
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    STRF_ASSERT(src <= src_end);
    detail::output_buffer_interchar_copy<DestCharT>(dest, src, src_end);
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf32>::encode_fill
    ( strf::transcode_dest<CharT>& dest, std::ptrdiff_t count, char32_t ch )
{
    strf::detail::write_fill(dest, count, static_cast<CharT>(ch));
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf32>::write_replacement_char
    ( strf::transcode_dest<CharT>& dest )
{
    dest.ensure(1);
    *dest.buffer_ptr() = 0xFFFD;
    dest.advance();
}


template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;
    using strf::detail::utf8_decode_last_2_of_4;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0, ch2 = 0, ch3 = 0;
    unsigned x = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();

    for (;src_it < src_end; ++dest_it) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* seq_begin = src_it;
        ++src_it;
        STRF_IF_LIKELY (ch0 < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch0);
        } else if (0xC0 == (ch0 & 0xE0)) {
            STRF_IF_LIKELY ( ch0 > 0xC1
                          && src_it != src_end
                          && is_utf8_continuation(ch1 = detail::cast_u8(*src_it)))
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(utf8_decode(ch0, ch1));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            STRF_IF_LIKELY ( src_it != src_end
                          && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(((ch1 & 0x3F) << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            STRF_IF_LIKELY (( src_it != src_end
                          && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                          && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                                   , surr_poli )
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) ))
            {
                STRF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>((x << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                 && first_2_of_4_are_valid(x = utf8_decode_first_2_of_4(ch0, ch1))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch2 = detail::cast_u8(*src_it))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch3 = detail::cast_u8(*src_it)) )
        {
            STRF_CHECK_DEST_SIZE(2);
            x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 +  (x >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 +  (x & 0x3FF));
            ++dest_it;
            ++src_it;
        } else {
            invalid_sequence:
            if (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence(1, "UTF-8", seq_begin, src_it - seq_begin);
            }
            STRF_CHECK_DEST;
            *dest_it = 0xFFFD;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::ptrdiff_t size = 0;
    unsigned ch0 = 0, ch1 = 0;
    const auto *src_it = src;
    while(src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        ++size;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && src_it != src_end && is_utf8_continuation(*src_it)) {
                ++src_it;
            }
        } else if (0xE0 == ch0) {
            if (   src_it != src_end
              && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
              && ++src_it != src_end
              && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, surr_poli)
              && ++src_it != src_end
              && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if (   src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
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
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    using strf::detail::utf8_decode;
    unsigned ch0 = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();
    for (;src_it < src_end; ++dest_it) {
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        STRF_CHECK_DEST;
        STRF_IF_LIKELY (ch0 < 0x80) {
            *dest_it = static_cast<DestCharT>(ch0);
        } else if (ch0 < 0xE0) {
            const auto ch32 = strf::detail::utf8_decode(ch0, detail::cast_u8(src_it[0]));
            *dest_it = static_cast<DestCharT>(ch32);
            ++src_it;
        } else if (ch0 < 0xF0) {
            const auto ch32 = strf::detail::utf8_decode
                (ch0, detail::cast_u8(src_it[0]), detail::cast_u8(src_it[1]));
            *dest_it = static_cast<DestCharT>(ch32);
            src_it += 2;
        } else{
            STRF_CHECK_DEST_SIZE(2);
            const auto ch32 = strf::detail::utf8_decode
                ( ch0
                , detail::cast_u8(src_it[0])
                , detail::cast_u8(src_it[1])
                , detail::cast_u8(src_it[2]) );
            const auto x = ch32 - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 +  (x >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 +  (x & 0x3FF));
            ++dest_it;
            src_it += 3;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    std::ptrdiff_t size = 0;
    const auto *src_it = src;
    while(src_it < src_end) {
        const unsigned ch0 = detail::cast_u8(*src_it);
        const int x = ( ch0 < 0x80 ? 0x09
                      : ch0 < 0xE0 ? 0x0A
                      : ch0 < 0xF0 ? 0x0B
                      :              0x14);
        src_it += x &  7;
        size   += x >> 3;
    }
    return size;
}

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();

    while (src_it < src_end) {
        unsigned const ch = detail::cast_u16(*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (strf::detail::not_surrogate(ch)) {
            three_bytes:
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
            dest_it += 3;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src_it != src_end
               && strf::detail::is_low_surrogate(*src_it) )
        {
            STRF_CHECK_DEST_SIZE(4);
            const unsigned ch2 = detail::cast_u16(*src_it);
            ++src_it;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  codepoint));
            dest_it += 4;
        } else if (surr_poli == strf::surrogate_policy::lax) {
            goto three_bytes;
        } else { // invalid sequece
            if (err_notifier) {
                dest.advance_to(dest_it);
                err_notifier->invalid_sequence(2, "UTF-16", src_it - 1, 1);
            }
            STRF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::surrogate_policy surr_poli )
{
    (void) surr_poli;
    std::ptrdiff_t size = 0;
    for(const auto *it = src; it < src_end; ++it) {
        unsigned const ch = detail::cast_u16(*it);
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

template <typename SrcCharT, typename DestCharT>
STRF_HD void strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier )
{
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = dest.buffer_end();

    while (src_it < src_end) {
        unsigned const ch = detail::cast_u16(*src_it);
        const auto dest_space = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : strf::detail::not_high_surrogate(ch) ? 3 : 4);
            if (required_space > dest_space) {
                STRF_RECYCLE;
            }
        }
        ++src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (strf::detail::not_high_surrogate(ch)) {
            dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
            dest_it += 3;
        } else {
            const unsigned ch2 = detail::cast_u16(*src_it);
            ++src_it;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  codepoint));
            dest_it += 4;
        }
    }
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD std::ptrdiff_t strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end )
{
    std::ptrdiff_t size = 0;
    for(const auto *it = src; it < src_end; ) {
        unsigned const ch = detail::cast_u16(*it);
        const bool surrogate = detail::is_high_surrogate(ch);
        const int x = ( ch < 0x80   ? 0x9
                      : ch < 0x800  ? 0xA
                      : surrogate   ? 0x14
                      :               0xB );
        size += x & 7;
        it   += x >> 3;
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
using utf_t = strf::static_charset<CharT, strf::detail::csid_utf_impl<sizeof(CharT)>::csid>;

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

