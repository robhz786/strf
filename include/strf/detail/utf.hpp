#ifndef STRF_DETAIL_UTF_HPP
#define STRF_DETAIL_UTF_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>
#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

#define STRF_UTF_CHECK_DST                                      \
    STRF_IF_UNLIKELY (dst_it >= dst_end) {                     \
        return {seq_begin, dst_it, reason::insufficient_output_space};    \
    }

#define STRF_UTF_CHECK_DST_SIZE(SIZE)                           \
    STRF_IF_UNLIKELY (dst_it + (SIZE) > dst_end) {             \
        return {seq_begin, dst_it, reason::insufficient_output_space};    \
    }

namespace detail {

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dst<CharT>& dst
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1 ) noexcept
{
    auto *p = dst.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 2;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dst.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p += seq_size;
        }
        dst.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dst.flush();
        STRF_IF_UNLIKELY (!dst.good()) {
            return;
        }
        p = dst.buffer_ptr();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dst<CharT>& dst
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2 ) noexcept
{
    auto *p = dst.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 3;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dst.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p += seq_size;
        }
        dst.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dst.flush();
        STRF_IF_UNLIKELY (!dst.good()) {
            return;
        }
        p = dst.buffer_ptr();
        count -= space;
    }
}

template <typename CharT>
inline STRF_HD void repeat_sequence
    ( strf::transcode_dst<CharT>& dst
    , std::ptrdiff_t count
    , CharT ch0
    , CharT ch1
    , CharT ch2
    , CharT ch3 ) noexcept
{
    auto *p = dst.buffer_ptr();
    constexpr std::ptrdiff_t seq_size = 4;
    std::ptrdiff_t space = 0;
    std::ptrdiff_t inner_count = 0;
    while (1) {
        space = (dst.buffer_end() - p) / seq_size;
        inner_count = (space < count ? space : count);
        for (; inner_count > 0; --inner_count) {
            p[0] = ch0;
            p[1] = ch1;
            p[2] = ch2;
            p[3] = ch3;
            p += seq_size;
        }
        dst.advance_to(p);
        STRF_IF_LIKELY (count <= space) {
            return;
        }
        dst.flush();
        STRF_IF_UNLIKELY (!dst.good()) {
            return;
        }
        p = dst.buffer_ptr();
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

inline STRF_HD unsigned utf8_decode_first_2_of_3(unsigned ch0, unsigned ch1)
{
    return static_cast<unsigned>(((ch0 & 0x0F) << 6) | (ch1 & 0x3F));
}

inline STRF_HD bool first_2_of_3_are_valid(unsigned x, bool lax_surrogate_policy)
{
    return (lax_surrogate_policy || (x >> 5) != 0x1B);
}
inline STRF_HD bool first_2_of_3_are_valid
    ( unsigned ch0
    , unsigned ch1
    , bool lax_surr )
{
    return first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), lax_surr);
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

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DstCharT) == 1, "Incompatible character type for UTF-8");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t size
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags);

    static STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags ) noexcept
    {
        const auto s = (src_end - src);
        if (s <= limit) {
            return {s, src_end, transcode_stop_reason::completed};
        }
        while (limit > 0 && detail::is_utf8_continuation(src[limit])) {
            --limit;
        }
        return {limit, src + limit, transcode_stop_reason::insufficient_output_space};
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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_continue_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DstCharT) == 2, "Incompatible character type for UTF-16");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_continue_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DstCharT) == 4, "Incompatible character type for UTF-32");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_continue_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DstCharT) == 1, "Incompatible character type for UTF-8");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode_lax_surr_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end );

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode_strict_surr_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_continue_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DstCharT) == 2, "Incompatible character type for UTF-16");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags);

    static STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags ) noexcept
    {
        const auto size = (src_end - src);
        if (size <= limit) {
            return {size, src_end, transcode_stop_reason::completed};
        }
        if ( limit > 0
          && detail::is_high_surrogate(src[limit - 1])
          && detail::is_low_surrogate(src[limit]) )
        {
            --limit;
        }
        return {limit, src + limit, transcode_stop_reason::insufficient_output_space};
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

private:

    template <bool HasLimit, bool StopOnInvalidSeq>
    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DstCharT) == 4, "Incompatible character type for UTF-32");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_continue_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DstCharT) == 1, "Incompatible character type for UTF-1");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    template <bool StopOnLimit, bool StopOnInvalidSeq, bool StricSurrPoli>
    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DstCharT) == 2, "Incompatible character type for UTF-16");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags flags ) noexcept;

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
        , strf::transcode_flags flags ) noexcept;

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

private:

    template <bool StopOnLimit, bool StopOnInvalidSeq, bool StricSurrPoli>
    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit ) noexcept;
};

template <typename SrcCharT, typename DstCharT>
class static_transcoder<SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DstCharT) == 4, "Incompatible character type for UTF-32");

    using src_char_type = SrcCharT;
    using dst_char_type = DstCharT;

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags );

    static STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags )
    {
        return detail::bypass_unsafe_transcode(src, src_end, dst, err_notifier, flags);
    }

    static STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , std::ptrdiff_t limit
        , strf::transcode_flags )
    {
        const auto s = (src_end - src);
        if (s <= limit) {
            return {s, src_end, transcode_stop_reason::completed};
        }
        return {limit, src + limit, transcode_stop_reason::insufficient_output_space};
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
        return detail::bypass_unsafe_transcode<SrcCharT, DstCharT>;
    }
    static STRF_HD strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func() noexcept
    {
        return unsafe_transcode_size;
    }
};

template <typename SrcCharT, typename DstCharT>
using utf8_to_utf8 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >;
template <typename SrcCharT, typename DstCharT>
using utf8_to_utf16 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >;
template <typename SrcCharT, typename DstCharT>
using utf8_to_utf32 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >;

template <typename SrcCharT, typename DstCharT>
using utf16_to_utf8 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >;
template <typename SrcCharT, typename DstCharT>
using utf16_to_utf16 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16 >;
template <typename SrcCharT, typename DstCharT>
using utf16_to_utf32 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >;

template <typename SrcCharT, typename DstCharT>
using utf32_to_utf8 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >;
template <typename SrcCharT, typename DstCharT>
using utf32_to_utf16 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >;
template <typename SrcCharT, typename DstCharT>
using utf32_to_utf32 = strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf32 >;

template <typename SrcCharT, typename DstCharT>
using utf_to_utf = strf::static_transcoder
    < SrcCharT, DstCharT, strf::get_csid_utf<SrcCharT>(), strf::get_csid_utf<DstCharT>() >;

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
        ( CharT* dst, char32_t ch ) noexcept;

    static STRF_HD void encode_fill
        ( strf::transcode_dst<CharT>&, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::transcode_dst<CharT>& );

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
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::tag<DstCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DstCharT>(id);
    }
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DstCharT>;
        if (id == strf::get_csid_utf<DstCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DstCharT>{}};
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
        (CharT* dst, char32_t ch) noexcept;

    static STRF_HD void encode_fill
        ( strf::transcode_dst<CharT>&, std::ptrdiff_t count, char32_t ch );

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints_fast
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD strf::count_codepoints_result<CharT> count_codepoints
        ( const CharT* src, const CharT* src_end
        , std::ptrdiff_t max_count ) noexcept;

    static STRF_HD void write_replacement_char
        ( strf::transcode_dst<CharT>& );

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
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::tag<DstCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DstCharT>(id);
    }
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DstCharT>;
        if (id == strf::get_csid_utf<DstCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DstCharT>{}};
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
        (CharT* dst, char32_t ch) noexcept
    {
        *dst = static_cast<CharT>(ch);
        return dst + 1;
    }
    static STRF_HD void encode_fill
        ( strf::transcode_dst<CharT>&, std::ptrdiff_t count, char32_t ch );

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
        , std::ptrdiff_t max_count ) noexcept
    {
        STRF_ASSERT(src <= src_end);
        auto src_size = src_end - src;
        if (max_count <= src_size) {
            return {max_count, src + max_count};
        }
        return {src_size, src_end};
    }

    static STRF_HD void write_replacement_char
        ( strf::transcode_dst<CharT>& );

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
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::tag<DstCharT>, strf::charset_id id) noexcept
    {
        return find_transcoder_to<DstCharT>(id);
    }
    template <typename DstCharT>
    static STRF_HD strf::dynamic_transcoder<CharT, DstCharT>
    find_transcoder_to(strf::charset_id id) noexcept
    {
        using transcoder_type = strf::dynamic_transcoder<CharT, DstCharT>;
        if (id == strf::get_csid_utf<DstCharT>()) {
            return transcoder_type{strf::utf_to_utf<CharT, DstCharT>{}};
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

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    <SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst_it
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
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
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* const seq_begin = src_it;
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
              && first_2_of_3_are_valid(x = utf8_decode_first_2_of_3(ch0, ch1), lax_surr)
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
            if (err_notifier) {
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, dst_it, reason::invalid_sequence};
            }
            ch32 = 0xFFFD;
        }

        STRF_UTF_CHECK_DST;
        *dst_it = static_cast<DstCharT>(ch32);
        ++dst_it;
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, limit, flags);
    }
    return transcode_size_continue_on_inv_seq_(src, src_end, limit, flags);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;

    using stop_reason = strf::transcode_stop_reason;

    unsigned ch0 = 0, ch1 = 0;
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        const auto* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        if (ch0 >= 0x80) {
            if (0xC0 == (ch0 & 0xE0)) {
                if( ch0 > 0xC1 && src_it != src_end
                 && is_utf8_continuation(detail::cast_u8(*src_it)) ) {
                    ++src_it;
                } else goto invalid_sequence;
            } else if (0xE0 == ch0) {
                if (   src_it != src_end
                  && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
                  && ++src_it != src_end
                  && is_utf8_continuation(detail::cast_u8(*src_it)) )
                {
                    ++src_it;
                } else goto invalid_sequence;
            } else if (0xE0 == (ch0 & 0xF0)) {
                if (   src_it != src_end
                  && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                  && first_2_of_3_are_valid(utf8_decode_first_2_of_3(ch0, ch1), lax_surr)
                  && ++src_it != src_end
                  && is_utf8_continuation(detail::cast_u8(*src_it)) )
                {
                    ++src_it;
                } else goto invalid_sequence;
            } else if ( src_it != src_end
                     && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                     && first_2_of_4_are_valid(utf8_decode_first_2_of_4(ch0, ch1))
                     && ++src_it != src_end
                     && is_utf8_continuation(detail::cast_u8(*src_it))
                     && ++src_it != src_end
                     && is_utf8_continuation(detail::cast_u8(*src_it)) )
            {
                ++src_it;
            } else {
                invalid_sequence:
                return {size, seq_begin, stop_reason::invalid_sequence};
            }
            if (size >= limit) {
                return {size, seq_begin, stop_reason::insufficient_output_space};
            }
            ++size;
        }
    }
    return {size, src_end, stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size_continue_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        if (size >= limit){
            return {size, src_it, strf::transcode_stop_reason::insufficient_output_space};
        }
        ++size;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
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
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
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
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    if (src >= src_end) {
        return {src, dst, reason::completed};
    }
    if (dst >= dst_end) {
        return {src, dst, reason::insufficient_output_space};
    }
    const auto *src_it = src;
    auto *dst_it = dst;

    while (src_it < src_end) {
        const SrcCharT* const seq_begin = src_it;
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
        STRF_UTF_CHECK_DST;
        *dst_it = static_cast<DstCharT>(ch32);
        ++dst_it;
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags ) noexcept
{
    const auto *src_it = src;
    std::ptrdiff_t size = 0;

    if (src_end - src <= limit) {
        for (;src_it < src_end; ++src_it) {
            const unsigned ch = detail::cast_u8(*src_it);
            size += detail::not_utf8_continuation(ch);
        }
        // to-do: check if the below code is faster:
        //
        //     unsigned ch0 = detail::cast_u8(*src_it);
        //     src_it += ( ch0 < 0x80 ? 1
        //               : ch0 < 0xE0 ? 2
        //               : ch0 < 0xF0 ? 3
        //               :              4);
    } else {
        for (;src_it < src_end; ++src_it) {
            if (size >= limit){
                return {size, src_it, strf::transcode_stop_reason::insufficient_output_space};
            }
            const unsigned ch = detail::cast_u8(*src_it);
            size += detail::not_utf8_continuation(ch);
        }
    }
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0, ch2 = 0, ch3 = 0;
    const auto *src_it = src;
    auto *dst_it = dst;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while(src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* seq_begin = src_it;
        ++src_it;
        if(ch0 < 0x80) {
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch0);
            ++dst_it;
        } else if(0xC0 == (ch0 & 0xE0)) {
            if ( ch0 > 0xC1
              && src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))) {
                STRF_UTF_CHECK_DST_SIZE(2);
                ++src_it;
                dst_it[0] = static_cast<DstCharT>(ch0);
                dst_it[1] = static_cast<DstCharT>(ch1);
                dst_it += 2;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            if (   src_it != src_end
              && (0xA0 == ((ch1 = detail::cast_u8(*src_it)) & 0xE0))
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_UTF_CHECK_DST_SIZE(3);
                ++src_it;
                dst_it[0] = static_cast<DstCharT>(ch0);
                dst_it[1] = static_cast<DstCharT>(ch1);
                dst_it[2] = static_cast<DstCharT>(ch2);
                dst_it += 3;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_UTF_CHECK_DST_SIZE(3);
                ++src_it;
                dst_it[0] = static_cast<DstCharT>(ch0);
                dst_it[1] = static_cast<DstCharT>(ch1);
                dst_it[2] = static_cast<DstCharT>(ch2);
                dst_it += 3;
            } else goto invalid_sequence;
        } else if ( src_it != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src_it != src_end
                 && is_utf8_continuation(ch2 = detail::cast_u8(*src_it))
                 && ++src_it != src_end
                 && is_utf8_continuation(ch3 = detail::cast_u8(*src_it)) )
        {
            STRF_UTF_CHECK_DST_SIZE(4);
            ++src_it;
            dst_it[0] = static_cast<DstCharT>(ch0);
            dst_it[1] = static_cast<DstCharT>(ch1);
            dst_it[2] = static_cast<DstCharT>(ch2);
            dst_it[3] = static_cast<DstCharT>(ch3);
            dst_it += 4;
        } else {
            invalid_sequence:
            if (err_notifier) {
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, dst_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DST_SIZE(3);
            dst_it[0] = static_cast<DstCharT>('\xEF');
            dst_it[1] = static_cast<DstCharT>('\xBF');
            dst_it[2] = static_cast<DstCharT>('\xBD');
            dst_it += 3;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");

    auto len = src_end - src;
    const auto dst_space = dst_end - dst;
    if (len <= dst_space) {
        detail::copy_n(src, len, dst);
        STRF_ASSERT(src_end == src + len);
        return {src_end, dst + len, reason::completed};
    }
    auto next_src = src + dst_space;
    while (detail::is_utf8_continuation(*next_src) && next_src > src) {
        --next_src;
    }
    const auto count = next_src - src;
    strf::detail::copy_n(src, count, dst);
    return {next_src, dst + count, reason::insufficient_output_space};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, limit, flags);
    }
    return transcode_size_continue_on_inv_seq_(src, src_end, limit, flags);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::not_utf8_continuation;

    const SrcCharT* const src_begin = src;
    const SrcCharT* const src_limit = src + limit;
    unsigned ch0 = 0, ch1 = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while (src < src_end) {
        const auto* const seq_begin = src;
        ch0 = detail::cast_u8(*src);
        ++src;
        if (ch0 >= 0x80) {
            if (0xC0 == (ch0 & 0xE0)) {
                if ( ch0 <= 0xC1 || src == src_end || not_utf8_continuation(*src)) {
                    goto invalid_sequence;
                }
            } else if (0xE0 == ch0) {
                if (   src != src_end
                  && (((ch1 = detail::cast_u8(*src)) & 0xE0) == 0xA0)
                  && ++src != src_end
                  && is_utf8_continuation(* src) )
                {
                    ++src;
                } else goto invalid_sequence;
            } else if (0xE0 == (ch0 & 0xF0)) {
                if (   src != src_end
                  && is_utf8_continuation(ch1 = detail::cast_u8(*src))
                  && first_2_of_3_are_valid(ch0, ch1, lax_surr)
                  && ++src != src_end
                  && is_utf8_continuation(*src) )
                {
                    ++src;
                } else goto invalid_sequence;
            } else if ( src != src_end
                     && is_utf8_continuation(ch1 = detail::cast_u8(*src))
                     && first_2_of_4_are_valid(ch0, ch1)
                     && ++src != src_end
                     && is_utf8_continuation(detail::cast_u8(*src))
                     && ++src != src_end
                     && is_utf8_continuation(*src) )
            {
                ++src;
            } else {
                invalid_sequence:
                const auto size = seq_begin - src_begin;
                return {size, seq_begin, strf::transcode_stop_reason::invalid_sequence};
            }
            if (src > src_limit) {
                const auto size = seq_begin - src_begin;
                return {size, seq_begin, strf::transcode_stop_reason::insufficient_output_space};

            }
        }
    }
    const auto size = src - src_begin;
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size_continue_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    const SrcCharT* src_it = src;
    std::ptrdiff_t size = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while (src_it < src_end) {
        if (size >= limit){
            return {size, src_it, strf::transcode_stop_reason::insufficient_output_space};
        }
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
            size += 3;
            if (   src_it != src_end
              && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
              && ++src_it != src_end
              && is_utf8_continuation(* src_it) )
            {
                ++src_it;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            size += 3;
            if ( src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
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
    return {size, src_end, strf::transcode_stop_reason::completed};
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
    , std::ptrdiff_t max_count ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0;
    std::ptrdiff_t count = 0;
    const auto *it = src;
    constexpr bool lax_surr = false;
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
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
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
    ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x80) {
        strf::detail::write_fill(dst, count, static_cast<CharT>(ch));
    } else if (ch < 0x800) {
        auto ch0 = static_cast<CharT>(0xC0 | ((ch & 0x7C0) >> 6));
        auto ch1 = static_cast<CharT>(0x80 |  (ch &  0x3F));
        strf::detail::repeat_sequence<CharT>(dst, count, ch0, ch1);
    } else if (ch <  0x10000) {
        auto ch0 = static_cast<CharT>(0xE0 | ((ch & 0xF000) >> 12));
        auto ch1 = static_cast<CharT>(0x80 | ((ch &  0xFC0) >> 6));
        auto ch2 = static_cast<CharT>(0x80 |  (ch &   0x3F));
        strf::detail::repeat_sequence<CharT>(dst, count, ch0, ch1, ch2);
    } else if (ch < 0x110000) {
        auto ch0 = static_cast<CharT>(0xF0 | ((ch & 0x1C0000) >> 18));
        auto ch1 = static_cast<CharT>(0x80 | ((ch &  0x3F000) >> 12));
        auto ch2 = static_cast<CharT>(0x80 | ((ch &    0xFC0) >> 6));
        auto ch3 = static_cast<CharT>(0x80 |  (ch &     0x3F));
        strf::detail::repeat_sequence<CharT>(dst, count, ch0, ch1, ch2, ch3);
    } else {
        auto ch0 = static_cast<CharT>('\xEF');
        auto ch1 = static_cast<CharT>('\xBF');
        auto ch2 = static_cast<CharT>('\xBD');
        strf::detail::repeat_sequence<CharT>(dst, count, ch0, ch1, ch2);
    }
}

template <typename CharT>
STRF_HD CharT*
static_charset<CharT, strf::csid_utf8>::encode_char
    ( CharT* dst
    , char32_t ch ) noexcept
{
    STRF_IF_LIKELY (ch < 0x80) {
        *dst = static_cast<CharT>(ch);
        return dst + 1;
    }
    if (ch < 0x800) {
        dst[0] = static_cast<CharT>(0xC0 | ((ch & 0x7C0) >> 6));
        dst[1] = static_cast<CharT>(0x80 |  (ch &  0x3F));
        return dst + 2;
    }
    if (ch <  0x10000) {
        dst[0] = static_cast<CharT>(0xE0 | ((ch & 0xF000) >> 12));
        dst[1] = static_cast<CharT>(0x80 | ((ch &  0xFC0) >> 6));
        dst[2] = static_cast<CharT>(0x80 |  (ch &   0x3F));
        return dst + 3;
    }
    if (ch < 0x110000) {
        dst[0] = static_cast<CharT>(0xF0 | ((ch & 0x1C0000) >> 18));
        dst[1] = static_cast<CharT>(0x80 | ((ch &  0x3F000) >> 12));
        dst[2] = static_cast<CharT>(0x80 | ((ch &    0xFC0) >> 6));
        dst[3] = static_cast<CharT>(0x80 |  (ch &     0x3F));
        return dst + 4;
    }
    dst[0] = static_cast<CharT>('\xEF');
    dst[1] = static_cast<CharT>('\xBF');
    dst[2] = static_cast<CharT>('\xBD');
    return dst + 3;
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dst_it = dst;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    for(;src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else if (ch < 0x800) {
            STRF_UTF_CHECK_DST_SIZE(2);
            dst_it[0] = static_cast<DstCharT>(0xC0 | (0x1F & (ch >> 6)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 2;
        } else if (ch < 0x10000) {
            STRF_IF_LIKELY (lax_surr || strf::detail::not_surrogate(ch))
            {
                STRF_UTF_CHECK_DST_SIZE(3);
                dst_it[0] = static_cast<DstCharT>(0xE0 | (0x0F & (ch >> 12)));
                dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
                dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF &  ch ));
                dst_it += 3;
            } else goto invalid_sequence;
        } else if (ch < 0x110000) {
            STRF_UTF_CHECK_DST_SIZE(4);
            dst_it[0] = static_cast<DstCharT>(0xF0 | (0x07 & (ch >> 18)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 12)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[3] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 4;
        } else {
            invalid_sequence:
            if (err_notifier) {
                err_notifier->invalid_sequence(4, "UTF-32", (const void*)src_it, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src_it, dst_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DST_SIZE(3);
            dst_it[0] = static_cast<DstCharT>('\xEF');
            dst_it[1] = static_cast<DstCharT>('\xBF');
            dst_it[2] = static_cast<DstCharT>('\xBD');
            dst_it += 3;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if ((src_end - src) * 4 > limit) {
        if (strf::with_stop_on_invalid_sequence(flags)) {
            if (strf::with_strict_surrogate_policy(flags)) {
                return transcode_size_<true, true, true>(src, src_end, limit);
            }
            return transcode_size_<true, true, false>(src, src_end, limit);
        }
        if (strf::with_strict_surrogate_policy(flags)) {
            return transcode_size_<true, false, true>(src, src_end, limit);
        }
        return transcode_size_<true, false, false>(src, src_end, limit);
    }
    if (strf::with_stop_on_invalid_sequence(flags)) {
        if (strf::with_strict_surrogate_policy(flags)) {
            return transcode_size_<false, true, true>(src, src_end, limit);
        }
        return transcode_size_<false, true, false>(src, src_end, limit);
    }
    if (strf::with_strict_surrogate_policy(flags)) {
        return transcode_size_<false, false, true>(src, src_end, limit);
    }
    return transcode_size_<false, false, false>(src, src_end, limit);
}

template <typename SrcCharT, typename DstCharT>
template <bool StopOnLimit, bool StopOnInvalidSeq, bool StricSurrPoli>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
< SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >::transcode_size_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    decltype(limit) size = 0;
    using stop_reason = strf::transcode_stop_reason;
    (void) limit;
    for(;src < src_end; ++src) {
        if (StopOnLimit && !StopOnInvalidSeq && size >= limit) {
            return {size, src, stop_reason::insufficient_output_space};
        }
        const auto ch = detail::cast_u32(*src);
        const bool isSurrogate = StricSurrPoli && detail::is_surrogate(ch);
        const bool invalid =  isSurrogate || ch >= 0x110000;
        if (StopOnInvalidSeq && invalid) {
            return {size, src, stop_reason::invalid_sequence};
        }
        const auto ch_size = invalid ? 3 : (1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000));
        size += ch_size;
        if (StopOnLimit && StopOnInvalidSeq && size > limit) {
            return {size - ch_size, src, stop_reason::insufficient_output_space};
        }
    }
    return {size, src_end, stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    if (src >= src_end) {
        return {src, dst, reason::completed};
    }
    if (dst == dst_end) {
        return {src, dst, reason::insufficient_output_space};
    }
    const auto *src_it = src;
    auto *dst_it = dst;
    for(;src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);

        const auto dst_space = dst_end - dst_it;
        STRF_IF_UNLIKELY (dst_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : ch < 0x10000 ? 3 : 4);
            if (required_space > dst_space) {
                return {src_it, dst_it, reason::insufficient_output_space};
            }
        }
        STRF_IF_LIKELY (ch < 0x80) {
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else if (ch < 0x800) {
            dst_it[0] = static_cast<DstCharT>(0xC0 | (0x1F & (ch >> 6)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 2;
        } else if (ch < 0x10000) {
            dst_it[0] = static_cast<DstCharT>(0xE0 | (0x0F & (ch >> 12)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF &  ch ));
            dst_it += 3;
        } else if (ch < 0x110000) {
            dst_it[0] = static_cast<DstCharT>(0xF0 | (0x07 & (ch >> 18)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 12)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[3] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 4;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags ) noexcept
{
    limit = limit <= 0 ? 0 : limit;
    std::ptrdiff_t size = 0;
    const std::ptrdiff_t pre_limit = limit - 4;
    
    while (size < pre_limit) {
        if (src == src_end) {
            goto completed;
        }
        auto ch = detail::cast_u32(*src);
        const auto ch_size = ( ch < 0x80 ? 1
                             : ch < 0x800 ? 2
                             : ch < 0x10000 ? 3 : 4 );
        size += ch_size;
        ++src;
    }
    for(;src < src_end; ++src) {
        if (size >= limit) {
            return {size, src, strf::transcode_stop_reason::insufficient_output_space};
        }
        auto ch = detail::cast_u32(*src);
        const auto ch_size = ( ch < 0x80 ? 1
                               : ch < 0x800 ? 2
                               : ch < 0x10000 ? 3 : 4 );
        const auto new_size = size + ch_size;
        if (new_size > limit) {
            return {size, src, strf::transcode_stop_reason::insufficient_output_space};
        }
        size = new_size;
    }

  completed:
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::write_replacement_char
    ( strf::transcode_dst<CharT>& dst )
{
    dst.ensure(3);
    auto *dst_it = dst.buffer_ptr();
    dst_it[0] = static_cast<CharT>('\xEF');
    dst_it[1] = static_cast<CharT>('\xBF');
    dst_it[2] = static_cast<CharT>('\xBD');
    dst_it += 3;
    dst.advance_to(dst_it);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    const auto *src_it = src;
    while (src_it < src_end) {
        const SrcCharT* const seq_begin = src_it;
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
            } else if (strf::with_strict_surrogate_policy(flags)) {
                if (err_notifier) {
                    err_notifier->invalid_sequence(2, "UTF-16", seq_begin, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, dst_it, reason::invalid_sequence};
                }
                ch32 = 0xFFFD;
            }
        }
        STRF_UTF_CHECK_DST;
        *dst_it = static_cast<DstCharT>(ch32);
        ++dst_it;
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags) && strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, limit);
    }
    return transcode_size_continue_on_inv_seq_(src, src_end, limit);
}
template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size_continue_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    unsigned ch = 0;
    std::ptrdiff_t count = 0;
    const auto *src_it = src;
    const SrcCharT* src_it_next = nullptr;
    for(; src_it < src_end; src_it = src_it_next) {
        if (count >= limit) {
            return {count, src_it, strf::transcode_stop_reason::insufficient_output_space};
        }
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
    return {count, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    unsigned ch = 0;
    std::ptrdiff_t count = 0;
    const SrcCharT* src_next = nullptr;
    for(; src < src_end; src = src_next) {
        src_next = src + 1;
        ch = detail::cast_u16(*src);
        if (strf::detail::is_surrogate(ch)) {
            if ( strf::detail::is_high_surrogate(ch)
              && src_next != src_end
              && strf::detail::is_low_surrogate(*src_next)) {
                ++src_next;
            } else {
                return {count, src, strf::transcode_stop_reason::invalid_sequence};
            }
        }
        if (count >= limit) {
           return {count, src, strf::transcode_stop_reason::insufficient_output_space};
        }
        ++count;
    }
    return {count, src_end, strf::transcode_stop_reason::completed};
}


template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags flags)
{
    using reason = strf::transcode_stop_reason;

    auto *dst_it = dst;
    const auto *src_it = src;
    if (strf::with_strict_surrogate_policy(flags)) {
        while (src_it < src_end) {
            const SrcCharT* const seq_begin = src_it;
            const unsigned ch = detail::cast_u16(*src_it);
            unsigned ch32 = ch;
            ++src_it;
            STRF_IF_UNLIKELY (strf::detail::is_high_surrogate(ch)) {
                const unsigned ch2 = detail::cast_u16(*src_it);
                ch32 = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
                ++src_it;
            }
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch32);
            ++dst_it;
        }
    } else {
        while (src_it < src_end) {
            const SrcCharT* const seq_begin = src_it;
            const unsigned ch = detail::cast_u16(*src_it);
            unsigned ch32 = ch;
            ++src_it;
            STRF_IF_UNLIKELY (src_it != src_end && strf::detail::is_high_surrogate(ch)) {
                const unsigned ch2 = detail::cast_u16(*src_it);
                if (strf::detail::is_low_surrogate(ch2)) {
                    ch32 = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
                    ++src_it;
                }
            }
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch32);
            ++dst_it;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    std::ptrdiff_t size = 0;
    if (strf::with_strict_surrogate_policy(flags)) {
        while (src < src_end) {
            if (size >= limit) {
                goto insufficient_output_space;
            }
            const unsigned ch = detail::cast_u16(*src);
            src += strf::detail::is_high_surrogate(ch) ? 2 : 1;
            ++size;
        }
    } else {
        while (src < src_end) {
            if (size >= limit) {
                goto insufficient_output_space;
            }
            const unsigned ch = detail::cast_u16(*src);
            if (++src != src_end
             && strf::detail::is_high_surrogate(ch)
             && strf::detail::is_low_surrogate(*src) )
            {
                ++src;
            }
            ++size;
        }
    }
    return {size, src_end, transcode_stop_reason::completed};

  insufficient_output_space:
    return {size, src, strf::transcode_stop_reason::insufficient_output_space};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dst_it = dst;
    while (src_it < src_end) {
        const SrcCharT* const seq_begin = src_it;
        unsigned ch = detail::cast_u16(*src_it);
        ++src_it;
        STRF_IF_UNLIKELY (strf::detail::is_surrogate(ch)) {
            unsigned ch2 = 0;
            if ( strf::detail::is_high_surrogate(ch)
              && src_it != src_end
              && strf::detail::is_low_surrogate(ch2 = detail::cast_u16(*src_it)) )
            {
                ++src_it;
                STRF_UTF_CHECK_DST_SIZE(2);
                dst_it[0] = static_cast<DstCharT>(ch);
                dst_it[1] = static_cast<DstCharT>(ch2);
                dst_it += 2;
                continue;
            }
            if (strf::with_strict_surrogate_policy(flags)){
                if (err_notifier) {
                    err_notifier->invalid_sequence(2, "UTF-16", seq_begin, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, dst_it, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
        }
        STRF_UTF_CHECK_DST;
        *dst_it = static_cast<DstCharT>(ch);
        ++dst_it;
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");

    if (src < src_end) {
        auto len = src_end - src;
        STRF_ASSERT(src < src_end);
        const auto dst_space = dst_end - dst;
        if (len <= dst_space) {
            detail::copy_n(src, len, dst);
            STRF_ASSERT(src_end == src + len);
            return {src_end, dst + len, reason::completed};
        }
        auto count = dst_space;
        if (dst_space != 0) {
            if (detail::is_low_surrogate(src[count]) &&
                detail::is_high_surrogate(src[count - 1])) {
                --count;
            }
            strf::detail::copy_n(src, count, dst);
        }
        return {src + count, dst + count, reason::insufficient_output_space};
    }
    return {src, dst, reason::completed};
}


template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    const bool stopOnInvSeq = ( strf::with_strict_surrogate_policy(flags)
                             && strf::with_stop_on_invalid_sequence(flags) );
    const bool hasLimit = (src_end - src > limit);
    if (hasLimit) {
        if (stopOnInvSeq) {
            return transcode_size_<true, true>(src, src_end, limit);
        }
        return transcode_size_<true, false>(src, src_end, limit);
    }
    if (stopOnInvSeq) {
        return transcode_size_<false, true>(src, src_end, limit);
    }
    return transcode_size_<false, false>(src, src_end, limit);

}

template <typename SrcCharT, typename DstCharT>
template <bool HasLimit, bool StopOnInvalidSeq>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    using reason = strf::transcode_stop_reason;
    (void) limit;
    unsigned ch = 0;
    const auto* const src_begin = src;
    const SrcCharT* src_next = nullptr;
    const SrcCharT* src_limit = src + limit;
    for(; src < src_end; src = src_next) {
        if (HasLimit && !StopOnInvalidSeq && src_next >= src_limit) {
            return {src - src_begin, src, reason::insufficient_output_space};
        }
        src_next = src + 1;
        ch = detail::cast_u16(*src);
        if (strf::detail::is_surrogate(ch)) {
            if ( strf::detail::is_high_surrogate(ch)
              && src_next != src_end
              && strf::detail::is_low_surrogate(*src_next))
            {
                ++src_next;
            } else if (StopOnInvalidSeq) {
                const auto size = src - src_begin;
                return {size, src, reason::invalid_sequence};
            }
        }
        if (HasLimit && StopOnInvalidSeq && src_next > src_limit) {
            //src = src_next == src_limit ? src_next : src;
            const auto size = src - src_begin;
            return {size, src, reason::insufficient_output_space};
        }
    }
    const auto size = src - src_begin;
    return {size, src_end, reason::completed};
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
    , std::ptrdiff_t max_count ) noexcept
{
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
    ( CharT* dst
    , char32_t ch ) noexcept
{
    STRF_IF_LIKELY (ch < 0x10000) {
        *dst = static_cast<CharT>(ch);
        return dst + 1;
    }
    if (ch < 0x110000) {
        const char32_t sub_codepoint = ch - 0x10000;
        dst[0] = static_cast<CharT>(0xD800 + (sub_codepoint >> 10));
        dst[1] = static_cast<CharT>(0xDC00 + (sub_codepoint &  0x3FF));
        return dst + 2;
    }
    *dst = 0xFFFD;
    return dst + 1;
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf16>::encode_fill
    ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch )
{
    STRF_IF_LIKELY (ch < 0x10000) {
        strf::detail::write_fill<CharT>(dst, count, static_cast<CharT>(ch));
    } else if (ch < 0x110000) {
        const char32_t sub_codepoint = ch - 0x10000;
        auto ch0 = static_cast<CharT>(0xD800 + (sub_codepoint >> 10));
        auto ch1 = static_cast<CharT>(0xDC00 + (sub_codepoint &  0x3FF));
        strf::detail::repeat_sequence<CharT>(dst, count, ch0, ch1);
    } else {
        strf::detail::write_fill<CharT>(dst, count, 0xFFFD);
    }
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dst_it = dst;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    for ( ; src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x10000) {
            STRF_IF_LIKELY (lax_surr || strf::detail::not_surrogate(ch))
            {
                STRF_UTF_CHECK_DST;
                *dst_it = static_cast<DstCharT>(ch);
                ++dst_it;
            } else goto invalid_char;
        } else if (ch < 0x110000) {
            STRF_UTF_CHECK_DST_SIZE(2);
            const auto sub_codepoint = ch - 0x10000;
            dst_it[0] = static_cast<DstCharT>(0xD800 | (sub_codepoint >> 10));
            dst_it[1] = static_cast<DstCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dst_it += 2;
        } else {
            invalid_char:
            if (err_notifier) {
                err_notifier->invalid_sequence(4, "UTF-32", seq_begin, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, dst_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DST;
            *dst_it = 0xFFFD;
            ++dst_it;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    const bool HasLimit = (src_end - src) * 2 > limit;
    const bool StopOnInvalidSeq = strf::with_stop_on_invalid_sequence(flags);
    const bool StrictSurrPoli = strf::with_strict_surrogate_policy(flags);

    if (HasLimit) {
        if (StopOnInvalidSeq) {
            if (StrictSurrPoli) {
                return transcode_size_<true, true, true>(src, src_end, limit);
            }
            return transcode_size_<true, true, false>(src, src_end, limit);
        }
        if (StrictSurrPoli) {
            return transcode_size_<true, false, true>(src, src_end, limit);
        }
        return transcode_size_<true, false, false>(src, src_end, limit);
    }
    if (StopOnInvalidSeq) {
        if (StrictSurrPoli) {
            return transcode_size_<false, true, true>(src, src_end, limit);
        }
        return transcode_size_<false, true, false>(src, src_end, limit);
    }
    if (StrictSurrPoli) {
        return transcode_size_<false, false, true>(src, src_end, limit);
    }
    return transcode_size_<false, false, false>(src, src_end, limit);
}

template <typename SrcCharT, typename DstCharT>
template <bool StopOnLimit, bool StopOnInvalidSeq, bool StricSurrPoli>
    STRF_HD strf::transcode_size_result<SrcCharT> static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >::transcode_size_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    decltype(limit) size = 0;
    using reason = strf::transcode_stop_reason;
    (void) limit;
    for ( ; src < src_end; ++src) {
        if (StopOnLimit && !StopOnInvalidSeq && size >= limit) {
            return {size, src, reason::insufficient_output_space};
        }
        const auto ch = detail::cast_u32(*src);
        const bool isSurrogate = StricSurrPoli && detail::is_surrogate(ch);
        const bool invalid =  isSurrogate || ch >= 0x110000;
        if (StopOnInvalidSeq && invalid) {
            return {size, src, reason::invalid_sequence};
        }
        const auto ch_size = invalid ? 1 : (ch < 0x10000 ? 1 : 2);
        size += ch_size;
        if (StopOnLimit && StopOnInvalidSeq && size > limit) {
            return {size - ch_size, src, reason::insufficient_output_space};
        }
    }
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dst_it = dst;
    for ( ; src_it < src_end; ++src_it) {
        unsigned const ch = detail::cast_u32(*src_it);
        const auto dst_size = dst_end - dst_it;
        STRF_IF_UNLIKELY (dst_size < 2) {
            const int required_size = ch < 0x10000 ? 1 : 2;
            if (dst_size < required_size) {
                return {src_it, dst_it, reason::insufficient_output_space};
            }
        }
        STRF_IF_LIKELY (ch < 0x10000) {
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else  {
            const auto sub_codepoint = ch - 0x10000;
            dst_it[0] = static_cast<DstCharT>(0xD800 | (sub_codepoint >> 10));
            dst_it[1] = static_cast<DstCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dst_it += 2;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags) noexcept
{
    using reason = strf::transcode_stop_reason;
    (void) flags;
    decltype(limit) size = 0;
    if ((src_end - src) * 2 <= limit) {
        for ( ; src < src_end; ++src) {
            unsigned const ch = detail::cast_u32(*src);
            size += 1 + (0x10000 <= ch);
        }
    } else {
        for ( ; src < src_end; ++src) {
            unsigned const ch = detail::cast_u32(*src);
            const auto ch_size = 1 + (0x10000 <= ch);
            size += ch_size;
            if (size >= limit) {
                if (size > limit) {
                    return {size - ch_size, src, reason::completed};
                }
                return {size, src + 1, reason::completed};
            }
        }
    }
    return {size, src_end, reason::completed};
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf16>::write_replacement_char
    ( strf::transcode_dst<CharT>& dst )
{
    dst.ensure(1);
    *dst.buffer_ptr() = 0xFFFD;
    dst.advance();
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    if (strf::with_lax_surrogate_policy(flags)) {
        for (; src < src_end; ++src) {
            const SrcCharT* const seq_begin = src;
            unsigned ch = detail::cast_u32(*src);
            STRF_IF_UNLIKELY (ch >= 0x110000) {
                if (err_notifier) {
                    err_notifier->invalid_sequence(4, "UTF-32", src, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, dst_it, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        }
    } else {
        for(; src < src_end; ++src) {
            const SrcCharT* const seq_begin = src;
            unsigned ch = detail::cast_u32(*src);
            STRF_IF_UNLIKELY (ch >= 0x110000 || strf::detail::is_surrogate(ch)) {
                if (err_notifier) {
                    err_notifier->invalid_sequence(4, "UTF-32", src, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, dst_it, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        }
    }
    return {src, dst_it, reason::completed};
}


template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf32, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    if ( ! strf::with_stop_on_invalid_sequence(flags)) {
        if (src_end - src <= limit) {
            return {src_end - src, src_end, reason::completed};
        } else {
            return {src + limit - src, src_end, reason::insufficient_output_space};
        }
    }
    const auto* const src_begin = src;
    if (src_end - src <= limit) {
        if (strf::with_strict_surrogate_policy(flags)) {
            for ( ; src < src_end; ++src) {
                if (*src >= 0x110000 || detail::is_surrogate(*src)) {
                    return {src - src_begin, src, reason::invalid_sequence};
                }
            }
        } else {
            for ( ; src < src_end; ++src) {
                if (*src >= 0x110000) {
                    return {src - src_begin, src, reason::invalid_sequence};
                }
            }
        }
    } else {
        const auto* src_limit = src_begin + limit;
        if (strf::with_strict_surrogate_policy(flags)) {
            for ( ; src < src_end; ++src) {
                if (*src >= 0x110000 || detail::is_surrogate(*src)) {
                    return {src - src_begin, src, reason::invalid_sequence};
                }
                if (src >= src_limit) {
                    return {src - src_begin, src, reason::insufficient_output_space};
                }
            }
        } else {
            for ( ; src < src_end; ++src) {
                if (*src >= 0x110000) {
                    return {src - src_begin, src, reason::invalid_sequence};
                }
                if (src >= src_limit) {
                    return {src - src_begin, src, reason::insufficient_output_space};
                }
            }
        }
    }
    return {src_end - src_begin, src_end, reason::completed};
}

template <typename CharT>
STRF_HD void static_charset<CharT, strf::csid_utf32>::encode_fill
    ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch )
{
    strf::detail::write_fill(dst, count, static_cast<CharT>(ch));
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf32>::write_replacement_char
    ( strf::transcode_dst<CharT>& dst )
{
    dst.ensure(1);
    *dst.buffer_ptr() = 0xFFFD;
    dst.advance();
}


template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
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
    auto *dst_it = dst;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    for (;src_it < src_end; ++dst_it) {
        const SrcCharT* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch0 < 0x80) {
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch0);
        } else if (0xC0 == (ch0 & 0xE0)) {
            STRF_IF_LIKELY ( ch0 > 0xC1
                          && src_it != src_end
                          && is_utf8_continuation(ch1 = detail::cast_u8(*src_it)))
            {
                STRF_UTF_CHECK_DST;
                *dst_it = static_cast<DstCharT>(utf8_decode(ch0, ch1));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            STRF_IF_LIKELY ( src_it != src_end
                          && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_UTF_CHECK_DST;
                *dst_it = static_cast<DstCharT>(((ch1 & 0x3F) << 6) | (ch2 & 0x3F));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            STRF_IF_LIKELY (( src_it != src_end
                          && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
                          && first_2_of_3_are_valid( x = utf8_decode_first_2_of_3(ch0, ch1)
                                                   , lax_surr )
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) ))
            {
                STRF_UTF_CHECK_DST;
                *dst_it = static_cast<DstCharT>((x << 6) | (ch2 & 0x3F));
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
            STRF_UTF_CHECK_DST_SIZE(2);
            x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
            dst_it[0] = static_cast<DstCharT>(0xD800 +  (x >> 10));
            dst_it[1] = static_cast<DstCharT>(0xDC00 +  (x & 0x3FF));
            ++dst_it;
            ++src_it;
        } else {
            invalid_sequence:
            if (err_notifier) {
                err_notifier->invalid_sequence(1, "UTF-8", seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, dst_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DST;
            *dst_it = 0xFFFD;
        }
    }
    return {src_it, dst_it, reason::completed};
}
template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, limit, flags);
    }
    return transcode_size_continue_on_inv_seq_(src, src_end, limit, flags);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;
    using stop_reason = strf::transcode_stop_reason;

    std::ptrdiff_t size = 0;
    unsigned ch0 = 0, ch1 = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src < src_end) {
        const auto* const seq_begin = src;
        const auto previous_size = size;
        ch0 = detail::cast_u8(*src);
        ++src;
        ++size;
        STRF_IF_LIKELY (ch0 >= 0x80) {
            if (0xC0 == (ch0 & 0xE0)) {
                STRF_IF_LIKELY ( ch0 > 0xC1
                              && src != src_end
                              && is_utf8_continuation(ch1 = detail::cast_u8(*src)))
                {
                    ++src;
                } else goto invalid_sequence;
            } else if (0xE0 == ch0) {
                STRF_IF_LIKELY ( src != src_end
                              && (((ch1 = detail::cast_u8(*src)) & 0xE0) == 0xA0)
                              && ++src != src_end
                              && is_utf8_continuation(detail::cast_u8(*src)) )
                {
                    ++src;
                } else goto invalid_sequence;
            } else if (0xE0 == (ch0 & 0xF0)) {
                STRF_IF_LIKELY (( src != src_end
                              && is_utf8_continuation(ch1 = detail::cast_u8(*src))
                              && first_2_of_3_are_valid( utf8_decode_first_2_of_3(ch0, ch1)
                                                       , lax_surr )
                              && ++src != src_end
                              && is_utf8_continuation(detail::cast_u8(*src)) ))
                {
                    ++src;
                } else goto invalid_sequence;
            } else if ( src != src_end
                     && is_utf8_continuation(ch1 = detail::cast_u8(*src))
                     && first_2_of_4_are_valid(utf8_decode_first_2_of_4(ch0, ch1))
                     && ++src != src_end
                     && is_utf8_continuation(detail::cast_u8(*src))
                     && ++src != src_end
                     && is_utf8_continuation(detail::cast_u8(*src)) )
            {
                ++size;
                ++src;
            } else {
                invalid_sequence:
                return {size - 1, seq_begin, stop_reason::invalid_sequence};
            }
            if (size > limit) {
                return {previous_size, seq_begin, stop_reason::insufficient_output_space};
            }
        }
    }
    return {size, src_end, stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size_continue_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using stop_reason = strf::transcode_stop_reason;

    std::ptrdiff_t size = 0;
    unsigned ch0 = 0, ch1 = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while(src < src_end) {
        if (size >= limit) {
            return {size, src, stop_reason::insufficient_output_space};
        }
        ch0 = detail::cast_u8(*src);
        ++src;
        ++size;
        if (0xC0 == (ch0 & 0xE0)) {
            if (ch0 > 0xC1 && src != src_end && is_utf8_continuation(*src)) {
                ++src;
            }
        } else if (0xE0 == ch0) {
            if (   src != src_end
              && (((ch1 = detail::cast_u8(*src)) & 0xE0) == 0xA0)
              && ++src != src_end
              && is_utf8_continuation(* src) )
            {
                ++src;
            }
        } else if (0xE0 == (ch0 & 0xF0)) {
            if ( src != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src))
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
              && ++src != src_end
              && is_utf8_continuation(* src) )
            {
                ++src;
            }
        } else if (   src != src_end
                 && is_utf8_continuation(ch1 = detail::cast_u8(*src))
                 && first_2_of_4_are_valid(ch0, ch1)
                 && ++src != src_end && is_utf8_continuation(*src)
                 && ++src != src_end && is_utf8_continuation(*src) )
        {
            ++src;
            ++size;
        }
    }
    return {size, src_end, stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    using strf::detail::utf8_decode;
    unsigned ch0 = 0;
    const auto *src_it = src;
    auto *dst_it = dst;
    for (;src_it < src_end; ++dst_it) {
        const SrcCharT* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        STRF_UTF_CHECK_DST;
        STRF_IF_LIKELY (ch0 < 0x80) {
            *dst_it = static_cast<DstCharT>(ch0);
        } else if (ch0 < 0xE0) {
            const auto ch32 = strf::detail::utf8_decode(ch0, detail::cast_u8(src_it[0]));
            *dst_it = static_cast<DstCharT>(ch32);
            ++src_it;
        } else if (ch0 < 0xF0) {
            const auto ch32 = strf::detail::utf8_decode
                (ch0, detail::cast_u8(src_it[0]), detail::cast_u8(src_it[1]));
            *dst_it = static_cast<DstCharT>(ch32);
            src_it += 2;
        } else{
            STRF_UTF_CHECK_DST_SIZE(2);
            const auto ch32 = strf::detail::utf8_decode
                ( ch0
                , detail::cast_u8(src_it[0])
                , detail::cast_u8(src_it[1])
                , detail::cast_u8(src_it[2]) );
            const auto x = ch32 - 0x10000;
            dst_it[0] = static_cast<DstCharT>(0xD800 +  (x >> 10));
            dst_it[1] = static_cast<DstCharT>(0xDC00 +  (x & 0x3FF));
            ++dst_it;
            src_it += 3;
        }
    }
    return {src_it, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags ) noexcept
{
    using stop_reason = strf::transcode_stop_reason;

    std::ptrdiff_t size = 0;
    while (src < src_end) {
        if (size >= limit) {
            return {size, src, stop_reason::insufficient_output_space};
        }
        const unsigned ch0 = detail::cast_u8(*src);
        const int x = ( ch0 < 0x80 ? 0x09
                      : ch0 < 0xE0 ? 0x0A
                      : ch0 < 0xF0 ? 0x0B
                      :              0x14);
        size += x >> 3;
        src  += x &  7;
    }
    return {size, src_end, stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    (void) err_notifier;
    const auto *src_it = src;
    auto *dst_it = dst;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u16(*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_UTF_CHECK_DST;
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else if (ch < 0x800) {
            STRF_UTF_CHECK_DST_SIZE(2);
            dst_it[0] = static_cast<DstCharT>(0xC0 | (0x1F & (ch >> 6)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 2;
        } else if (strf::detail::not_surrogate(ch)) {
            three_bytes:
            STRF_UTF_CHECK_DST_SIZE(3);
            dst_it[0] = static_cast<DstCharT>(0xE0 | (0x0F & (ch >> 12)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF &  ch ));
            dst_it += 3;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src_it != src_end
               && strf::detail::is_low_surrogate(*src_it) )
        {
            STRF_UTF_CHECK_DST_SIZE(4);
            const unsigned ch2 = detail::cast_u16(*src_it);
            ++src_it;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dst_it[0] = static_cast<DstCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dst_it[3] = static_cast<DstCharT>(0x80 | (0xBF &  codepoint));
            dst_it += 4;
        } else if (lax_surr) {
            goto three_bytes;
        } else { // invalid sequece
            if (err_notifier) {
                err_notifier->invalid_sequence(2, "UTF-16", src_it - 1, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src_it - 1, dst_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DST_SIZE(3);
            dst_it[0] = static_cast<DstCharT>('\xEF');
            dst_it[1] = static_cast<DstCharT>('\xBF');
            dst_it[2] = static_cast<DstCharT>('\xBD');
            dst_it += 3;
        }
    }
    return {src_it, dst_it, reason::completed};
}


template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags) && strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, limit);
    }
    return transcode_size_continue_on_inv_seq_(src, src_end, limit);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    std::ptrdiff_t size = 0;
    for (; src < src_end; ++src) {
        unsigned const ch = detail::cast_u16(*src);
        const auto previous_size = size;
        STRF_IF_LIKELY (ch < 0x80) {
            ++size;
        } else if (ch < 0x800) {
            size += 2;
        } else if (detail::is_surrogate(ch)) {
            if ( strf::detail::is_high_surrogate(ch)
              && src + 1 != src_end
              && strf::detail::is_low_surrogate(*(src + 1)) )
            {
                size += 4;
                ++src;
            } else {
                return {size, src, strf::transcode_stop_reason::invalid_sequence};
            }
        } else {
            size += 3;
        }
        if (size > limit) {
            return {previous_size, src, strf::transcode_stop_reason::insufficient_output_space};
        }
    }
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size_continue_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit ) noexcept
{
    std::ptrdiff_t size = 0;
    for ( ; src < src_end; ++src) {
        if (size >= limit) {
            return {size, src, strf::transcode_stop_reason::insufficient_output_space};
        }
        unsigned const ch = detail::cast_u16(*src);
        STRF_IF_LIKELY (ch < 0x80) {
            ++size;
        } else if (ch < 0x800) {
            size += 2;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src + 1 != src_end
               && strf::detail::is_low_surrogate(*(src + 1)) )
        {
            size += 4;
            ++src;
        } else {
            size += 3;
        }
    }
    return {size, src_end, strf::transcode_stop_reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    if (src >= src_end) {
        return {src, dst, reason::completed};
    }
    if (dst == dst_end) {
        return {src, dst, reason::insufficient_output_space};
    }
    if (strf::with_strict_surrogate_policy(flags)) {
        return unsafe_transcode_strict_surr_(src, src_end, dst, dst_end);
    }
    return unsafe_transcode_lax_surr_(src, src_end, dst, dst_end);
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_lax_surr_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;
    unsigned int ch2 = 0;
    while (src < src_end) {
        const SrcCharT* const seq_begin = src;
        unsigned const ch = detail::cast_u16(*src);
        const auto dst_space = dst_end - dst_it;
        STRF_IF_UNLIKELY (dst_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : 3 );
            if (required_space > dst_space) {
                return {seq_begin, dst_it, reason::insufficient_output_space};
            }
        }
        ++src;
        STRF_IF_LIKELY (ch < 0x80) {
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else if (ch < 0x800) {
            dst_it[0] = static_cast<DstCharT>(0xC0 | (0x1F & (ch >> 6)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 2;
        } else if (strf::detail::not_high_surrogate(ch)
                || src == src_end
                || strf::detail::not_low_surrogate(ch2 = detail::cast_u16(*src))) {
            dst_it[0] = static_cast<DstCharT>(0xE0 | (0x0F & (ch >> 12)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF &  ch ));
            dst_it += 3;
        } else {
            ++src;
            STRF_UTF_CHECK_DST_SIZE(4);
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dst_it[0] = static_cast<DstCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dst_it[3] = static_cast<DstCharT>(0x80 | (0xBF &  codepoint));
            dst_it += 4;
        }
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_strict_surr_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end )
{
    using reason = strf::transcode_stop_reason;
    auto *dst_it = dst;

    while (src < src_end) {
        const SrcCharT* const seq_begin = src;
        unsigned const ch = detail::cast_u16(*src);
        const auto dst_space = dst_end - dst_it;
        STRF_IF_UNLIKELY (dst_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : strf::detail::not_high_surrogate(ch) ? 3 : 4);
            if (required_space > dst_space) {
                return {seq_begin, dst_it, reason::insufficient_output_space};
            }
        }
        ++src;
        STRF_IF_LIKELY (ch < 0x80) {
            *dst_it = static_cast<DstCharT>(ch);
            ++dst_it;
        } else if (ch < 0x800) {
            dst_it[0] = static_cast<DstCharT>(0xC0 | (0x1F & (ch >> 6)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF &  ch));
            dst_it += 2;
        } else if (strf::detail::not_high_surrogate(ch)) {
            dst_it[0] = static_cast<DstCharT>(0xE0 | (0x0F & (ch >> 12)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (ch >> 6)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF &  ch ));
            dst_it += 3;
        } else {
            const unsigned ch2 = detail::cast_u16(*src);
            ++src;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dst_it[0] = static_cast<DstCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dst_it[1] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dst_it[2] = static_cast<DstCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dst_it[3] = static_cast<DstCharT>(0x80 | (0xBF &  codepoint));
            dst_it += 4;
        }
    }
    return {src, dst_it, reason::completed};
}

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DstCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags)) {
        std::ptrdiff_t size = 0;
        while (src < src_end) {
            if (size >= limit) {
                return {size, src, strf::transcode_stop_reason::insufficient_output_space};
            }
            const unsigned ch = detail::cast_u16(*src);
            const bool surrogate = detail::is_high_surrogate(ch);
            const int x = ( ch < 0x80   ? 0x9
                          : ch < 0x800  ? 0xA
                          : surrogate   ? 0x14
                          :               0xB );
            size += x & 7;
            src  += x >> 3;
        }
        return {size, src_end, strf::transcode_stop_reason::completed};
    }
    auto r = transcode_size_continue_on_inv_seq_(src, src_end, limit);
    return {r.ssize, r.src_ptr, r.stop_reason};
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

#undef STRF_UTF_RECYCLE
#undef STRF_UTF_CHECK_DST
#undef STRF_UTF_CHECK_DST_SIZE

#endif  // STRF_DETAIL_UTF_HPP

