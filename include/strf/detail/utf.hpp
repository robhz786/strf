#ifndef STRF_DETAIL_UTF_HPP
#define STRF_DETAIL_UTF_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>
#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

#define STRF_UTF_RECYCLE                                   \
    dest.advance_to(dest_it);                              \
    dest.flush();                                          \
    STRF_IF_UNLIKELY (!dest.good()) {                      \
        return {seq_begin, reason::bad_destination};       \
    }                                                      \
    dest_it = dest.buffer_ptr();                           \
    dest_end = dest.buffer_end();                          \

#define STRF_UTF_CHECK_DEST                                \
    STRF_IF_UNLIKELY (dest_it >= dest_end) {               \
        STRF_UTF_RECYCLE;                                  \
    }

#define STRF_UTF_CHECK_DEST_SIZE(SIZE)                     \
    STRF_IF_UNLIKELY (dest_it + (SIZE) > dest_end) {       \
        STRF_UTF_RECYCLE;                                  \
    }

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

template <typename CharT>
inline STRF_HD CharT* get_initial_dest_end_(strf::transcode_dest<CharT>& dst)
{
    // This function is to be used to set initial value of the `dest_end`
    // variable in the many transcode functions that follow.

    // The purpose is to cause the transcode function to (almost) immediately
    // return strf::stop_reason::bad_destination if anything is to written in
    // dst but dst is in "bad" state.

    // We don't want that to happen however, when the stop reason shall be
    // invalid_sequence or unsupported codepoint

    return dst.good() ? dst.buffer_end() : dst.buffer_ptr();
}

} // namespace detail

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-8");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags);

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags ) noexcept
    {
        return {src_end - src, src_end, unsafe_transcode_size_stop_reason::completed};
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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 1, "Incompatible character type for UTF-8");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-8");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode_lax_surr_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest );

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode_strict_surr_
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 2, "Incompatible character type for UTF-16");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags);

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags ) noexcept
    {
        return {src_end - src, src_end, strf::unsafe_transcode_size_stop_reason::completed};
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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32>
{
public:
    static_assert(sizeof(SrcCharT) == 2, "Incompatible character type for UTF-16");
    static_assert(sizeof(DestCharT) == 4, "Incompatible character type for UTF-32");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

private:

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_non_stop_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size_stop_on_inv_seq_
        ( const SrcCharT* src
        , const SrcCharT* src_end ) noexcept;
};

template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8>
{
public:
    static_assert(sizeof(SrcCharT) == 4, "Incompatible character type for UTF-32");
    static_assert(sizeof(DestCharT) == 1, "Incompatible character type for UTF-1");

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) noexcept;

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

    using src_char_type = SrcCharT;
    using dst_char_type = DestCharT;

    static STRF_HD strf::transcode_result<SrcCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags );

    static STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags );

    static STRF_HD strf::unsafe_transcode_result<SrcCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags )
    {
        return detail::bypass_unsafe_transcode(src, src_end, dest, err_notifier, flags);
    }

    static STRF_HD strf::unsafe_transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags )
    {
        return {src_end - src, src_end, unsafe_transcode_size_stop_reason::completed};
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
        return detail::bypass_unsafe_transcode<SrcCharT, DestCharT>;
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
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    <SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
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
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
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
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, reason::invalid_sequence};
            }
            ch32 = 0xFFFD;
        }

        STRF_UTF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, flags);
    }
    return transcode_size_non_stop_(src, src_end, flags);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;

    using reason = strf::transcode_size_stop_reason;

    unsigned ch0 = 0, ch1 = 0;
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        const auto* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        ++size;
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
                return {size - 1, seq_begin, reason::invalid_sequence};
            }
        }
    }
    return {size, src_end, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end
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
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::unsafe_transcode_stop_reason;
    if (src >= src_end) {
        return {src, reason::completed};
    }
    if (!dest.good()) {
        return {src, reason::bad_destination};
    }
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);

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
        STRF_UTF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags ) noexcept
{
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
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
    return {size, src_end, strf::unsafe_transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    unsigned ch0 = 0, ch1 = 0, ch2 = 0, ch3 = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while(src_it < src_end) {
        ch0 = detail::cast_u8(*src_it);
        const SrcCharT* seq_begin = src_it;
        ++src_it;
        if(ch0 < 0x80) {
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch0);
            ++dest_it;
        } else if(0xC0 == (ch0 & 0xE0)) {
            if ( ch0 > 0xC1
              && src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))) {
                STRF_UTF_CHECK_DEST_SIZE(2);
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
                STRF_UTF_CHECK_DEST_SIZE(3);
                ++src_it;
                dest_it[0] = static_cast<DestCharT>(ch0);
                dest_it[1] = static_cast<DestCharT>(ch1);
                dest_it[2] = static_cast<DestCharT>(ch2);
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (0xE0 == (ch0 & 0xF0)) {
            if (   src_it != src_end
              && is_utf8_continuation(ch1 = detail::cast_u8(*src_it))
              && first_2_of_3_are_valid(ch0, ch1, lax_surr)
              && ++src_it != src_end
              && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_UTF_CHECK_DEST_SIZE(3);
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
            STRF_UTF_CHECK_DEST_SIZE(4);
            ++src_it;
            dest_it[0] = static_cast<DestCharT>(ch0);
            dest_it[1] = static_cast<DestCharT>(ch1);
            dest_it[2] = static_cast<DestCharT>(ch2);
            dest_it[3] = static_cast<DestCharT>(ch3);
            dest_it += 4;
        } else {
            invalid_sequence:
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence
                    (1, "UTF-8", (const void*)seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    if (src < src_end) {
        auto len = static_cast<std::size_t>(src_end - src);
        while (dest.good()) {
            const auto buf_space = dest.buffer_space();
            if (len <= buf_space) {
                detail::copy_n(src, len, dest.buffer_ptr());
                dest.advance(len);
                STRF_ASSERT(src_end == src + len);
                return {src_end, reason::completed};
            }
            auto next_src = src + buf_space;
            while (detail::is_utf8_continuation(*next_src) && next_src > src) {
                --next_src;
            }
            const auto count = next_src - src;
            strf::detail::copy_n(src, count, dest.buffer_ptr());
            src = next_src;
            len -= count;
            dest.advance(count);
            dest.flush();
        };
        return {src, reason::bad_destination};
    }
    return {src, reason::completed};

}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, flags);
    }
    return transcode_size_non_stop_(src, src_end, flags);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::not_utf8_continuation;

    const SrcCharT* const src_begin = src;
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
                return {size, seq_begin, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    }
    const auto size = src - src_begin;
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end
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
    return {size, src_end, strf::transcode_size_stop_reason::completed};
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
    const bool lax_surr = surr_poli == strf::surrogate_policy::lax;
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
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    for(;src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_UTF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (ch < 0x10000) {
            STRF_IF_LIKELY (lax_surr || strf::detail::not_surrogate(ch))
            {
                STRF_UTF_CHECK_DEST_SIZE(3);
                dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
                dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
                dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
                dest_it += 3;
            } else goto invalid_sequence;
        } else if (ch < 0x110000) {
            STRF_UTF_CHECK_DEST_SIZE(4);
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (ch >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 4;
        } else {
            invalid_sequence:
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence(4, "UTF-32", (const void*)src_it, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src_it, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    (void) flags;
    const auto *src_it = src;
    std::ptrdiff_t count = 0;
    const bool non_stop = !strf::with_stop_on_invalid_sequence(flags);
    if ( strf::with_strict_surrogate_policy(flags)) {
        for(;src_it < src_end; ++src_it) {
            auto ch = detail::cast_u32(*src_it);
            STRF_IF_LIKELY (ch < 0xD800 || (0xDFFF < ch && ch < 0x110000)) {
                count += 1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000);
            } else if (non_stop) {
                count += 3;
            } else {
                return {count, src_it, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    } else {
        for(;src_it < src_end; ++src_it) {
            auto ch = detail::cast_u32(*src_it);
            STRF_IF_LIKELY (ch < 0x110000) {
                count += 1 + (ch >= 0x80) + (ch >= 0x800) + (ch >= 0x10000);
            } else if (non_stop) {
                count += 3;
            } else {
                return {count, src_it, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    }
    return {count, src_end, strf::transcode_size_stop_reason::completed};
}


template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::unsafe_transcode_stop_reason;
    if (src >= src_end) {
        return {src, reason::completed};
    }
    if (!dest.good()) {
        return {src, reason::bad_destination};
    }
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    for(;src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);

        const auto dest_space = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : ch < 0x10000 ? 3 : 4);
            if (required_space > dest_space) {
                STRF_UTF_RECYCLE;
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
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags ) noexcept
{
    const auto *src_it = src;
    std::ptrdiff_t size = 0;
    for(;src_it < src_end; ++src_it) {
        auto ch = detail::cast_u32(*src_it);
        size += ( ch < 0x80 ? 1
                : ch < 0x800 ? 2
                : ch < 0x10000 ? 3 : 4 );
    }
    return {size, src_end, strf::unsafe_transcode_size_stop_reason::completed};
}

template <typename CharT>
STRF_HD void
static_charset<CharT, strf::csid_utf8>::write_replacement_char
    ( strf::transcode_dest<CharT>& dest )
{
    dest.ensure(3);
    auto *dest_it = dest.buffer_ptr();
    dest_it[0] = static_cast<CharT>('\xEF');
    dest_it[1] = static_cast<CharT>('\xBF');
    dest_it[2] = static_cast<CharT>('\xBD');
    dest_it += 3;
    dest.advance_to(dest_it);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
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
                dest.advance_to(dest_it);
                if (err_notifier) {
                    err_notifier->invalid_sequence(2, "UTF-16", seq_begin, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, reason::invalid_sequence};
                }
                ch32 = 0xFFFD;
            }
        }
        STRF_UTF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch32);
        ++dest_it;
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags) && strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end);
    }
    return transcode_size_non_stop_(src, src_end);
}
template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
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
    return {count, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
    unsigned ch = 0;
    std::ptrdiff_t count = 0;
    const SrcCharT* src_next = nullptr;
    for(; src < src_end; src = src_next) {
        src_next = src + 1;
        ch = detail::cast_u16(*src);
        ++count;
        if (strf::detail::is_surrogate(ch)) {
            if ( strf::detail::is_high_surrogate(ch)
              && src_next != src_end
              && strf::detail::is_low_surrogate(*src_next)) {
                ++src_next;
            } else {
                return {count - 1, src, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    }
    return {count, src_end, strf::transcode_size_stop_reason::completed};
}


template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags flags)
{
    using reason = strf::unsafe_transcode_stop_reason;

    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
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
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch32);
            ++dest_it;
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
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch32);
            ++dest_it;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf32 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    std::ptrdiff_t size = 0;
    if (strf::with_strict_surrogate_policy(flags)) {
        while (src < src_end) {
            const unsigned ch = detail::cast_u16(*src);
            src += strf::detail::is_high_surrogate(ch) ? 2 : 1;
            ++size;
        }
    } else {
        while (src < src_end) {
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
    return {size, src_end, unsafe_transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
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
                STRF_UTF_CHECK_DEST_SIZE(2);
                dest_it[0] = static_cast<DestCharT>(ch);
                dest_it[1] = static_cast<DestCharT>(ch2);
                dest_it += 2;
                continue;
            }
            if (strf::with_strict_surrogate_policy(flags)){
                dest.advance_to(dest_it);
                if (err_notifier) {
                    err_notifier->invalid_sequence(2, "UTF-16", seq_begin, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
        }
        STRF_UTF_CHECK_DEST;
        *dest_it = static_cast<DestCharT>(ch);
        ++dest_it;
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::transcode_stop_reason;
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    if (src < src_end) {
        auto len = static_cast<std::size_t>(src_end - src);
        while (dest.good()) {
            STRF_ASSERT(src < src_end);
            const auto buf_space = dest.buffer_space();
            if (len <= buf_space) {
                detail::copy_n(src, len, dest.buffer_ptr());
                dest.advance(len);
                STRF_ASSERT(src_end == src + len);
                return {src_end, reason::completed};
            }
            if (buf_space != 0) {
                auto count = buf_space;
                if (detail::is_low_surrogate(src[count]) &&
                    detail::is_high_surrogate(src[count - 1])) {
                    --count;
                }
                strf::detail::copy_n(src, count, dest.buffer_ptr());
                src += count;
                len -= count;
                dest.advance(count);
            }
            dest.flush();
        };
        return {src, reason::bad_destination};
    }
    return {src, reason::completed};
}


template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags) && strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end);
    }
    return transcode_size_non_stop_(src, src_end);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
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
    return {count, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf16 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
    unsigned ch = 0;
    const auto* const src_begin = src;
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
                const auto size = src - src_begin;
                return {size, src, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    }
    const auto size = src - src_begin;
    return {size, src_end, strf::transcode_size_stop_reason::completed};
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
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    for ( ; src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);
        STRF_IF_LIKELY (ch < 0x10000) {
            STRF_IF_LIKELY (lax_surr || strf::detail::not_surrogate(ch))
            {
                STRF_UTF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(ch);
                ++dest_it;
            } else goto invalid_char;
        } else if (ch < 0x110000) {
            STRF_UTF_CHECK_DEST_SIZE(2);
            const auto sub_codepoint = ch - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 | (sub_codepoint >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 | (sub_codepoint &  0x3FF));
            dest_it += 2;
        } else {
            invalid_char:
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence(4, "UTF-32", seq_begin, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DEST;
            *dest_it = 0xFFFD;
            ++dest_it;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    (void) flags;
    std::ptrdiff_t size = 0;
    if (strf::with_stop_on_invalid_sequence(flags)) {
        if (strf::with_strict_surrogate_policy(flags)) {
            for ( ; src < src_end; ++src) {
                unsigned const ch = detail::cast_u32(*src);
                if (detail::is_surrogate(ch) || ch >= 0x110000) {
                    return {size, src, strf::transcode_size_stop_reason::invalid_sequence};
                }
                size += 1 + (0x10000 <= ch);
            }
        } else {
            for ( ; src < src_end; ++src) {
                unsigned const ch = detail::cast_u32(*src);
                if (ch >= 0x110000) {
                    return {size, src, strf::transcode_size_stop_reason::invalid_sequence};
                }
                size += 1 + (0x10000 <= ch);
            }
        }
    } else {
        for ( ; src < src_end; ++src) {
            unsigned const ch = detail::cast_u32(*src);
            size += 1 + (0x10000 <= ch && ch < 0x110000);
        }
    }
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::unsafe_transcode_stop_reason;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    for ( ; src_it < src_end; ++src_it) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u32(*src_it);

        const auto dest_size = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_size < 2) {
            const int required_size = ch < 0x10000 ? 1 : 2;
            if (dest_size < required_size) {
                STRF_UTF_RECYCLE;
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
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags) noexcept
{
    (void) flags;
    std::ptrdiff_t size = 0;
    for ( ; src < src_end; ++src) {
        unsigned const ch = detail::cast_u32(*src);
        size += 1 + (0x10000 <= ch);
    }
    return {size, src_end, strf::unsafe_transcode_size_stop_reason::completed};
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
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    if (strf::with_lax_surrogate_policy(flags)) {
        for (; src < src_end; ++src) {
            const SrcCharT* const seq_begin = src;
            unsigned ch = detail::cast_u32(*src);
            STRF_IF_UNLIKELY (ch >= 0x110000) {
                dest.advance_to(dest_it);
                if (err_notifier) {
                    err_notifier->invalid_sequence(4, "UTF-32", src, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        }
    } else {
        for(; src < src_end; ++src) {
            const SrcCharT* const seq_begin = src;
            unsigned ch = detail::cast_u32(*src);
            STRF_IF_UNLIKELY (ch >= 0x110000 || strf::detail::is_surrogate(ch)) {
                dest.advance_to(dest_it);
                if (err_notifier) {
                    err_notifier->invalid_sequence(4, "UTF-32", src, 1);
                }
                if (strf::with_stop_on_invalid_sequence(flags)) {
                    return {seq_begin, reason::invalid_sequence};
                }
                ch = 0xFFFD;
            }
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        }
    }
    dest.advance_to(dest_it);
    return {src, reason::completed};
}


template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf32, strf::csid_utf32 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags )
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        if (strf::with_strict_surrogate_policy(flags)) {
            for (auto it = src; it < src_end; ++it) {
                if (*it >= 0x110000 || detail::is_surrogate(*it)) {
                    return {it - src, it, transcode_size_stop_reason::invalid_sequence};
                }
            }
        } else {
            for (auto it = src; it < src_end; ++it) {
                if (*it >= 0x110000) {
                    return {it - src, it, transcode_size_stop_reason::invalid_sequence};
                }
            }
        }
    }
    return {src_end - src, src_end, transcode_size_stop_reason::completed};
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
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
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
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    for (;src_it < src_end; ++dest_it) {
        const SrcCharT* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch0 < 0x80) {
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch0);
        } else if (0xC0 == (ch0 & 0xE0)) {
            STRF_IF_LIKELY ( ch0 > 0xC1
                          && src_it != src_end
                          && is_utf8_continuation(ch1 = detail::cast_u8(*src_it)))
            {
                STRF_UTF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(utf8_decode(ch0, ch1));
                ++src_it;
            } else goto invalid_sequence;
        } else if (0xE0 == ch0) {
            STRF_IF_LIKELY ( src_it != src_end
                          && (((ch1 = detail::cast_u8(*src_it)) & 0xE0) == 0xA0)
                          && ++src_it != src_end
                          && is_utf8_continuation(ch2 = detail::cast_u8(*src_it)) )
            {
                STRF_UTF_CHECK_DEST;
                *dest_it = static_cast<DestCharT>(((ch1 & 0x3F) << 6) | (ch2 & 0x3F));
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
                STRF_UTF_CHECK_DEST;
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
            STRF_UTF_CHECK_DEST_SIZE(2);
            x = utf8_decode_last_2_of_4(x, ch2, ch3) - 0x10000;
            dest_it[0] = static_cast<DestCharT>(0xD800 +  (x >> 10));
            dest_it[1] = static_cast<DestCharT>(0xDC00 +  (x & 0x3FF));
            ++dest_it;
            ++src_it;
        } else {
            invalid_sequence:
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence(1, "UTF-8", seq_begin, src_it - seq_begin);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {seq_begin, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DEST;
            *dest_it = 0xFFFD;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}
template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end, flags);
    }
    return transcode_size_non_stop_(src, src_end, flags);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;
    using strf::detail::utf8_decode_first_2_of_3;
    using strf::detail::utf8_decode_first_2_of_4;

    std::ptrdiff_t size = 0;
    unsigned ch0 = 0, ch1 = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src < src_end) {
        const auto* const seq_begin = src;
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
                return {size - 1, seq_begin, strf::transcode_size_stop_reason::invalid_sequence};
            }
        }
    }
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    using strf::detail::utf8_decode;
    using strf::detail::not_surrogate;
    using strf::detail::is_utf8_continuation;
    using strf::detail::first_2_of_3_are_valid;
    using strf::detail::first_2_of_4_are_valid;

    std::ptrdiff_t size = 0;
    unsigned ch0 = 0, ch1 = 0;
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);
    while(src < src_end) {
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
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    using reason = strf::unsafe_transcode_stop_reason;
    using strf::detail::utf8_decode;
    unsigned ch0 = 0;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    for (;src_it < src_end; ++dest_it) {
        const SrcCharT* const seq_begin = src_it;
        ch0 = detail::cast_u8(*src_it);
        ++src_it;
        STRF_UTF_CHECK_DEST;
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
            STRF_UTF_CHECK_DEST_SIZE(2);
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
    return {src_it, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf16 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags ) noexcept
{
    std::ptrdiff_t size = 0;
    const auto *src_it = src;
    while (src_it < src_end) {
        const unsigned ch0 = detail::cast_u8(*src_it);
        const int x = ( ch0 < 0x80 ? 0x09
                      : ch0 < 0xE0 ? 0x0A
                      : ch0 < 0xF0 ? 0x0B
                      :              0x14);
        src_it += x &  7;
        size   += x >> 3;
    }
    return {size, src_end, strf::unsafe_transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags )
{
    using reason = strf::transcode_stop_reason;
    (void) err_notifier;
    const auto *src_it = src;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    const bool lax_surr = strf::with_lax_surrogate_policy(flags);

    while (src_it < src_end) {
        const SrcCharT* const seq_begin = src_it;
        unsigned const ch = detail::cast_u16(*src_it);
        ++src_it;
        STRF_IF_LIKELY (ch < 0x80) {
            STRF_UTF_CHECK_DEST;
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            STRF_UTF_CHECK_DEST_SIZE(2);
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (strf::detail::not_surrogate(ch)) {
            three_bytes:
            STRF_UTF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
            dest_it += 3;
        } else if ( strf::detail::is_high_surrogate(ch)
               && src_it != src_end
               && strf::detail::is_low_surrogate(*src_it) )
        {
            STRF_UTF_CHECK_DEST_SIZE(4);
            const unsigned ch2 = detail::cast_u16(*src_it);
            ++src_it;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  codepoint));
            dest_it += 4;
        } else if (lax_surr) {
            goto three_bytes;
        } else { // invalid sequece
            dest.advance_to(dest_it);
            if (err_notifier) {
                err_notifier->invalid_sequence(2, "UTF-16", src_it - 1, 1);
            }
            if (strf::with_stop_on_invalid_sequence(flags)) {
                return {src_it - 1, reason::invalid_sequence};
            }
            STRF_UTF_CHECK_DEST_SIZE(3);
            dest_it[0] = static_cast<DestCharT>('\xEF');
            dest_it[1] = static_cast<DestCharT>('\xBF');
            dest_it[2] = static_cast<DestCharT>('\xBD');
            dest_it += 3;
        }
    }
    dest.advance_to(dest_it);
    return {src_it, reason::completed};
}




template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags) && strf::with_stop_on_invalid_sequence(flags)) {
        return transcode_size_stop_on_inv_seq_(src, src_end);
    }
    return transcode_size_non_stop_(src, src_end);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size_stop_on_inv_seq_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
    std::ptrdiff_t size = 0;
    for (; src < src_end; ++src) {
        unsigned const ch = detail::cast_u16(*src);
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
                return {size, src, strf::transcode_size_stop_reason::invalid_sequence};
            }
        } else {
            size += 3;
        }
    }
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::transcode_size_non_stop_
    ( const SrcCharT* src
    , const SrcCharT* src_end ) noexcept
{
    std::ptrdiff_t size = 0;
    for ( ; src < src_end; ++src) {
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
    return {size, src_end, strf::transcode_size_stop_reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest
    , strf::transcoding_error_notifier*
    , strf::transcode_flags flags )
{
    using reason = strf::unsafe_transcode_stop_reason;
    if (src >= src_end) {
        return {src, reason::completed};
    }
    if (!dest.good()) {
        return {src, reason::bad_destination};
    }
    if (strf::with_strict_surrogate_policy(flags)) {
        return unsafe_transcode_strict_surr_(src, src_end, dest);
    }
    return unsafe_transcode_lax_surr_(src, src_end, dest);
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_lax_surr_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest )
{
    using reason = strf::unsafe_transcode_stop_reason;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);
    unsigned int ch2 = 0;
    while (src < src_end) {
        const SrcCharT* const seq_begin = src;
        unsigned const ch = detail::cast_u16(*src);
        const auto dest_space = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : 3 );
            if (required_space > dest_space) {
                STRF_UTF_RECYCLE;
            }
        }
        ++src;
        STRF_IF_LIKELY (ch < 0x80) {
            *dest_it = static_cast<DestCharT>(ch);
            ++dest_it;
        } else if (ch < 0x800) {
            dest_it[0] = static_cast<DestCharT>(0xC0 | (0x1F & (ch >> 6)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF &  ch));
            dest_it += 2;
        } else if (strf::detail::not_high_surrogate(ch)
                || src == src_end
                || strf::detail::not_low_surrogate(ch2 = detail::cast_u16(*src))) {
            dest_it[0] = static_cast<DestCharT>(0xE0 | (0x0F & (ch >> 12)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (ch >> 6)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF &  ch ));
            dest_it += 3;
        } else {
            ++src;
            STRF_UTF_CHECK_DEST_SIZE(4);
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  codepoint));
            dest_it += 4;
        }
    }
    dest.advance_to(dest_it);
    return {src, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_strict_surr_
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_dest<DestCharT>& dest )
{
    using reason = strf::unsafe_transcode_stop_reason;
    auto *dest_it = dest.buffer_ptr();
    auto *dest_end = detail::get_initial_dest_end_(dest);

    while (src < src_end) {
        const SrcCharT* const seq_begin = src;
        unsigned const ch = detail::cast_u16(*src);
        const auto dest_space = dest_end - dest_it;
        STRF_IF_UNLIKELY (dest_space < 4) {
            const int required_space = ( ch < 0x80 ? 1
                                       : ch < 0x800 ? 2
                                       : strf::detail::not_high_surrogate(ch) ? 3 : 4);
            if (required_space > dest_space) {
                STRF_UTF_RECYCLE;
            }
        }
        ++src;
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
            const unsigned ch2 = detail::cast_u16(*src);
            ++src;
            const unsigned codepoint = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
            dest_it[0] = static_cast<DestCharT>(0xF0 | (0x07 & (codepoint >> 18)));
            dest_it[1] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 12)));
            dest_it[2] = static_cast<DestCharT>(0x80 | (0xBF & (codepoint >> 6)));
            dest_it[3] = static_cast<DestCharT>(0x80 | (0xBF &  codepoint));
            dest_it += 4;
        }
    }
    dest.advance_to(dest_it);
    return {src, reason::completed};
}

template <typename SrcCharT, typename DestCharT>
STRF_HD strf::unsafe_transcode_size_result<SrcCharT> strf::static_transcoder
    < SrcCharT, DestCharT, strf::csid_utf16, strf::csid_utf8 >::unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::transcode_flags flags ) noexcept
{
    if (strf::with_strict_surrogate_policy(flags)) {
        std::ptrdiff_t size = 0;
        while (src < src_end) {
            const unsigned ch = detail::cast_u16(*src);
            const bool surrogate = detail::is_high_surrogate(ch);
            const int x = ( ch < 0x80   ? 0x9
                          : ch < 0x800  ? 0xA
                          : surrogate   ? 0x14
                          :               0xB );
            size += x & 7;
            src  += x >> 3;
        }
        return {size, src_end, strf::unsafe_transcode_size_stop_reason::completed};
    }
    auto r = transcode_size_non_stop_(src, src_end);
    return {r.size, r.ptr, r.stop_reason};
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
#undef STRF_UTF_CHECK_DEST
#undef STRF_UTF_CHECK_DEST_SIZE

#endif  // STRF_DETAIL_UTF_HPP

