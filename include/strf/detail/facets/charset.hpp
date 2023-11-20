#ifndef STRF_DETAIL_FACETS_CHARSET_HPP
#define STRF_DETAIL_FACETS_CHARSET_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/output_buffer_functions.hpp>

namespace strf {

enum class transcode_flags {
    none = 0,
    surrogate_policy = 1,
    lax_surrogate_policy = 1,
    stop_on_invalid_sequence = 1 << 1,
    stop_on_unsupported_codepoint = 1 << 2
};

STRF_HD constexpr transcode_flags operator|(transcode_flags a, transcode_flags b)
{
    return static_cast<transcode_flags>
        (static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

STRF_HD constexpr transcode_flags operator&(transcode_flags a, transcode_flags b)
{
    return static_cast<transcode_flags>
        (static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

STRF_HD constexpr transcode_flags& operator |=(transcode_flags& a, transcode_flags b)
{
    return a = a | b;
}

STRF_HD constexpr transcode_flags& operator &=(transcode_flags& a, transcode_flags b)
{
    return a = a & b;
}

STRF_HD constexpr bool with_strict_surrogate_policy(transcode_flags f)
{
    return transcode_flags::none == (f & transcode_flags::lax_surrogate_policy);
}
STRF_HD constexpr bool with_lax_surrogate_policy(transcode_flags f)
{
    return transcode_flags::none != (f & transcode_flags::lax_surrogate_policy);
}
STRF_HD constexpr bool with_stop_on_invalid_sequence(transcode_flags f)
{
    return transcode_flags::none != (f & transcode_flags::stop_on_invalid_sequence);
}
STRF_HD constexpr bool with_stop_on_unsupported_codepoint(transcode_flags f)
{
    return transcode_flags::none != (f & transcode_flags::stop_on_unsupported_codepoint);
}

struct transcoding_error_notifier_c;

class transcoding_error_notifier {
public:
    virtual STRF_HD ~transcoding_error_notifier() STRF_DEFAULT_IMPL;

    transcoding_error_notifier() = default;
    transcoding_error_notifier(const transcoding_error_notifier&) = default;
    transcoding_error_notifier(transcoding_error_notifier&&) = default;
    transcoding_error_notifier& operator=(const transcoding_error_notifier&) = default;
    transcoding_error_notifier& operator=(transcoding_error_notifier&&) = default;

    virtual STRF_HD void invalid_sequence
        ( int code_unit_size
        , const char* charset_name
        , const void* sequence_ptr
        , std::ptrdiff_t code_units_count )
    {
        (void) code_unit_size;
        (void) charset_name;
        (void) sequence_ptr;
        (void) code_units_count;
    }

    virtual STRF_HD void unsupported_codepoint
        ( const char* charset_name
        , unsigned codepoint )
    {
        (void) charset_name;
        (void) codepoint;
    }
};

struct transcoding_error_notifier_nullptr {
    using category = transcoding_error_notifier_c;

    constexpr STRF_HD transcoding_error_notifier* get() const { return nullptr; }
};

struct transcoding_error_notifier_ptr {
    using category = transcoding_error_notifier_c;

    constexpr transcoding_error_notifier_ptr() noexcept = default;

    STRF_HD constexpr explicit transcoding_error_notifier_ptr(transcoding_error_notifier* p) noexcept
        : ptr(p)
    {
    }

    STRF_HD constexpr bool operator==(const transcoding_error_notifier_ptr& other) const {
        return ptr == other.ptr;
    }

    constexpr STRF_HD transcoding_error_notifier* get() const { return ptr; }

    transcoding_error_notifier* ptr = nullptr;
};

struct transcoding_error_notifier_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::transcoding_error_notifier_nullptr get_default() noexcept
    {
        return {};
    }
};

enum class charset_id : std::uint32_t{};

// generated at https://www.random.org/bytes/
constexpr strf::charset_id csid_ascii        = (strf::charset_id) 0x9dea526b;
constexpr strf::charset_id csid_utf8         = (strf::charset_id) 0x04650346;
constexpr strf::charset_id csid_utf16        = (strf::charset_id) 0x0439cb08;
constexpr strf::charset_id csid_utf32        = (strf::charset_id) 0x67be80a2;
constexpr strf::charset_id csid_iso_8859_1   = (strf::charset_id) 0xcf00a401;
constexpr strf::charset_id csid_iso_8859_2   = (strf::charset_id) 0xcf00a402;
constexpr strf::charset_id csid_iso_8859_3   = (strf::charset_id) 0xcf00a403;
constexpr strf::charset_id csid_iso_8859_4   = (strf::charset_id) 0xcf00a404;
constexpr strf::charset_id csid_iso_8859_5   = (strf::charset_id) 0xcf00a405;
constexpr strf::charset_id csid_iso_8859_6   = (strf::charset_id) 0xcf00a406;
constexpr strf::charset_id csid_iso_8859_7   = (strf::charset_id) 0xcf00a407;
constexpr strf::charset_id csid_iso_8859_8   = (strf::charset_id) 0xcf00a408;
constexpr strf::charset_id csid_iso_8859_9   = (strf::charset_id) 0xcf00a409;
constexpr strf::charset_id csid_iso_8859_10  = (strf::charset_id) 0xcf00a410;
constexpr strf::charset_id csid_iso_8859_11  = (strf::charset_id) 0xcf00a411;
constexpr strf::charset_id csid_iso_8859_13  = (strf::charset_id) 0xcf00a413;
constexpr strf::charset_id csid_iso_8859_14  = (strf::charset_id) 0xcf00a414;
constexpr strf::charset_id csid_iso_8859_15  = (strf::charset_id) 0xcf00a415;
constexpr strf::charset_id csid_iso_8859_16  = (strf::charset_id) 0xcf00a416;
constexpr strf::charset_id csid_windows_1250 = (strf::charset_id) 0x5cff7250;
constexpr strf::charset_id csid_windows_1251 = (strf::charset_id) 0x5cff7251;
constexpr strf::charset_id csid_windows_1252 = (strf::charset_id) 0x5cff7252;
constexpr strf::charset_id csid_windows_1253 = (strf::charset_id) 0x5cff7253;
constexpr strf::charset_id csid_windows_1254 = (strf::charset_id) 0x5cff7254;
constexpr strf::charset_id csid_windows_1255 = (strf::charset_id) 0x5cff7255;
constexpr strf::charset_id csid_windows_1256 = (strf::charset_id) 0x5cff7256;
constexpr strf::charset_id csid_windows_1257 = (strf::charset_id) 0x5cff7257;
constexpr strf::charset_id csid_windows_1258 = (strf::charset_id) 0x5cff7258;

namespace detail {

template <std::size_t> struct csid_utf_impl;
template <> struct csid_utf_impl<1>
{
    constexpr static strf::charset_id csid = csid_utf8;
};
template <> struct csid_utf_impl<2>
{
    constexpr static strf::charset_id csid = csid_utf16;
};
template <> struct csid_utf_impl<4>
{
    constexpr static strf::charset_id csid = csid_utf32;
};

} // namespace detail

template <typename CharT>
constexpr STRF_HD strf::charset_id get_csid_utf() noexcept
{
    return strf::detail::csid_utf_impl<sizeof(CharT)>::csid;
}

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename CharT>
constexpr strf::charset_id csid_utf = strf::get_csid_utf<CharT>();

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename CharT>
struct charset_c;

template <typename CharT, strf::charset_id>
class static_charset;

template < typename SrcCharT, typename DstCharT
         , strf::charset_id Src, strf::charset_id Dst>
class static_transcoder;

template <typename SrcCharT, typename DstCharT>
class dynamic_transcoder;

constexpr int invalid_char_len = -1;

enum class transcode_stop_reason : std::uint32_t {
    completed,
    insufficient_output_space,
    unsupported_codepoint,
    invalid_sequence,
};

template <typename SrcCharT, typename DstCharT>
struct transcode_result
{
    const SrcCharT* src_ptr;
    DstCharT* dst_ptr;
    transcode_stop_reason stop_reason;
};

template <typename SrcCharT>
struct transcode_size_result
{
    std::ptrdiff_t ssize;
    const SrcCharT* src_ptr;
    transcode_stop_reason stop_reason;
};

constexpr auto ssize_max = std::numeric_limits<std::ptrdiff_t>::max();

template <typename CharT>
using transcode_dst = strf::output_buffer<CharT, 3>;

constexpr std::ptrdiff_t transcode_dst_min_buffer_size = 8;

template <typename SrcCharT, typename DstCharT>
using transcode_f = strf::transcode_result<SrcCharT, DstCharT> (*)
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier
    , strf::transcode_flags flags );

template <typename SrcCharT, typename DstCharT>
using unsafe_transcode_f = strf::transcode_f<SrcCharT, DstCharT>;

template <typename SrcCharT>
using transcode_size_f = transcode_size_result<SrcCharT> (*)
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags );

template <typename SrcCharT>
using unsafe_transcode_size_f = transcode_size_f<SrcCharT>;

template <typename CharT>
using write_replacement_char_f = void (*)
    ( strf::transcode_dst<CharT>& );

// assume transcode_flags::lax_surrogate_policy
using validate_f = int (*)(char32_t ch);

// assume transcode_flags::lax_surrogate_policy
using encoded_char_size_f = int (*)(char32_t ch);

// assume transcode_flags::lax_surrogate_policy
template <typename CharT>
using encode_char_f = CharT*(*)
    ( CharT* dst, char32_t ch );

template <typename CharT>
using encode_fill_f = void (*)
    ( strf::transcode_dst<CharT>&, std::ptrdiff_t count, char32_t ch );

namespace detail {

template <typename CharT>
void trivial_fill_f
    ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch )
{
    // same as strf::detail::write_fill<CharT>
    auto narrow_ch = static_cast<CharT>(ch);
    STRF_IF_LIKELY (count <= dst.buffer_sspace()) {
        strf::detail::str_fill_n<CharT>(dst.buffer_ptr(), count, narrow_ch);
        dst.advance(count);
    } else {
        write_fill_continuation<CharT>(dst, count, narrow_ch);
    }
}

} // namespace detail

template <typename CharT>
struct count_codepoints_result {
    std::ptrdiff_t count;
    const CharT* ptr;
};

template <typename CharT>
using count_codepoints_fast_f = strf::count_codepoints_result<CharT> (*)
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count );

template <typename CharT>
using count_codepoints_f = strf::count_codepoints_result<CharT> (*)
    ( const CharT* src
    , const CharT* src_end
    , std::ptrdiff_t max_count );

template <typename CharT>
using codepoints_fast_count_f
    STRF_DEPRECATED_MSG("codepoints_fast_count_f was renamed to count_codepoints_fast_f")
    = count_codepoints_fast_f<CharT>;

template <typename CharT>
using codepoints_robust_count_f
    STRF_DEPRECATED_MSG("codepoints_robust_count_f was renamed to count_codepoints_f")
     = count_codepoints_f<CharT>;

template <typename CharT>
using decode_unit_f = char32_t (*)
    ( CharT );

template <typename SrcCharT, typename DstCharT>
using find_transcoder_f =
    strf::dynamic_transcoder<SrcCharT, DstCharT> (*)
    ( strf::charset_id );

template <typename SrcCharT, typename DstCharT>
class dynamic_transcoder
{
public:
    using src_code_unit = SrcCharT;
    using dst_code_unit = DstCharT;

    template <strf::charset_id SrcId, strf::charset_id DstId>
    constexpr explicit STRF_HD dynamic_transcoder
        ( strf::static_transcoder<SrcCharT, DstCharT, SrcId, DstId> t ) noexcept
        : transcode_func_(t.transcode_func())
        , transcode_size_func_(t.transcode_size_func())
        , unsafe_transcode_func_(t.unsafe_transcode_func())
        , unsafe_transcode_size_func_(t.unsafe_transcode_size_func())
    {
    }

    constexpr STRF_HD dynamic_transcoder() noexcept
        : transcode_func_(nullptr)
        , transcode_size_func_(nullptr)
        , unsafe_transcode_func_(nullptr)
        , unsafe_transcode_size_func_(nullptr)
    {
    }

    STRF_HD strf::transcode_result<SrcCharT, DstCharT> transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags ) const
    {
        return transcode_func_(src, src_end, dst, dst_end, err_notifier, flags);
    }

    STRF_HD strf::transcode_result<SrcCharT, DstCharT> unsafe_transcode
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , DstCharT* dst
        , DstCharT* dst_end
        , strf::transcoding_error_notifier* err_notifier
        , strf::transcode_flags flags ) const
    {
        return unsafe_transcode_func_(src, src_end, dst, dst_end, err_notifier, flags);
    }

    STRF_HD strf::transcode_size_result<SrcCharT> transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) const
    {
        return transcode_size_func_(src, src_end, flags);
    }

    STRF_HD strf::transcode_size_result<SrcCharT> unsafe_transcode_size
        ( const SrcCharT* src
        , const SrcCharT* src_end
        , strf::transcode_flags flags ) const
    {
        return unsafe_transcode_size_func_(src, src_end, flags);
    }

    constexpr STRF_HD strf::transcode_f<SrcCharT, DstCharT>
    transcode_func() const noexcept
    {
        return transcode_func_;
    }

    constexpr STRF_HD strf::transcode_size_f<SrcCharT>
    transcode_size_func() const noexcept
    {
        return transcode_size_func_;
    }

    constexpr STRF_HD strf::unsafe_transcode_f<SrcCharT, DstCharT>
    unsafe_transcode_func() const noexcept
    {
        return unsafe_transcode_func_;
    }

    constexpr STRF_HD strf::unsafe_transcode_size_f<SrcCharT>
    unsafe_transcode_size_func() const noexcept
    {
        return unsafe_transcode_size_func_;
    }

private:

    strf::transcode_f<SrcCharT, DstCharT> transcode_func_;
    strf::transcode_size_f<SrcCharT> transcode_size_func_;
    strf::unsafe_transcode_f<SrcCharT, DstCharT> unsafe_transcode_func_;
    strf::unsafe_transcode_size_f<SrcCharT> unsafe_transcode_size_func_;
};

template <typename CharT>
struct dynamic_charset_data
{
    constexpr STRF_HD dynamic_charset_data
        ( const char* name_
        , strf::charset_id id_
        , char32_t replacement_char_
        , int replacement_char_size_
        , strf::validate_f validate_func_
        , strf::encoded_char_size_f encoded_char_size_func_
        , strf::encode_char_f<CharT> encode_char_func_
        , strf::encode_fill_f<CharT> encode_fill_func_
        , strf::count_codepoints_fast_f<CharT> count_codepoints_fast_func_
        , strf::count_codepoints_f<CharT> count_codepoints_func_
        , strf::write_replacement_char_f<CharT> write_replacement_char_func_
        , strf::decode_unit_f<CharT> decode_unit_func_
        , strf::dynamic_transcoder<CharT, CharT> sanitizer_
        , strf::dynamic_transcoder<char32_t, CharT> from_u32_
        , strf::dynamic_transcoder<CharT, char32_t> to_u32_
        , strf::find_transcoder_f<wchar_t, CharT> find_transcoder_from_wchar_ = nullptr
        , strf::find_transcoder_f<CharT, wchar_t> find_transcoder_to_wchar_ = nullptr
        , strf::find_transcoder_f<char16_t, CharT> find_transcoder_from_char16_ = nullptr
        , strf::find_transcoder_f<CharT, char16_t> find_transcoder_to_char16_ = nullptr
        , strf::find_transcoder_f<char, CharT> find_transcoder_from_char_ = nullptr
        , strf::find_transcoder_f<CharT, char> find_transcoder_to_char_ = nullptr

#if defined (__cpp_char8_t)
        , strf::find_transcoder_f<char8_t, CharT> find_transcoder_from_char8_ = nullptr
        , strf::find_transcoder_f<CharT, char8_t> find_transcoder_to_char8_ = nullptr
#else
        , const void* find_transcoder_from_char8_ = nullptr
        , const void* find_transcoder_to_char8_ = nullptr
#endif
        ) noexcept
        : name(name_)
        , id(id_)
        , replacement_char(replacement_char_)
        , replacement_char_size(replacement_char_size_)
        , validate_func(validate_func_)
        , encoded_char_size_func(encoded_char_size_func_)
        , encode_char_func(encode_char_func_)
        , encode_fill_func(encode_fill_func_)
        , count_codepoints_fast_func(count_codepoints_fast_func_)
        , count_codepoints_func(count_codepoints_func_)
        , write_replacement_char_func(write_replacement_char_func_)
        , decode_unit_func(decode_unit_func_)
        , sanitizer(sanitizer_)
        , from_u32(from_u32_)
        , to_u32(to_u32_)
        , find_transcoder_from_wchar(find_transcoder_from_wchar_)
        , find_transcoder_to_wchar(find_transcoder_to_wchar_)
        , find_transcoder_from_char16(find_transcoder_from_char16_)
        , find_transcoder_to_char16(find_transcoder_to_char16_)
        , find_transcoder_from_char(find_transcoder_from_char_)
        , find_transcoder_to_char(find_transcoder_to_char_)
        , find_transcoder_from_char8(find_transcoder_from_char8_)
        , find_transcoder_to_char8(find_transcoder_to_char8_)
    {
    }

    const char* name;
    strf::charset_id id;
    char32_t replacement_char;
    int replacement_char_size;
    strf::validate_f validate_func;
    strf::encoded_char_size_f encoded_char_size_func;
    strf::encode_char_f<CharT> encode_char_func;
    strf::encode_fill_f<CharT> encode_fill_func;
    strf::count_codepoints_fast_f<CharT> count_codepoints_fast_func;
    strf::count_codepoints_f<CharT> count_codepoints_func;
    strf::write_replacement_char_f<CharT> write_replacement_char_func;
    strf::decode_unit_f<CharT> decode_unit_func;

    strf::dynamic_transcoder<CharT, CharT> sanitizer;
    strf::dynamic_transcoder<char32_t, CharT> from_u32;
    strf::dynamic_transcoder<CharT, char32_t> to_u32;

    strf::find_transcoder_f<wchar_t, CharT> find_transcoder_from_wchar = nullptr;
    strf::find_transcoder_f<CharT, wchar_t> find_transcoder_to_wchar = nullptr;

    strf::find_transcoder_f<char16_t, CharT> find_transcoder_from_char16 = nullptr;
    strf::find_transcoder_f<CharT, char16_t> find_transcoder_to_char16 = nullptr;

    strf::find_transcoder_f<char, CharT> find_transcoder_from_char = nullptr;
    strf::find_transcoder_f<CharT, char> find_transcoder_to_char = nullptr;

#if defined (__cpp_char8_t)
    strf::find_transcoder_f<char8_t, CharT> find_transcoder_from_char8 = nullptr;
    strf::find_transcoder_f<CharT, char8_t> find_transcoder_to_char8 = nullptr;

#else
    const void* find_transcoder_from_char8 = nullptr;
    const void* find_transcoder_to_char8 = nullptr;

#endif
};

template <typename CharT>
class dynamic_charset
{
public:

    using code_unit = CharT;
    using char_type STRF_DEPRECATED = CharT;

    STRF_HD explicit dynamic_charset
        ( const strf::dynamic_charset_data<CharT>& data )
        : data_(&data)
    {
    }

    template <strf::charset_id CharsetID>
    explicit dynamic_charset(strf::static_charset<CharT, CharsetID> scs)
        : dynamic_charset(scs.to_dynamic())
    {
    }

    STRF_HD bool operator==(const dynamic_charset& other) const noexcept
    {
        return data_->id == other.data_->id;
    }
    STRF_HD bool operator!=(const dynamic_charset& other) const noexcept
    {
        return data_->id != other.data_->id;
    }

    STRF_HD void swap(dynamic_charset& other) noexcept
    {
        auto tmp = data_;
        data_ = other.data_;
        other.data_ = tmp;
    }
    STRF_HD dynamic_charset<CharT> to_dynamic() const noexcept
    {
        return *this;
    }
    STRF_HD const char* name() const noexcept
    {
        return data_->name;
    };
    constexpr STRF_HD strf::charset_id id() const noexcept
    {
        return data_->id;
    }
    constexpr STRF_HD char32_t replacement_char() const noexcept
    {
        return data_->replacement_char;
    }
    constexpr STRF_HD int replacement_char_size() const noexcept
    {
        return data_->replacement_char_size;
    }
    constexpr STRF_HD int validate(char32_t ch) const // noexcept
    {
        return data_->validate_func(ch);
    }
    constexpr STRF_HD int encoded_char_size(char32_t ch) const // noexcept
    {
        return data_->encoded_char_size_func(ch);
    }
    STRF_HD code_unit* encode_char(code_unit* dst, char32_t ch) const // noexcept
    {
        return data_->encode_char_func(dst, ch);
    }
    STRF_HD void encode_fill
        ( strf::transcode_dst<CharT>& dst, std::ptrdiff_t count, char32_t ch ) const
    {
        data_->encode_fill_func(dst, count, ch);
    }
    STRF_HD strf::count_codepoints_result<code_unit> count_codepoints_fast
        ( const code_unit* src, const code_unit* src_end, std::ptrdiff_t max_count ) const
    {
        return data_->count_codepoints_fast_func(src, src_end, max_count);
    }
    STRF_HD strf::count_codepoints_result<code_unit> count_codepoints
        ( const code_unit* src, const code_unit* src_end
        , std::ptrdiff_t max_count ) const
    {
        return data_->count_codepoints_func(src, src_end, max_count);
    }
    STRF_HD void write_replacement_char(strf::transcode_dst<CharT>& dst) const
    {
        data_->write_replacement_char_func(dst);
    }
    STRF_HD char32_t decode_unit(code_unit ch) const
    {
        return data_->decode_unit_func(ch);
    }
    STRF_HD strf::encode_char_f<CharT> encode_char_func() const noexcept
    {
        return data_->encode_char_func;
    }
    STRF_HD strf::encode_fill_f<CharT> encode_fill_func() const noexcept
    {
        return data_->encode_fill_func;
    }
    STRF_HD strf::validate_f validate_func() const noexcept
    {
        return data_->validate_func;
    }
    STRF_HD strf::write_replacement_char_f<CharT>
    write_replacement_char_func() const noexcept
    {
        return data_->write_replacement_char_func;
    }
    STRF_HD strf::dynamic_transcoder<char32_t, CharT> from_u32() const
    {
        return data_->from_u32;
    }
    STRF_HD strf::dynamic_transcoder<CharT, char32_t> to_u32() const
    {
        return data_->to_u32;
    }
    STRF_HD strf::dynamic_transcoder<CharT, CharT> sanitizer() const
    {
        return data_->sanitizer;
    }
    STRF_HD strf::dynamic_transcoder<CharT, wchar_t> find_transcoder_to
        ( strf::tag<wchar_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_to_wchar) {
            return data_->find_transcoder_to_wchar(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<wchar_t, CharT> find_transcoder_from
        ( strf::tag<wchar_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_from_wchar) {
            return data_->find_transcoder_from_wchar(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<CharT, char16_t> find_transcoder_to
        ( strf::tag<char16_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_to_char16) {
            return data_->find_transcoder_to_char16(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<char16_t, CharT> find_transcoder_from
        ( strf::tag<char16_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_from_char16) {
            return data_->find_transcoder_from_char16(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<CharT, char> find_transcoder_to
        ( strf::tag<char>, strf::charset_id id) const
    {
        if (data_->find_transcoder_to_char) {
            return data_->find_transcoder_to_char(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<char, CharT> find_transcoder_from
        ( strf::tag<char>, strf::charset_id id) const
    {
        if (data_->find_transcoder_from_char) {
            return data_->find_transcoder_from_char(id);
        }
        return {};
    }

#if defined (__cpp_char8_t)

    STRF_HD strf::dynamic_transcoder<CharT, char8_t> find_transcoder_to
        ( strf::tag<char8_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_to_char8) {
            return data_->find_transcoder_to_char8(id);
        }
        return {};
    }
    STRF_HD strf::dynamic_transcoder<char8_t, CharT> find_transcoder_from
        ( strf::tag<char8_t>, strf::charset_id id) const
    {
        if (data_->find_transcoder_from_char8) {
            return data_->find_transcoder_from_char8(id);
        }
        return {};
    }

#endif // defined (__cpp_char8_t)

private:
    const strf::dynamic_charset_data<CharT>* data_;
};

template <typename CharT>
struct charset_c;

namespace detail {

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_result<SrcCharT, DstCharT> bypass_unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier*
    , strf::transcode_flags )
{
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");
    using reason = strf::transcode_stop_reason;
    if (src < src_end) {
        auto len = src_end - src;
        const auto dst_space = detail::zero_if_negative(dst_end - dst);
        if (len <= dst_space) {
            detail::copy_n(src, detail::cast_unsigned(len), dst);
            STRF_ASSERT(src_end == src + len);
            return {src_end, dst + len, reason::completed};
        }
        strf::detail::copy_n(src, detail::cast_unsigned(dst_space), dst);
        return {src + dst_space, dst + dst_space, reason::insufficient_output_space};
    }
    return {src, dst, reason::completed};
}

} // namespace detail
} // namespace strf

#include <strf/detail/utf.hpp>

namespace strf {

template <typename CharT>
struct charset_c
{
    static constexpr bool constrainable = false;
    static constexpr STRF_HD strf::utf_t<CharT> get_default() noexcept
    {
        return {};
    }
};

template <typename Facet>
struct facet_traits;

template <typename CharT, strf::charset_id CSId>
struct facet_traits<strf::static_charset<CharT, CSId>>
{
    using category = strf::charset_c<CharT>;
};

template <typename CharT>
struct facet_traits<strf::dynamic_charset<CharT>>
{
    using category = strf::charset_c<CharT>;
};

namespace detail {

template <typename SrcCharset, typename DstCharset>
class has_static_transcoder_impl
{
    using src_code_unit = typename SrcCharset::code_unit;
    using dst_code_unit = typename DstCharset::code_unit;

    template <strf::charset_id SrcId, strf::charset_id DstId>
    static
    decltype( strf::static_transcoder
                < src_code_unit, dst_code_unit, SrcId, DstId >()
            , std::true_type() )
    test_( strf::static_charset<src_code_unit, SrcId>*
         , strf::static_charset<dst_code_unit, DstId>* )
    {
        return {};
    }

    static std::false_type test_(...)
    {
        return {};
    }

    using result_ = decltype(test_((SrcCharset*)nullptr, (DstCharset*)nullptr));

public:

    static constexpr bool value = result_::value;
};

template <typename SrcCharset, typename DstCharset>
constexpr STRF_HD bool has_static_transcoder()
{
    return has_static_transcoder_impl<SrcCharset, DstCharset>::value;
}

template <typename SrcCharT, typename DstCharset>
class has_find_transcoder_from_impl
{
    template < typename S
             , typename D
             , typename = decltype
                 ( std::declval<const D&>().find_transcoder_from
                   (strf::tag<S>{}, strf::csid_utf8) ) >
    static std::true_type test(strf::tag<S> stag, const D* d);

    template <typename S, typename D>
    static std::false_type test(...);

public:

    static constexpr bool value
    = decltype(test<SrcCharT, DstCharset>(strf::tag<SrcCharT>(), nullptr))::value;
};

template <typename DstCharT, typename SrcCharset>
class has_find_transcoder_to_impl
{
    template < typename D
             , typename S
             , typename = decltype
                 ( std::declval<const S&>().find_transcoder_from
                   (strf::tag<D>(), strf::csid_utf8) ) >
    static std::true_type test(strf::tag<D> dtag, const S* s);

    template <typename D, typename S>
    static std::false_type test(...);

public:
    static constexpr bool value
    = decltype(test<DstCharT, SrcCharset>(strf::tag<DstCharT>(), nullptr))::value;
};

template <typename DstCharT, typename SrcCharset>
STRF_HD constexpr bool has_find_transcoder_to()
{
    return has_find_transcoder_to_impl<DstCharT, SrcCharset>::value;
}

template <typename SrcCharT, typename DstCharset>
STRF_HD constexpr bool has_find_transcoder_from()
{
    return has_find_transcoder_from_impl<SrcCharT, DstCharset>::value;
}

template <bool HasFindTo, bool HasFindFrom, typename Transcoder>
struct transcoder_finder_2;

template <typename Transcoder>
struct transcoder_finder_2<true, true, Transcoder>
{
private:

    template < typename SrcCharset, typename DstCharset, typename Transc1
             , typename SrcTag = strf::tag<typename SrcCharset::code_unit> >
    constexpr static STRF_HD Transcoder do_find
        ( SrcCharset src_cs
        , DstCharset dst_cs
        , Transc1 t )
    {
        return ( t.transcode_func() != nullptr
               ? t
               : dst_cs.find_transcoder_from(SrcTag{}, src_cs.id()) );
    }

public:

    template < typename SrcCharset, typename DstCharset
             , typename DstTag = strf::tag<typename DstCharset::code_unit> >
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DstCharset dst_cs)
    {
        return do_find(src_cs, dst_cs, src_cs.find_transcoder_to(DstTag{}, dst_cs.id()));
    }
};

template <typename Transcoder>
struct transcoder_finder_2<true, false, Transcoder>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DstCharset dst_cs)
    {
        return src_cs.find_transcoder_to
            ( strf::tag<typename DstCharset::code_unit>{}
            , dst_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, true, Transcoder>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DstCharset dst_cs)
    {
        return dst_cs.find_transcoder_from
            ( strf::tag<typename SrcCharset::code_unit>{}
            , src_cs.id() );
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, false, Transcoder>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset, DstCharset)
    {
        return {};
    }
};

template < bool HasStaticTranscoder
         , typename SrcCharT
         , typename DstCharT >
struct transcoder_finder;

template <typename SrcCharT, typename DstCharT >
struct transcoder_finder<true, SrcCharT, DstCharT>
{
    template < strf::charset_id SrcId, strf::charset_id DstId>
    constexpr static STRF_HD
    strf::static_transcoder<SrcCharT, DstCharT, SrcId, DstId> find
        ( strf::static_charset<SrcCharT, SrcId>
        , strf::static_charset<DstCharT, DstId> ) noexcept
    {
        return {};
    }
};

template <>
struct transcoder_finder<false, char32_t, char32_t>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD strf::utf32_to_utf32<char32_t, char32_t> find
        (SrcCharset, DstCharset )
    {
        return {};
    }
};

template <typename SrcCharT>
struct transcoder_finder<false, SrcCharT, char32_t>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD decltype(auto) find(SrcCharset src_cs, DstCharset)
    {
        return src_cs.to_u32();
    }
};

template <typename DstCharT>
struct transcoder_finder<false, char32_t, DstCharT>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD decltype(auto) find(SrcCharset, DstCharset dst_cs) noexcept
    {
        return dst_cs.from_u32();
    }
};

template <typename CharT>
struct transcoder_finder<false, CharT, CharT>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD
    strf::dynamic_transcoder<CharT, CharT>
    find(SrcCharset src_cs, DstCharset dst_cs )
    {
        return ( src_cs.id() == dst_cs.id()
               ? strf::dynamic_transcoder<CharT, CharT>{ src_cs.sanitizer() }
               : strf::detail::transcoder_finder_2
                   < strf::detail::has_find_transcoder_to<CharT, SrcCharset>()
                   , strf::detail::has_find_transcoder_from<CharT, DstCharset>()
                   , strf::dynamic_transcoder<CharT, CharT> >
                   ::find(src_cs, dst_cs) );
    }
};

template <typename SrcCharT, typename DstCharT >
struct transcoder_finder<false, SrcCharT, DstCharT>
{
    template <typename SrcCharset, typename DstCharset>
    constexpr static STRF_HD
    strf::dynamic_transcoder<SrcCharT, DstCharT>
    find(SrcCharset src_cs, DstCharset dst_cs )
    {
        return strf::detail::transcoder_finder_2
            < strf::detail::has_find_transcoder_to<DstCharT, SrcCharset>()
            , strf::detail::has_find_transcoder_from<SrcCharT, DstCharset>()
            , strf::dynamic_transcoder<SrcCharT, DstCharT> >
            ::find(src_cs, dst_cs);
    }
};

} // namespace detail

template < typename SrcCharset
         , typename DstCharset
         , typename Finder =
             detail::transcoder_finder
                 < strf::detail::has_static_transcoder<SrcCharset, DstCharset>()
                 , typename SrcCharset::code_unit
                 , typename DstCharset::code_unit > >
constexpr STRF_HD decltype(auto) find_transcoder(SrcCharset src_cs, DstCharset dst_cs)
{
    return Finder::find(src_cs, dst_cs);
}

namespace detail {

template <typename T>
struct is_static_charset: std::false_type
{};
template <typename Ch, strf::charset_id CsId>
struct is_static_charset<strf::static_charset<Ch, CsId>>: std::true_type
{};

template <typename T>
struct is_dynamic_charset: std::false_type
{};

template <typename Ch>
struct is_dynamic_charset<strf::dynamic_charset<Ch>>: std::true_type
{};

template <typename T>
struct is_charset: std::false_type
{};
template <typename Ch, strf::charset_id CsId>
struct is_charset<strf::static_charset<Ch, CsId>>: std::true_type
{};
template <typename Ch>
struct is_charset<strf::dynamic_charset<Ch>>: std::true_type
{};

} // namespace detail

template <typename SrcCharT, typename DstCharT>
struct decode_encode_result
{
    const SrcCharT* stale_src_ptr;
    DstCharT* dst_ptr;
    std::int32_t u32dist;
    transcode_stop_reason stop_reason;
};

template <typename SrcCharT>
struct decode_encode_size_result
{
    std::ptrdiff_t ssize;
    const SrcCharT* stale_src_ptr;
    std::int32_t u32dist;
    transcode_stop_reason stop_reason;
};

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using stop_reason = strf::transcode_stop_reason;
    char32_t pivot_buff[16] = {};
    while (true) {
        char32_t* pivot = pivot_buff;
        char32_t* pivot_end = pivot_buff + sizeof(pivot_buff) / sizeof(pivot_buff[0]);
        const auto src_res = to_u32(src, src_end, pivot, pivot_end, err_notifier, flags);
        const auto dst_res = from_u32
            (pivot_buff, src_res.dst_ptr, dst, dst_end, err_notifier, flags);

        STRF_ASSERT(src_res.stop_reason != stop_reason::unsupported_codepoint);
        STRF_ASSERT(dst_res.stop_reason != stop_reason::invalid_sequence);

        if (dst_res.stop_reason == stop_reason::completed) {
            if (src_res.stop_reason == stop_reason::insufficient_output_space) {
                src = src_res.src_ptr;
                dst = dst_res.dst_ptr;
                continue;
            }
            STRF_ASSERT(src_res.stop_reason == stop_reason::completed ||
                        src_res.stop_reason == stop_reason::invalid_sequence);
            return {src_res.src_ptr, dst_res.dst_ptr, 0, src_res.stop_reason};
        }
        const auto distance = static_cast<int32_t>(dst_res.src_ptr - pivot);
        return {src, dst_res.dst_ptr, distance, dst_res.stop_reason};
    }
}

#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::decode_encode
        ( to_u32, from_u32, src.data(), src.data() + src.size()
        , dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    return strf::decode_encode
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_func()
        , src, src_end, dst, dst_end, err_notifier, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::decode_encode
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_func()
        , src, dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode
        ( src_charset, dst_charset, src, src_end, dst, dst_end, err_notifier, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> decode_encode
    ( std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode
        ( src_charset, dst_charset, src, dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using stop_reason = strf::transcode_stop_reason;
    if (!dst.good()) {
        return {0, src, 0, stop_reason::insufficient_output_space};
    }
    char32_t pivot_buff[16] = {};
    char32_t* const pivot_end = pivot_buff + sizeof(pivot_buff) / sizeof(pivot_buff[0]);
    std::ptrdiff_t count = 0;
    auto* dst_it = dst.buffer_ptr();
    auto* dst_end = dst.buffer_end();
    while (true) {
        const auto src_res = to_u32(src, src_end, pivot_buff, pivot_end, err_notifier, flags);

        char32_t* pivot_it = pivot_buff;
        while (true) {
            const auto dst_res = from_u32
                (pivot_it, src_res.dst_ptr, dst_it, dst_end, err_notifier, flags);

            count += dst_res.dst_ptr - dst_it;
            if (dst_res.stop_reason == stop_reason::completed) {
                if (src_res.stop_reason == stop_reason::insufficient_output_space) {
                    src = src_res.src_ptr;
                    dst_it = dst_res.dst_ptr;
                    break; // i.e. go back to to_u23(...) line
                }
                STRF_ASSERT(src_res.stop_reason == stop_reason::completed ||
                            src_res.stop_reason == stop_reason::invalid_sequence);
                dst.advance_to(dst_res.dst_ptr);
                return {count, src_res.src_ptr, 0, src_res.stop_reason};
            }
            dst.advance_to(dst_res.dst_ptr);
            if (dst_res.stop_reason == stop_reason::insufficient_output_space) {
                dst.recycle();
                if (!dst.good()) {
                    const auto u32dist = static_cast<int32_t>(dst_res.src_ptr - pivot_buff);
                    return {count, src, u32dist, stop_reason::insufficient_output_space};
                }
                // there is still something remaining in the pivot buffer to be encoded
                dst_it = dst.buffer_ptr();
                dst_end = dst.buffer_end();
                pivot_it = const_cast<char32_t*>(dst_res.src_ptr);
            } else {
                STRF_ASSERT(dst_res.stop_reason == stop_reason::unsupported_codepoint);
                const auto u32dist = static_cast<int32_t>(dst_res.src_ptr - pivot_buff);
                return {count, src, u32dist, stop_reason::unsupported_codepoint};
            }
        }
    }
}

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    return strf::decode_encode
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_func()
        , src, src_end, dst, err_notifier, flags );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode
        ( src_charset, dst_charset, src, src_end, dst, err_notifier, flags );
}

template <typename SrcCharT>
STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_stop_reason = strf::transcode_stop_reason;
    using size_stop_reason = strf::transcode_stop_reason;

    constexpr auto pivot_buff_size = 16;
    char32_t pivot_buff[pivot_buff_size] = {};
    std::ptrdiff_t size = 0;
    while (true) {
        const auto src_res = to_u32
            ( src, src_end, pivot_buff, pivot_buff + pivot_buff_size, nullptr, flags);
        const auto size_res = size_calc_func(pivot_buff, src_res.dst_ptr, limit, flags);

        STRF_ASSERT(src_res.stop_reason != src_stop_reason::unsupported_codepoint);
        STRF_ASSERT(size_res.stop_reason != size_stop_reason::invalid_sequence);
        STRF_ASSERT(size_res.ssize <= limit);
        limit -= size_res.ssize;
        size  += size_res.ssize;

        if (size_res.stop_reason == size_stop_reason::completed) {
            if (src_res.stop_reason == src_stop_reason::insufficient_output_space) {
                src = src_res.src_ptr;
                continue;
            }
            STRF_ASSERT(src_res.stop_reason == src_stop_reason::completed ||
                        src_res.stop_reason == src_stop_reason::invalid_sequence);
            return {size, src_res.src_ptr, 0, static_cast<size_stop_reason>(src_res.stop_reason)};
        }
        const auto distance = static_cast<int32_t>(size_res.src_ptr - pivot_buff);
        return {size, src, distance, size_res.stop_reason};
    }
}

#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::decode_encode_size
        ( to_u32, size_calc_func, src.data(), src.data() + src.size(), limit, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::decode_encode_size
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_size_func()
        , src, src_end, limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode_size
        ( src_charset, dst_charset, src, src_end, limit, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::decode_encode_size
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_size_func()
        , src, limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
inline STRF_HD strf::decode_encode_size_result<SrcCharT> decode_encode_size
    ( std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode_size(src_charset, dst_charset, src, limit, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

template<typename SrcCharT, typename DstCharT>
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::decode_encode
        (to_u32, from_u32, src, src_end, dst, dst_end, err_notifier, flags);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template<typename SrcCharT, typename DstCharT>
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_f<char32_t, DstCharT> from_u32
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::unsafe_decode_encode
        ( to_u32, from_u32, src.data(), src.data() + src.size()
        , dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW


template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    return strf::unsafe_decode_encode
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_func()
        , src, src_end, dst, dst_end, err_notifier, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::unsafe_decode_encode
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_func()
        , src, dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode
        ( src_charset, dst_charset, src, src_end, dst, dst_end, err_notifier, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_decode_encode
    ( std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode
        ( src_charset, dst_charset, src, dst, dst_end, err_notifier, flags );
}

#endif // STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using stop_reason = strf::transcode_stop_reason;
    constexpr auto pivot_buff_size = 32;
    char32_t pivot_buff[pivot_buff_size] = {};
    std::ptrdiff_t size = 0;
    while (true) {
        const auto src_res = to_u32
            ( src, src_end, pivot_buff, pivot_buff + pivot_buff_size,  nullptr, flags);
        const auto size_res = size_calc_func(pivot_buff, src_res.dst_ptr, limit, flags);

        STRF_ASSERT(src_res.stop_reason != stop_reason::unsupported_codepoint);
        STRF_ASSERT(size_res.ssize <= limit);

        limit -= size_res.ssize;
        size  += size_res.ssize;

        if (size_res.stop_reason == stop_reason::completed) {
            if (src_res.stop_reason == stop_reason::insufficient_output_space) {
                src = src_res.src_ptr;
                continue;
            }
            return {size, src_res.src_ptr, 0, src_res.stop_reason};
        }
        const auto src_u32len_dist = static_cast<std::int32_t>(size_res.src_ptr - pivot_buff);
        return {size, src, src_u32len_dist, size_res.stop_reason};

    }
}

#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_size_f<char32_t> size_calc_func
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::unsafe_decode_encode_size
        (to_u32, size_calc_func, src.data(), src.data() + src.size(), limit, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::unsafe_decode_encode_size
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_size_func()
        , src, src_end, limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode_size
        ( src_charset, dst_charset, src, src_end, limit, flags );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::unsafe_decode_encode_size
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_size_func()
        , src, limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_decode_encode_size
    ( std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode_size(src_charset, dst_charset, src, limit, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.transcode_func();
    if (func != nullptr) {
        auto res = func(src, src_end, dst, dst_end, err_notifier, flags);
        return {res.src_ptr, res.dst_ptr, 0, res.stop_reason};
    }
    return strf::decode_encode
        ( src_charset, dst_charset
        , src, src_end, dst, dst_end, err_notifier, flags );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode
        (src_charset, dst_charset, src, src_end, dst, dst_end, err_notifier, flags);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::transcode
        ( src_charset, dst_charset, src.data(), src.data() + src.size()
        , dst, dst_end, err_notifier, flags );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> transcode
    ( std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode
        (src_charset, dst_charset, src, dst, dst_end, err_notifier, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT, typename DstCharT>
STRF_HD strf::transcode_size_result<SrcCharT> transcode
    ( transcode_f<SrcCharT, DstCharT> func
    , const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using stop_reason = strf::transcode_stop_reason;
    auto dst_it = dst.buffer_ptr();
    auto dst_end = dst.buffer_end();
    if (!dst.good()) {
        return {0, src, stop_reason::insufficient_output_space};
    }
    std::ptrdiff_t count = 0;
    while (1) {
        const auto res = func(src, src_end, dst_it, dst_end, err_notifier, flags);
        count += res.dst_ptr - dst_it;
        dst.advance_to(res.dst_ptr);
        if (res.stop_reason != stop_reason::insufficient_output_space) {
            return {count, res.src_ptr, res.stop_reason};
        }
        dst.recycle();
        if (!dst.good()) {
            return {count, res.src_ptr, stop_reason::insufficient_output_space};
        }
        dst_it = dst.buffer_ptr();
        dst_end = dst.buffer_end();
        src = res.src_ptr;
    }
}

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.transcode_func();
    if (func != nullptr) {
        auto res = transcode(func, src, src_end, dst, err_notifier, flags);
        return {res.ssize, res.src_ptr, 0, res.stop_reason};
    }
    const auto src_to_u32 = src_charset.to_u32().transcode_func();
    const auto u32_to_dst = dst_charset.from_u32().transcode_func();
    return decode_encode(src_to_u32, u32_to_dst, src, src_end, dst, err_notifier, flags);
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode
        (src_charset, dst_charset, src, src_end, dst, err_notifier, flags);
}

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.transcode_size_func();
    if (func != nullptr) {
        auto res = func(src, src_end, limit, flags);
        return {res.ssize, res.src_ptr, 0, res.stop_reason};
    }
    return strf::decode_encode_size
        ( src_charset, dst_charset, src, src_end, limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
STRF_HD strf::decode_encode_size_result<SrcCharT> transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode_size(src_charset, dst_charset, src, src_end, limit, flags);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::transcode_size
        ( src_charset, dst_charset, src.data(), src.data() + src.size(), limit, flags );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> transcode_size
    ( std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode_size(src_charset, dst_charset, src, limit, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.unsafe_transcode_func();
    if (func != nullptr) {
        auto res = func(src, src_end, dst, dst_end, err_notifier, flags);
        return {res.src_ptr, res.dst_ptr, 0, res.stop_reason};
    }
    return strf::unsafe_decode_encode
        ( src_charset, dst_charset, src, src_end, dst, dst_end, err_notifier, flags);
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode
        (src_charset, dst_charset, src, src_end, dst, dst_end, err_notifier, flags);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::unsafe_transcode
        ( src_charset, dst_charset, src.data()
        , src.data() + src.size(), dst, dst_end, err_notifier, flags );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_result<SrcCharT, DstCharT> unsafe_transcode
    ( std::basic_string_view<SrcCharT> src
    , DstCharT* dst
    , DstCharT* dst_end
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode
        (src_charset, dst_charset, src, dst, dst_end, err_notifier, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW


template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.unsafe_transcode_func();
    if (func != nullptr) {
        auto res = strf::transcode(func, src, src_end, dst, err_notifier, flags);
        return {res.ssize, res.src_ptr, 0, res.stop_reason};
    }
    const auto src_to_u32 = src_charset.to_u32().unsafe_transcode_func();
    const auto u32_to_dst = dst_charset.from_u32().unsafe_transcode_func();
    return strf::decode_encode(src_to_u32, u32_to_dst, src, src_end, dst, err_notifier, flags);
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , strf::destination<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode
        (src_charset, dst_charset, src, src_end, dst, err_notifier, flags);
}

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.unsafe_transcode_size_func();
    if (func != nullptr) {
        auto res = func(src, src_end, limit, flags);
        return {res.ssize, res.src_ptr, 0, res.stop_reason};
    }
    return strf::unsafe_decode_encode_size(src_charset, dst_charset, src, src_end, limit, flags);
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode_size
    ( const SrcCharT* src
    , const SrcCharT* src_end
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode_size(src_charset, dst_charset, src, src_end, limit, flags);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    return strf::unsafe_transcode_size
        (src_charset, dst_charset, src.data(), src.data() + src.size(), limit, flags);
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD strf::decode_encode_size_result<SrcCharT> unsafe_transcode_size
    ( std::basic_string_view<SrcCharT> src
    , std::ptrdiff_t limit
    , strf::transcode_flags flags = strf::transcode_flags::none )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode_size(src_charset, dst_charset, src, limit, flags);
}

#endif // STRF_HAS_STD_STRING_VIEW

// backwards compatibility:

#if defined(STRF_DONT_DEPRECATE_CHAR_ENCODING)
#  define STRF_CHAR_ENCODING_DEPRECATED
#else
#  define STRF_CHAR_ENCODING_DEPRECATED STRF_DEPRECATED
#endif

using char_encoding_id STRF_CHAR_ENCODING_DEPRECATED =
    strf::charset_id;

template <typename CharT, strf::charset_id Id>
using static_char_encoding STRF_CHAR_ENCODING_DEPRECATED =
    strf::static_charset<CharT, Id>;

template <typename CharT>
using dynamic_char_encoding_data STRF_CHAR_ENCODING_DEPRECATED =
    strf::dynamic_charset_data<CharT>;

template <typename CharT>
using dynamic_char_encoding STRF_CHAR_ENCODING_DEPRECATED =
    strf::dynamic_charset<CharT>;

template <typename CharT>
using char_encoding_c STRF_CHAR_ENCODING_DEPRECATED =
    strf::charset_c<CharT>;

} // namespace strf

#endif // STRF_DETAIL_FACETS_CHARSET_HPP

