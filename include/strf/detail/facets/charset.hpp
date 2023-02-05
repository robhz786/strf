#ifndef STRF_DETAIL_FACETS_CHARSET_HPP
#define STRF_DETAIL_FACETS_CHARSET_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/output_buffer_functions.hpp>

namespace strf {

template <typename> struct facet_traits;

enum class surrogate_policy : bool
{
    strict = false, lax = true
};

struct surrogate_policy_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::surrogate_policy get_default() noexcept
    {
        return strf::surrogate_policy::strict;
    }
};

template <>
struct facet_traits<strf::surrogate_policy>
{
    using category = strf::surrogate_policy_c;
};

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

template < typename SrcCharT, typename DestCharT
         , strf::charset_id Src, strf::charset_id Dest>
class static_transcoder;

template <typename SrcCharT, typename DestCharT>
class dynamic_transcoder;

constexpr int invalid_char_len = -1;

template <typename CharT>
using transcode_dest = strf::output_buffer<CharT, 3>;

constexpr std::ptrdiff_t transcode_dest_min_buffer_size = 8;

template <typename SrcCharT, typename DestCharT>
using transcode_f = void (*)
    ( strf::transcode_dest<DestCharT>& dest
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli );

template <typename SrcCharT, typename DestCharT>
using unsafe_transcode_f = void (*)
    ( strf::transcode_dest<DestCharT>& dest
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcoding_error_notifier* err_notifier );

template <typename SrcCharT>
using transcode_size_f = std::ptrdiff_t (*)
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy surr_poli );

template <typename SrcCharT>
using unsafe_transcode_size_f = std::ptrdiff_t (*)
    ( const SrcCharT* src
    , std::ptrdiff_t src_size );

template <typename CharT>
using write_replacement_char_f = void (*)
    ( strf::transcode_dest<CharT>& );

// assume surragate_policy::lax
using validate_f = int (*)(char32_t ch);

// assume surragates_policy::lax
using encoded_char_size_f = int (*)(char32_t ch);

// assume surrogate_policy::lax
template <typename CharT>
using encode_char_f = CharT*(*)
    ( CharT* dest, char32_t ch );

template <typename CharT>
using encode_fill_f = void (*)
    ( strf::transcode_dest<CharT>&, std::ptrdiff_t count, char32_t ch );

namespace detail {

template <typename CharT>
void trivial_fill_f
    ( strf::transcode_dest<CharT>& dest, std::ptrdiff_t count, char32_t ch )
{
    // same as strf::detail::write_fill<CharT>
    auto narrow_ch = static_cast<CharT>(ch);
    STRF_IF_LIKELY (count <= dest.buffer_sspace()) {
        strf::detail::str_fill_n<CharT>(dest.buffer_ptr(), count, narrow_ch);
        dest.advance(count);
    } else {
        write_fill_continuation<CharT>(dest, count, narrow_ch);
    }
}

} // namespace detail

struct count_codepoints_result {
    std::ptrdiff_t count;
    std::ptrdiff_t pos;
};

template <typename CharT>
using count_codepoints_fast_f = strf::count_codepoints_result (*)
    ( const CharT* src
    , std::ptrdiff_t src_size
    , std::ptrdiff_t max_count );

template <typename CharT>
using count_codepoints_f = strf::count_codepoints_result (*)
    ( const CharT* src
    , std::ptrdiff_t src_size
    , std::ptrdiff_t max_count
    , strf::surrogate_policy surr_poli );


using codepoints_count_result
    STRF_DEPRECATED_MSG("codepoints_count_result was renamed to count_codepoints_result")
     = count_codepoints_result;

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

template <typename SrcCharT, typename DestCharT>
using find_transcoder_f =
    strf::dynamic_transcoder<SrcCharT, DestCharT> (*)
    ( strf::charset_id );

template <typename SrcCharT, typename DestCharT>
class dynamic_transcoder
{
public:

    template <strf::charset_id SrcId, strf::charset_id DestId>
    constexpr explicit STRF_HD dynamic_transcoder
        ( strf::static_transcoder<SrcCharT, DestCharT, SrcId, DestId> t ) noexcept
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

    STRF_HD void transcode
        ( strf::transcode_dest<DestCharT>& dest
        , const SrcCharT* src
        , std::ptrdiff_t src_size
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli ) const
    {
        transcode_func_(dest, src, src_size, err_notifier, surr_poli);
    }

    STRF_HD void unsafe_transcode
        ( strf::transcode_dest<DestCharT>& dest
        , const SrcCharT* src
        , std::ptrdiff_t src_size
        , strf::transcoding_error_notifier* err_notifier ) const
    {
        unsafe_transcode_func_(dest, src, src_size, err_notifier);
    }

    STRF_HD std::ptrdiff_t transcode_size
        ( const SrcCharT* src
        , std::ptrdiff_t src_size
        , strf::surrogate_policy surr_poli ) const
    {
        return transcode_size_func_(src, src_size, surr_poli);
    }

    STRF_HD std::ptrdiff_t unsafe_transcode_size
        ( const SrcCharT* src
        , std::ptrdiff_t src_size ) const
    {
        return unsafe_transcode_size_func_(src, src_size);
    }

    constexpr STRF_HD strf::transcode_f<SrcCharT, DestCharT>
    transcode_func() const noexcept
    {
        return transcode_func_;
    }

    constexpr STRF_HD strf::transcode_size_f<SrcCharT>
    transcode_size_func() const noexcept
    {
        return transcode_size_func_;
    }

    constexpr STRF_HD strf::unsafe_transcode_f<SrcCharT, DestCharT>
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

    strf::transcode_f<SrcCharT, DestCharT> transcode_func_;
    strf::transcode_size_f<SrcCharT> transcode_size_func_;
    strf::unsafe_transcode_f<SrcCharT, DestCharT> unsafe_transcode_func_;
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
    STRF_HD code_unit* encode_char(code_unit* dest, char32_t ch) const // noexcept
    {
        return data_->encode_char_func(dest, ch);
    }
    STRF_HD void encode_fill
        ( strf::transcode_dest<CharT>& dest, std::ptrdiff_t count, char32_t ch ) const
    {
        data_->encode_fill_func(dest, count, ch);
    }
    STRF_HD strf::count_codepoints_result count_codepoints_fast
        ( const code_unit* src, std::ptrdiff_t src_size, std::ptrdiff_t max_count ) const
    {
        return data_->count_codepoints_fast_func(src, src_size, max_count);
    }
    STRF_HD strf::count_codepoints_result count_codepoints
        ( const code_unit* src, std::ptrdiff_t src_size
        , std::ptrdiff_t max_count, strf::surrogate_policy surr_poli ) const
    {
        return data_->count_codepoints_func(src, src_size, max_count, surr_poli);
    }
    STRF_HD void write_replacement_char(strf::transcode_dest<CharT>& dest) const
    {
        data_->write_replacement_char_func(dest);
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

template <typename SrcCharset, typename DestCharset>
class has_static_transcoder_impl
{
    using src_char_type = typename SrcCharset::code_unit;
    using dest_char_type = typename DestCharset::code_unit;

    template <strf::charset_id SrcId, strf::charset_id DestId>
    static
    decltype( strf::static_transcoder
                < src_char_type, dest_char_type, SrcId, DestId >()
            , std::true_type() )
    test_( strf::static_charset<src_char_type, SrcId>*
         , strf::static_charset<dest_char_type, DestId>* )
    {
        return {};
    }

    static std::false_type test_(...)
    {
        return {};
    }

    using result_ = decltype(test_((SrcCharset*)nullptr, (DestCharset*)nullptr));

public:

    static constexpr bool value = result_::value;
};

template <typename SrcCharset, typename DestCharset>
constexpr STRF_HD bool has_static_transcoder()
{
    return has_static_transcoder_impl<SrcCharset, DestCharset>::value;
}

template <typename SrcCharT, typename DestCharset>
class has_find_transcoder_from_impl
{
    template <typename S, typename D>
    static auto test(strf::tag<S> stag, const D* d)
        -> decltype( d->find_transcoder_from(stag, strf::csid_utf8)
                   , std::true_type() );

    template <typename S, typename D>
    static std::false_type test(...);

public:

    static constexpr bool value
    = decltype(test<SrcCharT, DestCharset>(strf::tag<SrcCharT>(), nullptr))::value;
};

template <typename DestCharT, typename SrcCharset>
class has_find_transcoder_to_impl
{
    template <typename D, typename S>
    static auto test(strf::tag<D> dtag, const S* s)
    -> decltype( s->find_transcoder_from(dtag, strf::csid_utf8)
               , std::true_type() );

    template <typename D, typename S>
    static std::false_type test(...);

public:
    static constexpr bool value
    = decltype(test<DestCharT, SrcCharset>(strf::tag<DestCharT>(), nullptr))::value;
};

template <typename DestCharT, typename SrcCharset>
STRF_HD constexpr bool has_find_transcoder_to()
{
    return has_find_transcoder_to_impl<DestCharT, SrcCharset>::value;
}

template <typename SrcCharT, typename DestCharset>
STRF_HD constexpr bool has_find_transcoder_from()
{
    return has_find_transcoder_from_impl<SrcCharT, DestCharset>::value;
}

template <bool HasFindTo, bool HasFindFrom, typename Transcoder>
struct transcoder_finder_2;

template <typename Transcoder>
struct transcoder_finder_2<true, true, Transcoder>
{
private:

    template < typename SrcCharset, typename DestCharset, typename Transc1
             , typename SrcTag = strf::tag<typename SrcCharset::code_unit> >
    constexpr static STRF_HD Transcoder do_find
        ( SrcCharset src_cs
        , DestCharset dest_cs
        , Transc1 t )
    {
        return ( t.transcode_func() != nullptr
               ? t
               : dest_cs.find_transcoder_from(SrcTag{}, src_cs.id()) );
    }

public:

    template < typename SrcCharset, typename DestCharset
             , typename DestTag = strf::tag<typename DestCharset::code_unit> >
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        return do_find(src_cs, dest_cs, src_cs.find_transcoder_to(DestTag{}, dest_cs.id()));
    }
};

template <typename Transcoder>
struct transcoder_finder_2<true, false, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        return src_cs.find_transcoder_to
            ( strf::tag<typename DestCharset::code_unit>{}
            , dest_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, true, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        return dest_cs.find_transcoder_from
            ( strf::tag<typename SrcCharset::code_unit>{}
            , src_cs.id() );
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, false, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset, DestCharset)
    {
        return {};
    }
};

template < bool HasStaticTranscoder
         , typename SrcCharT
         , typename DestCharT >
struct transcoder_finder;

template <typename SrcCharT, typename DestCharT >
struct transcoder_finder<true, SrcCharT, DestCharT>
{
    template < strf::charset_id SrcId, strf::charset_id DestId>
    constexpr static STRF_HD
    strf::static_transcoder<SrcCharT, DestCharT, SrcId, DestId> find
        ( strf::static_charset<SrcCharT, SrcId>
        , strf::static_charset<DestCharT, DestId> ) noexcept
    {
        return {};
    }
};

template <>
struct transcoder_finder<false, char32_t, char32_t>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD strf::utf32_to_utf32<char32_t, char32_t> find
        (SrcCharset, DestCharset )
    {
        return {};
    }
};

template <typename SrcCharT>
struct transcoder_finder<false, SrcCharT, char32_t>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto find(SrcCharset src_cs, DestCharset)
        -> decltype(src_cs.to_u32())
    {
        return src_cs.to_u32();
    }
};

template <typename DestCharT>
struct transcoder_finder<false, char32_t, DestCharT>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto find(SrcCharset, DestCharset dest_cs) noexcept
        -> decltype(dest_cs.from_u32())
    {
        return dest_cs.from_u32();
    }
};

template <typename CharT>
struct transcoder_finder<false, CharT, CharT>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD
    strf::dynamic_transcoder<CharT, CharT>
    find(SrcCharset src_cs, DestCharset dest_cs )
    {
        return ( src_cs.id() == dest_cs.id()
               ? strf::dynamic_transcoder<CharT, CharT>{ src_cs.sanitizer() }
               : strf::detail::transcoder_finder_2
                   < strf::detail::has_find_transcoder_to<CharT, SrcCharset>()
                   , strf::detail::has_find_transcoder_from<CharT, DestCharset>()
                   , strf::dynamic_transcoder<CharT, CharT> >
                   ::find(src_cs, dest_cs) );
    }
};

template <typename SrcCharT, typename DestCharT >
struct transcoder_finder<false, SrcCharT, DestCharT>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD
    strf::dynamic_transcoder<SrcCharT, DestCharT>
    find(SrcCharset src_cs, DestCharset dest_cs )
    {
        return strf::detail::transcoder_finder_2
            < strf::detail::has_find_transcoder_to<DestCharT, SrcCharset>()
            , strf::detail::has_find_transcoder_from<SrcCharT, DestCharset>()
            , strf::dynamic_transcoder<SrcCharT, DestCharT> >
            ::find(src_cs, dest_cs);
    }
};

} // namespace detail

template < typename SrcCharset
         , typename DestCharset
         , typename Finder =
             detail::transcoder_finder
                 < strf::detail::has_static_transcoder<SrcCharset, DestCharset>()
                 , typename SrcCharset::code_unit
                 , typename DestCharset::code_unit > >
constexpr STRF_HD auto find_transcoder(SrcCharset src_cs, DestCharset dest_cs)
    -> decltype(Finder::find(src_cs, dest_cs))
{
    return Finder::find(src_cs, dest_cs);
}

namespace detail {

template <typename DestCharT>
class buffered_encoder: public strf::transcode_dest<char32_t>
{
public:

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    STRF_HD buffered_encoder
        ( strf::transcode_f<char32_t, DestCharT> func
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli )
        : strf::transcode_dest<char32_t>( buff_, buff_size_ )
        , transcode_(func)
        , dest_(dest)
        , err_notifier_(err_notifier)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto *p = this->buffer_ptr();
        STRF_IF_LIKELY (p != buff_ && dest_.good()) {
            transcode_( dest_, buff_, (p - buff_)
                      , err_notifier_, surr_poli_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_f<char32_t, DestCharT> transcode_;
    strf::transcode_dest<DestCharT>& dest_;
    strf::transcoding_error_notifier* err_notifier_;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::ptrdiff_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};


template <typename DestCharT>
STRF_HD void buffered_encoder<DestCharT>::recycle()
{
    auto *p = this->buffer_ptr();
    this->set_buffer_ptr(buff_);
    STRF_IF_LIKELY (p != buff_ && dest_.good()) {
        this->set_good(false);
        transcode_( dest_, buff_, (p - buff_)
                  , err_notifier_, surr_poli_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::transcode_dest<char32_t>
{
public:

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    STRF_HD buffered_size_calculator
        ( strf::transcode_size_f<char32_t> func, strf::surrogate_policy surr_poli )
        : strf::transcode_dest<char32_t>(buff_, buff_size_)
        , size_func_(func)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD std::ptrdiff_t get_sum()
    {
        flush();
        return sum_;
    }

private:

    strf::transcode_size_f<char32_t> size_func_;
    std::ptrdiff_t sum_ = 0;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::ptrdiff_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD void buffered_size_calculator::recycle()
{
    auto *p = this->buffer_ptr();
    STRF_IF_LIKELY (p != buff_) {
        this->set_buffer_ptr(buff_);
        sum_ += size_func_(buff_, (p - buff_), surr_poli_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)


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
STRF_HD void decode_encode
    ( strf::transcode_dest<DstCharT>& dst
    , strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy poli )
{
    strf::detail::buffered_encoder<DstCharT> tmp{from_u32, dst, err_notifier, poli};
    to_u32(tmp, src, src_size, err_notifier, poli);
    tmp.finish();
}

template <typename SrcCharT, typename DstCharT>
STRF_HD void decode_encode
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    strf::detail::buffered_encoder<DstCharT> tmp{from_u32, dst, err_notifier, poli};
    to_u32(tmp, src, src_size, err_notifier, poli);
    tmp.finish();
}


#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT, typename DstCharT>
STRF_HD void decode_encode
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DstCharT> from_u32
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    return strf::decode_encode
        ( to_u32, from_u32, src.data(), static_cast<std::ptrdiff_t>(src.size()), dst, err_notifier, poli );
}

#endif // STRF_HAS_STD_STRING_DECLARATIO

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    return strf::decode_encode
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_func()
        , src, src_size, dst, err_notifier, poli );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    return strf::decode_encode
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_func()
        , src, dst, err_notifier, poli );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void decode_encode
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
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
        ( src_charset, dst_charset, src, src_size, dst, err_notifier, poli );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void decode_encode
    ( std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
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
        ( src_charset, dst_charset, src, dst, err_notifier, poli );
}

#endif // STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
STRF_HD std::ptrdiff_t decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, poli};
    to_u32(acc, src, src_size, nullptr, poli);
    return acc.get_sum();
}

#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
inline STRF_HD std::ptrdiff_t decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , std::basic_string_view<SrcCharT> src
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    return strf::decode_encode_size
        ( to_u32, size_calc_func, src.data(), static_cast<std::ptrdiff_t>(src.size()), poli );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::decode_encode_size
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_size_func()
        , src, src_size, poli );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t decode_encode_size
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode_size
        ( src_charset, dst_charset, src, src_size, poli );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::decode_encode_size
        ( src_charset.to_u32().transcode_func()
        , dst_charset.from_u32().transcode_size_func()
        , src, poli );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
inline STRF_HD std::ptrdiff_t decode_encode_size
    ( std::basic_string_view<SrcCharT> src
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::decode_encode_size(src_charset, dst_charset, src, poli);
}

#endif // STRF_HAS_STD_STRING_VIEW

namespace detail {

template <typename DestCharT>
class unsafe_buffered_encoder: public strf::transcode_dest<char32_t>
{
public:

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    STRF_HD unsafe_buffered_encoder
        ( strf::unsafe_transcode_f<char32_t, DestCharT> func
        , strf::transcode_dest<DestCharT>& dest
        , strf::transcoding_error_notifier* err_notifier )
        : strf::transcode_dest<char32_t>( buff_, buff_size_ )
        , transcode_(func)
        , dest_(dest)
        , err_notifier_(err_notifier)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto *p = this->buffer_ptr();
        STRF_IF_LIKELY (p != buff_ && dest_.good()) {
            transcode_( dest_, buff_, (p - buff_), err_notifier_);
        }
        this->set_good(false);
    }

private:

    strf::unsafe_transcode_f<char32_t, DestCharT> transcode_;
    strf::transcode_dest<DestCharT>& dest_;
    strf::transcoding_error_notifier* err_notifier_;
    constexpr static const std::ptrdiff_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

template <typename DestCharT>
STRF_HD void unsafe_buffered_encoder<DestCharT>::recycle()
{
    auto *p = this->buffer_ptr();
    this->set_buffer_ptr(buff_);
    STRF_IF_LIKELY (p != buff_ && dest_.good()) {
        this->set_good(false);
        transcode_(dest_, buff_, (p - buff_), err_notifier_);
        this->set_good(true);
    }
}

class unsafe_buffered_size_calculator: public strf::transcode_dest<char32_t>
{
public:

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    STRF_HD unsafe_buffered_size_calculator
        ( strf::unsafe_transcode_size_f<char32_t> func )
        : strf::transcode_dest<char32_t>(buff_, buff_size_)
        , size_func_(func)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD std::ptrdiff_t get_sum()
    {
        flush();
        return sum_;
    }

private:

    strf::unsafe_transcode_size_f<char32_t> size_func_;
    std::ptrdiff_t sum_ = 0;
    constexpr static const std::ptrdiff_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD void unsafe_buffered_size_calculator::recycle()
{
    auto *p = this->buffer_ptr();
    STRF_IF_LIKELY (p != buff_) {
        this->set_buffer_ptr(buff_);
        sum_ += size_func_(buff_, (p - buff_));
    }
}

#endif //! defined(STRF_OMIT_IMPL)

} // namespace detail

template<typename SrcCharT, typename DstCharT>
STRF_HD void unsafe_decode_encode
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_f<char32_t, DstCharT> from_u32
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr )
{
    strf::detail::unsafe_buffered_encoder<DstCharT> tmp{from_u32, dst, err_notifier};
    to_u32(tmp, src, src_size, err_notifier);
    tmp.finish();
}

#ifdef STRF_HAS_STD_STRING_VIEW

template<typename SrcCharT, typename DstCharT>
inline STRF_HD void unsafe_decode_encode
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_f<char32_t, DstCharT> from_u32
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr)
{
    return strf::unsafe_decode_encode
        ( to_u32, from_u32, src.data(), static_cast<std::ptrdiff_t>(src.size()), dst, err_notifier );
}

#endif // STRF_HAS_STD_STRING_VIEW


template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void unsafe_decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr)
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    return strf::unsafe_decode_encode
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_func()
        , src, src_size, dst, err_notifier );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void unsafe_decode_encode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr)
{
    return strf::unsafe_decode_encode
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_func()
        , src, dst, err_notifier );
}

#endif // STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void unsafe_decode_encode
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr)
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
        ( src_charset, dst_charset, src, src_size, dst, err_notifier );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void unsafe_decode_encode
    ( std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr)
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
        ( src_charset, dst_charset, src, dst, err_notifier );
}

#endif // STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , std::ptrdiff_t src_size)
{
    strf::detail::unsafe_buffered_size_calculator acc{size_calc_func};
    to_u32(acc, src, src_size, nullptr);
    return acc.get_sum();
}

#ifdef STRF_HAS_STD_STRING_VIEW

template <typename SrcCharT>
inline STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( strf::unsafe_transcode_f<SrcCharT, char32_t> to_u32
    , strf::unsafe_transcode_size_f<char32_t> size_calc_func
    , std::basic_string_view<SrcCharT> src)
{
    return strf::unsafe_decode_encode_size
        (to_u32, size_calc_func, src.data(), static_cast<std::ptrdiff_t>(src.size()));
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::unsafe_decode_encode_size
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_size_func()
        , src, src_size );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( const SrcCharT* src
    , std::ptrdiff_t src_size )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode_size
        ( src_charset, dst_charset, src, src_size );
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    return strf::unsafe_decode_encode_size
        ( src_charset.to_u32().unsafe_transcode_func()
        , dst_charset.from_u32().unsafe_transcode_size_func()
        , src );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
inline STRF_HD std::ptrdiff_t unsafe_decode_encode_size
    ( std::basic_string_view<SrcCharT> src )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_decode_encode_size(src_charset, dst_charset, src);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD void do_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.transcode_func();
    if (func != nullptr) {
        return func(dst, src, src_size, err_notifier, poli);
    }
    return strf::decode_encode
        ( src_charset, dst_charset
        , src, src_size, dst, err_notifier, poli );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
STRF_HD void do_transcode
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::do_transcode
        (src_charset, dst_charset, src, src_size, dst, err_notifier, poli);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void do_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    return strf::do_transcode
        (src_charset, dst_charset, src.data(), static_cast<std::ptrdiff_t>(src.size()), dst, err_notifier, poli);
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void do_transcode
    ( std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::do_transcode
        (src_charset, dst_charset, src, dst, err_notifier, poli);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD std::ptrdiff_t transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.transcode_size_func();
    if (func != nullptr) {
        return func(src, src_size, poli);
    }
    return strf::decode_encode_size
        ( src_charset, dst_charset, src, src_size, poli );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0>
STRF_HD std::ptrdiff_t transcode_size
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode_size(src_charset, dst_charset, src, src_size, poli);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    return strf::transcode_size
        ( src_charset, dst_charset, src.data(), static_cast<std::ptrdiff_t>(src.size()), poli );
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD std::ptrdiff_t transcode_size
    ( std::basic_string_view<SrcCharT> src
    , strf::surrogate_policy poli = strf::surrogate_policy::strict )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::transcode_size(src_charset, dst_charset, src, poli);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD void do_unsafe_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename DstCharset::code_unit, DstCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.unsafe_transcode_func();
    if (func != nullptr) {
        return func( dst, src, src_size, err_notifier);
    }
    return strf::unsafe_decode_encode
        ( src_charset, dst_charset, src, src_size, dst, err_notifier );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
STRF_HD void do_unsafe_transcode
    ( const SrcCharT* src
    , std::ptrdiff_t src_size
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::do_unsafe_transcode
        (src_charset, dst_charset, src, src_size, dst, err_notifier);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset, typename DstCharset
         , typename SrcCharT, typename DstCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD void do_unsafe_transcode
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr )
{
    return strf::do_unsafe_transcode
        ( src_charset, dst_charset, src.data(), static_cast<std::ptrdiff_t>(src.size()), dst, err_notifier );
}

template < template <class> class SrcCharsetTmpl
         , template <class> class DstCharsetTmpl
         , typename SrcCharT
         , typename DstCharT >
inline STRF_HD void do_unsafe_transcode
    ( std::basic_string_view<SrcCharT> src
    , strf::transcode_dest<DstCharT>& dst
    , strf::transcoding_error_notifier* err_notifier = nullptr )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharsetTmpl<DstCharT>;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(detail::is_static_charset<dst_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");
    static_assert(std::is_same<typename dst_charset_t::code_unit, DstCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::do_unsafe_transcode
        (src_charset, dst_charset, src, dst, err_notifier);
}

#endif // STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
STRF_HD std::ptrdiff_t unsafe_transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , const SrcCharT* src
    , std::ptrdiff_t src_size )
{
    static_assert(std::is_same<typename SrcCharset::code_unit, SrcCharT>::value, "");

    const auto transcoder = strf::find_transcoder(src_charset, dst_charset);
    const auto func = transcoder.unsafe_transcode_size_func();
    if (func != nullptr) {
        return func(src, src_size);
    }
    return strf::unsafe_decode_encode_size(src_charset, dst_charset, src, src_size);
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD std::ptrdiff_t unsafe_transcode_size
    ( const SrcCharT* src
    , std::ptrdiff_t src_size )
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode_size(src_charset, dst_charset, src, src_size);
}

#ifdef STRF_HAS_STD_STRING_VIEW

template < typename SrcCharset
         , typename DstCharset
         , typename SrcCharT
         , detail::enable_if_t<detail::is_charset<SrcCharset>::value, int> = 0
         , detail::enable_if_t<detail::is_charset<DstCharset>::value, int> = 0 >
inline STRF_HD std::ptrdiff_t unsafe_transcode_size
    ( SrcCharset src_charset
    , DstCharset dst_charset
    , std::basic_string_view<SrcCharT> src )
{
    return strf::unsafe_transcode_size(src_charset, dst_charset, src.data(), static_cast<std::ptrdiff_t>(src.size()));
}

template < template <class> class SrcCharsetTmpl
         , typename DstCharset
         , typename SrcCharT
         , typename DstCharT = typename DstCharset::code_unit
         , detail::enable_if_t<detail::is_static_charset<DstCharset>::value, int> = 0 >
STRF_HD std::ptrdiff_t unsafe_transcode_size(std::basic_string_view<SrcCharT> src)
{
    using src_charset_t = SrcCharsetTmpl<SrcCharT>;
    using dst_charset_t = DstCharset;

    static_assert(detail::is_static_charset<src_charset_t>::value, "");
    static_assert(std::is_same<typename src_charset_t::code_unit, SrcCharT>::value, "");

    constexpr src_charset_t src_charset;
    constexpr dst_charset_t dst_charset;

    return strf::unsafe_transcode_size(src_charset, dst_charset, src);
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

