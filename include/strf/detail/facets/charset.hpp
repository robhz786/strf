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
    virtual STRF_HD ~transcoding_error_notifier() {}

    virtual STRF_HD void invalid_sequence
        ( std::size_t code_unit_size
        , const char* charset_name
        , const void* sequence_ptr
        , std::size_t code_units_count )
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

    STRF_HD constexpr transcoding_error_notifier_ptr(transcoding_error_notifier* p) noexcept
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

constexpr std::size_t invalid_char_len = (std::size_t)-1;

template <typename CharT>
using transcode_dest = strf::output_buffer<CharT, 3>;

constexpr std::size_t transcode_dest_min_buffer_size = 8;

template <typename SrcCharT, typename DestCharT>
using transcode_f = void (*)
    ( strf::transcode_dest<DestCharT>& dest
    , const SrcCharT* src
    , std::size_t src_size
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli );

template <typename SrcCharT>
using transcode_size_f = std::size_t (*)
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli );

template <typename CharT>
using write_replacement_char_f = void (*)
    ( strf::transcode_dest<CharT>& );

// assume surragate_policy::lax
using validate_f = std::size_t (*)(char32_t ch);

// assume surragates_policy::lax
using encoded_char_size_f = std::size_t (*)(char32_t ch);

// assume surrogate_policy::lax
template <typename CharT>
using encode_char_f = CharT*(*)
    ( CharT* dest, char32_t ch );

template <typename CharT>
using encode_fill_f = void (*)
    ( strf::transcode_dest<CharT>&, std::size_t count, char32_t ch );

namespace detail {

template <typename CharT>
void trivial_fill_f
    ( strf::transcode_dest<CharT>& dest, std::size_t count, char32_t ch )
{
    // same as strf::detail::write_fill<CharT>
    CharT narrow_ch = static_cast<CharT>(ch);
    STRF_IF_LIKELY (count <= dest.buffer_space()) {
        strf::detail::str_fill_n<CharT>(dest.buffer_ptr(), count, narrow_ch);
        dest.advance(count);
    } else {
        write_fill_continuation<CharT>(dest, count, narrow_ch);
    }
}

} // namespace detail

struct count_codepoints_result {
    std::size_t count;
    std::size_t pos;
};

template <typename CharT>
using count_codepoints_fast_f = strf::count_codepoints_result (*)
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count );

template <typename CharT>
using count_codepoints_f = strf::count_codepoints_result (*)
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count
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
    {
    }

    constexpr STRF_HD dynamic_transcoder() noexcept
        : transcode_func_(nullptr)
        , transcode_size_func_(nullptr)
    {
    }

    STRF_HD void transcode
        ( strf::transcode_dest<DestCharT>& dest
        , const SrcCharT* src
        , std::size_t src_size
        , strf::transcoding_error_notifier* err_notifier
        , strf::surrogate_policy surr_poli ) const
    {
        transcode_func_(dest, src, src_size, err_notifier, surr_poli);
    }

    STRF_HD std::size_t transcode_size
        ( const SrcCharT* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli ) const
    {
        return transcode_size_func_(src, src_size, surr_poli);
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

private:

    strf::transcode_f<SrcCharT, DestCharT> transcode_func_;
    strf::transcode_size_f<SrcCharT> transcode_size_func_;
};

template <typename CharT>
struct dynamic_charset_data
{
    constexpr STRF_HD dynamic_charset_data
        ( const char* name_
        , strf::charset_id id_
        , char32_t replacement_char_
        , std::size_t replacement_char_size_
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
    std::size_t replacement_char_size;
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

    STRF_HD dynamic_charset
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
    constexpr STRF_HD std::size_t replacement_char_size() const noexcept
    {
        return data_->replacement_char_size;
    }
    constexpr STRF_HD std::size_t validate(char32_t ch) const // noexcept
    {
        return data_->validate_func(ch);
    }
    constexpr STRF_HD std::size_t encoded_char_size(char32_t ch) const // noexcept
    {
        return data_->encoded_char_size_func(ch);
    }
    STRF_HD code_unit* encode_char(code_unit* dest, char32_t ch) const // noexcept
    {
        return data_->encode_char_func(dest, ch);
    }
    STRF_HD void encode_fill
        ( strf::transcode_dest<CharT>& dest, std::size_t count, char32_t ch ) const
    {
        data_->encode_fill_func(dest, count, ch);
    }
    STRF_HD strf::count_codepoints_result count_codepoints_fast
        ( const code_unit* src, std::size_t src_size, std::size_t max_count ) const
    {
        return data_->count_codepoints_fast_func(src, src_size, max_count);
    }
    STRF_HD strf::count_codepoints_result count_codepoints
        ( const code_unit* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) const
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

    using result_ = decltype(test_((SrcCharset*)0, (DestCharset*)0));

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
    = decltype(test<SrcCharT, DestCharset>(strf::tag<SrcCharT>(), 0))::value;
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
    = decltype(test<DestCharT, SrcCharset>(strf::tag<DestCharT>(), 0))::value;
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
        auto p = this->buffer_ptr();
        STRF_IF_LIKELY (p != buff_ && dest_.good()) {
            transcode_( dest_, buff_, static_cast<std::size_t>(p - buff_)
                      , err_notifier_, surr_poli_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_f<char32_t, DestCharT> transcode_;
    strf::transcode_dest<DestCharT>& dest_;
    strf::transcoding_error_notifier* err_notifier_;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};


template <typename DestCharT>
STRF_HD void buffered_encoder<DestCharT>::recycle()
{
    auto p = this->buffer_ptr();
    this->set_buffer_ptr(buff_);
    STRF_IF_LIKELY (p != buff_ && dest_.good()) {
        this->set_good(false);
        transcode_( dest_, buff_, static_cast<std::size_t>(p - buff_)
                  , err_notifier_, surr_poli_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::transcode_dest<char32_t>
{
public:

    STRF_HD buffered_size_calculator
        ( strf::transcode_size_f<char32_t> func, strf::surrogate_policy surr_poli )
        : strf::transcode_dest<char32_t>(buff_, buff_size_)
        , size_func_(func)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD std::size_t get_sum()
    {
        flush();
        return sum_;
    }

private:

    strf::transcode_size_f<char32_t> size_func_;
    std::size_t sum_ = 0;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

STRF_FUNC_IMPL STRF_HD void buffered_size_calculator::recycle()
{
    auto p = this->buffer_ptr();
    STRF_IF_LIKELY (p != buff_) {
        this->set_buffer_ptr(buff_);
        sum_ += size_func_(buff_, static_cast<std::size_t>(p - buff_), surr_poli_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

template<typename SrcCharT, typename DestCharT>
STRF_HD void decode_encode
    ( strf::transcode_dest<DestCharT>& dest
    , strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DestCharT> from_u32
    , const SrcCharT* src
    , std::size_t src_size
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_encoder<DestCharT> tmp{from_u32, dest, err_notifier, surr_poli};
    to_u32(tmp, src, src_size, err_notifier, surr_poli);
    tmp.finish();
}

template<typename SrcCharT>
STRF_HD std::size_t decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , std::size_t src_size
    , strf::transcoding_error_notifier* err_notifier
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, surr_poli};
    to_u32(acc, src, src_size, err_notifier, surr_poli);
    return acc.get_sum();
}


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

#endif  // STRF_DETAIL_FACETS_CHARSET_HPP

