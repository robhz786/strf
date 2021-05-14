#ifndef STRF_DETAIL_FACETS_CHARSET_HPP
#define STRF_DETAIL_FACETS_CHARSET_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuff.hpp>

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

struct invalid_seq_notifier_c;

class invalid_seq_notifier
{
public:

    using category = invalid_seq_notifier_c;

    typedef void(*notify_fptr)();

    constexpr invalid_seq_notifier() noexcept = default;
    constexpr invalid_seq_notifier(const invalid_seq_notifier&) noexcept = default;

    constexpr STRF_HD explicit invalid_seq_notifier(notify_fptr f) noexcept
        : notify_func_(f)
    {
    }
    constexpr STRF_HD invalid_seq_notifier& operator=(notify_fptr f) noexcept
    {
        notify_func_ = f;
        return *this;
    }
    constexpr STRF_HD invalid_seq_notifier& operator=(const invalid_seq_notifier& other) noexcept
    {
        notify_func_ = other.notify_func_;
        return *this;
    }
    constexpr STRF_HD bool operator==(const invalid_seq_notifier& other) noexcept
    {
        return notify_func_ == other.notify_func_;
    }
    constexpr STRF_HD operator bool () const noexcept
    {
        return notify_func_ != nullptr;
    }
    constexpr STRF_HD void notify() const noexcept
    {
        STRF_IF_UNLIKELY (notify_func_) {
            notify_func_();
        }
    }

private:
    notify_fptr notify_func_ = nullptr;
};

struct invalid_seq_notifier_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::invalid_seq_notifier get_default() noexcept
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
constexpr strf::charset_id csid_utf = strf::detail::csid_utf_impl<sizeof(CharT)>::csid;

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

template <typename SrcCharT, typename DestCharT>
using transcode_f = void (*)
    ( strf::basic_outbuff<DestCharT>& ob
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli );

template <typename SrcCharT>
using transcode_size_f = std::size_t (*)
    ( const SrcCharT* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli );

template <typename CharT>
using write_replacement_char_f = void (*)
    ( strf::basic_outbuff<CharT>& );

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
    ( strf::basic_outbuff<CharT>&, std::size_t count, char32_t ch );

namespace detail {

template <typename CharT>
void trivial_fill_f
    ( strf::basic_outbuff<CharT>& ob, std::size_t count, char32_t ch )
{
    // same as strf::detail::write_fill<CharT>
    CharT narrow_ch = static_cast<CharT>(ch);
    STRF_IF_LIKELY (count <= ob.space()) {
        strf::detail::str_fill_n<CharT>(ob.pointer(), count, narrow_ch);
        ob.advance(count);
    } else {
        write_fill_continuation<CharT>(ob, count, narrow_ch);
    }
}

} // namespace detail

struct codepoints_count_result {
    std::size_t count;
    std::size_t pos;
};

template <typename CharT>
using codepoints_fast_count_f = strf::codepoints_count_result (*)
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count );

template <typename CharT>
using codepoints_robust_count_f = strf::codepoints_count_result (*)
    ( const CharT* src
    , std::size_t src_size
    , std::size_t max_count
    , strf::surrogate_policy surr_poli );

template <typename CharT>
using decode_char_f = char32_t (*)
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
        ( strf::basic_outbuff<DestCharT>& ob
        , const SrcCharT* src
        , std::size_t src_size
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli ) const
    {
        transcode_func_(ob, src, src_size, inv_seq_notifier, surr_poli);
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
    const char* name;
    strf::charset_id id;
    char32_t replacement_char;
    std::size_t replacement_char_size;
    strf::validate_f validate_func;
    strf::encoded_char_size_f encoded_char_size_func;
    strf::encode_char_f<CharT> encode_char_func;
    strf::encode_fill_f<CharT> encode_fill_func;
    strf::codepoints_fast_count_f<CharT> codepoints_fast_count_func;
    strf::codepoints_robust_count_f<CharT> codepoints_robust_count_func;
    strf::write_replacement_char_f<CharT> write_replacement_char_func;
    strf::decode_char_f<CharT> decode_char_func;

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
    void* find_transcoder_from_char8 = nullptr;
    void* find_transcoder_to_char8 = nullptr;

#endif
};

template <typename CharT>
class dynamic_charset
{
public:

    using char_type = CharT;

    dynamic_charset(const dynamic_charset& ) = default;

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

    STRF_HD dynamic_charset& operator=(const dynamic_charset& other) noexcept
    {
        data_ = other.data_;
        return *this;
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
    STRF_HD char_type* encode_char(char_type* dest, char32_t ch) const // noexcept
    {
        return data_->encode_char_func(dest, ch);
    }
    STRF_HD void encode_fill
        ( strf::basic_outbuff<CharT>& ob, std::size_t count, char32_t ch ) const
    {
        data_->encode_fill_func(ob, count, ch);
    }
    STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const char_type* src, std::size_t src_size, std::size_t max_count ) const
    {
        return data_->codepoints_fast_count_func(src, src_size, max_count);
    }
    STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const char_type* src, std::size_t src_size
        , std::size_t max_count, strf::surrogate_policy surr_poli ) const
    {
        return data_->codepoints_robust_count_func(src, src_size, max_count, surr_poli);
    }
    STRF_HD void write_replacement_char(strf::basic_outbuff<CharT>& ob) const
    {
        data_->write_replacement_char_func(ob);
    }
    STRF_HD char32_t decode_char(char_type ch) const
    {
        return data_->decode_char_func(ch);
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
    using src_char_type = typename SrcCharset::char_type;
    using dest_char_type = typename DestCharset::char_type;

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
constexpr bool has_static_transcoder =
    has_static_transcoder_impl<SrcCharset, DestCharset>::value;

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
constexpr bool has_find_transcoder_to =
    has_find_transcoder_to_impl<DestCharT, SrcCharset>::value;

template <typename SrcCharT, typename DestCharset>
constexpr bool has_find_transcoder_from =
    has_find_transcoder_from_impl<SrcCharT, DestCharset>::value;


template <bool HasFindTo, bool HasFindFrom, typename Transcoder>
struct transcoder_finder_2;

template <typename Transcoder>
struct transcoder_finder_2<true, true, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        constexpr strf::tag<typename SrcCharset::char_type> src_tag;
        constexpr strf::tag<typename DestCharset::char_type> dest_tag;
        auto t = src_cs.find_transcoder_to(dest_tag, dest_cs.id());
        if (t.transcode_func() != nullptr) {
            return t;
        }
        return dest_cs.find_transcoder_from(src_tag, src_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<true, false, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        constexpr strf::tag<typename DestCharset::char_type> dest_tag;
        return src_cs.find_transcoder_to(dest_tag, dest_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, true, Transcoder>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD Transcoder find(SrcCharset src_cs, DestCharset dest_cs)
    {
        constexpr strf::tag<typename SrcCharset::char_type> src_tag;
        return dest_cs.find_transcoder_from(src_tag, src_cs.id());
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
    constexpr static STRF_HD auto find
        (SrcCharset src_cs, DestCharset )
    {
        return src_cs.to_u32();
    }
};

template <typename DestCharT>
struct transcoder_finder<false, char32_t, DestCharT>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto find
        (SrcCharset, DestCharset dest_cs) noexcept
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
        if (src_cs.id() == dest_cs.id()) {
            return strf::dynamic_transcoder<CharT, CharT>
                { src_cs.sanitizer() };
        }
        return strf::detail::transcoder_finder_2
            < strf::detail::has_find_transcoder_to<CharT, SrcCharset>
            , strf::detail::has_find_transcoder_from<CharT, DestCharset>
            , strf::dynamic_transcoder<CharT, CharT> >
            ::find(src_cs, dest_cs);
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
            < strf::detail::has_find_transcoder_to<DestCharT, SrcCharset>
            , strf::detail::has_find_transcoder_from<SrcCharT, DestCharset>
            , strf::dynamic_transcoder<SrcCharT, DestCharT> >
            ::find(src_cs, dest_cs);
    }
};

} // namespace detail

template <typename SrcCharset, typename DestCharset>
constexpr STRF_HD decltype(auto) find_transcoder
    ( SrcCharset src_cs, DestCharset dest_cs )
{
    return detail::transcoder_finder
        < strf::detail::has_static_transcoder<SrcCharset, DestCharset>
        , typename SrcCharset::char_type
        , typename DestCharset::char_type >
        ::find(src_cs, dest_cs);
}

namespace detail {

template <typename DestCharT>
class buffered_encoder: public strf::basic_outbuff<char32_t>
{
public:

    STRF_HD buffered_encoder
        ( strf::transcode_f<char32_t, DestCharT> func
        , strf::basic_outbuff<DestCharT>& ob
        , strf::invalid_seq_notifier inv_seq_notifier
        , strf::surrogate_policy surr_poli )
        : strf::basic_outbuff<char32_t>( buff_, buff_size_ )
        , transcode_(func)
        , ob_(ob)
        , inv_seq_notifier_(inv_seq_notifier)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto p = this->pointer();
        STRF_IF_LIKELY (p != buff_ && ob_.good()) {
            transcode_( ob_, buff_, static_cast<std::size_t>(p - buff_)
                      , inv_seq_notifier_, surr_poli_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_f<char32_t, DestCharT> transcode_;
    strf::basic_outbuff<DestCharT>& ob_;
    strf::invalid_seq_notifier inv_seq_notifier_;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};


template <typename DestCharT>
STRF_HD void buffered_encoder<DestCharT>::recycle()
{
    auto p = this->pointer();
    this->set_pointer(buff_);
    STRF_IF_LIKELY (p != buff_ && ob_.good()) {
        this->set_good(false);
        transcode_( ob_, buff_, static_cast<std::size_t>(p - buff_)
                  , inv_seq_notifier_, surr_poli_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::basic_outbuff<char32_t>
{
public:

    STRF_HD buffered_size_calculator
        ( strf::transcode_size_f<char32_t> func, strf::surrogate_policy surr_poli )
        : strf::basic_outbuff<char32_t>(buff_, buff_size_)
        , size_func_(func)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD std::size_t get_sum()
    {
        recycle();
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
    auto p = this->pointer();
    STRF_IF_LIKELY (p != buff_) {
        this->set_pointer(buff_);
        sum_ += size_func_(buff_, static_cast<std::size_t>(p - buff_), surr_poli_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

template<typename SrcCharT, typename DestCharT>
STRF_HD void decode_encode
    ( strf::basic_outbuff<DestCharT>& ob
    , strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_f<char32_t, DestCharT> from_u32
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_encoder<DestCharT> tmp{from_u32, ob, inv_seq_notifier, surr_poli};
    to_u32(tmp, src, src_size, inv_seq_notifier, surr_poli);
    tmp.finish();
}

template<typename SrcCharT>
STRF_HD std::size_t decode_encode_size
    ( strf::transcode_f<SrcCharT, char32_t> to_u32
    , strf::transcode_size_f<char32_t> size_calc_func
    , const SrcCharT* src
    , std::size_t src_size
    , strf::invalid_seq_notifier inv_seq_notifier
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, surr_poli};
    to_u32(acc, src, src_size, inv_seq_notifier, surr_poli);
    return acc.get_sum();
}


// backwards compatibility:

#if defined(STRF_DONT_DEPRECATE_CHAR_ENCODING)
#  define STRF_CHAR_ENCODING_DEPRECATED
#else
#  define STRF_CHAR_ENCODING_DEPRECATED [[deprecated]]
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

