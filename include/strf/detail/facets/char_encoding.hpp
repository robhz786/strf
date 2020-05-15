#ifndef STRF_DETAIL_FACETS_CHAR_ENCODING_HPP
#define STRF_DETAIL_FACETS_CHAR_ENCODING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>

namespace strf {

template <typename> struct facet_traits;

enum class invalid_seq_policy
{
    replace, stop
};

class invalid_sequence: public strf::stringify_error
{
    using strf::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "strf: encoding conversion error";
    }
};

namespace detail {

inline STRF_HD void handle_invalid_sequence()
{

#if defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
    throw strf::invalid_sequence();
#else // defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)

#  ifndef __CUDA_ARCH__
    std::abort();
#  else
    asm("trap;");
#  endif

#endif // defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
}

} // namespace detail

struct invalid_seq_policy_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::invalid_seq_policy get_default() noexcept
    {
        return strf::invalid_seq_policy::replace;
    }
};

template <>
struct facet_traits<strf::invalid_seq_policy>
{
    using category = strf::invalid_seq_policy_c;
    static constexpr bool store_by_value = true;
};

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
    static constexpr bool store_by_value = true;
};

enum class char_encoding_id : unsigned
{
    utf8,
    mutf8,

    utf16_little_endian,
    utf16_big_endian,
    utf16,

    utf32_little_endian,
    utf32_big_endian,
    utf32,

    ascii,

    iso_8859_1,
    iso_8859_2,
    iso_8859_3,
    iso_8859_4,
    iso_8859_5,
    iso_8859_6,
    iso_8859_7,
    iso_8859_8,
    iso_8859_9,
    iso_8859_10,
    iso_8859_11,
    iso_8859_12,
    iso_8859_13,
    iso_8859_14,
    iso_8859_15,
    iso_8859_16,

    windows_1250,
    windows_1251,
    windows_1252,
    windows_1253,
    windows_1254,
    windows_1255,
    windows_1256,
    windows_1257,
    windows_1258,

    ebcdic_cp37,
    ebcdic_cp930,
    ebcdic_cp1047,

    cp437,
    cp720,
    cp737,
    cp850,
    cp852,
    cp855,
    cp857,
    cp858,
    cp860,
    cp861,
    cp862,
    cp863,
    cp865,
    cp866,
    cp869,
    cp872,

    mac_os_roman,

    koi8_r,
    koi8_u,
    koi7,
    mik,
    iscii,
    tscii,
    viscii,

    iso_2022_jp,
    iso_2022_jp_1,
    iso_2022_jp_2,
    iso_2022_jp_2004,
    iso_2022_kr,
    iso_2022_cn,
    iso_2022_cn_ext,

    // etc ... TODO
    // https://en.wikipedia.org/wiki/Character_encoding#Common_character_encodings
    // https://docs.python.org/2.4/lib/standard-encodings.html
};

template <typename CharT>
struct char_encoding_c;

template <strf::char_encoding_id>
class static_underlying_char_encoding;

template <strf::char_encoding_id Src, strf::char_encoding_id Dest>
class static_underlying_transcoder;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_underlying_transcoder;

constexpr std::size_t invalid_char_len = (std::size_t)-1;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using transcode_f = void (*)
    ( strf::underlying_outbuf<DestCharSize>& ob
    , const strf::underlying_char_type<SrcCharSize>* src
    , std::size_t src_size
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli );

template <std::size_t SrcCharSize>
using transcode_size_f = std::size_t (*)
    ( const strf::underlying_char_type<SrcCharSize>* src
    , std::size_t src_size
    , strf::surrogate_policy surr_poli );

template <std::size_t CharSize>
using write_replacement_char_f = void (*)
    ( strf::underlying_outbuf<CharSize>& );

// assume surragate_policy::lax
using validate_f = std::size_t (*)(char32_t ch);

// assume surragates_policy::lax and strf::invalid_seq_policy::replace
using encoded_char_size_f = std::size_t (*)(char32_t ch);

// assume surrogate_policy::lax and strf::invalid_seq_policy::replace
template <std::size_t CharSize>
using encode_char_f = strf::underlying_char_type<CharSize>*(*)
    ( strf::underlying_char_type<CharSize>* dest, char32_t ch );

template <std::size_t CharSize>
using encode_fill_f = void (*)
    ( strf::underlying_outbuf<CharSize>&, std::size_t count, char32_t ch
    , strf::invalid_seq_policy inv_seq_poli, strf::surrogate_policy surr_poli );

struct codepoints_count_result {
    std::size_t count;
    std::size_t pos;
};

template <std::size_t CharSize>
using codepoints_fast_count_f = strf::codepoints_count_result (*)
    ( const strf::underlying_char_type<CharSize>* src
    , std::size_t src_size
    , std::size_t max_count );

template <std::size_t CharSize>
using codepoints_robust_count_f = strf::codepoints_count_result (*)
    ( const strf::underlying_char_type<CharSize>* src
    , std::size_t src_size
    , std::size_t max_count
    , strf::surrogate_policy surr_poli );

template <std::size_t CharSize>
using decode_char_f = char32_t (*)
    ( strf::underlying_char_type<CharSize> );

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using find_transcoder_f =
    strf::dynamic_underlying_transcoder<SrcCharSize, DestCharSize> (*)
    ( strf::char_encoding_id );

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_underlying_transcoder
{
public:

    template <strf::char_encoding_id Src, strf::char_encoding_id Dest>
    constexpr explicit STRF_HD dynamic_underlying_transcoder
        ( strf::static_underlying_transcoder<Src, Dest> t ) noexcept
        : transcode_func_(t.transcode_func())
        , transcode_size_func_(t.transcode_size_func())
    {
    }

    constexpr STRF_HD dynamic_underlying_transcoder() noexcept
        : transcode_func_(nullptr)
        , transcode_size_func_(nullptr)
    {
    }

    STRF_HD void transcode
        ( strf::underlying_outbuf<DestCharSize>& ob
        , const strf::underlying_char_type<SrcCharSize>* src
        , std::size_t src_size
        , strf::invalid_seq_policy inv_seq_poli
        , strf::surrogate_policy surr_poli ) const
    {
        transcode_func_(ob, src, src_size, inv_seq_poli, surr_poli);
    }

    STRF_HD std::size_t transcode_size
        ( const strf::underlying_char_type<SrcCharSize>* src
        , std::size_t src_size
        , strf::surrogate_policy surr_poli ) const
    {
        return transcode_size_func_(src, src_size, surr_poli);
    }

    constexpr STRF_HD strf::transcode_f<SrcCharSize, DestCharSize>
    transcode_func() const noexcept
    {
        return transcode_func_;
    }

    constexpr STRF_HD strf::transcode_size_f<SrcCharSize>
    transcode_size_func() const noexcept
    {
        return transcode_size_func_;
    }

private:

    strf::transcode_f<SrcCharSize, DestCharSize> transcode_func_;
    strf::transcode_size_f<SrcCharSize> transcode_size_func_;
};

template <std::size_t CharSize>
struct dynamic_underlying_char_encoding_data
{
    const char* name;
    strf::char_encoding_id id;
    char32_t replacement_char;
    std::size_t replacement_char_size;
    strf::validate_f validate_func;
    strf::encoded_char_size_f encoded_char_size_func;
    strf::encode_char_f<CharSize> encode_char_func;
    strf::encode_fill_f<CharSize> encode_fill_func;
    strf::codepoints_fast_count_f<CharSize> codepoints_fast_count_func;
    strf::codepoints_robust_count_f<CharSize> codepoints_robust_count_func;
    strf::write_replacement_char_f<CharSize> write_replacement_char_func;
    strf::decode_char_f<CharSize> decode_char_func;

    strf::dynamic_underlying_transcoder<4, CharSize> from_u32;
    strf::dynamic_underlying_transcoder<CharSize, 4> to_u32;
    strf::dynamic_underlying_transcoder<CharSize, CharSize> sanitizer;

    strf::find_transcoder_f<1, CharSize> transcoder_from_1byte_encoding;
    strf::find_transcoder_f<2, CharSize> transcoder_from_2bytes_encoding;

    strf::find_transcoder_f<CharSize, 1> transcoder_to_1byte_encoding;
    strf::find_transcoder_f<CharSize, 2> transcoder_to_2bytes_encoding;
};

template <std::size_t CharSize>
class dynamic_underlying_char_encoding
{
    using char_type_ = strf::underlying_char_type<CharSize>;

public:

    static constexpr std::size_t char_size = CharSize;

    dynamic_underlying_char_encoding(const dynamic_underlying_char_encoding& ) = default;

    STRF_HD dynamic_underlying_char_encoding
        ( const strf::dynamic_underlying_char_encoding_data<CharSize>& data )
        : data_(&data)
    {
    }

    STRF_HD dynamic_underlying_char_encoding& operator=(const dynamic_underlying_char_encoding& other) noexcept
    {
        data_ = other.data_;
        return *this;
    }
    STRF_HD bool operator==(const dynamic_underlying_char_encoding& other) const noexcept
    {
        return data_->id == other.data_->id;
    }
    STRF_HD bool operator!=(const dynamic_underlying_char_encoding& other) const noexcept
    {
        return data_->id != other.data_->id;
    }

    STRF_HD void swap(dynamic_underlying_char_encoding& other) noexcept
    {
        auto tmp = data_;
        data_ = other.data_;
        other.data_ = tmp;
    }
    STRF_HD const char* name() const noexcept
    {
        return data_->name;
    };
    constexpr STRF_HD strf::char_encoding_id id() const noexcept
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
    STRF_HD char_type_* encode_char(char_type_* dest, char32_t ch) const // noexcept
    {
        return data_->encode_char_func(dest, ch);
    }
    STRF_HD void encode_fill
        ( strf::underlying_outbuf<CharSize>& ob, std::size_t count, char32_t ch
        , strf::invalid_seq_policy inv_seq_poli, strf::surrogate_policy surr_poli ) const
    {
        data_->encode_fill_func(ob, count, ch, inv_seq_poli, surr_poli);
    }
    STRF_HD strf::codepoints_count_result codepoints_fast_count
        ( const char_type_* src, std::size_t src_size, std::size_t max_count ) const
    {
        return data_->codepoints_fast_count_func(src, src_size, max_count);
    }
    STRF_HD strf::codepoints_count_result codepoints_robust_count
        ( const char_type_* src, std::size_t* src_size
        , std::size_t max_count ) const
    {
        return data_->codepoints_robust_count_func(src, src_size, max_count);
    }
    STRF_HD void write_replacement_char(strf::underlying_outbuf<CharSize>& ob) const
    {
        data_->write_replacement_char_func(ob);
    }
    STRF_HD char32_t decode_char(char_type_ ch) const
    {
        return data_->decode_char_func(ch);
    }

    STRF_HD strf::encode_char_f<char_size> encode_char_func() const noexcept
    {
        return data_->encode_char_func;
    }
    STRF_HD strf::encode_fill_f<char_size> encode_fill_func() const noexcept
    {
        return data_->encode_fill_func;
    }
    STRF_HD strf::write_replacement_char_f<char_size>
    write_replacement_char_func() const noexcept
    {
        return data_->write_replacement_char_func;
    }

    strf::dynamic_underlying_transcoder<4, CharSize> from_u32() const
    {
        return data_->from_u32;
    }
    strf::dynamic_underlying_transcoder<CharSize, 4> to_u32() const
    {
        return data_->to_u32;
    }
    strf::dynamic_underlying_transcoder<CharSize, CharSize> sanitizer() const
    {
        return data_->sanitizer;
    }

    strf::dynamic_underlying_transcoder<CharSize, 1> find_transcoder_to
        ( std::integral_constant<std::size_t, 1>, strf::char_encoding_id id) const
    {
        if (data_->transcoder_to_1byte_encoding) {
            return data_->transcoder_to_1byte_encoding(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<CharSize, 2> find_transcoder_to
        ( std::integral_constant<std::size_t, 2>, strf::char_encoding_id id) const
    {
        if (data_->transcoder_to_2bytes_encoding) {
            return data_->transcoder_to_2bytes_encoding(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<1, CharSize> find_transcoder_from
        ( std::integral_constant<std::size_t, 1>, strf::char_encoding_id id) const
    {
        if (data_->transcoder_from_1byte_encoding) {
            return data_->transcoder_from_1byte_encoding(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<2, CharSize> find_transcoder_from
        ( std::integral_constant<std::size_t, 2>, strf::char_encoding_id id) const
    {
        if (data_->transcoder_from_2bytes_encoding) {
            return data_->transcoder_from_2bytes_encoding(id);
        }
        return {};
    }

    const strf::dynamic_underlying_char_encoding_data<CharSize>* data_;
};

template <typename CharT>
struct char_encoding_c;

template <typename CharT, strf::char_encoding_id CSID>
class static_char_encoding: public strf::static_underlying_char_encoding<CSID>
{
public:
    static_assert( sizeof(CharT) == strf::static_underlying_char_encoding<CSID>::char_size
                 , "Incompatible character size" );
    using category = strf::char_encoding_c<CharT>;
};

} // namespace strf

#include <strf/detail/utf.hpp>

namespace strf {

template <typename CharT>
struct char_encoding_c
{
    static constexpr bool constrainable = false;
    static constexpr STRF_HD strf::utf<CharT> get_default() noexcept
    {
        return {};
    }
};

namespace detail {

template <typename SrcEncoding, typename DestEncoding>
class has_static_transcoder_impl
{
    template <strf::char_encoding_id SrcId, strf::char_encoding_id DestId>
    static
    decltype(strf::static_underlying_transcoder<SrcId, DestId>(), std::true_type())
    test_( strf::static_underlying_char_encoding<SrcId>*
         , strf::static_underlying_char_encoding<DestId>* )
    {
        return {};
    }

    static std::false_type test_(...)
    {
        return {};
    }

    using result_ = decltype(test_((SrcEncoding*)0, (DestEncoding*)0));

public:

    static constexpr bool value = result_::value;
};

template <typename SrcEncoding, typename DestEncoding>
constexpr bool has_static_transcoder =
    has_static_transcoder_impl<SrcEncoding, DestEncoding>::value;

template <std::size_t SrcCharSize, typename DestEncoding>
class has_find_transcoder_from_impl
{
    template <std::size_t N>
    using itag = std::integral_constant<std::size_t, N>;

    template <std::size_t S, typename D>
    static auto test(itag<S>, const D* d)
    -> decltype( d->find_transcoder_from(itag<S>(), strf::char_encoding_id::utf8)
               , std::true_type() );

    template <std::size_t S, typename D>
    static std::false_type test(...);

public:

    static constexpr bool value
    = decltype(test<SrcCharSize, DestEncoding>(itag<SrcCharSize>(), 0))::value;
};

template <std::size_t DestCharSize, typename SrcEncoding>
class has_find_transcoder_to_impl
{
    template <std::size_t N>
    using itag = std::integral_constant<std::size_t, N>;

    template <std::size_t D, typename S>
    static auto test(itag<D>, const S* s)
    -> decltype( s->find_transcoder_from(itag<D>(), strf::char_encoding_id::utf8)
               , std::true_type() );

    template <std::size_t D, typename S>
    static std::false_type test(...);

public:
    static constexpr bool value
    = decltype(test<DestCharSize, SrcEncoding>(itag<DestCharSize>(), 0))::value;
};

template <std::size_t DestCharSize, typename SrcEncoding>
constexpr bool has_find_transcoder_to =
    has_find_transcoder_to_impl<DestCharSize, SrcEncoding>::value;

template <std::size_t SrcCharSize, typename DestEncoding>
constexpr bool has_find_transcoder_from =
    has_find_transcoder_from_impl<SrcCharSize, DestEncoding>::value;


template <bool HasFindTo, bool HasFindFrom, typename Transcoder>
struct transcoder_finder_2;

template <typename Transcoder>
struct transcoder_finder_2<true, true, Transcoder>
{
    template <std::size_t N>
    using itag = std::integral_constant<std::size_t, N>;

    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD Transcoder find(SrcEncoding src_cs, DestEncoding dest_cs)
    {
        auto t = src_cs.find_transcoder_to(itag<DestEncoding::char_size>(), dest_cs.id());
        if (t.transcode_func() != nullptr) {
            return t;
        }
        return dest_cs.find_transcoder_from(itag<SrcEncoding::char_size>(), src_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<true, false, Transcoder>
{
    template <std::size_t N>
    using itag = std::integral_constant<std::size_t, N>;

    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD Transcoder find(SrcEncoding src_cs, DestEncoding dest_cs)
    {
        return src_cs.find_transcoder_to(itag<DestEncoding::char_size>(), dest_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, true, Transcoder>
{
    template <std::size_t N>
    using itag = std::integral_constant<std::size_t, N>;

    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD Transcoder find(SrcEncoding src_cs, DestEncoding dest_cs)
    {
        return dest_cs.find_transcoder_from(itag<SrcEncoding::char_size>(), src_cs.id());
    }
};

template <typename Transcoder>
struct transcoder_finder_2<false, false, Transcoder>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD Transcoder find(SrcEncoding, DestEncoding)
    {
        return {};
    }
};

template < bool HasStaticTranscoder
         , std::size_t SrcCharSize
         , std::size_t DestCharSize >
struct transcoder_finder;

template <std::size_t SrcCharSize, std::size_t DestCharSize >
struct transcoder_finder<true, SrcCharSize, DestCharSize>
{
    template < strf::char_encoding_id Src, strf::char_encoding_id Dest>
    constexpr static STRF_HD strf::static_underlying_transcoder<Src, Dest> find
        ( strf::static_underlying_char_encoding<Src>
        , strf::static_underlying_char_encoding<Dest> ) noexcept
    {
        return {};
    }
};

template <>
struct transcoder_finder<false, 4, 4>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD strf::utf32_to_utf32 find
        (SrcEncoding, DestEncoding )
    {
        return {};
    }
};

template <std::size_t SrcCharSize>
struct transcoder_finder<false, SrcCharSize, 4>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD auto find
        (SrcEncoding src_cs, DestEncoding )
    {
        return src_cs.to_u32();
    }
};

template <std::size_t DestCharSize>
struct transcoder_finder<false, 4, DestCharSize>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD auto find
        (const SrcEncoding&, const DestEncoding& dest_cs) noexcept
    {
        return dest_cs.from_u32();
    }
};

template <std::size_t CharSize>
struct transcoder_finder<false, CharSize, CharSize>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD
    strf::dynamic_underlying_transcoder<CharSize, CharSize>
    find(SrcEncoding src_cs, DestEncoding dest_cs )
    {
        if (src_cs.id() == dest_cs.id()) {
            return strf::dynamic_underlying_transcoder<CharSize, CharSize>
                { src_cs.sanitizer() };
        }
        return strf::detail::transcoder_finder_2
            < strf::detail::has_find_transcoder_to<CharSize, SrcEncoding>
            , strf::detail::has_find_transcoder_from<CharSize, DestEncoding>
            , strf::dynamic_underlying_transcoder<CharSize, CharSize> >
            ::find(src_cs, dest_cs);
    }
};

template <std::size_t SrcCharSize, std::size_t DestCharSize >
struct transcoder_finder<false, SrcCharSize, DestCharSize>
{
    template <typename SrcEncoding, typename DestEncoding>
    constexpr static STRF_HD
    strf::dynamic_underlying_transcoder<SrcCharSize, DestCharSize>
    find(SrcEncoding src_cs, DestEncoding dest_cs )
    {
        return strf::detail::transcoder_finder_2
            < strf::detail::has_find_transcoder_to<DestCharSize, SrcEncoding>
            , strf::detail::has_find_transcoder_from<SrcCharSize, DestEncoding>
            , strf::dynamic_underlying_transcoder<SrcCharSize, DestCharSize> >
            ::find(src_cs, dest_cs);
    }
};

} // namespace detail

template <typename SrcEncoding, typename DestEncoding>
constexpr STRF_HD decltype(auto) find_transcoder
    ( const SrcEncoding& src_cs, const DestEncoding& dest_cs )
{
    return detail::transcoder_finder
        < strf::detail::has_static_transcoder<SrcEncoding, DestEncoding>
        , SrcEncoding::char_size
        , DestEncoding::char_size>
        ::find(src_cs, dest_cs);
}

namespace detail {

template <std::size_t DestCharSize>
class buffered_encoder: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_encoder
        ( strf::transcode_f<4, DestCharSize> func
        , strf::underlying_outbuf<DestCharSize>& ob
        , strf::invalid_seq_policy inv_seq_poli
        , strf::surrogate_policy surr_poli )
        : strf::underlying_outbuf<4>( buff_, buff_size_ )
        , transcode_(func)
        , ob_(ob)
        , inv_seq_poli_(inv_seq_poli)
        , surr_poli_(surr_poli)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto p = this->pointer();
        if (p != buff_ && ob_.good()) {
            transcode_( ob_, buff_, static_cast<std::size_t>(p - buff_)
                      , inv_seq_poli_, surr_poli_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_f<4, DestCharSize> transcode_;
    strf::underlying_outbuf<DestCharSize>& ob_;
    strf::invalid_seq_policy inv_seq_poli_;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};


template <std::size_t DestCharSize>
STRF_HD void buffered_encoder<DestCharSize>::recycle()
{
    auto p = this->pointer();
    this->set_pointer(buff_);
    if (p != buff_ && ob_.good()) {
        this->set_good(false);
        transcode_( ob_, buff_, static_cast<std::size_t>(p - buff_)
                  , inv_seq_poli_, surr_poli_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_size_calculator
        ( strf::transcode_size_f<4> func, strf::surrogate_policy surr_poli )
        : strf::underlying_outbuf<4>(buff_, buff_size_)
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

    strf::transcode_size_f<4> size_func_;
    std::size_t sum_ = 0;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD void buffered_size_calculator::recycle()
{
    auto p = this->pointer();
    if (p != buff_) {
        this->set_pointer(buff_);
        sum_ += size_func_(buff_, static_cast<std::size_t>(p - buff_), surr_poli_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

template<std::size_t SrcCharSize, std::size_t DestCharSize>
STRF_HD void decode_encode
    ( strf::underlying_outbuf<DestCharSize>& ob
    , strf::transcode_f<SrcCharSize, 4> to_u32
    , strf::transcode_f<4, DestCharSize> from_u32
    , const underlying_char_type<SrcCharSize>* src
    , std::size_t src_size
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_encoder<DestCharSize> tmp{from_u32, ob, inv_seq_poli, surr_poli};
    to_u32(tmp, src, src_size, inv_seq_poli, surr_poli);
    tmp.finish();
}

template<std::size_t SrcCharSize>
STRF_HD std::size_t decode_encode_size
    ( strf::transcode_f<SrcCharSize, 4> to_u32
    , strf::transcode_size_f<4> size_calc_func
    , const underlying_char_type<SrcCharSize>* src
    , std::size_t src_size
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, surr_poli};
    to_u32(acc, src, src_size, inv_seq_poli, surr_poli);
    return acc.get_sum();
}

template <typename CharT>
class dynamic_char_encoding: public strf::dynamic_underlying_char_encoding<sizeof(CharT)>
{
public:

    using category = strf::char_encoding_c<CharT>;

    explicit dynamic_char_encoding(const strf::dynamic_underlying_char_encoding<sizeof(CharT)>& u)
        : strf::dynamic_underlying_char_encoding<sizeof(CharT)>(u)
    {
    }

    template <strf::char_encoding_id EncodingID>
    explicit dynamic_char_encoding(strf::static_char_encoding<CharT, EncodingID> scs)
        : dynamic_char_encoding(scs.to_dynamic())
    {
    }
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_CHAR_ENCODING_HPP

