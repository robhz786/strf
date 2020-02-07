#ifndef STRF_DETAIL_FACETS_CHARSETS_HPP
#define STRF_DETAIL_FACETS_CHARSETS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>

namespace strf {

template <typename> class facet_trait;

enum class invalid_seq_policy
{
    replace, stop
};

class invalid_sequence: public strf::stringify_error
{
    using strf::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: charset conversion error";
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
class facet_trait<strf::invalid_seq_policy>
{
public:
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
class facet_trait<strf::surrogate_policy>
{
public:
    using category = strf::surrogate_policy_c;
    static constexpr bool store_by_value = true;
};

enum class charset_id : unsigned
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

constexpr std::size_t invalid_char_len = (std::size_t)-1;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using transcode_func = void (*)
    ( strf::underlying_outbuf<DestCharSize>& ob
    , const strf::underlying_char_type<SrcCharSize>* begin
    , const strf::underlying_char_type<SrcCharSize>* end
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli );

template <std::size_t SrcCharSize>
using transcode_size_func = std::size_t (*)
    ( const strf::underlying_char_type<SrcCharSize>* begin
    , const strf::underlying_char_type<SrcCharSize>* end
    , strf::surrogate_policy surr_poli );

template <std::size_t CharSize>
using write_replacement_char_func = void (*)
    ( strf::underlying_outbuf<CharSize>& );

// assume allow_surragates::lax
using validate_func = std::size_t (*)(char32_t ch);

// assume allow_surragates::lax and strf::invalid_seq_policy::replace
using encoded_char_size_func = std::size_t (*)(char32_t ch);

// assume allow_surragates::lax and strf::invalid_seq_policy::replace
template <std::size_t CharSize>
using encode_char_func = strf::underlying_char_type<CharSize>*(*)
    ( strf::underlying_char_type<CharSize>* dest, char32_t ch );

template <std::size_t CharSize>
using encode_fill_func = void (*)
    ( strf::underlying_outbuf<CharSize>&, std::size_t count, char32_t ch
    , strf::invalid_seq_policy inv_seq_poli, strf::surrogate_policy surr_poli );

template <std::size_t CharSize>
using codepoints_count_func = std::size_t (*)
    ( const strf::underlying_char_type<CharSize>* begin
    , const strf::underlying_char_type<CharSize>* end
    , std::size_t max_count );

template <std::size_t CharSize>
using decode_single_char_func = char32_t (*)
    ( strf::underlying_char_type<CharSize> );

template <typename CharT>
struct charset_c;

template <strf::charset_id>
class static_charset;

template <strf::charset_id Src, strf::charset_id Dest>
class static_transcoder;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_transcoder
{
public:

    template <strf::charset_id Src, strf::charset_id Dest>
    constexpr dynamic_transcoder(strf::static_transcoder<Src, Dest> t) noexcept
        : transcode(t.transcode)
        , necessary_size(t.necessary_size)
    {
    }

    constexpr dynamic_transcoder() noexcept
        : transcode(nullptr)
        , necessary_size(nullptr)
    {
    }

    strf::transcode_func<SrcCharSize, DestCharSize> transcode;
    strf::transcode_size_func<SrcCharSize> necessary_size;
};

namespace detail {

template <typename SrcCharset, typename DestCharset>
constexpr STRF_HD
strf::dynamic_transcoder<SrcCharset::char_size, DestCharset::char_size>
get_transcoder_impl(strf::rank<0>, const SrcCharset&, const DestCharset&, ...)
{
    return {};
}

template < strf::charset_id Src
         , strf::charset_id Dest
         , std::size_t SrcCharSize = static_charset<Src>::char_size
         , std::size_t DestCharSize = static_charset<Dest>::char_size >
constexpr STRF_HD strf::static_transcoder<Src, Dest> get_transcoder_impl
    ( strf::rank<1>
    , strf::static_charset<Src>
    , strf::static_charset<Dest>
    , std::index_sequence<strf::static_transcoder<Src, Dest>::char_size>* )
{
    return {};
}

} // namespace detail

template
    < typename SrcCharset
    , typename DestCharset
    , std::size_t SrcCharSize = SrcCharset::char_size
    , std::size_t DestCharSize = DestCharset::char_size >
constexpr STRF_HD
decltype(auto) get_transcoder(const SrcCharset& src_cs, const DestCharset& dest_cs)
{
    return detail::get_transcoder_impl(strf::rank<2>(), src_cs, dest_cs, nullptr);
}

namespace detail {

template <std::size_t DestCharSize>
class buffered_encoder: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_encoder
        ( strf::transcode_func<4, DestCharSize> func
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
        auto p = this->pos();
        if (p != buff_ && ob_.good()) {
            transcode_(ob_, buff_, p, inv_seq_poli_, surr_poli_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_func<4, DestCharSize> transcode_;
    strf::underlying_outbuf<DestCharSize>& ob_;
    strf::invalid_seq_policy inv_seq_poli_;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};


template <std::size_t DestCharSize>
STRF_HD void buffered_encoder<DestCharSize>::recycle()
{
    auto p = this->pos();
    this->set_pos(buff_);
    if (p != buff_ && ob_.good()) {
        this->set_good(false);
        transcode_(ob_, buff_, p, inv_seq_poli_, surr_poli_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_size_calculator
        ( strf::transcode_size_func<4> func, strf::surrogate_policy surr_poli )
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

    strf::transcode_size_func<4> size_func_;
    std::size_t sum_ = 0;
    strf::surrogate_policy surr_poli_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD void buffered_size_calculator::recycle()
{
    auto p = this->pos();
    if (p != buff_) {
        this->set_pos(buff_);
        sum_ += size_func_(buff_, p, surr_poli_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

template<std::size_t SrcCharSize, std::size_t DestCharSize>
STRF_HD void decode_encode
    ( strf::underlying_outbuf<DestCharSize>& ob
    , strf::transcode_func<SrcCharSize, 4> to_u32
    , strf::transcode_func<4, DestCharSize> from_u32
    , const underlying_char_type<SrcCharSize>* str
    , const underlying_char_type<SrcCharSize>* str_end
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_encoder<DestCharSize> tmp{from_u32, ob, inv_seq_poli, surr_poli};
    to_u32(tmp, str, str_end, inv_seq_poli, surr_poli);
    tmp.finish();
}

template<std::size_t SrcCharSize>
STRF_HD std::size_t decode_encode_size
    ( strf::transcode_func<SrcCharSize, 4> to_u32
    , strf::transcode_size_func<4> size_calc_func
    , const underlying_char_type<SrcCharSize>* str
    , const underlying_char_type<SrcCharSize>* str_end
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, surr_poli};
    to_u32(acc, str, str_end, inv_seq_poli, surr_poli);
    return acc.get_sum();
}

} // namespace strf

#endif  // STRF_DETAIL_FACETS_CHARSETS_HPP

