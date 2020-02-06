#ifndef STRF_DETAIL_FACETS_ENCODINGS_HPP
#define STRF_DETAIL_FACETS_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>

namespace strf {

template <typename> class facet_trait;

enum class encoding_error
{
    replace, stop
};

class encoding_failure: public strf::stringify_error
{
    using strf::stringify_error::stringify_error;

    const char* what() const noexcept override
    {
        return "Boost.Stringify: encoding conversion error";
    }
};

namespace detail {

inline STRF_HD void handle_encoding_failure()
{

#if defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
    throw strf::encoding_failure();
#else // defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)

#  ifndef __CUDA_ARCH__
    std::abort();
#  else
    asm("trap;");
#  endif

#endif // defined(__cpp_exceptions) && !defined(__CUDA_ARCH__)
}

} // namespace detail

struct encoding_error_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::encoding_error get_default() noexcept
    {
        return strf::encoding_error::replace;
    }
};

template <>
class facet_trait<strf::encoding_error>
{
public:
    using category = strf::encoding_error_c;
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

enum class encoding_id : unsigned
{
    eid_utf8,
    eid_mutf8,

    eid_utf16_little_endian,
    eid_utf16_big_endian,
    eid_utf16,

    eid_utf32_little_endian,
    eid_utf32_big_endian,
    eid_utf32,

    eid_pure_ascii,

    eid_iso_8859_1,
    eid_iso_8859_2,
    eid_iso_8859_3,
    eid_iso_8859_4,
    eid_iso_8859_5,
    eid_iso_8859_6,
    eid_iso_8859_7,
    eid_iso_8859_8,
    eid_iso_8859_9,
    eid_iso_8859_10,
    eid_iso_8859_11,
    eid_iso_8859_12,
    eid_iso_8859_13,
    eid_iso_8859_14,
    eid_iso_8859_15,
    eid_iso_8859_16,

    eid_windows_1250,
    eid_windows_1251,
    eid_windows_1252,
    eid_windows_1253,
    eid_windows_1254,
    eid_windows_1255,
    eid_windows_1256,
    eid_windows_1257,
    eid_windows_1258,

    eid_ebcdic_cp37,
    eid_ebcdic_cp930,
    eid_ebcdic_cp1047,

    eid_cp437,
    eid_cp720,
    eid_cp737,
    eid_cp850,
    eid_cp852,
    eid_cp855,
    eid_cp857,
    eid_cp858,
    eid_cp860,
    eid_cp861,
    eid_cp862,
    eid_cp863,
    eid_cp865,
    eid_cp866,
    eid_cp869,
    eid_cp872,

    eid_mac_os_roman,

    eid_koi8_r,
    eid_koi8_u,
    eid_koi7,
    eid_mik,
    eid_iscii,
    eid_tscii,
    eid_viscii,

    eid_iso_2022_jp,
    eid_iso_2022_jp_1,
    eid_iso_2022_jp_2,
    eid_iso_2022_jp_2004,
    eid_iso_2022_kr,
    eid_iso_2022_cn,
    eid_iso_2022_cn_ext,

    // etc ... TODO
    // https://en.wikipedia.org/wiki/Character_encoding#Common_character_encodings
    // https://docs.python.org/2.4/lib/standard-encodings.html
};

constexpr std::size_t invalid_char_len = (std::size_t)-1;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using transcode_func = void (*)
    ( strf::underlying_outbuf<DestCharSize>& ob
    , const strf::underlying_outbuf_char_type<SrcCharSize>* begin
    , const strf::underlying_outbuf_char_type<SrcCharSize>* end
    , strf::encoding_error err_hdl
    , strf::surrogate_policy allow_surr );

template <std::size_t SrcCharSize>
using transcode_size_func = std::size_t (*)
    ( const strf::underlying_outbuf_char_type<SrcCharSize>* begin
    , const strf::underlying_outbuf_char_type<SrcCharSize>* end
    , strf::surrogate_policy allow_surr );

template <std::size_t CharSize>
using write_replacement_char_func = void (*)
    ( strf::underlying_outbuf<CharSize>& );

// assume allow_surragates::lax
using validate_func = std::size_t (*)(char32_t ch);

// assume allow_surragates::lax and strf::encoding_error::replace
using encoded_char_size_func = std::size_t (*)(char32_t ch);

// assume allow_surragates::lax and strf::encoding_error::replace
template <std::size_t CharSize>
using encode_char_func = strf::underlying_outbuf_char_type<CharSize>*(*)
    ( strf::underlying_outbuf_char_type<CharSize>* dest, char32_t ch );

template <std::size_t CharSize>
using encode_fill_func = void (*)
    ( strf::underlying_outbuf<CharSize>&, std::size_t count, char32_t ch
    , strf::encoding_error err_hdl, strf::surrogate_policy allow_surr );

template <std::size_t CharSize>
using codepoints_count_func = std::size_t (*)
    ( const strf::underlying_outbuf_char_type<CharSize>* begin
    , const strf::underlying_outbuf_char_type<CharSize>* end
    , std::size_t max_count );

template <std::size_t CharSize>
using decode_single_char_func = char32_t (*)
    ( strf::underlying_outbuf_char_type<CharSize> );

template <typename CharT>
struct encoding_c;

template <strf::encoding_id>
class static_encoding;

template <strf::encoding_id Src, strf::encoding_id Dest>
class static_transcoder;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_transcoder
{
public:

    template <strf::encoding_id Src, strf::encoding_id Dest>
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

template <typename SrcEnc, typename DestEnc>
constexpr STRF_HD
strf::dynamic_transcoder<SrcEnc::char_size, DestEnc::char_size>
get_transcoder_impl(strf::rank<0>, const SrcEnc&, const DestEnc&, ...)
{
    return {};
}

template < strf::encoding_id Src
         , strf::encoding_id Dest
         , std::size_t SrcCharSize = static_encoding<Src>::char_size
         , std::size_t DestCharSize = static_encoding<Dest>::char_size >
constexpr STRF_HD strf::static_transcoder<Src, Dest> get_transcoder_impl
    ( strf::rank<1>
    , strf::static_encoding<Src>
    , strf::static_encoding<Dest>
    , std::index_sequence<strf::static_transcoder<Src, Dest>::char_size>* )
{
    return {};
}

} // namespace detail

template
    < typename SrcEnc
    , typename DestEnc
    , std::size_t SrcCharSize = SrcEnc::char_size
    , std::size_t DestCharSize = DestEnc::char_size >
constexpr STRF_HD
decltype(auto) get_transcoder(const SrcEnc& src_enc, const DestEnc& dest_enc)
{
    return detail::get_transcoder_impl(strf::rank<2>(), src_enc, dest_enc, nullptr);
}

namespace detail {

template <std::size_t DestCharSize>
class buffered_encoder: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_encoder
        ( strf::transcode_func<4, DestCharSize> func
        , strf::underlying_outbuf<DestCharSize>& ob
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr )
        : strf::underlying_outbuf<4>( buff_, buff_size_ )
        , transcode_(func)
        , ob_(ob)
        , enc_err_(enc_err)
        , allow_surr_(allow_surr)
    {
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto p = this->pos();
        if (p != buff_ && ob_.good()) {
            transcode_(ob_, buff_, p, enc_err_, allow_surr_);
        }
        this->set_good(false);
    }

private:

    strf::transcode_func<4, DestCharSize> transcode_;
    strf::underlying_outbuf<DestCharSize>& ob_;
    strf::encoding_error enc_err_;
    strf::surrogate_policy allow_surr_;
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
        transcode_(ob_, buff_, p, enc_err_, allow_surr_);
        this->set_good(true);
    }
}

class buffered_size_calculator: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_size_calculator
        ( strf::transcode_size_func<4> func, strf::surrogate_policy allow_surr )
        : strf::underlying_outbuf<4>(buff_, buff_size_)
        , size_func_(func)
        , allow_surr_(allow_surr)
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
    strf::surrogate_policy allow_surr_;
    constexpr static const std::size_t buff_size_ = 32;
    char32_t buff_[buff_size_];
};

#if ! defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD void buffered_size_calculator::recycle()
{
    auto p = this->pos();
    if (p != buff_) {
        this->set_pos(buff_);
        sum_ += size_func_(buff_, p, allow_surr_);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

template<std::size_t SrcCharSize, std::size_t DestCharSize>
STRF_HD void decode_encode
    ( strf::underlying_outbuf<DestCharSize>& ob
    , strf::transcode_func<SrcCharSize, 4> to_u32
    , strf::transcode_func<4, DestCharSize> from_u32
    , const underlying_outbuf_char_type<SrcCharSize>* str
    , const underlying_outbuf_char_type<SrcCharSize>* str_end
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr )
{
    strf::detail::buffered_encoder<DestCharSize> tmp{from_u32, ob, enc_err, allow_surr};
    to_u32(tmp, str, str_end, enc_err, allow_surr);
    tmp.finish();
}

template<std::size_t SrcCharSize>
STRF_HD std::size_t decode_encode_size
    ( strf::transcode_func<SrcCharSize, 4> to_u32
    , strf::transcode_size_func<4> size_calc_func
    , const underlying_outbuf_char_type<SrcCharSize>* str
    , const underlying_outbuf_char_type<SrcCharSize>* str_end
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, allow_surr};
    to_u32(acc, str, str_end, enc_err, allow_surr);
    return acc.get_sum();
}

} // namespace strf

#endif  // STRF_DETAIL_FACETS_ENCODINGS_HPP

