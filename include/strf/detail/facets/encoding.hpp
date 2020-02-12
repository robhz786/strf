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

template <typename CharT>
struct charset_c;

template <strf::charset_id>
class static_underlying_charset;

template <strf::charset_id Src, strf::charset_id Dest>
class static_underlying_transcoder;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_underlying_transcoder;

constexpr std::size_t invalid_char_len = (std::size_t)-1;

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using transcode_f = void (*)
    ( strf::underlying_outbuf<DestCharSize>& ob
    , const strf::underlying_char_type<SrcCharSize>* begin
    , const strf::underlying_char_type<SrcCharSize>* end
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli );

template <std::size_t SrcCharSize>
using transcode_size_f = std::size_t (*)
    ( const strf::underlying_char_type<SrcCharSize>* begin
    , const strf::underlying_char_type<SrcCharSize>* end
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

template <std::size_t CharSize>
using codepoints_count_f = std::size_t (*)
    ( const strf::underlying_char_type<CharSize>* begin
    , const strf::underlying_char_type<CharSize>* end
    , std::size_t max_count );

template <std::size_t CharSize>
using decode_single_char_f = char32_t (*)
    ( strf::underlying_char_type<CharSize> );

template <std::size_t SrcCharSize, std::size_t DestCharSize>
using find_transcoder_f =
    strf::dynamic_underlying_transcoder<SrcCharSize, DestCharSize> (*)
    ( strf::charset_id );

template <std::size_t SrcCharSize, std::size_t DestCharSize>
class dynamic_underlying_transcoder
{
public:

    template <strf::charset_id Src, strf::charset_id Dest>
    constexpr STRF_HD dynamic_underlying_transcoder
        ( strf::static_underlying_transcoder<Src, Dest> t ) noexcept
        : transcode_func_(t.transcode)
        , necessary_size_func_(t.necessary_size)
    {
    }

    constexpr STRF_HD dynamic_underlying_transcoder() noexcept
        : transcode_func_(nullptr)
        , necessary_size_func_(nullptr)
    {
    }

    constexpr STRF_HD strf::transcode_f<SrcCharSize, DestCharSize>
    transcode_func() const noexcept
    {
        return transcode_func_;
    }

    constexpr STRF_HD strf::transcode_size_f<SrcCharSize>
    necessary_size_func() const noexcept
    {
        return necessary_size_func_;
    }

    constexpr bool empty() const noexcept
    {
        return transcode_func_ == nullptr;
    }

private:

    strf::transcode_f<SrcCharSize, DestCharSize> transcode_func_;
    strf::transcode_size_f<SrcCharSize> necessary_size_func_;
};

template <std::size_t CharSize>
struct dynamic_underlying_charset_data
{
    const char* name;
    strf::charset_id id;
    char32_t replacement_char;
    std::size_t replacement_char_size;
    strf::validate_f validate_func;
    strf::encoded_char_size_f encoded_char_size_func;
    strf::encode_char_f<CharSize> encode_char_func;
    strf::encode_fill_f<CharSize> encode_fill_func;
    strf::codepoints_count_f<CharSize> codepoints_count_func;
    strf::write_replacement_char_f<CharSize> write_replacement_char_func;
    strf::decode_single_char_f<CharSize> decode_single_char_func;

    strf::dynamic_underlying_transcoder<4, CharSize> from_u32;
    strf::dynamic_underlying_transcoder<CharSize, 4> to_u32;
    strf::dynamic_underlying_transcoder<CharSize, CharSize> sanitizer;

    strf::find_transcoder_f<1, CharSize> transcoder_from_1byte_charset;
    strf::find_transcoder_f<2, CharSize> transcoder_from_2bytes_charset;

    strf::find_transcoder_f<CharSize, 1> transcoder_to_1byte_charset;
    strf::find_transcoder_f<CharSize, 2> transcoder_to_2bytes_charset;
};

template <std::size_t CharSize>
class dynamic_underlying_charset
{
    using char_type_ = strf::underlying_char_type<CharSize>;

public:

    static constexpr std::size_t char_size = CharSize;

    dynamic_underlying_charset(const dynamic_underlying_charset& ) = default;

    STRF_HD dynamic_underlying_charset
        ( const strf::dynamic_underlying_charset_data<CharSize>& data )
        : data_(&data)
    {
    }

    STRF_HD dynamic_underlying_charset& operator=(const dynamic_underlying_charset& other) noexcept
    {
        data_ = other.data_;
        return *this;
    }
    STRF_HD bool operator==(const dynamic_underlying_charset& other) const noexcept
    {
        return data_->id == other.data_->id;
    }
    STRF_HD bool operator!=(const dynamic_underlying_charset& other) const noexcept
    {
        return data_->id != other.data_->id;
    }

    STRF_HD void swap(dynamic_underlying_charset& other) noexcept
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
    STRF_HD std::size_t codepoints_count
        ( const char_type_* begin, const char_type_* end
        , std::size_t max_count ) const
    {
        return data_->codepoints_count(begin, end, max_count);
    }
    STRF_HD void write_replacement_char(strf::underlying_outbuf<CharSize>& ob) const
    {
        data_->write_replacement_char_func(ob);
    }
    STRF_HD char32_t decode_single_char(char_type_ ch) const
    {
        return data_->decode_single_char_func(ch);
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
    template <std::size_t DestCharSize>
    auto find_transcoder_to(strf::charset_id id) const
    {
        return find_transcoder_to_
            ( std::integral_constant<std::size_t, DestCharSize>{}, id );
    }
    template <std::size_t SrcCharSize>
    auto find_transcoder_from(strf::charset_id id) const
    {
        return find_transcoder_from_
            ( std::integral_constant<std::size_t, SrcCharSize>{}, id );
    }

private:

    strf::dynamic_underlying_transcoder<CharSize, 1> find_transcoder_to_
        ( std::integral_constant<std::size_t, 1>, strf::charset_id id) const
    {
        if (data_->transcoder_to_1byte_charset) {
            return data_->transcoder_to_1byte_charset(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<CharSize, 2> find_transcoder_to_
        ( std::integral_constant<std::size_t, 2>, strf::charset_id id) const
    {
        if (data_->transcoder_to_2bytes_charset) {
            return data_->transcoder_to_2bytes_charset(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<1, CharSize> find_transcoder_from_
        ( std::integral_constant<std::size_t, 1>, strf::charset_id id) const
    {
        if (data_->transcoder_from_1byte_charset) {
            return data_->transcoder_from_1byte_charset(id);
        }
        return {};
    }

    strf::dynamic_underlying_transcoder<2, CharSize> find_transcoder_from_
        ( std::integral_constant<std::size_t, 2>, strf::charset_id id) const
    {
        if (data_->transcoder_from_2bytes_charset) {
            return data_->transcoder_from_2bytes_charset(id);
        }
        return {};
    }

    const strf::dynamic_underlying_charset_data<CharSize>* data_;
};

} // namespace strf

#include <strf/detail/utf_encodings.hpp>

namespace strf {

template <typename CharT>
struct charset_c
{
    static constexpr bool constrainable = false;
    static constexpr STRF_HD strf::utf<CharT> get_default() noexcept
    {
        return {};
    }
};

namespace detail {

template <std::size_t SrcCharSize, std::size_t DestCharSize>
struct transcoder_finder
{
    template < strf::charset_id Src, strf::charset_id Dest>
    constexpr static STRF_HD auto get
        ( const strf::rank<4>&
        , strf::static_underlying_charset<Src>
        , strf::static_underlying_charset<Dest> ) noexcept
        -> decltype(strf::static_underlying_transcoder<Src, Dest>())
    {
        return {};
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<3>, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype( src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id())
                   , dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id()) )
    {
        auto t = src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id());
        if ( ! t.empty() ) {
            return t;
        }
        return dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id());
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<2>&, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype(src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id()))
    {
        return src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id());
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<1>&, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype(dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id()))
    {
        return dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id());
    }

    template <strf::charset_id Src, strf::charset_id Dest >
    constexpr static STRF_HD
        strf::dynamic_underlying_transcoder
            < strf::static_underlying_charset<Src>::char_size
            , strf::static_underlying_charset<Dest>::char_size >
    get( const strf::rank<0>&
       , strf::static_underlying_charset<Src>
       , strf::static_underlying_charset<Dest> ) noexcept
    {
        return {};
    }
};

template <std::size_t CharSize>
struct transcoder_finder<CharSize, CharSize>
{
    template <strf::charset_id Src, strf::charset_id Dest>
    constexpr static STRF_HD auto get
        ( const strf::rank<4>&
        , strf::static_underlying_charset<Src>
        , strf::static_underlying_charset<Dest> ) noexcept
        -> decltype(strf::static_underlying_transcoder<Src, Dest>())
    {
        return {};
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<3>, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype( src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id())
                   , dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id()) )
    {
        if (src_cs.id() == dest_cs.id()){
            return src_cs.sanitizer();
        }
        auto t = src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id());
        if ( ! t.empty() ) {
            return t;
        }
        return dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id());
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<2>&, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype(src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id()))
    {
        if (src_cs.id() == dest_cs.id()){
            return src_cs.sanitizer();
        }
        return src_cs.template find_transcoder_to<DestCharset::char_size>(dest_cs.id());
    }

    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto get
        ( const strf::rank<1>&, SrcCharset src_cs, DestCharset dest_cs )
        -> decltype(dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id()))
    {
        if (src_cs.id() == dest_cs.id()){
            return src_cs.sanitizer();
        }
        return dest_cs.template find_transcoder_from<SrcCharset::char_size>(src_cs.id());
    }

    template <strf::charset_id Src, strf::charset_id Dest >
    constexpr static STRF_HD
        strf::dynamic_underlying_transcoder
            < strf::static_underlying_charset<Src>::char_size
            , strf::static_underlying_charset<Dest>::char_size >
    get( const strf::rank<0>&
       , strf::static_underlying_charset<Src> src_cs
       , strf::static_underlying_charset<Dest> dest_cs ) noexcept
    {
        if (src_cs.id() == dest_cs.id()){
            return src_cs.sanitizer();
        }
        return {};
    }
};


template <>
struct transcoder_finder<4, 4>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD strf::utf32_to_utf32
    get(strf::rank<0>, const SrcCharset&, const DestCharset&) noexcept
    {
        return {};
    }
};

template <std::size_t SrcCharSize>
struct transcoder_finder<SrcCharSize, 4>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto
    get(strf::rank<0>, const SrcCharset& src_cs, const DestCharset&) noexcept
    {
        return src_cs.to_u32();
    }
};

template <std::size_t DestCharSize>
struct transcoder_finder<4, DestCharSize>
{
    template <typename SrcCharset, typename DestCharset>
    constexpr static STRF_HD auto
    get(strf::rank<0>, const SrcCharset&, const DestCharset& dest_cs) noexcept
    {
        return dest_cs.from_u32();
    }
};

} // namespace detail

template
    < typename SrcCharset
    , typename DestCharset
    , std::size_t SrcCharSize = SrcCharset::char_size
    , std::size_t DestCharSize = DestCharset::char_size >
constexpr STRF_HD
decltype(auto) get_transcoder(const SrcCharset& src_cs, const DestCharset& dest_cs)
{
    return detail::transcoder_finder<SrcCharSize, DestCharSize>
        ::get(strf::rank<4>(), src_cs, dest_cs);
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
        auto p = this->pos();
        if (p != buff_ && ob_.good()) {
            transcode_(ob_, buff_, p, inv_seq_poli_, surr_poli_);
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
    , strf::transcode_f<SrcCharSize, 4> to_u32
    , strf::transcode_f<4, DestCharSize> from_u32
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
    ( strf::transcode_f<SrcCharSize, 4> to_u32
    , strf::transcode_size_f<4> size_calc_func
    , const underlying_char_type<SrcCharSize>* str
    , const underlying_char_type<SrcCharSize>* str_end
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::buffered_size_calculator acc{size_calc_func, surr_poli};
    to_u32(acc, str, str_end, inv_seq_poli, surr_poli);
    return acc.get_sum();
}

template <typename CharT>
class dynamic_charset: public strf::dynamic_underlying_charset<sizeof(CharT)>
{
public:

    using category = strf::charset_c<CharT>;

    explicit dynamic_charset(const strf::dynamic_underlying_charset<sizeof(CharT)>& u)
        : strf::dynamic_underlying_charset<sizeof(CharT)>(u)
    {
    }

    template < typename T
             , typename = std::enable_if_t
                 < std::is_same<category, typename T::category>::value
                && ! std::is_same<dynamic_charset<CharT>, T>::value > >
    explicit dynamic_charset(T cs)
        : dynamic_charset(cs.to_dynamic())
    {
    }
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_CHARSETS_HPP

