#ifndef STRF_DETAIL_FACETS_ENCODINGS_HPP
#define STRF_DETAIL_FACETS_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>

namespace strf {

template <typename> class facet_trait;

enum class encoding_error
{
    replace, stop
};

struct encoding_error_c
{
    static constexpr bool constrainable = false;

    static constexpr STRF_HD strf::encoding_error get_default()
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

    static constexpr STRF_HD strf::surrogate_policy get_default()
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

template <std::size_t CharInSize, std::size_t CharOutSize>
struct underlying_transcoder
{
    using char_in = strf::underlying_outbuf_char_type<CharInSize>;
    using char_out = strf::underlying_outbuf_char_type<CharOutSize>;

    typedef void (&transcode_func_ref)
        ( strf::underlying_outbuf<CharOutSize>&
        , const char_in* src
        , const char_in* src_end
        , strf::encoding_error err_hdl
        , strf::surrogate_policy allow_surr );

    typedef std::size_t (&size_func_ref)
        ( const char_in* src
        , const char_in* src_end
        , strf::surrogate_policy allow_surr );

    transcode_func_ref transcode;
    size_func_ref necessary_size;
};

template <std::size_t CharSize>
struct underlying_encoding
{
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    typedef std::size_t (&validate_func_ref)(char32_t ch);

    typedef char_type* (&encode_char_func_ref)(char_type* dest, char32_t ch);

    typedef void (&encode_fill_func_ref)
        ( strf::underlying_outbuf<CharSize>&
        , std::size_t count
        , char32_t ch
        , strf::encoding_error err_hdl
        , strf::surrogate_policy allow_surr );

    typedef std::size_t (&codepoints_count_func_ref)
        ( const char_type* begin
        , const char_type* end
        , std::size_t max_count );

    typedef void (&write_replacement_char_func_ref)
        ( strf::underlying_outbuf<CharSize>& );

    typedef char32_t (&decode_char_func_ref)(char_type ch);

    typedef const strf::underlying_transcoder<CharSize, 1>* (*to8_func_ptr)
        ( const strf::underlying_encoding<1>& enc );

    typedef const strf::underlying_transcoder<1, CharSize>*
        (*from8_func_ptr)
        ( const strf::underlying_encoding<1>& enc );

    typedef const strf::underlying_transcoder<CharSize, 2>*
        (*to16_func_ptr)
        ( const strf::underlying_encoding<2>& enc );

    typedef const strf::underlying_transcoder<2, CharSize>*
        (*from16_func_ptr)
        ( const strf::underlying_encoding<2>& enc );

    typedef const strf::underlying_transcoder<CharSize, 4>*
        (*to32_func_ptr)
        ( const strf::underlying_encoding<4>& enc );

    typedef const strf::underlying_transcoder<4, CharSize>*
        (*from32_func_ptr)
        ( const strf::underlying_encoding<4>& enc );

    strf::underlying_transcoder<4, CharSize> from_u32;
    strf::underlying_transcoder<CharSize, 4> to_u32;
    strf::underlying_transcoder<CharSize, CharSize> sanitizer;

    validate_func_ref validate;
    encode_char_func_ref encode_char_fn;
    encode_fill_func_ref encode_fill;
    codepoints_count_func_ref codepoints_count;
    write_replacement_char_func_ref write_replacement_char;
    decode_char_func_ref decode_single_char;

    from8_func_ptr from8;
    to8_func_ptr to8;
    from16_func_ptr from16;
    to16_func_ptr to16;
    from32_func_ptr from32;
    to32_func_ptr to32;

    const char* name;
    strf::encoding_id id;
    std::size_t replacement_char_size;
    char32_t u32equivalence_begin;
    char32_t u32equivalence_end;

    STRF_HD std::size_t char_size(char32_t ch) const
    {
        auto size = validate(ch);
        bool is_valid = (size != (std::size_t)-1);
        return is_valid * size + (!is_valid) * replacement_char_size;
    }

    STRF_HD char_type* encode_char(char_type* dest, char32_t ch) const
    {
        return encode_char_fn(dest, ch);
    }
    STRF_HD void encode_char( strf::underlying_outbuf<CharSize>& ob
                            , char32_t ch
                            , strf::encoding_error err_hdl ) const
    {
        using char_type = strf::underlying_outbuf_char_type<CharSize>;
        if (u32equivalence_begin <= ch && ch < u32equivalence_end) {
            ob.ensure(1);
            *ob.pos() = static_cast<char_type>(ch);
            ob.advance();
        } else {
            auto s = validate(ch);
            if(s != (std::size_t)-1) {
                ob.ensure(s);
                ob.advance_to(this->encode_char(ob.pos(), ch));
            } else {
                if(err_hdl == strf::encoding_error::stop) {
                    strf::detail::handle_encoding_failure();
                }
                this->write_replacement_char(ob);
            }
        }
    }
};

template <typename CharIn, typename CharOut>
using transcoder_engine
    = strf::underlying_transcoder< sizeof(CharIn), sizeof(CharOut) >;

template <typename CharT>
using encoding_engine
    = strf::underlying_encoding<sizeof(CharT)>;

template <typename CharIn, typename CharOut>
class transcoder
{
    using _inner_char_type_in
        = typename strf::underlying_outbuf<sizeof(CharIn)>::char_type;

public:

    using engine_type = strf::transcoder_engine<CharIn, CharOut>;

    constexpr STRF_HD transcoder(const engine_type& info) noexcept
        : _impl(&info)
    {
    }

    transcoder& operator=(const transcoder& cp) noexcept
    {
        _impl = cp._impl;
        return *this;
    }

    STRF_HD bool operator==(const transcoder& cmp) const
    {
        return _impl = cmp._impl;
    }

    STRF_HD void transcode
        ( strf::basic_outbuf<CharOut>& ob
        , const CharIn* src
        , const CharIn* src_end
        , strf::encoding_error err_hdl
        , strf::surrogate_policy allow_surr ) const
    {
        _impl->transcode( ob.as_underlying()
                        , reinterpret_cast<const _inner_char_type_in*>(src)
                        , reinterpret_cast<const _inner_char_type_in*>(src_end)
                        , err_hdl
                        , allow_surr );
    }

    STRF_HD std::size_t necessary_size
        ( const CharIn* src
        , const CharIn* src_end
        , strf::surrogate_policy allow_surr ) const
    {
        return _impl->necessary_size
            ( reinterpret_cast<const _inner_char_type_in*>(src)
            , reinterpret_cast<const _inner_char_type_in*>(src_end)
            , allow_surr );
    }

    STRF_HD const strf::underlying_transcoder<sizeof(CharIn), sizeof(CharOut)>&
    as_underlying() const
    {
        return *_impl;
    }

private:

    const engine_type* _impl;
};


template <typename CharT>
class encoding
{
    using _impl_char_type
        = typename strf::underlying_outbuf<sizeof(CharT)>::char_type;

    template <typename>
    friend class strf::encoding;

public:

    using char_type = CharT;

    STRF_HD explicit encoding(const strf::encoding_engine<CharT>& eng)
        : _impl(&eng)
    {
    }

        STRF_HD encoding(const encoding& other)  noexcept
        : _impl(other._impl)
    {
    }

    STRF_HD encoding& operator=(const encoding& cp)
    {
        _impl = cp._impl;
        return *this;
    }
    STRF_HD bool operator==(const encoding& cmp) const
    {
        return _impl == cmp._impl;
    }
    STRF_HD strf::transcoder<char_type, char32_t> to_u32() const
    {
        return {_impl->to_u32};
    }
    STRF_HD strf::transcoder<char32_t, char_type> from_u32() const
    {
        return {_impl->from_u32};
    }
    STRF_HD strf::transcoder<char_type, char_type> sanitizer() const
    {
        return {_impl->sanitizer};
    }
    STRF_HD std::size_t validate(char32_t ch) const
    {
        return _impl->validate(ch);
    }
    STRF_HD std::size_t char_size(char32_t ch) const
    {
        auto size = _impl->validate(ch);
        bool is_valid = (size != (std::size_t)-1);
        return is_valid * size + (!is_valid) * replacement_char_size();
    }
    STRF_HD void encode_fill
        ( strf::basic_outbuf<char_type>& ob
        , std::size_t count
        , char32_t ch
        , strf::encoding_error err_hdl
        , strf::surrogate_policy allow_surr ) const
    {
        _impl->encode_fill( ob.as_underlying()
                          , count, ch, err_hdl, allow_surr );
    }
    STRF_HD std::size_t codepoints_count( const char_type* src_begin
                                , const char_type* src_end
                                , std::size_t max_count ) const
    {
        return _impl->codepoints_count
            ( reinterpret_cast<const _impl_char_type*>(src_begin)
            , reinterpret_cast<const _impl_char_type*>(src_end)
            , max_count );
    }
    STRF_HD void write_replacement_char
        ( strf::basic_outbuf<char_type>& ob ) const
    {
        _impl->write_replacement_char(ob.as_underlying());
    }
    STRF_HD char32_t decode_single_char(char_type ch) const
    {
        return _impl->decode_single_char(ch);
    }
    STRF_HD char_type* encode_char(char_type* dest, char32_t ch) const
    {
        auto rdest = reinterpret_cast<_impl_char_type*>(dest);
        return reinterpret_cast<char_type*>(_impl->encode_char(rdest, ch));
    }
    STRF_HD const char* name() const
    {
        return _impl->name;
    }
    STRF_HD strf::encoding_id id() const
    {
        return _impl->id;
    }
    STRF_HD std::size_t replacement_char_size() const
    {
        return _impl->replacement_char_size;
    }
    STRF_HD char32_t u32equivalence_begin() const
    {
        return _impl->u32equivalence_begin;
    }
    STRF_HD char32_t u32equivalence_end() const
    {
        return _impl->u32equivalence_end;
    }

    template <typename CharT2>
    const strf::transcoder_engine<char_type, CharT2>*
    STRF_HD transcoder_engine_to(strf::encoding<CharT2> e) const;

    STRF_HD const strf::underlying_encoding<sizeof(CharT)>& as_underlying() const
    {
        return *_impl;
    }

private:

    const strf::encoding_engine<CharT>* _impl;
};

namespace detail {

template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<CharInSize, 1>* enc_to_enc
    ( const strf::underlying_encoding<CharInSize>& from_enc
    , const strf::underlying_encoding<1>& to_enc)
{
    return from_enc.to8 ? from_enc.to8(to_enc) : nullptr;
}
template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<CharInSize, 2>* enc_to_enc
    ( const strf::underlying_encoding<CharInSize>& from_enc
    , const strf::underlying_encoding<2>& to_enc)
{
    return from_enc.to16 ? from_enc.to16(to_enc) : nullptr;
}
template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<CharInSize, 4>* enc_to_enc
    ( const strf::underlying_encoding<CharInSize>& from_enc
    , const strf::underlying_encoding<4>& to_enc)
{
    return from_enc.to32 ? from_enc.to32(to_enc) : nullptr;
}

template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<1, CharInSize>* enc_from_enc
    ( const strf::underlying_encoding<CharInSize>& to_enc
    , const strf::underlying_encoding<1>& from_enc)
{
    return to_enc.from8 ? to_enc.from8(from_enc) : nullptr;
}
template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<2, CharInSize>* enc_from_enc
    ( const strf::underlying_encoding<CharInSize>& to_enc
    , const strf::underlying_encoding<2>& from_enc)
{
    return to_enc.from16 ? to_enc.from16(from_enc) : nullptr;
}
template <std::size_t CharInSize>
const STRF_HD strf::underlying_transcoder<4, CharInSize>* enc_from_enc
    ( const strf::underlying_encoding<CharInSize>& to_enc
    , const strf::underlying_encoding<4>& from_enc)
{
    return to_enc.from32 ? to_enc.from32(from_enc) : nullptr;
}

template <std::size_t CharInSize, std::size_t CharOutSize>
struct get_transcoder_helper
{
    static STRF_HD const strf::underlying_transcoder<CharInSize, CharOutSize>* get
        ( const strf::underlying_encoding<CharInSize>& src_encoding
        , const strf::underlying_encoding<CharOutSize>& dest_encoding )
    {
        const strf::underlying_transcoder<CharInSize, CharOutSize>* t
            = strf::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : strf::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <std::size_t CharSize>
struct get_transcoder_helper<CharSize, CharSize>
{
    static STRF_HD const strf::underlying_transcoder<CharSize, CharSize>* get
        ( const strf::underlying_encoding<CharSize>& src_encoding
        , const strf::underlying_encoding<CharSize>& dest_encoding )
    {
        if (src_encoding.id == dest_encoding.id) {
            return & src_encoding.sanitizer;
        }
        const strf::underlying_transcoder<CharSize, CharSize>* t
            = strf::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : strf::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <>
struct get_transcoder_helper<4, 4>
{
    static STRF_HD const strf::underlying_transcoder<4, 4>* get
        ( const strf::underlying_encoding<4>& src_encoding
        , const strf::underlying_encoding<4>& dest_encoding )
    {
        if (src_encoding.id == dest_encoding.id) {
            return & src_encoding.sanitizer;
        }
        const strf::underlying_transcoder<4, 4>* t
            = strf::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : strf::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <std::size_t CharOutSize>
struct get_transcoder_helper<4, CharOutSize>
{
    static STRF_HD const strf::underlying_transcoder<4, CharOutSize>* get
        ( const strf::underlying_encoding<4>& src_encoding
        , const strf::underlying_encoding<CharOutSize>& dest_encoding )
    {
        if (src_encoding.id == strf::encoding_id::eid_utf32) {
            return & dest_encoding.from_u32;
        }
        const strf::underlying_transcoder<4, CharOutSize>* t
            = strf::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : strf::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <std::size_t CharInSize>
struct get_transcoder_helper<CharInSize, 4>
{
    static STRF_HD const strf::underlying_transcoder<CharInSize, 4>* get
        ( const strf::underlying_encoding<CharInSize>& src_encoding
        , const strf::underlying_encoding<4>& dest_encoding )
    {
        if (dest_encoding.id == strf::encoding_id::eid_utf32) {
            return & src_encoding.to_u32;
        }
        const strf::underlying_transcoder<CharInSize, 4>* t
            = strf::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : strf::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

} // namespace detail

template <typename CharIn>
template <typename CharOut>
inline STRF_HD const strf::transcoder_engine<CharIn, CharOut>*
encoding<CharIn>::transcoder_engine_to(strf::encoding<CharOut> e) const
{
    using impl = strf::detail::get_transcoder_helper
        < sizeof(CharIn), sizeof(CharOut) >;

    return impl::get(*_impl, *e._impl);
}

template <std::size_t CharInSize, std::size_t CharOutSize>
inline STRF_HD const strf::underlying_transcoder<CharInSize, CharOutSize>*
get_transcoder( const strf::underlying_encoding<CharInSize>& src_encoding
              , const strf::underlying_encoding<CharOutSize>& dest_encoding )
{
    return strf::detail::get_transcoder_helper<CharInSize, CharOutSize>
        ::get(src_encoding, dest_encoding);
}

namespace detail {

constexpr const std::size_t mini_buffer32_size = 16;

template <std::size_t CharOutSize>
class buffered_encoder: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_encoder
        ( const strf::underlying_encoding<CharOutSize>& enc
        , strf::underlying_outbuf<CharOutSize>& ob
        , strf::encoding_error err_hdl
        , strf::surrogate_policy allow_surr )
        : strf::underlying_outbuf<4>( _mini_buffer
                                    , strf::detail::mini_buffer32_size )
        , _enc(enc)
        , _ob(ob)
        , _err_hdl(err_hdl)
        , _allow_surr(allow_surr)
    {
        _begin = this->pos();
    }

    STRF_HD void recycle() override;

    STRF_HD void finish()
    {
        auto p = this->pos();
        if (p != _begin && _ob.good()) {
            _enc.from_u32.transcode(_ob, _begin, p, _err_hdl, _allow_surr);
        }
        this->set_good(false);
    }

private:

    const strf::underlying_encoding<CharOutSize>& _enc;
    strf::underlying_outbuf<CharOutSize>& _ob;
    char32_t* _begin;
    strf::encoding_error _err_hdl;
    strf::surrogate_policy _allow_surr;
    char32_t _mini_buffer[mini_buffer32_size];
};

template <std::size_t CharOutSize>
STRF_HD void buffered_encoder<CharOutSize>::recycle()
{
    auto p = this->pos();
    this->set_pos(_begin);
    if (p != _begin && _ob.good()) {
        this->set_good(false);
        _enc.from_u32.transcode(_ob, _begin, p, _err_hdl, _allow_surr);
        this->set_good(true);
    }
}

template <std::size_t CharOutSize>
class buffered_size_calculator: public strf::underlying_outbuf<4>
{
public:

    STRF_HD buffered_size_calculator
        ( const strf::underlying_encoding<CharOutSize>& enc
        , strf::surrogate_policy allow_surr )
        : strf::underlying_outbuf<4>
            ( _mini_buffer
            , strf::detail::mini_buffer32_size )
        , _enc(enc)
        , _allow_surr(allow_surr)
    {
        _begin = this->pos();
    }

    STRF_HD void recycle() override;

    STRF_HD std::size_t get_sum()
    {
        recycle();
        return _sum;
    }

private:

    const strf::underlying_encoding<CharOutSize>& _enc;
    char32_t* _begin;
    std::size_t _sum = 0;
    strf::surrogate_policy _allow_surr;
    char32_t _mini_buffer[mini_buffer32_size];
};

template <std::size_t CharOutSize>
STRF_HD void buffered_size_calculator<CharOutSize>::recycle()
{
    auto end = this->pos();
    if (end != _begin) {
        this->set_pos(_begin);
        _sum += _enc.from_u32.necessary_size(_begin, end, _allow_surr);
    }
}

} // namespace detail


template<std::size_t CharInSize, std::size_t CharOutSize>
inline STRF_HD void decode_encode
    ( strf::underlying_outbuf<CharOutSize>& ob
    , const strf::underlying_outbuf_char_type<CharInSize>* src
    , const strf::underlying_outbuf_char_type<CharInSize>* src_end
    , const strf::underlying_encoding<CharInSize>& src_encoding
    , const strf::underlying_encoding<CharOutSize>& dest_encoding
    , strf::encoding_error err_hdl
    , strf::surrogate_policy allow_surr )
{
    strf::detail::buffered_encoder<CharOutSize> dest
        { dest_encoding, ob, err_hdl, allow_surr };

    src_encoding.to_u32.transcode(dest, src, src_end, err_hdl, allow_surr);
    dest.finish();
}

template<std::size_t CharInSize, std::size_t CharOutSize>
inline STRF_HD std::size_t decode_encode_size
    ( const strf::underlying_outbuf_char_type<CharInSize>* src
    , const strf::underlying_outbuf_char_type<CharInSize>* src_end
    , const strf::underlying_encoding<CharInSize>& src_encoding
    , const strf::underlying_encoding<CharOutSize>& dest_encoding
    , strf::surrogate_policy allow_surr )
{
    strf::detail::buffered_size_calculator<CharOutSize> calc
        { dest_encoding, allow_surr };

    constexpr strf::encoding_error err_hdl = strf::encoding_error::replace;
    src_encoding.to_u32.transcode(calc, src, src_end, err_hdl, allow_surr);

    return calc.get_sum();
}


#if defined(STRF_OMIT_IMPL)

namespace detail {

const STRF_HD strf::underlying_encoding<1>& utf8_impl();
const STRF_HD strf::underlying_encoding<2>& utf16_impl();
const STRF_HD strf::underlying_encoding<4>& utf32_impl();
const STRF_HD strf::underlying_encoding<1>& windows_1252_impl();
const STRF_HD strf::underlying_encoding<1>& iso_8859_1_impl();
const STRF_HD strf::underlying_encoding<1>& iso_8859_3_impl();
const STRF_HD strf::underlying_encoding<1>& iso_8859_15_impl();

} // namespace detail

#endif // defined(STRF_OMIT_IMPL)

} // namespace strf

#if ! defined(STRF_OMIT_IMPL)

#include <strf/detail/utf_encodings.hpp>
#include <strf/detail/single_byte_encodings.hpp>

#endif // defined(STRF_OMIT_IMPL)

namespace strf {

template <typename CharT>
inline STRF_HD strf::encoding<CharT> utf16()
{
    static_assert(sizeof(CharT) >= 2, "incompatible character type for UTF-16");
    return strf::encoding<CharT>{strf::detail::utf16_impl()};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> utf32()
{
    static_assert(sizeof(CharT) >= 4, "incompatible character type for UTF-32");
    const auto& info = strf::detail::utf32_impl();
    return strf::encoding<CharT>{info};
}

namespace detail {

inline STRF_HD const strf::underlying_encoding<2>&
get_w_encoding_impl(std::integral_constant<std::size_t, 2>)
{
    return strf::detail::utf16_impl();
}

inline STRF_HD const strf::underlying_encoding<4>&
get_w_encoding_impl(std::integral_constant<std::size_t, 4>)
{
    return strf::detail::utf32_impl();
}

} // namespace detail

inline STRF_HD strf::encoding<wchar_t> wchar_encoding()
{
    std::integral_constant<std::size_t, sizeof(wchar_t)> tag;
    const auto& info = strf::detail::get_w_encoding_impl(tag);
    return strf::encoding<wchar_t>{info};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> utf8()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return strf::encoding<CharT>{strf::detail::utf8_impl()};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> windows_1252()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return strf::encoding<CharT>{strf::detail::windows_1252_impl()};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> iso_8859_1()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return strf::encoding<CharT>{strf::detail::iso_8859_1_impl()};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> iso_8859_3()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return strf::encoding<CharT>{strf::detail::iso_8859_3_impl()};
}

template <typename CharT>
inline STRF_HD strf::encoding<CharT> iso_8859_15()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return strf::encoding<CharT>{strf::detail::iso_8859_15_impl()};
}

template <typename CharT>
struct encoding_c;

template <typename Facet>
class facet_trait;

template <typename CharT>
class facet_trait<strf::encoding<CharT> >
{
public:
    using category = strf::encoding_c<CharT>;
    static constexpr bool store_by_value = true;
};

template <>
struct encoding_c<char>
{
    static constexpr bool constrainable = false;

    static STRF_HD encoding<char> get_default()
    {
        return strf::utf8<char>();
    }
};

#if defined(__cpp_char8_t)

template <>
struct encoding_c<char8_t>
{
    static constexpr bool constrainable = false;

    static STRF_HD encoding<char8_t> get_default()
    {
        return strf::utf8<char8_t>();
    }
};

#endif

template <>
struct encoding_c<char16_t>
{
    static constexpr bool constrainable = false;

    static STRF_HD encoding<char16_t> get_default()
    {
        return strf::utf16<char16_t>();
    }
};

template <>
struct encoding_c<char32_t>
{
    static constexpr bool constrainable = false;

    static STRF_HD encoding<char32_t> get_default()
    {
        return strf::utf32<char32_t>();
    }
};

template <>
struct encoding_c<wchar_t>
{
    static constexpr bool constrainable = false;

    static STRF_HD encoding<wchar_t> get_default()
    {
        return strf::wchar_encoding();
    }
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_ENCODINGS_HPP

