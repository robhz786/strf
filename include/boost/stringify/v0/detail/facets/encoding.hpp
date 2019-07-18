#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> class output_buffer;
template <typename> class facet_trait;

enum class encoding_error
{
    replace, stop, ignore
};

struct encoding_error_c
{
    static constexpr bool constrainable = false;

    static constexpr stringify::v0::encoding_error get_default()
    {
        return stringify::v0::encoding_error::replace;
    }
};

template <>
class facet_trait<stringify::v0::encoding_error>
{
public:
    using category = stringify::v0::encoding_error_c;
    static constexpr bool store_by_value = true;
};

enum class surrogate_policy : bool
{
    strict = false, lax = true
};

struct surrogate_policy_c
{
    static constexpr bool constrainable = false;

    static constexpr stringify::v0::surrogate_policy get_default()
    {
        return stringify::v0::surrogate_policy::strict;
    }
};

template <>
class facet_trait<stringify::v0::surrogate_policy>
{
public:
    using category = stringify::v0::surrogate_policy_c;
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

namespace detail {

template <typename CharIn, typename CharOut>
struct transcoder_impl
{
    typedef void (&transcode_func_ref)
        ( stringify::v0::output_buffer_base<CharOut>&
        , const CharIn* src
        , const CharIn* src_end
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr );

    typedef std::size_t (&size_func_ref)
        ( const CharIn* src
        , const CharIn* src_end
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr );

    transcode_func_ref transcode;
    size_func_ref necessary_size;
};

template <typename CharT>
struct encoding_impl
{
    using char_type = CharT;

    typedef std::size_t (&validate_func_ref)(char32_t ch);
    typedef CharT* (&encode_char_func_ref)(CharT* dest, char32_t ch);
    typedef void (&encode_fill_func_ref)
        ( stringify::v0::output_buffer_base<CharT>&
        , std::size_t count
        , char32_t ch
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr );
    typedef std::size_t (&codepoints_count_func_ref)
        ( const CharT* begin
        , const CharT* end
        , std::size_t max_count );
    typedef void (&write_replacement_char_func_ref)
        ( stringify::v0::output_buffer_base<CharT>& );
    typedef char32_t (&decode_char_func_ref)(CharT ch);
    typedef const stringify::v0::detail::transcoder_impl<CharT, std::uint8_t>* (*to8_func_ptr)
        ( const stringify::v0::detail::encoding_impl<std::uint8_t>& enc );
    typedef const stringify::v0::detail::transcoder_impl<std::uint8_t, CharT>* (*from8_func_ptr)
        ( const stringify::v0::detail::encoding_impl<std::uint8_t>& enc );
    typedef const stringify::v0::detail::transcoder_impl<CharT, char16_t>* (*to16_func_ptr)
        ( const stringify::v0::detail::encoding_impl<char16_t>& enc );
    typedef const stringify::v0::detail::transcoder_impl<char16_t, CharT>* (*from16_func_ptr)
        ( const stringify::v0::detail::encoding_impl<char16_t>& enc );
    typedef const stringify::v0::detail::transcoder_impl<CharT, char32_t>* (*to32_func_ptr)
        ( const stringify::v0::detail::encoding_impl<char32_t>& enc );
    typedef const stringify::v0::detail::transcoder_impl<char32_t, CharT>* (*from32_func_ptr)
        ( const stringify::v0::detail::encoding_impl<char32_t>& enc );

    stringify::v0::detail::transcoder_impl<char32_t, CharT> from_u32;
    stringify::v0::detail::transcoder_impl<CharT, char32_t> to_u32;
    stringify::v0::detail::transcoder_impl<CharT, CharT> sanitizer;

    validate_func_ref validate;
    encode_char_func_ref encode_char;
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
    stringify::v0::encoding_id id;
    std::size_t replacement_char_size;
    char32_t u32equivalence_begin;
    char32_t u32equivalence_end;
};

} // namespace detail

template <typename CharIn, typename CharOut>
using transcoder_engine
    = stringify::v0::detail::transcoder_impl
        < stringify::v0::underlying_char_type<CharIn>
        , stringify::v0::underlying_char_type<CharOut> >;

template <typename CharT>
using encoding_engine
    = stringify::v0::detail::encoding_impl
        < stringify::v0::underlying_char_type<CharT> >;


template <typename CharIn, typename CharOut>
class transcoder
{
    using _inner_char_type_in = stringify::v0::underlying_char_type<CharIn>;
    using _inner_char_type_out = stringify::v0::underlying_char_type<CharOut>;
    using _impl_ob_type =
        stringify::v0::output_buffer_base<_inner_char_type_out>;

public:

    using engine_type = stringify::v0::transcoder_engine<CharIn, CharOut>;

    constexpr transcoder(const engine_type& info) noexcept
        : _impl(&info)
    {
    }

    constexpr transcoder(const transcoder&) noexcept = default;

    transcoder& operator=(const transcoder& cp) noexcept
    {
        _impl = cp._impl;
        return *this;
    }

    bool operator==(const transcoder& cmp) const
    {
        return _impl = cmp._impl;
    }

    void transcode
        ( stringify::v0::output_buffer<CharOut>& ob
        , const CharIn* src
        , const CharIn* src_end
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr ) const
    {
        _impl->transcode( static_cast<_impl_ob_type&>(ob)
                        , reinterpret_cast<const _inner_char_type_in*>(src)
                        , reinterpret_cast<const _inner_char_type_in*>(src_end)
                        , err_hdl
                        , allow_surr );
    }

    std::size_t necessary_size
        ( const CharIn* src
        , const CharIn* src_end
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr ) const
    {
        return _impl->necessary_size
            ( reinterpret_cast<const _inner_char_type_in*>(src)
            , reinterpret_cast<const _inner_char_type_in*>(src_end)
            , err_hdl
            , allow_surr );
    }

    engine_type & get_engine() const
    {
        return *_impl;
    }

private:

    const engine_type* _impl;
};


template <typename CharT>
class encoding
{
    using _impl_char_type = stringify::v0::underlying_char_type<CharT>;
    using _impl_ob_type = stringify::v0::output_buffer_base<_impl_char_type>;

    template <typename>
    friend class stringify::v0::encoding;

public:

    using char_type = CharT;

    explicit encoding(const stringify::v0::encoding_engine<CharT>& eng)
        : _impl(&eng)
    {
    }

    encoding(const encoding&) = default;

    encoding& operator=(const encoding& cp)
    {
        _impl = cp._impl;
        return *this;
    }
    bool operator==(const encoding& cmp) const
    {
        return _impl == cmp._impl;
    }
    stringify::v0::transcoder<char_type, char32_t> to_u32() const
    {
        return {_impl->to_u32};
    }
    stringify::v0::transcoder<char32_t, char_type> from_u32() const
    {
        return {_impl->from_u32};
    }
    stringify::v0::transcoder<char_type, char_type> sanitizer() const
    {
        return {_impl->sanitizer};
    }
    std::size_t validate(char32_t ch) const
    {
        return _impl->validate(ch);
    }
    std::size_t char_size(char32_t ch, stringify::v0::encoding_error err_hdl) const
    {
        auto size = _impl->validate(ch);
        bool is_valid = (size != (std::size_t)-1);
        bool shall_replace = (err_hdl == stringify::v0::encoding_error::replace);
        return is_valid * size
             + (!is_valid && shall_replace) * replacement_char_size();
    }
    void encode_fill
        ( stringify::v0::output_buffer<char_type>& ob
        , std::size_t count
        , char32_t ch
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr ) const
    {
        _impl->encode_fill( static_cast<_impl_ob_type&>(ob)
                          , count, ch, err_hdl, allow_surr );
    }
    std::size_t codepoints_count( const char_type* src_begin
                                , const char_type* src_end
                                , std::size_t max_count ) const
    {
        return _impl->codepoints_count
            ( reinterpret_cast<const _impl_char_type*>(src_begin)
            , reinterpret_cast<const _impl_char_type*>(src_end)
            , max_count );
    }
    void write_replacement_char
        ( stringify::v0::output_buffer<char_type>& ob ) const
    {
        _impl->write_replacement_char(static_cast<_impl_ob_type&>(ob));
    }
    char32_t decode_single_char(char_type ch) const
    {
        return _impl->decode_single_char(ch);
    }
    char_type* encode_char(char_type* dest, char32_t ch) const
    {
        auto rdest = reinterpret_cast<_impl_char_type*>(dest);
        return reinterpret_cast<char_type*>(_impl->encode_char(rdest, ch));
    }
    void encode_char
        ( stringify::v0::output_buffer<char_type>& ob
        , char32_t ch
        , stringify::v0::encoding_error err_hdl ) const
    {
        if (u32equivalence_begin() <= ch && ch < u32equivalence_end())
        {
            ob.ensure(1);
            *ob.pos() = static_cast<CharT>(ch);
            ob.advance();
        }
        else
        {
            auto s = validate(ch);
            if(s != (std::size_t)-1)
            {
                ob.ensure(s);
                ob.advance_to(this->encode_char(ob.pos(), ch));
            }
            else if(err_hdl == stringify::v0::encoding_error::replace)
            {
                this->write_replacement_char(ob);
            }
            else if(err_hdl == stringify::v0::encoding_error::stop)
            {
                ob.set_encoding_error();
            }
        }
    }
    const char* name() const
    {
        return _impl->name;
    }
    stringify::v0::encoding_id id() const
    {
        return _impl->id;
    }
    std::size_t replacement_char_size() const
    {
        return _impl->replacement_char_size;
    }
    char32_t u32equivalence_begin() const
    {
        return _impl->u32equivalence_begin;
    }
    char32_t u32equivalence_end() const
    {
        return _impl->u32equivalence_end;
    }

    template <typename CharT2>
    const stringify::v0::transcoder_engine<char_type, CharT2>*
    transcoder_engine_to(stringify::v0::encoding<CharT2> e) const;

    const stringify::v0::encoding_engine<CharT>& get_impl() const
    {
        return *_impl;
    }

private:

    const stringify::v0::encoding_engine<CharT>* _impl;
};

namespace detail {

template <typename CharIn>
const stringify::v0::detail::transcoder_impl<CharIn, std::uint8_t>* enc_to_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<std::uint8_t>& enc2)
{
    return enc1.to8 ? enc1.to8(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::detail::transcoder_impl<CharIn, char16_t>* enc_to_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<char16_t>& enc2)
{
    return enc1.to16 ? enc1.to16(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::detail::transcoder_impl<CharIn, char32_t>* enc_to_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<char32_t>& enc2)
{
    return enc1.to32 ? enc1.to32(enc2) : nullptr;
}

template <typename CharIn>
const stringify::v0::detail::transcoder_impl<std::uint8_t, CharIn>* enc_from_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<std::uint8_t>& enc2)
{
    return enc1.from8 ? enc1.from8(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::detail::transcoder_impl<char16_t, CharIn>* enc_from_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<char16_t>& enc2)
{
    return enc1.from16 ? enc1.from16(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::detail::transcoder_impl<char32_t, CharIn>* enc_from_enc
    ( const stringify::v0::detail::encoding_impl<CharIn>& enc1
    , const stringify::v0::detail::encoding_impl<char32_t>& enc2)
{
    return enc1.from32 ? enc1.from32(enc2) : nullptr;
}

template <typename CharIn, typename CharOut>
struct get_transcoder_helper
{
    static const stringify::v0::detail::transcoder_impl<CharIn, CharOut>* get
        ( const stringify::v0::detail::encoding_impl<CharIn>& src_encoding
        , const stringify::v0::detail::encoding_impl<CharOut>& dest_encoding )
    {
        const stringify::v0::detail::transcoder_impl<CharIn, CharOut>* t
            = stringify::v0::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <typename CharT>
struct get_transcoder_helper<CharT, CharT>
{
    static const stringify::v0::detail::transcoder_impl<CharT, CharT>* get
        ( const stringify::v0::detail::encoding_impl<CharT>& src_encoding
        , const stringify::v0::detail::encoding_impl<CharT>& dest_encoding )
    {
        if (src_encoding.id == dest_encoding.id)
        {
            return & src_encoding.sanitizer;
        }
        const stringify::v0::detail::transcoder_impl<CharT, CharT>* t
            = stringify::v0::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <>
struct get_transcoder_helper<char32_t, char32_t>
{
    using CharT = char32_t;
    static const stringify::v0::detail::transcoder_impl<CharT, CharT>* get
        ( const stringify::v0::detail::encoding_impl<CharT>& src_encoding
        , const stringify::v0::detail::encoding_impl<CharT>& dest_encoding )
    {
        if (src_encoding.id == dest_encoding.id)
        {
            return & src_encoding.sanitizer;
        }
        const stringify::v0::detail::transcoder_impl<CharT, CharT>* t
            = stringify::v0::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <typename CharOut>
struct get_transcoder_helper<char32_t, CharOut >
{
    static const stringify::v0::detail::transcoder_impl<char32_t, CharOut>* get
        ( const stringify::v0::detail::encoding_impl<char32_t>& src_encoding
        , const stringify::v0::detail::encoding_impl<CharOut>& dest_encoding )
    {
        if (src_encoding.id == stringify::v0::encoding_id::eid_utf32)
        {
            return & dest_encoding.from_u32;
        }
        const stringify::v0::detail::transcoder_impl<char32_t, CharOut>* t
            = stringify::v0::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

template <typename CharIn>
struct get_transcoder_helper<CharIn, char32_t>
{
    static const stringify::v0::detail::transcoder_impl<CharIn, char32_t>* get
        ( const stringify::v0::detail::encoding_impl<CharIn>& src_encoding
        , const stringify::v0::detail::encoding_impl<char32_t>& dest_encoding )
    {
        if (dest_encoding.id == stringify::v0::encoding_id::eid_utf32)
        {
            return & src_encoding.to_u32;
        }
        const stringify::v0::detail::transcoder_impl<CharIn, char32_t>* t
            = stringify::v0::detail::enc_from_enc(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::enc_to_enc(src_encoding, dest_encoding);
    }
};

} // namespace detail

template <typename CharIn>
template <typename CharOut>
inline const stringify::v0::transcoder_engine<CharIn, CharOut>*
encoding<CharIn>::transcoder_engine_to(stringify::v0::encoding<CharOut> e) const
{
    using impl = stringify::v0::detail::get_transcoder_helper
        < typename stringify::v0::encoding<CharIn>::_impl_char_type
        , typename stringify::v0::encoding<CharOut>::_impl_char_type >;

    return impl::get(*_impl, *e._impl);
}

template <typename CharIn, typename CharOut>
inline const stringify::v0::transcoder_engine<CharIn, CharOut>*
get_transcoder( stringify::v0::encoding<CharIn> src_encoding
              , stringify::v0::encoding<CharOut> dest_encoding )
{
    return src_encoding.transcoder_engine_to(dest_encoding);
}

namespace detail {

constexpr std::size_t global_mini_buffer32_size = 16;

inline char32_t* global_mini_buffer32()
{
    thread_local static char32_t buff[global_mini_buffer32_size];
    return buff;
}

template <typename CharOut>
class buffered_encoder: public stringify::v0::output_buffer<char32_t>
{
public:

    buffered_encoder
        ( stringify::v0::encoding<CharOut>& enc
        , stringify::v0::output_buffer<CharOut>& ob
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr )
        : stringify::v0::output_buffer<char32_t>
            ( stringify::v0::detail::global_mini_buffer32()
            , stringify::v0::detail::global_mini_buffer32_size )
        , _enc(enc)
        , _ob(ob)
        , _err_hdl(err_hdl)
        , _allow_surr(allow_surr)
    {
        _begin = this->pos();
    }

    void recycle() override;

private:

    stringify::v0::encoding<CharOut> _enc;
    stringify::v0::output_buffer<CharOut>& _ob;
    char32_t* _begin;
    stringify::v0::encoding_error _err_hdl;
    stringify::v0::surrogate_policy _allow_surr;
};

template <typename CharOut>
void buffered_encoder<CharOut>::recycle()
{
    auto end = this->pos();
    if (end != _begin)
    {
        this->set_pos(_begin);
        _enc.from_u32().transcode(_ob, _begin, end, _err_hdl, _allow_surr);
    }
}

template <typename CharOut>
class buffered_size_calculator: public stringify::v0::output_buffer<char32_t>
{
public:

    buffered_size_calculator
        ( stringify::v0::encoding<CharOut>& enc
        , stringify::v0::encoding_error err_hdl
        , stringify::v0::surrogate_policy allow_surr )
        : stringify::v0::output_buffer<char32_t>
            ( stringify::v0::detail::global_mini_buffer32()
            , stringify::v0::detail::global_mini_buffer32_size )
        , _enc(enc)
        , _err_hdl(err_hdl)
        , _allow_surr(allow_surr)
    {
        _begin = this->pos();
    }

    void recycle() override;

    std::size_t get_sum()
    {
        recycle();
        return _sum;
    }

private:

    stringify::v0::encoding<CharOut> _enc;
    char32_t* _begin;
    std::size_t _sum = 0;
    stringify::v0::encoding_error _err_hdl;
    stringify::v0::surrogate_policy _allow_surr;
};

template <typename CharOut>
void buffered_size_calculator<CharOut>::recycle()
{
    auto end = this->pos();
    if (end != _begin)
    {
        this->set_pos(_begin);
        _sum += _enc.from_u32().necessary_size(_begin, end, _err_hdl, _allow_surr);
    }
}

} // namespace detail


template<typename CharIn, typename CharOut>
inline void decode_encode
    ( stringify::v0::output_buffer<CharOut>& ob
    , const CharIn* src
    , const CharIn* src_end
    , stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding
    , stringify::v0::encoding_error err_hdl
    , stringify::v0::surrogate_policy allow_surr )
{
    stringify::v0::detail::buffered_encoder<CharOut> dest
        { dest_encoding, ob, err_hdl, allow_surr };

    src_encoding.to_u32().transcode(dest, src, src_end, err_hdl, allow_surr);
    dest.recycle();
}

template<typename CharIn, typename CharOut>
inline std::size_t decode_encode_size
    ( const CharIn* src
    , const CharIn* src_end
    , stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding
    , stringify::v0::encoding_error err_hdl
    , stringify::v0::surrogate_policy allow_surr )
{
    stringify::v0::detail::buffered_size_calculator<CharOut> calc
        { dest_encoding, err_hdl, allow_surr };

    src_encoding.to_u32().transcode(calc, src, src_end, err_hdl, allow_surr);

    return calc.get_sum();
}


#if defined(BOOST_STRINGIFY_OMIT_IMPL)

namespace detail {

const stringify::v0::detail::encoding_impl<std::uint8_t>& utf8_impl();
const stringify::v0::detail::encoding_impl<char16_t>& utf16_impl();
const stringify::v0::detail::encoding_impl<char32_t>& utf32_impl();
const stringify::v0::detail::encoding_impl<std::uint8_t>& windows_1252_impl();
const stringify::v0::detail::encoding_impl<std::uint8_t>& iso_8859_1_impl();
const stringify::v0::detail::encoding_impl<std::uint8_t>& iso_8859_3_impl();
const stringify::v0::detail::encoding_impl<std::uint8_t>& iso_8859_15_impl();

} // namespace detail

#endif // defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_V0_NAMESPACE_END

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

#include <boost/stringify/v0/detail/utf_encodings.hpp>
#include <boost/stringify/v0/detail/single_byte_encodings.hpp>

#endif // defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
inline stringify::v0::encoding<CharT> utf16()
{
    static_assert(sizeof(CharT) >= 2, "incompatible character type for UTF-16");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::utf16_impl()};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> utf32()
{
    static_assert(sizeof(CharT) >= 4, "incompatible character type for UTF-32");
    const auto& info = stringify::v0::detail::utf32_impl();
    return stringify::v0::encoding<CharT>{info};
}

namespace detail {

inline const stringify::v0::detail::encoding_impl<char16_t>&
get_w_encoding_impl(std::integral_constant<std::size_t, 2>)
{
    return stringify::v0::detail::utf16_impl();
}

inline const stringify::v0::detail::encoding_impl<char32_t>&
get_w_encoding_impl(std::integral_constant<std::size_t, 4>)
{
    return stringify::v0::detail::utf32_impl();
}

} // namespace detail

inline stringify::v0::encoding<wchar_t> wchar_encoding()
{
    std::integral_constant<std::size_t, sizeof(wchar_t)> tag;
    const auto& info = stringify::v0::detail::get_w_encoding_impl(tag);
    return stringify::v0::encoding<wchar_t>{info};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> utf8()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::utf8_impl()};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> windows_1252()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::windows_1252_impl()};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> iso_8859_1()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::iso_8859_1_impl()};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> iso_8859_3()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::iso_8859_3_impl()};
}

template <typename CharT>
inline stringify::v0::encoding<CharT> iso_8859_15()
{
    static_assert(sizeof(CharT) == 1, "incompatible character type for UTF-8");
    return stringify::v0::encoding<CharT>{stringify::v0::detail::iso_8859_15_impl()};
}

template <typename CharT>
struct encoding_c;

template <typename Facet>
class facet_trait;

template <typename CharT>
class facet_trait<stringify::v0::encoding<CharT> >
{
public:
    using category = stringify::v0::encoding_c<CharT>;
    static constexpr bool store_by_value = true;
};

template <>
struct encoding_c<char>
{
    static constexpr bool constrainable = false;

    static encoding<char> get_default()
    {
        return stringify::v0::utf8<char>();
    }
};

#if defined(__cpp_char8_t)

template <>
struct encoding_c<char8_t>
{
    static constexpr bool constrainable = false;

    static encoding<char8_t> get_default()
    {
        return stringify::v0::utf8<char8_t>();
    }
};

#endif

template <>
struct encoding_c<char16_t>
{
    static constexpr bool constrainable = false;

    static encoding<char16_t> get_default()
    {
        return stringify::v0::utf16<char16_t>();
    }
};

template <>
struct encoding_c<char32_t>
{
    static constexpr bool constrainable = false;

    static encoding<char32_t> get_default()
    {
        return stringify::v0::utf32<char32_t>();
    }
};

template <>
struct encoding_c<wchar_t>
{
    static constexpr bool constrainable = false;

    static encoding<wchar_t> get_default()
    {
        return stringify::v0::wchar_encoding();
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP

