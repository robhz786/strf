#ifndef BOOST_STRINGIFY_V0_DETAIL_TRANSCODING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_TRANSCODING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class error_handling
{
    stop, replace, ignore
};

enum class cv_result
{
    success, invalid_char, insufficient_space
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
    typedef stringify::v0::cv_result (&transcode_func_ref)
        ( const CharIn** src
        , const CharIn* src_end
        , CharOut** dest
        , CharOut* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    typedef std::size_t (&size_func_ref)
        ( const CharIn* src
        , const CharIn* src_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr );

    transcode_func_ref transcode;
    size_func_ref necessary_size;
};

template <typename CharT>
struct encoding_impl
{
    using char_type = CharT;

    typedef std::size_t (&validate_func_ref)(char32_t ch);
    typedef stringify::v0::cv_result (&encode_char_func_ref)
        ( CharT** dest
        , CharT* dest_end
        , char32_t ch
        , stringify::v0::error_handling err_hdl );
    typedef stringify::v0::cv_result (&encode_fill_func_ref)
        ( CharT** dest
        , CharT* dest_end
        , std::size_t& count
        , char32_t ch
        , stringify::v0::error_handling err_hdl );
    typedef std::size_t (&codepoints_count_func_ref)
        ( const CharT* begin
        , const CharT* end
        , std::size_t max_count );
    typedef bool (&write_replacement_char_func_ref)
        ( CharT** dest
        , CharT* dest_end );
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

    std::size_t char_size(char32_t ch, stringify::v0::error_handling err_hdl) const
    {
        auto s = validate(ch);
        return s != (std::size_t)-1
            ? s
            : err_hdl == stringify::v0::error_handling::replace
            ? replacement_char_size
            : 0 ;
    }

};


template <std::size_t N>
struct char_type_of_size_impl;

template <> struct char_type_of_size_impl<1>{using type = std::uint8_t;};
template <> struct char_type_of_size_impl<2>{using type = char16_t;};
template <> struct char_type_of_size_impl<4>{using type = char32_t;};

template <typename CharT>
using info_char_type = typename char_type_of_size_impl<sizeof(CharT)>::type;

} // namespace detail

template <typename CharIn, typename CharOut>
using transcoder_impl_type
    = stringify::v0::detail::transcoder_impl
        < stringify::v0::detail::info_char_type<CharIn>
        , stringify::v0::detail::info_char_type<CharOut> >;

template <typename CharT>
using encoding_impl_type
    = stringify::v0::detail::encoding_impl
        < stringify::v0::detail::info_char_type<CharT> >;


template <typename CharIn, typename CharOut>
class transcoder
{
    using _inner_char_type_in = stringify::v0::detail::info_char_type<CharIn>;
    using _inner_char_type_out = stringify::v0::detail::info_char_type<CharOut>;

public:

    using impl_type = stringify::v0::detail::transcoder_impl
        < _inner_char_type_in
        , _inner_char_type_out >;

    constexpr transcoder(const impl_type& info)
        : _impl(&info)
    {
    }

    constexpr transcoder(const transcoder&) = default;

    transcoder& operator=(const transcoder& cp)
    {
        _impl = cp._impl;
        return *this;
    }

    bool operator==(const transcoder& cmp) const
    {
        return _impl = cmp._impl;
    }

    stringify::v0::cv_result transcode
        ( const CharIn** src
        , const CharIn* src_end
        , CharOut** dest
        , CharOut* dest_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr ) const
    {
        return _impl->transcode
            ( reinterpret_cast<const _inner_char_type_in**>(src)
            , reinterpret_cast<const _inner_char_type_in*>(src_end)
            , reinterpret_cast<_inner_char_type_out**>(dest)
            , reinterpret_cast<_inner_char_type_out*>(dest_end)
            , err_hdl
            , allow_surr );
    }

    std::size_t necessary_size
        ( const CharIn* src
        , const CharIn* src_end
        , stringify::v0::error_handling err_hdl
        , bool allow_surr ) const
    {
        return _impl->necessary_size
            ( reinterpret_cast<const _inner_char_type_in*>(src)
            , reinterpret_cast<const _inner_char_type_in*>(src_end)
            , err_hdl
            , allow_surr );
    }

    const impl_type& get_impl() const
    {
        return *_impl;
    }

private:

    const impl_type* _impl;
};


template <typename CharT>
class encoding
{
    using _impl_char_type = stringify::v0::detail::info_char_type<CharT>;

    template <typename>
    friend class stringify::v0::encoding;

public:

    using impl_type = stringify::v0::detail::encoding_impl<_impl_char_type>;
    using char_type = CharT;

    explicit encoding(const impl_type& info)
        : _impl(&info)
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
    stringify::v0::transcoder<CharT, char32_t> to_u32() const
    {
        return {_impl->to_u32};
    }
    stringify::v0::transcoder<char32_t, CharT> from_u32() const
    {
        return {_impl->from_u32};
    }
    stringify::v0::transcoder<CharT, CharT> sanitizer() const
    {
        return {_impl->sanitizer};
    }
    std::size_t validate(char32_t ch) const
    {
        return _impl->validate(ch);
    }
    stringify::v0::cv_result encode_char
        ( CharT** dest
        , CharT* dest_end
        , char32_t ch
        , stringify::v0::error_handling err_hdl ) const
    {
        return _impl->encode_char
            ( reinterpret_cast<_impl_char_type**>(dest)
            , reinterpret_cast<_impl_char_type*>(dest_end)
            , ch, err_hdl );
    }
    stringify::v0::cv_result encode_fill
        ( CharT** dest
        , CharT* dest_end
        , std::size_t& count
        , char32_t ch
        , stringify::v0::error_handling err_hdl ) const
    {
        return _impl->encode_fill
            ( reinterpret_cast<_impl_char_type**>(dest)
            , reinterpret_cast<_impl_char_type*>(dest_end)
            , count, ch, err_hdl );
    }
    std::size_t codepoints_count
        ( const CharT* src_begin
        , const CharT* src_end
        , std::size_t max_count ) const
    {
        return _impl->codepoints_count
            ( reinterpret_cast<const _impl_char_type*>(src_begin)
            , reinterpret_cast<const _impl_char_type*>(src_end)
            , max_count );
    }
    bool write_replacement_char
        ( CharT** dest
        , CharT* dest_end ) const
    {
        return _impl->write_replacement_char
            ( reinterpret_cast<_impl_char_type**>(dest)
            , reinterpret_cast<_impl_char_type*>(dest_end) );
    }
    char32_t decode_single_char(CharT ch) const
    {
        return _impl->decode_single_char(ch);
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
    std::size_t char_size(char32_t ch, stringify::v0::error_handling err_hdl) const
    {
        auto s = _impl->validate(ch);
        return s != (std::size_t)-1
            ? s
            : err_hdl == stringify::v0::error_handling::replace
            ? _impl->replacement_char_size
            : 0 ;
    }

    template <typename CharT2>
    const stringify::v0::transcoder_impl_type<CharT, CharT2>*
    transcoder_impl_to(stringify::v0::encoding<CharT2> e) const;

    const impl_type& get_impl() const
    {
        return *_impl;
    }

private:

    const impl_type* _impl;
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
struct get_transcoder_impl_helper
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
struct get_transcoder_impl_helper<CharT, CharT>
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
struct get_transcoder_impl_helper<char32_t, char32_t>
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
struct get_transcoder_impl_helper<char32_t, CharOut >
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
struct get_transcoder_impl_helper<CharIn, char32_t>
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
inline const stringify::v0::transcoder_impl_type<CharIn, CharOut>*
encoding<CharIn>::transcoder_impl_to(stringify::v0::encoding<CharOut> e) const
{
    using impl = stringify::v0::detail::get_transcoder_impl_helper
        < typename stringify::v0::encoding<CharIn>::_impl_char_type
        , typename stringify::v0::encoding<CharOut>::_impl_char_type >;

    return impl::get(*_impl, *e._impl);
}

template <typename CharIn, typename CharOut>
inline const stringify::v0::transcoder_impl_type<CharIn, CharOut>*
get_transcoder_impl
    ( stringify::v0::encoding<CharIn> src_encoding
    , stringify::v0::encoding<CharOut> dest_encoding )
{
    return src_encoding.transcoder_impl_to(dest_encoding);
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

inline stringify::v0::encoding<char> utf8()
{
    return stringify::v0::encoding<char>{stringify::v0::detail::utf8_impl()};
}

inline stringify::v0::encoding<char16_t> utf16()
{
    return stringify::v0::encoding<char16_t>{stringify::v0::detail::utf16_impl()};
}

inline stringify::v0::encoding<char32_t> utf32()
{
    const auto& info = stringify::v0::detail::utf32_impl();
    return stringify::v0::encoding<char32_t>{info};
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

inline stringify::v0::encoding<char> windows_1252()
{
    return stringify::v0::encoding<char>{stringify::v0::detail::windows_1252_impl()};
}

inline stringify::v0::encoding<char> iso_8859_1()
{
    return stringify::v0::encoding<char>{stringify::v0::detail::iso_8859_1_impl()};
}

inline stringify::v0::encoding<char> iso_8859_3()
{
    return stringify::v0::encoding<char>{stringify::v0::detail::iso_8859_3_impl()};
}

inline stringify::v0::encoding<char> iso_8859_15()
{
    return stringify::v0::encoding<char>{stringify::v0::detail::iso_8859_15_impl()};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_TRANSCODING_HPP

