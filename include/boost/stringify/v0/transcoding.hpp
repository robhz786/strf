#ifndef BOOST_STRINGIFY_V0_TRANSCODING_HPP
#define BOOST_STRINGIFY_V0_TRANSCODING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class error_handling
{
    emit_error, stop = emit_error, replace, ignore
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


template <typename CharIn, typename CharOut>
struct transcoder
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
struct encoding
{
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
    typedef std::size_t (&replacement_char_size_func_ref) ();
    typedef bool (&write_replacement_char_func_ref)
        ( CharT** dest
        , CharT* dest_end );
    typedef const stringify::v0::transcoder<CharT, char>* (*to8_func_ptr)
        ( const stringify::v0::encoding<char>& enc );
    typedef const stringify::v0::transcoder<char, CharT>* (*from8_func_ptr)
        ( const stringify::v0::encoding<char>& enc );
    typedef const stringify::v0::transcoder<CharT, char16_t>* (*to16_func_ptr)
        ( const stringify::v0::encoding<char16_t>& enc );
    typedef const stringify::v0::transcoder<char16_t, CharT>* (*from16_func_ptr)
        ( const stringify::v0::encoding<char16_t>& enc );
    typedef const stringify::v0::transcoder<CharT, char32_t>* (*to32_func_ptr)
        ( const stringify::v0::encoding<char32_t>& enc );
    typedef const stringify::v0::transcoder<char32_t, CharT>* (*from32_func_ptr)
        ( const stringify::v0::encoding<char32_t>& enc );
    typedef const stringify::v0::transcoder<CharT, wchar_t>* (*tow_func_ptr)
        ( const stringify::v0::encoding<wchar_t>& enc );
    typedef const stringify::v0::transcoder<wchar_t, CharT>* (*fromw_func_ptr)
        ( const stringify::v0::encoding<wchar_t>& enc );

    stringify::v0::transcoder<char32_t, CharT> from_u32;
    stringify::v0::transcoder<CharT, char32_t> to_u32;
    stringify::v0::transcoder<CharT, CharT> sanitizer;

    validate_func_ref validate;
    encode_char_func_ref encode_char;
    encode_fill_func_ref encode_fill;
    codepoints_count_func_ref codepoints_count;
    write_replacement_char_func_ref write_replacement_char;

    from8_func_ptr from8;
    to8_func_ptr to8;
    from16_func_ptr from16;
    to16_func_ptr to16;
    from32_func_ptr from32;
    to32_func_ptr to32;
    fromw_func_ptr fromw;
    tow_func_ptr tow;

    const char* name;
    stringify::v0::encoding_id id;
    std::size_t replacement_char_size;
    // char32_t max_char;
    // unsigned max_char_size;
    // bool monotonic_size;
};


namespace detail {

template <typename CharIn>
const stringify::v0::transcoder<CharIn, char>* to_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char>& enc2)
{
    return enc1.to8 ? enc1.to8(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<CharIn, char16_t>* to_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char16_t>& enc2)
{
    return enc1.to16 ? enc1.to16(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<CharIn, char32_t>* to_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char32_t>& enc2)
{
    return enc1.to32 ? enc1.to32(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<CharIn, wchar_t>* to_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<wchar_t>& enc2)
{
    return enc1.tow ? enc1.tow(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<char, CharIn>* from_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char>& enc2)
{
    return enc1.from8 ? enc1.from8(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<char16_t, CharIn>* from_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char16_t>& enc2)
{
    return enc1.from16 ? enc1.from16(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<char32_t, CharIn>* from_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<char32_t>& enc2)
{
    return enc1.from32 ? enc1.from32(enc2) : nullptr;
}
template <typename CharIn>
const stringify::v0::transcoder<wchar_t, CharIn>* from_tmp
    ( const stringify::v0::encoding<CharIn>& enc1
    , const stringify::v0::encoding<wchar_t>& enc2)
{
    return enc1.fromw ? enc1.fromw(enc2) : nullptr;
}

template <typename CharIn, typename CharOut>
struct get_transcoder_impl
{
    static const transcoder<CharIn, CharOut>* get
        ( const encoding<CharIn>& src_encoding
        , const encoding<CharOut>& dest_encoding )
    {
        const transcoder<CharIn, CharOut>* t
            = stringify::v0::detail::from_tmp(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::to_tmp(src_encoding, dest_encoding);
    }
};

template <typename CharT>
struct get_transcoder_impl<CharT, CharT>
{
    static const transcoder<CharT, CharT>* get
        ( const encoding<CharT>& src_encoding
        , const encoding<CharT>& dest_encoding )
    {
        if (src_encoding.id == dest_encoding.id)
        {
            return & src_encoding.sanitizer;
        }
        const transcoder<CharT, CharT>* t
            = stringify::v0::detail::from_tmp(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::to_tmp(src_encoding, dest_encoding);
    }
};

template <typename CharOut>
struct get_transcoder_impl<char32_t, CharOut >
{
    static const transcoder<char32_t, CharOut>* get
        ( const encoding<char32_t>& src_encoding
        , const encoding<CharOut>& dest_encoding )
    {
        if (src_encoding.id == stringify::v0::encoding_id::eid_utf32)
        {
            return & dest_encoding.from_u32;
        }
        const transcoder<char32_t, CharOut>* t
            = stringify::v0::detail::from_tmp(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::to_tmp(src_encoding, dest_encoding);
    }
};

template <typename CharIn>
struct get_transcoder_impl<CharIn, char32_t>
{
    static const transcoder<CharIn, char32_t>* get
        ( const encoding<CharIn>& src_encoding
        , const encoding<char32_t>& dest_encoding )
    {
        if (dest_encoding.id == stringify::v0::encoding_id::eid_utf32)
        {
            return & src_encoding.to_u32;
        }
        const transcoder<CharIn, char32_t>* t
            = stringify::v0::detail::from_tmp(dest_encoding, src_encoding);
        return t != nullptr ? t
            : stringify::v0::detail::to_tmp(src_encoding, dest_encoding);
    }
};

} // namespace detail

template <typename CharIn, typename CharOut>
inline const transcoder<CharIn, CharOut>* get_transcoder
    ( const encoding<CharIn>& src_encoding
    , const encoding<CharOut>& dest_encoding )
{
    using impl = stringify::v0::detail::get_transcoder_impl<CharIn, CharOut>;
    return impl::get(src_encoding, dest_encoding);
}

#if defined(BOOST_STRINGIFY_OMIT_IMPL)

const stringify::v0::encoding<char>& utf8();
const stringify::v0::encoding<char16_t>& utf16();
const stringify::v0::encoding<char32_t>& utf32();
const stringify::v0::encoding<wchar_t>& wchar_encoding();

#endif // defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_V0_NAMESPACE_END


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

#include <boost/stringify/v0/detail/encodings.hpp>

#endif // defined(BOOST_STRINGIFY_OMIT_IMPL)

#endif  // BOOST_STRINGIFY_V0_TRANSCODING_HPP

