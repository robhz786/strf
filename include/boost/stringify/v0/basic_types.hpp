#ifndef BOOST_STRINGIFY_V0_BASIC_TYPES_HPP
#define BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

constexpr std::size_t min_buff_size = 60;

template <typename CharIn>  class encoding_info;
template <typename CharIn, typename CharOut> class transcoder;
template <typename CharOut> class output_writer;
template <typename CharOut> struct encoding_category;
struct allow_surrogates_category;
struct encoding_error_category;
struct tag {};

class error_signal
{
public:

    typedef void (*func_ptr)(void);

    explicit error_signal() noexcept
        : m_variant(e_omit)
    {
    }
    explicit error_signal(char32_t ch) noexcept
        : m_variant(e_char)
    {
        m_char = ch;
    }
    explicit error_signal(func_ptr func) noexcept
        : m_variant(e_function)
    {
        m_function = func;
    }
    explicit error_signal(std::error_code ec) noexcept
        : m_variant(e_error_code)
    {
        new (&m_error_code_storage) std::error_code(ec);
    }
    error_signal(const error_signal& other) noexcept
    {
        init_from(other);
    }
    ~error_signal() noexcept
    {
        if(m_variant == e_error_code)
        {
            error_code_ptr()->~error_code();
        }
    }
    void reset() noexcept
    {
        if(m_variant == e_error_code)
        {
            error_code_ptr()->~error_code();
        }
        m_variant = e_omit;
    }
    void reset(char32_t ch) noexcept
    {
        if(m_variant == e_error_code)
        {
            error_code_ptr()->~error_code();
        }
        m_variant = e_char;
        m_char = ch;
    }
    void reset(func_ptr func) noexcept
    {
        if(m_variant == e_error_code)
        {
            error_code_ptr()->~error_code();
        }
        m_variant = e_function;
        m_function = func;
    }
    void reset(std::error_code ec) noexcept
    {
        if(m_variant == e_error_code)
        {
            * error_code_ptr() = ec;
        }
        else
        {
            new (&m_error_code_storage) std::error_code(ec);
            m_variant = e_error_code;
        }
    }
    error_signal& operator=(const error_signal& other) noexcept;

    bool operator==(const error_signal& other) noexcept;

    bool skip() const noexcept
    {
        return m_variant == e_omit;
    }
    bool has_char() const noexcept
    {
        return m_variant == e_char;
    }
    bool has_error_code() const noexcept
    {
        return m_variant == e_error_code;
    }
    bool has_function() const noexcept
    {
        return m_variant == e_function;
    }

    char32_t get_char() const noexcept
    {
        return m_char;
    }
    const std::error_code& get_error_code() const noexcept
    {
        return *error_code_ptr();
    }
    func_ptr get_function() const noexcept
    {
        return m_function;
    }

private:

    void init_from(const error_signal& other);

    std::error_code* error_code_ptr()
    {
        return reinterpret_cast<std::error_code*>(&m_error_code_storage);
    }
    const std::error_code* error_code_ptr() const
    {
        return reinterpret_cast<const std::error_code*>(&m_error_code_storage);
    }

    using error_code_storage_type
    = std::aligned_storage_t<sizeof(std::error_code), alignof(std::error_code)>;

    union
    {
        char32_t m_char;
        error_code_storage_type m_error_code_storage;
        func_ptr m_function;
    };
    enum { e_omit, e_char, e_error_code, e_function } m_variant;
};


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
error_signal& error_signal::operator=(const error_signal& other) noexcept
{
    if (m_variant == e_error_code)
    {
        error_code_ptr()->~error_code();
    }
    init_from(other);
    return *this;
}

BOOST_STRINGIFY_INLINE
bool error_signal::operator==(const error_signal& other) noexcept
{
    if (m_variant != other.m_variant)
    {
        return false;
    }
    switch (m_variant)
    {
        case e_omit: return true;
        case e_char: return m_char == other.m_char;
        case e_function: return m_function == other.m_function;
        default:     return get_error_code() == other.get_error_code();
    }
}

BOOST_STRINGIFY_INLINE
void error_signal::init_from(const error_signal& other)
{
    m_variant = other.m_variant;
    switch (other.m_variant)
    {
        case e_char:
            m_char = other.m_char;
            break;
        case e_omit:
            break;
        case e_function:
            m_function = other.m_function;
            break;
        default:
            new (&m_error_code_storage) std::error_code(other.get_error_code());
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)


enum class cv_result
{
    success,
    invalid_char,
    insufficient_space
};

class u32output
{
public:

    virtual ~u32output()
    {
    }

    BOOST_STRINGIFY_NODISCARD
    virtual stringify::v0::cv_result put32(char32_t ch) = 0;

    /**
    @retval true success
    @retval false insufficient space
    */
    BOOST_STRINGIFY_NODISCARD
    virtual bool signalize_error() = 0;
};

template <typename CharIn>
struct decoder_result
{
    const CharIn* src_it;
    stringify::v0::cv_result result;
};

template <typename CharIn>
class decoder
{
public:

    virtual ~decoder()
    {
    }

    BOOST_STRINGIFY_NODISCARD
    virtual stringify::v0::decoder_result<CharIn> decode
        ( stringify::v0::u32output& dest
        , const CharIn* begin
        , const CharIn* end
        , bool allow_surrogates
        ) const = 0;

    /**
    dont treat surrogates as invalid
    @retval 0x0FFFFFF on error
    */
    // virtual char32_t decode(CharIn ch) const = 0; //todo

    virtual std::size_t remaining_codepoints_count
        ( std::size_t minuend
        , const CharIn* begin
        , const CharIn* end
        ) const = 0;
};

template <typename CharOut>
struct char_cv_result
{
    std::size_t count;
    CharOut* dest_it;
};

template <typename CharOut>
class encoder
{
public:

    virtual ~encoder()
    {
    }

    std::size_t necessary_size
        ( char32_t ch
        , const stringify::v0::error_signal& esig
        , bool allow_surrogates ) const;

    virtual std::size_t validate
        ( char32_t ch
        , bool allow_surrogates ) const = 0;

    /**
    if ch is invalid or not supported return {0, nullptr}
    */
    virtual stringify::v0::char_cv_result<CharOut> encode
        ( std::size_t count
        , char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool allow_surrogates ) const = 0;

    /**
    return
    - success: != nullptr && <=end
    - space is insufficient : end + 1
    - ch is invalid or not supported: nullptr
    */
    virtual CharOut* encode
        ( char32_t ch
        , CharOut* dest
        , CharOut* dest_end
        , bool allow_surrogates ) const = 0;
};

template <typename CharOut>
std::size_t encoder<CharOut>::necessary_size
    ( char32_t ch
    , const stringify::v0::error_signal& esig
    , bool allow_surrogates ) const
{
    auto size = validate(ch, allow_surrogates);
    if (size != 0)
    {
        return size;
    }
    if (esig.has_char())
    {
        size = validate(esig.get_char(), allow_surrogates);
        if (size != 0)
        {
            return size;
        }
        return 1; // size of U'?'
    }
    return 0;
}


template <typename CharOut>
stringify::v0::char_cv_result<CharOut> emit_error
    ( stringify::v0::error_signal& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , std::size_t count
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    if(err_sig.skip())
    {
        return { count, dest };
    }
    if(err_sig.has_char())
    {
        char32_t ch = err_sig.get_char();
        auto r = encoder.encode(ch, dest, dest_end, allow_surrogates);
        if(nullptr == r.dest_it)
        {
            BOOST_ASSERT(0 == r.count);
            ch = U'?';
            err_sig.reset(ch);
            r = encoder.encode(ch, dest, dest_end, allow_surrogates);
        }
        BOOST_ASSERT(nullptr != r.dest_it);
        return r;
    }
    if(err_sig.has_error_code())
    {
        return { 0, nullptr };
    }
    BOOST_ASSERT(err_sig.has_function());
    err_sig.get_function() ();
    return { count, dest };
}

template <typename CharOut>
CharOut* emit_error
    ( stringify::v0::error_signal& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    if(err_sig.skip())
    {
        return dest;
    }
    if(err_sig.has_char())
    {
        char32_t ch = err_sig.get_char();
        auto it = encoder.encode(ch, dest, dest_end, allow_surrogates);
        if(nullptr == it)
        {
            ch = U'?';
            err_sig.reset(ch);
            it = encoder.encode(ch, dest, dest_end, allow_surrogates);
        }
        BOOST_ASSERT(it != nullptr);
        return it;
    }
    if(err_sig.has_error_code())
    {
        return nullptr;
    }
    BOOST_ASSERT(err_sig.has_function());
    err_sig.get_function() ();
    return dest;
}


template <typename CharOut>
inline stringify::v0::char_cv_result<CharOut> emit_error
    ( stringify::v0::error_signal&& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , std::size_t count
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    return stringify::v0::emit_error
        ( err_sig, encoder, count, dest, dest_end, allow_surrogates );
}

template <typename CharOut>
inline stringify::v0::char_cv_result<CharOut> emit_error
    ( const stringify::v0::error_signal& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , std::size_t count
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    return stringify::v0::emit_error
        ( stringify::v0::error_signal{ err_sig }
        , encoder
        , count
        , dest
        , dest_end
        , allow_surrogates );
}

template <typename CharOut>
inline CharOut* emit_error
    ( stringify::v0::error_signal&& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    return stringify::v0::emit_error
        ( err_sig
        , encoder
        , dest
        , dest_end
        , allow_surrogates );
}

template <typename CharOut>
inline CharOut* emit_error
    ( const stringify::v0::error_signal& err_sig
    , const stringify::v0::encoder<CharOut>& encoder
    , CharOut* dest
    , CharOut* dest_end
    , bool allow_surrogates )
{
    return stringify::v0::emit_error
        ( stringify::v0::error_signal{ err_sig }
        , encoder
        , dest
        , dest_end
        , allow_surrogates );
}

template <typename CharT>
class encoding
{
public:

    using category = stringify::v0::encoding_category<CharT>;

    constexpr encoding(const stringify::v0::encoding_info<CharT>& info)
        : m_info(&info)
    {
    }

    constexpr encoding(const encoding&) = default;

    constexpr encoding& operator=(encoding other)
    {
        m_info = other.m_info;
    }


    template <typename Ch>
    constexpr bool operator==(encoding<Ch> other) const
    {
        return m_info->equivalent(other.info());
    }

    template <typename Ch>
    constexpr bool operator!=(encoding<Ch> other) const
    {
        return ! operator==(other);
    }

    constexpr const stringify::v0::encoding_info<CharT>& info() const
    {
        return *m_info;
    }

    constexpr const stringify::v0::decoder<CharT>& decoder() const
    {
        return m_info->decoder();
    }

    constexpr const stringify::v0::encoder<CharT>& encoder() const
    {
        return m_info->encoder();
    }

    template <typename CharOut>
    const stringify::v0::transcoder<CharT, CharOut>*
    to(stringify::v0::encoding<CharOut> enc) const;

    template <typename CharOut>
    const stringify::v0::transcoder<CharT, CharOut>*
    sani_to(stringify::v0::encoding<CharOut> enc) const;

private:

    const stringify::v0::encoding_info<CharT>* m_info;
};

enum encoding_id : unsigned
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

    eid_mac_os_roman,
    eid_koi8_r,
    eid_koi8_u,
    eid_koi7,
    eid_mik,
    eid_iscii,
    eid_tscii,
    eid_viscii,

    iso_2022_jp,
    iso_2022_jp_1,
    iso_2022_jp_2,
    iso_2022_jp_2004,
    iso_2022_kr,
    iso_2022_cn,
    iso_2022_cn_ext,

    // etc ... TODO
    // https://en.wikipedia.org/wiki/Character_encoding#Common_character_encodings
};


template <typename CharT>
class encoding_info
{
public:

    encoding_info
        ( const stringify::v0::decoder<CharT>& decoder
        , const stringify::v0::encoder<CharT>& encoder
        , unsigned enc_id )
        noexcept
        : m_decoder(decoder)
        , m_encoder(encoder)
        , m_id(enc_id)
    {
    }

    virtual ~encoding_info()
    {
    }

    virtual const transcoder<CharT, char>*     sani_to(encoding<char> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<CharT, wchar_t>*  sani_to(encoding<wchar_t> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<CharT, char16_t>* sani_to(encoding<char16_t> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<CharT, char32_t>* sani_to(encoding<char32_t> enc) const
        { (void)enc; return nullptr; }

    virtual const transcoder<char,     CharT>* sani_from(encoding<char> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<wchar_t,  CharT>* sani_from(encoding<wchar_t> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<char16_t, CharT>* sani_from(encoding<char16_t> enc) const
        { (void)enc; return nullptr; }
    virtual const transcoder<char32_t, CharT>* sani_from(encoding<char32_t> enc) const
        { (void)enc; return nullptr; }

    virtual const transcoder<CharT, char>*     to(encoding<char> enc) const
        { return sani_to(enc); }
    virtual const transcoder<CharT, wchar_t>*  to(encoding<wchar_t> enc) const
        { return sani_to(enc); }
    virtual const transcoder<CharT, char16_t>* to(encoding<char16_t> enc) const
        { return sani_to(enc); }
    virtual const transcoder<CharT, char32_t>* to(encoding<char32_t> enc) const
        { return sani_to(enc); }

    virtual const transcoder<char,     CharT>* from(encoding<char> enc) const
        { return sani_from(enc); }
    virtual const transcoder<wchar_t,  CharT>* from(encoding<wchar_t> enc) const
        { return sani_from(enc); }
    virtual const transcoder<char16_t, CharT>* from(encoding<char16_t> enc) const
        { return sani_from(enc); }
    virtual const transcoder<char32_t, CharT>* from(encoding<char32_t> enc) const
        { return sani_from(enc); }

    constexpr const stringify::v0::decoder<CharT>& decoder() const
    {
        return m_decoder;
    }

    constexpr const stringify::v0::encoder<CharT>& encoder() const
    {
        return m_encoder;
    }

    template <typename C>
    bool equivalent(const stringify::v0::encoding_info<C>& other) const
    {
        return m_id == other.m_id;
    }

private:

    template <typename>
    friend class stringify::v0::encoding_info;

    const stringify::v0::decoder<CharT>& m_decoder;
    const stringify::v0::encoder<CharT>& m_encoder;
    const unsigned m_id;
};

template <typename CharT>
template <typename CharOut>
const stringify::v0::transcoder<CharT, CharOut>* encoding<CharT>::to
    (stringify::v0::encoding<CharOut> enc) const
{
    auto trans = m_info->to(enc.info());
    return trans != nullptr ? trans : enc.info().from(*m_info);
}

template <typename CharT>
template <typename CharOut>
const stringify::v0::transcoder<CharT, CharOut>* encoding<CharT>::sani_to
    (stringify::v0::encoding<CharOut> enc) const
{
    auto trans = m_info->sani_to(enc.info());
    return trans != nullptr ? trans : enc.info().sani_from(*m_info);
}

class encoding_error final
{
public:

    using category = stringify::v0::encoding_error_category;

    encoding_error()
    {
    }

    template <typename Arg>
    encoding_error(const Arg& arg)
        : m_err_sig(arg)
    {
    }

    const stringify::v0::error_signal& on_error() const
    {
        return m_err_sig;
    }
    template <typename T> encoding_error on_error(T sig) const &
    {
        return { stringify::v0::error_signal{sig} };
    }
    template <typename T> encoding_error& on_error(T sig) &
    {
        m_err_sig = stringify::v0::error_signal{sig};
        return *this;
    }
    template <typename T> encoding_error&& on_error(T sig) &&
    {
        return static_cast<encoding_error&& >(on_error(sig));
    }

private:

    stringify::v0::error_signal m_err_sig;
};


class allow_surrogates
{
public:

    using category = allow_surrogates_category;

    constexpr allow_surrogates(bool v) : m_value(v) {}

    constexpr allow_surrogates(const allow_surrogates& ) = default;

    constexpr bool value() const
    {
        return m_value;
    }
    constexpr allow_surrogates value(bool k) const &
    {
        return {k};
    }
    constexpr allow_surrogates& value(bool k) &
    {
        m_value = k;
        return *this;
    }
    constexpr allow_surrogates&& value(bool k) &&
    {
        return static_cast<allow_surrogates&&>(value(k));
    }

private:

    bool m_value;
};


template <typename CharIn, typename CharOut>
struct str_cv_result
{
    const CharIn* src_it;
    CharOut* dest_it;
    stringify::v0::cv_result result;
};


template <typename CharIn, typename CharOut>
class transcoder
{
public:

    virtual ~transcoder()
    {
    }

    using char_cv_result = stringify::v0::char_cv_result<CharOut>;
    using str_cv_result = stringify::v0::str_cv_result<CharIn, CharOut>;

    // from string

    virtual str_cv_result convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const = 0;

    virtual std::size_t necessary_size
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const = 0;

    // from single char

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const = 0;

    virtual char_cv_result convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const = 0;

    virtual std::size_t necessary_size
        ( CharIn ch
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const = 0;
};

template <typename CharOut>
class piecemeal_input
{
    enum e_status {waiting_more, successfully_complete, error_reported};

public:

    virtual ~piecemeal_input()
    {
    }

    virtual CharOut* get_some(CharOut* dest_begin, CharOut* dest_end) = 0;

    bool more() const
    {
        return m_status == waiting_more;
    }

    bool success() const
    {
        return m_status == successfully_complete;
    }

    std::error_code get_error() const
    {
        return m_err;
    }

protected:

    void report_error(std::error_code err)
    {
        BOOST_ASSERT(err != std::error_code{});
        BOOST_ASSERT(m_status == waiting_more);
        m_err = err;
        m_status = error_reported;
    }

    void report_success()
    {
        BOOST_ASSERT(m_status == waiting_more);
        m_status = successfully_complete;
    }

    // // helper functions
    // struct encode_char_result
    // {
    //     CharOut* it;
    //     char32_t ch;
    // };

    // encode_char_result encode_char
    //     ( const stringify::v0::encoder<CharOut>& encoder
    //     , const stringify::v0::error_signal& err_sig
    //     , char32_t ch
    //     , CharOut* dest
    //     , CharOut* dest_end
    //     , bool allow_surrogates );

private:

    std::error_code m_err;
    e_status m_status = waiting_more;
};


// template <typename CharOut>
// typename piecemeal_input<CharOut>::encode_char_result
// piecemeal_input<CharOut>::encode_char
//     ( const stringify::v0::encoder<CharOut>& encoder
//     , const stringify::v0::error_signal& err_sig
//     , char32_t ch
//     , CharOut* dest
//     , CharOut* dest_end
//     , bool allow_surrogates )
// {
//     auto it = encoder.encode(ch, dest, dest_end, allow_surrogates);
//     if (it != nullptr)
//     {
//         return {it, ch};
//     }
//     if(err_sig.has_char())
//     {
//         auto ech = err_sig.get_char();
//         if (ch != ech)
//         {
//             return encode_char( encoder, err_sig, ech
//                               , dest, dest_end, allow_surrogates );
//         }
//         auto it = encoder.encode(U'?', dest, dest_end, allow_surrogates);
//         BOOST_ASSERT(it != nullptr);
//         return {it, U'?'};
//     }
//     if(err_sig.has_error_code())
//     {
//         this->report_error(err_sig.get_error_code());
//         return {nullptr, ch};
//     }
//     if(err_sig.has_function())
//     {
//         err_sig.get_function() ();
//     }
//     return {dest, U'\0'};
// }

namespace detail {

template <typename CharIn, typename CharOut>
class str_pm_input: public stringify::v0::piecemeal_input<CharOut>
{
public:
    str_pm_input
        ( const stringify::v0::transcoder<CharIn, CharOut>& trans
        , stringify::v0::error_signal err_sig
        , bool allow_surrogates
        , const CharIn* src_it
        , const CharIn* src_end )
        noexcept
        : m_trans(trans)
        , m_err_sig(err_sig)
        , m_allow_surrogates(allow_surrogates)
        , m_src_it(src_it)
        , m_src_end(src_end)
    {
    }

    virtual CharOut* get_some(CharOut* dest_begin, CharOut* dest_end) override;

private:

    const stringify::v0::transcoder<CharIn, CharOut>& m_trans;
    const stringify::v0::error_signal m_err_sig;
    const bool m_allow_surrogates;
    const CharIn* m_src_it;
    const CharIn* m_src_end;
};

template <typename CharOut>
class char32_pm_input: public stringify::v0::piecemeal_input<CharOut>
{
public:
    char32_pm_input
        ( const stringify::v0::encoder<CharOut>& encoder
        , stringify::v0::error_signal err_sig
        , bool allow_surrogates
        , char32_t ch ) noexcept
        : m_encoder(encoder)
        , m_err_sig(err_sig)
        , m_allow_surrogates(allow_surrogates)
        , m_char(ch)
    {
    }

    virtual CharOut* get_some(CharOut* dest_begin, CharOut* dest_end) override;

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    const stringify::v0::error_signal m_err_sig;
    const bool m_allow_surrogates;
    char32_t m_char;
};

template <typename CharOut>
class repeated_char32_pm_input: public stringify::v0::piecemeal_input<CharOut>
{
public:
    repeated_char32_pm_input
        ( const stringify::v0::encoder<CharOut>& encoder
        , stringify::v0::error_signal err_sig
        , bool allow_surrogates
        , std::size_t count
        , char32_t ch )
        noexcept
        : m_encoder(encoder)
        , m_err_sig(err_sig)
        , m_allow_surrogates(allow_surrogates)
        , m_count(count)
        , m_char(ch)
    {
    }

    virtual CharOut* get_some(CharOut* dest_begin, CharOut* dest_end) override;

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    const stringify::v0::error_signal m_err_sig;
    const bool m_allow_surrogates;
    std::size_t m_count;
    char32_t m_char;
};

} // namespace detail

template <typename CharOut>
struct output_writer_init
{
    template <typename FPack>
    output_writer_init(const FPack& fp)
        : m_encoding
            { fp.template get_facet
                < stringify::v0::encoding_category<CharOut>
                , stringify::v0::output_writer<CharOut> >() }
        , m_encoding_err
            { fp.template get_facet
                < stringify::v0::encoding_error_category
                , stringify::v0::output_writer<CharOut> >
                ().on_error() }
        , m_allow_surr
            { fp.template get_facet
                < stringify::v0::allow_surrogates_category
                , stringify::v0::output_writer<CharOut> >
                ().value() }
    {
    }

private:

    template <typename>
    friend class output_writer;

    const stringify::v0::encoding<CharOut>& m_encoding;
    const stringify::v0::error_signal& m_encoding_err;
    bool m_allow_surr;
};

struct codepoint_validation_result
{
    std::size_t size;
    char32_t ch;
    bool error_emitted;
};

template <typename CharOut>
class output_writer
{
public:
    using char_type = CharOut;

    output_writer(stringify::v0::output_writer_init<CharOut> init)
        : m_encoding(init.m_encoding)
        , m_encoding_err(init.m_encoding_err)
        , m_allow_surr(init.m_allow_surr)
    {
    }

    const stringify::v0::encoder<CharOut>& encoder() const
    {
        return m_encoding.encoder();
    }

    stringify::v0::encoding<CharOut> encoding() const
    {
        return m_encoding;
    }

    bool allow_surrogates() const
    {
        return m_allow_surr;
    }

    const auto& on_encoding_error() const
    {
        return m_encoding_err;
    }

    stringify::v0::codepoint_validation_result validate(char32_t ch);

    std::size_t necessary_size(char32_t ch) const
    {
        return encoder().necessary_size(ch, m_encoding_err, m_allow_surr);
    }

    template <typename CharIn>
    bool put
        ( const stringify::v0::transcoder<CharIn, CharOut>& trans
        , const CharIn* src_begin
        , const CharIn* src_end )
    {
        stringify::v0::detail::str_pm_input<CharIn, CharOut> src
            { trans, m_encoding_err, m_allow_surr, src_begin, src_end };
        return put(src);
    }

    bool put32(std::size_t count, char32_t ch)
    {
        stringify::v0::detail::repeated_char32_pm_input<CharOut> src
            { encoder(), m_encoding_err, m_allow_surr, count, ch };
        return put(src);
    }

    bool put32(char32_t ch)
    {
        stringify::v0::detail::char32_pm_input<CharOut> src
            { encoder(), m_encoding_err, m_allow_surr, ch};
        return put(src);
    }

    virtual ~output_writer()
    {
    }

    virtual void set_error(std::error_code err) = 0;

    virtual bool good() const = 0;

    virtual bool put(stringify::v0::piecemeal_input<CharOut>& src) = 0;

    virtual bool put(const CharOut* str, std::size_t size) = 0;

    virtual bool put(CharOut ch) = 0;

    virtual bool put(std::size_t count, CharOut ch) = 0;

protected:

    CharOut* encode(CharOut* dest_it, CharOut* dest_end, char32_t ch) const
    {
        return encoder().encode(ch, dest_it, dest_end, allow_surrogates());
    }

    auto encode(CharOut* dest_it, CharOut* dest_end, std::size_t count, char32_t ch) const
    {
        return encoder().encode(count, ch, dest_it, dest_end, allow_surrogates());
    }

    constexpr static std::size_t buff_size = sizeof(CharOut) == 1 ? 6 : 2;
    CharOut buff[buff_size];

private:

    const stringify::v0::encoding<CharOut> m_encoding;
    const stringify::v0::error_signal m_encoding_err;
    bool m_allow_surr;
};


template <typename CharOut>
stringify::v0::codepoint_validation_result
output_writer<CharOut>::validate(char32_t ch)
{
    auto s = encoder().validate(ch, m_allow_surr);
    if (s != 0)
    {
        return {s, ch, false};
    }
    if(m_encoding_err.has_char())
    {
        s = encoder().validate(m_encoding_err.get_char(), m_allow_surr);
        if (s != 0)
        {
            return {s, m_encoding_err.get_char(), false};
        }
        return {s, U'?', false};
    }
    if(m_encoding_err.has_error_code())
    {
        set_error(m_encoding_err.get_error_code());
        return {0, ch, true};
    }
    if(m_encoding_err.has_function())
    {
        m_encoding_err.get_function() ();
    }
    return {0, ch, false};
}

namespace detail {

template <typename CharOut>
class length_accumulator: public stringify::v0::u32output
{
public:

    length_accumulator
        ( const stringify::v0::encoder<CharOut>& encoder
        , const stringify::v0::error_signal& err_sig
        , bool allow_surr
        )
        : m_encoder(encoder)
        , m_err_sig(err_sig)
        , m_allow_surr(allow_surr)
    {
    }

    stringify::v0::cv_result put32(char32_t ch) override
    {
        m_length += m_encoder.necessary_size(ch, m_err_sig, m_allow_surr);
        return stringify::v0::cv_result::success;
    }

    bool signalize_error() override
    {
        if(m_err_sig.has_char())
        {
            (void) this->put32(m_err_sig.get_char());
            return true;
        }
        if(m_err_sig.skip())
        {
            return true;
        }
        if(m_err_sig.has_function())
        {
            m_err_sig.get_function() ();
        }
        m_status = stringify::v0::cv_result::invalid_char;
        return false;
    }

    std::size_t get_length() const
    {
        return m_length;
    }

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    std::size_t m_length = 0;
    stringify::v0::error_signal m_err_sig;
    stringify::v0::cv_result m_status = stringify::v0::cv_result::success;
    bool m_allow_surr;
};


template <typename CharOut>
class encoder_adapter: public stringify::v0::u32output
{
public:
    encoder_adapter
        ( const stringify::v0::encoder<CharOut>& encoder
        , CharOut* dest_it
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surr
        )
        : m_encoder(encoder)
        , m_dest_it(dest_it)
        , m_dest_end(dest_end)
        , m_insufficient_space(dest_end + 1)
        , m_err_sig(err_sig)
        , m_allow_surr(allow_surr)
    {
    }

    stringify::v0::cv_result put32(char32_t ch) override
    {
        if (m_status == stringify::v0::cv_result::success)
        {
            CharOut* it = m_encoder.encode(ch, m_dest_it, m_dest_end, m_allow_surr);
            if (it != nullptr)
            {
                if (it != m_insufficient_space)
                {
                    m_dest_it = it;
                }
                else
                {
                    m_status = stringify::v0::cv_result::insufficient_space;
                }
            }
            else
            {
                signalize_error();
            }
        }
        return m_status;
    }

    bool signalize_error() override
    {
        auto it = stringify::v0::emit_error
            ( m_err_sig
            , m_encoder
            , m_dest_it
            , m_dest_end
            , m_allow_surr );

        if(nullptr == it)
        {
            m_status = stringify::v0::cv_result::invalid_char;
            return false;
        }
        if(m_dest_end + 1 == it)
        {
            m_status = stringify::v0::cv_result::insufficient_space;
            return false;
        }
        m_dest_it = it;
        return true;
    }

    stringify::v0::cv_result result() const
    {
        return m_status;
    }

    CharOut* dest_it() const
    {
        return m_dest_it;
    }

private:

    //bool signal_error_as_char()
    //{
    //     auto ch = m_err_sig.get_char();
    //     CharOut* it = m_encoder.encode(ch, m_dest_it, m_dest_end, m_allow_surr);
    //     if (it == nullptr)
    //     {
    //         if(ch != U'?')
    //         {
    //             m_err_sig.reset(U'?');
    //             return signal_error_as_char();
    //         }
    //         m_status = stringify::v0::cv_result::invalid_char;
    //         return false;
    //     }
    //     if(it == m_dest_end + 1)
    //     {
    //         m_status = stringify::v0::cv_result::insufficient_space;
    //         return false;
    //     }
    //     m_dest_it = it;
    //     return true;
    //}

    const stringify::v0::encoder<CharOut>& m_encoder;
    CharOut* m_dest_it;
    CharOut* const m_dest_end;
    CharOut* const m_insufficient_space;;
    stringify::v0::error_signal m_err_sig;
    stringify::v0::cv_result m_status = stringify::v0::cv_result::success;
    bool m_allow_surr;
};


template <typename CharIn, typename CharOut>
class decode_encode: public stringify::v0::transcoder<CharIn, CharOut>
{
public:

    decode_encode
        ( const stringify::v0::decoder<CharIn>& decoder
        , const stringify::v0::encoder<CharOut>& encoder )
        noexcept
        : m_decoder(decoder)
        , m_encoder(encoder)
    {
    }

    using char_cv_result = stringify::v0::char_cv_result<CharOut>;
    using str_cv_result = stringify::v0::str_cv_result<CharIn, CharOut>;

    virtual str_cv_result convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool allow_surrogates
        ) const override;

    virtual std::size_t necessary_size
        ( const CharIn* src_begin
        , const CharIn* src_end
        , const stringify::v0::error_signal & err_sig
        , bool allow_surrogates
        ) const override;

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal & err_sig
        , bool allow_surrogates
        ) const override;

    virtual char_cv_result convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal & err_sig
        , bool allow_surrogates
        ) const override;

    virtual std::size_t necessary_size
        ( CharIn ch
        , const stringify::v0::error_signal & err_sig
        , bool allow_surrogates
        ) const override;

private:

    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::encoder<CharOut>& m_encoder;

};

template <typename CharIn, typename CharOut>
stringify::v0::str_cv_result<CharIn, CharOut>
decode_encode<CharIn, CharOut>::convert
    ( const CharIn* src_begin
    , const CharIn* src_end
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal& err_sig
    , bool allow_surrogates
    ) const
{
    stringify::v0::detail::encoder_adapter<CharOut> adapter
        ( m_encoder, dest_begin, dest_end, err_sig, allow_surrogates );
    auto r = m_decoder.decode
        ( adapter, src_begin, src_end, allow_surrogates );

    return {r.src_it, adapter.dest_it(), adapter.result()};
}

template <typename CharIn, typename CharOut>
std::size_t decode_encode<CharIn, CharOut>::necessary_size
    ( const CharIn* src_begin
    , const CharIn* src_end
    , const stringify::v0::error_signal & err_sig
    , bool allow_surrogates
    ) const
{
    stringify::v0::detail::length_accumulator<CharOut> adapter
        ( m_encoder, err_sig, allow_surrogates );
    (void)m_decoder.decode
        ( adapter, src_begin, src_end, allow_surrogates );
    return adapter.get_length();
}

template <typename CharIn, typename CharOut>
CharOut* decode_encode<CharIn, CharOut>::convert
    ( CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal & err_sig
    , bool allow_surrogates
    ) const
{

    // TODO
    (void) ch;
    (void) dest_begin;
    (void) dest_end;
    (void) err_sig;
    (void) allow_surrogates;
    BOOST_ASSERT(dest_begin == dest_end);
    return dest_end + 1;
}

template <typename CharIn, typename CharOut>
stringify::v0::char_cv_result<CharOut> decode_encode<CharIn, CharOut>::convert
    ( std::size_t count
    , CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , const stringify::v0::error_signal & err_sig
    , bool allow_surrogates
    ) const
{
    //todo

    (void) count;
    (void) ch;
    (void) dest_begin;
    (void) dest_end;
    (void) err_sig;
    (void) allow_surrogates;
    using result_type = stringify::v0::char_cv_result<CharOut>;
    BOOST_ASSERT(dest_begin == dest_end);
    return result_type{0, dest_begin};
}

template <typename CharIn, typename CharOut>
std::size_t decode_encode<CharIn, CharOut>::necessary_size
    ( CharIn ch
    , const stringify::v0::error_signal & err_sig
    , bool allow_surrogates
    ) const
{
    (void) ch;
    (void) err_sig;
    (void) allow_surrogates;
    BOOST_ASSERT(ch == 0xFFFFFFF);
    return 0;  //todo
}

} // namespace detail


template <typename CharIn, typename CharOut>
class string_writer
{

    using decode_encode
    = stringify::v0::detail::decode_encode<CharIn, CharOut>;

public:

    string_writer
        ( stringify::v0::output_writer<CharOut>& out
        , const stringify::v0::encoding<CharIn> input_enc
        , bool sani )
        // noexcept
        : m_out(out)
        , m_transcoder(nullptr)
    {
        if (sani || input_enc != out.encoding())
        {
            init_transcoder(out, input_enc, sani);
        }
    }

    ~string_writer()
    {
        auto * de = reinterpret_cast<decode_encode*>(&m_pool);
        if(de == m_transcoder)
        {
            de->~decode_encode();
        }
    }

    stringify::v0::output_writer<CharOut>& destination() const
    {
        return m_out;
    }

    bool write(const CharIn* begin, std::size_t count) const
    {
        if (m_transcoder == nullptr)
        {
            return m_out.put(reinterpret_cast<const CharOut*>(begin), count);
        }
        else
        {
            return m_out.put(*m_transcoder, begin, begin + count);
        }
    }
    bool write(const CharIn* begin, const CharIn* end) const
    {
        if (m_transcoder == nullptr)
        {
            return m_out.put(reinterpret_cast<const CharOut*>(begin), end - begin);
        }
        else
        {
            return m_out.put(*m_transcoder, begin, end);
        }
    }

    // bool repeat(std::size_t count, CharIn ch) const //todo
    // {
    //     return m_transcoder == nullptr
    //         ? m_out.repeat(count, static_cast<CharOut>(ch))
    //         : m_out.put(*m_transcoder, count, ch);
    // }

    std::size_t necessary_size(const CharIn* begin, const CharIn* end) const
    {
        return m_transcoder == nullptr
            ? (end - begin)
            : m_transcoder->necessary_size
                ( begin
                , end
                , on_encoding_error()
                , allow_surrogates() );
    }
    std::size_t necessary_size(const CharIn* begin, std::size_t count) const
    {
        return m_transcoder == nullptr
            ? count
            : m_transcoder->necessary_size
                ( begin
                , begin + count
                , on_encoding_error()
                , allow_surrogates() );
    }


    bool put32(std::size_t count, char32_t ch) const
    {
        return m_out.put32(count, ch);
    }

    bool put32(char32_t ch) const
    {
        return m_out.put32(ch);
    }

    std::size_t necessary_size(char32_t ch) const
    {
        return m_out.necessary_size(ch);
    }

    const auto& on_encoding_error() const
    {
        return m_out.on_encoding_error();
    }

    bool allow_surrogates() const
    {
        return m_out.allow_surrogates();
    }


private:

    void init_transcoder
        ( stringify::v0::output_writer<CharOut>& out
        , const stringify::v0::encoding<CharIn> input_enc
        , bool sani );

    stringify::v0::output_writer<CharOut>& m_out;
    const stringify::v0::transcoder<CharIn, CharOut> *m_transcoder = nullptr;

    using storage_type = typename std::aligned_storage
        < sizeof(decode_encode), alignof(decode_encode) >
        :: type;
    storage_type m_pool;
};

template <typename CharIn, typename CharOut>
void string_writer<CharIn, CharOut>::init_transcoder
    ( stringify::v0::output_writer<CharOut>& out
    , const stringify::v0::encoding<CharIn> input_enc
    , bool sani )
{

    m_transcoder = sani
        ? input_enc.sani_to(out.encoding())
        : input_enc.to(out.encoding()) ;

    if (m_transcoder == nullptr)
    {
        auto * de = reinterpret_cast<decode_encode*>(&m_pool);
        new (de) decode_encode(input_enc.decoder(), out.encoder());
        m_transcoder = de;
    }
}

namespace detail{

template
    < class QFromFmt
    , class QToFmt
    , template <class, class ...> class ValueWithFmt
    , class ValueType
    , class ... QFmts >
struct mp_replace_fmt
{
    template <class QF>
    using f = std::conditional_t
        < std::is_same<QFromFmt, QF>::value
        , QToFmt
        , QF >;

    using type = ValueWithFmt< ValueType, f<QFmts> ... >;
};

template <class QFmt, class T>
struct fmt_helper_impl;

template
    < class QFmt
    , template <class, class ...> class ValueWithFmt
    , class ValueType
    , class ... QFmts>
struct fmt_helper_impl<QFmt, ValueWithFmt<ValueType, QFmts ...> >
{
    using derived_type = ValueWithFmt<ValueType, QFmts ...>;

    template <class QToFmt>
    using adapted_derived_type =
        typename stringify::v0::detail::mp_replace_fmt
            < QFmt, QToFmt, ValueWithFmt, ValueType, QFmts ...>::type;
};

template <class QFmt>
struct fmt_helper_impl<QFmt, void>
{
    using derived_type = typename QFmt::template fn<void>;

    template <class QToFmt>
    using adapted_derived_type = typename QToFmt::template fn<void>;
};

} // namespace detail

template <typename QFmt, typename Der>
using fmt_helper = stringify::v0::detail::fmt_helper_impl<QFmt, Der>;

template <typename QFmt, typename Der>
using fmt_derived
= typename stringify::v0::fmt_helper<QFmt, Der>::derived_type;

template <typename ValueType, class ... Fmts>
class value_with_format;

template <typename ValueType, class ... Fmts>
class value_with_format
    : public Fmts::template fn<value_with_format<ValueType, Fmts ...>> ...
{
public:

    template <typename U>
    using replace_value_type = stringify::v0::value_with_format<U, Fmts ...>;

    template <typename ... OhterFmts>
    using replace_fmts = stringify::v0::value_with_format<ValueType, OhterFmts ...>;

    constexpr value_with_format(const value_with_format&) = default;
    constexpr value_with_format(value_with_format&&) = default;

    explicit constexpr value_with_format(const ValueType& v)
        : m_value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , const stringify::v0::value_with_format<OtherValueType, Fmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename Fmts::template fn<value_with_format<OtherValueType, Fmts...>>& >(f) )
        ...
        , m_value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , const stringify::v0::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts::template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , m_value(static_cast<ValueType&&>(v))
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( const stringify::v0::value_with_format<ValueType, OtherFmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename OtherFmts::template fn<value_with_format<ValueType, OtherFmts ...>>& >(f) )
        ...
        , m_value(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( const stringify::v0::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts::template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , m_value(static_cast<ValueType&&>(f.value()))
    {
    }

    constexpr const ValueType& value() const
    {
        return m_value;
    }

private:

    ValueType m_value;
};


enum class alignment {left, right, internal, center};

namespace detail
{
  template <class T> class alignment_format_impl;
  template <class T> class empty_alignment_format_impl;
}

struct alignment_format
{
    template <class T> using fn = stringify::v0::detail::alignment_format_impl<T>;
};

struct empty_alignment_format
{
    template <class T> using fn = stringify::v0::detail::empty_alignment_format_impl<T>;
};

namespace detail {

template <class T = void>
class alignment_format_impl
{
    using derived_type = stringify::v0::fmt_derived<alignment_format, T>;

public:

    constexpr alignment_format_impl()
    {
        static_assert(std::is_base_of<alignment_format_impl, derived_type>::value, "");
    }

    constexpr alignment_format_impl(const alignment_format_impl&) = default;

    template <typename U>
    constexpr alignment_format_impl(const alignment_format_impl<U>& u)
        : m_fill(u.fill())
        , m_width(u.width())
        , m_alignment(u.alignment())
    {
    }

    template <typename U>
    constexpr alignment_format_impl(const empty_alignment_format_impl<U>&)
    {
    }

    ~alignment_format_impl()
    {
    }

    constexpr derived_type&& operator<(int width) &&
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator<(int width) &
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator>(int width) &&
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator>(int width) &
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator^(int width) &&
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator^(int width) &
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator%(int width) &&
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator%(int width) &
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& fill(char32_t ch) &&
    {
        m_fill = ch;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& fill(char32_t ch) &
    {
        m_fill = ch;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& width(int w) &&
    {
        m_width = w;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& width(int w) &
    {
        m_width = w;
        return static_cast<derived_type&>(*this);
    }
    constexpr int width() const
    {
        return m_width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return m_alignment;
    }
    constexpr char32_t fill() const
    {
        return m_fill;
    }

private:

    template <typename>
    friend class alignment_format_impl;

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
};

template <class T>
class empty_alignment_format_impl
{
    using helper = stringify::v0::fmt_helper<empty_alignment_format, T>;
    using derived_type = typename helper::derived_type;
    using adapted_derived_type
    = typename helper::template adapted_derived_type<stringify::v0::alignment_format>;

    constexpr adapted_derived_type make_adapted() const
    {
        return adapted_derived_type{static_cast<const derived_type&>(*this)};
    }
public:

    constexpr empty_alignment_format_impl()
    {
    }

    constexpr empty_alignment_format_impl(const empty_alignment_format_impl&) = default;

    template <typename U>
    constexpr empty_alignment_format_impl(const empty_alignment_format_impl<U>&)
    {
    }

    ~empty_alignment_format_impl()
    {
    }

    constexpr adapted_derived_type operator<(int width) const
    {
        return make_adapted() < width;
    }
    constexpr adapted_derived_type operator>(int width) const
    {
        return make_adapted() > width;
    }
    constexpr adapted_derived_type operator^(int width) const
    {
        return make_adapted() ^ width;
    }
    constexpr adapted_derived_type operator%(int width) const
    {
        return make_adapted() % width;
    }
    constexpr adapted_derived_type fill(char32_t ch) const
    {
        return make_adapted().fill(ch);
    }
    constexpr adapted_derived_type width(int w) const
    {
        return make_adapted().width(w);
    }

    constexpr int width() const
    {
        return 0;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return stringify::v0::alignment::right;
    }
    constexpr char32_t fill() const
    {
        return U' ';
    }
};

} // namespace detail

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual std::size_t necessary_size() const = 0;

    virtual void write() const = 0;

    virtual int remaining_width(int w) const = 0;
};


template <typename CharOut>
class printers_receiver
{
public:

    virtual ~printers_receiver()
    {
    }

    virtual bool put(const stringify::v0::printer<CharOut>& ) = 0;
};

namespace detail {

template <typename CharOut>
class width_subtracter: public printers_receiver<CharOut>
{
public:

    width_subtracter(int w)
        : m_width(w)
    {
    }

    bool put(const stringify::v0::printer<CharOut>& p) override;

    int remaining_width() const
    {
        return m_width;
    }

private:

    int m_width;
};

template <typename CharOut>
bool width_subtracter<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    m_width = p.remaining_width(m_width);
    return m_width > 0;
}

template <typename CharOut>
class necessary_size_sum: public printers_receiver<CharOut>
{
public:

    necessary_size_sum() = default;

    bool put(const stringify::v0::printer<CharOut>& p) override;

    std::size_t accumulated_length() const
    {
        return m_len;
    }

private:

    std::size_t m_len = 0;
};

template <typename CharOut>
bool necessary_size_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    m_len += p.necessary_size();
    return true;
}

template <typename CharOut>
class serial_writer: public printers_receiver<CharOut>
{
public:

    serial_writer() = default;

    bool put(const stringify::v0::printer<CharOut>& p) override;
};

template <typename CharOut>
bool serial_writer<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    p.write();
    return true;
}

} // namespace detail

template <typename CharOut>
class dynamic_join_printer: public stringify::v0::printer<CharOut>
{
public:

    dynamic_join_printer(stringify::v0::output_writer<CharOut>& out)
        : m_out(out)
    {
    }

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

protected:

    virtual stringify::v0::alignment_format::fn<void> formatting() const;

    virtual void compose(stringify::v0::printers_receiver<CharOut>& out) const = 0;

private:

    void write_with_fill(int fillcount) const;

    void write_without_fill() const;

    stringify::v0::output_writer<CharOut>& m_out;
};

template <typename CharOut>
stringify::v0::alignment_format::fn<void>
dynamic_join_printer<CharOut>::formatting() const
{
    return {};
}

template <typename CharOut>
std::size_t dynamic_join_printer<CharOut>::necessary_size() const
{
    std::size_t fill_len = 0;
    const auto fmt = formatting();
    if(fmt.width() > 0)
    {
        stringify::v0::detail::width_subtracter<CharOut> wds{fmt.width()};
        compose(wds);
        std::size_t fillcount = wds.remaining_width();
        fill_len = m_out.necessary_size(fmt.fill()) * fillcount;
    }

    stringify::v0::detail::necessary_size_sum<CharOut> s;
    compose(s);
    return s.accumulated_length() + fill_len;
}

template <typename CharOut>
int dynamic_join_printer<CharOut>::remaining_width(int w) const
{
    const auto fmt_width = formatting().width();
    if (fmt_width > w)
    {
        return 0;
    }

    stringify::v0::detail::width_subtracter<CharOut> s{w};
    compose(s);
    int rw = s.remaining_width();
    return (w - rw < fmt_width) ? (w - fmt_width) : rw;
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write() const
{
    auto fmt = formatting();
    auto fillcount = fmt.width();
    if(fillcount > 0)
    {
        stringify::v0::detail::width_subtracter<CharOut> wds{fillcount};
        compose(wds);
        fillcount = wds.remaining_width();
    }
    if(fillcount > 0)
    {
        write_with_fill(fillcount);
    }
    else
    {
        write_without_fill();
    }
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write_without_fill() const
{
    stringify::v0::detail::serial_writer<CharOut> s;
    compose(s);
}

template <typename CharOut>
void dynamic_join_printer<CharOut>::write_with_fill(int fillcount) const
{
    auto fmt = formatting();
    switch (fmt.alignment())
    {
        case stringify::v0::alignment::left:
        {
            write_without_fill();
            m_out.put32(fillcount, fmt.fill());
            break;
        }
        case stringify::v0::alignment::center:
        {
            auto halfcount = fillcount / 2;
            m_out.put32(halfcount, fmt.fill());
            write_without_fill();
            m_out.put32(fillcount - halfcount, fmt.fill());
            break;
        }
        //case stringify::v0::alignment::internal:
        //case stringify::v0::alignment::right:
        default:
        {
            m_out.put32(fillcount, fmt.fill());
            write_without_fill();
        }
    }
}


template <typename T>
struct identity
{
    using type = T;
};

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharIn>
struct asm_string_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: stringify::v0::string_input_tag<CharIn>
{
};



template <typename CharIn>
struct is_asm_string_of
{
    template <typename T>
    using fn = std::is_same<stringify::v0::asm_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_asm_string: std::false_type
{
};

template <typename CharIn>
struct is_asm_string<stringify::v0::is_asm_string_of<CharIn>> : std::true_type
{
};

/*
template <typename CharOut>
bool output_writer<CharOut>::signal_encoding_error()
{
    const auto& err_sig = m_encoding.on_error();
    if (err_sig.has_char())
    {
        CharOut* it = encoder().encode
            ( err_sig.get_char()
            , buff
            , buff + buff_size
            , m_encoding.allow_surrogates() );

        if (it != nullptr && it != buff + buff_size + 1)
        {
            return put(buff, it - buff);
        }
        return put('?');
    }
    if (err_sig.has_error_code())
    {
        set_error(err_sig.get_error_code());
    }
    else if (err_sig.has_function())
    {
        err_sig.get_function()();
    }
    return false;
}
*/
namespace detail {

template <typename CharIn, typename CharOut>
CharOut* str_pm_input<CharIn, CharOut>::get_some
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    auto res = m_trans.convert
        ( m_src_it
        , m_src_end
        , dest_begin
        , dest_end
        , m_err_sig
        , m_allow_surrogates );

    if(res.result == stringify::v0::cv_result::success)
    {
        this->report_success();
    }
    else if(res.result == stringify::v0::cv_result::invalid_char)
    {
        BOOST_ASSERT(m_err_sig.has_error_code());
        this->report_error(m_err_sig.get_error_code());
    }
    BOOST_ASSERT(m_src_it <= res.src_it);
    BOOST_ASSERT(res.src_it <= m_src_end);
    BOOST_ASSERT(dest_begin <= res.dest_it);
    BOOST_ASSERT(res.dest_it <= dest_end);
    m_src_it = res.src_it;
    return res.dest_it;
}

template <typename CharOut>
CharOut* char32_pm_input<CharOut>::get_some
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    CharOut* it = m_encoder.encode
        ( m_char
        , dest_begin
        , dest_end
        , m_allow_surrogates );

    if(it == nullptr)
    {
        BOOST_ASSERT(m_err_sig.has_error_code());
        this->report_error(m_err_sig.get_error_code());
        return dest_begin;
    }
    if(it != dest_end + 1)
    {
        BOOST_ASSERT(dest_begin < it && it <= dest_end);
        this->report_success();
        return it;
    }
    return dest_begin;
}

template <typename CharOut>
CharOut* repeated_char32_pm_input<CharOut>::get_some
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    auto res = m_encoder.encode
        ( m_count
        , m_char
        , dest_begin
        , dest_end
        , m_allow_surrogates );

    if(res.dest_it == nullptr)
    {
        BOOST_ASSERT(m_err_sig.has_error_code());
        this->report_error(m_err_sig.get_error_code());
        return dest_begin;
    }
    if(res.count == m_count)
    {
        BOOST_ASSERT(dest_begin < res.dest_it || m_count == 0);
        BOOST_ASSERT(res.dest_it <= dest_end);
        this->report_success();
    }
    m_count -= res.count;
    return res.dest_it;
}

} // namespace detail

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

namespace detail{

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class decode_encode<char32_t, char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_pm_input<char32_t, char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_pm_input<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_pm_input<wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_pm_input<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_pm_input<char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_pm_input<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_pm_input<wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_pm_input<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_pm_input<char32_t>;

} // namespace detail

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_writer<char32_t, char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class output_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class output_writer<wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class output_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class output_writer<char32_t>;

#endif

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

