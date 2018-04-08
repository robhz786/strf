#ifndef BOOST_STRINGIFY_V0_BASIC_TYPES_HPP
#define BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharIn>  class encoding_info;
template <typename CharIn>  class decoder;
template <typename CharOut> class encoder;
template <typename CharIn, typename CharOut> class transcoder;
template <typename CharOut> class output_writer;
template <typename CharOut> struct output_encoding_category;
template <typename CharIn> struct input_encoding_category;
template <typename CharOut> class output_encoding;
template <typename CharIn> class input_encoding;

class error_signal
{
public:

    typedef void (*func_ptr)(void);

    explicit error_signal(char32_t ch = U'\uFFFD') noexcept
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

    error_signal& operator=(const error_signal& other) noexcept;

    bool operator==(const error_signal& other) noexcept;

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

    const std::error_code* error_code_ptr() const
    {
        return reinterpret_cast<const std::error_code*>(&m_error_code_storage);
    }

    enum { e_char, e_error_code, e_function } m_variant;
    using error_code_storage_type
    = std::aligned_storage_t<sizeof(std::error_code), alignof(std::error_code)>;

    union
    {
        char32_t m_char;
        error_code_storage_type m_error_code_storage;
        func_ptr m_function;
    };
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
        case e_function:
            m_function = other.m_function;
            break;
        default:
            new (&m_error_code_storage) std::error_code(other.get_error_code());
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

template <typename CharT>
class encoding_id
{
public:

    constexpr encoding_id(const stringify::v0::encoding_info<CharT>& info)
        : m_info(&info)
    {
    }

    constexpr encoding_id(const encoding_id&) = default;

    constexpr encoding_id& operator=(encoding_id other)
    {
        m_info = other.m_info;
    }


    template <typename Ch>
    constexpr bool operator==(encoding_id<Ch> other) const
    {
        return m_info->equivalent(other.info());
    }

    template <typename Ch>
    constexpr bool operator!=(encoding_id<Ch> other) const
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
    to(stringify::v0::encoding_id<CharOut> id) const;

    template <typename CharOut>
    const stringify::v0::transcoder<CharT, CharOut>*
    sani_to(stringify::v0::encoding_id<CharOut> id) const;

private:

    const stringify::v0::encoding_info<CharT>* m_info;
};

template <typename CharT>
class encoding_info
{
public:

    encoding_info
        ( const stringify::v0::decoder<CharT>& decoder
        , const stringify::v0::encoder<CharT>& encoder )
        noexcept
        : m_decoder(decoder)
        , m_encoder(encoder)
        , m_equiv(this)
    {
    }

    template <typename CharT_2>
    encoding_info
        ( const stringify::v0::decoder<CharT>& decoder
        , const stringify::v0::encoder<CharT>& encoder
        , const encoding_info<CharT_2>& equiv )
        noexcept
        : m_decoder(decoder)
        , m_encoder(encoder)
        , m_equiv
            ( sizeof(CharT) == sizeof(CharT_2)
            ? (const void*)&equiv
            : (const void*)this )
    {
    }

    virtual ~encoding_info() = default;

    virtual const transcoder<CharT, char>*     sani_to(encoding_id<char> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<CharT, wchar_t>*  sani_to(encoding_id<wchar_t> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<CharT, char16_t>* sani_to(encoding_id<char16_t> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<CharT, char32_t>* sani_to(encoding_id<char32_t> eid) const
        { (void)eid; return nullptr; }

    virtual const transcoder<char,     CharT>* sani_from(encoding_id<char> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<wchar_t,  CharT>* sani_from(encoding_id<wchar_t> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<char16_t, CharT>* sani_from(encoding_id<char16_t> eid) const
        { (void)eid; return nullptr; }
    virtual const transcoder<char32_t, CharT>* sani_from(encoding_id<char32_t> eid) const
        { (void)eid; return nullptr; }

    virtual const transcoder<CharT, char>*     to(encoding_id<char> eid) const
        { return sani_to(eid); }
    virtual const transcoder<CharT, wchar_t>*  to(encoding_id<wchar_t> eid) const
        { return sani_to(eid); }
    virtual const transcoder<CharT, char16_t>* to(encoding_id<char16_t> eid) const
        { return sani_to(eid); }
    virtual const transcoder<CharT, char32_t>* to(encoding_id<char32_t> eid) const
        { return sani_to(eid); }

    virtual const transcoder<char,     CharT>* from(encoding_id<char> eid) const
        { return sani_from(eid); }
    virtual const transcoder<wchar_t,  CharT>* from(encoding_id<wchar_t> eid) const
        { return sani_from(eid); }
    virtual const transcoder<char16_t, CharT>* from(encoding_id<char16_t> eid) const
        { return sani_from(eid); }
    virtual const transcoder<char32_t, CharT>* from(encoding_id<char32_t> eid) const
        { return sani_from(eid); }

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
        return m_equiv == other.m_equiv;
    }

private:

    template <typename>
    friend class stringify::v0::encoding_info;

    const stringify::v0::decoder<CharT>& m_decoder;
    const stringify::v0::encoder<CharT>& m_encoder;
    const void* m_equiv;
};

template <typename CharT>
template <typename CharOut>
const stringify::v0::transcoder<CharT, CharOut>* encoding_id<CharT>::to
    (stringify::v0::encoding_id<CharOut> id) const
{
    auto trans = m_info->to(id.info());
    return trans != nullptr ? trans : id.info().from(*m_info);
}

template <typename CharT>
template <typename CharOut>
const stringify::v0::transcoder<CharT, CharOut>* encoding_id<CharT>::sani_to
    (stringify::v0::encoding_id<CharOut> id) const
{
    auto trans = m_info->sani_to(id.info());
    return trans != nullptr ? trans : id.info().sani_from(*m_info);
}

template <typename CharIn>
class input_encoding final
{
public:
    using category = stringify::v0::input_encoding_category<CharIn>;
    stringify::v0::encoding_id<CharIn> id;

    const stringify::v0::decoder<CharIn>& decoder() const
    {
        return id.decoder();
    }
};

template <typename CharOut>
class output_encoding final
{
public:

    using category = stringify::v0::output_encoding_category<CharOut>;
    stringify::v0::encoding_id<CharOut> id;
    stringify::v0::error_signal err_sig;  // todo make private
    bool keep_surr = false;  // todo make private

    const stringify::v0::encoder<CharOut>& encoder() const
    {
        return id.encoder();
    }
    const stringify::v0::error_signal& on_error() const
    {
        return err_sig;
    }
    template <typename T> output_encoding on_error(T sig) const &
    {
        return {id, stringify::v0::error_signal{sig}, keep_surr};
    }
    template <typename T> output_encoding& on_error(T sig) &
    {
        err_sig = stringify::v0::error_signal{sig};
        return *this;
    }
    template <typename T> output_encoding&& on_error(T sig) &&
    {
        return static_cast<output_encoding&& >(on_error(sig));
    }

    bool keep_surrogates() const
    {
        return keep_surr;
    }
    output_encoding keep_surrogates(bool k) const &
    {
        return {id, err_sig, k};
    }
    output_encoding& keep_surrogates(bool k) &
    {
        keep_surr = k;
        return *this;
    }
    output_encoding&& keep_surrogates(bool k) &&
    {
        return static_cast<output_encoding&&>(keep_surrogates(k));
    }

};

class u32output
{
public:

    virtual ~u32output()
    {
    }

    virtual bool put32(char32_t ch) = 0;

    virtual bool signal_error() = 0;
};

enum class cv_result
{
    success,
    invalid_char,
    insufficient_space
};

template <typename CharOut>
struct char_cv_result
{
    std::size_t count;
    CharOut* dest_it;
};

template <typename CharIn, typename CharOut>
struct str_cv_result
{
    const CharIn* src_it;
    CharOut* dest_it;
    stringify::v0::cv_result result;
};


template <typename CharIn>
class decoder
{
public:

    virtual ~decoder() = default;

    virtual const CharIn* decode
        ( stringify::v0::u32output& dest
        , const CharIn* begin
        , const CharIn* end
        , bool keep_surrogates
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
class encoder
{
public:

    virtual ~encoder() = default;

    virtual std::size_t length
        ( char32_t ch
        , bool keep_surrogates )
        const = 0;

    virtual stringify::v0::char_cv_result<CharOut> convert
        ( std::size_t count
        , char32_t ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool keep_surrogates
        ) const = 0;

    /**
       return
       - success: != nullptr && <=end
       - space is insufficient : end + 1
       - ch is invalid or not supported: nullptr
     */
    virtual CharOut* convert
        ( char32_t ch
        , CharOut* dest
        , CharOut* dest_end
        , bool keep_surrogates
        ) const = 0;
};


template <typename CharIn, typename CharOut>
class transcoder
{
public:

    virtual ~transcoder() = default;

    using char_cv_result = stringify::v0::char_cv_result<CharOut>;
    using str_cv_result = stringify::v0::str_cv_result<CharIn, CharOut>;

    // from string

    virtual str_cv_result convert
        ( const CharIn* src_begin
        , const CharIn* src_end
        , CharOut* dest_begin
        , CharOut* dest_end
        , const stringify::v0::error_signal& err_sig
        , bool keep_surrogates
        ) const = 0;

    virtual std::size_t required_size
        ( const CharIn* begin
        , const CharIn* end
        , bool keep_surrogates
        ) const = 0;

    // from single char

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool keep_surrogates
        ) const = 0;

    virtual char_cv_result convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool keep_surrogates
        ) const = 0;

    virtual std::size_t required_size
        ( CharIn ch
        , bool keep_surrogates
        ) const = 0;
};


template <typename CharOut>
class source
{
public:

    virtual ~source() = default;

    virtual CharOut* get(CharOut* dest_begin, CharOut* dest_end) = 0;

    bool more()
    {
        return m_status == stringify::v0::cv_result::insufficient_space;
    }

    bool success()
    {
        return m_status == stringify::v0::cv_result::success;
    }

protected:

    stringify::v0::cv_result m_status
    = stringify::v0::cv_result::insufficient_space;
};

namespace detail {

template <typename CharIn, typename CharOut>
class str_source: public stringify::v0::source<CharOut>
{
public:
    str_source
        ( const stringify::v0::transcoder<CharIn, CharOut>& trans
        , stringify::v0::output_writer<CharOut>& out
        , const CharIn* src_it
        , const CharIn* src_end )
        noexcept
        : m_trans(trans)
        , m_out(out)
        , m_src_it(src_it)
        , m_src_end(src_end)
    {
    }

    virtual CharOut* get(CharOut* dest_begin, CharOut* dest_end) override;

private:

    const stringify::v0::transcoder<CharIn, CharOut>& m_trans;
    stringify::v0::output_writer<CharOut>& m_out;
    const CharIn* m_src_it;
    const CharIn* m_src_end;
};

template <typename CharOut>
class char32_source: public stringify::v0::source<CharOut>
{
public:
    char32_source(stringify::v0::output_writer<CharOut>& out, char32_t ch) noexcept
        : m_out(out)
        , m_char(ch)
    {
    }

    virtual CharOut* get(CharOut* dest_begin, CharOut* dest_end) override;

private:

    stringify::v0::output_writer<CharOut>& m_out;
    char32_t m_char;
};

template <typename CharOut>
class repeated_char32_source: public stringify::v0::source<CharOut>
{
public:
    repeated_char32_source
        ( stringify::v0::output_writer<CharOut>& out
        , std::size_t count
        , char32_t ch )
        noexcept
        : m_out(out)
        , m_count(count)
        , m_char(ch)
    {
    }

    virtual CharOut* get(CharOut* dest_begin, CharOut* dest_end) override;

private:

    stringify::v0::output_writer<CharOut>& m_out;
    std::size_t m_count;
    char32_t m_char;
};

} // namespace detail

template <typename CharOut>
struct output_writer_init
{
    template <typename FTuple>
    output_writer_init(const FTuple& ft)
        : m_encoding
            { ft.template get_facet
                < stringify::v0::output_encoding_category<CharOut>
                , stringify::v0::output_writer<char> >() }
    {
    }

    const stringify::v0::output_encoding<CharOut>& m_encoding;
};


template <typename CharOut>
class output_writer
{
public:
    using char_type = CharOut;

    output_writer(stringify::v0::output_writer_init<CharOut> init)
        : m_encoding(init.m_encoding)
    {
    }

    const stringify::v0::encoder<CharOut>& encoder() const
    {
        return m_encoding.id.encoder();
    }

    stringify::v0::encoding_id<CharOut> encoding() const
    {
        return m_encoding.id;
    }

    bool keep_surrogates() const
    {
        return m_encoding.keep_surrogates();
    }

    const auto& on_error() const
    {
        return m_encoding.on_error();
    }

    std::size_t required_size(char32_t ch) const
    {
        return encoder().length(ch, keep_surrogates());
    }

    bool signal_encoding_error();

    void set_error_invalid_char()
    {
        set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    }


    template <typename CharIn>
    bool put
        ( const stringify::v0::transcoder<CharIn, CharOut>& trans
        , const CharIn* src_begin
        , const CharIn* src_end )
    {
        stringify::v0::detail::str_source<CharIn, CharOut> src
            ( trans, *this, src_begin, src_end );
        return put(src);
    }

    bool put32(std::size_t count, char32_t ch)
    {
        stringify::v0::detail::repeated_char32_source<CharOut> src{*this, count, ch};
        return put(src);
    }

    bool put32(char32_t ch)
    {
        stringify::v0::detail::char32_source<CharOut> src{*this, ch};
        return put(src);
    }

    virtual ~output_writer()
    {
    }

    virtual void set_error(std::error_code err) = 0;

    virtual bool good() const = 0;

    virtual bool put(stringify::v0::source<CharOut>& src) = 0;

    virtual bool put(const CharOut* str, std::size_t size) = 0;

    virtual bool put(CharOut ch) = 0;

    virtual bool put(std::size_t count, CharOut ch) = 0;

protected:

    CharOut* encode(CharOut* dest_it, CharOut* dest_end, char32_t ch) const
    {
        return encoder().convert(ch, dest_it, dest_end, keep_surrogates());
    }

    auto encode(CharOut* dest_it, CharOut* dest_end, std::size_t count, char32_t ch) const
    {
        return encoder().convert(count, ch, dest_it, dest_end, keep_surrogates());
    }

    constexpr static std::size_t buff_size = sizeof(CharOut) == 1 ? 6 : 2;
    CharOut buff[buff_size];

private:

    const stringify::v0::output_encoding<CharOut> m_encoding;
};

namespace detail {

template <typename CharOut>
class length_accumulator: public stringify::v0::u32output
{
public:

    length_accumulator
        ( const stringify::v0::encoder<CharOut>& encoder
        , bool keep_surr
        )
        : m_encoder(encoder)
        , m_keep_surr(keep_surr)
    {
    }

    bool put32(char32_t ch) override
    {
        m_length += m_encoder.length(ch, m_keep_surr);
        return true;
    }

    bool signal_error() override
    {
        if(m_err_sig.has_char())
        {
            auto ch = m_err_sig.get_char();
            return put32(ch);
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
    bool m_keep_surr;
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
        , bool keep_surr
        )
        : m_encoder(encoder)
        , m_dest_it(dest_it)
        , m_dest_end(dest_end)
        , m_insufficient_space(dest_end + 1)
        , m_err_sig(err_sig)
        , m_keep_surr(keep_surr)
    {
    }

    bool put32(char32_t ch) override
    {
        if (m_status == stringify::v0::cv_result::success)
        {
            CharOut* it = m_encoder.convert(ch, m_dest_it, m_dest_end, m_keep_surr);
            if (it != nullptr)
            {
                if (it != m_insufficient_space)
                {
                    m_dest_it = it;
                    return true;
                }
                else
                {
                    m_status = stringify::v0::cv_result::insufficient_space;
                    return false;
                }
            }
            return signal_error();
        }
        return false;
    }

    bool signal_error() override
    {
        if(m_err_sig.has_char())
        {
            auto ch = m_err_sig.get_char();
            CharOut* it = m_encoder.convert(ch, m_dest_it, m_dest_end, m_keep_surr);
            if (it == nullptr)
            {
                m_status = stringify::v0::cv_result::invalid_char;
                return false;
            }
            if (it == m_dest_end + 1)
            {
                m_status = stringify::v0::cv_result::insufficient_space;
                return false;
            }
            m_dest_it = it;
            return true;
        }
        if(m_err_sig.has_function())
        {
            m_err_sig.get_function() ();
        }
        m_status = stringify::v0::cv_result::invalid_char;
        return false;
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

    const stringify::v0::encoder<CharOut>& m_encoder;
    CharOut* m_dest_it;
    CharOut* const m_dest_end;
    CharOut* const m_insufficient_space;;
    stringify::v0::error_signal m_err_sig;
    stringify::v0::cv_result m_status = stringify::v0::cv_result::success;
    bool m_keep_surr;
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
        , bool keep_surrogates
        ) const override;

    virtual std::size_t required_size
        ( const CharIn* src_begin
        , const CharIn* src_end
        , bool keep_surrogates
        ) const override;

    virtual CharOut* convert
        ( CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool keep_surrogates
        ) const override;

    virtual char_cv_result convert
        ( std::size_t count
        , CharIn ch
        , CharOut* dest_begin
        , CharOut* dest_end
        , bool keep_surrogates
        ) const override;

    virtual std::size_t required_size
        ( CharIn ch
        , bool keep_surrogates
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
    , bool keep_surrogates
    ) const
{
    stringify::v0::detail::encoder_adapter<CharOut> adapter
        ( m_encoder, dest_begin, dest_end, err_sig, keep_surrogates );
    const CharIn* src_it = m_decoder.decode
        ( adapter, src_begin, src_end, keep_surrogates );

    return {src_it, adapter.dest_it(), adapter.result()};
}

template <typename CharIn, typename CharOut>
std::size_t decode_encode<CharIn, CharOut>::required_size
    ( const CharIn* src_begin
    , const CharIn* src_end
    , bool keep_surrogates
    ) const
{
    stringify::v0::detail::length_accumulator<CharOut> adapter
        ( m_encoder, keep_surrogates );
    m_decoder.decode
        ( adapter, src_begin, src_end, keep_surrogates );
    return adapter.get_length();
}

template <typename CharIn, typename CharOut>
CharOut* decode_encode<CharIn, CharOut>::convert
    ( CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , bool keep_surrogates
    ) const
{

     //todo
    (void) ch;
    (void) keep_surrogates;
    (void) dest_begin;
    (void) dest_end;
    BOOST_ASSERT(dest_begin == dest_end);
    return dest_end + 1;
}

template <typename CharIn, typename CharOut>
stringify::v0::char_cv_result<CharOut> decode_encode<CharIn, CharOut>::convert
    ( std::size_t count
    , CharIn ch
    , CharOut* dest_begin
    , CharOut* dest_end
    , bool keep_surrogates
    ) const
{
    //todo

    (void) count;
    (void) ch;
    (void) dest_begin;
    (void) dest_end;
    (void) keep_surrogates;
    using result_type = stringify::v0::char_cv_result<CharOut>;
    BOOST_ASSERT(dest_begin == dest_end);
    return result_type{0, dest_begin};
}

template <typename CharIn, typename CharOut>
std::size_t decode_encode<CharIn, CharOut>::required_size
    ( CharIn ch
    , bool keep_surrogates
    ) const
{
    (void) ch;
    (void) keep_surrogates;
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
        , const stringify::v0::encoding_id<CharIn> input_enc_id
        , bool sani )
        // noexcept
        : m_out(out)
        , m_transcoder(nullptr)
    {
        if (sani || input_enc_id != out.encoding())
        {
            init_transcoder(out, input_enc_id, sani);
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

    std::size_t length(const CharIn* begin, const CharIn* end) const
    {
        return m_transcoder == nullptr
            ? (end - begin)
            : m_transcoder->required_size(begin, end, keep_surrogates());
    }
    std::size_t length(const CharIn* begin, std::size_t count) const
    {
        return m_transcoder == nullptr
            ? count
            : m_transcoder->required_size(begin, begin + count, keep_surrogates());
    }


    bool put32(std::size_t count, char32_t ch) const
    {
        return m_out.put32(count, ch);
    }

    bool put32(char32_t ch) const
    {
        return m_out.put32(ch);
    }

    std::size_t required_size(char32_t ch) const
    {
        return m_out.required_size(ch);
    }

    const auto& on_error() const
    {
        return m_out.on_error();
    }

    bool keep_surrogates() const
    {
        return m_out.keep_surrogates();
    }


private:

    void init_transcoder
        ( stringify::v0::output_writer<CharOut>& out
        , const stringify::v0::encoding_id<CharIn> input_enc_id
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
    , const stringify::v0::encoding_id<CharIn> input_enc_id
    , bool sani )
{

    m_transcoder = sani
        ? input_enc_id.sani_to(out.encoding())
        : input_enc_id.to(out.encoding()) ;

    if (m_transcoder == nullptr)
    {
        auto * de = reinterpret_cast<decode_encode*>(&m_pool);
        new (de) decode_encode(input_enc_id.decoder(), out.encoder());
        m_transcoder = de;
    }
}


template <typename CharIn>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual std::size_t length() const = 0;

    virtual void write() const = 0;

    virtual int remaining_width(int w) const = 0;
};


enum class alignment {left, right, internal, center};

template <class T = void>
class align_formatting
{
    using child_type = typename std::conditional
        < std::is_void<T>::value
        , align_formatting<void>
        , T
        > :: type;

public:

    template <typename U>
    using other = stringify::v0::align_formatting<U>;

    constexpr align_formatting()
    {
        static_assert(std::is_base_of<align_formatting, child_type>::value, "");
    }

    constexpr align_formatting(const align_formatting&) = default;

    template <typename U>
    constexpr align_formatting(const align_formatting<U>& other)
        : m_fill(other.m_fill)
        , m_width(other.m_width)
        , m_alignment(other.m_alignment)
    {
    }

    ~align_formatting() = default;

    template <typename U>
    constexpr child_type& format_as(const align_formatting<U>& other) &
    {
        m_fill = other.m_fill;
        m_width = other.m_width;
        m_alignment = other.m_alignment;
        return static_cast<child_type&>(*this);
    }

    template <typename U>
    constexpr child_type&& format_as(const align_formatting<U>& other) &&
    {
        return static_cast<child_type&&>(format_as(other));
    }

    constexpr child_type&& operator<(int width) &&
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator<(int width) &
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator>(int width) &&
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator>(int width) &
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator^(int width) &&
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator^(int width) &
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator%(int width) &&
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator%(int width) &
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& fill(char32_t ch) &&
    {
        m_fill = ch;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& fill(char32_t ch) &
    {
        m_fill = ch;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& width(int w) &&
    {
        m_width = w;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& width(int w) &
    {
        m_width = w;
        return static_cast<child_type&>(*this);
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
    friend class align_formatting;

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
};

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


template <typename CharOut>
bool output_writer<CharOut>::signal_encoding_error()
{
    const auto& err_sig = m_encoding.on_error();
    if (err_sig.has_char())
    {
        CharOut* it = encoder().convert
            ( err_sig.get_char()
            , buff
            , buff + buff_size
            , m_encoding.keep_surrogates() );

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

namespace detail {

template <typename CharIn, typename CharOut>
CharOut* str_source<CharIn, CharOut>::get
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    auto res = m_trans.convert
        ( m_src_it
        , m_src_end
        , dest_begin
        , dest_end
        , m_out.on_error()
        , m_out.keep_surrogates() );

    this->m_status = res.result;
    if(res.result == stringify::v0::cv_result::invalid_char)
    {
        if(m_out.on_error().has_error_code())
        {
            m_out.set_error(m_out.on_error().get_error_code());
        }
        else if(m_out.on_error().has_function())
        {
            m_out.on_error().get_function()();
        }
        else
        {
            m_out.set_error_invalid_char();
        }
    }
    BOOST_ASSERT(m_src_it <= res.src_it);
    BOOST_ASSERT(res.src_it <= m_src_end);
    BOOST_ASSERT(dest_begin <= res.dest_it);
    BOOST_ASSERT(res.dest_it <= dest_end);
    m_src_it = res.src_it;
    return res.dest_it;
}

template <typename CharOut>
CharOut* char32_source<CharOut>::get
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    CharOut* it = m_out.encoder().convert
        ( m_char
        , dest_begin
        , dest_end
        , m_out.keep_surrogates() );

    if(it == nullptr)
    {

        if(m_out.on_error().has_char())
        {
            auto echar = m_out.on_error().get_char();
            m_char = echar == m_char ? U'?' : echar;
            return get(dest_begin, dest_end);
        }
        else if(m_out.on_error().has_error_code())
        {
            m_out.set_error(m_out.on_error().get_error_code());
        }
        else if(m_out.on_error().has_function())
        {
            m_out.on_error().get_function()();
        }
        this->m_status = stringify::v0::cv_result::invalid_char;
        return dest_begin;
    }
    if(it != dest_end + 1)
    {
        BOOST_ASSERT(dest_begin < it && it <= dest_end);
        this->m_status = stringify::v0::cv_result::success;
        return it;
    }
    this->m_status = stringify::v0::cv_result::insufficient_space;
    return dest_begin;
}

template <typename CharOut>
CharOut* repeated_char32_source<CharOut>::get
    ( CharOut* dest_begin
    , CharOut* dest_end )
{
    auto res = m_out.encoder().convert
        ( m_count
        , m_char
        , dest_begin
        , dest_end
        , m_out.keep_surrogates() );

    if(res.dest_it == nullptr)
    {
        if(m_out.on_error().has_char())
        {
            auto echar = m_out.on_error().get_char();
            m_char = echar == m_char ? U'?' : echar;
            return get(dest_begin, dest_end);
        }
        else if(m_out.on_error().has_error_code())
        {
            m_out.set_error(m_out.on_error().get_error_code());
        }
        else if(m_out.on_error().has_function())
        {
            m_out.on_error().get_function()();
        }
        this->m_status = stringify::v0::cv_result::invalid_char;
        return dest_begin;
    }
    if(res.count == m_count)
    {
        BOOST_ASSERT(dest_begin < res.dest_it || m_count == 0);
        BOOST_ASSERT(res.dest_it <= dest_end);
        this->m_status = stringify::v0::cv_result::success;
    }
    else
    {
        m_count -= res.count;
        this->m_status = stringify::v0::cv_result::insufficient_space;
    }
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

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<wchar_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class str_source<char32_t, char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_source<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_source<wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_source<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_source<char32_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_source<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_source<wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_source<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class repeated_char32_source<char32_t>;

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

