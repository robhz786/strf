#ifndef BOOST_STRINGIFY_V0_FACETS_ENCODER_HPP
#define BOOST_STRINGIFY_V0_FACETS_ENCODER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/output_writer.hpp>
#include <boost/stringify/v0/ftuple.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> struct encoder_tag;

template <typename CharT> class encoder
{
public:

    virtual ~encoder()
    {
    };

    virtual std::size_t length(char32_t ch) const = 0;

    virtual bool encode
        ( stringify::v0::output_writer<CharT>& destination
        , std::size_t count
        , char32_t ch
        ) const = 0;

};

inline bool put_utf8_replacement_char
    ( stringify::v0::output_writer<char>& ow
    , std::size_t count
    )
{
    return ow.repeat(count, '\xEF', '\xBF', '\xBD');
}

template <typename CharT>
inline bool put_utf16_replacement_char
    ( stringify::v0::output_writer<CharT>& ow
    , std::size_t count
    )
{
    return ow.repeat(count, static_cast<CharT>(u'\uFFFD'));
}

template <typename CharT>
inline bool from_utf32_throw
    ( stringify::v0::output_writer<CharT>&
    , std::size_t
    )
{
    throw std::system_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}

template <typename CharT>
inline bool from_utf32_set_error_code
    ( stringify::v0::output_writer<CharT>& ow
    , std::size_t
    )
{
    ow.set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}



template <typename CharT>
using from_utf32_err_func
= bool(*)(stringify::v0::output_writer<CharT>&, std::size_t);


template <typename ErrHandlingFunc = from_utf32_err_func<char>>
class u8encoder: public stringify::v0::encoder<char>
{
public:

    using category = stringify::v0::encoder_tag<char>;

    u8encoder
        ( ErrHandlingFunc err_func
        , bool wtf8 = false  // tolerate surrogates
        )
        : m_err_func(err_func)
        , m_wtf8(wtf8)
    {
    }

    u8encoder(const u8encoder& other)
        : m_err_func(other.m_err_func)
        , m_wtf8(other.m_wtf8)
    {
    }

    u8encoder(u8encoder&& other)
        : m_err_func(std::move(other.m_err_func))
        , m_wtf8(other.m_wtf8)
    {
    }


    ~u8encoder() = default;

    std::size_t length(char32_t ch) const noexcept override
    {
        return (ch <     0x80 ? 1 :
                ch <    0x800 ? 2 :
                ! m_wtf8 && is_surrogate(ch) ? 4 :
                ch <  0x10000 ? 3 :
                4); // 0xFFFD
    }

    u8encoder & wtf8(bool b = true) &
    {
        m_wtf8 = b;
        return *this;
    }

    u8encoder && wtf8(bool b = true) &&
    {
        m_wtf8 = b;
        return std::move(*this);
    }

    u8encoder wtf8(bool b = true) const &
    {
        return {m_err_func, b};
    }

    u8encoder & tolerate_surrogates(bool b = true) &
    {
        m_wtf8 = b;
        return *this;
    }

    u8encoder && tolerate_surrogates(bool b = true) &&
    {
        m_wtf8 = b;
        return std::move(*this);
    }

    u8encoder tolerate_surrogates(bool b = true) const &
    {
        return {m_err_func, b};
    }

    bool encode
        ( stringify::v0::output_writer<char>& destination
        , std::size_t count
        , char32_t ch
        ) const override
    {
        if (ch < 0x80)
        {
            if (count == 1)
            {
                return destination.put(static_cast<char>(ch));
            }
            return destination.repeat(count, static_cast<char>(ch));
        }
        if (ch < 0x800)
        {
            return destination.repeat
                ( count
                , static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6))
                , static_cast<char>(0x80 |  (ch &  0x3F))
                );
        }
        if (ch <  0x10000)
        {
            if (is_surrogate(ch) && ! m_wtf8)
            {
                return m_err_func(destination, count);
            }
            return destination.repeat
                ( count
                , static_cast<char>(0xE0 | ((ch & 0xF000) >> 12))
                , static_cast<char>(0x80 | ((ch &  0xFC0) >> 6))
                , static_cast<char>(0x80 |  (ch &   0x3F))
                );
        }
        if (ch < 0x110000)
        {
            return destination.repeat
                ( count
                , static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18))
                , static_cast<char>(0x80 | ((ch &  0x3F000) >> 12))
                , static_cast<char>(0x80 | ((ch &    0xFC0) >> 6))
                , static_cast<char>(0x80 |  (ch &     0x3F))
                );
        }
        return m_err_func(destination, count);
    }

private:

    static bool is_surrogate(unsigned long ch) noexcept
    {
        return ch >> 11 == 0x1B;
    }

    ErrHandlingFunc m_err_func;
    bool m_wtf8;
};


inline u8encoder<from_utf32_err_func<char>> make_u8encoder(bool wtf8 = false)
{
    return {put_utf8_replacement_char, wtf8};
}

template <typename F>
u8encoder<F> make_u8encoder(F err_func,  bool wtf8 = false)
{
    return {err_func, wtf8};
}

template <typename CharT, typename ErrHandlingFunc = from_utf32_err_func<CharT>>
class u16encoder: public stringify::v0::encoder<CharT>
{
public:

    using category = stringify::v0::encoder_tag<CharT>;

    u16encoder(ErrHandlingFunc err_func, bool tolerate_surrogates = false)
        : m_err_func(err_func)
        , m_tolerate_surr(tolerate_surrogates)
    {
    }

    u16encoder(const u16encoder& cp)
        : m_err_func(cp.m_err_func)
        , m_tolerate_surr(cp.m_tolerate_surr)
    {
    }

    u16encoder(u16encoder&& mv)
        : m_err_func(std::move(mv.m_err_func))
        , m_tolerate_surr(mv.m_tolerate_surr)
    {
    }

    ~u16encoder() = default;

    std::size_t length(char32_t ch) const noexcept override
    {
        return single_char_range(ch) ? 1 : 2;
    }

    bool encode
        ( stringify::v0::output_writer<CharT>& destination
        , std::size_t count
        , char32_t ch
        ) const override
    {
        if (single_char_range(ch))
        {
            if (count == 1)
            {
                return destination.put(static_cast<CharT>(ch));
            }
            else
            {
                return destination.repeat(count, static_cast<CharT>(ch));
            }
        }
        else if (two_chars_range(ch))
        {
            char32_t sub_codepoint = ch - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            return destination.repeat
                ( count
                , static_cast<CharT>(high_surrogate)
                , static_cast<CharT>(low_surrogate)
                );
        }
        else
        {
           return m_err_func(destination, count);
        }
    }

    u16encoder & tolerate_surrogates(bool b = true) &
    {
        m_tolerate_surr = b;
        return *this;
    }

    u16encoder && tolerate_surrogates(bool b = true) &&
    {
        m_tolerate_surr = b;
        return std::move(*this);
    }

    u16encoder tolerate_surrogates(bool b = true) const &
    {
        return {m_err_func, b};
    }

private:

    bool single_char_range(char32_t ch) const noexcept
    {
        return ch < 0x10000 && (ch < 0xd800 || ch > 0xdfff || m_tolerate_surr);
    }

    bool two_chars_range(char32_t ch) const noexcept
    {
        return 0xffff < ch && ch < 0x110000;
    }

    ErrHandlingFunc m_err_func;
    bool m_tolerate_surr;
};


template <typename CharT>
inline u16encoder<CharT, from_utf32_err_func<CharT>> make_u16encoder
    ( bool tolerate_surrogates = false )
{
    return {put_utf16_replacement_char<char16_t>, tolerate_surrogates};
}

template <typename CharT, typename F>
u16encoder<CharT, F> make_u16encoder
    (F err_func, bool tolerate_surrogates = false)
{
    return {err_func, tolerate_surrogates};
}

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

namespace detail
{

inline auto make_u16encoders(bool tolerate_surrogates, std::true_type)
{
    return stringify::v0::make_ftuple
        ( stringify::v0::u16encoder<char16_t>
              { put_utf16_replacement_char<char16_t>
              , tolerate_surrogates }
        , stringify::v0::u16encoder<wchar_t>
              { put_utf16_replacement_char<wchar_t>
              , tolerate_surrogates }
        );
}

inline auto make_u16encoders(bool tolerate_surrogates, std::false_type)
{
    return stringify::v0::u16encoder<char16_t>
              { put_utf16_replacement_char<char16_t>
              , tolerate_surrogates };
}

} // namespace detail

inline auto make_u16encoders(bool tolerate_surrogates = false)
{
    return stringify::v0::detail::make_u16encoders
        ( tolerate_surrogates
        , stringify::v0::detail::wchar_is_16
        );
}

#else // defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

inline auto make_u16encoders(bool tolerate_surrogates = false)
{
    return stringify::v0::u16encoder<char16_t>
              { put_utf16_replacement_char<char16_t>
              , tolerate_surrogates };
}

#endif // defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

template <typename CharT>
class utf32_to_utf32: public stringify::v0::encoder<CharT>
{
public:

    using category = stringify::v0::encoder_tag<CharT>;

    utf32_to_utf32()
    {
    }

    utf32_to_utf32(const utf32_to_utf32&)
    {
    }

    ~utf32_to_utf32() = default;

    std::size_t length(char32_t) const noexcept override
    {
        return 1;
    }

    bool encode
        ( stringify::v0::output_writer<CharT>& destination
        , std::size_t count
        , char32_t ch
        ) const override
    {
        return destination.repeat(count, static_cast<CharT>(ch));
    }
};


class default_utf32_to_wstr : public stringify::v0::encoder<wchar_t>
{
public:

    using category = stringify::v0::encoder_tag<wchar_t>;

    std::size_t length(char32_t) const noexcept override;

    bool encode
        ( stringify::v0::output_writer<wchar_t>& destination
        , std::size_t count
        , char32_t ch
        ) const override;
};


template <> struct encoder_tag<char>
{
    static auto get_default() noexcept
    -> const stringify::v0::u8encoder<stringify::v0::from_utf32_err_func<char>> &;
};

template <> struct encoder_tag<char16_t>
{
    static auto get_default() noexcept
        -> const stringify::v0::u16encoder
            <char16_t, stringify::v0::from_utf32_err_func<char16_t>>&;
};

template <> struct encoder_tag<char32_t>
{
    static const stringify::v0::utf32_to_utf32<char32_t>& get_default() noexcept;
};

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

template <> struct encoder_tag<wchar_t>
{

    static decltype(auto) get_default() noexcept
    {
        constexpr bool wchar_is_32 = sizeof(wchar_t) == sizeof(char32_t);
        return get_default(std::integral_constant<bool, wchar_is_32>{});
    }

private:

    using utf32_impl = stringify::v0::utf32_to_utf32<wchar_t>;

    using utf16_impl = stringify::v0::u16encoder
              < wchar_t
              , stringify::v0::from_utf32_err_func<wchar_t>
              >;

    static const utf32_impl& get_default(std::true_type) noexcept;

    static const utf16_impl& get_default(std::false_type) noexcept;

};

#endif // ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)



#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u8encoder<stringify::v0::from_utf32_err_func<char>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u16encoder<char16_t, stringify::v0::from_utf32_err_func<char16_t>>;


#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class utf32_to_utf32<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u16encoder<wchar_t, stringify::v0::from_utf32_err_func<wchar_t>>;

#endif // ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)


#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)



#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE
const stringify::v0::u8encoder<stringify::v0::from_utf32_err_func<char>>&
encoder_tag<char>::get_default() noexcept
{
    const static stringify::v0::u8encoder
        <stringify::v0::from_utf32_err_func<char>>
        x{put_utf8_replacement_char};

    return x;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::u16encoder
    <char16_t, stringify::v0::from_utf32_err_func<char16_t>>&
encoder_tag<char16_t>::get_default() noexcept
{
    const static stringify::v0::u16encoder
        <char16_t, stringify::v0::from_utf32_err_func<char16_t>>
        x{put_utf16_replacement_char<char16_t>};

    return x;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::utf32_to_utf32<char32_t>&
encoder_tag<char32_t>::get_default() noexcept
{
    const static stringify::v0::utf32_to_utf32<char32_t> x;

    return x;
}

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

BOOST_STRINGIFY_INLINE
const stringify::v0::encoder_tag<wchar_t>::utf16_impl&
encoder_tag<wchar_t>::get_default(std::false_type) noexcept
{
    const static utf16_impl x{put_utf16_replacement_char<wchar_t>};

    return x;
}

BOOST_STRINGIFY_INLINE
const stringify::v0::encoder_tag<wchar_t>::utf32_impl&
encoder_tag<wchar_t>::get_default(std::true_type) noexcept
{
    const static utf32_impl x;

    return x;
}

#endif // ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_ENCODER_HPP

