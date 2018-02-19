#ifndef BOOST_STRINGIFY_V0_FACETS_DECODER_HPP
#define BOOST_STRINGIFY_V0_FACETS_DECODER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/ftuple.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> struct decoder_category;

//[ u32output_class
class u32output
{
public:

    virtual ~u32output()
    {
    }

    virtual bool put(char32_t ch) = 0;

    virtual void set_error(std::error_code err) = 0;
};
//]


//[ decoder_class_template
template <typename CharT> class decoder
{
public:

    using category = stringify::v0::decoder_category<CharT>;

    virtual ~decoder() = default;

    virtual void decode
        ( stringify::v0::u32output& dest
        , const CharT* str
        , const CharT* end
        ) const = 0;
};
//]


typedef bool (*decoder_err_func)(stringify::v0::u32output&);

inline bool decoder_err_put_replacement_char(stringify::v0::u32output& rec)
{
    return rec.put(U'\uFFFD');
}


template <typename ErrHandlingFunc>
//requires requires(ErrHandlingFunc func, stringify::v0::u32output& out)
//{
//    {func(out) -> bool};
//};
class u8decoder: public stringify::v0::decoder<char>
{

    u8decoder
        ( ErrHandlingFunc err_func
        , bool tolerate_overlong
        , bool mutf8
        , bool wtf8
        )
        : m_err_func(err_func)
        , m_tolerate_overlong(tolerate_overlong)
        , m_mutf8(mutf8)
        , m_wtf8(wtf8)
    {
    }

public:

    u8decoder(const u8decoder& cp) // = default
        : m_err_func(cp.m_err_func)
        , m_tolerate_overlong(cp.m_tolerate_overlong)
        , m_mutf8(cp.m_mutf8)
        , m_wtf8(cp.m_wtf8)

    {
    }

    u8decoder(u8decoder&& mv) // = default
        : m_err_func(std::move(mv.m_err_func))
        , m_tolerate_overlong(mv.m_tolerate_overlong)
        , m_mutf8(mv.m_mutf8)
        , m_wtf8(mv.m_wtf8)
    {
    }


    u8decoder(ErrHandlingFunc err_func)
        : m_err_func(err_func)
    {
    }


    virtual ~u8decoder() = default;


    virtual void decode
        ( stringify::v0::u32output& destination
        , const char* str
        , const char* end
        ) const override
    {
        unsigned ch1, ch2, ch3, x;
        bool shall_continue = true;
        bool failed_previous = false;

        while(str != end && shall_continue)
        {
            unsigned ch0 = *str;
            ++str;
            if (0x80 > (x = ch0) ||

                (0xC0 == (ch0 & 0xE0)
                 ? (str != end && is_continuation(ch1 = *str) &&
                    (++str, is_valid_2_bytes(x = decode(ch0, ch1))))

                 : 0xE0 == (ch0 & 0xF0)
                 ? (str   != end && is_continuation(ch1 = *str) &&
                    ++str != end && is_continuation(ch2 = *str) &&
                    (++str, is_valid_3_bytes(x = decode(ch0, ch1, ch2))))

                 : (0xF0 == (ch0 & 0xF8) &&
                    str   != end && is_continuation(ch1 = *str) &&
                    ++str != end && is_continuation(ch2 = *str) &&
                    ++str != end && is_continuation(ch3 = *str) &&
                    (++str, is_valid_4_bytes(x = decode(ch0, ch1, ch2, ch3))))))
            {
                shall_continue = destination.put(x);
                failed_previous = false;
            }
            else if ( ! (failed_previous && is_continuation(ch0)))
            {
                shall_continue = m_err_func(destination);
                failed_previous = true;
            }
        }
    }


    u8decoder& tolerate_overlong(bool _ = true) &
    {
        m_tolerate_overlong = _;
        return *this;
    }
    u8decoder&& tolerate_overlong(bool _ = true) &&
    {
        m_tolerate_overlong = _;
        return std::move(*this);
    }
    u8decoder tolerate_overlong(bool _ = true) const &
    {
        return {m_err_func, _, m_mutf8, m_wtf8};
    }


    u8decoder& mutf8(bool _ = true) &
    {
        m_mutf8 = _;
        return *this;
    }
    u8decoder&& mutf8(bool _ = true) &&
    {
        m_mutf8 = _;
        return std::move(*this);
    }
    u8decoder mutf8(bool _ = true) const &
    {
        return {m_err_func, m_tolerate_overlong, _, m_wtf8};
    }


    u8decoder& wtf8(bool _ = true) &
    {
        m_wtf8 = _;
        return *this;
    }
    u8decoder&& wtf8(bool _ = true) &&
    {
        m_wtf8 = _;
        return std::move(*this);
    }
    u8decoder wtf8(bool _ = true) const &
    {
        return {m_err_func, m_tolerate_overlong, m_mutf8, _};
    }


private:

    static bool is_continuation(unsigned ch)
    {
        return (ch & 0xC0) == 0x80;
    }

    static unsigned decode(unsigned ch0, unsigned ch1)
    {
        return (((ch0 & 0x1F) << 6) |
                ((ch1 & 0x3F) << 0));
    }

    static unsigned decode(unsigned ch0, unsigned ch1, unsigned ch2)
    {
        return (((ch0 & 0x0F) << 12) |
                ((ch1 & 0x3F) <<  6) |
                ((ch2 & 0x3F) <<  0));
    }

    static
    unsigned decode(unsigned ch0, unsigned ch1, unsigned ch2, unsigned ch3)
    {
        return (((ch0 & 0x07) << 18) |
                ((ch1 & 0x3F) << 12) |
                ((ch2 & 0x3F) <<  6) |
                ((ch3 & 0x3F) <<  0));
    }

    bool is_valid_2_bytes(unsigned x) const
    {
        return (m_tolerate_overlong || 0x7F < x || (m_mutf8 && x == 0));
    }

    bool is_valid_3_bytes(unsigned x) const
    {
        return ((m_tolerate_overlong || 0x7ff < x) &&
                (m_wtf8 || 0xDFFF < x || x < 0xD800 ));
    }

    bool is_valid_4_bytes(unsigned x) const
    {
        return ( m_tolerate_overlong || 0xFFFF < x ) && x < 0x110000;
    }


    void err_func(stringify::v0::u32output& dest) const
    {
        dest.put(0xFFFE);
    }

    ErrHandlingFunc m_err_func;
    bool m_tolerate_overlong = false;
    bool m_mutf8 = false;
    bool m_wtf8 = false;
};

template <typename ErrHandlingFunc>
stringify::v0::u8decoder<ErrHandlingFunc>
make_u8decoder(ErrHandlingFunc err_func)
{
    return {err_func};
}

inline
stringify::v0::u8decoder<bool(*)(u32output&)>
make_u8decoder()
{
    return {stringify::v0::decoder_err_put_replacement_char};
}


template <typename CharT, typename ErrHandlingFunc>
class u16decoder: public stringify::v0::decoder<CharT>
{
public:

    u16decoder(ErrHandlingFunc err_func)
        : m_err_func(err_func)
    {
    }

    u16decoder(const u16decoder& cp) // = default
        : m_err_func(cp.m_err_func)
    {
    }

    u16decoder(u16decoder&& mv) // = default
        : m_err_func(std::move(mv.m_err_func))
    {
    }

    virtual ~u16decoder() = default;

    virtual void decode
        ( stringify::v0::u32output& dest
        , const CharT* str
        , const CharT* end
        ) const override
    {
        bool shall_continue = true;
        unsigned long ch, ch2;

        while(str != end && shall_continue)
        {
            ch = *str;
            ++str;
            if (! is_surrogate(ch))
            {
                shall_continue = dest.put(static_cast<char32_t>(ch));
            }
            else if (is_high_surrogate(ch)
                     && str != end
                     && is_low_surrogate(ch2 = *str))
            {
                ch = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
                shall_continue = dest.put(ch);
                ++str;
            }
            else
            {
                shall_continue = m_err_func(dest);
            }
        }
    }

private:

    static bool is_surrogate(unsigned long ch) noexcept
    {
        return ch >> 11 == 0x1B;
    }

    static bool is_high_surrogate(unsigned long ch) noexcept
    {
        return ch >> 10 == 0x36;
    }

    static bool is_low_surrogate(unsigned long ch) noexcept
    {
        return ch >> 10 == 0x37;
    }


    ErrHandlingFunc m_err_func;
    bool m_allow_alone_surrogates;
};


template <typename CharT>
class lax_u16decoder: public stringify::v0::decoder<CharT>
{
public:

    void decode
        ( stringify::v0::u32output& dest
        , const CharT* str
        , const CharT* end
        ) const override
    {
        bool shall_continue = true;
        unsigned long ch, ch2;

        while(str != end && shall_continue)
        {
            ch = *str;
            ++str;
            if (is_high_surrogate(ch)
                && str != end
                && is_low_surrogate(ch2 = *str))
            {
                ch = 0x10000 + (((ch & 0x3FF) << 10) | (ch2 & 0x3FF));
                ++str;
            }
            shall_continue = dest.put(ch);
        }
    }

private:

    static bool is_high_surrogate(unsigned long ch) noexcept
    {
        return ch >> 10 == 0x36;
    }

    static bool is_low_surrogate(unsigned long ch) noexcept
    {
        return ch >> 10 == 0x37;
    }

};

template <typename CharT, typename ErrHandlingFunc>
stringify::v0::u16decoder<CharT, ErrHandlingFunc>
make_u16decoder(ErrHandlingFunc err_func)
{
    return {err_func};
}

template <typename CharT>
stringify::v0::u16decoder<CharT, stringify::v0::decoder_err_func>
make_u16decoder()
{
    return {stringify::v0::decoder_err_put_replacement_char};
}

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)





namespace detail
{

inline auto make_lax_u16decoders(std::true_type)
{
    return stringify::v0::make_ftuple
        ( stringify::v0::lax_u16decoder<char16_t>{}
        , stringify::v0::lax_u16decoder<wchar_t>{}
        );
}

inline auto make_lax_u16decoders(std::false_type)
{
    return stringify::v0::lax_u16decoder<char16_t>{};
}


template <typename ErrHandlingFunc>
auto make_u16decoders(ErrHandlingFunc err_func, std::true_type)
{
    return stringify::v0::make_ftuple
        ( stringify::v0::u16decoder<char16_t, ErrHandlingFunc>{err_func}
        , stringify::v0::u16decoder<wchar_t, ErrHandlingFunc>{err_func}
        );
}


template <typename ErrHandlingFunc>
auto make_u16decoders(ErrHandlingFunc err_func, std::false_type)
{
    return stringify::v0::u16decoder<char16_t, ErrHandlingFunc>{err_func};
}


inline auto make_u16decoders(std::true_type)
{
    auto func = stringify::v0::decoder_err_put_replacement_char;
    return stringify::v0::make_ftuple
        ( stringify::v0::u16decoder<char16_t, stringify::v0::decoder_err_func>{func}
        , stringify::v0::u16decoder<wchar_t, stringify::v0::decoder_err_func>{func}
        );
}

inline auto make_u16decoders(std::false_type)
{
    return stringify::v0::u16decoder<char16_t, stringify::v0::decoder_err_func>
    {stringify::v0::decoder_err_put_replacement_char};
}

} // namespace detail

inline auto make_lax_u16decoders()
{
    return stringify::v0::detail::make_lax_u16decoders
        (stringify::v0::detail::wchar_is_16);
}

inline auto make_u16decoders()
{
    return stringify::v0::detail::make_u16decoders
        (stringify::v0::detail::wchar_is_16);
}

template <typename ErrHandlingFunc>
auto make_u16decoders(ErrHandlingFunc err_func)
{
    return stringify::v0::detail::make_u16decoders
        (err_func, stringify::v0::detail::wchar_is_16);
}

#else  // if defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

inline auto make_lax_u16decoders()
{
    return stringify::v0::lax_u16dcoder<char16_t>{};
}

inline auto make_u16decoders()
{
    return stringify::v0::detail::make_u16decoder<char16_t>();
}

template <typename ErrHandlingFunc>
auto make_u16decoders(ErrHandlingFunc err_func)
{
    return stringify::v0::detail::make_u16decoder<char16_t>(err_func);
}


#endif // defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)


template <typename CharT>
class u32decoder: public stringify::v0::decoder<CharT>
{
public:

    virtual ~u32decoder() = default;

    virtual void decode
        ( stringify::v0::u32output& dest
        , const CharT* str
        , const CharT* end
        ) const override
    {
        for(; str != end; ++str)
        {
            dest.put(static_cast<char32_t>(*str));
        }
    }
};

template <> struct decoder_category<char>
{
    static const auto& get_default() noexcept
    {
        using decoder_type =
            stringify::v0::u8decoder<stringify::v0::decoder_err_func>;

        const static decoder_type x
            { stringify::v0::decoder_err_put_replacement_char };

        return x;
    }
};

template <> struct decoder_category<char16_t>
{
    static const auto& get_default() noexcept
    {
        using decoder_type
            = stringify::v0::u16decoder
                <char16_t, stringify::v0::decoder_err_func>;

        const static decoder_type x
            { stringify::v0::decoder_err_put_replacement_char };

        return x;
    }
};

template <> struct decoder_category<char32_t>
{
    static const stringify::v0::u32decoder<char32_t>&
    get_default() noexcept
    {
        const static stringify::v0::u32decoder<char32_t> x{};
        return x;
    }
};

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

template <> struct decoder_category<wchar_t>
{

    static const auto& get_default() noexcept
    {
        return get_default(stringify::v0::detail::wchar_is_32);
    }

private:

    using wstr_decoder_u16 =
        stringify::v0::u16decoder<wchar_t, stringify::v0::decoder_err_func>;
    using wstr_decoder_u32 =
        stringify::v0::u32decoder<wchar_t>;

    static const wstr_decoder_u32& get_default(std::true_type) noexcept
    {
        const static wstr_decoder_u32 x{};
        return x;
    }

    static const wstr_decoder_u16& get_default(std::false_type) noexcept
    {
        const static wstr_decoder_u16 x
            { stringify::v0::decoder_err_put_replacement_char };
        return x;
    }

};

#endif // ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u8decoder <stringify::v0::decoder_err_func>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u16decoder<char16_t, stringify::v0::decoder_err_func>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class u32decoder<char32_t>;

#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class u32decoder<wchar_t>;

#endif // ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

#endif //  defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_DECODER_HPP

