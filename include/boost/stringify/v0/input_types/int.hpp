#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/arg_format.hpp>
#include <boost/stringify/v0/facets/encoder.hpp>
#include <boost/stringify/v0/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/number_of_digits.hpp>
#include <boost/stringify/v0/formatter.hpp>
#include <boost/assert.hpp>
#include <cstdint>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template
    < typename IntT
    , typename unsigned_IntT = typename std::make_unsigned<IntT>::type
    >
typename std::enable_if<std::is_signed<IntT>::value, unsigned_IntT>::type
unsigned_abs(IntT value)
{
    return ( value > 0
           ? static_cast<unsigned_IntT>(value)
           : 1 + static_cast<unsigned_IntT>(-(value + 1)));
}

template<typename IntT>
typename std::enable_if<std::is_unsigned<IntT>::value, IntT>::type
unsigned_abs(IntT value)
{
    return value;
}

} // namespace detail

template <typename IntT>
class int_with_format: public int_format<int_with_format<IntT> >
{

public:

    template <typename T>
    using fmt_tmpl = stringify::v0::int_format<T>;

    using fmt_type = fmt_tmpl<int_with_format>;

    int_with_format() = default;

    int_with_format(IntT value)
        : m_value(value)
    {
    }

    int_with_format(IntT value, const fmt_type& fmt)
        : fmt_type(fmt)
        , m_value(value)
    {
    }

    int_with_format(const fmt_type& fmt)
        : fmt_type(fmt)
    {
    }

    int_with_format(const int_with_format&) = default;

    constexpr IntT value() const
    {
        return m_value;
    }

    constexpr void value(IntT v)
    {
        m_value = v;
    }

private:

    IntT m_value = 0;
};


template <typename IntT, typename CharT>
class int_formatter: public formatter<CharT>
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    static constexpr bool is_signed = std::is_signed<IntT>::value;

public:

    using input_type  = IntT ;
    using char_type   = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;

    template <typename FTuple>
    int_formatter
        ( const FTuple& ft
        , const stringify::v0::int_with_format<IntT>& value
        ) noexcept
        : int_formatter
            ( value
            , get_encoder(ft)
            , get_numpunct<8>(ft)
            , get_numpunct<10>(ft)
            , get_numpunct<16>(ft)
            )
    {
    }


    int_formatter
        ( const stringify::v0::int_with_format<IntT>& value
        , const stringify::v0::encoder<CharT>& encoder
        , const stringify::v0::numpunct<8>& numpunct_oct
        , const stringify::v0::numpunct<10>& numpunct_dec
        , const stringify::v0::numpunct<16>& numpunct_hex
        ) noexcept;

    virtual ~int_formatter();

    std::size_t length() const override;

    void write(writer_type& out) const override;

    int remaining_width(int w) const override;

private:

    boost::stringify::v0::int_with_format<IntT> m_input;
    const stringify::v0::encoder<CharT>& m_encoder;
    const stringify::v0::numpunct_base& m_numpunct;
    int m_fillcount = 0;

    template <typename FTuple>
    const stringify::v0::encoder<CharT>& get_encoder(const FTuple& ft) noexcept
    {
        using tag = stringify::v0::encoder_category<CharT>;
        return ft.template get_facet<tag, input_type>();
    }
    template <int Base, typename FTuple>
    const stringify::v0::numpunct<Base>& get_numpunct(const FTuple& ft) noexcept
    {
        using tag = stringify::v0::numpunct_category<Base>;
        return ft.template get_facet<tag, input_type>();
    }

    bool showsign() const
    {
        return is_signed && (m_input.showpos() || m_input.value() < 0);
    }

    unsigned_type unsigned_value() const noexcept
    {
        if(m_input.base() == 10)
        {
            return stringify::v0::detail::unsigned_abs(m_input.value());
        }
        return static_cast<unsigned_type>(m_input.value());
    }

    std::size_t length_fill() const
    {
        if (m_fillcount > 0)
        {
            return m_fillcount * m_encoder.length(m_input.fill());
        }
        return 0;
    }

    std::size_t length_body() const
    {
        switch(m_input.base())
        {
            case  10:
                return length_digits<10>() + (showsign() ? 1 : 0);

            case 16:
                return length_digits<16>() + (m_input.showbase() ? 2 : 0);

            default:
                BOOST_ASSERT(m_input.base() == 8);
                return length_digits<8>() + (m_input.showbase() ? 1 : 0);
        }
    }

    template <unsigned Base>
    std::size_t length_digits() const noexcept
    {
        unsigned num_digits
            = stringify::v0::detail::number_of_digits<Base>(unsigned_value());

        if (unsigned num_seps = m_numpunct.thousands_sep_count(num_digits))
        {
            auto sep_len = m_encoder.length(m_numpunct.thousands_sep());
            return num_digits + num_seps * sep_len;
        }
        return num_digits;
    }

    void write_with_fill(writer_type& out) const
    {
        switch(m_input.alignment())
        {
            case stringify::v0::alignment::left:
                write_sign(out);
                write_base(out);
                write_digits(out);
                write_fill(out, m_fillcount);
                break;

            case stringify::v0::alignment::internal:
                write_sign(out);
                write_base(out);
                write_fill(out, m_fillcount);
                write_digits(out);
                break;

            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                write_fill(out, halfcount);
                write_sign(out);
                write_base(out);
                write_digits(out);
                write_fill(out, m_fillcount - halfcount);
                break;
            }
            default:
                write_fill(out, m_fillcount);
                write_sign(out);
                write_base(out);
                write_digits(out);
        }
    }

    void write_sign(writer_type& out) const
    {
        if (is_signed && m_input.base() == 10)
        {
            if (m_input.value() < 0)
            {
                out.put(CharT('-'));
            }
            else if(showsign())
            {
                out.put(CharT('+'));
            }
        }
    }

    void write_without_fill(writer_type& out) const
    {
        switch(m_input.base())
        {
            case 16:
                write_base(out);
                write_digits_t<16>(out);
                break;

            case  8:
                write_base(out);
                write_digits_t<8>(out);
                break;

            default:
                BOOST_ASSERT(m_input.base() == 10);
                write_sign(out);
                write_digits_t<10>(out);
        }
    }

    void write_base(writer_type& out) const
    {
        if(m_input.base() != 10 && m_input.showbase())
        {
            out.put(CharT('0'));
            if(m_input.base() == 16)
            {
                out.put(m_input.uppercase() ? CharT('X'): CharT('x'));
            }
        }
    }

    void write_digits(writer_type& out) const
    {
        switch (m_input.base())
        {
            case 10:
                write_digits_t<10>(out);
                break;

            case 16:
                write_digits_t<16>(out);
                break;

            default:
                BOOST_ASSERT(m_input.base() == 8);
                write_digits_t<8>(out);
        }
    }

    template <unsigned Base>
    void write_digits_t(writer_type& out) const
    {
        constexpr std::size_t buff_size = sizeof(IntT) * 6;
        CharT buff[buff_size];
        CharT* end = &buff[buff_size - 1];
        auto* begin = write_digits_backwards<Base>(unsigned_value(), end);
        unsigned num_digits = static_cast<unsigned>(end - begin);
        if (m_numpunct.thousands_sep_count(num_digits) == 0)
        {
            out.put(begin, num_digits);
        }
        else
        {
            write_digits_with_punctuation(out, begin, num_digits);
        }
    }

    void write_digits_with_punctuation
        ( writer_type& out
        , const CharT* digits
        , unsigned num_digits
        ) const
    {
        char32_t thousands_sep = m_numpunct.thousands_sep();
        unsigned char groups[sizeof(IntT) * 6];
        auto* it = m_numpunct.groups(num_digits, groups);
        out.put(digits, *it);
        digits += *it;
        while(--it >= groups)
        {
            m_encoder.encode(out, 1, thousands_sep);
            out.put(digits, *it);
            digits += *it;
        }
    }

    template <unsigned Base, typename OutputIterator>
    OutputIterator*
    write_digits_backwards(unsigned_type value, OutputIterator* it) const
    {
        while(value >= Base)
        {
            *--it = character_of_digit(value % Base);
            value /= Base;
        }
        *--it = character_of_digit(static_cast<unsigned>(value));
        return it;
    }

    CharT character_of_digit(unsigned digit) const noexcept
    {
        if (digit < 10)
        {
            return CharT('0') + digit;
        }
        const CharT char_a = m_input.uppercase() ? 'A' : 'a';
        return  char_a + digit - 10;
    }

    void determinate_fill()
    {
        int content_width = width_body();
        if(m_input.width() < 0)
        {
            m_input.width(0);
        }
        if(content_width < m_input.width())
        {
            m_fillcount = m_input.width() - content_width;
        }
        else
        {
            m_input.width(content_width);
        }
    }

    void write_fill(writer_type& out, int count) const
    {
        m_encoder.encode(out, count, m_input.fill());
    }

    int width_body() const noexcept
    {
        unsigned num_digits = 0;
        unsigned extra_chars = 0;
        auto uv = unsigned_value();
        switch(m_input.base())
        {
            case 10:
            {
                num_digits = stringify::v0::detail::number_of_digits<10>(uv);
                extra_chars = showsign() ? 1 : 0;
                break;
            }
            case 16:
            {
                num_digits = stringify::v0::detail::number_of_digits<16>(uv);
                extra_chars = m_input.showbase() ? 2 : 0;
                break;
            }
            default:
            {
                BOOST_ASSERT(m_input.base() == 8);
                num_digits = stringify::v0::detail::number_of_digits<8>(uv);
                extra_chars = m_input.showbase() ? 1 : 0;
            }
        }

        unsigned num_separators = m_numpunct.thousands_sep_count(num_digits);
        return num_digits + extra_chars + num_separators;
    }
};

template <typename IntT, typename CharT>
int_formatter<IntT, CharT>::int_formatter
    ( const stringify::v0::int_with_format<IntT>& valuef
    , const stringify::v0::encoder<CharT>& encoder
    , const stringify::v0::numpunct<8>& numpunct_oct
    , const stringify::v0::numpunct<10>& numpunct_dec
    , const stringify::v0::numpunct<16>& numpunct_hex
    ) noexcept
    : m_input{valuef}
    , m_encoder{encoder}
    , m_numpunct
        ( valuef.base() == 10
          ? static_cast<const stringify::v0::numpunct_base&>(numpunct_dec)
          : valuef.base() == 16
          ? static_cast<const stringify::v0::numpunct_base&>(numpunct_hex)
          : static_cast<const stringify::v0::numpunct_base&>(numpunct_oct)
        )
{
    determinate_fill();
}

template <typename IntT, typename CharT>
int_formatter<IntT, CharT>::~int_formatter()
{
}

template <typename IntT, typename CharT>
std::size_t int_formatter<IntT, CharT>::length() const
{
    return length_body() + length_fill();
}

template <typename IntT, typename CharT>
void int_formatter<IntT, CharT>::write(writer_type& out) const
{
    if (m_fillcount > 0)
    {
        write_with_fill(out);
    }
    else
    {
        write_without_fill(out);
    }
}

template <typename IntT, typename CharT>
int int_formatter<IntT, CharT>::remaining_width(int w) const
{
    return w > m_input.width() ? (w - m_input.width()) : 0;
}


template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<short, CharT>
boost_stringify_make_formatter(const FTuple& ft, short x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<int, CharT>
boost_stringify_make_formatter(const FTuple& ft, int x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<long, CharT>
boost_stringify_make_formatter(const FTuple& ft, long x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<long long, CharT>
boost_stringify_make_formatter(const FTuple& ft, long long x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<unsigned short, CharT>
boost_stringify_make_formatter(const FTuple& ft, unsigned short x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<unsigned int, CharT>
boost_stringify_make_formatter(const FTuple& ft, unsigned int x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<unsigned long, CharT>
boost_stringify_make_formatter(const FTuple& ft, unsigned long x)
{
    return {ft, x};
}
template <typename CharT, typename FTuple>
inline stringify::v0::int_formatter<unsigned long long, CharT>
boost_stringify_make_formatter(const FTuple& ft, unsigned long long x)
{
    return {ft, x};
}


template <typename CharT, typename FTuple, typename IntT>
inline stringify::v0::int_formatter<IntT, CharT>
boost_stringify_make_formatter
    ( const FTuple& ft
    , const stringify::v0::int_with_format<IntT>& x
    )
{
    return {ft, x};
}

inline stringify::v0::int_with_format<short>
boost_stringify_fmt(short x)
{
    return {x};
}
inline stringify::v0::int_with_format<int>
boost_stringify_fmt(int x)
{
    return {x};
}
inline stringify::v0::int_with_format<long>
boost_stringify_fmt(long x)
{
    return {x};
}
inline stringify::v0::int_with_format<long long>
boost_stringify_fmt(long long x)
{
    return {x};
}
inline stringify::v0::int_with_format<unsigned short>
boost_stringify_fmt(unsigned short x)
{
    return {x};
}
inline stringify::v0::int_with_format<unsigned>
boost_stringify_fmt(unsigned x)
{
    return {x};
}
inline stringify::v0::int_with_format<unsigned long>
boost_stringify_fmt(unsigned long x)
{
    return {x};
}
inline stringify::v0::int_with_format<unsigned long long>
boost_stringify_fmt(unsigned long long x)
{
    return {x};
}



#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
