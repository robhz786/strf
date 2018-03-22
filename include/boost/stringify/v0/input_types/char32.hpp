#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/arg_format.hpp>
#include <boost/stringify/v0/formatter.hpp>
#include <boost/stringify/v0/facets/encoder.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN


template <typename CharT>
class char_with_format
    : public stringify::v0::char_format<char_with_format<CharT> >
{
public:

    template <typename T>
    using fmt_tmpl = stringify::v0::char_format<T>;

    using fmt_type = fmt_tmpl<char_with_format>;

    constexpr char_with_format(CharT value)
        : m_value(value)
    {
    }

    constexpr char_with_format(CharT value, const fmt_type& fmt)
        : fmt_type(fmt)
        , m_value(value)
    {
    }

    constexpr char_with_format(const char_with_format&) = default;

    constexpr CharT value() const
    {
        return m_value;
    }

private:

    CharT m_value = 0;
};

template <typename CharT>
class char32_formatter: public formatter<CharT>
{
    using input_type = char32_t;
    using writer_type = stringify::v0::output_writer<CharT>;

public:

    template <typename FTuple>
    char32_formatter
        ( const FTuple& ft
        , const stringify::v0::char_with_format<char32_t>& input
        ) noexcept
        : char32_formatter(input, get_encoder(ft), get_width_calculator(ft))
    {
    }

    char32_formatter
        ( const stringify::v0::char_with_format<char32_t>& input
        , const stringify::v0::encoder<CharT>& encoder
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    virtual ~char32_formatter();

    std::size_t length() const override;

    void write(writer_type& out) const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::char_with_format<char32_t> m_fmt;
    const stringify::v0::encoder<CharT>& m_encoder;
    int m_fillcount = 0;

    template <typename FTuple>
    static const auto& get_encoder(const FTuple& ft)
    {
        using category = stringify::v0::encoder_category<CharT>;
        return ft.template get_facet<category, input_type>();
    }

    template <typename FTuple>
    static const auto& get_width_calculator(const FTuple& ft)
    {
        using category = stringify::v0::width_calculator_category;
        return ft.template get_facet<category, input_type>();
    }

    void determinate_fill_and_width(const stringify::v0::width_calculator& wcalc)
    {
        int content_width = 0;
        if(m_fmt.width() < 0)
        {
            m_fmt.width(0);
        }
        if (m_fmt.count() > 0)
        {
            content_width = m_fmt.count() * wcalc.width_of(m_fmt.value());
        }
        if (content_width >= m_fmt.width())
        {
            m_fillcount = 0;
            m_fmt.width(content_width);
        }
        else
        {
            m_fillcount = m_fmt.width() - content_width;
        }
    }
};


template <typename CharT>
char32_formatter<CharT>::char32_formatter
    ( const stringify::v0::char_with_format<char32_t>& input
    , const stringify::v0::encoder<CharT>& encoder
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_fmt(input)
    , m_encoder(encoder)
{
    determinate_fill_and_width(wcalc);
}


template <typename CharT>
char32_formatter<CharT>::~char32_formatter()
{
}


template <typename CharT>
std::size_t char32_formatter<CharT>::length() const
{
    std::size_t len = 0;
    if (m_fmt.count() > 0)
    {
        len = m_fmt.count() * m_encoder.length(m_fmt.value());
    }
    if (m_fillcount > 0)
    {
        len += m_fillcount * m_encoder.length(m_fmt.fill());
    }
    return len;
}


template <typename CharT>
void char32_formatter<CharT>::write(writer_type& out) const
{
    if (m_fillcount == 0)
    {
        m_encoder.encode(out, m_fmt.count(), m_fmt.value());
    }
    else
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                m_encoder.encode(out, m_fmt.count(), m_fmt.value());
                m_encoder.encode(out, m_fillcount, m_fmt.fill());
                break;
            }
            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                m_encoder.encode(out, halfcount, m_fmt.fill());
                m_encoder.encode(out, m_fmt.count(), m_fmt.value());
                m_encoder.encode(out, m_fillcount - halfcount, m_fmt.fill());
                break;
            }
            default:
            {
                m_encoder.encode(out, m_fillcount, m_fmt.fill());
                m_encoder.encode(out, m_fmt.count(), m_fmt.value());
            }
        }
    }
}


template <typename CharT>
int char32_formatter<CharT>::remaining_width(int w) const
{
    if (w > m_fmt.width())
    {
        return w - m_fmt.width();
    }
    return 0;
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

template
    < typename CharT
    , typename FTuple
    , typename = typename std::enable_if<!std::is_same<CharT, char32_t>::value>::type
    >
inline stringify::v0::char32_formatter<CharT>
boost_stringify_make_formatter(const FTuple& ft, char32_t ch)
{
    return {ft, ch};
}

template
    < typename CharT
    , typename FTuple
    , typename = typename std::enable_if<!std::is_same<CharT, char32_t>::value>::type
    >
inline stringify::v0::char32_formatter<CharT>
boost_stringify_make_formatter
    ( const FTuple& ft
    , const stringify::v0::char_with_format<char32_t>& ch
    )
{
    return {ft, ch};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED



