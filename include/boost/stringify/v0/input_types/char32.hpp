#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/formatter.hpp>
#include <boost/stringify/v0/char_flags.hpp>
#include <boost/stringify/v0/facets/encoder.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

using char_fmt
    = stringify::v0::basic_arg_fmt_with_count<int, 0, int, 1, '<', '^'>;

namespace detail {

template <typename CharT>
class char32_formatter: public formatter<CharT>
{
    using input_type = char32_t;
    using writer_type = stringify::v0::output_writer<CharT>;

public:

    using second_arg = stringify::v0::char_fmt;

    template <typename FTuple>
    char32_formatter
        ( const FTuple& ft
        , char32_t ch
        , const second_arg& fmt = {}
        ) noexcept
        : char32_formatter(get_encoder(ft), get_width_calculator(ft), ch, fmt)
    {
    }

    char32_formatter
        ( const stringify::v0::encoder<CharT>& encoder
        , const stringify::v0::width_calculator& wcalc
        , char32_t ch
        , const second_arg& fmt = {}
        ) noexcept
        : m_encoder(encoder)
        , m_char(ch)
        , m_fmt(fmt)
    {
        determinate_fill_and_width(wcalc);
    }

    std::size_t length() const override
    {
        std::size_t len = 0;
        if (m_fmt.count() > 0)
        {
            len = m_fmt.count() * m_encoder.length(m_char);
        }
        if (m_fillcount > 0)
        {
            len += m_fillcount * m_encoder.length(m_fmt.fill());
        }
        return len;
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount == 0)
        {
            m_encoder.encode(out, m_fmt.count(), m_char);
        }
        else if(m_fmt.has_char<'<'>())
        {
            m_encoder.encode(out, m_fmt.count(), m_char);
            m_encoder.encode(out, m_fillcount, m_fmt.fill());
        }
        else if(m_fmt.has_char<'^'>())
        {
            auto halfcount = m_fillcount / 2;
            m_encoder.encode(out, halfcount, m_fmt.fill());
            m_encoder.encode(out, m_fmt.count(), m_char);
            m_encoder.encode(out, m_fillcount - halfcount, m_fmt.fill());
        }
        else
        {
            m_encoder.encode(out, m_fillcount, m_fmt.fill());
            m_encoder.encode(out, m_fmt.count(), m_char);
        }
    }

    int remaining_width(int w) const override
    {
        if (w > 0 && static_cast<char_fmt::width_type>(w) > m_fmt.width())
        {
            return w - static_cast<int>(m_fmt.width());
        }
        return 0;
    }

private:

    const stringify::v0::encoder<CharT>& m_encoder;
    const char32_t m_char;
    stringify::v0::char_fmt m_fmt;
    char_fmt::width_type m_fillcount = 0;

    template <typename FTuple>
    static const auto& get_encoder(const FTuple& ft)
    {
        using category = stringify::v0::encoder_tag<CharT>;
        return ft.template get_facet<category, input_type>();
    }

    template <typename FTuple>
    static const auto& get_width_calculator(const FTuple& ft)
    {
        using category = stringify::v0::width_calculator_tag;
        return ft.template get_facet<category, input_type>();
    }

    void determinate_fill_and_width(const stringify::v0::width_calculator& wcalc)
    {
        char_fmt::width_type content_width = 0;
        if (m_fmt.count() > 0)
        {
            content_width = m_fmt.count() * wcalc.width_of(m_char);
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

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_formatter<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


struct char32_input_traits
{
    template <typename CharT, typename FTuple>
    using formatter
    = stringify::v0::detail::char32_formatter<CharT>;
};

} // namespace detail

stringify::v0::detail::char32_input_traits
boost_stringify_input_traits_of(char32_t);


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED



