#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/stringifier.hpp>
#include <boost/stringify/v0/char_flags.hpp>
#include <boost/stringify/v0/facets/alignment.hpp>
#include <boost/stringify/v0/facets/conversion_from_utf32.hpp>
#include <boost/stringify/v0/facets/fill.hpp>
#include <boost/stringify/v0/facets/width.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct char_argf
{
    using char_flags_type = stringify::v0::char_flags<'<', '>', '='>;

    constexpr char_argf(int w)
        : width(w)
    {
    }
    constexpr char_argf(const char* f, std::size_t c=1)
        : count(c), flags(f)
    {
    }
    constexpr char_argf(int w, const char* f, std::size_t c=1)
        : count(c), width(w), flags(f)
    {
    }

    constexpr char_argf(const char_argf&) = default;

    std::size_t count = 1;
    int width = -1;
    char_flags_type flags;
};

namespace detail {

template <typename CharT>
class char32_stringifier: public stringifier<CharT>
{
    using input_type = char32_t;
    using writer_type = stringify::v0::output_writer<CharT>;
    using from32_tag = stringify::v0::conversion_from_utf32_tag<CharT>;
    using to32_tag = stringify::v0::conversion_to_utf32_tag<CharT>;
    using wcalc_tag = stringify::v0::width_calculator_tag;
    using argf_reader = stringify::v0::conventional_argf_reader<input_type>;

public:

    using second_arg = stringify::v0::char_argf;

    template <typename FTuple>
    char32_stringifier
        ( const FTuple& ft
        , char32_t ch
        ) noexcept
        : m_from32cv(get_facet<from32_tag>(ft))
        , m_width(get_facet<width_tag>(ft).width())
        , m_char(ch)
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_alignment(get_facet<alignment_tag>(ft).value())
    {
        determinate_fill_and_width(get_facet<wcalc_tag>(ft));
    }

    template <typename FTuple>
    char32_stringifier
        ( const FTuple& ft
        , char32_t ch
        , const second_arg& argf
        ) noexcept
        : m_from32cv(get_facet<from32_tag>(ft))
        , m_count(argf.count)
        , m_width(argf_reader::get_width(argf, ft))
        , m_char(ch)
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_alignment(argf_reader::get_alignment(argf, ft))
    {
        determinate_fill_and_width(get_facet<wcalc_tag>(ft));
    }


    std::size_t length() const override
    {
        std::size_t len = 0;
        if (m_count > 0)
        {
            len = m_count * m_from32cv.length(m_char);
        }
        if (m_fillcount > 0)
        {
            len += m_fillcount * m_from32cv.length(m_fillchar);
        }
        return len;
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount == 0)
        {
            m_from32cv.write(out, m_char, m_count);
        }
        else if(m_alignment == stringify::v0::alignment::left)
        {
            m_from32cv.write(out, m_char, m_count);
            m_from32cv.write(out, m_fillchar, m_fillcount);
        }
        else
        {
            m_from32cv.write(out, m_fillchar, m_fillcount);
            m_from32cv.write(out, m_char, m_count);
        }
    }

    int remaining_width(int w) const override
    {
        if (w > 0 && std::size_t(w) > m_width)
        {
            return w - static_cast<int>(m_width);
        }
        return 0;
    }

private:

    const stringify::v0::conversion_from_utf32<CharT>& m_from32cv;
    const std::size_t m_count = 1;
    std::size_t m_width;
    const char32_t m_char;
    const char32_t m_fillchar;
    int m_fillcount = 0;
    const stringify::v0::alignment m_alignment;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_type>();
    }

    void determinate_fill_and_width(const stringify::v0::width_calculator& wcalc)
    {
        std::size_t content_width = 0;
        if ( m_count > 0 )
        {
            content_width = m_count * wcalc.width_of(m_char);
        }
        if (content_width >= m_width)
        {
            m_fillcount = 0;
            m_width = content_width;
        }
        else
        {
            int fillwidth = static_cast<int>(m_width - content_width);
            m_fillcount = fillwidth / wcalc.width_of(m_fillchar);
        }
    }
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_stringifier<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_stringifier<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_stringifier<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_stringifier<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


struct char32_input_traits
{
    template <typename CharT, typename FTuple>
    using stringifier
    = stringify::v0::detail::char32_stringifier<CharT>;
};

} // namespace detail

stringify::v0::detail::char32_input_traits
boost_stringify_input_traits_of(char32_t);


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED



