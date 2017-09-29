#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_types/char32.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharT>
class char_stringifier: public stringifier<CharT>
{
    using input_type = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;
    using from32_tag = stringify::v0::conversion_from_utf32_tag<CharT>;
    using wcalc_tag = stringify::v0::width_calculator_tag;
    using argf_reader = stringify::v0::conventional_argf_reader<input_type>;

public:

    using second_arg = stringify::v0::char_argf;

    template <typename FTuple>
    char_stringifier
        ( const FTuple& ft
        , char32_t ch
        ) noexcept
        : m_from32cv(get_facet<from32_tag>(ft))
        , m_width(get_facet<width_tag>(ft).width())
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_alignment(get_facet<alignment_tag>(ft).value())
        , m_char(ch)
    {
        determinate_fill_and_width(get_facet<wcalc_tag>(ft));
    }

    template <typename FTuple>
    char_stringifier
        ( const FTuple& ft
        , char32_t ch
        , const second_arg& argf
        ) noexcept
        : m_from32cv(get_facet<from32_tag>(ft))
        , m_count(argf.count)
        , m_width(argf_reader::get_width(argf, ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_alignment(argf_reader::get_alignment(argf, ft))
        , m_char(ch)
    {
        determinate_fill_and_width(get_facet<wcalc_tag>(ft));
    }


    std::size_t length() const override
    {
        std::size_t len = m_count;
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
            out.repeat(m_count, m_char);
        }
        else if(m_alignment == stringify::v0::alignment::left)
        {
            out.repeat(m_count, m_char);
            m_from32cv.write(out, m_fillcount, m_fillchar);
        }
        else
        {
            m_from32cv.write(out, m_fillcount, m_fillchar);
            out.repeat(m_count, m_char);
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
    const char32_t m_fillchar;
    int m_fillcount = 0;
    const stringify::v0::alignment m_alignment;
    const CharT m_char;

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
            char32_t ch32 = m_char; // todo: use convertion_to_utf32 facet ?
            content_width = m_count * wcalc.width_of(ch32);
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

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_stringifier<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_stringifier<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_stringifier<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_stringifier<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


template <typename CharIn>
struct char_input_traits
{

private:

    template <typename CharOut>
    struct checker
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        using stringifier
        = boost::stringify::v0::detail::char_stringifier<CharOut>;
    };

public:

    template <typename CharOut, typename>
    using stringifier = typename checker<CharOut>::stringifier;
};

} //namepace detail


boost::stringify::v0::detail::char_input_traits<char>
boost_stringify_input_traits_of(char);

boost::stringify::v0::detail::char_input_traits<char16_t>
boost_stringify_input_traits_of(char16_t);

boost::stringify::v0::detail::char_input_traits<wchar_t>
boost_stringify_input_traits_of(wchar_t);

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED



