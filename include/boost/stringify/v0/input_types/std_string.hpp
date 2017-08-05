#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_types/char_ptr.hpp>
#include <string>
#include <type_traits>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename Traits, typename CharT>
class std_string_stringifier
{

public:

    using char_type = CharT;
    using input_type  = std::basic_string<CharT, Traits>;
    using writer_type = boost::stringify::v0::output_writer<CharT>;
    using second_arg = boost::stringify::v0::detail::string_argf;

private:

    using cv_tag = boost::stringify::v0::conversion_from_utf32_tag<CharT>;
    using wcalc_tag = boost::stringify::v0::width_calculator_tag;
    using argf_reader = boost::stringify::v0::conventional_argf_reader<input_type>;

public:

    template <typename FTuple>
    std_string_stringifier
        ( const FTuple& ft
        , const input_type& str
        , second_arg argf
        ) noexcept
        : m_str(str)
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(argf_reader::get_width(argf, ft))
        , m_alignment(argf_reader::get_alignment(argf, ft))
    {
        init(get_facet<wcalc_tag>(ft));
    }

    template <typename FTuple>
    std_string_stringifier(const FTuple& ft, const input_type& str) noexcept
        : m_str(str)
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(get_facet<width_tag>(ft).width())
        , m_alignment(get_facet<alignment_tag>(ft).value())
    {
        init(get_facet<wcalc_tag>(ft));
    }

    std::size_t length() const
    {
        if (m_fillcount > 0)
        {
            return m_str.length() +  m_conv.length(m_fillchar) * m_fillcount;
        }
        return m_str.length();
    }

    void write(writer_type& out) const
    {
        if (m_fillcount > 0)
        {
            if(m_alignment == boost::stringify::v0::alignment::left)
            {
                out.put(&m_str[0], m_str.length());
                write_fill(out);
            }
            else
            {
                write_fill(out);
                out.put(&m_str[0], m_str.length());
            }
        }
        else
        {
            out.put(&m_str[0], m_str.length());
        }
    }

    int remaining_width(int w) const
    {
        return w > m_total_width ? w - m_total_width : 0;
    }


private:

    const input_type& m_str;
    const boost::stringify::v0::conversion_from_utf32<CharT>& m_conv;
    const char32_t m_fillchar;
    width_t m_fillcount = 0;
    width_t m_total_width;
    boost::stringify::v0::alignment m_alignment;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_type>();
    }

    template <typename WidthCalculator>
    void init(const WidthCalculator& wcalc)
    {
        int content_width = wcalc.width_of(m_str.c_str(), m_str.length());
        if(content_width < m_total_width)
        {
            auto fillchar_width = wcalc.width_of(m_fillchar);
            m_fillcount = (m_total_width - content_width) / fillchar_width;
        }
        else
        {
            m_total_width = content_width;
        }
    }

    void write_fill(writer_type& out) const
    {
        m_conv.write(out, m_fillchar, m_fillcount);
    }
};


template <typename CharIn, typename CharTraits>
struct std_string_input_traits
{
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        using stringifier = boost::stringify::v0::detail::std_string_stringifier
            <CharTraits, CharOut>;
    };

public:

    template <typename Output, typename FTuple>
    using stringifier = typename helper<Output>::stringifier;
};

} // namespace detail


template<typename CharT, typename CharTraits>
auto boost_stringify_input_traits_of(const std::basic_string<CharT, CharTraits>& str)
    -> boost::stringify::v0::detail::std_string_input_traits<CharT, CharTraits>;

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif //BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED
