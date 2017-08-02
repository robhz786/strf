#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>
#include <boost/stringify/v0/facets/conversion_from_utf32.hpp>
#include <type_traits>

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {

template <typename CharT>
class char32_stringifier
{

    using cv_tag = boost::stringify::v0::conversion_from_utf32_tag<CharT>;
    using wc_tag = boost::stringify::v0::width_calculator_tag;

public:

    using input_type = char32_t;
    using char_type = CharT;
    using writer_type = boost::stringify::v0::output_writer<CharT>;;

    template <typename FTuple>
    char32_stringifier(const FTuple& fmt, char32_t ch) noexcept
        : m_conv(fmt.template get_facet<cv_tag, char32_t>())
        , m_char32(ch)
        , m_width(fmt.template get_facet<wc_tag, char32_t>().width_of(ch))
    {
    }

    std::size_t length() const
    {
        return m_conv.length(m_char32);
    }
    
    void write(writer_type& out) const
    {
        m_conv.write(out, m_char32);
    }

    int remaining_width(int w) const
    {
        return std::max(0, w - m_width);
    }


private:

    const boost::stringify::v0::conversion_from_utf32<CharT>& m_conv;
    char32_t m_char32;
    int m_width;

};


struct char32_input_traits
{
    template <typename CharT, typename FTuple>
    using stringifier
    = boost::stringify::v0::detail::char32_stringifier<CharT>;
};

} // namespace detail

boost::stringify::v0::detail::char32_input_traits
boost_stringify_input_traits_of(char32_t);


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR32_HPP_INCLUDED



