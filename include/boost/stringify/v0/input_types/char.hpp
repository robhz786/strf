#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>
#include <boost/stringify/v0/input_types/char32.hpp>
#include <boost/stringify/v0/type_traits.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharT, typename FTuple>
class char_stringifier
{

public:

    using char_type = CharT;
    using input_type = char_type;
    using writer_type = boost::stringify::v0::output_writer<CharT>;
    using ftuple_type = FTuple ;

    char_stringifier(const FTuple& fmt, CharT ch) noexcept
        : m_fmt(fmt)
        , m_char(ch)
    {
    }

    std::size_t length() const
    {
        return 1;
    }

    void write(writer_type& out) const
    {
        out.put(m_char);
    }

    int remaining_width(int w) const
    {
        auto calc = boost::stringify::v0::get_width_calculator<input_type>(m_fmt);
        return w - calc.width_of(m_char);
    }


private:

    const FTuple& m_fmt;
    CharT m_char;
};

template <typename CharIn>
struct char_input_traits
{

private:

    template <typename CharOut, typename FTuple>
    struct checker
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        using stringifier
        = boost::stringify::v0::detail::char_stringifier<CharOut, FTuple>;
    };

public:

    template <typename CharOut, typename FTuple>
    using stringifier = typename checker<CharOut, FTuple>::stringifier;
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



