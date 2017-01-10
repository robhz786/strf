#ifndef BOOST_STRINGIFY_CUSTOM_FILL_HPP
#define BOOST_STRINGIFY_CUSTOM_FILL_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/ftuple.hpp>
#include <boost/stringify/custom_width.hpp>
#include <boost/stringify/custom_char32_conversion.hpp>
#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>

namespace boost {
namespace stringify {

struct fill_tag;

template
    < char32_t fill_char = U' '
    , template <class> class Filter = boost::stringify::accept_any_type  
    >
class fimpl_fill_static_single_char
{
public:

    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;
    
    template
        < typename CharT
        , typename InputType
        , typename Output
        , typename Formatting
        >
    void fill(width_type width, Output& out, const Formatting& fmt) const noexcept
    {
        decltype(auto) char32_writer
            = boost::stringify::get_char32_writer<CharT, InputType>(fmt);
        boost::stringify::width_t ch_width = fill_char_width<CharT, InputType>(fmt);
        for ( boost::stringify::width_t acc_width = ch_width
            ; acc_width <= width
            ; acc_width += ch_width
            )
        {
            char32_writer.write(fill_char, out);
        }
    }

    template <typename CharT, typename InputType, typename Formatting>
    std::size_t length(width_type width, const Formatting& fmt) const noexcept
    {
        std::size_t ch_length
            = get_char32_writer<CharT, InputType>(fmt).length(fill_char);
        boost::stringify::width_t ch_width = fill_char_width<CharT, InputType>(fmt);
        return ch_length * std::size_t(width / ch_width);
    }

private:

    template <typename CharT, typename InputType, typename Formatting>
    boost::stringify::width_t fill_char_width(const Formatting& fmt) const noexcept
    {
        boost::stringify::width_accumulator<Formatting, InputType, CharT> acc;
        acc.add(fill_char);
        return acc.result();
    }
};


template <template <class> class Filter = boost::stringify::accept_any_type>
class fimpl_fill_single_char
{
public:
    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;

    fimpl_fill_single_char(char32_t fill) : m_fill_char(fill)
    {
    }

    fimpl_fill_single_char(const fimpl_fill_single_char&) = default;

    template
        < typename CharT
        , typename InputType
        , typename Output
        , typename Formatting
        >
    void fill(width_type width, Output& out, const Formatting& fmt) const noexcept
    {
        decltype(auto) char32_writer = get_char32_writer<CharT, InputType>(fmt);
        boost::stringify::width_t ch_width = fill_char_width<CharT, InputType>(fmt);
        for ( boost::stringify::width_t acc_width = ch_width
            ; acc_width <= width
            ; acc_width += ch_width
            )
        {
            char32_writer.write(m_fill_char, out);
        }
    }

    template <typename CharT, typename InputType, typename Formatting>
    std::size_t length(width_type width, const Formatting& fmt) const noexcept
    {
        std::size_t ch_length
            = get_char32_writer<CharT, InputType>(fmt).length(m_fill_char);
        boost::stringify::width_t ch_width = fill_char_width<CharT, InputType>(fmt);
        return ch_length * std::size_t(width / ch_width);
    }

private:

    template <typename CharT, typename InputType, typename Formatting>
    boost::stringify::width_t fill_char_width(const Formatting& fmt) const noexcept
    {
        boost::stringify::width_accumulator<Formatting, InputType, CharT> acc;
        acc.add(m_fill_char);
        return acc.result();
    }

    char32_t m_fill_char;
};


struct fill_tag
{
    typedef 
        boost::stringify::fimpl_fill_static_single_char
            < U' '
            , boost::stringify::accept_any_type
            >
        default_impl;
};

template
    < char32_t fillChar
    , template <class> class Filter = boost::stringify::accept_any_type
    >
auto fill()
{
    return fimpl_fill_static_single_char<fillChar, Filter>();
}

template <template <class> class Filter = boost::stringify::accept_any_type>
auto fill(char32_t fillChar)
{
    return fimpl_fill_single_char<Filter>(fillChar);
}


template <typename InputType, typename Formatting>
decltype(auto) get_filler(const Formatting& fmt)
{
    return fmt.template get<boost::stringify::fill_tag, InputType>();
}

template <typename CharT, typename InputType, typename Output, typename Formatting>
void write_fill
( boost::stringify::width_t width
  , Output& out
  , const Formatting& fmt
)
{
    boost::stringify::get_filler<InputType>(fmt)
        . template fill<CharT, InputType>(width, out, fmt);
}

template <typename CharT, typename InputType, typename Formatting>
std::size_t fill_length
    ( boost::stringify::width_t width
    , const Formatting& fmt
    )
{
    return boost::stringify::get_filler<InputType>(fmt)
        . template length<CharT, InputType>(width, fmt);
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_FILL_HPP

