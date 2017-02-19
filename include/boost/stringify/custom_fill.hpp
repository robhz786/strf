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
    < char32_t FillChar = U' '
    , template <class> class Filter = boost::stringify::true_trait  
    >
class fill_impl_t
{
public:

    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;
    
    template<typename CharT, typename Output, typename FTuple>
    void fill(width_type width, Output& out, const FTuple& fmt) const
    {
        decltype(auto) char32_writer
            = boost::stringify::get_char32_writer<CharT, category>(fmt);
        int count = quantity<CharT>(width, fmt);
        
        if (CharT ch = char32_writer.convert_if_length_is_1(FillChar))
        {
            out.put(ch, count);
        }
        else
        {
            while(count -- > 0)
            {
                char32_writer.write(FillChar, out);
            }
        }
    }

    template <typename CharT, typename FTuple>
    std::size_t length(width_type width, const FTuple& fmt) const
    {
        std::size_t ch_length
            = get_char32_writer<CharT, category>(fmt).length(FillChar);
        return ch_length * quantity<CharT>(width, fmt);
    }

private:
    
    template <typename CharT, typename FTuple>
    int quantity(width_type width, const FTuple& fmt) const
    {
        boost::stringify::width_t ch_width =
            boost::stringify::get_width_calculator<category>(fmt)
            .width_of(FillChar);

        if (ch_width == 0 || ch_width == 1)
        {
            return width;
        }
        return width / ch_width;
    }
};


template <template <class> class Filter = boost::stringify::true_trait>
class fill_impl
{
public:
    fill_impl(char32_t ch)
        : m_fillchar(ch)
    {
    }

    
    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;
    
    template <typename CharT, typename Output, typename FTuple>
    void fill(width_type width, Output& out, const FTuple& fmt) const
    {
        const auto& char32_writer
            = boost::stringify::get_char32_writer<CharT, category>(fmt);
        int count = quantity<CharT>(width, fmt);

        if (CharT ch = char32_writer.convert_if_length_is_1(m_fillchar))
        {
            out.put(ch, count);
        }
        else
        {
            while(count -- > 0)
            {
                char32_writer.write(m_fillchar, out);
            }
        }
    }

    template <typename CharT, typename FTuple>
    std::size_t length(width_type width, const FTuple& fmt) const
    {
        std::size_t ch_length
            = get_char32_writer<CharT, category>(fmt).length(m_fillchar);
        return ch_length * quantity<CharT>(width, fmt);
    }

private:

    char32_t m_fillchar;
    
    template <typename CharT, typename FTuple>
    int quantity(width_type width, const FTuple& fmt) const
    {
        boost::stringify::width_t ch_width =
            boost::stringify::get_width_calculator<category>(fmt)
            .width_of(m_fillchar);

        if (ch_width == 0 || ch_width == 1)
        {
            return width;
        }
        return width / ch_width;
    }
};


struct fill_tag
{
    typedef 
        boost::stringify::fill_impl_t<U' ', boost::stringify::true_trait>
        default_impl;
};


template <typename T>
struct ic_fill: public std::false_type
{
};


template <>
struct ic_fill<boost::stringify::fill_tag>: public std::true_type
{
};


auto fill(char32_t fillChar)
{
    return fill_impl<boost::stringify::true_trait>(fillChar);
}


template <template <class> class Filter>
auto fill_if(char32_t fillChar)
{
    return fill_impl<Filter>(fillChar);
}


template <char32_t fillChar>
auto fill_t = fill_impl_t<fillChar, boost::stringify::true_trait>();


template <char32_t Char, template <class> class Filter>
auto fill_t_if = fill_impl_t<Char, Filter>();


template <typename InputType, typename FTuple>
decltype(auto) get_filler(const FTuple& fmt)
{
    return fmt.template get<boost::stringify::fill_tag, InputType>();
}


template <typename CharT, typename InputType, typename Output, typename FTuple>
void write_fill
( boost::stringify::width_t width
  , Output& out
  , const FTuple& fmt
)
{
    boost::stringify::get_filler<InputType>(fmt)
        . template fill<CharT>(width, out, fmt);
}


template <typename CharT, typename InputType, typename FTuple>
std::size_t fill_length
    ( boost::stringify::width_t width
    , const FTuple& fmt
    )
{
    return boost::stringify::get_filler<InputType>(fmt)
        . template length<CharT>(width, fmt);
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_FILL_HPP

