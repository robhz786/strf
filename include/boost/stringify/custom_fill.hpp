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
    
    template
        < typename CharT
        , typename InputType
        , typename Output
        , typename Formatting
        >
    void fill(width_type width, Output& out, const Formatting& fmt) const
    {
        decltype(auto) char32_writer
            = boost::stringify::get_char32_writer<CharT, InputType>(fmt);
        int count = quantity<CharT, InputType>(width, fmt);
        
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

    template <typename CharT, typename InputType, typename Formatting>
    std::size_t length(width_type width, const Formatting& fmt) const
    {
        std::size_t ch_length
            = get_char32_writer<CharT, InputType>(fmt).length(FillChar);
        return ch_length * quantity<CharT, InputType>(width, fmt);
    }

private:
    
    template <typename CharT, typename InputType, typename Formatting>
    int quantity(width_type width, const Formatting& fmt) const
    {
        boost::stringify::width_t ch_width =
            boost::stringify::get_width_calculator<InputType>(fmt)
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
    
    template
        < typename CharT
        , typename InputType
        , typename Output
        , typename Formatting
        >
    void fill(width_type width, Output& out, const Formatting& fmt) const
    {
        const auto& char32_writer
            = boost::stringify::get_char32_writer<CharT, InputType>(fmt);
        int count = quantity<CharT, InputType>(width, fmt);

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

    template <typename CharT, typename InputType, typename Formatting>
    std::size_t length(width_type width, const Formatting& fmt) const
    {
        std::size_t ch_length
            = get_char32_writer<CharT, InputType>(fmt).length(m_fillchar);
        return ch_length * quantity<CharT, InputType>(width, fmt);
    }

private:

    char32_t m_fillchar;
    
    template <typename CharT, typename InputType, typename Formatting>
    int quantity(width_type width, const Formatting& fmt) const
    {
        boost::stringify::width_t ch_width =
            boost::stringify::get_width_calculator<InputType>(fmt)
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

