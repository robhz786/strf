#ifndef BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/input_char_ptr.hpp>
#include <string>
#include <type_traits>

namespace boost {
namespace stringify{
namespace detail {

template <class CharT, typename Traits, typename Output, class FTuple>
class std_string_stringifier
{

public:
    
    using input_type  = std::basic_string<CharT, Traits>;
    using output_type = Output;
    using ftuple_type = FTuple;
    using arg_format_type = boost::stringify::detail::string_arg_format
        <input_type, FTuple>;

   std_string_stringifier
        ( const FTuple& fmt
        , const input_type& str
        , arg_format_type argfmt
        ) noexcept
        : m_fmt(fmt)
        , m_str(str)
        , m_total_width(argfmt.get_width(fmt))
        , m_padding_width(padding_width())
        , m_alignment(argfmt.get_alignment(fmt))
    {
    }
    
    std_string_stringifier(const FTuple& fmt, const input_type& str) noexcept
        : m_fmt(fmt)
        , m_str(str)
        , m_total_width(get_facet<width_tag>().width())
        , m_padding_width(padding_width())
        , m_alignment(get_facet<alignment_tag>().value())
    {
    }

    std::size_t length() const
    {
        if (m_padding_width > 0)
        {
            return m_str.length() + 
                boost::stringify::fill_length<CharT, input_type>
                (m_padding_width, m_fmt);
        }
        return m_str.length();
    }

    void write(Output& out) const
    {
        if (m_padding_width > 0)
        {
            if(m_alignment == boost::stringify::alignment::left)
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
        if(m_total_width > w)
        {
            return 0;
        }
        if(m_padding_width > 0)
        {
            return w - m_total_width;
        }
        return
            boost::stringify::get_width_calculator<input_type>(m_fmt)
            .remaining_width(w, m_str.c_str(), m_str.length());
    }


private:

    const FTuple& m_fmt;
    const input_type& m_str;
    const width_t m_total_width;
    const width_t m_padding_width;
    boost::stringify::alignment m_alignment;

    template <typename fmt_tag>
    decltype(auto) get_facet() const noexcept
    {
        return m_fmt.template get<fmt_tag, input_type>();
    }
    
    void write_fill(Output& out) const
    {
        boost::stringify::write_fill<CharT, input_type>
                (m_padding_width, out, m_fmt);
    }
    
    width_t padding_width() const
    {
        return
            boost::stringify::get_width_calculator<input_type>(m_fmt)
            .remaining_width(m_total_width, &m_str[0], m_str.length());
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

        template <typename Output, typename FTuple>
        using stringifier
        = boost::stringify::detail::std_string_stringifier
            <CharOut, CharTraits, Output, FTuple>;
    };

public:

    template <typename CharT, typename Output, typename FTuple>
    using stringifier
    = typename helper<CharT>::template stringifier<Output, FTuple>;
};

} // namespace detail


template
    < typename String
    , typename CharT = typename String::value_type
    , typename CharTraits = typename String::traits_type
    >
auto boost_stringify_input_traits_of(const String& str)
    -> std::enable_if_t
        < std::is_same<String, std::basic_string<CharT, CharTraits> >::value
        , boost::stringify::detail::std_string_input_traits<CharT, CharTraits>
        >;
/*    
    -> std::enable_if_t
        < std::is_same<typename CharTraits::char_type, CharT>::value
       && std::is_convertible<decltype(str[0]), const CharT&>::value
       && std::is_convertible<decltype(str.length()), std::size_t>::value
       && std::is_same
              < typename std::iterator_traits<typename String::iterator>::iterator_category
              , std::random_access_iterator_tag
              >::value  
        , boost::stringify::detail::std_string_input_traits<CharT, CharTraits>
        >;
*/


} // namespace stringify
} // namespace boost


#endif
