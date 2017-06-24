#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_types/char_ptr.hpp>
#include <string>
#include <type_traits>

namespace boost {
namespace stringify{
inline namespace v0 {
namespace detail {

template <typename Traits, typename Output, class FTuple>
class std_string_stringifier
{

public:

    using char_type = typename Output::char_type;
    using input_type  = std::basic_string<char_type, Traits>;
    using output_type = Output;
    using ftuple_type = FTuple;
    using second_arg = boost::stringify::v0::detail::string_arg_format
        <input_type, FTuple>;

   std_string_stringifier
        ( const FTuple& fmt
        , const input_type& str
        , second_arg argfmt
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
                boost::stringify::v0::fill_length<char_type, input_type>
                (m_padding_width, m_fmt);
        }
        return m_str.length();
    }

    void write(Output& out) const
    {
        if (m_padding_width > 0)
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
        if(m_total_width > w)
        {
            return 0;
        }
        if(m_padding_width > 0)
        {
            return w - m_total_width;
        }
        return
            boost::stringify::v0::get_width_calculator<input_type>(m_fmt)
            .remaining_width(w, m_str.c_str(), m_str.length());
    }


private:

    const FTuple& m_fmt;
    const input_type& m_str;
    const width_t m_total_width;
    const width_t m_padding_width;
    boost::stringify::v0::alignment m_alignment;

    template <typename FacetCategory>
    decltype(auto) get_facet() const noexcept
    {
        return boost::stringify::v0::get_facet<FacetCategory, input_type>(m_fmt);
    }
    
    void write_fill(Output& out) const
    {
        boost::stringify::v0::write_fill<char_type, input_type>
                (m_padding_width, out, m_fmt);
    }
    
    width_t padding_width() const
    {
        return
            boost::stringify::v0::get_width_calculator<input_type>(m_fmt)
            .remaining_width(m_total_width, &m_str[0], m_str.length());
    }
};


template <typename CharIn, typename CharTraits>
struct std_string_input_traits
{
private:

    template <typename Output, typename FTuple>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(typename Output::char_type), "");

        using stringifier = boost::stringify::v0::detail::std_string_stringifier
            <CharTraits, Output, FTuple>;
    };

public:

    template <typename Output, typename FTuple>
    using stringifier = typename helper<Output, FTuple>::stringifier;
};

} // namespace detail


template<typename CharT, typename CharTraits>
auto boost_stringify_input_traits_of(const std::basic_string<CharT, CharTraits>& str)
    -> boost::stringify::v0::detail::std_string_input_traits<CharT, CharTraits>;

} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif //BOOST_STRINGIFY_V0_INPUT_TYPES_STD_STRING_HPP_INCLUDED
