#ifndef BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/input_char32.hpp>
#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename CharT, typename Output, typename Formatting>
class char_stringifier
    : public boost::stringify::stringifier<CharT, Output, Formatting>
{
    typedef boost::stringify::stringifier<CharT, Output, Formatting> base;
    
public:

    typedef CharT input_type;
    typedef CharT char_type;
    typedef Output output_type;
    typedef Formatting ftuple_type;
    
    char_stringifier(const Formatting& fmt, CharT _character) noexcept
        : m_fmt(fmt)
        , m_char(_character)
    {
    }

    virtual std::size_t length() const override
    {
        return 1;
    }
    
    void write(Output& out) const override
    {
        out.put(m_char);
    }
    
private:
   
    const Formatting& m_fmt; 
    CharT m_char;
};

template <typename CharIn>
struct char_input_traits
{
    
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");
        
        template <typename Output, typename Formatting>
        using stringifier
        = boost::stringify::detail::char_stringifier
            <CharOut, Output, Formatting>;
    };
    
public:
    
    template <typename CharT, typename Output, typename Formatting>
    using stringifier
    = typename helper<CharT>::template stringifier<Output, Formatting>;
};

} //namepace detail


boost::stringify::detail::char_input_traits<char>
boost_stringify_input_traits_of(char);

boost::stringify::detail::char_input_traits<char16_t>
boost_stringify_input_traits_of(char16_t);

boost::stringify::detail::char_input_traits<wchar_t>
boost_stringify_input_traits_of(wchar_t);


} // namespace stringify
} // namespace boost

#endif



