#ifndef BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/stringifier.hpp>

namespace boost {
namespace stringify{
namespace detail {

template <class CharT, typename Traits, typename Output, class Formatting>
class std_string_stringifier
    : public boost::stringify::stringifier<CharT, Output, Formatting>
{
    typedef boost::stringify::stringifier<CharT, Output, Formatting> base;

public:

    typedef const std::basic_string<CharT, Traits>& input_type;
    typedef CharT char_type;
    typedef Output output_type;
    typedef Formatting ftuple_type;

    std_string_stringifier
        ( const Formatting& fmt
        , const std::basic_string<CharT, Traits>& str
        ) noexcept
        : m_fmt(fmt)
        , m_str(str)
    {
    }

    virtual std::size_t length() const noexcept override
    {
        return m_str.length();
    }

    void write(Output& out) const noexcept(base::noexcept_output) override
    {
        if(m_str)
        {
            out.put(m_str.c_str(), m_str.length());
        }
    }
    
private:
    
    const Formatting& m_fmt;
    const std::basic_string<CharT, Traits>& m_str;
    
};

template <typename CharIn, typename CharTraits>
struct std_string_input_traits
{
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        template <typename Output, typename Formatting>
        using stringifier
        = boost::stringify::detail::std_string_stringifier
            <CharOut, CharTraits, Output, Formatting>;
    };

public:

    template <typename CharT, typename Output, typename Formatting>
    using stringifier
    = typename helper<CharT>::template stingificator<Output, Formatting>;
};

} // namespace detail


template <typename CharT, typename CharTraits>
boost::stringify::detail::std_string_input_traits<CharT, CharTraits>
boost_stringify_input_traits_of(const std::basic_string<CharT, CharTraits>&);


} // namespace stringify
} // namespace boost


#endif
