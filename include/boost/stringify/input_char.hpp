#ifndef BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <type_traits>

namespace boost
{
namespace stringify
{

template <typename charT, typename Output, typename Formating>
class input_char: public boost::stringify::input_base<charT, Output, Formating>
{
    typedef boost::stringify::input_base<charT, Output, Formating> base;
    
public:
    
    input_char() noexcept
        : m_char()
    {
    }

    input_char(charT _character) noexcept
        : m_char(_character)
    {
    }

    void set(charT _character) noexcept
    {
        m_char = _character;
    }

    virtual std::size_t length(const Formating&) const noexcept override
    {
        return 1;
    }
    
    void write
        ( Output& out
        , const Formating& fmt
        ) const noexcept(base::noexcept_output) override
    {
        out.put(m_char);
    }
    
private:
    
    charT m_char;
};


template <typename charT, typename Output, typename Formating>
boost::stringify::input_char<charT, Output, Formating>
argf(charT c) noexcept
{
    return c;
}


template <typename charT, typename Output, typename Formating>
typename std::enable_if
    < std::is_same<charT, wchar_t>::value && sizeof(charT) == sizeof(char32_t)
    , boost::stringify::input_char<charT, Output, Formating>
    > :: type
argf(char32_t c) noexcept
{
    return c;
}

} // namespace stringify
} // namespace boost

#endif



