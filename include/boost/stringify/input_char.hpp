#ifndef BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <type_traits>

namespace boost
{
namespace stringify
{

template <typename CharT, typename Output, typename Formatting>
class input_char: public boost::stringify::input_base<CharT, Output, Formatting>
{
    typedef boost::stringify::input_base<CharT, Output, Formatting> base;
    
public:
    
    input_char() noexcept
        : m_char()
    {
    }

    input_char(CharT _character) noexcept
        : m_char(_character)
    {
    }

    void set(CharT _character) noexcept
    {
        m_char = _character;
    }

    virtual std::size_t length(const Formatting&) const noexcept override
    {
        return 1;
    }
    
    void write
        ( Output& out
        , const Formatting& fmt
        ) const noexcept(base::noexcept_output) override
    {
        out.put(m_char);
    }
    
private:
    
    CharT m_char;
};


template <typename CharT, typename Output, typename Formatting>
typename std::enable_if
    < ! std::is_same<CharT, char32_t>::value // because there is input_char32
    , boost::stringify::input_char<CharT, Output, Formatting>
    > :: type
argf(CharT c) noexcept
{
    return c;
}

} // namespace stringify
} // namespace boost

#endif



