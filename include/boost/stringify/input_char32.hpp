#ifndef BOOST_STRINGIFY_INPUT_CHAR32_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR32_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <boost/stringify/custom_char32_conversion.hpp>
#include <type_traits>

namespace boost
{
namespace stringify
{

template <typename CharT, typename Output, typename Formatting>
class input_char32: public boost::stringify::input_base<CharT, Output, Formatting>
{
    typedef boost::stringify::input_base<CharT, Output, Formatting> base;
    
public:
    
    input_char32() noexcept
        : m_char32()
    {
    }

    input_char32(char32_t ch) noexcept
        : m_char32(ch)
    {
    }

    void set(char32_t ch) noexcept
    {
        m_char32 = ch;
    }

    virtual std::size_t length(const Formatting& fmt) const noexcept override
    {
        return boost::stringify::get_char32_writer<CharT, char32_t>(fmt)
            .length(m_char32);
    }
    
    void write
        ( Output& out
        , const Formatting& fmt
        ) const noexcept(base::noexcept_output) override
    {
        return boost::stringify::get_char32_writer<CharT, char32_t>(fmt)
            .write(m_char32, out);
    }
    
private:
    char32_t m_char32;
};


template <typename CharT, typename Output, typename Formatting>
boost::stringify::input_char32<CharT, Output, Formatting>
argf(char32_t c) noexcept
{
    return c;
}

} // namespace stringify
} // namespace boost

#endif



