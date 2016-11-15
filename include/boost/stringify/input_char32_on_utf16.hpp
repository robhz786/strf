#ifndef BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP
#define BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP

namespace boost
{
namespace stringify
{

template <typename charT, typename Output, typename Formating>
class char32_to_utf16: public boost::stringify::input_base<charT, Output, Formating>
{
    typedef boost::stringify::input_base<charT, Output, Formating> base;
    
public:

    char32_to_utf16() noexcept
        : m_char32(0xFFFFFFFF)
    {
    }

    char32_to_utf16(char32_t _codepoint) noexcept
        : m_char32(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
        m_char32 = _codepoint;
    }

    virtual std::size_t length(const Formating&) const noexcept
    {
        if(single_char_range())
        {
            return 1;
        }
        if(two_chars_range())
        {
            return 2;
        }
        return 0;
    }
    
    void write
        ( Output& out
        , const Formating& fmt
        ) const noexcept(base::noexcept_output) override
    {
        if (single_char_range())
        {
            out.put(static_cast<charT>(m_char32));
        }
        else if (two_chars_range())
        {
            char32_t sub_codepoint = m_char32 - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            out.put(static_cast<charT>(high_surrogate));
            out.put(static_cast<charT>(low_surrogate));
        }
    }

 
private:

    char32_t m_char32;

    bool single_char_range() const
    {
        return m_char32 < 0xd800 || (0xdfff < m_char32 && m_char32 <  0x10000);
    }

    bool two_chars_range() const
    {
        return 0xffff < m_char32 && m_char32 < 0x110000;
    }
};

template <typename charT, typename Output, typename Formating>
inline typename std::enable_if
    < (sizeof(charT) == sizeof(char16_t))
    , boost::stringify::char32_to_utf16<charT, Output, Formating>
    >
    ::type
argf(char32_t c) noexcept
{
    return c;
}


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP */

