#ifndef BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP
#define BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP

namespace boost
{
namespace stringify
{

template <typename Output, typename Formating>
class char32_to_utf8: public boost::stringify::input_base<char, Output, Formating>
{
    typedef boost::stringify::input_base<char, Output, Formating> base;
    
public:

    char32_to_utf8() noexcept
        : m_char32(0XFFFFFFFF)
    {
    }

    char32_to_utf8(char32_t _codepoint) noexcept
        : m_char32(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
        m_char32 = _codepoint;
    }

    virtual std::size_t length(const Formating&) const noexcept override
    {
        return (m_char32 <     0x80 ? 1 :
                m_char32 <    0x800 ? 2 :
                m_char32 <  0x10000 ? 3 :
                m_char32 < 0x110000 ? 4 :
                /* invalid codepoit */ 0);
    }
    
    void write
        ( Output& out
        , const Formating& fmt
        ) const noexcept(base::noexcept_output) override
    {
        if (m_char32 <     0x80)
        {
            out.put(static_cast<char>(m_char32));
        }
        else if (m_char32 <    0x800)
        {
            out.put(static_cast<char>(0xC0 | ((m_char32 & 0x7C0) >> 6)));
            out.put(static_cast<char>(0x80 |  (m_char32 &  0x3F)));
        }
        else if (m_char32 <  0x10000)
        {
            out.put(static_cast<char>(0xE0 | ((m_char32 & 0xF000) >> 12)));
            out.put(static_cast<char>(0x80 | ((m_char32 &  0xFC0) >> 6)));
            out.put(static_cast<char>(0x80 |  (m_char32 &   0x3F)));
        }
        else if (m_char32 < 0x110000)
        {
            out.put(static_cast<char>(0xF0 | ((m_char32 & 0x1C0000) >> 18)));
            out.put(static_cast<char>(0x80 | ((m_char32 &  0x3F000) >> 12)));
            out.put(static_cast<char>(0x80 | ((m_char32 &    0xFC0) >> 6)));
            out.put(static_cast<char>(0x80 |  (m_char32 &     0x3F)));
        }
    }

private:

    char32_t m_char32;
};


template <typename CharT, typename Output, typename Formating>
inline typename std::enable_if
    < (sizeof(CharT) == sizeof(char))
    , boost::stringify::char32_to_utf8<Output, Formating>
    >
    ::type
argf(char32_t c) noexcept
{
    return c;
}


} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP */

