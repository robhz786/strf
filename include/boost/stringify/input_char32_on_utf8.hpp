#ifndef BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP
#define BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP

namespace boost
{
namespace stringify
{

template <typename Formating>
class char32_to_utf8: public boost::stringify::input_base<char, Formating>
{
public:

    char32_to_utf8() noexcept
        : codepoint(0XFFFFFFFF)
    {
    }

    char32_to_utf8(char32_t _codepoint) noexcept
        : codepoint(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
        codepoint = _codepoint;
    }

    virtual std::size_t length(const Formating&) const noexcept
    {
        return (codepoint <     0x80 ? 1 :
                codepoint <    0x800 ? 2 :
                codepoint <  0x10000 ? 3 :
                codepoint < 0x110000 ? 4 :
                /* invalid codepoit */ 0);
    }

    virtual char* write_without_termination_char(char* out, const Formating&)
        const noexcept
    {
        return (codepoint <     0x80 ? write_utf8_range1(out) :
                codepoint <    0x800 ? write_utf8_range2(out) :
                codepoint <  0x10000 ? write_utf8_range3(out) :
                codepoint < 0x110000 ? write_utf8_range4(out) :
                /* invalid codepoit */ out);
    }

    // virtual void write
    //     ( boost::stringify::simple_ostream<char>& out
    //     , const Formating& fmt
    //     ) const
    // {
    //     char buff[4];
    //     write_without_termination_char(buff, fmt);
    //     out.write(buff, 4);
    // }

private:

    char32_t codepoint;

    char* write_utf8_range1(char* out) const noexcept
    {
        *out =  static_cast<char>(codepoint);
        return ++out;
    }

    char* write_utf8_range2(char* out) const noexcept
    {
        *  out = static_cast<char>(0xC0 | ((codepoint & 0x7C0) >> 6));
        *++out = static_cast<char>(0x80 |  (codepoint &  0x3F));
        return ++out;
    }

    char* write_utf8_range3(char* out) const noexcept
    {
        *  out = static_cast<char>(0xE0 | ((codepoint & 0xF000) >> 12));
        *++out = static_cast<char>(0x80 | ((codepoint &  0xFC0) >> 6));
        *++out = static_cast<char>(0x80 |  (codepoint &   0x3F));
        return ++out;
    }

    char* write_utf8_range4(char* out) const noexcept
    {
        *  out = static_cast<char>(0xF0 | ((codepoint & 0x1C0000) >> 18));
        *++out = static_cast<char>(0x80 | ((codepoint &  0x3F000) >> 12));
        *++out = static_cast<char>(0x80 | ((codepoint &    0xFC0) >> 6));
        *++out = static_cast<char>(0x80 |  (codepoint &     0x3F));
        return ++out;
    }
};


template <typename charT, typename Formating>
inline typename std::enable_if
    < (sizeof(charT) == sizeof(char))
    , boost::stringify::char32_to_utf8<Formating>
    >
    ::type
argf(char32_t c) noexcept
{
    return c;
}


} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF8_HPP */

