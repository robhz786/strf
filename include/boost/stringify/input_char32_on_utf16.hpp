#ifndef BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP
#define BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP

namespace boost
{
namespace stringify
{

template <typename charT, typename Formating>
class char32_to_utf16: public boost::stringify::input_base<charT, Formating>
{
public:

    char32_to_utf16() noexcept
        : codepoint(0xFFFFFFFF)
    {
    }

    char32_to_utf16(char32_t _codepoint) noexcept
        : codepoint(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
        codepoint = _codepoint;
    }

    virtual std::size_t length(const Formating&) const noexcept
    {
        return (single_char_range() ? 1 :
                two_chars_range()   ? 2 : 0);
    }

    virtual charT* write_without_termination_char(charT* out, const Formating&)
        const noexcept
    {
        if (single_char_range())
        {
            *out++ = static_cast<charT>(codepoint);
        }
        else if (two_chars_range())
        {
            char32_t sub_codepoint = codepoint - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            *out++ = static_cast<charT>(high_surrogate);
            *out++ = static_cast<charT>(low_surrogate);
        }
        return out;
    }

    // virtual void write_ostream
    //     ( boost::stringify::simple_ostream<charT>& out
    //     , const Formating&
    //     ) const
    // {
    //     if (single_char_range())
    //     {
    //         out.put(static_cast<charT>(codepoint));
    //     }
    //     else if (two_chars_range())
    //     {
    //         char32_t sub_codepoint = codepoint - 0x10000;
    //         char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
    //         char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
    //         out.put(static_cast<charT>(high_surrogate));
    //         out.put(static_cast<charT>(low_surrogate));
    //     }
    // }

private:

    char32_t codepoint;

    bool single_char_range() const
    {
        return codepoint < 0xd800 || (0xdfff < codepoint && codepoint <  0x10000);
    }

    bool two_chars_range() const
    {
        return 0xffff < codepoint && codepoint < 0x110000;
    }
};

template <typename charT, typename Formating>
inline typename std::enable_if
    < (sizeof(charT) == sizeof(char16_t))
    , boost::stringify::char32_to_utf16<charT, Formating>
    >
    ::type
argf(char32_t c) noexcept
{
    return c;
}


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_INPUT_CHAR32_ON_UTF16_HPP */

