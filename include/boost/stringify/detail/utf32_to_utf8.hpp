#ifndef BOOST_STRINGIFY_DETAIL_UTF32_TO_UTF8_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_UTF32_TO_UTF8_HPP_INCLUDED

#include <boost/stringify/input_char.hpp>

namespace boost {
namespace stringify {
namespace detail {

template<typename CharT, typename Formatting>
struct utf32_to_utf8: boost::stringify::input_base<char, Formatting>
{
    const CharT* str;

    utf32_to_utf8() noexcept:
                     str(0)
    {
    }

    utf32_to_utf8(const CharT* _str) noexcept :
                                      str(_str)
    {
    }

    void set(const CharT* _str) noexcept
    {
        str = _str;
    }

    virtual std::size_t length(const Formatting& fmt) const noexcept
    {
        std::size_t len = 0;
        for (const CharT* it = str; *it != CharT(); ++it)
            len += char32_to_utf8<Formatting>(*it).length(fmt);

        return len;
    }

    virtual char* write_without_termination_char
        ( char* out
        , const Formatting& fmt
        )
        const noexcept
    {
        for (const CharT* it = str; *it != CharT(); ++it)
        {
            out = char32_to_utf8<Formatting>(*it)
                .write_without_termination_char(out, fmt);
        }
        return out;
    }

    virtual void write_ostream
        ( boost::stringify::simple_ostream<char>& out
        , const Formatting& fmt
        )
        const
    {
        for (const CharT* it = str; *it != CharT() && out.good(); ++it)
            char32_to_utf8<Formatting>(*it).write(out, fmt);
    }
};

}; //namespace boost
}; //namespace stringify
}; //namespace detail

#endif

