#ifndef BOOST_STRINGIFY_DETAIL_UTF32_TO_UTF8_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_UTF32_TO_UTF8_HPP_INCLUDED

#include <boost/stringify/detail/char_writer.hpp>

namespace boost {
namespace stringify {
namespace detail {

template<typename charT, typename Formating>
struct utf32_to_utf8: boost::stringify::str_writer<char, Formating>
{
    const charT* str;

    utf32_to_utf8() noexcept:
                     str(0)
    {
    }

    utf32_to_utf8(const charT* _str) noexcept :
                                      str(_str)
    {
    }

    void set(const charT* _str) noexcept
    {
        str = _str;
    }

    virtual std::size_t minimal_length(const Formating& fmt) const noexcept
    {
        std::size_t len = 0;
        for (const charT* it = str; *it != charT(); ++it)
            len += char32_to_utf8<Formating>(*it).minimal_length(fmt);

        return len;
    }

    virtual char* write_without_termination_char
        ( char* out
        , const Formating& fmt
        )
        const noexcept
    {
        for (const charT* it = str; *it != charT(); ++it)
        {
            out = char32_to_utf8<Formating>(*it)
                .write_without_termination_char(out, fmt);
        }
        return out;
    }

    virtual void write
        ( boost::stringify::simple_ostream<char>& out
        , const Formating& fmt
        )
        const
    {
        for (const charT* it = str; *it != charT() && out.good(); ++it)
            char32_to_utf8<Formating>(*it).write(out, fmt);
    }
};

}; //namespace boost
}; //namespace stringify
}; //namespace detail

#endif

