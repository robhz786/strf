#ifndef BOOST_STRINGIFY_DETAIL_STRING_WRITER_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <limits>
#include <boost/stringify/str_writer.hpp>

namespace boost
{
namespace stringify
{
namespace detail
{

template <class charT, class traits, class Formating>
struct std_string_writer: public boost::stringify::str_writer<charT, Formating>
{
    const std::basic_string<charT, traits>* str;    

    std_string_writer() noexcept
        : str(0)
    {
    }

    std_string_writer(const std::basic_string<charT, traits>& _str) noexcept
        : str(&_str)
    {
    }

    void set(const std::basic_string<charT, traits>& _str) noexcept
    {
        str = &_str;
    }

    virtual std::size_t minimal_length(const Formating&) const noexcept
    {
        return str ? str->length() : 0;
    }

    virtual charT* write_without_termination_char
        ( charT* out
        , const Formating&
        ) const noexcept
    {
        if( ! str)
        {
            return out;
        }
        return std::copy(str->begin(), str->end(), out);
    }

    virtual void write(boost::stringify::simple_ostream<charT>& out, const Formating&) const
    {
        if(str)
        {
            out.write(str->c_str(), str->length());
        }
    }
};


template<typename charT, typename Formating>
struct char_ptr_writer: boost::stringify::str_writer<charT, Formating>
{
    char_ptr_writer() noexcept:
                       str(0),
                       len(0)
    {
    }

    char_ptr_writer(const charT* _str) noexcept
        : str(_str)
        , len((std::numeric_limits<std::size_t>::max) ())
    {
    }

    void set(const charT* _str) noexcept
    {
        str = _str;
        len = (std::numeric_limits<std::size_t>::max) ();
    }

    virtual std::size_t minimal_length(const Formating&) const noexcept
    {
        return get_length();
    }

    virtual charT* write_without_termination_char(charT* out, const Formating&) const noexcept
    {
        return std::copy(str, str + get_length(), out);
    }

    virtual void write(boost::stringify::simple_ostream<charT>& out, const Formating&) const
    {
        if(str)
        {
            out.write(str, get_length());
        }
    }


private:
    const charT* str;
    mutable std::size_t len;

    std::size_t get_length() const noexcept
    {
        if (len == (std::numeric_limits<std::size_t>::max) ())
        {
            try
            {
                len = std::char_traits<charT>::length(str);
            }
            catch(...)
            {
                len = 0;
            }
        }
        return len;
    }

}; 

template <typename charT, typename Formating>
inline
boost::stringify::detail::char_ptr_writer<charT, Formating> argf(const charT* str) noexcept
{
    return str;
}

template <typename charT, typename Formating, typename traits>
inline
boost::stringify::detail::std_string_writer<charT, traits, Formating>
argf(const std::basic_string<charT, traits>& str) noexcept
{
    return str;
}

} // namespace detail
} // namespace stringify
} // namespace boost


#endif
