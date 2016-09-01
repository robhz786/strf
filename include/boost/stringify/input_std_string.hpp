#ifndef BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_STD_STRING_HPP_HPP_INCLUDED

#include <string>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template <class charT, class traits, class Formating>
struct input_std_string: public boost::stringify::input_base<charT, Formating>
{
    const std::basic_string<charT, traits>* str;    

    input_std_string() noexcept
        : str(0)
    {
    }

    input_std_string(const std::basic_string<charT, traits>& _str) noexcept
        : str(&_str)
    {
    }

    void set(const std::basic_string<charT, traits>& _str) noexcept
    {
        str = &_str;
    }

    virtual std::size_t length(const Formating&) const noexcept
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

    // virtual void write
    //     ( boost::stringify::simple_ostream<charT>& out
    //     , const Formating&
    //     ) const
    // {
    //     if(str)
    //     {
    //         out.write(str->c_str(), str->length());
    //     }
    // }
};


template <typename charT, typename Formating, typename traits>
inline
boost::stringify::input_std_string<charT, traits, Formating>
argf(const std::basic_string<charT, traits>& str) noexcept
{
    return str;
}

} // namespace stringify
} // namespace boost


#endif
