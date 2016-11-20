#ifndef BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED

#include <string>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template <class CharT, typename Traits, typename Output, class Formating>
class input_std_string: public boost::stringify::input_base<CharT, Output, Formating>
{
    typedef boost::stringify::input_base<CharT, Output, Formating> base;
    
public:
    
    const std::basic_string<CharT, Traits>* str;    

    input_std_string() noexcept
        : str(0)
    {
    }

    input_std_string(const std::basic_string<CharT, Traits>& _str) noexcept
        : str(&_str)
    {
    }

    void set(const std::basic_string<CharT, Traits>& _str) noexcept
    {
        str = &_str;
    }

    virtual std::size_t length(const Formating&) const noexcept override
    {
        return str ? str->length() : 0;
    }

    void write
        ( Output& out
        , const Formating&
        ) const noexcept(base::noexcept_output) override
    {
        if(str)
        {
            out.put(str->c_str(), str->length());
        }
    }
};


template <typename CharT, typename Traits, typename Output, typename Formating>
inline
boost::stringify::input_std_string<CharT, Traits, Output, Formating>
argf(const std::basic_string<CharT, Traits>& str) noexcept
{
    return str;
}

} // namespace stringify
} // namespace boost


#endif
