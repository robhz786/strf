#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template<typename charT, typename Formating>
struct input_char_ptr: boost::stringify::input_base<charT, Formating>
{
    input_char_ptr() noexcept:
                       str(0),
                       len(0)
    {
    }

    input_char_ptr(const charT* _str) noexcept
        : str(_str)
        , len((std::numeric_limits<std::size_t>::max) ())
    {
    }

    void set(const charT* _str) noexcept
    {
        str = _str;
        len = (std::numeric_limits<std::size_t>::max) ();
    }

    virtual std::size_t length(const Formating&) const noexcept
    {
        return get_length();
    }

    virtual charT* write_without_termination_char(charT* out, const Formating&) const noexcept
    {
        return std::copy(str, str + get_length(), out);
    }

    // virtual void write
    //     ( boost::stringify::simple_ostream<charT>& out
    //     , const Formating&
    //     ) const
    // {
    //     if(str)
    //     {
    //         out.write(str, get_length());
    //     }
    // }


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
boost::stringify::input_char_ptr<charT, Formating> argf(const charT* str) noexcept
{
    return str;
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_INPUT_CHAR_PTR */

