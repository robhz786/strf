#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template<typename CharT, typename Output, typename Formating>
struct input_char_ptr: boost::stringify::input_base<CharT, Output, Formating>
{
    typedef const CharT* input_type;

    typedef boost::stringify::input_base<CharT, Output, Formating> base;
    
    typedef
        boost::stringify::width_accumulator<Formating, input_type, CharT>
        width_accumulator_type;
    
public:
   
    input_char_ptr() noexcept
        : str(0)
        , len(0)
    {
    }

    input_char_ptr(const CharT* _str) noexcept
        : str(_str)
        , len((std::numeric_limits<std::size_t>::max) ())
    {
    }

    void set(const CharT* _str) noexcept
    {
        str = _str;
        len = (std::numeric_limits<std::size_t>::max) ();
    }

    virtual std::size_t length(const Formating& fmt) const noexcept override
    {
        return str_length() + fill_length(fmt);
    }

    void write
        ( Output& out
        , const Formating& fmt
        ) const noexcept(base::noexcept_output) override
    {
        auto fillwidth = fill_width(fmt);
        if (fillwidth > 0)
        {
            boost::stringify::get_filler<CharT, input_type>(fmt)
                .template fill<width_accumulator_type>(out, fillwidth);
        }

        if(str)
        {
            out.put(str, str_length());
        }
    }


private:
    const CharT* str;
    mutable std::size_t len;

    std::size_t str_length() const noexcept
    {
        if (len == (std::numeric_limits<std::size_t>::max) ())
        {
            len = std::char_traits<CharT>::length(str);
        }
        return len;
    }

    std::size_t fill_length(const Formating& fmt) const noexcept
    {
        auto fillwidth = fill_width(fmt);
        if (fillwidth > 0)
        {
            return boost::stringify::get_filler<CharT, input_type>(fmt)
                .template length<width_accumulator_type>(fillwidth);
        }
        return 0;
    }

    boost::stringify::width_t fill_width(const Formating& fmt) const noexcept
    {
        auto total_width = boost::stringify::get_width<input_type>(fmt);
        if(total_width > 0)
        {
            width_accumulator_type acc;
            if(acc.add(str, str_length(), total_width))
            {
                boost::stringify::width_t nonfill_width = acc.result();
                if (nonfill_width < total_width)
                {
                    return total_width - nonfill_width;
                }
            }
        }
        return 0;
    }
}; 

template <typename CharT, typename Output, typename Formating>
inline
boost::stringify::input_char_ptr<CharT, Output, Formating>
argf(const CharT* str) noexcept
{
    return str;
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_INPUT_CHAR_PTR */

