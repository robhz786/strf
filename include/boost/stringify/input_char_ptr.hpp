#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template<typename charT, typename Output, typename Formating>
struct input_char_ptr: boost::stringify::input_base<charT, Output, Formating>
{
    typedef boost::stringify::input_base<charT, Output, Formating> base;
    
    typedef
        typename Formating::template fimpl_type
            < boost::stringify::ftype_width_calculator<char>
            , const char*
            >
        width_calculator_type;


    typedef
        typename width_calculator_type::accumulator_type
        width_accumulator_type;

public:

    input_char_ptr() noexcept
        : str(0)
        , len(0)
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

    virtual std::size_t length(const Formating& fmt) const noexcept
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
            write_fill(out, fillwidth, fmt);
        }

        if(str)
        {
            out.put(str, str_length());
        }
    }


private:
    const charT* str;
    mutable std::size_t len;

    std::size_t str_length() const noexcept
    {
        if (len == (std::numeric_limits<std::size_t>::max) ())
        {
            len = std::char_traits<charT>::length(str);
        }
        return len;
    }

    std::size_t fill_length(const Formating& fmt) const noexcept
    {
        auto fillwidth = fill_width(fmt);
        if (fillwidth > 0)
        {
            return fmt_fill(fmt)
                .template length<width_accumulator_type>(fillwidth);
        }
        return 0;
    }
    
    void write_fill
        ( Output& out
        , boost::stringify::width_t fillwidth
        , const Formating& fmt
        )  const noexcept
    {
        return fmt_fill(fmt)
            .template fill<Output, width_accumulator_type>(out, fillwidth);
    }
   
    boost::stringify::width_t
    fill_width(const Formating& fmt) const noexcept
    {
        auto width = fmt_width(fmt).width();
        if(width > 0)
        {
            width_accumulator_type acc;
            if(acc.add(str, str_length(), width))
            {
                return width - acc.result();
            }
        }
        return 0;
    }

    decltype(auto) fmt_width(const Formating& fmt) const noexcept
    {
        return fmt.template get<boost::stringify::ftype_width, const char*>();
    }

    decltype(auto) fmt_fill(const Formating& fmt) const noexcept
    {
        return fmt.template get<boost::stringify::ftype_fill<char>, const char*>();
    }
    
}; 

template <typename charT, typename Output, typename Formating>
inline
boost::stringify::input_char_ptr<charT, Output, Formating>
argf(const charT* str) noexcept
{
    return str;
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_INPUT_CHAR_PTR */

