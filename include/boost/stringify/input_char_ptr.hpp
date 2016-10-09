#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/input_base.hpp>

namespace boost
{
namespace stringify
{

template<typename charT, typename traits, typename Formating>
struct input_char_ptr: boost::stringify::input_base<charT, Formating>
{
private:

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

    virtual charT* write_without_termination_char
        ( charT* out
        , const Formating& fmt
        ) const noexcept
    {
        auto fillwidth = fill_width(fmt);
        if (fillwidth > 0)
        {
            out = write_fill(out, fillwidth, fmt);
        }
        return traits::copy(out, str, str_length()) + str_length();
    }

    // virtual void write
    //     ( boost::stringify::simple_ostream<charT>& out
    //     , const Formating&
    //     ) const
    // {
    //     if(str)
    //     {
    //         out.write(str, str_length());
    //     }
    // }


private:
    const charT* str;
    mutable std::size_t len;

    std::size_t str_length() const noexcept
    {
        if (len == (std::numeric_limits<std::size_t>::max) ())
        {
            len = traits::length(str);
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
    
    charT* write_fill
        ( charT* out
        , boost::stringify::width_t fillwidth
        , const Formating& fmt
        )  const noexcept
    {
        return fmt_fill(fmt)
            .template fill<traits, width_accumulator_type>(out, fillwidth);
    }
    
    boost::stringify::width_t
    fill_width(const Formating& fmt) const noexcept
    {
        auto width = fmt_width(fmt).width();
        if (width > 0)
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

template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_char_ptr<charT, traits, Formating>
argf(const charT* str) noexcept
{
    return str;
}


} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_INPUT_CHAR_PTR */

