#ifndef BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>
#include <boost/stringify/detail/uint_traits.hpp>

namespace boost
{
namespace stringify
{

template <typename intT, typename charT, typename Formating>
struct input_int: public boost::stringify::input_base<charT, Formating>
{
private:
    typedef typename std::make_unsigned<intT>::type  unsigned_intT;
    typedef boost::stringify::detail::uint_traits<unsigned_intT> uint_traits;

public:

    input_int() noexcept
        : value(0)
        , abs_value(0)
    {
    }

    input_int(intT _value) noexcept
    {
        set(_value);
    }

    void set(intT _value) noexcept
    {
        value = (_value);
        abs_value = (value > 0
                     ? static_cast<unsigned_intT>(value)
                     : static_cast<unsigned_intT>(-(value+1)) +1 );
    }


    virtual std::size_t minimal_length(const Formating&) const noexcept
    {
        return
            ( uint_traits::number_of_digits(static_cast<unsigned_intT>(value))
            + (value < 0 ? 1 : 0)
            );
    }

    virtual charT* write_without_termination_char
        ( charT* out
        , const Formating&
        ) const noexcept
    {
        if (value < 0)
            *out++ = boost::stringify::detail::the_sign_minus<charT>();

        out += uint_traits::number_of_digits(abs_value);
        unsigned_intT it_value = abs_value;
        charT* end = out;
        do
        {
            *--out = correspondig_character_of_digit(it_value % 10);
        }
        while(it_value /= 10);

        return end;
    }

    virtual void write(boost::stringify::simple_ostream<charT>& out, const Formating&) const
    {
        if (value < 0)
            out.put(boost::stringify::detail::the_sign_minus<charT>());

        auto div{uint_traits::greatest_power_of_10_less_than(abs_value)};
        do
        {
            out.put
                ( correspondig_character_of_digit((abs_value / div) % 10)
                );
        }
        while(div /= 10);
    }

private:
    intT value;
    unsigned_intT abs_value;

    charT correspondig_character_of_digit(unsigned int digit) const noexcept
    {
        return boost::stringify::detail::the_digit_zero<charT>() + digit;
    }
};


template <typename charT, typename Formating>
inline                                    
boost::stringify::input_int<int, charT, Formating>
argf(int i) noexcept
{                                               
    return i;                                     
}

template <typename charT, typename Formating>
inline                                    
boost::stringify::input_int<int, charT, Formating>
argf(int i, const char*) noexcept
{                                               
    return i;                                     
}


template <typename charT, typename Formating>
inline
boost::stringify::input_int<long, charT, Formating>
argf(long i) noexcept
{
    return i;
}

template <typename charT, typename Formating>
inline
boost::stringify::input_int<long long, charT, Formating>
argf(long long i) noexcept
{
    return i;
}

template <typename charT, typename Formating>
inline
boost::stringify::input_int<unsigned int, charT, Formating>
argf(unsigned int i) noexcept
{
    return i;
}

template <typename charT, typename Formating>
inline
boost::stringify::input_int<unsigned long, charT, Formating>
argf(unsigned long i) noexcept
{
    return i;
}

template <typename charT, typename Formating>
inline
boost::stringify::input_int<unsigned long long, charT, Formating>
argf(unsigned long long i) noexcept
{
    return i;
}


}//namespace stringify
}//namespace boost


#endif
