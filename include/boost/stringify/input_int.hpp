#ifndef BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <boost/stringify/formater_tuple.hpp>
#include <boost/stringify/fmt_showpos.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>
#include <boost/stringify/detail/char_flags.hpp>
#include <boost/stringify/detail/uint_traits.hpp>

#include <boost/logic/tribool.hpp>

namespace boost
{
namespace stringify
{
namespace detail
{

class local_formatting_int
{

public:

    constexpr local_formatting_int(const char* charflags, double d = 0.0) : m_charflags(charflags)
    {
    }

    constexpr local_formatting_int()
    {
    }

    constexpr local_formatting_int(const local_formatting_int&) = default;
    
    local_formatting_int& operator=(const local_formatting_int&) = default;
    
    boost::tribool showpos() const
    {
        if (m_charflags.has_char('-'))
            return false;

        if (m_charflags.has_char('+'))
            return true;
     
        return boost::logic::indeterminate;
    }

private:

    typedef boost::stringify::detail::char_flags<'+', '-'> char_flags_type;
    char_flags_type m_charflags;    
};


} // namespace detail




template <typename intT, typename charT, typename traits, typename Formating>
struct input_int: public boost::stringify::input_base<charT, Formating>
{
private:
    typedef typename std::make_unsigned<intT>::type unsigned_intT;
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
                     : 1 + static_cast<unsigned_intT>(-(value + 1)));
    }

    typedef boost::stringify::detail::local_formatting_int local_formatting;

    
    void set
        ( intT _value
        , local_formatting fmt
        ) noexcept
    {
        set(_value);
        m_local_fmt = fmt;
    }

    virtual std::size_t length(const Formating& fmt) const noexcept
    {
        return uint_traits::number_of_digits(abs_value) + (has_sign(fmt) ? 1 : 0);
    }

    virtual charT* write_without_termination_char
        ( charT* out
        , const Formating& fmt
        ) const noexcept
    {
        out = write_sign(out, fmt);
        return write_digits(out, fmt);
    }
/*
    virtual void write
        ( boost::stringify::simple_ostream<charT>& out
        , const Formating&
        ) const
    {
        if (value < 0)
            out.put(boost::stringify::detail::the_sign_minus<charT>());

        auto div = uint_traits::greatest_power_of_10_less_than(abs_value);
        do
        {
            out.put(character_of_digit((abs_value / div) % 10));
        }
        while(div /= 10);
    }
*/
private:
    intT value;
    unsigned_intT abs_value; // TODO optimaze ( use a union when intT is usigned )
    local_formatting m_local_fmt;
    
    bool has_sign(const Formating& fmt) const noexcept
    {
        /*constexpr*/ if( std::is_signed<intT>::value)
        {
            return value < 0 || showpos(fmt);
        }
        return false;
    }

    charT* write_sign(charT* out, const Formating& fmt) const noexcept
    {
        /*constexpr*/ if( std::is_signed<intT>::value)
        {
            if (value < 0)
            {
                traits::assign(*out++, boost::stringify::detail::the_sign_minus<charT>());
            }
            else if(showpos(fmt))
            {
                traits::assign(*out++, boost::stringify::detail::the_sign_plus<charT>());
            }
        }
        return out;
    }

    charT* write_digits(charT* out, const Formating& fmt) const noexcept
    {
        out += uint_traits::number_of_digits(abs_value);
        unsigned_intT it_value = abs_value;
        charT* end = out;
        do
        {
            traits::assign(*--out, character_of_digit(it_value % 10));
        }
        while(it_value /= 10);
        return end;
    }

    bool showpos(const Formating& fmt) const noexcept
    {
        boost::tribool local_showpos = m_local_fmt.showpos();
        if(indeterminate(local_showpos))
        {
            return fmt.template get<boost::stringify::ftype_showpos, intT>().show();
        }
        return local_showpos;
    }
        
    charT character_of_digit(unsigned int digit) const noexcept
    {
        return boost::stringify::detail::the_digit_zero<charT>() + digit;
    }
};


template <typename charT, typename traits, typename Formating>
inline                                    
boost::stringify::input_int<int, charT, traits, Formating>
argf(int i) noexcept
{                                               
    return i;                                     
}

template <typename charT, typename traits, typename Formating>
inline                                    
boost::stringify::input_int<int, charT, traits, Formating>
argf(int i, const char*) noexcept
{                                               
    return i;                                     
}


template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_int<long, charT, traits, Formating>
argf(long i) noexcept
{
    return i;
}

template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_int<long long, charT, traits, Formating>
argf(long long i) noexcept
{
    return i;
}

template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_int<unsigned int, charT, traits, Formating>
argf(unsigned int i) noexcept
{
    return i;
}

template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_int<unsigned long, charT, traits, Formating>
argf(unsigned long i) noexcept
{
    return i;
}

template <typename charT, typename traits, typename Formating>
inline
boost::stringify::input_int<unsigned long long, charT, traits, Formating>
argf(unsigned long long i) noexcept
{
    return i;
}


}//namespace stringify
}//namespace boost


#endif
