#ifndef BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/stringifier.hpp>
#include <boost/stringify/ftuple.hpp>
#include <boost/stringify/custom_showpos.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>
#include <boost/stringify/detail/char_flags.hpp>
#include <boost/stringify/detail/uint_traits.hpp>
#include <boost/stringify/detail/int_digits.hpp>

#include <boost/logic/tribool.hpp>

namespace boost {
namespace stringify {
namespace detail {

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


template <typename intT, typename CharT, typename Output, typename Formatting>
struct int_stringifier
    : public boost::stringify::stringifier<CharT, Output, Formatting>
{
    typedef boost::stringify::stringifier<CharT, Output, Formatting> base;
    typedef typename std::make_unsigned<intT>::type unsigned_intT;
    typedef boost::stringify::detail::uint_traits<unsigned_intT> uint_traits;
    using base::noexcept_output;
    using base::random_access_output;
public:

    int_stringifier() noexcept
        : m_value(0)
        , m_abs_value(0)
    {
    }

    int_stringifier(intT _value) noexcept
    {
        set(_value);
    }

    void set(intT _value) noexcept
    {
        m_value = (_value);
        m_abs_value = (m_value > 0
                     ? static_cast<unsigned_intT>(m_value)
                     : 1 + static_cast<unsigned_intT>(-(m_value + 1)));
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

    virtual std::size_t length(const Formatting& fmt) const noexcept override
    {
        return length_digits() + (has_sign(fmt) ? 1 : 0);
    }

    virtual void write
        ( Output& out
        , const Formatting& fmt
        ) const noexcept(noexcept_output) override
    {
        write_sign(out, fmt);
        write_digits(out, fmt);
    }

private:

    virtual std::size_t length_digits() const noexcept
    {
        return uint_traits::number_of_digits(m_abs_value);
    }
    

    intT m_value;
    unsigned_intT m_abs_value; // TODO optimaze ( use a union when intT is usigned )
    local_formatting m_local_fmt;
    
    bool has_sign(const Formatting& fmt) const noexcept
    {
        /*constexpr*/ if( std::is_signed<intT>::value)
        {
            return m_value < 0 || showpos(fmt);
        }
        return false;
    }

    void write_sign(Output& out, const Formatting& fmt) const noexcept(noexcept_output)
    {
        /*constexpr*/ if( std::is_signed<intT>::value)
        {
            if (m_value < 0)
            {
                out.put(boost::stringify::detail::the_sign_minus<CharT>());
            }
            else if(showpos(fmt))
            {
                out.put(boost::stringify::detail::the_sign_plus<CharT>());
            }
        }
    }
    
    void write_digits(Output& out, const Formatting& fmt) const noexcept(noexcept_output)
    {
        boost::stringify::detail::int_digits<unsigned_intT, 10> digits(m_abs_value);
        while(! digits.empty())
        {
            out.put(character_of_digit(digits.pop()));
        }
    }

    bool showpos(const Formatting& fmt) const noexcept
    {
        boost::tribool local_showpos = m_local_fmt.showpos();
        if(indeterminate(local_showpos))
        {
            return boost::stringify::get_showpos<intT>(fmt);
        }
        return local_showpos;
    }
        
    CharT character_of_digit(unsigned int digit) const noexcept
    {
        if (digit < 10)
        {
            return boost::stringify::detail::the_digit_zero<CharT>() + digit;
        }
        return boost::stringify::detail::the_character_a<CharT>() + digit - 10;
    }
};


template <typename IntT>
struct int_input_traits
{
    template <typename CharT, typename Output, typename Formatting>
    using stringifier =
        boost::stringify::detail::int_stringifier<IntT, CharT, Output, Formatting>;
};

} // namespace detail

boost::stringify::detail::int_input_traits<short>
boost_stringify_input_traits_of(short);

boost::stringify::detail::int_input_traits<int>
boost_stringify_input_traits_of(int);

boost::stringify::detail::int_input_traits<long>
boost_stringify_input_traits_of(long);

boost::stringify::detail::int_input_traits<long long>
boost_stringify_input_traits_of(long long);

boost::stringify::detail::int_input_traits<unsigned short>
boost_stringify_input_traits_of(unsigned short);

boost::stringify::detail::int_input_traits<unsigned>
boost_stringify_input_traits_of(unsigned);

boost::stringify::detail::int_input_traits<unsigned long>
boost_stringify_input_traits_of(unsigned long);

boost::stringify::detail::int_input_traits<unsigned long long>
boost_stringify_input_traits_of(unsigned long long);

}//namespace stringify
}//namespace boost


#endif
