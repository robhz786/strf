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

namespace boost {
namespace stringify {
namespace detail {

struct int_arg_format
{
    typedef boost::stringify::width_t width_t;
    typedef boost::stringify::detail::char_flags
        <'+', '-', '<', '>', 'o', 'd', 'x', 'X', 'c', 'C', '#', '$'>
        char_flags_type;
    
    constexpr int_arg_format(const char* flags, width_t w = -1
        )
        : m_flags(flags)
        , m_width(w)  
    {
    }

    constexpr int_arg_format(boost::stringify::width_t w)
        : m_width(w)  
    {
    }
    
    constexpr int_arg_format()
    {
    }

    constexpr int_arg_format(const int_arg_format&) = default;
    
    int_arg_format& operator=(const int_arg_format&) = default;

    template <typename IntputType, typename FTuple>
    bool get_showpos(const FTuple& fmt) const noexcept
    {
        if (m_flags.has_char('-'))
        {
            return false;
        }
        else if (m_flags.has_char('+'))
        {
            return true;
        }
        else
        {
            return boost::stringify::get_showpos<IntputType>(fmt);
        }
    }
    
    char_flags_type m_flags;
    boost::stringify::width_t m_width = -1;
};


template
    < typename intT
    , typename unsigned_intT = typename std::make_unsigned<int>::type
    >
typename std::enable_if<std::is_signed<intT>::value, unsigned_intT>::type
unsigned_abs(intT value)
{
    return ( value > 0
           ? static_cast<unsigned_intT>(value)
           : 1 + static_cast<unsigned_intT>(-(value + 1)));
}


template<typename intT>
typename std::enable_if<std::is_unsigned<intT>::value, intT>::type
unsigned_abs(intT value)
{
    return value;
}


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

    typedef intT input_type;
    typedef CharT char_type;
    typedef Output output_type;
    typedef Formatting ftuple_type;
    typedef boost::stringify::detail::int_arg_format arg_format_type;
    
    int_stringifier(const Formatting& fmt, intT value) noexcept
        : m_fmt(fmt)
        , m_value(value)
        , m_showpos(boost::stringify::get_showpos<input_type>(fmt))  
    {
    }

    int_stringifier(const Formatting& fmt, intT value, arg_format_type argf) noexcept
        : m_fmt(fmt)
        , m_value(value)
        , m_showpos(argf.get_showpos<input_type>(fmt))
    {
    }
   
    virtual std::size_t length() const noexcept override
    {
        return length_digits() + (has_sign() ? 1 : 0);
    }

    virtual void write(Output& out) const noexcept(noexcept_output) override
    {
        write_sign(out);
        write_digits(out);
    }

private:
   
    const Formatting& m_fmt;
    const intT m_value;
    const bool m_showpos;
    
    virtual std::size_t length_digits() const noexcept
    {
        return uint_traits::number_of_digits(unsigned_abs(m_value));
    }
    
    bool has_sign() const noexcept
    {
        if( std::is_signed<intT>::value)
        {
            return m_value < 0 || m_showpos;
        }
        return false;
    }

    void write_sign(Output& out) const noexcept(noexcept_output)
    {
        if( std::is_signed<intT>::value)
        {
            if (m_value < 0)
            {
                out.put(boost::stringify::detail::the_sign_minus<CharT>());
            }
            else if(m_showpos)
            {
                out.put(boost::stringify::detail::the_sign_plus<CharT>());
            }
        }
    }
    
    void write_digits(Output& out) const noexcept(noexcept_output)
    {
        boost::stringify::detail::int_digits<unsigned_intT, 10>
            digits(unsigned_abs(m_value));
        while(! digits.empty())
        {
            out.put(character_of_digit(digits.pop()));
        }
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
