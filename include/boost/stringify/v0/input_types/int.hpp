#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/facets/encodings.hpp>
#include <boost/stringify/v0/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/number_of_digits.hpp>
#include <boost/assert.hpp>
#include <cstdint>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template
    < typename IntT
    , typename unsigned_IntT = typename std::make_unsigned<IntT>::type
    >
typename std::enable_if<std::is_signed<IntT>::value, unsigned_IntT>::type
unsigned_abs(IntT value)
{
    return ( value > 0
           ? static_cast<unsigned_IntT>(value)
           : 1 + static_cast<unsigned_IntT>(-(value + 1)));
}

template<typename IntT>
typename std::enable_if<std::is_unsigned<IntT>::value, IntT>::type
unsigned_abs(IntT value)
{
    return value;
}

} // namespace detail

template <class T>
class int_formatting: public stringify::v0::align_formatting<T>
{

    using child_type = typename std::conditional
        < std::is_same<T, void>::value
        , int_formatting<void>
        , T
        > :: type;

public:

    template <typename U>
    friend class int_formatting;
    
    template <typename U>
    using other = stringify::v0::int_formatting<U>;
    
    constexpr int_formatting() = default;

    constexpr int_formatting(const int_formatting&) = default;

    ~int_formatting() = default;

    template <typename U>
    constexpr child_type& format_as(const int_formatting<U> & other) &
    {
        align_formatting<T>::format_as(other);
        m_base = other.m_base;
        m_showbase = other.m_showbase;
        m_showpos = other.m_showpos;
        m_uppercase = other.m_uppercase;
        return static_cast<child_type&>(*this);
    }

    template <typename U>
    constexpr child_type&& format_as(const int_formatting<U> & other) &&
    {
        return static_cast<child_type&&>(format_as(other));
    }
    
    constexpr child_type&& uphex() &&
    {
        m_base = 16;
        m_uppercase = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& uphex() &
    {
        m_base = 16;
        m_uppercase = true;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& hex() &&
    {
        m_base = 16;
        m_uppercase = false;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& hex() &
    {
        m_base = 16;
        m_uppercase = false;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& dec() &&
    {
        m_base = 10;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& dec() &
    {
        m_base = 10;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& oct() &&
    {
        m_base = 8;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& oct() &
    {
        m_base = 8;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator+() &&
    {
        m_showpos = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator+() &
    {
        m_showpos = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type&& operator~() &&
    {
        m_showbase = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator~() &
    {
        m_showbase = true;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& uppercase(bool u) &&
    {
        m_uppercase = u;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& uppercase(bool u) &
    {
        m_uppercase = u;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& showbase(bool s) &&
    {
        m_showbase = s;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& showbase(bool s) &
    {
        m_showbase = s;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& showpos(bool s) &&
    {
        m_showpos = s;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& showpos(bool s) &
    {
        m_showpos = s;
        return static_cast<child_type&>(*this);
    }
    constexpr unsigned base() const
    {
        return m_base;
    }
    constexpr bool showbase() const
    {
        return m_showbase;
    }
    constexpr bool showpos() const
    {
        return m_showpos;
    }
    constexpr bool uppercase() const
    {
        return m_uppercase;
    }

private:

    unsigned short m_base = 10;
    bool m_showbase = false;
    bool m_showpos = false;
    bool m_uppercase = false;

};

template <typename IntT>
class int_with_formatting: public int_formatting<int_with_formatting<IntT> >
{

public:

    //int_with_formatting() = default;

    int_with_formatting(IntT value)
        : m_value(value)
    {
    }

    int_with_formatting(const int_with_formatting&) = default;

    constexpr IntT value() const
    {
        return m_value;
    }

    constexpr void value(IntT v)
    {
        m_value = v;
    }

private:

    IntT m_value = 0;
};


template <typename IntT, typename CharT>
class int_printer: public printer<CharT>
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    static constexpr bool is_signed = std::is_signed<IntT>::value;

public:

    using input_type  = IntT ;
    using char_type   = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;

    template <typename FPack>
    int_printer
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& ft
        , const stringify::v0::int_with_formatting<IntT>& value
        ) noexcept
        : int_printer
            ( out
            , value
            , get_numpunct<8>(ft)
            , get_numpunct<10>(ft)
            , get_numpunct<16>(ft)
            )
    {
    }


    int_printer
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::int_with_formatting<IntT>& value
        , const stringify::v0::numpunct<8>& numpunct_oct
        , const stringify::v0::numpunct<10>& numpunct_dec
        , const stringify::v0::numpunct<16>& numpunct_hex
        ) noexcept;

    virtual ~int_printer();

    std::size_t length() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharT>& m_out;
    boost::stringify::v0::int_with_formatting<IntT> m_input;
    const stringify::v0::numpunct_base& m_numpunct;
    int m_fillcount = 0;

    template <int Base, typename FPack>
    const stringify::v0::numpunct<Base>& get_numpunct(const FPack& ft) noexcept
    {
        using tag = stringify::v0::numpunct_category<Base>;
        return ft.template get_facet<tag, input_type>();
    }

    bool showsign() const
    {
        return is_signed && (m_input.showpos() || m_input.value() < 0);
    }

    unsigned_type unsigned_value() const noexcept
    {
        if(m_input.base() == 10)
        {
            return stringify::v0::detail::unsigned_abs(m_input.value());
        }
        return static_cast<unsigned_type>(m_input.value());
    }

    std::size_t length_fill() const
    {
        if (m_fillcount > 0)
        {
            return m_fillcount * m_out.required_size(m_input.fill());
        }
        return 0;
    }

    std::size_t length_body() const
    {
        switch(m_input.base())
        {
            case  10:
                return length_digits<10>() + (showsign() ? 1 : 0);

            case 16:
                return length_digits<16>() + (m_input.showbase() ? 2 : 0);

            default:
                BOOST_ASSERT(m_input.base() == 8);
                return length_digits<8>() + (m_input.showbase() ? 1 : 0);
        }
    }

    template <unsigned Base>
    std::size_t length_digits() const noexcept
    {
        unsigned num_digits
            = stringify::v0::detail::number_of_digits<Base>(unsigned_value());

        if (unsigned num_seps = m_numpunct.thousands_sep_count(num_digits))
        {
            auto sep_len = m_out.required_size(m_numpunct.thousands_sep());
            return num_digits + num_seps * sep_len;
        }
        return num_digits;
    }

    void write_with_fill() const
    {
        switch(m_input.alignment())
        {
            case stringify::v0::alignment::left:
                write_sign();
                write_base();
                write_digits();
                m_out.put32(m_fillcount, m_input.fill());
                break;

            case stringify::v0::alignment::internal:
                write_sign();
                write_base();
                m_out.put32(m_fillcount, m_input.fill());
                write_digits();
                break;

            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                m_out.put32(halfcount, m_input.fill());
                write_sign();
                write_base();
                write_digits();
                m_out.put32(m_fillcount - halfcount, m_input.fill());
                break;
            }
            default:
                m_out.put32(m_fillcount, m_input.fill());                
                write_sign();
                write_base();
                write_digits();
        }
    }

    void write_sign() const
    {
        if (is_signed && m_input.base() == 10)
        {
            if (m_input.value() < 0)
            {
                m_out.put(CharT('-'));
            }
            else if(showsign())
            {
                m_out.put(CharT('+'));
            }
        }
    }

    void write_without_fill() const
    {
        switch(m_input.base())
        {
            case 16:
                write_base();
                write_digits_t<16>();
                break;

            case  8:
                write_base();
                write_digits_t<8>();
                break;

            default:
                BOOST_ASSERT(m_input.base() == 10);
                write_sign();
                write_digits_t<10>();
        }
    }

    void write_base() const
    {
        if(m_input.base() != 10 && m_input.showbase())
        {
            m_out.put(CharT('0'));
            if(m_input.base() == 16)
            {
                m_out.put(m_input.uppercase() ? CharT('X'): CharT('x'));
            }
        }
    }

    void write_digits() const
    {
        switch (m_input.base())
        {
            case 10:
                write_digits_t<10>();
                break;

            case 16:
                write_digits_t<16>();
                break;

            default:
                BOOST_ASSERT(m_input.base() == 8);
                write_digits_t<8>();
        }
    }

    template <unsigned Base>
    void write_digits_t() const
    {
        constexpr std::size_t buff_size = sizeof(IntT) * 6;
        CharT buff[buff_size];
        CharT* end = &buff[buff_size - 1];
        auto* begin = write_digits_backwards<Base>(unsigned_value(), end);
        unsigned num_digits = static_cast<unsigned>(end - begin);
        if (m_numpunct.thousands_sep_count(num_digits) == 0)
        {
            m_out.put(begin, num_digits);
        }
        else
        {
            write_digits_with_punctuation(begin, num_digits);
        }
    }

    void write_digits_with_punctuation
        ( const CharT* digits
        , unsigned num_digits
        ) const
    {
        char32_t thousands_sep = m_numpunct.thousands_sep();
        unsigned char groups[sizeof(IntT) * 6];
        constexpr std::size_t buff_size = sizeof(IntT) * 12;
        CharT buff[buff_size];
        CharT* it_buff = buff;
        CharT* buff_end = buff + buff_size;
        const auto& encoder = m_out.encoder();

        auto* it_grp = m_numpunct.groups(num_digits, groups);
        while(true)
        {
            unsigned grp_size = *it_grp;
            for(unsigned i = 0; i < grp_size; ++i)
            {
                *it_buff = *digits;
                ++it_buff;
                ++digits;
            }
            if(--it_grp < groups)
            {
                break;
            }
            it_buff = encoder.convert(thousands_sep, it_buff, buff_end, true);
            BOOST_ASSERT(it_buff != nullptr);
            BOOST_ASSERT(it_buff < buff_end);
        }
        m_out.put(buff, it_buff - buff);
    }

    template <unsigned Base, typename OutputIterator>
    OutputIterator*
    write_digits_backwards(unsigned_type value, OutputIterator* it) const
    {
        while(value >= Base)
        {
            *--it = character_of_digit(value % Base);
            value /= Base;
        }
        *--it = character_of_digit(static_cast<unsigned>(value));
        return it;
    }

    CharT character_of_digit(unsigned digit) const noexcept
    {
        if (digit < 10)
        {
            return CharT('0') + digit;
        }
        const CharT char_a = m_input.uppercase() ? 'A' : 'a';
        return  char_a + digit - 10;
    }

    void determinate_fill()
    {
        int content_width = width_body();
        if(m_input.width() < 0)
        {
            m_input.width(0);
        }
        if(content_width < m_input.width())
        {
            m_fillcount = m_input.width() - content_width;
        }
        else
        {
            m_input.width(content_width);
        }
    }

    int width_body() const noexcept
    {
        unsigned num_digits = 0;
        unsigned extra_chars = 0;
        auto uv = unsigned_value();
        switch(m_input.base())
        {
            case 10:
            {
                num_digits = stringify::v0::detail::number_of_digits<10>(uv);
                extra_chars = showsign() ? 1 : 0;
                break;
            }
            case 16:
            {
                num_digits = stringify::v0::detail::number_of_digits<16>(uv);
                extra_chars = m_input.showbase() ? 2 : 0;
                break;
            }
            default:
            {
                BOOST_ASSERT(m_input.base() == 8);
                num_digits = stringify::v0::detail::number_of_digits<8>(uv);
                extra_chars = m_input.showbase() ? 1 : 0;
            }
        }

        unsigned num_separators = m_numpunct.thousands_sep_count(num_digits);
        return num_digits + extra_chars + num_separators;
    }
};

template <typename IntT, typename CharT>
int_printer<IntT, CharT>::int_printer
    ( stringify::v0::output_writer<CharT>& out
    , const stringify::v0::int_with_formatting<IntT>& valuef
    , const stringify::v0::numpunct<8>& numpunct_oct
    , const stringify::v0::numpunct<10>& numpunct_dec
    , const stringify::v0::numpunct<16>& numpunct_hex
    ) noexcept
    : m_out(out)
    , m_input{valuef}
    , m_numpunct
        ( valuef.base() == 10
          ? static_cast<const stringify::v0::numpunct_base&>(numpunct_dec)
          : valuef.base() == 16
          ? static_cast<const stringify::v0::numpunct_base&>(numpunct_hex)
          : static_cast<const stringify::v0::numpunct_base&>(numpunct_oct)
        )
{
    determinate_fill();
}

template <typename IntT, typename CharT>
int_printer<IntT, CharT>::~int_printer()
{
}

template <typename IntT, typename CharT>
std::size_t int_printer<IntT, CharT>::length() const
{
    return length_body() + length_fill();
}

template <typename IntT, typename CharT>
void int_printer<IntT, CharT>::write() const
{
    if (m_fillcount > 0)
    {
        write_with_fill();
    }
    else
    {
        write_without_fill();
    }
}

template <typename IntT, typename CharT>
int int_printer<IntT, CharT>::remaining_width(int w) const
{
    return w > m_input.width() ? (w - m_input.width()) : 0;
}

// namespace detail {

// template <typename IntT>
// struct int_input_traits
// {
//     template <typename CharT, typename FPack>
//     static inline stringify::v0::int_printer<IntT, CharT> make_printer
//         ( const FPack& ft
//         , IntT ch
//         )
//     {
//         return {ft, ch};
//     }

//     template <typename CharT, typename FPack>
//     static inline stringify::v0::int_printer<IntT, CharT> make_printer
//         ( const FPack& ft
//         , const stringify::v0::int_with_formatting<IntT>& ch
//         )
//     {
//         return {ft, ch};
//     }

//     static inline stringify::v0::int_with_formatting<IntT> fmt(IntT ch)
//     {
//         return {ch};
//     }
// };
// }

// stringify::v0::detail::int_input_traits<short>
// stringify_get_input_traits (stringify::v0::int_with_formatting<short>);

// stringify::v0::detail::int_input_traits<int>
// stringify_get_input_traits (stringify::v0::int_with_formatting<int>);

// stringify::v0::detail::int_input_traits<long>
// stringify_get_input_traits (stringify::v0::int_with_formatting<long>);

// stringify::v0::detail::int_input_traits<long long>
// stringify_get_input_traits (stringify::v0::int_with_formatting<long long>);

// stringify::v0::detail::int_input_traits<unsigned short>
// stringify_get_input_traits (stringify::v0::int_with_formatting<unsigned short>);

// stringify::v0::detail::int_input_traits<unsigned int>
// stringify_get_input_traits (stringify::v0::int_with_formatting<unsigned int>);

// stringify::v0::detail::int_input_traits<unsigned long>
// stringify_get_input_traits (stringify::v0::int_with_formatting<unsigned long>);

// stringify::v0::detail::int_input_traits<unsigned long long>
// stringify_get_input_traits (stringify::v0::int_with_formatting<unsigned long long>);

// stringify::v0::detail::int_input_traits<short>
// stringify_get_input_traits(short);

// stringify::v0::detail::int_input_traits<int>
// stringify_get_input_traits(int);

// stringify::v0::detail::int_input_traits<long>
// stringify_get_input_traits(long);

// stringify::v0::detail::int_input_traits<long long>
// stringify_get_input_traits(long long);

// stringify::v0::detail::int_input_traits<unsigned short>
// stringify_get_input_traits(unsigned short);

// stringify::v0::detail::int_input_traits<unsigned int>
// stringify_get_input_traits(unsigned int);

// stringify::v0::detail::int_input_traits<unsigned long>
// stringify_get_input_traits(unsigned long);

// stringify::v0::detail::int_input_traits<unsigned long long>
// stringify_get_input_traits(unsigned long long);


template <typename CharT, typename FPack>
inline stringify::v0::int_printer<short, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , short x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<int, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , int x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<long, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , long x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<long long, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , long long x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned short, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , unsigned short x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned int, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , unsigned int x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned long, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , unsigned long x )
{
    return {out, ft, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned long long, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , unsigned long long x )
{
    return {out, ft, x};
}


template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::int_printer<IntT, CharT>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , const stringify::v0::int_with_formatting<IntT>& x
    )
{
    return {out, ft, x};
}

inline stringify::v0::int_with_formatting<short>
stringify_fmt(short x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<int>
stringify_fmt(int x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<long>
stringify_fmt(long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<long long>
stringify_fmt(long long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned short>
stringify_fmt(unsigned short x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned>
stringify_fmt(unsigned x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned long>
stringify_fmt(unsigned long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned long long>
stringify_fmt(unsigned long long x)
{
    return {x};
}

template <typename> struct is_int_number: public std::false_type {};
template <> struct is_int_number<short>: public std::true_type {};
template <> struct is_int_number<int>: public std::true_type {};
template <> struct is_int_number<long>: public std::true_type {};
template <> struct is_int_number<long long>: public std::true_type {};
template <> struct is_int_number<unsigned short>: public std::true_type {};
template <> struct is_int_number<unsigned int>: public std::true_type {};
template <> struct is_int_number<unsigned long>: public std::true_type {};
template <> struct is_int_number<unsigned long long>: public std::true_type {};


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<long long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<unsigned long long, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
