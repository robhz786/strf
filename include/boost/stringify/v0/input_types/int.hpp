#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/facets/encodings.hpp>
#include <boost/stringify/v0/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/int_digits.hpp>
#include <boost/assert.hpp>
#include <cstdint>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <class T>
class int_formatting: public stringify::v0::align_formatting<T>
{

    using child_type = T;

public:

    template <typename U>
    using fmt_other = stringify::v0::int_formatting<U>;

    constexpr int_formatting() = default;

    constexpr int_formatting(const int_formatting&) = default;

    ~int_formatting() = default;

    template <typename U>
    constexpr int_formatting(const int_formatting<U> & u)
        : stringify::v0::align_formatting<T>(u)
        , m_base(u.base())
        , m_showbase(u.showbase())
        , m_showpos(u.showpos())
        , m_uppercase(u.uppercase())
    {
    }

    constexpr child_type&& p(unsigned _) &&
    {
        m_precision = _;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& p(unsigned _) &
    {
        m_precision = _;
        return *this;
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
    constexpr unsigned precision() const
    {
        return m_precision;
    }
    constexpr unsigned short base() const
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

    unsigned m_precision = 0;
    unsigned short m_base = 10;
    bool m_showbase = false;
    bool m_showpos = false;
    bool m_uppercase = false;

};

template <typename IntT>
class int_with_formatting: public int_formatting<int_with_formatting<IntT> >
{

public:

    int_with_formatting(IntT value)
        : m_value(value)
    {
    }

    int_with_formatting(const int_with_formatting&) = default;


    template <typename U>
    int_with_formatting(IntT value, const int_formatting<U>& u)
        : int_formatting<int_with_formatting>(u)
        , m_value(value)
    {
    }

    constexpr IntT value() const
    {
        return m_value;
    }

private:

    IntT m_value = 0;
};


template <typename IntT, typename CharT>
class int_printer: public printer<CharT>
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    static constexpr bool is_signed = std::is_signed<IntT>::value;
    constexpr static unsigned max_digcount = (sizeof(IntT) * 8 + 2) / 3;

public:

    using input_type  = IntT ;
    using char_type   = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;

    template <typename FPack>
    int_printer
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& fp
        , const stringify::v0::int_with_formatting<IntT>& value
        ) noexcept
        : int_printer
            ( out
            , value
            , get_numpunct<8>(fp)
            , get_numpunct<10>(fp)
            , get_numpunct<16>(fp)
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

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    const stringify::v0::numpunct_base* m_punct;
    stringify::v0::output_writer<CharT>& m_out;
    boost::stringify::v0::int_with_formatting<IntT> m_fmt;
    unsigned short m_digcount;
    unsigned short m_sepcount;
    unsigned m_fillcount;

    template <int Base, typename FPack>
    const stringify::v0::numpunct<Base>& get_numpunct(const FPack& fp) noexcept
    {
        using tag = stringify::v0::numpunct_category<Base>;
        return fp.template get_facet<tag, input_type>();
    }

    bool showsign() const
    {
        return is_signed && (m_fmt.showpos() || m_fmt.value() < 0);
    }

    unsigned_type unsigned_value() const noexcept
    {
        if(m_fmt.base() == 10)
        {
            return stringify::v0::detail::unsigned_abs(m_fmt.value());
        }
        return static_cast<unsigned_type>(m_fmt.value());
    }

    std::size_t length_fill() const
    {
        if (m_fillcount > 0)
        {
            return m_fillcount * m_out.necessary_size(m_fmt.fill());
        }
        return 0;
    }

    std::size_t length_body() const
    {
        return length_complement() + length_digits();
    }

    std::size_t length_complement() const noexcept
    {
        if (m_fmt.base() == 10)
        {
            return showsign() ? 1 : 0;
        }
        else if (m_fmt.base() == 16)
        {
            return m_fmt.showbase() ? 2 : 0;
        }
        BOOST_ASSERT(m_fmt.base() == 8);
        return m_fmt.showbase() ? 1 : 0;
    }

    std::size_t length_digits() const noexcept
    {
        auto total_digcount
            = m_fmt.precision() > m_digcount
            ? m_fmt.precision() : m_digcount;
        if (m_sepcount > 0)
        {
            auto sep_len = m_out.necessary_size(m_punct->thousands_sep());
            return total_digcount + m_sepcount * sep_len;
        }
        return total_digcount;
    }

    void write_with_fill() const
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
                write_complement();
                write_digits();
                m_out.put32(m_fillcount, m_fmt.fill());
                break;

            case stringify::v0::alignment::internal:
                write_complement();
                m_out.put32(m_fillcount, m_fmt.fill());
                write_digits();
                break;

            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                m_out.put32(halfcount, m_fmt.fill());
                write_complement();
                write_digits();
                m_out.put32(m_fillcount - halfcount, m_fmt.fill());
                break;
            }
            default:
                m_out.put32(m_fillcount, m_fmt.fill());
                write_complement();
                write_digits();
        }
    }

    void write_complement() const
    {
        if (m_fmt.base() == 10)
        {
            if(is_signed)
            {
                if(m_fmt.value() < 0)
                {
                    m_out.put(CharT('-'));
                }
                else if( m_fmt.showpos())
                {
                    m_out.put(CharT('+'));
                }
            }
        }
        else if (m_fmt.showbase())
        {
            m_out.put(CharT('0'));
            if(m_fmt.base() == 16)
            {
                m_out.put(m_fmt.uppercase() ? CharT('X'): CharT('x'));
            }
        }
    }

    void write_digits() const
    {
        if(m_fmt.precision() > m_digcount)
        {
            m_out.put(m_fmt.precision() - m_digcount, CharT('0'));
        }
        if (m_sepcount == 0)
        {
            write_digits_nosep();
        }
        else
        {
            write_digits_sep();
        }
    }

    void write_digits_nosep() const
    {
        CharT dig_buff[max_digcount];
        CharT* dig_it = stringify::v0::detail::write_int_txtdigits_backwards
            ( m_fmt.value()
            , m_fmt.base()
            , m_fmt.uppercase()
            , dig_buff + max_digcount );
        BOOST_ASSERT(dig_it + m_digcount == dig_buff + max_digcount);
        m_out.put(dig_it, m_digcount);
    }

    void write_digits_sep() const
    {
        const auto& encoder = m_out.encoder();
        auto sep = m_out.validate(m_punct->thousands_sep());
        if(sep.error_emitted)
        {
            return;
        }
        if(sep.size == 0)
        {
            write_digits_nosep();
            return;
        }

        char dig_buff[max_digcount];
        char* dig_it = stringify::v0::detail::write_int_txtdigits_backwards
            ( m_fmt.value()
            , m_fmt.base()
            , m_fmt.uppercase()
            , dig_buff + max_digcount );

        unsigned char grp_buff[max_digcount];
        auto* grp_it = m_punct->groups(m_digcount, grp_buff);

        if (sep.size == 1)
        {
            CharT sep_ch;
            auto t = encoder.encode
                ( sep.ch, &sep_ch, &sep_ch + 1, m_out.allow_surrogates() );
            BOOST_ASSERT(t != &sep_ch + 2);
            BOOST_ASSERT(t != nullptr);
            (void) t;
            write_digits_littlesep(dig_it, grp_buff, grp_it, sep_ch);
        }
        else
        {
            write_digits_bigsep
                ( dig_it
                , grp_buff
                , grp_it
                , sep.ch
                , static_cast<unsigned>(sep.size) );
        }
    }

    void write_digits_littlesep
        ( char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , CharT sep ) const
    {
        CharT buff[max_digcount * 2];
        CharT* it = buff;
        for(unsigned i = *grp_it; i != 0; --i)
        {
            *it++ = *dig_it++;
        }
        do
        {
            *it++ = sep;
            for(unsigned i = *--grp_it; i != 0; --i)
            {
                *it++ = *dig_it++;
            }
        }
        while(grp_it > grp);
        BOOST_ASSERT(buff + m_digcount + m_sepcount == it);
        m_out.put(buff, m_digcount + m_sepcount);
    }

    void write_digits_bigsep
        ( char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , char32_t sep_char
        , unsigned sep_char_size ) const
    {
        stringify::v0::detail::intdigits_writer<IntT, CharT> writer
            { dig_it
            , grp
            , grp_it
            , m_out.encoder()
            , sep_char
            , sep_char_size };

        m_out.put(writer);
    }
};

template <typename IntT, typename CharT>
int_printer<IntT, CharT>::int_printer
    ( stringify::v0::output_writer<CharT>& out
    , const stringify::v0::int_with_formatting<IntT>& fmt
    , const stringify::v0::numpunct<8>& numpunct_oct
    , const stringify::v0::numpunct<10>& numpunct_dec
    , const stringify::v0::numpunct<16>& numpunct_hex
    ) noexcept
    : m_out(out)
    , m_fmt{fmt}
{
    auto extra_chars_count = 0;
    if (fmt.base() == 10)
    {
        m_punct = & numpunct_dec;
        m_digcount = stringify::v0::detail::count_digits<10>(fmt.value());
        m_sepcount = m_punct->thousands_sep_count(m_digcount);
        if(showsign())
        {
            extra_chars_count = 1;
        }
    }
    else if (fmt.base() == 16)
    {
        m_punct = & numpunct_hex;
        m_digcount = stringify::v0::detail::count_digits<16>(fmt.value());
        if(fmt.showbase())
        {
            extra_chars_count = 2;
        }
    }
    else
    {
        BOOST_ASSERT(fmt.base() == 8);
        m_digcount = stringify::v0::detail::count_digits<8>(fmt.value());
        m_punct = & numpunct_oct;
        if(fmt.showbase())
        {
            extra_chars_count = 1;
        }
    }
    m_sepcount = m_punct->thousands_sep_count(m_digcount);
    m_fillcount = 0;
    int content_width
        = static_cast<int>(fmt.precision() > m_digcount ? fmt.precision() : m_digcount)
        + static_cast<int>(m_sepcount)
        + extra_chars_count;

    if (m_fmt.width() > content_width)
    {
        m_fillcount = m_fmt.width() - content_width;
    }
    else
    {
        m_fmt.width(content_width);
        m_fillcount = 0;
    }

    BOOST_ASSERT(m_digcount <= max_digcount);
    BOOST_ASSERT(m_sepcount <= max_digcount);
}

template <typename IntT, typename CharT>
int_printer<IntT, CharT>::~int_printer()
{
}

template <typename IntT, typename CharT>
std::size_t int_printer<IntT, CharT>::necessary_size() const
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
        write_complement();
        write_digits();
    }
}

template <typename IntT, typename CharT>
int int_printer<IntT, CharT>::remaining_width(int w) const
{
    return w > m_fmt.width() ? (w - m_fmt.width()) : 0;
}


template <typename CharT, typename FPack>
inline stringify::v0::int_printer<short, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , short x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<int, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , int x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<long, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , long x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<long long, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , long long x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned short, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , unsigned short x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned int, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , unsigned int x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned long, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , unsigned long x )
{
    return {out, fp, x};
}
template <typename CharT, typename FPack>
inline stringify::v0::int_printer<unsigned long long, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , unsigned long long x )
{
    return {out, fp, x};
}


template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::int_printer<IntT, CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , const stringify::v0::int_with_formatting<IntT>& x
    )
{
    return {out, fp, x};
}

inline stringify::v0::int_with_formatting<short>
make_fmt(stringify::v0::tag, short x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<int>
make_fmt(stringify::v0::tag, int x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<long>
make_fmt(stringify::v0::tag, long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<long long>
make_fmt(stringify::v0::tag, long long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned short>
make_fmt(stringify::v0::tag, unsigned short x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned>
make_fmt(stringify::v0::tag, unsigned x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned long>
make_fmt(stringify::v0::tag, unsigned long x)
{
    return {x};
}
inline stringify::v0::int_with_formatting<unsigned long long>
make_fmt(stringify::v0::tag, unsigned long long x)
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
