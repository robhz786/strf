#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/conventional_argf_reader.hpp>
#include <boost/stringify/v0/char_flags.hpp>
#include <boost/stringify/v0/facets/alignment.hpp>
#include <boost/stringify/v0/facets/fill.hpp>
#include <boost/stringify/v0/facets/intbase.hpp>
#include <boost/stringify/v0/facets/case.hpp>
#include <boost/stringify/v0/facets/encoder.hpp>
#include <boost/stringify/v0/facets/showbase.hpp>
#include <boost/stringify/v0/facets/showpos.hpp>
#include <boost/stringify/v0/facets/width.hpp>
#include <boost/stringify/v0/facets/width.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/detail/characters_catalog.hpp>
#include <boost/stringify/v0/detail/number_of_digits.hpp>
#include <boost/stringify/v0/formatter.hpp>
#include <boost/assert.hpp>
#include <cstdint>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

struct int_argf
{
    using char_flags_type = stringify::v0::char_flags
        <'+', '-', '<', '>', '=', '^', 'o', 'd', 'x', 'X', 'c', 'C', '#', '$'>;

    constexpr int_argf(int w): width(w) {}
    constexpr int_argf(const char* f): flags(f) {}
    constexpr int_argf(int w, const char* f): width(w), flags(f) {}
    constexpr int_argf(const int_argf&) = default;

    int width = -1;
    char_flags_type flags;
};


template
    < typename intT
    , typename unsigned_intT = typename std::make_unsigned<intT>::type
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


template <typename intT, typename CharT>
class int_formatter: public formatter<CharT>
{
    using unsigned_intT = typename std::make_unsigned<intT>::type;
    using chars_catalog = stringify::v0::detail::characters_catalog;
    using from32_tag = stringify::v0::encoder_tag<CharT>;
    static constexpr bool is_signed = std::is_signed<intT>::value;

public:

    using input_type  = intT ;
    using char_type   = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;
    using second_arg = stringify::v0::detail::int_argf;

private:

    using argf_reader = stringify::v0::conventional_argf_reader<input_type>;

public:

    template <typename FTuple>
    int_formatter(const FTuple& ft, intT value) noexcept
        : m_value{value}
        , m_encoder{get_facet<stringify::v0::encoder_tag<CharT>>(ft)}
        , m_width{get_facet<stringify::v0::width_tag>(ft).width()}
        , m_fillchar{get_facet<stringify::v0::fill_tag>(ft).fill_char()}
        , m_alignment{get_facet<stringify::v0::alignment_tag>(ft).value()}
        , m_base(get_facet<stringify::v0::intbase_tag>(ft).value())
        , m_showpos
            { is_signed && value >= 0 &&
              get_facet<stringify::v0::showpos_tag>(ft).value() }
        , m_showbase{get_facet<stringify::v0::showbase_tag>(ft).value()}
        , m_uppercase{get_facet<stringify::v0::case_tag>(ft).uppercase()}
    {
        if(m_width > 0)
        {
            determinate_fill();
        }
    }

    template <typename FTuple>
    int_formatter(const FTuple& ft, intT value, second_arg argf) noexcept
        : m_value{value}
        , m_encoder{get_facet<stringify::v0::encoder_tag<CharT>>(ft)}
        , m_width{argf_reader::get_width(argf, ft)}
        , m_fillchar{get_facet<stringify::v0::fill_tag>(ft).fill_char()}
        , m_base(argf_reader::get_base(argf, ft))
        , m_showpos{is_signed && value >= 0 && argf_reader::get_showpos(argf, ft)}
        , m_showbase{argf_reader::get_showbase(argf, ft)}
        , m_uppercase{argf_reader::get_uppercase(argf, ft)}
    {
        if(m_width > 0)
        {
            m_alignment = argf_reader::get_alignment(argf, ft);
            determinate_fill();
        }
    }

    std::size_t length() const override
    {
        return length_body() + length_fill();
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount > 0)
        {
            write_with_fill(out);
        }
        else
        {
            write_without_fill(out);
        }
    }

    int remaining_width(int w) const override
    {
        if(m_width <= 0)
        {
            m_width = width_body();
        }
        return w > m_width ? w - m_width : 0;
    }

private:

    const intT m_value;
    const stringify::v0::encoder<CharT>& m_encoder;
    mutable int m_width;
    char32_t m_fillchar = U' ';
    int m_fillcount = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
    unsigned short m_base;
    bool m_showpos;
    bool m_showbase;
    bool m_uppercase;


    template <typename FacetCategory, typename FTuple>
    static const auto& get_facet(const FTuple& ft) noexcept
    {
        return ft.template get_facet<FacetCategory, input_type>();
    }

    unsigned_intT unsigned_value() const noexcept
    {
        if(m_base == 10)
        {
            return stringify::v0::detail::unsigned_abs(m_value);
        }
        return static_cast<unsigned_intT>(m_value);
    }

    std::size_t length_fill() const
    {
        if (m_fillcount > 0)
        {
            return m_fillcount * m_encoder.length(m_fillchar);
        }
        return 0;
    }

    std::size_t length_body() const
    {
        switch(m_base)
        {
            case 16:
                return length_digits<16>() + (m_showbase ? 2 : 0);
            case  8:
                return length_digits<8>() + (m_showbase ? 1 : 0);
        }
        BOOST_ASSERT(m_base == 10);
        return length_digits<10>() + (m_value < 0 || m_showpos ? 1 : 0);
    }

    template <unsigned Base>
    std::size_t length_digits() const noexcept
    {
        return stringify::v0::detail::number_of_digits<Base>(unsigned_value());
    }

    void write_with_fill(writer_type& out) const
    {
        switch (m_alignment)
        {
            case stringify::v0::alignment::left:
                write_sign(out);
                write_base(out);
                write_digits(out);
                write_fill(out, m_fillcount);
                break;

            case stringify::v0::alignment::right:
                write_fill(out, m_fillcount);
                write_sign(out);
                write_base(out);
                write_digits(out);
                break;


            case stringify::v0::alignment::internal:
                write_sign(out);
                write_base(out);
                write_fill(out, m_fillcount);
                write_digits(out);
                break;

            default:
            {
                BOOST_ASSERT(m_alignment == stringify::v0::alignment::center);
                auto halfcount = m_fillcount / 2;
                write_fill(out, halfcount);
                write_sign(out);
                write_base(out);
                write_digits(out);
                write_fill(out, m_fillcount - halfcount);
            }
        }
    }

    void write_sign(writer_type& out) const
    {
        if (std::is_signed<intT>::value && m_base == 10)
        {
            if (m_value < 0)
            {
                out.put(chars_catalog::minus<CharT>());
            }
            else if(m_showpos)
            {
                out.put(chars_catalog::plus<CharT>());
            }
        }
    }

    void write_without_fill(writer_type& out) const
    {
        switch(m_base)
        {
            case 16:
                write_base(out);
                write_digits_t<16>(out);
                break;

            case  8:
                write_base(out);
                write_digits_t<8>(out);
                break;

            default:
                BOOST_ASSERT(m_base == 10);
                write_sign(out);
                write_digits_t<10>(out);
        }
    }

    void write_base(writer_type& out) const
    {
        if(m_showbase)
        {
            if (m_base == 16)
            {
                out.put(chars_catalog::zero<CharT>());
                if (m_uppercase)
                {
                    out.put(chars_catalog::X<CharT>());
                }
                else
                {
                    out.put(chars_catalog::x<CharT>());
                }
            }
            else if(m_base == 8)
            {
                out.put(chars_catalog::zero<CharT>());
            }
        }
    }

    void write_digits(writer_type& out) const
    {
        switch (m_base)
        {
            case 10 : write_digits_t<10>(out); break;
            case 16 : write_digits_t<16>(out); break;
            default:
                BOOST_ASSERT(m_base == 8);
                write_digits_t<8>(out);
        }
    }

    template <unsigned Base>
    void write_digits_t(writer_type& out) const
    {
        constexpr std::size_t buff_size = sizeof(intT) * 6;
        CharT buff[buff_size];
        CharT* end = &buff[buff_size - 1];
        auto begin = write_digits_backwards<Base>(unsigned_value(), end);
        out.put(begin, (end - begin));
    }

    template <unsigned Base, typename OutputIterator>
    OutputIterator*
    write_digits_backwards(unsigned_intT value, OutputIterator* it) const
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
            return chars_catalog::zero<CharT>() + digit;
        }
        else if (m_uppercase)
        {
            return chars_catalog::A<CharT>() + digit - 10;
        }
        return chars_catalog::a<CharT>() + digit - 10;
    }

    void determinate_fill()
    {
        int content_width = width_body();
        if(content_width < m_width)
        {
            m_fillcount = m_width - content_width;
        }
        else
        {
            m_width = content_width;
        }
    }

    void write_fill(writer_type& out, int count) const
    {
        m_encoder.encode(out, count, m_fillchar);
    }

    int width_body() const noexcept
    {
        int bw = 0;
        auto uv = unsigned_value();
        if(m_base == 10)
        {
            if(m_value < 0 || m_showpos)
            {
                ++bw;
            }
            bw += stringify::v0::detail::number_of_digits<10>(uv);
        }
        else if(m_base == 16)
        {
            if (m_showbase)
            {
                bw += 2;
            }
            bw += stringify::v0::detail::number_of_digits<16>(uv);
        }
        else
        {
            BOOST_ASSERT(m_base == 8);
            if (m_showbase)
            {
                ++bw;
            }
            bw += stringify::v0::detail::number_of_digits<8>(uv);
        }
        return bw;
    }

};

template <typename IntT>
struct int_input_traits
{
    template <typename CharT, typename>
    using formatter =
        stringify::v0::detail::int_formatter<IntT, CharT>;
};

} // namespace detail

stringify::v0::detail::int_input_traits<short>
boost_stringify_input_traits_of(short);

stringify::v0::detail::int_input_traits<int>
boost_stringify_input_traits_of(int);

stringify::v0::detail::int_input_traits<long>
boost_stringify_input_traits_of(long);

stringify::v0::detail::int_input_traits<long long>
boost_stringify_input_traits_of(long long);

stringify::v0::detail::int_input_traits<unsigned short>
boost_stringify_input_traits_of(unsigned short);

stringify::v0::detail::int_input_traits<unsigned>
boost_stringify_input_traits_of(unsigned);

stringify::v0::detail::int_input_traits<unsigned long>
boost_stringify_input_traits_of(unsigned long);

stringify::v0::detail::int_input_traits<unsigned long long>
boost_stringify_input_traits_of(unsigned long long);

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

namespace detail
{
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<long long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned short, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_formatter<unsigned long long, wchar_t>;
}

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_flags
         <'+', '-', '<', '>', '=', 'o', 'd', 'x', 'X', 'c', 'C', '#', '$'>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
