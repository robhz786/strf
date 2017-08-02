#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/conventional_argf_reader.hpp>
#include <boost/stringify/v0/char_flags.hpp>
#include <boost/stringify/v0/output_writer.hpp>
#include <boost/stringify/v0/facets/alignment.hpp>
#include <boost/stringify/v0/facets/fill.hpp>
#include <boost/stringify/v0/facets/intbase.hpp>
#include <boost/stringify/v0/facets/case.hpp>
#include <boost/stringify/v0/facets/showbase.hpp>
#include <boost/stringify/v0/facets/showpos.hpp>
#include <boost/stringify/v0/facets/width.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/detail/characters_catalog.hpp>
#include <boost/stringify/v0/detail/number_of_digits.hpp>
#include <cstdint>

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {

struct int_argf
{
    using char_flags_type = boost::stringify::v0::char_flags
        <'+', '-', '<', '>', '=', 'o', 'd', 'x', 'X', 'c', 'C', '#', '$'>;

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
struct int_stringifier
{
    using unsigned_intT = typename std::make_unsigned<intT>::type;
    using width_t = boost::stringify::v0::width_t;
    using chars_catalog = boost::stringify::v0::detail::characters_catalog;
    using cv_tag = boost::stringify::v0::conversion_from_utf32_tag<CharT>;
    using wcalc_tag = boost::stringify::v0::width_calculator_tag;
    static constexpr bool is_signed = std::is_signed<intT>::value;

public:

    using input_type  = intT ;
    using char_type   = CharT;
    using output_type = boost::stringify::v0::output_writer<CharT>;
    using second_arg = boost::stringify::v0::detail::int_argf;

private:

    using argf_reader = boost::stringify::v0::conventional_argf_reader<input_type>;

public:

    template <typename FTuple>
    int_stringifier(const FTuple& ft, intT value) noexcept
        : m_value(value)
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(get_facet<width_tag>(ft).width())
        , m_alignment(get_facet<alignment_tag>(ft).value())
        , m_base(get_facet<intbase_tag>(ft).value())
        , m_showpos(is_signed && value >= 0 && get_facet<showpos_tag>(ft).value())
        , m_showbase(get_facet<showbase_tag>(ft).value())
        , m_uppercase(get_facet<case_tag>(ft).uppercase())
    {
        calculate_fill(get_facet<wcalc_tag>(ft));
    }

    template <typename FTuple>
    int_stringifier(const FTuple& ft, intT value, second_arg argf) noexcept
        : m_value(value)
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(argf_reader::get_width(argf, ft))
        , m_alignment(argf_reader::get_alignment(argf, ft))
        , m_base(argf_reader::get_base(argf, ft))
        , m_showpos(is_signed && value >= 0 && argf_reader::get_showpos(argf, ft))
        , m_showbase(argf_reader::get_showbase(argf, ft))
        , m_uppercase(argf_reader::get_uppercase(argf, ft))
    {
        calculate_fill(get_facet<wcalc_tag>(ft));
    }

    std::size_t length() const
    {
        return length_body() + length_fill();
    }

    void write(output_type& out) const
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

    int remaining_width(int w) const
    {
        return w > m_total_width ? w - m_total_width : 0;
    }

private:

    const intT m_value;
    const boost::stringify::v0::conversion_from_utf32<CharT>& m_conv;
    const char32_t m_fillchar;
    width_t m_fillcount = 0;
    width_t m_total_width;
    const boost::stringify::v0::alignment m_alignment;
    const unsigned short m_base;
    const bool m_showpos;
    const bool m_showbase;
    const bool m_uppercase;


    template <typename FacetCategory, typename FTuple>
    static const auto& get_facet(const FTuple& ft) noexcept
    {
        return ft.template get_facet<FacetCategory, input_type>();
    }

    unsigned_intT unsigned_value() const noexcept
    {
        if(m_base == 10)
        {
            return boost::stringify::v0::detail::unsigned_abs(m_value);
        }
        return static_cast<unsigned_intT>(m_value);
    }

    std::size_t length_fill() const
    {
        if (m_fillcount > 0)
        {
            return m_fillcount * m_conv.length(m_fillchar);
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
        return boost::stringify::v0::detail::number_of_digits<Base>(unsigned_value());
    }


    // template <unsigned Base>
    // std::size_t length_separators(unsigned num_digits) const noexcept
    // {
    //     auto separators_count =
    //         boost::stringify::v0::numgrouping_count<input_type, Base>
    //         (m_fmt, num_digits);

    //     if (separators_count > 0)
    //     {
    //         char32_t separator
    //             = boost::stringify::v0::thousands_sep<input_type, Base>(m_fmt);
    //         return separators_count
    //             * boost::stringify::v0::get_char32_length<CharT, input_type>
    //               (m_fmt, separator);
    //     }
    //     return 0;
    // }

    void write_with_fill(output_type& out) const
    {
        switch (m_alignment)
        {
            case boost::stringify::v0::alignment::left:
                write_sign(out);
                write_base(out);
                write_digits(out);
                write_fill(out);
                break;

            case boost::stringify::v0::alignment::right:
                write_fill(out);
                write_sign(out);
                write_base(out);
                write_digits(out);
                break;

            default:
                BOOST_ASSERT(m_alignment == boost::stringify::v0::alignment::internal);
                write_sign(out);
                write_base(out);
                write_fill(out);
                write_digits(out);
        }
    }

    void write_sign(output_type& out) const
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

    void write_without_fill(output_type& out) const
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

    void write_base(output_type& out) const
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

    void write_digits(output_type& out) const
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
    void write_digits_t(output_type& out) const
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

    template <typename WidthCalculator>
    void calculate_fill(const WidthCalculator& wcalc)
    {
        int content_width = width_body();
        if(content_width < m_total_width)
        {
            auto fillchar_width = wcalc.width_of(m_fillchar);
            m_fillcount = (m_total_width - content_width) / fillchar_width;
        }
        else
        {
            m_total_width = content_width;
        }
    }

    void write_fill(output_type& out) const
    {
        m_conv.write(out, m_fillchar, m_fillcount);
    }

    width_t width_body() const noexcept
    {
        width_t bw = 0;
        auto uv = unsigned_value();
        if(m_base == 10)
        {
            if(m_value < 0 || m_showpos)
            {
                ++bw;
            }
            bw += boost::stringify::v0::detail::number_of_digits<10>(uv);
        }
        else if(m_base == 16)
        {
            if (m_showbase)
            {
                bw += 2;
            }
            bw += boost::stringify::v0::detail::number_of_digits<16>(uv);
        }
        else
        {
            BOOST_ASSERT(m_base == 8);
            if (m_showbase)
            {
                ++bw;
            }
            bw += boost::stringify::v0::detail::number_of_digits<8>(uv);
        }
        return bw;
    }

};


template <typename IntT>
struct int_input_traits
{
    template <typename CharT, typename>
    using stringifier =
        boost::stringify::v0::detail::int_stringifier<IntT, CharT>;
};

} // namespace detail

boost::stringify::v0::detail::int_input_traits<short>
boost_stringify_input_traits_of(short);

boost::stringify::v0::detail::int_input_traits<int>
boost_stringify_input_traits_of(int);

boost::stringify::v0::detail::int_input_traits<long>
boost_stringify_input_traits_of(long);

boost::stringify::v0::detail::int_input_traits<long long>
boost_stringify_input_traits_of(long long);

boost::stringify::v0::detail::int_input_traits<unsigned short>
boost_stringify_input_traits_of(unsigned short);

boost::stringify::v0::detail::int_input_traits<unsigned>
boost_stringify_input_traits_of(unsigned);

boost::stringify::v0::detail::int_input_traits<unsigned long>
boost_stringify_input_traits_of(unsigned long);

boost::stringify::v0::detail::int_input_traits<unsigned long long>
boost_stringify_input_traits_of(unsigned long long);

} // inline namespace v0
} // namespace stringify
} // namespace boost


#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_INT_HPP_INCLUDED
