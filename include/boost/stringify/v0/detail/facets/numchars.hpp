#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/stringify/v0/detail/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/int_digits.hpp>
#include <cstdint>
#include <cmath> // std::ceil
#include <algorithm>
#include <numeric>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, unsigned Base>
struct numchars_c;

template <typename CharT>
class numchars
{
public:
    virtual ~numchars() {}

    virtual void print_base_indication
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual void print_pos_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual void print_neg_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual void print_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , bool negative ) const = 0;
    virtual void print_single_digit
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned digit ) const = 0;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros = 0 ) const = 0;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros = 0) const = 0;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_trailing_zeros ) const = 0;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , const char* digits
        , unsigned num_digits
        , unsigned num_trailing_zeros ) const = 0;
    virtual std::size_t integer_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const = 0;
    virtual void print_scientific_notation
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool print_point = false
        , unsigned trailing_zeros = 0 ) const = 0;
    virtual std::size_t scientific_notation_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool has_sign
        , bool print_point ) const = 0;
    virtual int scientific_notation_printwidth
        ( unsigned num_digits
        , int exponent
        , bool has_sign ) const = 0;
    /**
    @note `num_digits` is allowed to be greater than the number of digits in
          `digits`. In this case, the function writes leading zeros.
     */
    virtual void print_fractional_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , unsigned trailing_zeros = 0 ) const = 0;
    virtual std::size_t fractional_digits_printsize
        ( stringify::v0::encoding<CharT> enc
        , char32_t decimal_point
        , unsigned num_digits ) const = 0;
    virtual std::size_t fractional_digits_printwidth
        ( unsigned num_digits ) const = 0;
    virtual int integer_printwidth
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const = 0;
};

namespace detail {

template <typename CharT>
void print_amplified_integer_small_separator
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const std::uint8_t* groups
    , unsigned num_groups
    , CharT separator
    , const char* digits
    , unsigned num_digits )
{
    (void)enc;
    BOOST_ASSERT(num_groups != 0);
    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    while (num_digits > grp_size)
    {
        BOOST_ASSERT(grp_size + 1 <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        auto digits_2 = digits + grp_size;
        std::copy(digits, digits_2, it);
        it[grp_size] = separator;
        digits = digits_2;
        ob.advance(grp_size + 1);
        num_digits -= grp_size;
        BOOST_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        ob.ensure(num_digits);
        std::copy(digits, digits + num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        grp_size -= num_digits;
        ob.ensure(grp_size);
        std::char_traits<CharT>::assign(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups)
    {
        grp_size = *--grp_it;
        BOOST_ASSERT(grp_size + 1 <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        *it = separator;
        std::char_traits<CharT>::assign(it + 1, grp_size, '0');
        ob.advance(grp_size + 1);
    }
}

template <typename CharT>
void print_amplified_integer_big_separator
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const std::uint8_t* groups
    , unsigned num_groups
    , char32_t separator
    , std::size_t separator_size
    , const char* digits
    , unsigned num_digits )
{
    BOOST_ASSERT(num_groups != 0);
    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    while (num_digits > grp_size)
    {
        BOOST_ASSERT(grp_size + separator_size <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + separator_size);
        auto it = ob.pos();
        auto digits_2 = digits + grp_size;
        std::copy(digits, digits_2, it);
        digits = digits_2;
        ob.advance_to(enc.encode_char(it + grp_size, separator));
        num_digits -= grp_size;
        BOOST_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        ob.ensure(num_digits);
        std::copy(digits, digits + num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        grp_size -= num_digits;
        ob.ensure(grp_size);
        std::char_traits<CharT>::assign(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups)
    {
        grp_size = *--grp_it;
        BOOST_ASSERT(grp_size + separator_size <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + separator_size);
        auto it = enc.encode_char(ob.pos(), separator);
        std::char_traits<CharT>::assign(it, grp_size, '0');
        ob.advance_to(it + separator_size);
    }
}

template <typename CharT>
class numchars_default_common: public stringify::v0::numchars<CharT>
{
public:

    virtual void print_pos_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual void print_neg_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual void print_sign
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , bool negative ) const override;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , const char* digits
        , unsigned num_digits
        , unsigned num_trailing_zeros ) const override;
    virtual std::size_t fractional_digits_printsize
        ( stringify::v0::encoding<CharT> enc
        , char32_t decimal_point
        , unsigned num_digits ) const override;
    virtual std::size_t fractional_digits_printwidth
        ( unsigned num_digits ) const override;
    virtual std::size_t scientific_notation_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool has_sign
        , bool print_point ) const override;
    virtual int scientific_notation_printwidth
        ( unsigned num_digits
        , int exponent
        , bool has_sign ) const override;

    using numchars<CharT>::print_integer;

protected:

    static void _print_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , unsigned num_digits );

    static void _print_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups );
};

template <typename CharT>
void numchars_default_common<CharT>::print_pos_sign
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    ob.ensure(1);
    *ob.pos() = '+';
    ob.advance();
}

template <typename CharT>
void numchars_default_common<CharT>::print_neg_sign
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    ob.ensure(1);
    *ob.pos() = '-';
    ob.advance();
}

template <typename CharT>
void numchars_default_common<CharT>::print_sign
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , bool negative ) const
{
    (void)enc;
    ob.ensure(1);
    *ob.pos() = static_cast<CharT>('+') + (negative << 1);
    ob.advance();
}

template <typename CharT>
void numchars_default_common<CharT>::_print_digits
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const char* digits
    , unsigned num_digits )
{
    (void)enc;
    auto pos = ob.pos();
    auto end = ob.end();
    while(num_digits != 0)
    {
        if (pos == end)
        {
            ob.advance_to(pos);
            ob.recycle();
            pos = ob.pos();
            end = ob.end();
        }
        *pos = *digits;
        ++pos;
        ++digits;
        --num_digits;
    }
    ob.advance_to(pos);
}

template <typename CharT>
void numchars_default_common<CharT>::_print_digits
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const char* digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups )
{
    auto sep_size = enc.validate(separator);
    if (sep_size == (std::size_t)-1)
    {
        _print_digits(ob, enc, digits, num_digits);
        return;
    }

    ob.ensure(1);

    auto pos = ob.pos();
    auto end = ob.end();
    auto grp_it = groups + num_groups - 1;
    auto n = *grp_it;

    while(true)
    {
        *pos = *digits;
        ++pos;
        ++digits;
        if (--num_digits == 0)
        {
            break;
        }
        --n;
        if (pos == end || (n == 0 && pos + sep_size >= end))
        {
            ob.advance_to(pos);
            ob.recycle();
            pos = ob.pos();
            end = ob.end();
        }
        if (n == 0)
        {
            pos = enc.encode_char(pos, separator);
            n = *--grp_it;
        }
    }
    ob.advance_to(pos);
}

template <typename CharT>
void numchars_default_common<CharT>::print_amplified_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const stringify::v0::numpunct_base& punct
    , std::uint8_t* mem
    , const char* digits
    , unsigned num_digits
    , unsigned num_trailing_zeros ) const
{
    auto sep32 = punct.thousands_sep();
    CharT sep = static_cast<CharT>(sep32);;
    //char buff[detail::max_num_digits<decltype(digits), 8>];

    if (sep32 >= enc.u32equivalence_end() || sep32 < enc.u32equivalence_begin())
    {
        auto sep_size = enc.validate(sep32);
        if (sep_size == (std::size_t)-1 || sep_size == 0)
        {
            ob.ensure(num_digits);
            std::copy(digits, digits + num_digits, ob.pos());
            ob.advance(num_digits);
            ob.ensure(num_trailing_zeros);
            std::char_traits<CharT>::assign(ob.pos(), num_trailing_zeros, '0');
            return;
        }
        if (sep_size > 1)
        {
            auto num_groups = punct.groups(num_trailing_zeros + num_digits, mem);
            stringify::v0::detail::print_amplified_integer_big_separator<CharT>
                ( ob, enc, mem, num_groups, sep, sep_size
                , digits, num_digits );
            return;
        }
        enc.encode_char(&sep, sep32);
    }
    auto num_groups = punct.groups(num_trailing_zeros + num_digits, mem);
    stringify::v0::detail:: print_amplified_integer_small_separator<CharT>
        ( ob, enc, mem, num_groups, sep, digits, num_digits );
}

template <typename CharT>
std::size_t numchars_default_common<CharT>::fractional_digits_printsize
    ( stringify::v0::encoding<CharT> enc
    , char32_t decimal_point
    , unsigned num_digits ) const
{
    auto s = enc.validate(decimal_point);
    if (s == (size_t)-1)
    {
        s = enc.replacement_char_size();
    }
    return s + num_digits;
}

template <typename CharT>
std::size_t numchars_default_common<CharT>::scientific_notation_printsize
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , char32_t decimal_point
    , int exponent
    , bool has_sign
    , bool print_point ) const
{
    std::size_t psize = 0;
    print_point = print_point || num_digits > 1;
    if (print_point)
    {
        psize = enc.validate(decimal_point);
        if (psize == (size_t)-1)
        {
            psize = enc.replacement_char_size();
        }
    }
    unsigned e10u = std::abs(exponent);
    return has_sign + num_digits + psize + 2
        + detail::count_digits<10>(e10u) + (e10u < 10);
}

template <typename CharT>
int numchars_default_common<CharT>::scientific_notation_printwidth
     ( unsigned num_digits
     , int exponent
     , bool has_sign ) const
{
    unsigned e10u = std::abs(exponent);
    return has_sign + num_digits + (e10u < 10) + 2
        + detail::count_digits<10>(e10u);
}

template <typename CharT>
std::size_t numchars_default_common<CharT>::fractional_digits_printwidth
    ( unsigned num_digits ) const
{
    return num_digits;
}

} // namespace detail

template <typename CharT, unsigned Base>
class default_numchars;

template <typename CharT>
// made final to enable the implementation of has_i18n
class default_numchars<CharT, 10> final
    : public stringify::v0::detail::numchars_default_common<CharT>
{
public:

    using category = numchars_c<CharT, 10>;

    using stringify::v0::detail::numchars_default_common<CharT>::print_integer;
    using stringify::v0::detail::numchars_default_common<CharT>::print_amplified_integer;

    virtual void print_base_indication
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros = 0) const override;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned trailing_zeros ) const override;
    virtual void print_scientific_notation
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool print_point
        , unsigned trailing_zeros ) const override;
    virtual void print_fractional_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , unsigned trailing_zeros ) const override;
    virtual void print_single_digit
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned digit ) const override;
    virtual std::size_t integer_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;
    virtual int integer_printwidth
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;

protected:

    using stringify::v0::detail::numchars_default_common<CharT>::_print_digits;

private:

    void _print_digits_big_sep
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
void default_numchars<CharT, 10>::print_base_indication
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)ob;
    (void)enc;
}

template <typename CharT>
void default_numchars<CharT, 10>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    (void)enc;
    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 10>));
    ob.ensure(num_digits);
    ob.advance(num_digits);
    CharT* it = ob.pos();
    const char* arr = stringify::v0::detail::chars_00_to_99();

    auto uvalue = digits;
    while (uvalue > 99)
    {
        auto index = (uvalue % 100) << 1;
        it[-2] = arr[index];
        it[-1] = arr[index + 1];
        it -= 2;
        uvalue /= 100;
    }
    if (uvalue < 10)
    {
        it[-1] = static_cast<CharT>('0' + uvalue);
        BOOST_ASSERT(it + num_digits - 1 == ob.pos());
    }
    else
    {
        auto index = uvalue << 1;
        it[-2] = arr[index];
        it[-1] = arr[index + 1];
        BOOST_ASSERT(it + num_digits - 2 == ob.pos());
    }
}

template <typename CharT>
void default_numchars<CharT, 10>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const stringify::v0::numpunct_base& punct
    , std::uint8_t* mem
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<10>(digits));

    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    const auto num_groups = punct.groups(num_digits, mem);
    const auto * groups = mem;
    const auto separator = punct.thousands_sep();
    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        _print_digits_big_sep( ob, enc, digits, groups
                             , separator, num_digits, num_groups );
        return;
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 10>;
    static_assert( max_digits * 2 - 1 <= boost::min_size_after_recycle<CharT>()
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    ob.ensure(necessary_size);
    ob.advance(necessary_size);
    CharT* it = ob.pos();

    const char* arr = stringify::v0::detail::chars_00_to_99();
    auto n = *groups;
    CharT sep = static_cast<CharT>(separator);
    auto uvalue = digits;
    while (uvalue > 99)
    {
        auto index = (uvalue % 100) << 1;
        if (n > 1)
        {
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            n -= 2;
            if (n == 0)
            {
                it[-3] = sep;
                n = * ++groups;
                it -= 3;
            }
            else
            {
                it -= 2;
            }
        }
        else
        {
            it[-3] = arr[index];
            it[-2] = sep;
            it[-1] = arr[index + 1];
            n = * ++groups - 1;
            if (n == 0)
            {
                it[-4] = sep;
                it -= 4;
                n = * ++groups;
            }
            else
            {
                it -= 3;
            }
        }
        uvalue /= 100;
    }
    BOOST_ASSERT(n != 0);
    if (uvalue < 10)
    {
        it[-1] = static_cast<CharT>('0' + uvalue);
        BOOST_ASSERT(it + necessary_size - 1 == ob.pos());
    }
    else
    {
        auto index = uvalue << 1;
        if (n == 1)
        {
            it[-3] = arr[index];
            it[-2] = sep;
            it[-1] = arr[index + 1];
            BOOST_ASSERT(it + necessary_size - 3 == ob.pos());
        }
        else
        {
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            BOOST_ASSERT(it + necessary_size - 2 == ob.pos());
        }
    }
}

template <typename CharT>
void default_numchars<CharT, 10>::_print_digits_big_sep
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 10>;
    char buff[max_digits];
    auto char_digits = stringify::v0::detail::write_int_dec_txtdigits_backwards
        ( digits, buff + max_digits );
    _print_digits
        ( ob, enc, char_digits, groups, separator, num_digits, num_groups );
}

template <typename CharT>
void default_numchars<CharT, 10>::print_amplified_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned trailing_zeros ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 10>));

    ob.ensure(num_digits);
    stringify::v0::detail::write_int_dec_txtdigits_backwards
        ( digits, ob.pos() + num_digits );
    ob.advance(num_digits);
    stringify::v0::detail::write_fill(ob, trailing_zeros, CharT('0'));
}

template <typename CharT>
void default_numchars<CharT, 10>::print_scientific_notation
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , int exponent
    , bool print_point
    , unsigned trailing_zeros ) const
{
    BOOST_ASSERT(num_digits == detail::count_digits<10>(digits));

    CharT small_decimal_point = static_cast<CharT>(decimal_point);
    std::size_t psize = 1;
    print_point = print_point || num_digits > 1|| trailing_zeros != 0;
    if ( print_point
      && ( decimal_point >= enc.u32equivalence_end()
        || decimal_point < enc.u32equivalence_begin() ) )
    {
        psize = enc.validate(decimal_point);
        if (psize == (std::size_t)-1)
        {
            psize = enc.replacement_char_size();
        }
        else if (psize == 1)
        {
            enc.encode_char(&small_decimal_point, decimal_point);
        }
    }
    if (num_digits == 1)
    {
        ob.ensure(num_digits + print_point * psize);
        auto it = ob.pos();
        *it = '0' + digits;
        if (print_point)
        {
            if (psize == 1)
            {
                it[1] = small_decimal_point;
                ob.advance(2);
            }
            else
            {
                ob.advance_to(enc.encode_char(it + 1, decimal_point));
            }
        }
        else
        {
            ob.advance(1);
        }
    }
    else
    {
        ob.ensure(num_digits + psize);
        auto it = ob.pos() + num_digits + psize;
        const char* arr = stringify::v0::detail::chars_00_to_99();

        while(digits > 99)
        {
            auto index = (digits % 100) << 1;
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            it -= 2;
            digits /= 100;
        }
        CharT highest_digit;
        if (digits < 10)
        {
            highest_digit = static_cast<CharT>('0' + digits);
        }
        else
        {
            auto index = digits << 1;
            highest_digit = arr[index];
            * --it = arr[index + 1];
        }
        if (psize == 1)
        {
            * --it = small_decimal_point;
        }
        else
        {
            it -= psize;
            enc.encode_char(it, decimal_point);
        }
        * --it = highest_digit;
        BOOST_ASSERT(it == ob.pos());
        ob.advance(num_digits + psize);
    }
    if (trailing_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, trailing_zeros, CharT('0'));
    }

    unsigned adv = 4;
    CharT* it;
    unsigned e10u = std::abs(exponent);
    BOOST_ASSERT(e10u < 1000);

    if (e10u >= 100)
    {
        ob.ensure(5);
        it = ob.pos();
        it[4] = static_cast<CharT>('0' + e10u % 10);
        e10u /= 10;
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
        adv = 5;
    }
    else if (e10u >= 10)
    {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
    }
    else
    {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<CharT>('0' + e10u);
        it[2] = '0';
    }
    it[0] = 'e';
    it[1] = static_cast<CharT>('+' + ((exponent < 0) << 1));
    ob.advance(adv);
}

template <typename CharT>
void default_numchars<CharT, 10>::print_single_digit
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned digit ) const
{
    (void)enc;
    BOOST_ASSERT(digit < 10);
    ob.ensure(1);
    *ob.pos() = (CharT)'0' + digit;
    ob.advance(1);
}

template <typename CharT>
void default_numchars<CharT, 10>::print_fractional_digits
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , unsigned trailing_zeros ) const
{
    enc.encode_char(ob, decimal_point, stringify::v0::encoding_error::replace);
    if (num_digits != 0)
    {
        ob.ensure(num_digits);
        const auto end = ob.pos() + num_digits;
        auto it = detail::write_int_dec_txtdigits_backwards(digits, end);
        auto itz = end - num_digits;
        if (itz < it)
        {
            std::char_traits<CharT>::assign(itz, it - itz, '0');
        }
        ob.advance_to(end);
    }
    if (trailing_zeros != 0)
    {
        detail::write_fill(ob, trailing_zeros, CharT('0'));
    }
}

template <typename CharT>
std::size_t default_numchars<CharT, 10>::integer_printsize
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    (void) enc;
    (void) has_base_indication;
    return num_digits + has_sign;
}

template <typename CharT>
int default_numchars<CharT, 10>::integer_printwidth
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    (void)has_base_indication;
    return num_digits + has_sign;
}

template <typename CharT>
class default_numchars<CharT, 16> final
    : public stringify::v0::detail::numchars_default_common<CharT>
{
public:

    using category = numchars_c<CharT, 16>;

    using stringify::v0::detail::numchars_default_common<CharT>::print_integer;
    using stringify::v0::detail::numchars_default_common<CharT>::print_amplified_integer;

    virtual void print_base_indication
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros = 0) const override;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned trailing_zeros ) const override;
    virtual void print_scientific_notation
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool print_point
        , unsigned trailing_zeros ) const override;
    // virtual std::size_t scientific_notation_printsize
    //     ( stringify::v0::encoding<CharT> enc
    //     , unsigned num_digits
    //     , char32_t decimal_point
    //     , int exponent
    //     , bool has_sign
    //     , bool print_point ) const override;
    virtual void print_fractional_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , unsigned trailing_zeros ) const override;
    virtual void print_single_digit
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned digit ) const override;
    virtual std::size_t integer_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;
    virtual int integer_printwidth
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;

protected:

    using stringify::v0::detail::numchars_default_common<CharT>::_print_digits;

private:

    void _print_digits_big_sep
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
void default_numchars<CharT, 16>::print_base_indication
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    ob.ensure(2);
    ob.pos()[0] = '0';
    ob.pos()[1] = 'x';
    ob.advance(2);
}

template <typename CharT>
void default_numchars<CharT, 16>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    (void)enc;
    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    BOOST_ASSERT(num_digits <= sizeof(digits) * 2);
    ob.ensure(num_digits);
    ob.advance(num_digits);
    CharT* it = ob.pos();
    auto value = digits;
    constexpr char offset_a = 'a' - 10;
    while (value > 0xF)
    {
        std::uint8_t d = value & 0xF;
        --it;
        if (d < 10)
        {
            *it = static_cast<CharT>('0' + d);
        }
        else
        {
            *it = static_cast<CharT>(offset_a + d);
        }
        value = value >> 4;
    }
    --it;
    if (value < 10)
    {
        *it = static_cast<CharT>('0' + value);
    }
    else
    {
        *it = static_cast<CharT>(offset_a + value);
    }

    BOOST_ASSERT(it + num_digits == ob.pos());
}

template <typename CharT>
void default_numchars<CharT, 16>::_print_digits_big_sep
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 16>;
    char buff[max_digits];
    auto digarr = stringify::v0::detail::write_int_hex_txtdigits_backwards
        (digits, buff + max_digits);
    _print_digits
        ( ob, enc, digarr, groups, separator, num_digits, num_groups );
}

template <typename CharT>
void default_numchars<CharT, 16>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const stringify::v0::numpunct_base& punct
    , std::uint8_t* mem
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<16>(digits));

    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    const auto num_groups = punct.groups(num_digits, mem);
    const auto * groups = mem;
    const auto separator = punct.thousands_sep();
    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        _print_digits_big_sep( ob, enc, digits, groups
                             , separator, num_digits, num_groups );
        return;
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 16>;
    static_assert( max_digits * 2 - 1 <= boost::min_size_after_recycle<CharT>()
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    ob.ensure(necessary_size);
    ob.advance(necessary_size);
    CharT* it = ob.pos();
    auto n = *groups;
    auto value = digits;
    CharT sep = static_cast<CharT>(separator);
    constexpr char offset_digit_a = 'a' - 10;
    while (value > 0xF)
    {
        unsigned d = value & 0xF;
        --it;
        if (d < 10)
        {
            *it = static_cast<CharT>('0' + d);
        }
        else
        {
            *it = static_cast<CharT>(offset_digit_a + d);
        }
        if (--n == 0)
        {
            *--it = sep;
            n = *++groups;
        }
        value = value >> 4;
    }
    --it;
    if (value < 10)
    {
        *it = static_cast<CharT>('0' + value);
    }
    else
    {
        *it = static_cast<CharT>(offset_digit_a + value);
    }
    BOOST_ASSERT(it + necessary_size == ob.pos());
}

template <typename CharT>
void default_numchars<CharT, 16>::print_amplified_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned trailing_zeros ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 16>));
    ob.ensure(num_digits);
    stringify::v0::detail::write_int_hex_txtdigits_backwards
        ( digits, ob.pos() + num_digits );
    ob.advance(num_digits);
    stringify::v0::detail::write_fill(ob, trailing_zeros, CharT('0'));
}

template <typename CharT>
void default_numchars<CharT, 16>::print_scientific_notation
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , int exponent
    , bool print_point
    , unsigned trailing_zeros ) const
{
    // todo
    (void) ob;
    (void) enc;
    (void) digits;
    (void) num_digits;
    (void) decimal_point;
    (void) exponent;
    (void) print_point;
    (void) trailing_zeros;
}

// template <typename CharT>
// std::size_t default_numchars<CharT, 16>::scientific_notation_printsize
//     ( stringify::v0::encoding<CharT> enc
//     , unsigned num_digits
//     , char32_t decimal_point
//     , int exponent
//     , bool has_sign
//     , bool print_point ) const
// {
//     (void) enc;
//     (void) num_digits;
//     (void) decimal_point;
//     (void) exponent;
//     (void) has_sign;
//     (void) print_point;

//     return 0;
// }

template <typename CharT>
void default_numchars<CharT, 16>::print_fractional_digits
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , unsigned trailing_zeros ) const
{
    (void) ob;
    (void) enc;
    (void) digits;
    (void) num_digits;
    (void) decimal_point;
    (void) trailing_zeros;
}

template <typename CharT>
void default_numchars<CharT, 16>::print_single_digit
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned digit ) const
{
    (void)enc;
    BOOST_ASSERT(digit < 16);
    ob.ensure(1);
    *ob.pos() = (digit < 10) ? ((CharT)'0' + digit) : (CharT)'a' + (digit - 10);
    ob.advance(1);
}

template <typename CharT>
std::size_t default_numchars<CharT, 16>::integer_printsize
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    (void) enc;
    return num_digits + has_sign + (has_base_indication << 1);
}

template <typename CharT>
int default_numchars<CharT, 16>::integer_printwidth
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    return num_digits + has_sign + (has_base_indication << 1);
}

template <typename CharT>
class default_numchars<CharT, 8> final
    : public stringify::v0::detail::numchars_default_common<CharT>
{
public:

    default_numchars()
    {
    }

    using category = numchars_c<CharT, 8>;

    using stringify::v0::detail::numchars_default_common<CharT>::print_integer;
    using stringify::v0::detail::numchars_default_common<CharT>::print_amplified_integer;

    virtual void print_base_indication
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros ) const override;
    virtual void print_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const stringify::v0::numpunct_base& punct
        , std::uint8_t* mem
        , unsigned long long digits
        , unsigned num_digits
        , unsigned num_leading_zeros = 0) const override;
    virtual void print_amplified_integer
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , unsigned trailing_zeros ) const override;
    virtual std::size_t integer_printsize
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;
    virtual int integer_printwidth
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication ) const override;
    virtual void print_scientific_notation
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , int exponent
        , bool print_point
        , unsigned trailing_zeros ) const override;
    // virtual std::size_t scientific_notation_printsize
    //     ( stringify::v0::encoding<CharT> enc
    //     , unsigned num_digits
    //     , char32_t decimal_point
    //     , int exponent
    //     , bool has_sign
    //     , bool print_point ) const override;
    virtual void print_fractional_digits
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits
        , char32_t decimal_point
        , unsigned trailing_zeros ) const override;
    virtual void print_single_digit
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned digit ) const override;

protected:

    using stringify::v0::detail::numchars_default_common<CharT>::_print_digits;

private:

    void _print_digits_big_sep
        ( boost::basic_outbuf<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
void default_numchars<CharT, 8>::print_base_indication
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    ob.ensure(1);
    *ob.pos() = '0';
    ob.advance();
}

template <typename CharT>
void default_numchars<CharT, 8>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 8>));
    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    ob.ensure(num_digits);
    ob.advance(num_digits);
    CharT* it = ob.pos();
    auto value = digits;
    while (value > 0x7)
    {
        *--it = '0' + (value & 0x7);
        value = value >> 3;
    }
    *--it = static_cast<CharT>('0' + value);
    BOOST_ASSERT(it + num_digits == ob.pos());
}

template <typename CharT>
void default_numchars<CharT, 8>::print_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const stringify::v0::numpunct_base& punct
    , std::uint8_t* mem
    , unsigned long long digits
    , unsigned num_digits
    , unsigned num_leading_zeros ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<8>(digits));

    if (num_leading_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, num_leading_zeros, CharT('0'));
    }
    const auto num_groups = punct.groups(num_digits, mem);
    const auto * groups = mem;
    const auto separator = punct.thousands_sep();
    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        return _print_digits_big_sep( ob, enc, digits, groups
                                    , separator, num_digits, num_groups );
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 8>;
    static_assert( max_digits * 2 - 1 <= boost::min_size_after_recycle<CharT>()
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    ob.ensure(necessary_size);
    ob.advance(necessary_size);
    auto it = ob.pos();
    auto value = digits;
    auto n = *groups;
    CharT sep = static_cast<CharT>(separator);
    while (value > 0x7)
    {
        *--it = '0' + (value & 0x7);
        value = value >> 3;
        if (--n == 0)
        {
            *--it = sep;
            n = *++groups;
        }
    }
    *--it = static_cast<CharT>('0' + value);
    BOOST_ASSERT(it + necessary_size == ob.pos());
}

template <typename CharT>
void default_numchars<CharT, 8>::_print_digits_big_sep
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 8>;
    char buff[max_digits];
    auto digarr = stringify::v0::detail::write_int_oct_txtdigits_backwards
        (digits, buff + max_digits);
    _print_digits
        ( ob, enc, digarr, groups, separator, num_digits, num_groups );
}

template <typename CharT>
void default_numchars<CharT, 8>::print_amplified_integer
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , unsigned trailing_zeros ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 16>));
    ob.ensure(num_digits);
    stringify::v0::detail::write_int_oct_txtdigits_backwards
        ( digits, ob.pos() + num_digits );
    ob.advance(num_digits);
    stringify::v0::detail::write_fill(ob, trailing_zeros, CharT('0'));
}

template <typename CharT>
void default_numchars<CharT, 8>::print_scientific_notation
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , int exponent
    , bool print_point
    , unsigned trailing_zeros ) const
{
    // this function should never be called
    (void) ob;
    (void) enc;
    (void) digits;
    (void) num_digits;
    (void) decimal_point;
    (void) exponent;
    (void) print_point;
    (void) trailing_zeros;
}

// template <typename CharT>
// std::size_t default_numchars<CharT, 8>::scientific_notation_printsize
//     ( stringify::v0::encoding<CharT> enc
//     , unsigned num_digits
//     , char32_t decimal_point
//     , int exponent
//     , bool has_sign
//     , bool print_point ) const
// {
//     (void) enc;
//     (void) num_digits;
//     (void) decimal_point;
//     (void) exponent;
//     (void) has_sign;
//     (void) print_point;
//     return 0;
// }

template <typename CharT>
void default_numchars<CharT, 8>::print_fractional_digits
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , unsigned trailing_zeros ) const
{
    (void) ob;
    (void) enc;
    (void) digits;
    (void) num_digits;
    (void) decimal_point;
    (void) trailing_zeros;
}

template <typename CharT>
void default_numchars<CharT, 8>::print_single_digit
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned digit ) const
{
    (void)enc;
    BOOST_ASSERT(digit < 8);
    ob.ensure(1);
    *ob.pos() = (CharT)'0' + digit;
    ob.advance(1);
}

template <typename CharT>
std::size_t default_numchars<CharT, 8>::integer_printsize
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    (void) enc;
    (void) has_sign;
    return num_digits + has_base_indication;
}

template <typename CharT>
int default_numchars<CharT, 8>::integer_printwidth
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication ) const
{
    return num_digits + has_base_indication + has_sign;
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char8_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char8_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char8_t, 8>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char, 8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char16_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char16_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char16_t, 8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char32_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char32_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<char32_t, 8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<wchar_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<wchar_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class default_numchars<wchar_t, 8>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

template <typename CharT, unsigned Base>
struct numchars_c
{
    static const default_numchars<CharT, Base>& get_default()
    {
        static const stringify::v0::default_numchars<CharT, Base> x{};
        return x;
    }
};

namespace detail {

template <typename CharT, typename FPack, typename IntT, unsigned Base>
class has_i18n_impl
{
public:

    static std::true_type  test_numchars
        ( const stringify::v0::numchars<CharT>& );
    static std::false_type test_numchars
        ( const stringify::v0::default_numchars<CharT, Base>& );

    static std::true_type  test_numpunct(const stringify::v0::numpunct_base&);
    static std::false_type test_numpunct(const stringify::v0::default_numpunct<Base>&);

    static const FPack& fp();

    using has_numchars_type = decltype
        ( test_numchars
            ( get_facet<stringify::v0::numchars_c<CharT, Base>, IntT>(fp())) );

    using has_numpunct_type = decltype
        ( test_numpunct
            ( get_facet< stringify::v0::numpunct_c<Base>, IntT >(fp())) );

public:

    static constexpr bool has_i18n
        = has_numchars_type::value || has_numpunct_type::value;
};

template <typename CharT, typename FPack, typename IntT, unsigned Base>
constexpr bool has_i18n = has_i18n_impl<CharT, FPack, IntT, Base>::has_i18n;

static_assert(has_i18n<char, decltype(pack()), int, 10> == false, "");
static_assert(has_i18n<char, decltype(pack(monotonic_grouping<10>(3))), int, 10> == true, "");

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP

