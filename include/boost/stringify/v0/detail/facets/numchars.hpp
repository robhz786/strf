#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/stringify/v0/detail/int_digits.hpp>
#include <cstdint>
#include <cmath> // std::ceil
#include <algorithm>
#include <numeric>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, unsigned Base>
struct numchars_category;

template <typename CharT>
class numchars
{
public:

    numchars(int char_width = 1)
        : _char_width(char_width)
    {
    }

    virtual ~numchars() {}

    virtual bool print_base_indication
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual bool print_pos_sign
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual bool print_neg_sign
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual bool print_exp_base
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const = 0;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits ) const = 0;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const = 0;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , unsigned num_digits ) const = 0;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const = 0;
    virtual bool print_zeros
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned count ) const = 0;
    virtual std::size_t size
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const = 0;
    virtual int width
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const = 0;

    int char_width() const
    {
        return _char_width;
    }

private:

    int _char_width;
};

template <typename CharT>
class numchars_default_common: public stringify::v0::numchars<CharT>
{
public:

    virtual bool print_pos_sign
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;

    virtual bool print_neg_sign
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;

    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , unsigned num_digits ) const override;

    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , const char* digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const override;

    virtual bool print_zeros
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned count ) const override;
};

template <typename CharT>
bool numchars_default_common<CharT>::print_pos_sign
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() == 0 && !ob.recycle())
    {
        return false;
    }
    *ob.pos() = '+';
    ob.advance();
    return true;
}

template <typename CharT>
bool numchars_default_common<CharT>::print_neg_sign
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() == 0 && !ob.recycle())
    {
        return false;
    }
    *ob.pos() = '-';
    ob.advance();
    return true;
}

template <typename CharT>
bool numchars_default_common<CharT>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const char* digits
    , unsigned num_digits ) const
{
    (void)enc;
    auto pos = ob.pos();
    auto end = ob.end();
    while(num_digits != 0)
    {
        if (pos == end)
        {
            ob.advance_to(pos);
            if ( ! ob.recycle())
            {
                return false;
            }
            pos = ob.pos();
            end = ob.end();
        }
        *pos = *digits;
        ++pos;
        ++digits;
        --num_digits;
    }
    ob.advance_to(pos);
    return true;
}

template <typename CharT>
bool numchars_default_common<CharT>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const char* digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    auto sep_size = enc.validate(separator);
    if (sep_size == (std::size_t)-1)
    {
        return print_digits(ob, enc, digits, num_digits);
    }

    if (ob.size() == 0 && ! ob.recycle())
    {
        return false;
    }

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
            if ( ! ob.recycle())
            {
                return false;
            }
            pos = ob.pos();
            end = ob.end();
        }
        if (n == 0)
        {
            auto res = enc.encode_char( &pos, end, separator
                                      , stringify::v0::error_handling::ignore
                                      , false );
            (void)res;
            BOOST_ASSERT(res == stringify::v0::cv_result::success);
            n = *--grp_it;
        }
    }

    ob.advance_to(pos);
    return true;
}

template <typename CharT>
bool numchars_default_common<CharT>::print_zeros
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned count ) const
{
    (void) enc;
    constexpr CharT ch = '0';
    return stringify::v0::detail::write_fill(ob, count, ch);
}

template <typename CharT, unsigned Base>
class default_numchars;

template <typename CharT>
// made final to enable the implementation of has_i18n
class default_numchars<CharT, 10> final
    : public stringify::v0::numchars_default_common<CharT>
{
public:

    using category = numchars_category<CharT, 10>;

    using stringify::v0::numchars_default_common<CharT>::print_digits;

    virtual bool print_base_indication
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_exp_base
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const override;
    virtual std::size_t size
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;
    virtual int width
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;

private:

    bool _print_digits_big_sep
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
bool default_numchars<CharT, 10>::print_base_indication
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)ob;
    (void)enc;
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 10>::print_exp_base
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() == 0 && !ob.recycle())
    {
        return false;
    }
    *ob.pos() = 'e';
    ob.advance();
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 10>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 10>));
    if (ob.size() < num_digits && !ob.recycle())
    {
        return false;
    }
    BOOST_ASSERT(num_digits <= ob.size());

    const char* arr = stringify::v0::detail::chars_00_to_99();
    ob.advance(num_digits);
    CharT* it = ob.pos();
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
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 10>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<10>(digits));
    BOOST_ASSERT(num_groups <= num_digits);
    BOOST_ASSERT(0 == std::count(groups, groups + num_groups, 0));
    BOOST_ASSERT(num_digits == std::accumulate(groups, groups + num_groups, 0u));

    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        return _print_digits_big_sep( ob, enc, digits, groups
                                    , separator, num_digits, num_groups );
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 10>;
    static_assert( max_digits * 2 - 1 <= stringify::v0::min_buff_size
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    if (ob.size() < necessary_size && !ob.recycle())
    {
        return false;
    }
    BOOST_ASSERT(necessary_size <= ob.size());
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
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 10>::_print_digits_big_sep
    ( stringify::v0::output_buffer<CharT>& ob
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
    auto digarr = stringify::v0::detail::write_int_dec_txtdigits_backwards
        (digits, buff + max_digits);
    return numchars_default_common<CharT>::print_digits
        ( ob, enc, digarr, groups, separator, num_digits, num_groups );
}

template <typename CharT>
std::size_t default_numchars<CharT, 10>::size
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    (void) enc;
    (void) has_base_indication;
    return num_digits + has_sign + (has_exponent << 1);
}

template <typename CharT>
int default_numchars<CharT, 10>::width
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    (void)has_base_indication;
    return num_digits + has_sign + (has_exponent << 1);
}

template <typename CharT>
class default_numchars<CharT, 16> final
    : public stringify::v0::numchars_default_common<CharT>
{
public:

    using category = numchars_category<CharT, 16>;

    using stringify::v0::numchars_default_common<CharT>::print_digits;

    virtual bool print_base_indication
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_exp_base
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const override;
    virtual std::size_t size
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;
    virtual int width
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;

private:

    bool _print_digits_big_sep
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
bool default_numchars<CharT, 16>::print_base_indication
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() < 2 && !ob.recycle())
    {
        return false;
    }
    ob.pos()[0] = '0';
    ob.pos()[1] = 'x';
    ob.advance(2);
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 16>::print_exp_base
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() == 0 && !ob.recycle())
    {
        return false;
    }
    *ob.pos() = 'p';
    ob.advance();
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 16>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= sizeof(digits) * 2);
    if (ob.size() < num_digits && !ob.recycle())
    {
        return false;
    }
    BOOST_ASSERT(num_digits <= ob.size());

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
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 16>::_print_digits_big_sep
    ( stringify::v0::output_buffer<CharT>& ob
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
    return numchars_default_common<CharT>::print_digits
        ( ob, enc, digarr, groups, separator, num_digits, num_groups );
}

template <typename CharT>
bool default_numchars<CharT, 16>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<16>(digits));
    BOOST_ASSERT(num_groups <= num_digits);
    BOOST_ASSERT(0 == std::count(groups, groups + num_groups, 0));
    BOOST_ASSERT(num_digits == std::accumulate(groups, groups + num_groups, 0u));

    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        return _print_digits_big_sep( ob, enc, digits, groups
                                    , separator, num_digits, num_groups );
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 16>;
    static_assert( max_digits * 2 - 1 <= stringify::v0::min_buff_size
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    if (ob.size() < necessary_size && !ob.recycle())
    {
        return false;
    }
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
    return true;
}

template <typename CharT>
std::size_t default_numchars<CharT, 16>::size
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    (void) enc;
    return num_digits + has_sign
        + (has_base_indication << 1) + (has_exponent << 1);
}

template <typename CharT>
int default_numchars<CharT, 16>::width
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    return num_digits + has_sign
        + (has_base_indication << 1) + (has_exponent << 1);
}

template <typename CharT>
class default_numchars<CharT, 8> final
    : public stringify::v0::numchars_default_common<CharT>
{
public:

    default_numchars()
    {
    }

    using category = numchars_category<CharT, 8>;

    using stringify::v0::numchars_default_common<CharT>::print_digits;

    virtual bool print_base_indication
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_exp_base
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , unsigned num_digits ) const override;
    virtual bool print_digits
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const override;
    virtual std::size_t size
        ( stringify::v0::encoding<CharT> enc
        , unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;
    virtual int width
        ( unsigned num_digits
        , bool has_sign
        , bool has_base_indication
        , bool has_exponent ) const override;
private:

    bool _print_digits_big_sep
        ( stringify::v0::output_buffer<CharT>& ob
        , stringify::v0::encoding<CharT> enc
        , unsigned long long digits
        , const std::uint8_t* groups
        , char32_t separator
        , unsigned num_digits
        , unsigned num_groups ) const;
};

template <typename CharT>
bool default_numchars<CharT, 8>::print_base_indication
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    if (ob.size() == 0 && !ob.recycle())
    {
        return false;
    }
    *ob.pos() = '0';
    ob.advance();
    return true;
}

template <typename CharT>
bool default_numchars<CharT, 8>::print_exp_base
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc ) const
{
    (void)enc;
    ob.set_error(std::errc::not_supported);
    return false;
}

template <typename CharT>
bool default_numchars<CharT, 8>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits ) const
{
    (void)enc;
    BOOST_ASSERT(num_digits <= (detail::max_num_digits<decltype(digits), 8>));
    if (ob.size() < num_digits && !ob.recycle())
    {
        return false;
    }
    BOOST_ASSERT(num_digits <= ob.size());

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

    return true;
}

template <typename CharT>
bool default_numchars<CharT, 8>::print_digits
    ( stringify::v0::output_buffer<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , const std::uint8_t* groups
    , char32_t separator
    , unsigned num_digits
    , unsigned num_groups ) const
{
    BOOST_ASSERT(num_digits == stringify::v0::detail::count_digits<8>(digits));
    BOOST_ASSERT(num_groups <= num_digits);
    BOOST_ASSERT(0 == std::count(groups, groups + num_groups, 0));
    BOOST_ASSERT(num_digits == std::accumulate(groups, groups + num_groups, 0u));

    if ( separator < enc.u32equivalence_begin()
      || separator >= enc.u32equivalence_end() )
    {
        return _print_digits_big_sep( ob, enc, digits, groups
                                    , separator, num_digits, num_groups );
    }

    constexpr auto max_digits
        = stringify::v0::detail::max_num_digits<decltype(digits), 8>;
    static_assert( max_digits * 2 - 1 <= stringify::v0::min_buff_size
                 , "too many possible digits" );

    auto necessary_size = num_digits + num_groups - 1;
    if (ob.size() < necessary_size && !ob.recycle())
    {
        return false;
    }
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

    return true;
}

template <typename CharT>
bool default_numchars<CharT, 8>::_print_digits_big_sep
    ( stringify::v0::output_buffer<CharT>& ob
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
    return numchars_default_common<CharT>::print_digits
        ( ob, enc, digarr, groups, separator, num_digits, num_groups );
}

template <typename CharT>
std::size_t default_numchars<CharT, 8>::size
    ( stringify::v0::encoding<CharT> enc
    , unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    (void) enc;
    (void) has_exponent;
    (void) has_sign;
    return num_digits + has_base_indication;
}

template <typename CharT>
int default_numchars<CharT, 8>::width
    ( unsigned num_digits
    , bool has_sign
    , bool has_base_indication
    , bool has_exponent ) const
{
    (void) has_sign;
    (void) has_exponent;
    return num_digits + has_base_indication;
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

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

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

template <typename CharT, unsigned Base>
struct numchars_category
{
    static const default_numchars<CharT, Base>& get_default()
    {
        static const stringify::v0::default_numchars<CharT, Base> x{};
        return x;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_NUMCHARS_HPP

