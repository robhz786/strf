#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_INT_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_INT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/int_digits.hpp>
#include <boost/stringify/v0/detail/facets/numchars.hpp>
#include <boost/stringify/v0/detail/facets/numpunct.hpp>
#include <boost/assert.hpp>
#include <algorithm>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename CharT>
class i18n_int_printer: public printer<CharT>
{
public:

    template <typename FPack, typename IntT>
    i18n_int_printer
        ( const FPack& fp
        , IntT value ) noexcept
        : _chars(get_facet<stringify::v0::numchars_c<CharT, 10>, IntT>(fp))
        , _punct(get_facet<stringify::v0::numpunct_c<10>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
        , _digcount(stringify::v0::detail::count_digits<10>(value))
    {
        if (value < 0)
        {
            _uvalue = stringify::v0::detail::unsigned_abs(value);
            _negative = true;
        }
        else
        {
            _uvalue = value;
            _negative = false;
        }
        _sepcount = ( _punct.no_group_separation(_digcount)
                    ? 0
                    : _punct.thousands_sep_count(_digcount) );
    }

    std::size_t necessary_size() const override;

    int width(int) const override;

    void write(stringify::v0::output_buffer<CharT>& ob) const override;

private:

    void _write_with_punct(stringify::v0::output_buffer<CharT>& ob) const;

    const stringify::v0::numchars<CharT>& _chars;
    const stringify::v0::numpunct_base& _punct;
    stringify::v0::encoding<CharT> _encoding;
    unsigned long long _uvalue;
    unsigned _digcount;
    unsigned _sepcount;
    bool _negative;
};

template <typename CharT>
std::size_t i18n_int_printer<CharT>::necessary_size() const
{
    auto size = _chars.integer_printsize
        ( _encoding, _digcount, _negative, false );
    if (_sepcount != 0)
    {
        auto sepsize = _encoding.validate(_punct.thousands_sep());
        if (sepsize != std::size_t(-1))
        {
            size += sepsize * _sepcount;
        }
    }
    return size;
}

template <typename CharT>
int i18n_int_printer<CharT>::width(int) const
{
    return _sepcount + _chars.integer_printwidth(_digcount, _negative, false);
}

template <typename CharT>
void i18n_int_printer<CharT>::write(stringify::v0::output_buffer<CharT>& ob) const
{
    if (_negative)
    {
        _chars.print_neg_sign(ob, _encoding);
    }
    if (_sepcount != 0)
    {
        _write_with_punct(ob);
    }
    else
    {
        _chars.print_integer(ob, _encoding, _uvalue, _digcount);
    }
}

template <typename CharT>
void i18n_int_printer<CharT>::_write_with_punct
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    constexpr unsigned max_digits
        = stringify::v0::detail::max_num_digits<decltype(_uvalue), 10>;
    unsigned char groups[max_digits];
    _chars.print_integer( ob, _encoding, _punct, groups
                        , _uvalue, _digcount );
}


template <typename CharT>
class fast_int_printer: public printer<CharT>
{
public:

    template <typename IntT>
    fast_int_printer(IntT value)
    {
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        if (value < 0)
        {
            unsigned_IntT uvalue = stringify::v0::detail::unsigned_abs(value);
            _digcount = stringify::v0::detail::count_digits<10>(uvalue);
            _uvalue = uvalue;
            _negative = true;
        }
        else
        {
            _digcount = stringify::v0::detail::count_digits<10>(value);
            _uvalue = value;
            _negative = false;
        }
    }

    template <typename FP, typename IntT>
    fast_int_printer(const FP&, IntT value)
        : fast_int_printer(value)
    {
    }

    std::size_t necessary_size() const override;

    int width(int) const override;

    void write(stringify::v0::output_buffer<CharT>& ob) const override;

private:

    unsigned long long _uvalue;
    unsigned _digcount;
    bool _negative;
};

template <typename CharT>
std::size_t fast_int_printer<CharT>::necessary_size() const
{
    return _digcount + _negative;
}

template <typename CharT>
int fast_int_printer<CharT>::width(int) const
{
    return _digcount + _negative;
}

template <typename CharT>
void fast_int_printer<CharT>::write
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    unsigned size = _digcount + _negative;
    ob.ensure(size);
    ob.advance(size);
    CharT* it = write_int_dec_txtdigits_backwards(_uvalue, ob.pos());
    if (_negative)
    {
        it[-1] = '-';
    }
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_int_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_int_printer<char8_t>;
#endif

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_int_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_int_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, short, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, short x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, int, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, int x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, long, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, long long, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, long long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, unsigned short, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned short x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, unsigned int, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned int x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, unsigned long, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, unsigned long long, 10>
    , stringify::v0::detail::i18n_int_printer<CharT>
    , stringify::v0::detail::fast_int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned long long x)
{
    return {fp, x};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_INT_HPP

