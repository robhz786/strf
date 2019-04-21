#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/format_functions.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/stringify/v0/detail/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/int_digits.hpp>
#include <boost/assert.hpp>
#include <cstdint>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct int_format
{
    template <class T>
    class fn
    {
    public:
        using derived_type = stringify::v0::fmt_derived<alignment_format, T>;

        constexpr fn() = default;

        template <typename U>
        constexpr fn(const fn<U> & u)
            : _precision(u.precision())
            , _base(u.base())
            , _showbase(u.showbase())
            , _showpos(u.showpos())
        {
        }

        constexpr derived_type&& p(unsigned _) &&
        {
            _precision = _;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& p(unsigned _) &
        {
            _precision = _;
            return *this;
        }
        constexpr derived_type&& hex() &&
        {
            _base = 16;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& hex() &
        {
            _base = 16;
            return static_cast<derived_type&>(*this);
        }
        constexpr derived_type&& dec() &&
        {
            _base = 10;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& dec() &
        {
            _base = 10;
            return static_cast<derived_type&>(*this);
        }
        constexpr derived_type&& oct() &&
        {
            _base = 8;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& oct() &
        {
            _base = 8;
            return static_cast<derived_type&>(*this);
        }
        constexpr derived_type&& operator+() &&
        {
            _showpos = true;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& operator+() &
        {
            _showpos = true;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& operator~() &&
        {
            _showbase = true;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& operator~() &
        {
            _showbase = true;
            return static_cast<derived_type&>(*this);
        }
        constexpr derived_type&& showbase(bool s) &&
        {
            _showbase = s;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& showbase(bool s) &
        {
            _showbase = s;
            return static_cast<derived_type&>(*this);
        }
        constexpr derived_type&& showpos(bool s) &&
        {
            _showpos = s;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type& showpos(bool s) &
        {
            _showpos = s;
            return static_cast<derived_type&>(*this);
        }
        constexpr unsigned precision() const
        {
            return _precision;
        }
        constexpr unsigned short base() const
        {
            return _base;
        }
        constexpr bool showbase() const
        {
            return _showbase;
        }
        constexpr bool showpos() const
        {
            return _showpos;
        }

    private:

        unsigned _precision = 0;
        unsigned short _base = 10;
        bool _showbase = false;
        bool _showpos = false;
    };
};

namespace detail
{
template <typename IntT>
struct int_value
{
    IntT value;
};
}


template <typename IntT>
using int_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::int_value<IntT>
    , stringify::v0::int_format
    , stringify::v0::alignment_format >;


template <typename CharT>
class fmt_int_printer: public printer<CharT>
{

public:

    template <typename FPack, typename IntT, int Base>
    fmt_int_printer
        ( const FPack& fp
        , stringify::v0::int_with_format<IntT> value
        , std::integral_constant<int, Base> ) noexcept;

    ~fmt_int_printer();

    std::size_t necessary_size() const override;

    bool write( stringify::v0::output_buffer<CharT>& ob ) const override;

    int remaining_width(int w) const override;

private:

    unsigned long long _uvalue;
    const stringify::v0::numchars<CharT>& _chars;
    const stringify::v0::numpunct_base& _punct;
    const stringify::v0::encoding<CharT> _encoding;
    stringify::v0::encoding_policy _epoli;
    unsigned _digcount;
    unsigned _sepcount;
    unsigned _fillcount;
    unsigned _precision;
    stringify::v0::alignment_format::fn<void> _afmt;
    bool _showneg;
    bool _showpos;
    bool _showbase;

    template <typename IntT, int Base>
    void _init(stringify::v0::int_with_format<IntT> value);

    bool _write_fill
        ( stringify::v0::output_buffer<CharT>& ob
        , std::size_t count ) const
    {
        return _encoding.encode_fill
            ( ob, count, _afmt.fill(), _epoli.err_hdl(), _epoli.allow_surr() );
    }

    bool _write_complement(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits_sep(stringify::v0::output_buffer<CharT>& ob) const;
};

template <typename CharT>
template <typename FPack, typename IntT, int Base>
fmt_int_printer<CharT>::fmt_int_printer
    ( const FPack& fp
    , stringify::v0::int_with_format<IntT> value
    , std::integral_constant<int, Base> ) noexcept
    : _chars(get_facet<stringify::v0::numchars_category<CharT, Base>, IntT>(fp))
    , _punct(get_facet<stringify::v0::numpunct_category<Base>, IntT>(fp))
    , _encoding(get_facet<stringify::v0::encoding_category<CharT>, IntT>(fp))
    , _epoli(get_facet<stringify::v0::encoding_policy_category, IntT>(fp))
    , _afmt(value)
{
    _init<IntT, Base>(value);
}

template <typename CharT>
template <typename IntT, int Base>
void fmt_int_printer<CharT>::_init(stringify::v0::int_with_format<IntT> value)
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;

    _digcount = stringify::v0::detail::count_digits<Base>(value.value().value);
    _sepcount = _punct.thousands_sep_count(_digcount);
    _precision = value.precision();

    int complement_width;

#if defined(_MSC_VER)
#pragma warning ( push )
#pragma warning ( disable : 4127 )
#endif // defined(_MSC_VER)

    BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)

#if defined(_MSC_VER)
#pragma warning ( pop )
#endif // defined(_MSC_VER)

    {
        if (value.value().value < 0)
        {
            _uvalue = stringify::v0::detail::unsigned_abs(value.value().value);
            _showneg = true;
            _showpos = false;
            _showbase = false;
            complement_width = 1;
        }
        else
        {
            _uvalue = value.value().value;
            _showneg = false;
            _showpos = value.showpos();
            _showbase = false;
            complement_width = value.showpos();
        }
    }
    else
    {
        _uvalue = unsigned_type(value.value().value);
        _showneg = false;
        _showpos = false;
        _showbase = value.showbase();
        complement_width = ( Base == 8
                           ? value.showbase()
                           : (value.showbase() << 1) );
    }

    int  content_width;
    if (_precision > _digcount)
    {
        content_width
            = _chars.char_width()
            * (_precision + complement_width + _sepcount);
    }
    else
    {
        content_width
            = _chars.char_width()
            * (_digcount + complement_width + _sepcount);
    }

    if (_afmt.width() > content_width)
    {
        _fillcount = _afmt.width() - content_width;
    }
    else
    {
        _afmt.width(content_width);
        _fillcount = 0;
    }
}

template <typename CharT>
fmt_int_printer<CharT>::~fmt_int_printer()
{
}

template <typename CharT>
std::size_t fmt_int_printer<CharT>::necessary_size() const
{
    std::size_t s = _chars.size( _encoding
                               , std::max((unsigned)_digcount, _precision)
                               , _showneg || _showpos
                               , _showbase, false );
    if (_sepcount > 0)
    {
        auto sepsize = _encoding.validate(_punct.thousands_sep());
        if (sepsize != (std::size_t)-1)
        {
            s += _sepcount * sepsize;
        }
    }
    if (_fillcount > 0)
    {
        s += _fillcount * _encoding.char_size(_afmt.fill(), _epoli.err_hdl());
    }
    return s;
}

template <typename CharT>
int fmt_int_printer<CharT>::remaining_width(int w) const
{
    return w > _afmt.width() ? (w - _afmt.width()) : 0;
}

template <typename CharT>
bool fmt_int_printer<CharT>::write
        ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        return _write_complement(ob)
            && _write_digits(ob);
    }

    switch(_afmt.alignment())
    {
        case stringify::v0::alignment::left:
        {
            return _write_complement(ob)
                && _write_digits(ob)
                && _write_fill(ob, _fillcount);
        }
        case stringify::v0::alignment::internal:
        {
            return _write_complement(ob)
                && _write_fill(ob, _fillcount)
                && _write_digits(ob);
        }
        case stringify::v0::alignment::center:
        {
            auto halfcount = _fillcount / 2;
            return _write_fill(ob, halfcount)
                && _write_complement(ob)
                && _write_digits(ob)
                && _write_fill(ob, _fillcount - halfcount);
        }
        default:
        {
            return _write_fill(ob, _fillcount)
                && _write_complement(ob)
                && _write_digits(ob);
        }
    }
}

template <typename CharT>
inline bool fmt_int_printer<CharT>::_write_complement
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if(_showneg)
    {
        return _chars.print_neg_sign(ob, _encoding);
    }
    else if(_showpos)
    {
        return _chars.print_pos_sign(ob, _encoding);
    }
    else if (_showbase)
    {
        return _chars.print_base_indication(ob, _encoding);
    }
    return true;
}

template <typename CharT>
inline bool fmt_int_printer<CharT>::_write_digits
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if ( _precision > _digcount
      && ! _chars.print_zeros(ob, _encoding, _precision - _digcount) )
    {
        return false;
    }
    if (_sepcount == 0)
    {
        return _chars.print_digits(ob, _encoding, _uvalue, _digcount);
    }
    return _write_digits_sep(ob);
}

template <typename CharT>
bool fmt_int_printer<CharT>::_write_digits_sep
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    unsigned char grp_buff
        [stringify::v0::detail::max_num_digits< unsigned long long, 8>];
    auto* grp_it = _punct.groups(_digcount, grp_buff);
    (void)grp_it;
    BOOST_ASSERT((grp_it - grp_buff) == _sepcount);
    return _chars.print_digits( ob, _encoding, _uvalue, grp_buff
                              , _punct.thousands_sep()
                              , _digcount
                              , _sepcount + 1 );
}


template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::fmt_int_printer<CharT>
make_printer( const FPack& fp
            , const stringify::v0::int_with_format<IntT>& x )
{
    switch (x.base())
    {
        case 10: return {fp, x, std::integral_constant<int, 10>{}};
        case 16: return {fp, x, std::integral_constant<int, 16>{}};
        default: return {fp, x, std::integral_constant<int, 8>{}};
    }
}

inline auto make_fmt(stringify::v0::tag, short x)
{
    return stringify::v0::int_with_format<short>{{x}};
}
inline auto make_fmt(stringify::v0::tag, int x)
{
    return stringify::v0::int_with_format<int>{{x}};
}
inline auto make_fmt(stringify::v0::tag, long x)
{
    return stringify::v0::int_with_format<long>{{x}};
}
inline auto make_fmt(stringify::v0::tag, long long x)
{
    return stringify::v0::int_with_format<long long>{{x}};
}
inline auto make_fmt(stringify::v0::tag, unsigned short x)
{
    return stringify::v0::int_with_format<unsigned short>{{x}};
}
inline auto make_fmt(stringify::v0::tag, unsigned x)
{
    return  stringify::v0::int_with_format<unsigned>{{x}};
}
inline auto make_fmt(stringify::v0::tag, unsigned long x)
{
    return stringify::v0::int_with_format<unsigned long>{{x}};
}
inline auto make_fmt(stringify::v0::tag, unsigned long long x)
{
    return stringify::v0::int_with_format<unsigned long long>{{x}};
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

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED
