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

namespace detail {
template <typename IntT>
struct int_value
{
    IntT value;
};
} // namespace detail


template <typename IntT>
using int_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::int_value<IntT>
    , stringify::v0::int_format
    , stringify::v0::alignment_format >;

namespace detail {

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

    void write( boost::basic_outbuf<CharT>& ob ) const override;

    int width(int) const override;

private:

    unsigned long long _uvalue;
    const stringify::v0::numchars<CharT>& _chars;
    const stringify::v0::numpunct_base& _punct;
    const stringify::v0::encoding<CharT> _encoding;
    unsigned _digcount;
    unsigned _sepcount;
    unsigned _fillcount;
    unsigned _precision;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::alignment_format::fn<void> _afmt;
    stringify::v0::surrogate_policy _allow_surr;
    bool _negative;
    bool _showsign;
    bool _showbase;

    template <typename IntT, int Base, bool NoGroupSep, bool DefaultChars>
    void _init( stringify::v0::int_with_format<IntT> value
              , std::integral_constant<bool, NoGroupSep>
              , std::integral_constant<bool, DefaultChars> );

    void _write_fill
        ( boost::basic_outbuf<CharT>& ob
        , std::size_t count ) const
    {
        return _encoding.encode_fill
            ( ob, count, _afmt.fill(), _enc_err, _allow_surr );
    }

    void _write_complement(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits(boost::basic_outbuf<CharT>& ob) const;
};

template <typename CharT>
template <typename FPack, typename IntT, int Base>
inline fmt_int_printer<CharT>::fmt_int_printer
    ( const FPack& fp
    , stringify::v0::int_with_format<IntT> value
    , std::integral_constant<int, Base> ) noexcept
    : _chars(get_facet<stringify::v0::numchars_c<CharT, Base>, IntT>(fp))
    , _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
    , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    , _enc_err(get_facet<stringify::v0::encoding_error_c, IntT>(fp))
    , _afmt(value)
    , _allow_surr(get_facet<stringify::v0::surrogate_policy_c, IntT>(fp))
{
    using no_group_sep =
        decltype(detail::has_no_grouping
                 (get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp)));
    using numchars_type =
        decltype(get_facet<stringify::v0::numchars_c<CharT, Base>, IntT>(fp));
    using numchars_default =
        std::is_same< const numchars_type&
                    , const stringify::v0::default_numchars<CharT, Base>& >;
    _init<IntT, Base>(value, no_group_sep{}, numchars_default{});
}

#if defined(_MSC_VER)
#pragma warning ( push )
#pragma warning ( disable : 4127 )
#endif // defined(_MSC_VER)

template <typename CharT>
template <typename IntT, int Base, bool NoGroupSep, bool DefaultChars>
void fmt_int_printer<CharT>::_init
    ( stringify::v0::int_with_format<IntT> value
    , std::integral_constant<bool, NoGroupSep>
    , std::integral_constant<bool, DefaultChars> )
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;

    BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)
    {
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        unsigned_IntT uvalue = 1 + unsigned_IntT(-(value.value().value + 1));
        _negative = value.value().value < 0;
        _showsign = _negative || value.showpos();
        _uvalue = _negative * uvalue + (!_negative) * value.value().value;
        _showbase = false;
    }
    else
    {
        _uvalue = unsigned_type(value.value().value);
        _negative = false;
        _showsign = false;
        _showbase = value.showbase();
    }
    _digcount = stringify::v0::detail::count_digits<Base>(_uvalue);
    _precision = value.precision();

    BOOST_STRINGIFY_IF_CONSTEXPR (NoGroupSep)
    {
        _sepcount = 0;
    }
    else
    {
        _sepcount = _punct.thousands_sep_count(_digcount);
    }

    int content_width = 0;
    BOOST_STRINGIFY_IF_CONSTEXPR (DefaultChars)
    {
        content_width = std::max(_precision, _digcount)
            + _showsign
            + (_showbase << (Base == 16))
            + static_cast<int>(_sepcount);
    }
    else
    {
         content_width = _chars.integer_printwidth
             ( std::max(_precision, _digcount)
             , _showsign
             , _showbase )
             + static_cast<int>(_sepcount);
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

#if defined(_MSC_VER)
#pragma warning ( pop )
#endif // defined(_MSC_VER)

template <typename CharT>
fmt_int_printer<CharT>::~fmt_int_printer()
{
}

template <typename CharT>
std::size_t fmt_int_printer<CharT>::necessary_size() const
{
    std::size_t s = _chars.integer_printsize
        ( _encoding, std::max((unsigned)_digcount, _precision)
        , _showsign, _showbase );
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
        s += _fillcount * _encoding.char_size(_afmt.fill(), _enc_err);
    }
    return s;
}

template <typename CharT>
int fmt_int_printer<CharT>::width(int) const
{
    return _afmt.width();
}

template <typename CharT>
void fmt_int_printer<CharT>::write
        ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        _write_complement(ob);
        _write_digits(ob);
    }
    else
    {
        switch(_afmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                _write_complement(ob);
                _write_digits(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::alignment::internal:
            {
                _write_complement(ob);
                _write_fill(ob, _fillcount);
                _write_digits(ob);
                break;
            }
            case stringify::v0::alignment::center:
            {
                auto halfcount = _fillcount / 2;
                _write_fill(ob, halfcount);
                _write_complement(ob);
                _write_digits(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _write_complement(ob);
                _write_digits(ob);
            }
        }
    }
}

template <typename CharT>
inline void fmt_int_printer<CharT>::_write_complement
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if(_showsign)
    {
        _chars.print_sign(ob, _encoding, _negative);
    }
    else if (_showbase)
    {
        _chars.print_base_indication(ob, _encoding);
    }
}

template <typename CharT>
inline void fmt_int_printer<CharT>::_write_digits
    ( boost::basic_outbuf<CharT>& ob ) const
{
    unsigned zeros = (_precision > _digcount) * (_precision - _digcount);
    if (_sepcount == 0)
    {
        _chars.print_integer(ob, _encoding, _uvalue, _digcount, zeros);
    }
    else
    {
        unsigned char grp_buff
        [stringify::v0::detail::max_num_digits< unsigned long long, 8>];
        _chars.print_integer( ob, _encoding, _punct, grp_buff
                            , _uvalue, _digcount, zeros );
    }
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char8_t>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::detail::fmt_int_printer<CharT>
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

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED
