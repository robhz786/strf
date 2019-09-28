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

template <int Base>
struct int_format;

namespace detail {

template <class T, int Base>
class int_format_fn
{
    using helper = stringify::v0::fmt_helper<int_format<Base>, T>;

public:

    using derived_type = typename helper::derived_type;

private:

    template <int OtherBase>
    using adapted_derived_type
    = typename helper::template adapted_derived_type<int_format<OtherBase> >;

    template <int OtherBase>
    derived_type&& to_base(std::true_type /*same_base*/) &&
    {
        return static_cast<derived_type&&>(*this);
    }
    template <int OtherBase>
    const derived_type& to_base(std::true_type /*same_base*/) const &
    {
        return static_cast<const derived_type&>(*this);
    }
    template <int OtherBase>
    derived_type& to_base(std::true_type /*same_base*/) &
    {
        return static_cast<derived_type&>(*this);
    }
    template <int OtherBase>
    adapted_derived_type<OtherBase> to_base(std::false_type /*same_base*/) const &
    {
        return adapted_derived_type<OtherBase>
            { static_cast<const derived_type&>(*this) };
    }

    template <int OtherBase>
    using base_eq = std::integral_constant<bool, Base == OtherBase>;

    template <int OtherBase>
    decltype(auto) to_base() &&
    {
        return static_cast<int_format_fn&&>(*this)
            .to_base<OtherBase>(base_eq<OtherBase>{});
    }

    template <int OtherBase>
    decltype(auto) to_base() &
    {
        return to_base<OtherBase>(base_eq<OtherBase>{});
    }

    template <int OtherBase>
    decltype(auto) to_base() const &
    {
        return to_base<OtherBase>(base_eq<OtherBase>{});
    }

public:

    constexpr int_format_fn() = default;

    template <typename U, int OtherBase>
    constexpr int_format_fn(const int_format_fn<U, OtherBase> & u)
        : _precision(u.precision())
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
    constexpr decltype(auto) hex() &&
    {
        return static_cast<int_format_fn&&>(*this).to_base<16>();
    }
    constexpr decltype(auto) hex() &
    {
        return to_base<16>();
    }
    constexpr decltype(auto) dec() &&
    {
        return static_cast<int_format_fn&&>(*this).to_base<10>();
    }
    constexpr decltype(auto) dec() &
    {
        return to_base<10>();
    }
    constexpr decltype(auto) oct() &&
    {
        return static_cast<int_format_fn&&>(*this).to_base<8>();
    }
    constexpr decltype(auto) oct() &
    {
        return to_base<8>();
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
    constexpr int base() const
    {
        return Base;
    }
    constexpr unsigned precision() const
    {
        return _precision;
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
    bool _showbase = false;
    bool _showpos = false;
};

template <typename IntT>
struct int_value
{
    IntT value;
};

} // namespace detail

template <int Base>
struct int_format
{
    template <typename T>
    using fn = stringify::v0::detail::int_format_fn<T, Base>;
};

template <typename IntT, int Base = 10>
using int_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::int_value<IntT>
    , stringify::v0::int_format<Base>
    , stringify::v0::alignment_format >;

namespace detail {

template <typename CharT, int Base>
class fmt_int_printer: public printer<CharT>
{
public:

    template <typename FPack, typename IntT>
    fmt_int_printer
        ( const FPack& fp
        , stringify::v0::int_with_format<IntT, Base> value ) noexcept;

    ~fmt_int_printer();

    std::size_t necessary_size() const override;

    void write( boost::basic_outbuf<CharT>& ob ) const override;

    int width(int) const override;

private:

    unsigned long long _uvalue;
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
    std::uint8_t _prefixsize;

    template <typename IntT, bool NoGroupSep>
    void _init( stringify::v0::int_with_format<IntT, Base> value
              , std::integral_constant<bool, NoGroupSep> );

    void _write_fill
        ( boost::basic_outbuf<CharT>& ob
        , std::size_t count ) const
    {
        return _encoding.encode_fill
            ( ob, count, _afmt.fill(), _enc_err, _allow_surr );
    }

    void _write_complement(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits_no_sep(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits_with_sep(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits_little_sep(boost::basic_outbuf<CharT>& ob) const;
    void _write_digits_big_sep( boost::basic_outbuf<CharT>& ob
                              , std::size_t sep_size ) const;
};

template <typename CharT, int Base>
template <typename FPack, typename IntT>
inline fmt_int_printer<CharT, Base>::fmt_int_printer
    ( const FPack& fp
    , stringify::v0::int_with_format<IntT, Base> value ) noexcept
    : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
    , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    , _enc_err(get_facet<stringify::v0::encoding_error_c, IntT>(fp))
    , _afmt(value)
    , _allow_surr(get_facet<stringify::v0::surrogate_policy_c, IntT>(fp))
{
    using no_group_sep =
        decltype(detail::has_no_grouping
                 (get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp)));
    _init<IntT>(value, no_group_sep{});
}

#if defined(_MSC_VER)
#pragma warning ( push )
#pragma warning ( disable : 4127 )
#endif // defined(_MSC_VER)

template <typename CharT, int Base>
template <typename IntT, bool NoGroupSep>
void fmt_int_printer<CharT, Base>::_init
    ( stringify::v0::int_with_format<IntT, Base> value
    , std::integral_constant<bool, NoGroupSep> )
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;

    BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)
    {
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        unsigned_IntT uvalue = 1 + unsigned_IntT(-(value.value().value + 1));
        _negative = value.value().value < 0;
        _prefixsize = _negative || value.showpos();
        _uvalue = _negative * uvalue + (!_negative) * value.value().value;
    }
    else
    {
        _uvalue = unsigned_type(value.value().value);
        _negative = false;
        _prefixsize = value.showbase() << (Base == 16);
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

    int content_width = std::max(_precision, _digcount)
            + _prefixsize
            + static_cast<int>(_sepcount);

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

template <typename CharT, int Base>
fmt_int_printer<CharT, Base>::~fmt_int_printer()
{
}

template <typename CharT, int Base>
std::size_t fmt_int_printer<CharT, Base>::necessary_size() const
{
    std::size_t s = std::max(_precision, _digcount) + _prefixsize;

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

template <typename CharT, int Base>
int fmt_int_printer<CharT, Base>::width(int) const
{
    return _afmt.width();
}

template <typename CharT, int Base>
void fmt_int_printer<CharT, Base>::write
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

template <typename CharT, int Base>
inline void fmt_int_printer<CharT, Base>::_write_complement
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_prefixsize != 0)
    {
        ob.ensure(_prefixsize);
        BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)
        {
            * ob.pos() = static_cast<CharT>('+') + (_negative << 1);
            ob.advance(1);
        }
        else BOOST_STRINGIFY_IF_CONSTEXPR (Base == 8)
        {
            * ob.pos() = static_cast<CharT>('0');
            ob.advance(1);
        }
        else
        {
            ob.pos()[0] = static_cast<CharT>('0');
            ob.pos()[1] = static_cast<CharT>('x');
            ob.advance(2);
        }
    }
}

template <typename CharT, int Base>
inline void fmt_int_printer<CharT, Base>::_write_digits
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_precision > _digcount)
    {
        unsigned zeros = _precision - _digcount;
        stringify::v0::detail::write_fill(ob, zeros, CharT('0'));
    }
    if (_sepcount == 0)
    {
        _write_digits_no_sep(ob);
    }
    else
    {
        _write_digits_with_sep(ob);
    }
}

template <typename CharT, int Base>
inline void fmt_int_printer<CharT, Base>::_write_digits_no_sep
    ( boost::basic_outbuf<CharT>& ob ) const
{
    ob.ensure(_digcount);
    stringify::v0::detail::write_int_txtdigits_backwards<Base>
        ( _uvalue, ob.pos() + _digcount );
    ob.advance(_digcount);
}

template <typename CharT, int Base>
void fmt_int_printer<CharT, Base>::_write_digits_with_sep
    ( boost::basic_outbuf<CharT>& ob ) const
{
    auto sep_size = _encoding.validate(_punct.thousands_sep());
    constexpr auto max_digits = detail::max_num_digits< decltype(_uvalue)
                                                      , Base >;
    unsigned char groups_buff[max_digits];
    const auto num_groups = _punct.groups(_digcount, groups_buff);
    (void) num_groups;
    BOOST_ASSERT(num_groups == _sepcount + 1);

    switch(sep_size)
    {
        case 1:
            _write_digits_little_sep(ob);
            break;
        case (std::size_t)-1:
            _write_digits_no_sep(ob);
            break;
        default:
            _write_digits_big_sep(ob, sep_size);
    }
}

template <typename CharT, int Base>
void fmt_int_printer<CharT, Base>::_write_digits_little_sep
    ( boost::basic_outbuf<CharT>& ob ) const
{
    BOOST_ASSERT(1 == _encoding.validate(_punct.thousands_sep()));

    constexpr auto max_digits = detail::max_num_digits<unsigned long long, Base>;
    std::uint8_t groups_buff[max_digits];

    const auto num_groups = _punct.groups(_digcount, groups_buff);
    (void) num_groups;
    BOOST_ASSERT(num_groups == _sepcount + 1);

    CharT sep;
    _encoding.encode_char(&sep, _punct.thousands_sep());

    auto size = _digcount + _sepcount;
    detail::write_int_txtdigits_backwards_little_sep<Base>
        ( _uvalue, ob.pos() + size, sep, groups_buff );
    ob.advance(size);
}

template <typename CharT>
void write_digits_big_sep
    ( boost::basic_outbuf<CharT>& ob
    , const stringify::v0::encoding<CharT> encoding
    , const std::uint8_t* last_grp
    , unsigned char* digits
    , unsigned num_digits
    , char32_t sep
    , std::size_t sep_size )
{
    BOOST_ASSERT(sep_size != (std::size_t)-1);
    BOOST_ASSERT(sep_size != 1);
    BOOST_ASSERT(sep_size == encoding.validate(sep));

    ob.ensure(1);

    auto pos = ob.pos();
    auto end = ob.end();
    auto grp_it = last_grp;
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
            pos = encoding.encode_char(pos, sep);
            n = *--grp_it;
        }
    }
    ob.advance_to(pos);
}

template <typename CharT, int Base>
void fmt_int_printer<CharT, Base>::_write_digits_big_sep
    ( boost::basic_outbuf<CharT>& ob, std::size_t sep_size ) const
{
    constexpr auto max_digits = detail::max_num_digits<unsigned long long, Base>;
    unsigned char digits_buff[max_digits];
    unsigned char groups_buff[max_digits];

    const auto num_groups = _punct.groups(_digcount, groups_buff);
    BOOST_ASSERT(num_groups == _sepcount + 1);

    const auto dig_end = digits_buff + max_digits;
    auto digits = stringify::v0::detail::write_int_txtdigits_backwards<Base>
        ( _uvalue, dig_end );

    stringify::v0::detail::write_digits_big_sep
        ( ob, _encoding, groups_buff + num_groups - 1, digits, _digcount
        , _punct.thousands_sep(), sep_size );
}


#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char8_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char8_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char8_t, 16>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char16_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char16_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char16_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char32_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char32_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<char32_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<wchar_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<wchar_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<wchar_t, 16>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename IntT, int Base>
inline stringify::v0::detail::fmt_int_printer<CharT, Base>
make_printer( const FPack& fp
            , const stringify::v0::int_with_format<IntT, Base>& x )
{
    return {fp, x};
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
