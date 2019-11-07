#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED

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

struct int_format_data
{
    unsigned precision = 0;
    bool showbase = false;
    bool showpos = false;
};

constexpr bool operator==( stringify::v0::int_format_data lhs
                         , stringify::v0::int_format_data rhs) noexcept
{
    return lhs.precision == rhs.precision
        && lhs.showbase == rhs.showbase
        && lhs.showpos == rhs.showpos;
}

constexpr bool operator!=( stringify::v0::int_format_data lhs
                         , stringify::v0::int_format_data rhs) noexcept
{
    return ! (lhs == rhs);
}

template <class T, int Base>
class int_format_fn
{
private:

    template <int OtherBase>
    using _adapted_derived_type
        = stringify::v0::fmt_replace<T, int_format<Base>, int_format<OtherBase> >;

public:

    constexpr int_format_fn()  noexcept = default;

    template <typename U, int OtherBase>
    constexpr int_format_fn(const int_format_fn<U, OtherBase> & u) noexcept
        : _data(u.get_int_format_data())
    {
    }

    template < int B = 16 >
    constexpr std::enable_if_t<Base == B && B == 16, T&&>
    hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr std::enable_if_t<Base != B && B == 16, _adapted_derived_type<B>>
    hex() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 10 >
    constexpr std::enable_if_t<Base == B && B == 10, T&&>
    dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr std::enable_if_t<Base != B && B == 10, _adapted_derived_type<B>>
    dec() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 8 >
    constexpr std::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 8 >
    constexpr std::enable_if_t<Base != B && B == 8, _adapted_derived_type<B>>
    oct() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    constexpr T&& p(unsigned _) && noexcept
    {
        _data.precision = _;
        return static_cast<T&&>(*this);
    }
    T&& operator+() && noexcept
    {
        _data.showpos = true;
        return static_cast<T&&>(*this);
    }
    constexpr T&& operator~() && noexcept
    {
        _data.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr static int base() noexcept
    {
        return Base;
    }
    constexpr unsigned precision() const noexcept
    {
        return _data.precision;
    }
    constexpr bool showbase() const noexcept
    {
        return _data.showbase;
    }
    constexpr bool showpos() const noexcept
    {
        return _data.showpos;
    }
    constexpr stringify::v0::int_format_data get_int_format_data() const noexcept
    {
        return _data;
    }

private:

    stringify::v0::int_format_data _data;
};

template <typename IntT>
struct int_tag
{
    IntT value;
};

template <int Base>
struct int_format
{
    template <typename T>
    using fn = stringify::v0::int_format_fn<T, Base>;
};

template <typename IntT, int Base = 10, bool Align = false>
using int_with_format = stringify::v0::value_with_format
    < stringify::v0::int_tag<IntT>
    , stringify::v0::int_format<Base>
    , stringify::v0::alignment_format_q<Align> >;

namespace detail {

template <typename CharT, typename FPack, typename IntT, unsigned Base>
class has_intpunct_impl
{
public:

    static std::true_type  test_numpunct(const stringify::v0::numpunct_base&);
    static std::false_type test_numpunct(const stringify::v0::default_numpunct<Base>&);
    static std::false_type test_numpunct(const stringify::v0::no_grouping<Base>&);

    static const FPack& fp();

    using has_numpunct_type = decltype
        ( test_numpunct
            ( get_facet< stringify::v0::numpunct_c<Base>, IntT >(fp())) );
public:

    static constexpr bool value = has_numpunct_type::value;
};

template <typename CharT, typename FPack, typename IntT, unsigned Base>
constexpr bool has_intpunct = has_intpunct_impl<CharT, FPack, IntT, Base>::value;

template <typename CharT>
class int_printer: public printer<CharT>
{
public:

    template <typename Preview, typename IntT>
    int_printer(Preview& preview, IntT value)
    {
        _negative = value < 0;
        _uvalue = stringify::v0::detail::unsigned_abs(value);
        _digcount = stringify::v0::detail::count_digits<10>(_uvalue);
        auto _size = _digcount + _negative;
        preview.subtract_width(static_cast<std::int16_t>(_size));
        preview.add_size(_size);
    }

    template <typename FP, typename Preview, typename IntT>
    int_printer(const FP&, Preview& preview, IntT value)
        : int_printer(preview, value)
    {
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    unsigned long long _uvalue;
    unsigned _digcount;
    bool _negative;
};

template <typename CharT>
void int_printer<CharT>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    unsigned size = _digcount + _negative;
    ob.ensure(size);
    CharT* it = write_int_dec_txtdigits_backwards(_uvalue, ob.pos() + size);
    if (_negative)
    {
        it[-1] = '-';
    }
    ob.advance(size);
}

template <typename CharT>
class punct_int_printer: public printer<CharT>
{
public:

    template <typename FPack, typename Preview, typename IntT>
    punct_int_printer
        ( const FPack& fp
        , Preview& preview
        , IntT value ) noexcept
        : _punct(get_facet<stringify::v0::numpunct_c<10>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        _negative = value < 0;
        _uvalue = stringify::v0::detail::unsigned_abs(value);
        _digcount = stringify::v0::detail::count_digits<10>(_uvalue);
        _sepcount = ( _punct.no_group_separation(_digcount)
                    ? 0
                    : _punct.thousands_sep_count(_digcount) );

        preview.subtract_width
            ( static_cast<std::int16_t>(_sepcount + _digcount + _negative) );

        _calc_size(preview);
    }

    std::size_t necessary_size() const;

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    void _calc_size(stringify::v0::size_preview<false>&) const
    {
    }

    void _calc_size(stringify::v0::size_preview<true>&) const;

    const stringify::v0::numpunct_base& _punct;
    stringify::v0::encoding<CharT> _encoding;
    unsigned long long _uvalue;
    unsigned _digcount;
    unsigned _sepcount;
    bool _negative;
};

template <typename CharT>
void punct_int_printer<CharT>::_calc_size
    ( stringify::v0::size_preview<true>& preview ) const
{
    std::size_t size = _digcount + _negative;
    if (_sepcount != 0)
    {
        auto sepsize = _encoding.validate(_punct.thousands_sep());
        if (sepsize != std::size_t(-1))
        {
            size += sepsize * _sepcount;
        }
    }
    preview.add_size(size);
}

template <typename CharT>
void punct_int_printer<CharT>::print_to(stringify::v0::basic_outbuf<CharT>& ob) const
{
    if (_sepcount == 0)
    {
        ob.ensure(_negative + _digcount);
        auto it = ob.pos();
        if (_negative)
        {
            *it = static_cast<CharT>('-');
            ++it;
        }
        it += _digcount;
        stringify::v0::detail::write_int_txtdigits_backwards<10>(_uvalue, it);
        ob.advance_to(it);
    }
    else
    {
        if (_negative)
        {
            put(ob, static_cast<CharT>('-'));
        }
        stringify::v0::detail::write_int<10>( ob, _punct, _encoding
                                            , _uvalue, _digcount );
    }
}

template <typename CharT, int Base>
class partial_fmt_int_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack, typename Preview, typename IntT>
    partial_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , const stringify::v0::int_with_format<IntT, Base, false>& value )
        : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        _init<IntT, detail::has_intpunct<CharT, FPack, IntT, Base>>
            ( value.value().value, value.get_int_format_data() );
        preview.subtract_width(width());
        calc_size(preview);
    }

    template <typename FPack, typename Preview, typename IntT>
    partial_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , const stringify::v0::int_with_format<IntT, Base, true>& value )
        : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        _init<IntT, detail::has_intpunct<CharT, FPack, IntT, Base>>
            ( value.value().value, value.get_int_format_data() );
        preview.subtract_width(width());
        calc_size(preview);
    }

    std::int16_t width() const
    {
        return static_cast<std::int16_t>( std::max(_precision, _digcount)
                                        + _prefixsize
                                        + static_cast<int>(_sepcount) );
    }

    auto encoding() const
    {
        return _encoding;
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;
    void calc_size(stringify::v0::size_preview<false>& ) const
    {
    }
    void calc_size(stringify::v0::size_preview<true>& ) const;

    void write_complement(stringify::v0::basic_outbuf<CharT>& ob) const;
    void write_digits(stringify::v0::basic_outbuf<CharT>& ob) const;

private:

    const stringify::v0::numpunct_base& _punct;
    const stringify::v0::encoding<CharT> _encoding;
    unsigned long long _uvalue = 0;
    unsigned _digcount = 0;
    unsigned _sepcount = 0;
    unsigned _precision = 0;
    bool _negative = false;
    std::uint8_t _prefixsize = 0;

    template <typename IntT, bool HasPunct>
    void _init(IntT value, stringify::v0::int_format_data fmt);
};

template <typename CharT, int Base>
template <typename IntT, bool HasPunct>
void partial_fmt_int_printer<CharT, Base>::_init
    ( IntT value
    , stringify::v0::int_format_data fmt )
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)
    {
        _negative = value < 0;
        _prefixsize = _negative || fmt.showpos;
        _uvalue = stringify::v0::detail::unsigned_abs(value);
    }
    else
    {
        _uvalue = unsigned_type(value);
        _negative = false;
        _prefixsize = static_cast<unsigned>(fmt.showbase) << static_cast<unsigned>(Base == 16);
    }
    _digcount = stringify::v0::detail::count_digits<Base>(_uvalue);
    _precision = fmt.precision;

    BOOST_STRINGIFY_IF_CONSTEXPR (HasPunct)
    {
        _sepcount = _punct.thousands_sep_count(_digcount);
    }
    else
    {
        _sepcount = 0;
    }
}

template <typename CharT, int Base>
void partial_fmt_int_printer<CharT, Base>::calc_size
    ( stringify::v0::size_preview<true>& preview ) const
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
    preview.add_size(s);
}

template <typename CharT, int Base>
inline void partial_fmt_int_printer<CharT, Base>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_sepcount == 0)
    {
        ob.ensure(_prefixsize + _digcount);
        auto it = ob.pos();
        if (_prefixsize != 0)
        {
            BOOST_STRINGIFY_IF_CONSTEXPR (Base == 10)
            {
                * it = static_cast<CharT>('+') + (_negative << 1);
                ++ it;
            }
            else BOOST_STRINGIFY_IF_CONSTEXPR (Base == 8)
            {
                * it = static_cast<CharT>('0');
                ++ it;
            }
            else
            {
                it[0] = static_cast<CharT>('0');
                it[1] = static_cast<CharT>('x');
                it += 2;
            }
        }
        if (_precision > _digcount)
        {
            ob.advance_to(it);
            unsigned zeros = _precision - _digcount;
            stringify::v0::detail::write_fill(ob, zeros, CharT('0'));
            it = ob.pos();
            ob.ensure(_digcount);
        }
        it += _digcount;
        stringify::v0::detail::write_int_txtdigits_backwards<Base>(_uvalue, it);
        ob.advance_to(it);
    }
    else
    {
        write_complement(ob);
        if (_precision > _digcount)
        {
            unsigned zeros = _precision - _digcount;
            stringify::v0::detail::write_fill(ob, zeros, CharT('0'));
        }
        stringify::v0::detail::write_int<Base>( ob, _punct, _encoding
                                              , _uvalue, _digcount );
    }
}

template <typename CharT, int Base>
inline void partial_fmt_int_printer<CharT, Base>::write_complement
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
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
inline void partial_fmt_int_printer<CharT, Base>::write_digits
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_precision > _digcount)
    {
        unsigned zeros = _precision - _digcount;
        stringify::v0::detail::write_fill(ob, zeros, CharT('0'));
    }
    if (_sepcount == 0)
    {
        stringify::v0::detail::write_int<Base>(ob, _uvalue, _digcount);
    }
    else
    {
        stringify::v0::detail::write_int<Base>( ob, _punct, _encoding
                                              , _uvalue, _digcount );
    }
}

template <typename CharT, int Base>
class full_fmt_int_printer: public printer<CharT>
{
public:

    template <typename FPack, typename Preview, typename IntT>
    full_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , stringify::v0::int_with_format<IntT, Base, true> value ) noexcept;

    ~full_fmt_int_printer();

    void print_to( stringify::v0::basic_outbuf<CharT>& ob ) const override;

private:

    stringify::v0::detail::partial_fmt_int_printer<CharT, Base> _ichars;
    unsigned _fillcount = 0;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::alignment_format_data _afmt;
    stringify::v0::surrogate_policy _allow_surr;

    void _calc_fill_size(stringify::v0::size_preview<false>&) const
    {
    }

    void _calc_fill_size(stringify::v0::size_preview<true>& preview) const
    {
        if (_fillcount > 0)
        {
            preview.add_size( _fillcount
                            * _ichars.encoding().char_size(_afmt.fill, _enc_err) );
        }
    }

    void _write_fill
        ( stringify::v0::basic_outbuf<CharT>& ob
        , std::size_t count ) const
    {
        return _ichars.encoding().encode_fill
            ( ob, count, _afmt.fill, _enc_err, _allow_surr );
    }
};

template <typename CharT, int Base>
template <typename FPack, typename Preview, typename IntT>
inline full_fmt_int_printer<CharT, Base>::full_fmt_int_printer
    ( const FPack& fp
    , Preview& preview
    , stringify::v0::int_with_format<IntT, Base, true> value ) noexcept
    : _ichars(fp, preview, value)
    , _enc_err(get_facet<stringify::v0::encoding_error_c, IntT>(fp))
    , _afmt(value.get_alignment_format_data())
    , _allow_surr(get_facet<stringify::v0::surrogate_policy_c, IntT>(fp))
{
    auto content_width = _ichars.width();
    if (_afmt.width > content_width)
    {
        _fillcount = _afmt.width - content_width;
        preview.subtract_width(_fillcount);
    }
    _calc_fill_size(preview);
}

template <typename CharT, int Base>
full_fmt_int_printer<CharT, Base>::~full_fmt_int_printer()
{
}

template <typename CharT, int Base>
void full_fmt_int_printer<CharT, Base>::print_to
        ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        _ichars.print_to(ob);
    }
    else
    {
        switch(_afmt.alignment)
        {
            case stringify::v0::text_alignment::left:
            {
                _ichars.print_to(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::text_alignment::split:
            {
                _ichars.write_complement(ob);
                _write_fill(ob, _fillcount);
                _ichars.write_digits(ob);
                break;
            }
            case stringify::v0::text_alignment::center:
            {
                auto halfcount = _fillcount / 2;
                _write_fill(ob, halfcount);
                _ichars.print_to(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _ichars.print_to(ob);
            }
        }
    }
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_int_printer<char8_t>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t, 16>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t,  8>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t, 10>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t, 16>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class int_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_int_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_int_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_int_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_int_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, short, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, short x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, int, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, int x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, long x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, long long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, long long x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned short, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, unsigned short x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned int, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, unsigned int x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, unsigned long x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned long long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, Preview& preview, unsigned long long x)
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview, typename IntT, int Base>
inline stringify::v0::detail::full_fmt_int_printer<CharT, Base>
make_printer( const FPack& fp
            , Preview& preview
            , const stringify::v0::int_with_format<IntT, Base, true>& x )
{
    return {fp, preview, x};
}

template <typename CharT, typename FPack, typename Preview, typename IntT, int Base>
inline stringify::v0::detail::partial_fmt_int_printer<CharT, Base>
make_printer( const FPack& fp
            , Preview& preview
            , const stringify::v0::int_with_format<IntT, Base, false>& x )
{
    return {fp, preview, x};
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
