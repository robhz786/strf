#ifndef STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/encoding.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/int_digits.hpp>
#include <strf/detail/standard_lib_functions.hpp>
#include <cstdint>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/

namespace strf {

template <int Base>
struct int_format;

struct int_format_data
{
    unsigned precision = 0;
    bool showbase = false;
    bool showpos = false;
};

constexpr STRF_HD bool operator==( strf::int_format_data lhs
                                 , strf::int_format_data rhs) noexcept
{
    return lhs.precision == rhs.precision
        && lhs.showbase == rhs.showbase
        && lhs.showpos == rhs.showpos;
}

constexpr STRF_HD bool operator!=( strf::int_format_data lhs
                                 , strf::int_format_data rhs) noexcept
{
    return ! (lhs == rhs);
}

template <class T, int Base>
class int_format_fn
{
private:

    template <int OtherBase>
    using _adapted_derived_type
        = strf::fmt_replace<T, int_format<Base>, int_format<OtherBase> >;

public:

    constexpr STRF_HD int_format_fn()  noexcept { }

    template <typename U, int OtherBase>
    constexpr STRF_HD int_format_fn(const int_format_fn<U, OtherBase> & u) noexcept
        : _data(u.get_int_format_data())
    {
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 16, T&&>
    hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 16, _adapted_derived_type<B>>
    hex() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 10, T&&>
    dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 10, _adapted_derived_type<B>>
    dec() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 8, _adapted_derived_type<B>>
    oct() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 2, T&&>
    bin() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 2, _adapted_derived_type<B>>
    bin() &&
    {
        return _adapted_derived_type<B>{ static_cast<const T&>(*this) };
    }

    constexpr STRF_HD T&& p(unsigned _) && noexcept
    {
        _data.precision = _;
        return static_cast<T&&>(*this);
    }
    STRF_HD T&& operator+() && noexcept
    {
        _data.showpos = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& operator~() && noexcept
    {
        _data.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr static STRF_HD int base() noexcept
    {
        return Base;
    }
    constexpr STRF_HD unsigned precision() const noexcept
    {
        return _data.precision;
    }
    constexpr STRF_HD bool showbase() const noexcept
    {
        return _data.showbase;
    }
    constexpr STRF_HD bool showpos() const noexcept
    {
        return _data.showpos;
    }
    constexpr STRF_HD strf::int_format_data get_int_format_data() const noexcept
    {
        return _data;
    }

private:

    strf::int_format_data _data;
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
    using fn = strf::int_format_fn<T, Base>;
};

template <typename IntT, int Base = 10, bool Align = false>
using int_with_format = strf::value_with_format
    < strf::int_tag<IntT>
    , strf::int_format<Base>
    , strf::alignment_format_q<Align> >;

namespace detail {

template <typename FPack, typename IntT, unsigned Base>
class has_intpunct_impl
{
public:

    static STRF_HD std::true_type  test_numpunct(const strf::numpunct_base&);
    static STRF_HD std::false_type test_numpunct(const strf::default_numpunct<Base>&);
    static STRF_HD std::false_type test_numpunct(const strf::no_grouping<Base>&);

    static const FPack& fp();

    using has_numpunct_type = decltype
        ( test_numpunct
            ( get_facet< strf::numpunct_c<Base>, IntT >(fp())) );
public:

    static constexpr bool value = has_numpunct_type::value;
};

template <typename FPack, typename IntT, unsigned Base>
constexpr bool has_intpunct = has_intpunct_impl<FPack, IntT, Base>::value;

template <std::size_t CharSize>
class int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename Preview, typename IntT>
    STRF_HD int_printer(Preview& preview, IntT value)
    {
        _negative = value < 0;
        _uvalue = strf::detail::unsigned_abs(value);
        _digcount = strf::detail::count_digits<10>(_uvalue);
        auto _size = _digcount + _negative;
        preview.subtract_width(static_cast<std::int16_t>(_size));
        preview.add_size(_size);
    }

    template <typename FP, typename Preview, typename IntT, typename CharT>
    STRF_HD int_printer(const FP&, Preview& preview, IntT value, strf::tag<CharT>)
        : int_printer(preview, value)
    {
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    unsigned long long _uvalue;
    unsigned _digcount;
    bool _negative;
};

template <std::size_t CharSize>
STRF_HD void int_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    unsigned size = _digcount + _negative;
    ob.ensure(size);
    auto* it = write_int_dec_txtdigits_backwards(_uvalue, ob.pos() + size);
    if (_negative) {
        it[-1] = '-';
    }
    ob.advance(size);
}

template <std::size_t CharSize>
class punct_int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename IntT, typename CharT>
    STRF_HD punct_int_printer
        ( const FPack& fp
        , Preview& preview
        , IntT value
        , strf::tag<CharT> ) noexcept
        : _punct(get_facet<strf::numpunct_c<10>, IntT>(fp))
        , _encoding(get_facet<strf::encoding_c<CharT>, IntT>(fp).as_underlying())
    {
        _negative = value < 0;
        _uvalue = strf::detail::unsigned_abs(value);
        _digcount = strf::detail::count_digits<10>(_uvalue);
        _sepcount = ( _punct.no_group_separation(_digcount)
                    ? 0
                    : _punct.thousands_sep_count(_digcount) );

        preview.subtract_width
            ( static_cast<std::int16_t>(_sepcount + _digcount + _negative) );

        _calc_size(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    STRF_HD void _calc_size(strf::size_preview<false>&) const
    {
    }

    STRF_HD void _calc_size(strf::size_preview<true>&) const;

    const strf::numpunct_base& _punct;
    const strf::underlying_encoding<CharSize>& _encoding;
    unsigned long long _uvalue;
    unsigned _digcount;
    unsigned _sepcount;
    bool _negative;
};

template <std::size_t CharSize>
STRF_HD void punct_int_printer<CharSize>::_calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t size = _digcount + _negative;
    if (_sepcount != 0) {
        auto sepsize = _encoding.validate(_punct.thousands_sep());
        if (sepsize != std::size_t(-1)) {
            size += sepsize * _sepcount;
        }
    }
    preview.add_size(size);
}

template <std::size_t CharSize>
STRF_HD void punct_int_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    if (_sepcount == 0) {
        ob.ensure(_negative + _digcount);
        auto it = ob.pos();
        if (_negative) {
            *it = static_cast<char_type>('-');
            ++it;
        }
        it += _digcount;
        strf::detail::write_int_dec_txtdigits_backwards(_uvalue, it);
        ob.advance_to(it);
    } else {
        if (_negative) {
            put(ob, static_cast<char_type>('-'));
        }
        strf::detail::write_int<10>( ob, _punct, _encoding, _uvalue
                                   , _digcount, strf::lowercase );
    }
}

template <std::size_t CharSize, int Base>
class partial_fmt_int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename IntT, typename CharT>
    STRF_HD partial_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::int_with_format<IntT, Base, false>& value
        , strf::tag<CharT> tag_char )
        : partial_fmt_int_printer( fp, preview, value.value().value
                                 , value.get_int_format_data()
                                 , tag_char )
    {
    }

    template < typename FPack
             , typename Preview
             , typename IntT
             , typename CharT
             , typename IntTag = IntT >
    STRF_HD partial_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , IntT value
        , int_format_data fdata
        , strf::tag<CharT>
        , strf::tag<IntT, IntTag> = strf::tag<IntT, IntTag>{} )
        : _punct(get_facet<strf::numpunct_c<Base>, IntTag>(fp))
        , _encoding(get_facet<strf::encoding_c<CharT>, IntTag>(fp).as_underlying())
        , _lettercase(get_facet<strf::lettercase_c, IntTag>(fp))
    {
        _init<IntT, detail::has_intpunct<FPack, IntTag, Base>>
            ( value, fdata );
        preview.subtract_width(width());
        calc_size(preview);
    }

    STRF_HD std::int16_t width() const
    {
        return static_cast<std::int16_t>( strf::detail::max(_precision, _digcount)
                                        + _prefixsize
                                        + static_cast<int>(_sepcount) );
    }

    STRF_HD auto encoding() const
    {
        return _encoding;
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;
    STRF_HD void calc_size(strf::size_preview<false>& ) const
    {
    }
    STRF_HD void calc_size(strf::size_preview<true>& ) const;

    STRF_HD void write_complement(strf::underlying_outbuf<CharSize>& ob) const;
    STRF_HD void write_digits(strf::underlying_outbuf<CharSize>& ob) const;

private:

    const strf::numpunct_base& _punct;
    const strf::underlying_encoding<CharSize>& _encoding;
    unsigned long long _uvalue = 0;
    unsigned _digcount = 0;
    unsigned _sepcount = 0;
    unsigned _precision = 0;
    bool _negative = false;
    std::uint8_t _prefixsize = 0;
    strf::lettercase _lettercase;

    template <typename IntT, bool HasPunct>
    STRF_HD  void _init(IntT value, strf::int_format_data fmt);
};

template <std::size_t CharSize, int Base>
template <typename IntT, bool HasPunct>
STRF_HD void partial_fmt_int_printer<CharSize, Base>::_init
    ( IntT value
    , strf::int_format_data fmt )
{
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    STRF_IF_CONSTEXPR (Base == 10) {
        _negative = value < 0;
        _prefixsize = _negative || fmt.showpos;
        _uvalue = strf::detail::unsigned_abs(value);
    } else {
        _uvalue = unsigned_type(value);
        _negative = false;
        _prefixsize = static_cast<unsigned>(fmt.showbase)
            << static_cast<unsigned>(Base == 16 || Base == 2);
    }
    _digcount = strf::detail::count_digits<Base>(_uvalue);
    _precision = fmt.precision;

    STRF_IF_CONSTEXPR (HasPunct) {
        _sepcount = _punct.thousands_sep_count(_digcount);
    } else {
        _sepcount = 0;
    }
}

template <std::size_t CharSize, int Base>
STRF_HD void partial_fmt_int_printer<CharSize, Base>::calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t s = strf::detail::max(_precision, _digcount) + _prefixsize;
    if (_sepcount > 0) {
        auto sepsize = _encoding.validate(_punct.thousands_sep());
        if (sepsize != (std::size_t)-1) {
            s += _sepcount * sepsize;
        }
    }
    preview.add_size(s);
}

template <std::size_t CharSize, int Base>
STRF_HD inline void partial_fmt_int_printer<CharSize, Base>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_sepcount == 0) {
        ob.ensure(_prefixsize + _digcount);
        auto it = ob.pos();
        if (_prefixsize != 0) {
            STRF_IF_CONSTEXPR (Base == 10) {
                * it = static_cast<char_type>('+') + (_negative << 1);
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 8) {
                * it = static_cast<char_type>('0');
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 16) {
                it[0] = static_cast<char_type>('0');
                it[1] = static_cast<char_type>
                    ('X' | ((_lettercase != strf::uppercase) << 5));
                it += 2;
            } else {
                it[0] = static_cast<char_type>('0');
                it[1] = static_cast<char_type>
                    ('B' | ((_lettercase != strf::uppercase) << 5));
                it += 2;
            }
        }
        ob.advance_to(it);
        if (_precision > _digcount) {
            unsigned zeros = _precision - _digcount;
            strf::detail::write_fill(ob, zeros, char_type('0'));
        }
        strf::detail::write_int<Base>(ob, _uvalue, _digcount, _lettercase);
    } else {
        write_complement(ob);
        if (_precision > _digcount) {
            unsigned zeros = _precision - _digcount;
            strf::detail::write_fill(ob, zeros, char_type('0'));
        }
        strf::detail::write_int<Base>( ob, _punct, _encoding
                                     , _uvalue, _digcount, _lettercase );
    }
}

template <std::size_t CharSize, int Base>
inline STRF_HD void partial_fmt_int_printer<CharSize, Base>::write_complement
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_prefixsize != 0) {
        ob.ensure(_prefixsize);
        STRF_IF_CONSTEXPR (Base == 10) {
            * ob.pos() = static_cast<char_type>('+') + (_negative << 1);
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 8) {
            * ob.pos() = static_cast<char_type>('0');
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 16) {
            ob.pos()[0] = static_cast<char_type>('0');
            ob.pos()[1] = static_cast<char_type>
                ('X' | ((_lettercase != strf::uppercase) << 5));
            ob.advance(2);
        } else {
            ob.pos()[0] = static_cast<char_type>('0');
            ob.pos()[1] = static_cast<char_type>
                ('B' | ((_lettercase != strf::uppercase) << 5));
            ob.advance(2);
        }
    }
}

template <std::size_t CharSize, int Base>
inline STRF_HD void partial_fmt_int_printer<CharSize, Base>::write_digits
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_precision > _digcount) {
        unsigned zeros = _precision - _digcount;
        strf::detail::write_fill(ob, zeros, char_type('0'));
    }
    if (_sepcount == 0) {
        strf::detail::write_int<Base>(ob, _uvalue, _digcount, _lettercase);
    } else {
        strf::detail::write_int<Base>( ob, _punct, _encoding
                                     , _uvalue, _digcount, _lettercase );
    }
}

template <std::size_t CharSize, int Base>
class full_fmt_int_printer: public printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename IntT, typename CharT>
    STRF_HD full_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , strf::int_with_format<IntT, Base, true> value
        , strf::tag<CharT> ) noexcept;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD full_fmt_int_printer
        ( const FPack& fp
        , Preview& preview
        , const void* value
        , strf::alignment_format_data afdata
        , strf::tag<CharT> );

    STRF_HD ~full_fmt_int_printer();

    STRF_HD void print_to( strf::underlying_outbuf<CharSize>& ob ) const override;

private:

    strf::detail::partial_fmt_int_printer<CharSize, Base> _ichars;
    unsigned _fillcount = 0;
    strf::encoding_error _enc_err;
    strf::alignment_format_data _afmt;
    strf::surrogate_policy _allow_surr;

    STRF_HD  void _calc_fill_size(strf::size_preview<false>&) const
    {
    }

    STRF_HD  void _calc_fill_size(strf::size_preview<true>& preview) const
    {
        if (_fillcount > 0) {
            preview.add_size( _fillcount
                            * _ichars.encoding().char_size(_afmt.fill) );
        }
    }

    STRF_HD  void _write_fill
        ( strf::underlying_outbuf<CharSize>& ob
        , std::size_t count ) const
    {
        return _ichars.encoding().encode_fill
            ( ob, count, _afmt.fill, _enc_err, _allow_surr );
    }
};

template <std::size_t CharSize, int Base>
template <typename FPack, typename Preview, typename IntT, typename CharT>
inline STRF_HD full_fmt_int_printer<CharSize, Base>::full_fmt_int_printer
    ( const FPack& fp
    , Preview& preview
    , strf::int_with_format<IntT, Base, true> value
    , strf::tag<CharT> tag_char) noexcept
    : _ichars( fp, preview, value.value().value
             , value.get_int_format_data(), tag_char/*, strf::tag<IntT>()*/)
    , _enc_err(get_facet<strf::encoding_error_c, IntT>(fp))
    , _afmt(value.get_alignment_format_data())
    , _allow_surr(get_facet<strf::surrogate_policy_c, IntT>(fp))
{
    auto content_width = _ichars.width();
    if (_afmt.width > content_width) {
        _fillcount = _afmt.width - content_width;
        preview.subtract_width(static_cast<std::int16_t>(_fillcount));
    }
    _calc_fill_size(preview);
}

template <std::size_t CharSize, int Base>
template <typename FPack, typename Preview, typename CharT>
inline STRF_HD full_fmt_int_printer<CharSize, Base>::full_fmt_int_printer
    ( const FPack& fp
    , Preview& preview
    , const void* value
    , strf::alignment_format_data afdata
    , strf::tag<CharT> tag_char )
    : _ichars( fp, preview, reinterpret_cast<std::size_t>(value)
             , strf::int_format_data{0, true}
             , tag_char
             , strf::tag<std::size_t, const void*>() )
    , _enc_err(get_facet<strf::encoding_error_c, const void*>(fp))
    , _afmt(afdata)
    , _allow_surr(get_facet<strf::surrogate_policy_c, const void*>(fp))
{
    auto content_width = _ichars.width();
    if (_afmt.width > content_width) {
        _fillcount = _afmt.width - content_width;
        preview.subtract_width(static_cast<std::int16_t>(_fillcount));
    }
    _calc_fill_size(preview);
}

template <std::size_t CharSize, int Base>
STRF_HD full_fmt_int_printer<CharSize, Base>::~full_fmt_int_printer()
{
}

template <std::size_t CharSize, int Base>
STRF_HD void full_fmt_int_printer<CharSize, Base>::print_to
        ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_fillcount == 0) {
        _ichars.print_to(ob);
    } else {
        switch(_afmt.alignment) {
            case strf::text_alignment::left: {
                _ichars.print_to(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::split: {
                _ichars.write_complement(ob);
                _write_fill(ob, _fillcount);
                _ichars.write_digits(ob);
                break;
            }
            case strf::text_alignment::center: {
                auto halfcount = _fillcount / 2;
                _write_fill(ob, halfcount);
                _ichars.print_to(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default: {
                _write_fill(ob, _fillcount);
                _ichars.print_to(ob);
            }
        }
    }
}

#if defined(STRF_SEPARATE_COMPILATION)
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4, 16>;

STRF_EXPLICIT_TEMPLATE class int_printer<1>;
STRF_EXPLICIT_TEMPLATE class int_printer<2>;
STRF_EXPLICIT_TEMPLATE class int_printer<4>;

STRF_EXPLICIT_TEMPLATE class punct_int_printer<1>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<2>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, short, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, short x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, int, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, int x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, long, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, long x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, long long, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, long long x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, unsigned short, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, unsigned short x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, unsigned int, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, unsigned int x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, unsigned long, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, unsigned long x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_intpunct<FPack, unsigned long long, 10>
    , strf::detail::punct_int_printer<sizeof(CharT)>
    , strf::detail::int_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, unsigned long long x)
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview, typename IntT, int Base>
STRF_HD inline strf::detail::full_fmt_int_printer<sizeof(CharT), Base>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::int_with_format<IntT, Base, true>& x )
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview, typename IntT, int Base>
STRF_HD inline strf::detail::partial_fmt_int_printer<sizeof(CharT), Base>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::int_with_format<IntT, Base, false>& x )
{
    return {fp, preview, x, strf::tag<CharT>()};
}

inline STRF_HD auto make_fmt(strf::rank<1>, short x)
{
    return strf::int_with_format<short>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, int x)
{
    return strf::int_with_format<int>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, long x)
{
    return strf::int_with_format<long>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, long long x)
{
    return strf::int_with_format<long long>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, unsigned short x)
{
    return strf::int_with_format<unsigned short>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, unsigned x)
{
    return  strf::int_with_format<unsigned>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, unsigned long x)
{
    return strf::int_with_format<unsigned long>{{x}};
}
inline STRF_HD auto make_fmt(strf::rank<1>, unsigned long long x)
{
    return strf::int_with_format<unsigned long long>{{x}};
}

// void*

template < typename CharOut, typename FPack, typename Preview >
inline STRF_HD strf::detail::partial_fmt_int_printer<sizeof(CharOut), 16>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const void* p )
{
    return { fp, preview, reinterpret_cast<std::size_t>(p)
           , strf::int_format_data{0, true, false}
           , strf::tag<CharOut>()
           , strf::tag<std::size_t, const void*>() };
}


template < typename CharOut, typename FPack, typename Preview >
inline STRF_HD strf::detail::full_fmt_int_printer<sizeof(CharOut), 16>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format<const void*, strf::alignment_format> f )
{
    return { fp
           , preview
           , f.value()
           , f.get_alignment_format_data()
           , strf::tag<CharOut>() };
}

inline STRF_HD auto make_fmt(strf::rank<1>, const void* p)
{
    return strf::value_with_format<const void*, strf::alignment_format>(p);
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

} // namespace strf

#endif // STRF_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED
