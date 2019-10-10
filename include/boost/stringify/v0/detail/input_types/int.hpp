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

template <class T, int Base>
class int_format_fn
{
private:

    template <int OtherBase>
    using adapted_derived_type
        = stringify::v0::fmt_replace<T, int_format<Base>, int_format<OtherBase> >;

    template <int OtherBase>
    T&& to_base(std::true_type /*same_base*/) &&
    {
        return static_cast<T&&>(*this);
    }
    template <int OtherBase>
    adapted_derived_type<OtherBase> to_base(std::false_type /*same_base*/) const &
    {
        return adapted_derived_type<OtherBase>
            { static_cast<const T&>(*this) };
    }

    template <int OtherBase>
    using base_eq = std::integral_constant<bool, Base == OtherBase>;

    template <int OtherBase>
    decltype(auto) to_base() &&
    {
        return static_cast<int_format_fn&&>(*this)
            .to_base<OtherBase>(base_eq<OtherBase>{});
    }

public:

    constexpr int_format_fn()  noexcept = default;

    template <typename U, int OtherBase>
    constexpr int_format_fn(const int_format_fn<U, OtherBase> & u) noexcept
        : _data(u.get_int_format_data())
    {
    }

    constexpr decltype(auto) hex() && noexcept
    {
        return static_cast<int_format_fn&&>(*this).to_base<16>();
    }
    constexpr decltype(auto) dec() && noexcept
    {
        return static_cast<int_format_fn&&>(*this).to_base<10>();
    }
    constexpr decltype(auto) oct() && noexcept
    {
        return static_cast<int_format_fn&&>(*this).to_base<8>();
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

template <typename IntT, int Base = 10>
using int_with_format = stringify::v0::value_with_format
    < stringify::v0::int_tag<IntT>
    , stringify::v0::int_format<Base>
    , stringify::v0::empty_alignment_format >;

template <typename IntT, int Base>
using int_with_alignment_format = stringify::v0::value_with_format
    < stringify::v0::int_tag<IntT>
    , stringify::v0::int_format<Base>
    , stringify::v0::alignment_format >;


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

    template <typename IntT>
    int_printer(IntT value)
    {
        _negative = value < 0;
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        unsigned_IntT uvalue = 1 + unsigned_IntT(-(value +1));
        _uvalue = _negative * uvalue + ! _negative * value;
        _digcount = stringify::v0::detail::count_digits<10>(_uvalue);
    }

    template <typename FP, typename IntT>
    int_printer(const FP&, IntT value)
        : int_printer(value)
    {
    }

    std::size_t necessary_size() const override;

    int width(int) const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

private:

    unsigned long long _uvalue;
    unsigned _digcount;
    bool _negative;
};

template <typename CharT>
std::size_t int_printer<CharT>::necessary_size() const
{
    return _digcount + _negative;
}

template <typename CharT>
int int_printer<CharT>::width(int) const
{
    return _digcount + _negative;
}

template <typename CharT>
void int_printer<CharT>::write
    ( boost::basic_outbuf<CharT>& ob ) const
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

    template <typename FPack, typename IntT>
    punct_int_printer
        ( const FPack& fp
        , IntT value ) noexcept
        : _punct(get_facet<stringify::v0::numpunct_c<10>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        unsigned_IntT uvalue = 1 + unsigned_IntT(-(value +1));
        _negative = value < 0;
        _uvalue = _negative * uvalue + ! _negative * value;
        _digcount = stringify::v0::detail::count_digits<10>(_uvalue);
        _sepcount = ( _punct.no_group_separation(_digcount)
                    ? 0
                    : _punct.thousands_sep_count(_digcount) );
    }

    std::size_t necessary_size() const override;

    int width(int) const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

private:

    const stringify::v0::numpunct_base& _punct;
    stringify::v0::encoding<CharT> _encoding;
    unsigned long long _uvalue;
    unsigned _digcount;
    unsigned _sepcount;
    bool _negative;
};

template <typename CharT>
std::size_t punct_int_printer<CharT>::necessary_size() const
{
    auto size = _digcount + _negative;
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
int punct_int_printer<CharT>::width(int) const
{
    return _sepcount + _digcount + _negative;
}

template <typename CharT>
void punct_int_printer<CharT>::write(boost::basic_outbuf<CharT>& ob) const
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

    template <typename FPack, typename IntT>
    partial_fmt_int_printer( const FPack& fp, IntT value )
        : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
        , _precision(0)
    {
        using unsigned_type = typename std::make_unsigned<IntT>::type;
        if (Base == 10 && value < 0)
        {
            _uvalue = 1 + unsigned_type(-(value + 1));
            _negative = true;
            _prefixsize = 1;
        }
        else
        {
            _uvalue = unsigned_type(value);
            _negative = false;
            _prefixsize = 0;
        }
        _digcount = stringify::v0::detail::count_digits<Base>(_uvalue);

        BOOST_STRINGIFY_IF_CONSTEXPR (detail::has_intpunct<CharT, FPack, IntT, Base>)
        {
            _sepcount = _punct.thousands_sep_count(_digcount);
        }
        else
        {
            _sepcount = 0;
        }
    }

    template <typename FPack, typename IntT>
    partial_fmt_int_printer( const FPack& fp
                           , const stringify::v0::int_with_format<IntT, Base>& value )
        : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        _init<IntT, detail::has_intpunct<CharT, FPack, IntT, Base>>
            ( value.value().value, value.get_int_format_data() );
    }

    template <typename FPack, typename IntT>
    partial_fmt_int_printer
        ( const FPack& fp
        , const stringify::v0::int_with_alignment_format<IntT, Base>& value )
        : _punct(get_facet<stringify::v0::numpunct_c<Base>, IntT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, IntT>(fp))
    {
        _init<IntT, detail::has_intpunct<CharT, FPack, IntT, Base>>
            ( value.value().value, value.get_int_format_data() );
    }

    int width() const
    {
        return std::max(_precision, _digcount)
            + _prefixsize
            + static_cast<int>(_sepcount);
    }

    auto encoding() const
    {
        return _encoding;
    }

    void write(boost::basic_outbuf<CharT>& ob) const override;
    std::size_t necessary_size() const override;
    int width(int) const override
    {
        return width();
    }

    void write_complement(boost::basic_outbuf<CharT>& ob) const;
    void write_digits(boost::basic_outbuf<CharT>& ob) const;

private:

    const stringify::v0::numpunct_base& _punct;
    const stringify::v0::encoding<CharT> _encoding;
    unsigned long long _uvalue;
    unsigned _digcount;
    unsigned _sepcount;
    unsigned _precision;
    bool _negative;
    std::uint8_t _prefixsize;

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
        using unsigned_IntT = typename std::make_unsigned<IntT>::type;
        unsigned_IntT uvalue = 1 + unsigned_IntT(-(value + 1));
        _negative = value < 0;
        _prefixsize = _negative || fmt.showpos;
        _uvalue = _negative * uvalue + (!_negative) * value;
    }
    else
    {
        _uvalue = unsigned_type(value);
        _negative = false;
        _prefixsize = fmt.showbase << (Base == 16);
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
std::size_t partial_fmt_int_printer<CharT, Base>::necessary_size() const
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
    return s;
}

template <typename CharT, int Base>
inline void partial_fmt_int_printer<CharT, Base>::write
    ( boost::basic_outbuf<CharT>& ob ) const
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
inline void partial_fmt_int_printer<CharT, Base>::write_digits
    ( boost::basic_outbuf<CharT>& ob ) const
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

    template <typename FPack, typename IntT>
    full_fmt_int_printer
        ( const FPack& fp
        , stringify::v0::int_with_alignment_format<IntT, Base> value ) noexcept;

    ~full_fmt_int_printer();

    std::size_t necessary_size() const override;

    void write( boost::basic_outbuf<CharT>& ob ) const override;

    int width(int) const override;

private:

    stringify::v0::detail::partial_fmt_int_printer<CharT, Base> _ichars;
    unsigned _fillcount = 0;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::alignment_format_data _afmt;
    stringify::v0::surrogate_policy _allow_surr;

    void _write_fill
        ( boost::basic_outbuf<CharT>& ob
        , std::size_t count ) const
    {
        return _ichars.encoding().encode_fill
            ( ob, count, _afmt.fill, _enc_err, _allow_surr );
    }
};

template <typename CharT, int Base>
template <typename FPack, typename IntT>
inline full_fmt_int_printer<CharT, Base>::full_fmt_int_printer
    ( const FPack& fp
    , stringify::v0::int_with_alignment_format<IntT, Base> value ) noexcept
    : _ichars(fp, value)
    , _enc_err(get_facet<stringify::v0::encoding_error_c, IntT>(fp))
    , _afmt(value.get_alignment_format_data())
    , _allow_surr(get_facet<stringify::v0::surrogate_policy_c, IntT>(fp))
{
    int content_width = _ichars.width();
    if (_afmt.width > content_width)
    {
        _fillcount = _afmt.width - content_width;
    }
    else
    {
        _afmt.width = content_width;
        _fillcount = 0;
    }
}

template <typename CharT, int Base>
full_fmt_int_printer<CharT, Base>::~full_fmt_int_printer()
{
}

template <typename CharT, int Base>
std::size_t full_fmt_int_printer<CharT, Base>::necessary_size() const
{
    std::size_t s = _ichars.necessary_size();
    if (_fillcount > 0)
    {
        s += _fillcount * _ichars.encoding().char_size(_afmt.fill, _enc_err);
    }
    return s;
}

template <typename CharT, int Base>
int full_fmt_int_printer<CharT, Base>::width(int) const
{
    return _afmt.width;
}

template <typename CharT, int Base>
void full_fmt_int_printer<CharT, Base>::write
        ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        _ichars.write(ob);
    }
    else
    {
        switch(_afmt.alignment)
        {
            case stringify::v0::alignment_e::left:
            {
                _ichars.write(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::alignment_e::internal:
            {
                _ichars.write_complement(ob);
                _write_fill(ob, _fillcount);
                _ichars.write_digits(ob);
                break;
            }
            case stringify::v0::alignment_e::center:
            {
                auto halfcount = _fillcount / 2;
                _write_fill(ob, halfcount);
                _ichars.write(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _ichars.write(ob);
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

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, short, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, short x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, int, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, int x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, long long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, long long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned short, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned short x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned int, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned int x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_intpunct<CharT, FPack, unsigned long long, 10>
    , stringify::v0::detail::punct_int_printer<CharT>
    , stringify::v0::detail::int_printer<CharT> >::type
make_printer(const FPack& fp, unsigned long long x)
{
    return {fp, x};
}

template <typename CharT, typename FPack, typename IntT, int Base>
inline stringify::v0::detail::full_fmt_int_printer<CharT, Base>
make_printer( const FPack& fp
            , const stringify::v0::int_with_alignment_format<IntT, Base>& x )
{
    return {fp, x};
}

template <typename CharT, typename FPack, typename IntT, int Base>
inline stringify::v0::detail::partial_fmt_int_printer<CharT, Base>
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
