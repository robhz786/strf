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

        constexpr fn() = default;

        template <typename U>
        constexpr fn(const fn<U> & u)
            : _base(u.base())
            , _showbase(u.showbase())
            , _showpos(u.showpos())
            , _uppercase(u.uppercase())
        {
        }

        constexpr T&& p(unsigned _) &&
        {
            _precision = _;
            return static_cast<T&&>(*this);
        }
        constexpr T& p(unsigned _) &
        {
            _precision = _;
            return *this;
        }
        constexpr T&& uphex() &&
        {
            _base = 16;
            _uppercase = true;
            return static_cast<T&&>(*this);
        }
        constexpr T& uphex() &
        {
            _base = 16;
            _uppercase = true;
            return static_cast<T&>(*this);
        }
        constexpr T&& hex() &&
        {
            _base = 16;
            _uppercase = false;
            return static_cast<T&&>(*this);
        }
        constexpr T& hex() &
        {
            _base = 16;
            _uppercase = false;
            return static_cast<T&>(*this);
        }
        constexpr T&& dec() &&
        {
            _base = 10;
            return static_cast<T&&>(*this);
        }
        constexpr T& dec() &
        {
            _base = 10;
            return static_cast<T&>(*this);
        }
        constexpr T&& oct() &&
        {
            _base = 8;
            return static_cast<T&&>(*this);
        }
        constexpr T& oct() &
        {
            _base = 8;
            return static_cast<T&>(*this);
        }
        constexpr T&& operator*() &&
        {
            _i18n = true;
            return static_cast<T&&>(*this);
        }
        constexpr T& operator*() &
        {
            _i18n = true;
            return static_cast<T&>(*this);
        }
        constexpr T&& operator+() &&
        {
            _showpos = true;
            return static_cast<T&&>(*this);
        }
        constexpr T& operator+() &
        {
            _showpos = true;
            return static_cast<T&&>(*this);
        }
        constexpr T&& operator~() &&
        {
            _showbase = true;
            return static_cast<T&&>(*this);
        }
        constexpr T& operator~() &
        {
            _showbase = true;
            return static_cast<T&>(*this);
        }
        constexpr T&& uppercase(bool u) &&
        {
            _uppercase = u;
            return static_cast<T&&>(*this);
        }
        constexpr T& uppercase(bool u) &
        {
            _uppercase = u;
            return static_cast<T&>(*this);
        }
        constexpr T&& showbase(bool s) &&
        {
            _showbase = s;
            return static_cast<T&&>(*this);
        }
        constexpr T& showbase(bool s) &
        {
            _showbase = s;
            return static_cast<T&>(*this);
        }
        constexpr T&& showpos(bool s) &&
        {
            _showpos = s;
            return static_cast<T&&>(*this);
        }
        constexpr T& showpos(bool s) &
        {
            _showpos = s;
            return static_cast<T&>(*this);
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
        constexpr bool uppercase() const
        {
            return _uppercase;
        }
        constexpr bool has_i18n() const
        {
            return _i18n;
        }

    private:

        unsigned _precision = 0;
        unsigned short _base = 10;
        bool _showbase = false;
        bool _showpos = false;
        bool _uppercase = false;
        bool _i18n = false;
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


template <typename IntT, typename CharT>
class fmt_int_printer: public printer<CharT>
{
    static constexpr bool _is_signed = std::is_signed<IntT>::value;
    constexpr static unsigned _max_digcount = (sizeof(IntT) * 8 + 2) / 3;

public:

    using input_type  = IntT ;
    using char_type   = CharT;

    template <typename FPack>
    fmt_int_printer
        ( const FPack& fp
        , const stringify::v0::int_with_format<IntT>& value ) noexcept
        : _encoding(_get_facet<stringify::v0::encoding_category<CharT>>(fp))
        , _err_hdl(_get_facet<stringify::v0::encoding_policy_category>(fp).err_hdl())
        , _fmt(value)
    {
        _init( _get_facet<stringify::v0::numpunct_category<8>>(fp)
             , _get_facet<stringify::v0::numpunct_category<10>>(fp)
             , _get_facet<stringify::v0::numpunct_category<16>>(fp) );
    }

    ~fmt_int_printer();

    std::size_t necessary_size() const override;

    bool write( stringify::v0::output_buffer<CharT>& ob ) const override;

    int remaining_width(int w) const override;

private:

    const stringify::v0::encoding<CharT> _encoding;
    const stringify::v0::numpunct_base* _punct;
    stringify::v0::error_handling _err_hdl;
    stringify::v0::int_with_format<IntT> _fmt;

    unsigned short _digcount;
    unsigned short _sepcount = 0;
    unsigned _fillcount;

    template <typename Category, typename FPack>
    decltype(auto) _get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, IntT>();
    }

    void _init
        ( const stringify::v0::numpunct<8>& numpunct_oct
        , const stringify::v0::numpunct<10>& numpunct_dec
        , const stringify::v0::numpunct<16>& numpunct_hex ) noexcept;

    bool _showsign() const
    {
        return _is_signed && (_fmt.showpos() || _fmt.value().value < 0);
    }

    std::size_t _length_fill() const
    {
        if (_fillcount > 0)
        {
            return _fillcount * _encoding.char_size(_fmt.fill(), _err_hdl);
        }
        return 0;
    }

    std::size_t _length_body() const
    {
        return _length_complement() + _length_digits();
    }

    std::size_t _length_complement() const noexcept
    {
        if (_fmt.base() == 10)
        {
            return _showsign() ? 1 : 0;
        }
        else if (_fmt.base() == 16)
        {
            return _fmt.showbase() ? 2 : 0;
        }
        BOOST_ASSERT(_fmt.base() == 8);
        return _fmt.showbase() ? 1 : 0;
    }

    std::size_t _length_digits() const noexcept
    {
        auto total_digcount
            = _fmt.precision() > _digcount
            ? _fmt.precision() : _digcount;
        if (_sepcount > 0)
        {
            auto sep_len = _encoding.char_size( _punct->thousands_sep()
                                              , _err_hdl );
            return total_digcount + _sepcount * sep_len;
        }
        return total_digcount;
    }

    bool _write_fill
        ( stringify::v0::output_buffer<CharT>& ob
        , std::size_t count ) const
    {
        return stringify::v0::detail::write_fill( _encoding
                                                , ob
                                                , count
                                                , _fmt.fill()
                                                , _err_hdl );
    }

    bool _write_complement(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits(stringify::v0::output_buffer<CharT>& ob)  const;
    bool _write_digits_nosep(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits_nosep_buff(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits_sep(stringify::v0::output_buffer<CharT>& ob) const;
    bool _write_digits_littlesep
        ( stringify::v0::output_buffer<CharT>& ob
        , const char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , CharT sep_char ) const;
    bool _write_digits_bigsep
        ( stringify::v0::output_buffer<CharT>& ob
        , char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , char32_t sep_char
        , unsigned sep_char_size ) const;
};

template <typename IntT, typename CharT>
void fmt_int_printer<IntT, CharT>::_init
    ( const stringify::v0::numpunct<8>& numpunct_oct
    , const stringify::v0::numpunct<10>& numpunct_dec
    , const stringify::v0::numpunct<16>& numpunct_hex ) noexcept
{
    auto extra_chars_count = 0;
    if (_fmt.base() == 10)
    {
        _digcount = stringify::v0::detail::count_digits<10>(_fmt.value().value);
        if(_showsign())
        {
            extra_chars_count = 1;
        }
        if (_fmt.has_i18n())
        {
            _punct = & numpunct_dec;
            _sepcount = numpunct_dec.thousands_sep_count(_digcount);
        }
    }
    else if (_fmt.base() == 16)
    {
        _digcount = stringify::v0::detail::count_digits<16>(_fmt.value().value);
        if(_fmt.showbase())
        {
            extra_chars_count = 2;
        }
        if (_fmt.has_i18n())
        {
            _punct = & numpunct_hex;
            _sepcount = numpunct_hex.thousands_sep_count(_digcount);
        }
    }
    else
    {
        BOOST_ASSERT(_fmt.base() == 8);
        _digcount = stringify::v0::detail::count_digits<8>(_fmt.value().value);
        if(_fmt.showbase())
        {
            extra_chars_count = 1;
        }
        if (_fmt.has_i18n())
        {
            _punct = & numpunct_oct;
            _sepcount = numpunct_oct.thousands_sep_count(_digcount);
        }
    }

    _fillcount = 0;
    int content_width
        = static_cast<int>( _fmt.precision() > _digcount
                          ? _fmt.precision()
                          : _digcount )
        + static_cast<int>(_sepcount)
        + extra_chars_count;

    if (_fmt.width() > content_width)
    {
        _fillcount = _fmt.width() - content_width;
    }
    else
    {
        _fmt.width(content_width);
        _fillcount = 0;
    }

    BOOST_ASSERT(_digcount <= _max_digcount);
    BOOST_ASSERT(_sepcount <= _max_digcount);
}

template <typename IntT, typename CharT>
fmt_int_printer<IntT, CharT>::~fmt_int_printer()
{
}

template <typename IntT, typename CharT>
std::size_t fmt_int_printer<IntT, CharT>::necessary_size() const
{
    return _length_body() + _length_fill();
}

template <typename IntT, typename CharT>
int fmt_int_printer<IntT, CharT>::remaining_width(int w) const
{
    return w > _fmt.width() ? (w - _fmt.width()) : 0;
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::write
        ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        return _write_complement(ob)
            && _write_digits(ob);
    }

    switch(_fmt.alignment())
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

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_complement
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if (_fmt.base() == 10)
    {
        if(_is_signed)
        {
            if (ob.size() == 0)
            {
                if ( ! ob.recycle())
                {
                    return false;
                }
            }
            if(_fmt.value().value < 0)
            {
                *ob.pos() = '-';
                ob.advance(1);
            }
            else if( _fmt.showpos())
            {
                *ob.pos() = '+';
                ob.advance(1);
            }
        }
    }
    else if (_fmt.showbase())
    {
        if (ob.size() < 2)
        {
            if ( ! ob.recycle())
            {
                return false;
            }
        }
        if(_fmt.base() == 16)
        {
            ob.pos()[0] = '0';
            ob.pos()[1] = _fmt.uppercase() ? CharT('X'): CharT('x');
            ob.advance(2);
        }
        else
        {
            ob.pos()[0] = '0';
            ob.advance(1);
        }
    }
    return true;
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if(_fmt.precision() > _digcount)
    {
        if ( ! stringify::v0::detail::write_fill( ob
                                                , _fmt.precision() - _digcount
                                                , CharT('0') ) )
        {
            return false;
        }
    }
    if (_sepcount == 0)
    {
        return _write_digits_nosep(ob);
    }
    return _write_digits_sep(ob);
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits_nosep
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    if (ob.size() < _digcount)
    {
        if ( ! ob.recycle())
        {
            return false;
        }
        if (ob.size() < _digcount)
        {
            return _write_digits_nosep_buff(ob);
        }
    }
    ob.advance(_digcount);
    CharT* it = stringify::v0::detail::write_int_txtdigits_backwards
        ( _fmt.value().value
        , _fmt.base()
        , _fmt.uppercase()
        , ob.pos() );
    BOOST_ASSERT(it + _digcount == ob.pos());
    return true;
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits_nosep_buff
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    char tmp[3*sizeof(CharT)];
    char* tmp_end = tmp + sizeof(tmp) / sizeof(tmp[0]);
    char* it = stringify::v0::detail::write_int_txtdigits_backwards
        ( _fmt.value().value
        , _fmt.base()
        , _fmt.uppercase()
        , tmp_end );

    BOOST_ASSERT(it + _digcount == tmp_end);
    std::size_t space = ob.size();
    BOOST_ASSERT(space < _digcount);
    std::copy_n(it, space, ob.pos());
    unsigned count = _digcount - space;
    while (ob.recycle(ob.end()))
    {
        std::size_t space = ob.size();
        if (count <= space)
        {
            std::copy_n(it, count, ob.pos());
            ob.advance(count);
            return true;
        }
        std::copy_n(it, space, ob.pos());
        it += space;
    }
    return false;
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits_sep
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    char dig_buff[_max_digcount];
    char* dig_it = stringify::v0::detail::write_int_txtdigits_backwards
        ( _fmt.value().value
        , _fmt.base()
        , _fmt.uppercase()
        , dig_buff + _max_digcount );

    char32_t sep_char32 = _punct->thousands_sep();
    unsigned char grp_buff[_max_digcount];
    auto* grp_it = _punct->groups(_digcount, grp_buff);
    BOOST_ASSERT((grp_it - grp_buff) == _sepcount);

    if ( _encoding.u32equivalence_begin() <= sep_char32
      && _encoding.u32equivalence_end() > sep_char32 )
    {
        return _write_digits_littlesep( ob, dig_it
                                      , grp_buff, grp_it
                                      , (CharT)sep_char32 );
    }
    auto sep_char32_size = _encoding.validate(sep_char32);
    if (sep_char32_size == (std::size_t)-1)
    {
        return _write_digits_nosep(ob);
    }
    if (sep_char32_size == 1)
    {
        CharT sep_ch;
        CharT* sep_char_ptr = & sep_ch;
        auto res = _encoding.encode_char( &sep_char_ptr, sep_char_ptr + 1
                                        , sep_char32
                                        , stringify::v0::error_handling::stop );
        BOOST_ASSERT(res == stringify::v0::cv_result::success);
        (void) res;
        return _write_digits_littlesep( ob, dig_it
                                      , grp_buff, grp_it, sep_ch);
    }
    return _write_digits_bigsep( ob, dig_it
                               , grp_buff, grp_it
                               , sep_char32, sep_char32_size );
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits_littlesep
    ( stringify::v0::output_buffer<CharT>& ob
    , const char* dig_it
    , unsigned char* grp
    , unsigned char* grp_it
    , CharT sep_char ) const
{
    std::size_t necessary_size = (grp_it - grp) + _digcount;
    if (ob.size() < necessary_size && ! ob.recycle())
    {
        return false;
    }
    BOOST_ASSERT(ob.size() >= necessary_size);

    auto it = ob.pos();
    for(unsigned i = *grp_it; i != 0; --i)
    {
        *it++ = *dig_it++;
    }

    do
    {
        *it++ = sep_char;
        for(unsigned i = *--grp_it; i != 0; --i)
        {
            *it++ = *dig_it++;
        }
    }
    while(grp_it > grp);
    ob.set_pos(it);
    return true;
}

template <typename IntT, typename CharT>
bool fmt_int_printer<IntT, CharT>::_write_digits_bigsep
    ( stringify::v0::output_buffer<CharT>& ob
    , char* dig_it
    , unsigned char* grp
    , unsigned char* grp_it
    , char32_t sep_char
    , unsigned sep_char_size ) const
{
    {
        unsigned i = *grp_it;
        if (ob.size() < i && ! ob.recycle())
        {
            return false;
        }
        BOOST_ASSERT(ob.size() >= i); // TODO is this guaranteed ?
        auto it = ob.pos();
        for( ; i != 0; --i)
        {
            *it++ = *dig_it++;
        }
        ob.set_pos(it);
    }
    do
    {
        unsigned i = *--grp_it;
        if (ob.size() <  i + sep_char_size && ! ob.recycle())
        {
            return false;
        }
        BOOST_ASSERT(ob.size() >= i + sep_char_size); // TODO is this guaranteed ?
        auto it = ob.pos();
        auto res = _encoding.encode_char( &it, ob.end(), sep_char
                                        , stringify::v0::error_handling::stop );
        (void)res;
        BOOST_ASSERT(res == stringify::v0::cv_result::success);
        for(; i != 0; --i)
        {
            *it++ = *dig_it++;
        }
        ob.set_pos(it);
    }
    while(grp_it > grp);
    return true;
}


template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::fmt_int_printer<IntT, CharT>
make_printer
    ( const FPack& fp
    , const stringify::v0::int_with_format<IntT>& x )
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


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<long long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned int, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned int, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned int, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned int, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long long, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long long, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long long, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned long long, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FMT_INT_HPP_INCLUDED
