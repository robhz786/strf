#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_FMT_INT_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_FMT_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/facets/encoding.hpp>
#include <boost/stringify/v0/facets/numpunct.hpp>
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
    using unsigned_type = typename std::make_unsigned<IntT>::type;
    static constexpr bool is_signed = std::is_signed<IntT>::value;
    constexpr static unsigned max_digcount = (sizeof(IntT) * 8 + 2) / 3;

public:

    using input_type  = IntT ;
    using char_type   = CharT;

    template <typename FPack>
    fmt_int_printer
        ( const FPack& fp
        , const stringify::v0::int_with_format<IntT>& value ) noexcept
        : _encoding(get_facet<stringify::v0::encoding_category<CharT>>(fp))
        , _err_hdl(get_facet<stringify::v0::encoding_policy_category>(fp).err_hdl())
        , _fmt(value)
    {
        init( get_facet<stringify::v0::numpunct_category<8>>(fp)
            , get_facet<stringify::v0::numpunct_category<10>>(fp)
            , get_facet<stringify::v0::numpunct_category<16>>(fp) );
    }

    ~fmt_int_printer();

    std::size_t necessary_size() const override;

    stringify::v0::expected_buff_it<CharT> write
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const override;

    int remaining_width(int w) const override;

private:

    const stringify::v0::encoding<CharT>& _encoding;
    const stringify::v0::numpunct_base* _punct;
    stringify::v0::error_handling _err_hdl;
    stringify::v0::int_with_format<IntT> _fmt;

    unsigned short _digcount;
    unsigned short _sepcount = 0;
    unsigned _fillcount;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, IntT>();
    }

    void init
        ( const stringify::v0::numpunct<8>& numpunct_oct
        , const stringify::v0::numpunct<10>& numpunct_dec
        , const stringify::v0::numpunct<16>& numpunct_hex ) noexcept;

    bool showsign() const
    {
        return is_signed && (_fmt.showpos() || _fmt.value().value < 0);
    }

    unsigned_type unsigned_value() const noexcept
    {
        if(_fmt.base() == 10)
        {
            return stringify::v0::detail::unsigned_abs(_fmt.value().value);
        }
        return static_cast<unsigned_type>(_fmt.value().value);
    }

    std::size_t length_fill() const
    {
        if (_fillcount > 0)
        {
            return _fillcount * _encoding.char_size(_fmt.fill(), _err_hdl);
        }
        return 0;
    }

    std::size_t length_body() const
    {
        return length_complement() + length_digits();
    }

    std::size_t length_complement() const noexcept
    {
        if (_fmt.base() == 10)
        {
            return showsign() ? 1 : 0;
        }
        else if (_fmt.base() == 16)
        {
            return _fmt.showbase() ? 2 : 0;
        }
        BOOST_ASSERT(_fmt.base() == 8);
        return _fmt.showbase() ? 1 : 0;
    }

    std::size_t length_digits() const noexcept
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

    stringify::v0::expected_buff_it<CharT> write_fill
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler
        , std::size_t count ) const
    {
        return stringify::v0::detail::write_fill( _encoding
                                                , buff
                                                , recycler
                                                , count
                                                , _fmt.fill()
                                                , _err_hdl );
    }


    stringify::v0::expected_buff_it<CharT> write_complement
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        if (_fmt.base() == 10)
        {
            if(is_signed)
            {
                if (buff.it == buff.end)
                {
                    auto x = recycler.recycle(buff.it);
                    BOOST_STRINGIFY_RETURN_ON_ERROR(x);
                    buff = *x;
                }
                if(_fmt.value().value < 0)
                {
                    *buff.it = '-';
                    ++buff.it;
                }
                else if( _fmt.showpos())
                {
                    *buff.it = '+';
                    ++buff.it;
                }
            }
        }
        else if (_fmt.showbase())
        {
            if (buff.it + 1 >= buff.end)
            {
                auto x = recycler.recycle(buff.it);
                BOOST_STRINGIFY_RETURN_ON_ERROR(x);
                buff = *x;
            }
            BOOST_ASSERT (buff.it + 1 < buff.end);
            if(_fmt.base() == 16)
            {
                buff.it[0] = '0';
                buff.it[1] = _fmt.uppercase() ? CharT('X'): CharT('x');
                buff.it += 2;
            }
            else
            {
                buff.it[0] = '0';
                ++buff.it;
            }
        }
        return { stringify::v0::in_place_t{}, buff };
    }

    stringify::v0::expected_buff_it<CharT> write_digits
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        if(_fmt.precision() > _digcount)
        {
            auto x = stringify::v0::detail::write_fill
                ( buff
                , recycler
                , _fmt.precision() - _digcount
                , CharT('0') );
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
            buff = *x;
        }
        if (_sepcount == 0)
        {
            return write_digits_nosep(buff, recycler);
        }
        return write_digits_sep(buff, recycler);
    }

    stringify::v0::expected_buff_it<CharT> write_digits_nosep
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        if (buff.it + _digcount > buff.end)
        {
            auto x = recycler.recycle(buff.it);
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
            if ((*x).it + _digcount > (*x).end)
            {
                return write_digits_nosep_buff(*x, recycler);
            }
            buff = *x;
        }
        auto end = buff.it + _digcount;
        CharT* it = stringify::v0::detail::write_int_txtdigits_backwards
            ( _fmt.value().value
            , _fmt.base()
            , _fmt.uppercase()
            , end );
        BOOST_ASSERT(it == buff.it);
        return { stringify::v0::in_place_t{}
               , stringify::v0::buff_it<CharT>{end, buff.end} };
    }

    stringify::v0::expected_buff_it<CharT> write_digits_nosep_buff
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        char tmp[3*sizeof(CharT)];
        char* tmp_end = tmp + sizeof(tmp) / sizeof(tmp[0]);
        char* it = stringify::v0::detail::write_int_txtdigits_backwards
            ( _fmt.value().value
            , _fmt.base()
            , _fmt.uppercase()
            , tmp_end );

        BOOST_ASSERT(it + _digcount == tmp_end);
        std::size_t space = buff.end - buff.it;
        BOOST_ASSERT(space < _digcount);
        std::copy_n(it, space, buff.it);
        it += space;
        unsigned count = _digcount - space;
        auto x = recycler.recycle(buff.it);
        while (x)
        {
            std::size_t space = (*x).end - (*x).it;
            if (count <= space)
            {
                std::copy_n(it, count, (*x).it);
                (*x).it += count;
                break;
            }
            std::copy_n(it, space, (*x).it);
            it += space;
            x = recycler.recycle((*x).it + space) ;
        }
        return x;
    }

    stringify::v0::expected_buff_it<CharT> write_digits_sep
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        char dig_buff[max_digcount];
        char* dig_it = stringify::v0::detail::write_int_txtdigits_backwards
            ( _fmt.value().value
            , _fmt.base()
            , _fmt.uppercase()
            , dig_buff + max_digcount );

        unsigned char grp_buff[max_digcount];
        auto* grp_it = _punct->groups(_digcount, grp_buff);

        char32_t sep_char32 = _punct->thousands_sep();
        if (sep_char32 <= _encoding.max_corresponding_u32char)
        {
            return write_digits_littlesep( buff, recycler, dig_it
                                         , grp_buff, grp_it
                                         , (CharT)sep_char32 );
        }
        auto sep_char32_size = _encoding.validate(sep_char32);
        if (sep_char32_size == (std::size_t)-1)
        {
            return write_digits_nosep(buff, recycler);
        }
        if (sep_char32_size == 1)
        {
            CharT sep_ch;
            CharT* sep_char_ptr = & sep_ch;
            auto res = _encoding.encode_char( &sep_char_ptr, sep_char_ptr + 1
                                            , sep_char32
                                            , stringify::v0::error_handling::stop );
            BOOST_ASSERT(res == stringify::v0::cv_result::success);
            return write_digits_littlesep( buff, recycler, dig_it
                                         , grp_buff, grp_it, sep_ch);
        }
        return write_digits_bigsep( buff, recycler, dig_it
                                  , grp_buff, grp_it
                                  , sep_char32, sep_char32_size );
    }

    stringify::v0::expected_buff_it<CharT> write_digits_littlesep
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler
        , const char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , CharT sep_char ) const
    {
        std::size_t necessary_size = (grp_it - grp) + 1 + _digcount;
        if (buff.it + necessary_size > buff.end)
        {
            auto x = recycler.recycle(buff.it);
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
            buff = *x;
            BOOST_ASSERT(buff.it + necessary_size <= buff.end);
        }

        for(unsigned i = *grp_it; i != 0; --i)
        {
            *buff.it++ = *dig_it++;
        }

        do
        {
            *buff.it++ = sep_char;
            for(unsigned i = *--grp_it; i != 0; --i)
            {
                *buff.it++ = *dig_it++;
            }
        }
        while(grp_it > grp);

        return { stringify::v0::in_place_t{}, buff };
    }

    stringify::v0::expected_buff_it<CharT> write_digits_bigsep
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler
        , char* dig_it
        , unsigned char* grp
        , unsigned char* grp_it
        , char32_t sep_char
        , unsigned sep_char_size ) const
    {
        {
            unsigned i = *grp_it;
            if (buff.it + i > buff.end)
            {
                auto x = recycler.recycle(buff.it);
                BOOST_STRINGIFY_RETURN_ON_ERROR(x);
                buff = *x;
            }
            for( ; i != 0; --i)
            {
                *buff.it++ = *dig_it++;
            }
        }
        do
        {
            unsigned i = *--grp_it;
            if (buff.it + i + sep_char_size > buff.end)
            {
                auto x = recycler.recycle(buff.it);
                BOOST_STRINGIFY_RETURN_ON_ERROR(x);
                buff = *x;
            }
            auto res = _encoding.encode_char( &buff.it, buff.end, sep_char
                                            , stringify::v0::error_handling::stop );
            (void)res;
            BOOST_ASSERT(res == stringify::v0::cv_result::success);
            for(; i != 0; --i)
            {
                *buff.it++ = *dig_it++;
            }
        }
        while(grp_it > grp);
        return { stringify::v0::in_place_t{}, buff };
    }
};

template <typename IntT, typename CharT>
void fmt_int_printer<IntT, CharT>::init
    ( const stringify::v0::numpunct<8>& numpunct_oct
    , const stringify::v0::numpunct<10>& numpunct_dec
    , const stringify::v0::numpunct<16>& numpunct_hex ) noexcept
{
    auto extra_chars_count = 0;
    if (_fmt.base() == 10)
    {
        _digcount = stringify::v0::detail::count_digits<10>(_fmt.value().value);
        if(showsign())
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

    BOOST_ASSERT(_digcount <= max_digcount);
    BOOST_ASSERT(_sepcount <= max_digcount);
}

template <typename IntT, typename CharT>
fmt_int_printer<IntT, CharT>::~fmt_int_printer()
{
}

template <typename IntT, typename CharT>
std::size_t fmt_int_printer<IntT, CharT>::necessary_size() const
{
    return length_body() + length_fill();
}

template <typename IntT, typename CharT>
int fmt_int_printer<IntT, CharT>::remaining_width(int w) const
{
    return w > _fmt.width() ? (w - _fmt.width()) : 0;
}


template <typename IntT, typename CharT>
stringify::v0::expected_buff_it<CharT> fmt_int_printer<IntT, CharT>::write
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
{
    if (_fillcount == 0)
    {
        auto x = write_complement(buff, recycler);
        return x ? write_digits(*x, recycler) : x;
    }

    switch(_fmt.alignment())
    {
        case stringify::v0::alignment::left:
        {
            auto x = write_complement(buff, recycler);
            if(x) x = write_digits(*x, recycler);
            return x ? write_fill(*x, recycler, _fillcount) : x;
        }
        case stringify::v0::alignment::internal:
        {
            auto x = write_complement(buff, recycler);
            if(x) x = write_fill(*x, recycler, _fillcount);
            return x ? write_digits(*x, recycler) : x;
        }
        case stringify::v0::alignment::center:
        {
            auto halfcount = _fillcount / 2;
            auto x = write_fill(buff, recycler, halfcount);
            if(x) x = write_complement(*x, recycler);
            if(x) x = write_digits(*x, recycler);
            return x ? write_fill(*x, recycler, _fillcount - halfcount) : x;
        }
        default:
        {
            auto x = write_fill(buff, recycler, _fillcount);
            if(x) x = write_complement(*x, recycler);
            return x ? write_digits(*x, recycler) : x;
        }
    }
}


template <typename CharT, typename FPack, typename IntT>
inline stringify::v0::fmt_int_printer<IntT, CharT>
make_printer
    ( const FPack& fp
    , const stringify::v0::int_with_format<IntT>& x
    )
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

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<short, wchar_t>;
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
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned short, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned short, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned short, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_int_printer<unsigned short, wchar_t>;
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

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_FMT_INT_HPP_INCLUDED
