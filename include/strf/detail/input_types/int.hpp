#ifndef STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/charset.hpp>
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
    using adapted_derived_type_
        = strf::fmt_replace<T, int_format<Base>, int_format<OtherBase> >;

public:

    constexpr STRF_HD int_format_fn()  noexcept { }

    template <typename U, int OtherBase>
    constexpr STRF_HD int_format_fn(const int_format_fn<U, OtherBase> & u) noexcept
        : data_(u.get_int_format_data())
    {
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 16, T&&>
    hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 16, adapted_derived_type_<B>>
    hex() &&
    {
        return adapted_derived_type_<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 10, T&&>
    dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 10, adapted_derived_type_<B>>
    dec() &&
    {
        return adapted_derived_type_<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 8, adapted_derived_type_<B>>
    oct() &&
    {
        return adapted_derived_type_<B>{ static_cast<const T&>(*this) };
    }

    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 2, T&&>
    bin() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 2, adapted_derived_type_<B>>
    bin() &&
    {
        return adapted_derived_type_<B>{ static_cast<const T&>(*this) };
    }

    constexpr STRF_HD T&& p(unsigned _) && noexcept
    {
        data_.precision = _;
        return static_cast<T&&>(*this);
    }
    STRF_HD T&& operator+() && noexcept
    {
        data_.showpos = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& operator*() && noexcept
    {
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    [[deprecated]] // use instead operator*
    constexpr STRF_HD T&& operator~() && noexcept
    {
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr static STRF_HD int base() noexcept
    {
        return Base;
    }
    constexpr STRF_HD unsigned precision() const noexcept
    {
        return data_.precision;
    }
    constexpr STRF_HD bool showbase() const noexcept
    {
        return data_.showbase;
    }
    constexpr STRF_HD bool showpos() const noexcept
    {
        return data_.showpos;
    }
    constexpr STRF_HD strf::int_format_data get_int_format_data() const noexcept
    {
        return data_;
    }

private:

    strf::int_format_data data_;
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

template <typename IntT, int Base = 10, bool HasAlignment = false>
using int_with_format = strf::value_with_format
    < strf::int_tag<IntT>
    , strf::int_format<Base>
    , strf::alignment_format_q<HasAlignment> >;

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
inline STRF_HD auto make_fmt(strf::rank<1>, const void* p)
{
    return strf::value_with_format<const void*, strf::alignment_format>(p);
}

namespace detail {

template <std::size_t> class int_printer;
template <std::size_t> class punct_int_printer;
template <std::size_t, int> class partial_fmt_int_printer;
template <std::size_t, int> class full_fmt_int_printer;

template <typename FPack, typename IntT, unsigned Base>
class has_intpunct_impl
{
public:

    static STRF_HD std::true_type  test_numpunct(const strf::numpunct<Base>&);
    static STRF_HD std::false_type test_numpunct(const strf::default_numpunct<Base>&);
    static STRF_HD std::false_type test_numpunct(const strf::no_grouping<Base>&);

    static STRF_HD const FPack& fp();

    using has_numpunct_type = decltype
        ( test_numpunct
            ( get_facet< strf::numpunct_c<Base>, IntT >(fp())) );
public:

    static constexpr bool value = has_numpunct_type::value;
};

template <typename FPack, typename IntT, unsigned Base>
constexpr STRF_HD bool has_intpunct()
{
    return has_intpunct_impl<FPack, IntT, Base>::value;
}

template <typename CharT, typename Preview, typename IntT>
struct int_printer_input
{
    using printer_type = strf::detail::int_printer<sizeof(CharT)>;

    template<typename FPack>
    constexpr STRF_HD int_printer_input
        ( const FPack&, Preview& preview_, IntT arg_ )
        : preview(preview_)
        , value(arg_)
    {
    }

    constexpr STRF_HD int_printer_input(Preview& preview_, IntT arg_)
        : preview(preview_)
        , value(arg_)
    {
    }

    Preview& preview;
    IntT value;
};

template <typename CharT, typename FPack, typename Preview, typename IntT>
struct punct_int_printer_input
{
    using printer_type = strf::detail::punct_int_printer<sizeof(CharT)>;

    FPack fp;
    Preview& preview;
    IntT value;
};

template < typename CharT, bool WithPunct, typename FPack
         , typename Preview, typename IntT  >
constexpr STRF_HD std::enable_if_t
    < ! WithPunct
    , strf::detail::int_printer_input<CharT, Preview, IntT> >
make_int_printer_input(const FPack&, Preview& preview, IntT x)
{
    return {preview, x};
}

template < typename CharT, bool WithPunct, typename FPack
         , typename Preview, typename IntT  >
constexpr STRF_HD std::enable_if_t
    < WithPunct
    , strf::detail::punct_int_printer_input<CharT, FPack, Preview, IntT> >
make_int_printer_input(const FPack& fp, Preview& preview, IntT x)
{
    return {fp, preview, x};
}

template < typename CharT, typename FPack, typename Preview, typename IntT >
struct int_printable_traits
{
    using printer_input_type = std::conditional_t
        < strf::detail::has_intpunct<FPack, IntT, 10>()
        , strf::detail::punct_int_printer_input<CharT, FPack, Preview, IntT>
        , strf::detail::int_printer_input<CharT, Preview, IntT> >;

    constexpr static STRF_HD printer_input_type
    make_input(const FPack& fp, Preview& preview, IntT arg)
    {
        return {fp, preview, arg};
    }
};

template < typename CharT, typename FPack>
struct voidptr_printable_traits
{
    template <typename Preview>
    constexpr static STRF_HD auto
    make_input(const FPack& fp, Preview& preview, const void* arg)
    {
        auto new_fp = strf::pack
            ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
            , strf::get_facet<strf::lettercase_c, const void*>(fp)
            , strf::get_facet<strf::charset_c<CharT>, const void*>(fp) );

        return strf::make_printer_input<CharT>
            ( new_fp, preview, *strf::hex(reinterpret_cast<std::size_t>(arg)) );
    }
};


} // namespace detail

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, short>
get_printable_traits(Preview&, short)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, int>
get_printable_traits(Preview&, int)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, long>
get_printable_traits(Preview&, long)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, long long>
get_printable_traits(Preview&, long long)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, unsigned short>
get_printable_traits(Preview&, unsigned short)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, unsigned int>
get_printable_traits(Preview&, unsigned int)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, unsigned long>
get_printable_traits(Preview&, unsigned long)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::int_printable_traits<CharT, FPack, Preview, unsigned long long>
get_printable_traits(Preview&, unsigned long long)
{ return {}; }

template <typename CharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::voidptr_printable_traits<CharT, FPack>
get_printable_traits(Preview&, const void*)
{ return {}; }

template < typename CharT, typename FPack, typename Preview
         , typename IntT, int Base, bool HasAlignment >
struct printable_traits
    < CharT, FPack, Preview, strf::int_with_format<IntT, Base, HasAlignment> >
    : strf::usual_printable_traits
        < CharT, FPack
        , std::conditional_t
            < HasAlignment
            , strf::detail::full_fmt_int_printer<sizeof(CharT), Base>
            , strf::detail::partial_fmt_int_printer<sizeof(CharT), Base> > >
{
};

template < typename CharT, typename FPack, typename Preview >
struct printable_traits
    < CharT, FPack, Preview
    , strf::value_with_format<const void*, strf::alignment_format> >
{
    constexpr static STRF_HD auto make_input
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format<const void*, strf::alignment_format> arg )
    {
        auto new_fp = strf::pack
            ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
            , strf::get_facet<strf::lettercase_c, const void*>(fp)
            , strf::get_facet<strf::charset_c<CharT>, const void*>(fp) );

        return strf::make_printer_input<CharT>
            ( new_fp, preview
            , *strf::hex(reinterpret_cast<std::size_t>(arg.value()))
                .set(arg.get_alignment_format_data()) );
    }
};

namespace detail {

template <std::size_t CharSize>
class int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename Preview, typename IntT>
    STRF_HD int_printer(strf::detail::int_printer_input<CharT, Preview, IntT> i)
    {
        init_(i.preview, i.value);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    template <typename Preview>
    STRF_HD void init_(Preview& p, short value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, int value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, long value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, long long value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, unsigned short value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, unsigned int value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, unsigned long value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, unsigned long long value){ init2_(p, value); }

    template <typename Preview, typename IntT>
    STRF_HD void init2_(Preview& preview, IntT value)
    {
        negative_ = value < 0;
        uvalue_ = strf::detail::unsigned_abs(value);
        digcount_ = strf::detail::count_digits<10>(uvalue_);
        auto size_ = digcount_ + negative_;
        preview.subtract_width(static_cast<std::int16_t>(size_));
        preview.add_size(size_);
    }


    unsigned long long uvalue_;
    unsigned digcount_;
    bool negative_;
};

template <std::size_t CharSize>
STRF_HD void int_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    unsigned size = digcount_ + negative_;
    ob.ensure(size);
    auto* it = write_int_dec_txtdigits_backwards(uvalue_, ob.pointer() + size);
    if (negative_) {
        it[-1] = '-';
    }
    ob.advance(size);
}

template <std::size_t CharSize>
class punct_int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename FPack, typename Preview, typename IntT>
    STRF_HD punct_int_printer
        ( const strf::detail::punct_int_printer_input<CharT, FPack, Preview, IntT>& i )
    {
        decltype(auto) cs = get_facet<strf::charset_c<CharT>, IntT>(i.fp);

        uvalue_ = strf::detail::unsigned_abs(i.value);
        digcount_ = strf::detail::count_digits<10>(uvalue_);
        auto punct = get_facet<strf::numpunct_c<10>, IntT>(i.fp);
        if (! punct.no_group_separation(digcount_)) {
            groups_ = punct.groups();
            thousands_sep_ = punct.thousands_sep();
            std::size_t sepsize = cs.validate(thousands_sep_);
            if (sepsize != (std::size_t)-1) {
                sepsize_ = static_cast<unsigned>(sepsize);
                sepcount_ = punct.thousands_sep_count(digcount_);
                if (sepsize_ == 1) {
                    char_type little_sep[4];
                    cs.encode_char(little_sep, thousands_sep_);
                    thousands_sep_ = little_sep[0];
                } else {
                    encode_char_ = cs.encode_char_func();
                }
            }
        }
        negative_ = i.value < 0;
        i.preview.add_size(digcount_ + negative_ + sepsize_ * sepcount_);
        i.preview.subtract_width
            ( static_cast<std::int16_t>(sepcount_ + digcount_ + negative_) );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::encode_char_f<CharSize> encode_char_;
    strf::digits_groups_iterator groups_;
    char32_t thousands_sep_;
    unsigned long long uvalue_;
    unsigned digcount_;
    unsigned sepcount_ = 0;
    unsigned sepsize_ = 0;
    bool negative_;
};

template <std::size_t CharSize>
STRF_HD void punct_int_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    if (sepcount_ == 0) {
        ob.ensure(negative_ + digcount_);
        auto it = ob.pointer();
        if (negative_) {
            *it = static_cast<char_type>('-');
            ++it;
        }
        it += digcount_;
        strf::detail::write_int_dec_txtdigits_backwards(uvalue_, it);
        ob.advance_to(it);
    } else {
        if (negative_) {
            put(ob, static_cast<char_type>('-'));
        }
        if (sepsize_ == 1) {
            strf::detail::write_int_little_sep<10>
                ( ob, uvalue_, groups_, digcount_, sepcount_
                , static_cast<char_type>(thousands_sep_), strf::lowercase );
        } else {
            strf::detail::write_int_big_sep<10>
                ( ob, encode_char_, uvalue_, groups_, thousands_sep_, sepsize_
                , digcount_, strf::lowercase );
        }
    }
}

template <std::size_t CharSize, int Base>
class partial_fmt_int_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    template < typename CharT, typename FPack, typename Preview
             , typename ThisPrinter, typename IntT >
    STRF_HD partial_fmt_int_printer
        ( const strf::usual_printer_input
            < CharT, FPack, Preview, ThisPrinter
            , strf::int_with_format<IntT, Base, false> >& i )
        : partial_fmt_int_printer( i.fp, i.preview, i.arg.value().value
                                 , i.arg.get_int_format_data()
                                 , strf::tag<CharT>() )
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
        , strf::tag<CharT> )
        : lettercase_(get_facet<strf::lettercase_c, IntTag>(fp))
    {
        init_<IntT>( value, fdata );
        STRF_IF_CONSTEXPR (detail::has_intpunct<FPack, IntTag, Base>()) {
            auto punct = get_facet<strf::numpunct_c<Base>, IntTag>(fp);
            if ( ! punct.no_group_separation(digcount_)) {
                groups_ = punct.groups();
                thousands_sep_ = punct.thousands_sep();
                auto charset = get_facet<strf::charset_c<CharT>, IntTag>(fp);
                init_punct_(charset);
            }
        }
        preview.subtract_width(width());
        calc_size(preview);
    }

    STRF_HD std::int16_t width() const
    {
        return static_cast<std::int16_t>( strf::detail::max(precision_, digcount_)
                                        + prefixsize_
                                        + static_cast<int>(sepcount_) );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;
    STRF_HD void calc_size(strf::size_preview<false>& ) const
    {
    }
    STRF_HD void calc_size(strf::size_preview<true>& ) const;

    STRF_HD void write_complement(strf::underlying_outbuf<CharSize>& ob) const;
    STRF_HD void write_digits(strf::underlying_outbuf<CharSize>& ob) const;

private:

    strf::encode_char_f<CharSize> encode_char_;
    unsigned long long uvalue_ = 0;
    strf::digits_groups_iterator groups_;
    char32_t thousands_sep_;
    unsigned precision_ = 0;
    unsigned digcount_ = 0;
    unsigned sepcount_ = 0;
    unsigned sepsize_ = 0;
    strf::lettercase lettercase_;
    bool negative_ = false;
    std::uint8_t prefixsize_ = 0;

    template <typename IntT>
    STRF_HD void init_(IntT value, strf::int_format_data fmt);

    template <typename Charset>
    STRF_HD void init_punct_(const Charset& cs);
};

template <std::size_t CharSize, int Base>
template <typename IntT>
STRF_HD void partial_fmt_int_printer<CharSize, Base>::init_
    ( IntT value
    , strf::int_format_data fmt )
{
    using unsigned_type = std::make_unsigned_t<IntT>;
    STRF_IF_CONSTEXPR (Base == 10) {
        negative_ = value < 0;
        prefixsize_ = negative_ || fmt.showpos;
        uvalue_ = strf::detail::unsigned_abs(value);
    } else {
        uvalue_ = unsigned_type(value);
        negative_ = false;
        prefixsize_ = static_cast<unsigned>(fmt.showbase)
            << static_cast<unsigned>(Base == 16 || Base == 2);
    }
    digcount_ = strf::detail::count_digits<Base>(uvalue_);
    precision_ = fmt.precision;
}

template <std::size_t CharSize, int Base>
template <typename Charset>
STRF_HD void partial_fmt_int_printer<CharSize, Base>::init_punct_(const Charset& cs)
{
    std::size_t sepsize = cs.validate(thousands_sep_);
    if (sepsize != (std::size_t)-1) {
        sepsize_ = static_cast<unsigned>(sepsize);
        sepcount_ = strf::sep_count(groups_, digcount_);
        if (sepsize_ == 1) {
            char_type little_sep[4];
            cs.encode_char(little_sep, thousands_sep_);
            thousands_sep_ = little_sep[0];
        } else {
            encode_char_ = cs.encode_char_func();
        }
    }
}

template <std::size_t CharSize, int Base>
STRF_HD void partial_fmt_int_printer<CharSize, Base>::calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t s = strf::detail::max(precision_, digcount_) + prefixsize_;
    if (sepcount_ > 0) {
        s += sepcount_ * sepsize_;
    }
    preview.add_size(s);
}

template <std::size_t CharSize, int Base>
STRF_HD inline void partial_fmt_int_printer<CharSize, Base>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (sepcount_ == 0) {
        ob.ensure(prefixsize_ + digcount_);
        auto it = ob.pointer();
        if (prefixsize_ != 0) {
            STRF_IF_CONSTEXPR (Base == 10) {
                * it = static_cast<char_type>('+') + (negative_ << 1);
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 8) {
                * it = static_cast<char_type>('0');
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 16) {
                it[0] = static_cast<char_type>('0');
                it[1] = static_cast<char_type>
                    ('X' | ((lettercase_ != strf::uppercase) << 5));
                it += 2;
            } else {
                it[0] = static_cast<char_type>('0');
                it[1] = static_cast<char_type>
                    ('B' | ((lettercase_ != strf::uppercase) << 5));
                it += 2;
            }
        }
        ob.advance_to(it);
        if (precision_ > digcount_) {
            unsigned zeros = precision_ - digcount_;
            strf::detail::write_fill(ob, zeros, char_type('0'));
        }
        strf::detail::write_int<Base>(ob, uvalue_, digcount_, lettercase_);
    } else {
        write_complement(ob);
        if (precision_ > digcount_) {
            unsigned zeros = precision_ - digcount_;
            strf::detail::write_fill(ob, zeros, char_type('0'));
        }
        if (sepsize_ == 1) {
            strf::detail::write_int_little_sep<Base>
                ( ob, uvalue_, groups_, digcount_, sepcount_
                , static_cast<char_type>(thousands_sep_), strf::lowercase );
        } else {
            strf::detail::write_int_big_sep<Base>
                ( ob, encode_char_, uvalue_, groups_, thousands_sep_
                , sepsize_, digcount_, strf::lowercase );
        }
    }
}

template <std::size_t CharSize, int Base>
inline STRF_HD void partial_fmt_int_printer<CharSize, Base>::write_complement
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (prefixsize_ != 0) {
        ob.ensure(prefixsize_);
        STRF_IF_CONSTEXPR (Base == 10) {
            * ob.pointer() = static_cast<char_type>('+') + (negative_ << 1);
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 8) {
            * ob.pointer() = static_cast<char_type>('0');
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 16) {
            ob.pointer()[0] = static_cast<char_type>('0');
            ob.pointer()[1] = static_cast<char_type>
                ('X' | ((lettercase_ != strf::uppercase) << 5));
            ob.advance(2);
        } else {
            ob.pointer()[0] = static_cast<char_type>('0');
            ob.pointer()[1] = static_cast<char_type>
                ('B' | ((lettercase_ != strf::uppercase) << 5));
            ob.advance(2);
        }
    }
}

template <std::size_t CharSize, int Base>
inline STRF_HD void partial_fmt_int_printer<CharSize, Base>::write_digits
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (precision_ > digcount_) {
        unsigned zeros = precision_ - digcount_;
        strf::detail::write_fill(ob, zeros, char_type('0'));
    }
    if (sepcount_ == 0) {
        strf::detail::write_int<Base>(ob, uvalue_, digcount_, lettercase_);
    } else if (sepsize_ == 1) {
        strf::detail::write_int_little_sep<Base>
            ( ob, uvalue_, groups_, digcount_, sepcount_
            , static_cast<char_type>(thousands_sep_), strf::lowercase );
    } else {
        strf::detail::write_int_big_sep<Base>
            ( ob, encode_char_, uvalue_, groups_, thousands_sep_, sepsize_
            , digcount_, strf::lowercase );
    }
}

template <std::size_t CharSize, int Base>
class full_fmt_int_printer: public printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template < typename CharT, typename FPack, typename Preview
             , typename ThisPrinter, typename IntT >
    STRF_HD full_fmt_int_printer
        ( const strf::usual_printer_input
            < CharT, FPack, Preview, ThisPrinter
            , strf::int_with_format<IntT, Base, true> >& i ) noexcept;

    STRF_HD ~full_fmt_int_printer();

    STRF_HD void print_to( strf::underlying_outbuf<CharSize>& ob ) const override;

private:

    strf::detail::partial_fmt_int_printer<CharSize, Base> ichars_;
    strf::encode_fill_f<CharSize> encode_fill_;
    unsigned fillcount_ = 0;
    strf::invalid_seq_policy inv_seq_poli_;
    strf::alignment_format_data afmt_;
    strf::surrogate_policy surr_poli_;

    template <typename Charset>
    STRF_HD  void calc_fill_size_
        ( strf::size_preview<false>&
        , const Charset& ) const
    {
    }

    template <typename Charset>
    STRF_HD void calc_fill_size_
        ( strf::size_preview<true>& preview
        , const Charset& cs ) const
    {
        if (fillcount_ > 0) {
            preview.add_size(fillcount_* cs.encoded_char_size(afmt_.fill));
        }
    }

    STRF_HD  void write_fill_
        ( strf::underlying_outbuf<CharSize>& ob
        , std::size_t count ) const
    {
        return encode_fill_
            ( ob, count, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
};

template <std::size_t CharSize, int Base>
template < typename CharT, typename FPack, typename Preview
         , typename ThisPrinter, typename IntT >
inline STRF_HD full_fmt_int_printer<CharSize, Base>::full_fmt_int_printer
    ( const strf::usual_printer_input
        < CharT, FPack, Preview, ThisPrinter
        , strf::int_with_format<IntT, Base, true> >& i ) noexcept
    : ichars_( i.fp, i.preview, i.arg.value().value
             , i.arg.get_int_format_data(), strf::tag<CharT>() )
    , inv_seq_poli_(get_facet<strf::invalid_seq_policy_c, IntT>(i.fp))
    , afmt_(i.arg.get_alignment_format_data())
    , surr_poli_(get_facet<strf::surrogate_policy_c, IntT>(i.fp))
{
    auto content_width = ichars_.width();
    if (afmt_.width > content_width) {
        fillcount_ = afmt_.width - content_width;
        i.preview.subtract_width(static_cast<std::int16_t>(fillcount_));
    }
    decltype(auto) cs = get_facet<strf::charset_c<CharT>, IntT>(i.fp);
    encode_fill_ = cs.encode_fill_func();
    calc_fill_size_(i.preview, cs);
}

template <std::size_t CharSize, int Base>
STRF_HD full_fmt_int_printer<CharSize, Base>::~full_fmt_int_printer()
{
}

template <std::size_t CharSize, int Base>
STRF_HD void full_fmt_int_printer<CharSize, Base>::print_to
        ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (fillcount_ == 0) {
        ichars_.print_to(ob);
    } else {
        switch(afmt_.alignment) {
            case strf::text_alignment::left: {
                ichars_.print_to(ob);
                write_fill_(ob, fillcount_);
                break;
            }
            case strf::text_alignment::split: {
                ichars_.write_complement(ob);
                write_fill_(ob, fillcount_);
                ichars_.write_digits(ob);
                break;
            }
            case strf::text_alignment::center: {
                auto halfcount = fillcount_ / 2;
                write_fill_(ob, halfcount);
                ichars_.print_to(ob);
                write_fill_(ob, fillcount_ - halfcount);
                break;
            }
            default: {
                write_fill_(ob, fillcount_);
                ichars_.print_to(ob);
            }
        }
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class int_printer<1>;
STRF_EXPLICIT_TEMPLATE class int_printer<2>;
STRF_EXPLICIT_TEMPLATE class int_printer<4>;

STRF_EXPLICIT_TEMPLATE class punct_int_printer<1>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<2>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<4>;

STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<1, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<2, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<4, 16>;

STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<1,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<1, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<1, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<2,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<2, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<2, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<4,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<4, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<4, 16>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

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
