#ifndef STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_INT_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/int_digits.hpp>
#include <strf/detail/standard_lib_functions.hpp>

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

constexpr STRF_HD auto tag_invoke(strf::fmt_tag, signed char x) noexcept
    -> strf::int_with_format<signed char>
{
    return strf::int_with_format<signed char>{strf::int_tag<signed char>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, short x) noexcept
    -> strf::int_with_format<short>
{
    return strf::int_with_format<short>{strf::int_tag<short>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, int x) noexcept
    -> strf::int_with_format<int>
{
    return strf::int_with_format<int>{strf::int_tag<int>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, long x) noexcept
    -> strf::int_with_format<long>
{
    return strf::int_with_format<long>{strf::int_tag<long>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, long long x) noexcept
    -> strf::int_with_format<long long>
{
    return strf::int_with_format<long long>{strf::int_tag<long long>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, unsigned char x) noexcept
    -> strf::int_with_format<unsigned char>
{
    return strf::int_with_format<unsigned char>{strf::int_tag<unsigned char>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, unsigned short x) noexcept
    -> strf::int_with_format<unsigned short>
{
    return strf::int_with_format<unsigned short>{strf::int_tag<unsigned short>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, unsigned int x) noexcept
    -> strf::int_with_format<unsigned int>
{
    return strf::int_with_format<unsigned int>{strf::int_tag<unsigned int>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, unsigned long x) noexcept
    -> strf::int_with_format<unsigned long>
{
    return strf::int_with_format<unsigned long>{strf::int_tag<unsigned long>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, unsigned long long x) noexcept
    -> strf::int_with_format<unsigned long long>
{
    return strf::int_with_format<unsigned long long>{strf::int_tag<unsigned long long>{x}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, const void* x) noexcept
    -> strf::value_with_format<const void*, strf::alignment_format>
{
    return strf::value_with_format<const void*, strf::alignment_format>{x};
}

namespace detail {

template <typename> class int_printer;
template <typename> class punct_int_printer;
template <typename, int> class partial_fmt_int_printer;
template <typename, int> class full_fmt_int_printer;

template <typename T>
constexpr STRF_HD bool negative_impl_(const T& x, std::integral_constant<bool, true>) noexcept
{
    return x < 0;
}
template <typename T>
constexpr STRF_HD bool negative_impl_(const T&, std::integral_constant<bool, false>) noexcept
{
    return false;
}
template <typename T>
constexpr STRF_HD bool negative(const T& x) noexcept
{
    return strf::detail::negative_impl_(x, std::is_signed<T>());
}

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

template <typename CharT, typename IntT, typename Preview>
struct nopunct_int_printer_input
{
    using printer_type = strf::detail::int_printer<CharT>;

    template<typename FPack>
    constexpr STRF_HD nopunct_int_printer_input
        ( IntT arg_, Preview& preview_, const FPack&) noexcept
        : value(arg_)
        , preview(preview_)
    {
    }

    constexpr STRF_HD nopunct_int_printer_input(IntT arg_, Preview& preview_) noexcept
        : value(arg_)
        , preview(preview_)
    {
    }

    IntT value;
    Preview& preview;
};

template <typename CharT, typename IntT, typename Preview, typename FPack>
struct punct_int_printer_input
{
    using printer_type = strf::detail::punct_int_printer<CharT>;

    IntT value;
    Preview& preview;
    FPack fp;
};

template <typename CharT, typename IntT, typename Preview, typename FPack>
using int_printer_input = std::conditional_t
    < strf::detail::has_intpunct<FPack, IntT, 10>()
    , strf::detail::punct_int_printer_input<CharT, IntT, Preview, FPack>
    , strf::detail::nopunct_int_printer_input<CharT, IntT, Preview> >;

} // namespace detail

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , signed char x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, signed char, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , short x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, short, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , int x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, int, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , long x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, long, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , long long x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, long long, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , unsigned char x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, unsigned char, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , unsigned short x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, unsigned short, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , unsigned int x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, unsigned int, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , unsigned long x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, unsigned long, Preview, FPack>
{
    return {x, preview, fp};
}
template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , unsigned long long x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::detail::int_printer_input<CharT, unsigned long long, Preview, FPack>
{
    return {x, preview, fp};
}

template < typename CharT, typename Preview, typename FPack
         , typename IntT, int Base, bool HasAlignment >
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , strf::int_with_format<IntT, Base, HasAlignment> x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, strf::int_with_format<IntT, Base, HasAlignment>, Preview, FPack
        , std::conditional_t
            < HasAlignment
            , strf::detail::full_fmt_int_printer<CharT, Base>
            , strf::detail::partial_fmt_int_printer<CharT, Base> > >
{
    return {x, preview, fp};
}

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , const void* x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT
        , strf::int_with_format<std::size_t, 16, false>
        , Preview
        , decltype
            ( strf::pack
                ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
                , strf::get_facet<strf::lettercase_c, const void*>(fp)
                , strf::get_facet<strf::char_encoding_c<CharT>, const void*>(fp) ) )
        , strf::detail::partial_fmt_int_printer<CharT, 16> >
{
    auto new_fp = strf::pack
            ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
            , strf::get_facet<strf::lettercase_c, const void*>(fp)
            , strf::get_facet<strf::char_encoding_c<CharT>, const void*>(fp) );
    return {*strf::hex(strf::detail::bit_cast<std::size_t>(x)), preview, new_fp};
}

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , strf::value_with_format<const void*, strf::alignment_format> x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT
        , strf::int_with_format<std::size_t, 16, true>
        , Preview
        , decltype
            ( strf::pack
                ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
                , strf::get_facet<strf::lettercase_c, const void*>(fp)
                , strf::get_facet<strf::char_encoding_c<CharT>, const void*>(fp) ) )
        , strf::detail::full_fmt_int_printer<CharT, 16> >
{
    auto new_fp = strf::pack
            ( strf::get_facet<strf::numpunct_c<16>, const void*>(fp)
            , strf::get_facet<strf::lettercase_c, const void*>(fp)
            , strf::get_facet<strf::char_encoding_c<CharT>, const void*>(fp) );
    return { *strf::hex(strf::detail::bit_cast<std::size_t>(x.value()))
               .set(x.get_alignment_format_data())
           , preview, new_fp };
}

namespace detail {

template <typename CharT>
class int_printer: public strf::printer<CharT>
{
public:

    template <typename Preview, typename IntT>
    STRF_HD int_printer(strf::detail::nopunct_int_printer_input<CharT, IntT, Preview> i)
    {
        init_(i.preview, i.value);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:
    template <typename Preview>
    STRF_HD void init_(Preview& p, signed char value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, short value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, int value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, long value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, long long value){ init2_(p, value); }
    template <typename Preview>
    STRF_HD void init_(Preview& p, unsigned char value){ init2_(p, value); }
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
        negative_ = strf::detail::negative(value);
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

template <typename CharT>
STRF_HD void int_printer<CharT>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    unsigned size = digcount_ + negative_;
    ob.ensure(size);
    auto* it = write_int_dec_txtdigits_backwards(uvalue_, ob.pointer() + size);
    if (negative_) {
        it[-1] = '-';
    }
    ob.advance(size);
}

template <typename CharT>
class punct_int_printer: public strf::printer<CharT>
{
public:

    template <typename IntT, typename... T>
    STRF_HD punct_int_printer
        ( const strf::detail::punct_int_printer_input<CharT, IntT, T...>& i )
    {
        auto enc = get_facet<strf::char_encoding_c<CharT>, IntT>(i.fp);

        uvalue_ = strf::detail::unsigned_abs(i.value);
        digcount_ = strf::detail::count_digits<10>(uvalue_);
        auto punct = get_facet<strf::numpunct_c<10>, IntT>(i.fp);
        if (punct.any_group_separation(digcount_)) {
            grouping_ = punct.grouping();
            thousands_sep_ = punct.thousands_sep();
            std::size_t sepsize = enc.validate(thousands_sep_);
            if (sepsize != (std::size_t)-1) {
                sepsize_ = static_cast<unsigned>(sepsize);
                sepcount_ = punct.thousands_sep_count(digcount_);
                if (sepsize_ == 1) {
                    CharT little_sep[4];
                    enc.encode_char(little_sep, thousands_sep_);
                    thousands_sep_ = little_sep[0];
                } else {
                    encode_char_ = enc.encode_char_func();
                }
            }
        }
        negative_ = strf::detail::negative(i.value);
        i.preview.add_size(digcount_ + negative_ + sepsize_ * sepcount_);
        i.preview.subtract_width
            ( static_cast<std::int16_t>(sepcount_ + digcount_ + negative_) );
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    strf::encode_char_f<CharT> encode_char_;
    strf::digits_grouping grouping_;
    char32_t thousands_sep_;
    unsigned long long uvalue_;
    unsigned digcount_;
    unsigned sepcount_ = 0;
    unsigned sepsize_ = 0;
    bool negative_;
};

template <typename CharT>
STRF_HD void punct_int_printer<CharT>::print_to(strf::basic_outbuff<CharT>& ob) const
{
    if (sepcount_ == 0) {
        ob.ensure(negative_ + digcount_);
        auto it = ob.pointer();
        if (negative_) {
            *it = static_cast<CharT>('-');
            ++it;
        }
        it += digcount_;
        strf::detail::write_int_dec_txtdigits_backwards(uvalue_, it);
        ob.advance_to(it);
    } else {
        if (negative_) {
            put(ob, static_cast<CharT>('-'));
        }
        if (sepsize_ == 1) {
            strf::detail::write_int_little_sep<10>
                ( ob, uvalue_, grouping_, digcount_, sepcount_
                , static_cast<CharT>(thousands_sep_), strf::lowercase );
        } else {
            strf::detail::write_int_big_sep<10>
                ( ob, encode_char_, uvalue_, grouping_, thousands_sep_, sepsize_
                , digcount_, strf::lowercase );
        }
    }
}

template <typename CharT, int Base>
class partial_fmt_int_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD partial_fmt_int_printer
        ( const strf::usual_printer_input<T...> & i)
        : partial_fmt_int_printer
          ( i.arg.value().value, i.arg.get_int_format_data()
          , i.preview, i.fp )
    {
    }

    template < typename FPack
             , typename Preview
             , typename IntT
             , typename IntTag = IntT >
    STRF_HD partial_fmt_int_printer
        ( IntT value, int_format_data fdata, Preview& preview, const FPack& fp )
        : lettercase_(get_facet<strf::lettercase_c, IntTag>(fp))
    {
        init_<IntT>( value, fdata );
        STRF_IF_CONSTEXPR (detail::has_intpunct<FPack, IntTag, Base>()) {
            auto punct = get_facet<strf::numpunct_c<Base>, IntTag>(fp);
            if (punct.any_group_separation(digcount_)) {
                grouping_ = punct.grouping();
                thousands_sep_ = punct.thousands_sep();
                auto encoding = get_facet<strf::char_encoding_c<CharT>, IntTag>(fp);
                init_punct_(encoding);
            }
        }
        preview.subtract_width(width());
        calc_size(preview);
    }

    STRF_HD std::int16_t width() const
    {
        return static_cast<std::int16_t>( (precision_ > digcount_ ? precision_ : digcount_)
                                        + prefixsize_
                                        + static_cast<int>(sepcount_) );
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;
    STRF_HD void calc_size(strf::size_preview<false>& ) const
    {
    }
    STRF_HD void calc_size(strf::size_preview<true>& ) const;

    STRF_HD void write_complement(strf::basic_outbuff<CharT>& ob) const;
    STRF_HD void write_digits(strf::basic_outbuff<CharT>& ob) const;

private:

    strf::encode_char_f<CharT> encode_char_;
    unsigned long long uvalue_ = 0;
    strf::digits_grouping grouping_;
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

    template <typename Encoding>
    STRF_HD void init_punct_(Encoding enc);
};

template <typename CharT, int Base>
template <typename IntT>
STRF_HD void partial_fmt_int_printer<CharT, Base>::init_
    ( IntT value
    , strf::int_format_data fmt )
{
    using unsigned_type = std::make_unsigned_t<IntT>;
    STRF_IF_CONSTEXPR (Base == 10) {
        negative_ = strf::detail::negative(value);
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

template <typename CharT, int Base>
template <typename Encoding>
STRF_HD void partial_fmt_int_printer<CharT, Base>::init_punct_(Encoding enc)
{
    std::size_t sepsize = enc.validate(thousands_sep_);
    if (sepsize != (std::size_t)-1) {
        sepsize_ = static_cast<unsigned>(sepsize);
        sepcount_ = grouping_.separators_count(digcount_);
        if (sepsize_ == 1) {
            CharT little_sep[4];
            enc.encode_char(little_sep, thousands_sep_);
            thousands_sep_ = little_sep[0];
        } else {
            encode_char_ = enc.encode_char_func();
        }
    }
}

template <typename CharT, int Base>
STRF_HD void partial_fmt_int_printer<CharT, Base>::calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t s = prefixsize_ + (precision_ > digcount_ ? precision_ : digcount_);
    if (sepcount_ > 0) {
        s += sepcount_ * sepsize_;
    }
    preview.add_size(s);
}

template <typename CharT, int Base>
STRF_HD inline void partial_fmt_int_printer<CharT, Base>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    if (sepcount_ == 0) {
        ob.ensure(prefixsize_ + digcount_);
        auto it = ob.pointer();
        if (prefixsize_ != 0) {
            STRF_IF_CONSTEXPR (Base == 10) {
                * it = static_cast<CharT>('+') + (negative_ << 1);
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 8) {
                * it = static_cast<CharT>('0');
                ++ it;
            } else STRF_IF_CONSTEXPR (Base == 16) {
                it[0] = static_cast<CharT>('0');
                it[1] = static_cast<CharT>
                    ('X' | ((lettercase_ != strf::uppercase) << 5));
                it += 2;
            } else {
                it[0] = static_cast<CharT>('0');
                it[1] = static_cast<CharT>
                    ('B' | ((lettercase_ != strf::uppercase) << 5));
                it += 2;
            }
        }
        ob.advance_to(it);
        if (precision_ > digcount_) {
            unsigned zeros = precision_ - digcount_;
            strf::detail::write_fill(ob, zeros, CharT('0'));
        }
        strf::detail::write_int<Base>(ob, uvalue_, digcount_, lettercase_);
    } else {
        write_complement(ob);
        if (precision_ > digcount_) {
            unsigned zeros = precision_ - digcount_;
            strf::detail::write_fill(ob, zeros, CharT('0'));
        }
        if (sepsize_ == 1) {
            strf::detail::write_int_little_sep<Base>
                ( ob, uvalue_, grouping_, digcount_, sepcount_
                , static_cast<CharT>(thousands_sep_), strf::lowercase );
        } else {
            strf::detail::write_int_big_sep<Base>
                ( ob, encode_char_, uvalue_, grouping_, thousands_sep_
                , sepsize_, digcount_, strf::lowercase );
        }
    }
}

template <typename CharT, int Base>
inline STRF_HD void partial_fmt_int_printer<CharT, Base>::write_complement
    ( strf::basic_outbuff<CharT>& ob ) const
{
    if (prefixsize_ != 0) {
        ob.ensure(prefixsize_);
        STRF_IF_CONSTEXPR (Base == 10) {
            * ob.pointer() = static_cast<CharT>('+') + (negative_ << 1);
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 8) {
            * ob.pointer() = static_cast<CharT>('0');
            ob.advance(1);
        } else STRF_IF_CONSTEXPR (Base == 16) {
            ob.pointer()[0] = static_cast<CharT>('0');
            ob.pointer()[1] = static_cast<CharT>
                ('X' | ((lettercase_ != strf::uppercase) << 5));
            ob.advance(2);
        } else {
            ob.pointer()[0] = static_cast<CharT>('0');
            ob.pointer()[1] = static_cast<CharT>
                ('B' | ((lettercase_ != strf::uppercase) << 5));
            ob.advance(2);
        }
    }
}

template <typename CharT, int Base>
inline STRF_HD void partial_fmt_int_printer<CharT, Base>::write_digits
    ( strf::basic_outbuff<CharT>& ob ) const
{
    if (precision_ > digcount_) {
        unsigned zeros = precision_ - digcount_;
        strf::detail::write_fill(ob, zeros, CharT('0'));
    }
    if (sepcount_ == 0) {
        strf::detail::write_int<Base>(ob, uvalue_, digcount_, lettercase_);
    } else if (sepsize_ == 1) {
        strf::detail::write_int_little_sep<Base>
            ( ob, uvalue_, grouping_, digcount_, sepcount_
            , static_cast<CharT>(thousands_sep_), strf::lowercase );
    } else {
        strf::detail::write_int_big_sep<Base>
            ( ob, encode_char_, uvalue_, grouping_, thousands_sep_, sepsize_
            , digcount_, strf::lowercase );
    }
}

template <typename CharT, int Base>
class full_fmt_int_printer: public printer<CharT>
{
public:

    template <typename IntT, typename... T>
    STRF_HD full_fmt_int_printer
        ( const strf::usual_printer_input
            < CharT, strf::int_with_format<IntT, Base, true>, T... >& i ) noexcept;

    STRF_HD ~full_fmt_int_printer();

    STRF_HD void print_to( strf::basic_outbuff<CharT>& ob ) const override;

private:

    strf::detail::partial_fmt_int_printer<CharT, Base> ichars_;
    strf::encode_fill_f<CharT> encode_fill_;
    unsigned fillcount_ = 0;
    strf::alignment_format_data afmt_;

    template <typename Encoding>
    STRF_HD  void calc_fill_size_
        ( strf::size_preview<false>&
        , Encoding ) const
    {
    }

    template <typename Encoding>
    STRF_HD void calc_fill_size_
        ( strf::size_preview<true>& preview
        , Encoding enc ) const
    {
        if (fillcount_ > 0) {
            preview.add_size(fillcount_* enc.encoded_char_size(afmt_.fill));
        }
    }

    STRF_HD  void write_fill_
        ( strf::basic_outbuff<CharT>& ob
        , std::size_t count ) const
    {
        return encode_fill_(ob, count, afmt_.fill);
    }
};

template <typename CharT, int Base>
template < typename IntT, typename... T >
inline STRF_HD full_fmt_int_printer<CharT, Base>::full_fmt_int_printer
    ( const strf::usual_printer_input
          < CharT, strf::int_with_format<IntT, Base, true>, T... >& i ) noexcept
    : ichars_(i.arg.value().value, i.arg.get_int_format_data(), i.preview, i.fp)
    , afmt_(i.arg.get_alignment_format_data())
{
    auto content_width = ichars_.width();
    if (afmt_.width > content_width) {
        fillcount_ = afmt_.width - content_width;
        i.preview.subtract_width(static_cast<std::int16_t>(fillcount_));
    }
    auto enc = get_facet<strf::char_encoding_c<CharT>, IntT>(i.fp);
    encode_fill_ = enc.encode_fill_func();
    calc_fill_size_(i.preview, enc);
}

template <typename CharT, int Base>
STRF_HD full_fmt_int_printer<CharT, Base>::~full_fmt_int_printer()
{
}

template <typename CharT, int Base>
STRF_HD void full_fmt_int_printer<CharT, Base>::print_to
        ( strf::basic_outbuff<CharT>& ob ) const
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

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class int_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char8_t, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char8_t,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char8_t, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char8_t, 16>;
#endif

STRF_EXPLICIT_TEMPLATE class int_printer<char>;
STRF_EXPLICIT_TEMPLATE class int_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class int_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class int_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class punct_int_printer<char>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class punct_int_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char16_t, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<char32_t, 16>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t,  8>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t, 10>;
STRF_EXPLICIT_TEMPLATE class partial_fmt_int_printer<wchar_t, 16>;

STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char16_t,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char16_t, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char16_t, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char32_t,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char32_t, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<char32_t, 16>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<wchar_t,  8>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<wchar_t, 10>;
STRF_EXPLICIT_TEMPLATE class full_fmt_int_printer<wchar_t, 16>;

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
