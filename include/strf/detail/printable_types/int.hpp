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

struct int_format
{
    unsigned precision = 0;
    unsigned pad0width = 0;
    strf::showsign sign = strf::showsign::negative_only;
    bool showbase = false;
    bool punctuate = false;
    int base = 10;
};

using int_format_full_dynamic = int_format;

template <int Base>
struct int_format_static_base
{
    unsigned precision = 0;
    unsigned pad0width = 0;
    strf::showsign sign = strf::showsign::negative_only;
    bool showbase = false;
    bool punctuate = false;
    constexpr static int base = Base;

    constexpr STRF_HD operator strf::int_format_full_dynamic () const
    {
        return {precision, pad0width, sign, showbase, punctuate, base};
    }
};

struct default_int_format
{
    constexpr static unsigned precision = 0;
    constexpr static unsigned pad0width = 0;
    constexpr static strf::showsign sign = strf::showsign::negative_only;
    constexpr static bool showbase = false;
    constexpr static bool punctuate = false;
    constexpr static int base = 10;

    constexpr STRF_HD operator strf::int_format_static_base<10> () const
    {
        return {};
    }
    constexpr STRF_HD operator strf::int_format_full_dynamic () const
    {
        return {};
    }
};


template <int ToBase, int FromBase>
constexpr STRF_HD int_format_static_base<ToBase> change_base(int_format_static_base<FromBase> f) noexcept
{
    return {f.precision, f.pad0width, f.sign, f.punctuate, f.showbase};
}

template <int ToBase>
constexpr STRF_HD int_format_static_base<ToBase> change_base(default_int_format) noexcept
{
    return {};
}

template <int Base>
constexpr STRF_HD bool operator==( strf::int_format_static_base<Base> lhs
                                 , strf::int_format_static_base<Base> rhs ) noexcept
{
    return lhs.precision == rhs.precision
        && lhs.pad0width == rhs.pad0width
        && lhs.sign == rhs.sign
        && lhs.punctuate == rhs.punctuate
        && lhs.showbase == rhs.showbase;
}

template <int Base>
constexpr STRF_HD bool operator!=( strf::int_format_static_base<Base> lhs
                                 , strf::int_format_static_base<Base> rhs ) noexcept
{
    return ! (lhs == rhs);
}

template <class T>
class default_int_formatter_fn;

template <class T, int Base>
class int_formatter_static_base_fn;

template <class T>
class int_formatter_full_dynamic_fn;

struct default_int_formatter
{
    template <typename T>
    using fn = strf::default_int_formatter_fn<T>;
};

template <int Base>
struct int_formatter_static_base
{
    template <typename T>
    using fn = int_formatter_static_base_fn<T, Base>;
};

struct int_formatter_full_dynamic
{
    template <typename T>
    using fn = strf::int_formatter_full_dynamic_fn<T>;
};

using int_formatter = default_int_formatter;

template <class T>
class default_int_formatter_fn
{
    template <int Base>
    using adapted_derived_type_
        = strf::fmt_replace<T, default_int_formatter, int_formatter_static_base<Base> >;

public:

    constexpr default_int_formatter_fn() noexcept = default;
    constexpr default_int_formatter_fn(const default_int_formatter_fn&) noexcept = default;

    template <typename U>
    constexpr STRF_HD default_int_formatter_fn(const default_int_formatter_fn<U>&) noexcept
    {
    }
    constexpr STRF_HD explicit default_int_formatter_fn(default_int_format) noexcept
    {
    }
    constexpr STRF_HD T&& dec() && noexcept
    {
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD const T&& dec() const && noexcept
    {
        return static_cast<const T&&>(*this);
    }
    constexpr STRF_HD T& dec() & noexcept
    {
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD const T& dec() const & noexcept
    {
        return *static_cast<const T*>(this);
    }
    constexpr STRF_HD adapted_derived_type_<16> hex() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<16>>{}
               , int_format_static_base<16>{} };
    }
    constexpr STRF_HD adapted_derived_type_<8> oct() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<8>>{}
               , int_format_static_base<8>{} };
    }
    constexpr STRF_HD adapted_derived_type_<2> bin() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<2>>{}
               , int_format_static_base<2>{} };
    }
    constexpr STRF_HD adapted_derived_type_<10> p(unsigned _) const noexcept
    {
        int_format_static_base<10> data;
        data.precision = _;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD adapted_derived_type_<10> pad0(unsigned w) && noexcept
    {
        int_format_static_base<10> data;
        data.pad0width = w;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD adapted_derived_type_<10> operator+() && noexcept
    {
        int_format_static_base<10> data;
        data.sign = strf::showsign::positive_also;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD adapted_derived_type_<10> fill_sign() && noexcept
    {
        int_format_static_base<10> data;
        data.sign = strf::showsign::fill_instead_of_positive;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD adapted_derived_type_<10> operator~() && noexcept
    {
        int_format_static_base<10> data;
        data.sign = strf::showsign::fill_instead_of_positive;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD void operator*() && noexcept = delete;
    constexpr STRF_HD adapted_derived_type_<10> punct() && noexcept
    {
        int_format_static_base<10> data;
        data.punctuate = true;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD adapted_derived_type_<10> operator!() && noexcept
    {
        int_format_static_base<10> data;
        data.punctuate = true;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<10>>{}
               , data };
    }
    constexpr STRF_HD
    strf::fmt_replace<T, default_int_formatter, int_formatter_full_dynamic >
    base(int b) const
    {
        int_format_full_dynamic data;
        data.base = b;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_full_dynamic>{}
               , data };
    }
    constexpr static STRF_HD int base() noexcept
    {
        return 10;
    }
    constexpr static STRF_HD strf::default_int_format get_int_format() noexcept
    {
        return {};
    }
    constexpr STRF_HD T&& set_int_format(strf::default_int_format) && noexcept
    {
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& set_int_format(strf::default_int_format) & noexcept
    {
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD const T&&
    set_int_format(strf::default_int_format) const && noexcept
    {
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD const T&
    set_int_format(strf::default_int_format) const & noexcept
    {
        return static_cast<T&>(*this);
    }
    template <int Base>
    constexpr STRF_HD adapted_derived_type_<Base>
    set_int_format(const strf::int_format_static_base<Base>& data) const & noexcept
    {
        return adapted_derived_type_<Base>
               { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base<Base>>{}
               , data };
    }
    constexpr STRF_HD
    strf::fmt_replace<T, default_int_formatter, int_formatter_full_dynamic >
    set_int_format(strf::int_format_full_dynamic data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_full_dynamic>{}
               , data };
    }
};

template <class T, int Base>
class int_formatter_static_base_fn
{
private:

    template <int OtherBase>
    using adapted_derived_type_ = strf::fmt_replace
        < T
        , int_formatter_static_base<Base>
        , int_formatter_static_base<OtherBase> >;

public:

    constexpr STRF_HD int_formatter_static_base_fn()  noexcept = default;

    template <typename U>
    constexpr STRF_HD int_formatter_static_base_fn
        ( const int_formatter_static_base_fn<U, Base> & u ) noexcept
        : data_(u.get_int_format())
    {
    }

    constexpr STRF_HD explicit int_formatter_static_base_fn
        ( int_format_static_base<Base> data ) noexcept
        : data_(data)
    {
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 16, T&&> hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 16, adapted_derived_type_<B>>
    hex() const &
    {
        return adapted_derived_type_<B>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<B>>{}
            , strf::change_base<B>(data_) };
    }
    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 10, T&&> dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 10, adapted_derived_type_<B>>
    dec() const &
    {
        return adapted_derived_type_<B>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<B>>{}
            , strf::change_base<B>(data_) };
    }
    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 8 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 8, adapted_derived_type_<B>>
    oct() const &
    {
        return adapted_derived_type_<B>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<B>>{}
            , strf::change_base<B>(data_) };
    }
    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base == B && B == 2, T&&>
    bin() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 2 >
    constexpr STRF_HD std::enable_if_t<Base != B && B == 2, adapted_derived_type_<B>>
    bin() const &
    {
        return adapted_derived_type_<B>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<B>>{}
            , strf::change_base<B>(data_) };
    }
    constexpr STRF_HD T&& p(unsigned _) && noexcept
    {
        data_.precision = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& pad0(unsigned w) && noexcept
    {
        data_.pad0width = w;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    constexpr STRF_HD T&& operator+() && noexcept
    {
        static_assert(DecimalBase, "operator+ only allowed in decimal base");
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    constexpr STRF_HD T&& fill_sign() && noexcept
    {
        static_assert(DecimalBase, "fill_sign() only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    constexpr STRF_HD T&& operator~() && noexcept
    {
        static_assert(DecimalBase, "operator~ only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    constexpr STRF_HD T&& operator*() && noexcept
    {
        static_assert(!DecimalBase, "operator* not allowed in decimal base");
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& punct() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& operator!() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD
    strf::fmt_replace<T, int_formatter_static_base<Base>, int_formatter_full_dynamic >
    base(int b) const
    {
        int_format_full_dynamic data;
        data.base = b;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_full_dynamic>{}
               , data };
    }
    constexpr static STRF_HD int base() noexcept
    {
        return Base;
    }
    constexpr STRF_HD strf::int_format_static_base<Base> get_int_format() const noexcept
    {
        return data_;
    }
    constexpr STRF_HD T&& set_int_format(strf::int_format_static_base<Base> data) && noexcept
    {
        data_ = data;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& set_int_format(strf::int_format_static_base<Base> data) & noexcept
    {
        data_ = data;
        return static_cast<T&>(*this);
    }
    template <int OtherBase>
    constexpr STRF_HD std::enable_if_t<Base != OtherBase, adapted_derived_type_<OtherBase>>
    set_int_format(strf::int_format_static_base<OtherBase> data) const & noexcept
    {
        return adapted_derived_type_<OtherBase>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<OtherBase>>{}
            , data };
    }
    constexpr STRF_HD
    strf::fmt_replace<T, int_formatter_static_base<Base>, default_int_formatter >
    set_int_format(strf::default_int_format) const & noexcept
    {
        return { * static_cast<const T*>(this) };
    }
    constexpr STRF_HD
    strf::fmt_replace<T, int_formatter_static_base<Base>, int_formatter_full_dynamic >
    set_int_format(strf::int_format_full_dynamic data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_full_dynamic>{}
               , data };
    }

private:

    strf::int_format_static_base<Base> data_;
};

template <typename T>
class int_formatter_full_dynamic_fn
{
public:

    constexpr STRF_HD int_formatter_full_dynamic_fn()  noexcept = default;

    constexpr STRF_HD int_formatter_full_dynamic_fn
        ( int_format_full_dynamic data )  noexcept
        : data_(data)
    {
    }

    constexpr STRF_HD int_formatter_full_dynamic_fn
        ( const int_formatter_full_dynamic_fn& ) noexcept = default;

    template <typename U>
    constexpr STRF_HD explicit int_formatter_full_dynamic_fn
        ( const int_formatter_full_dynamic_fn<U>& other)  noexcept
        : data_(other.data_)
    {
    }

    constexpr STRF_HD T&& hex() &&
    {
        data_.base = 16;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& hex() &
    {
        data_.base = 16;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& dec() &&
    {
        data_.base = 10;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& dec() &
    {
        data_.base = 10;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& oct() &&
    {
        data_.base = 8;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& oct() &
    {
        data_.base = 8;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& bin() &&
    {
        data_.base = 2;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& bin() &
    {
        data_.base = 2;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& p(unsigned _) && noexcept
    {
        data_.precision = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& p(unsigned _) & noexcept
    {
        data_.precision = _;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& pad0(unsigned w) && noexcept
    {
        data_.pad0width = w;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& pad0(unsigned w) & noexcept
    {
        data_.pad0width = w;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& operator+() && noexcept
    {
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& operator+() & noexcept
    {
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& fill_sign() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& fill_sign() & noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& operator~() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& operator~() & noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& operator*() && noexcept
    {
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& operator*() & noexcept
    {
        data_.showbase = true;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& punct() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& punct() & noexcept
    {
        data_.punctuate = true;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& operator!() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& operator!() & noexcept
    {
        data_.punctuate = true;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&& base(int b) && noexcept
    {
        data_.base = b;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T& base(int b) & noexcept
    {
        data_.base = b;
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD T&&
    set_int_format(strf::int_format_full_dynamic data) && noexcept
    {
        data_ = data;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&
    set_int_format(strf::int_format_full_dynamic data) & noexcept
    {
        data_ = data;
        return static_cast<T&>(*this);
    }
    template <int Base>
    constexpr STRF_HD
    strf::fmt_replace<T, int_formatter_full_dynamic, int_formatter_static_base<Base> >
    set_int_format(strf::int_format_static_base<Base> data) const & noexcept
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base<Base>>{}
            , data };

    }
    constexpr STRF_HD
    strf::fmt_replace<T, int_formatter_full_dynamic, default_int_formatter >
    set_int_format(strf::default_int_format data) const & noexcept
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::default_int_formatter>{}
            , data };
    }

    constexpr STRF_HD const int_format_full_dynamic& get_int_format() const
    {
        return data_;
    }

private:

    int_format_full_dynamic data_;
};

namespace detail {

template <typename> class default_int_printer;
//template <typename> class punct_int_printer;
template <typename, int> class punct_fmt_int_printer;
template <typename> class int_printer_full_dynamic;

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

// template <typename FPack, typename IntT, unsigned Base>
// class has_intpunct_impl
// {
// public:
//
//     static STRF_HD std::true_type  test_numpunct(strf::numpunct<Base>);
//     static STRF_HD std::false_type test_numpunct(strf::default_numpunct<Base>);
//     static STRF_HD std::false_type test_numpunct(strf::no_grouping<Base>);
//
//     static STRF_HD const FPack& fp();
//
//     using has_numpunct_type = decltype
//         ( test_numpunct
//             ( get_facet< strf::numpunct_c<Base>, IntT >(fp())) );
// public:
//
//     static constexpr bool value = has_numpunct_type::value;
// };
//
// template <typename FPack, typename IntT, unsigned Base>
// constexpr STRF_HD bool has_intpunct()
// {
//     return has_intpunct_impl<FPack, IntT, Base>::value;
// }

template <typename IntT>
struct int_printing;

template <typename CharT, typename Preview, typename IntT>
struct default_int_printer_input
{
    using printer_type = strf::detail::default_int_printer<CharT>;

    template<typename FPack>
    constexpr STRF_HD default_int_printer_input
        ( Preview& preview_, const FPack&, IntT arg_) noexcept
        : preview(preview_)
        , value(arg_)
    {
    }

    template<typename FPack, typename PTraits>
    constexpr STRF_HD default_int_printer_input
        ( Preview& preview_
        , const FPack&
        , strf::value_with_formatters
            < PTraits
            , strf::int_formatter
            , strf::alignment_formatter > arg_ ) noexcept
        : preview(preview_)
        , value(arg_.value())
    {
    }

    Preview& preview;
    IntT value;
};

template <typename IntT>
struct int_printing
{
private:

    template <typename P, bool HasAlignment>
    using vwf_ = value_with_formatters
              < P
              , int_formatter
              , alignment_formatter_q<HasAlignment> >;

    template <typename P, int Base, bool HasAlignment>
    using vwf_b_ = value_with_formatters
              < P
              , int_formatter_static_base<Base>
              , alignment_formatter_q<HasAlignment> >;

    template <typename P, bool HasAlignment>
    using vwf_full_dynamic_ = value_with_formatters
              < P
              , int_formatter_full_dynamic
              , alignment_formatter_q<HasAlignment> >;


public:

    using override_tag = IntT;
    using forwarded_type = IntT;
    using formatters = strf::tag< strf::int_formatter
                                , strf::alignment_formatter >;

    template <typename CharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& facets,  IntT x ) noexcept
        -> strf::detail::default_int_printer_input<CharT, Preview, IntT>
    {
        return {preview, facets, x};
    }

    template < typename CharT, typename Preview, typename FPack
             , typename PTraits, int Base, bool HasAlignment >
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& facets
        , vwf_b_<PTraits, Base, HasAlignment> x )
        -> strf::usual_printer_input
            < CharT, Preview, FPack, vwf_b_<PTraits, Base, HasAlignment>
            , strf::detail::punct_fmt_int_printer<CharT, Base> >
    {
        return {preview, facets, x};
    }

    template < typename CharT, typename Preview, typename FPack
             , typename PTraits, bool HasAlignment>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& facets
        , vwf_<PTraits, HasAlignment> x )
        -> std::conditional_t
            < ! HasAlignment
            , strf::detail::default_int_printer_input<CharT, Preview, IntT>
            , strf::usual_printer_input
                < CharT, Preview, FPack, vwf_<PTraits, HasAlignment>
                , strf::detail::punct_fmt_int_printer<CharT, 10> > >
    {
        return {preview, facets, x};
    }

    template < typename CharT, typename Preview, typename FPack
             , typename PTraits, bool HasAlignment>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& facets
        , vwf_full_dynamic_<PTraits, HasAlignment> x )
        -> strf::usual_printer_input
                < CharT, Preview, FPack, vwf_full_dynamic_<PTraits, HasAlignment>
                , strf::detail::int_printer_full_dynamic<CharT> >
    {
        return {preview, facets, x};
    }
};

} // namespace detail

template <> struct print_traits<signed char>:
    public strf::detail::int_printing<signed char> {};
template <> struct print_traits<short>:
    public strf::detail::int_printing<short> {};
template <> struct print_traits<int>:
    public strf::detail::int_printing<int> {};
template <> struct print_traits<long>:
    public strf::detail::int_printing<long> {};
template <> struct print_traits<long long>:
    public strf::detail::int_printing<long long> {};

template <> struct print_traits<unsigned char>:
    public strf::detail::int_printing<unsigned char> {};
template <> struct print_traits<unsigned short>:
    public strf::detail::int_printing<unsigned short> {};
template <> struct print_traits<unsigned int>:
    public strf::detail::int_printing<unsigned int> {};
template <> struct print_traits<unsigned long>:
    public strf::detail::int_printing<unsigned long> {};
template <> struct print_traits<unsigned long long>:
    public strf::detail::int_printing<unsigned long long> {};

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, signed char) noexcept
    -> strf::detail::int_printing<signed char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, short) noexcept
    -> strf::detail::int_printing<short>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, int) noexcept
    -> strf::detail::int_printing<int>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, long) noexcept
    -> strf::detail::int_printing<long>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, long long) noexcept
    -> strf::detail::int_printing<long long>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, unsigned char) noexcept
    -> strf::detail::int_printing<unsigned char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, unsigned short) noexcept
    -> strf::detail::int_printing<unsigned short>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, unsigned int) noexcept
    -> strf::detail::int_printing<unsigned int>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, unsigned long) noexcept
    -> strf::detail::int_printing<unsigned long>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, unsigned long long) noexcept
    -> strf::detail::int_printing<unsigned long long>
    { return {}; }

namespace detail {

struct voidptr_printing
{
    using override_tag = const void*;
    using forwarded_type = const void*;
    using formatters = strf::tag<strf::alignment_formatter>;

    template <typename CharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& facets, const void* x ) noexcept
    {
        auto f1 = strf::get_facet<strf::numpunct_c<16>, const void*>(facets);
        auto f2 = strf::get_facet<strf::lettercase_c, const void*>(facets);
        auto f3 = strf::get_facet<strf::char_encoding_c<CharT>, const void*>(facets);
        auto fp2 = strf::pack(f1, f2, f3);
        auto x2 = *strf::hex(strf::detail::bit_cast<std::size_t>(x));
        return strf::make_default_printer_input<CharT>(preview, fp2, x2);
    }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& facets
        , strf::value_with_formatters<T...> x ) noexcept
    {
        auto f1 = strf::get_facet<strf::numpunct_c<16>, const void*>(facets);
        auto f2 = strf::get_facet<strf::lettercase_c, const void*>(facets);
        auto f3 = strf::get_facet<strf::char_encoding_c<CharT>, const void*>(facets);
        auto fp2 = strf::pack(f1, f2, f3);
        auto x2 = *strf::hex(strf::detail::bit_cast<std::size_t>(x.value()))
                             .set_alignment_format(x.get_alignment_format());
        return strf::make_default_printer_input<CharT>(preview, fp2, x2);
    }
};

} // namespace detail

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const void*) noexcept
    -> strf::detail::voidptr_printing
    { return {}; }

namespace detail {

template <typename CharT>
class default_int_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD default_int_printer(strf::detail::default_int_printer_input<T...> i)
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
STRF_HD void default_int_printer<CharT>::print_to
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

// template <typename CharT>
// class punct_int_printer: public strf::printer<CharT>
// {
// public:

//     template <typename... T>
//     STRF_HD punct_int_printer
//         ( const strf::detail::punct_int_printer_input<T...>& i )
//     {
//         using int_type = typename strf::detail::punct_int_printer_input<T...>::arg_type;
//         auto enc = get_facet<strf::char_encoding_c<CharT>, int_type>(i.facets);

//         uvalue_ = strf::detail::unsigned_abs(i.value);
//         digcount_ = strf::detail::count_digits<10>(uvalue_);
//         auto punct = get_facet<strf::numpunct_c<10>, int_type>(i.facets);
//         if (punct.any_group_separation(digcount_)) {
//             grouping_ = punct.grouping();
//             thousands_sep_ = punct.thousands_sep();
//             std::size_t sepsize = enc.validate(thousands_sep_);
//             if (sepsize != (std::size_t)-1) {
//                 sepsize_ = static_cast<unsigned>(sepsize);
//                 sepcount_ = punct.thousands_sep_count(digcount_);
//                 if (sepsize_ == 1) {
//                     CharT little_sep[4];
//                     enc.encode_char(little_sep, thousands_sep_);
//                     thousands_sep_ = little_sep[0];
//                 } else {
//                     encode_char_ = enc.encode_char_func();
//                 }
//             }
//         }
//         negative_ = strf::detail::negative(i.value);
//         i.preview.add_size(digcount_ + negative_ + sepsize_ * sepcount_);
//         i.preview.subtract_width
//             ( static_cast<std::int16_t>(sepcount_ + digcount_ + negative_) );
//     }

//     STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

// private:

//     strf::encode_char_f<CharT> encode_char_;
//     strf::digits_grouping grouping_;
//     char32_t thousands_sep_;
//     unsigned long long uvalue_;
//     unsigned digcount_;
//     unsigned sepcount_ = 0;
//     unsigned sepsize_ = 0;
//     bool negative_;
// };

// template <typename CharT>
// STRF_HD void punct_int_printer<CharT>::print_to(strf::basic_outbuff<CharT>& ob) const
// {
//     if (sepcount_ == 0) {
//         ob.ensure(negative_ + digcount_);
//         auto it = ob.pointer();
//         if (negative_) {
//             *it = static_cast<CharT>('-');
//             ++it;
//         }
//         it += digcount_;
//         strf::detail::write_int_dec_txtdigits_backwards(uvalue_, it);
//         ob.advance_to(it);
//     } else {
//         if (negative_) {
//             put(ob, static_cast<CharT>('-'));
//         }
//         if (sepsize_ == 1) {
//             strf::detail::write_int_little_sep<10>
//                 ( ob, uvalue_, grouping_, digcount_, sepcount_
//                 , static_cast<CharT>(thousands_sep_), strf::lowercase );
//         } else {
//             strf::detail::write_int_big_sep<10>
//                 ( ob, encode_char_, uvalue_, grouping_, thousands_sep_, sepsize_
//                 , digcount_, strf::lowercase );
//         }
//     }
// }

struct fmt_int_printer_data {
    unsigned long long uvalue;
    unsigned digcount;
    unsigned leading_zeros;
    unsigned left_fillcount;
    unsigned right_fillcount;
    char32_t fillchar;
    bool has_prefix;
    char sign;
};

struct punct_fmt_int_printer_data: public fmt_int_printer_data {
    unsigned sepcount;
    unsigned sepsize;
    char32_t sepchar;
    strf::digits_grouping grouping;
};

template
    < typename IntT
    , std::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base<10> ifmt
    , IntT value ) noexcept
{
    if (value >= 0) {
        data.uvalue = value;
        data.sign = static_cast<char>(ifmt.sign);
        data.has_prefix = ifmt.sign != strf::showsign::negative_only;
    } else {
        using uvalue_type = decltype(data.uvalue);
        STRF_IF_CONSTEXPR (sizeof(IntT) < sizeof(data.uvalue)) {
            std::make_signed_t<uvalue_type> wide_value = value;
            data.uvalue = static_cast<uvalue_type>(-wide_value);
        } else {
            data.uvalue = 1 + static_cast<uvalue_type>(-(value + 1));
        }
        data.sign = '-';
        data.has_prefix = true;
    }
}

template
    < typename UIntT
    , std::enable_if_t<std::is_unsigned<UIntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base<10>
    , UIntT uvalue ) noexcept
{
    data.sign = '\0';
    data.has_prefix = false;
    data.uvalue = uvalue;
}

template <typename IntT, int Base>
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base<Base>
    , IntT value ) noexcept
{
    data.uvalue = static_cast<decltype(data.uvalue)>(value);
}

struct punct_fmt_int_printer_data_init_result {
    unsigned sub_width;
    unsigned fillcount;
};

template <int Base>
STRF_HD punct_fmt_int_printer_data_init_result init_punct_fmt_int_printer_data
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base<Base> ifmt
    , strf::alignment_format afmt ) noexcept
#if defined(STRF_OMIT_IMPL)
    ;
#else
{
    data.digcount = strf::detail::count_digits<Base>(data.uvalue);
    data.sepsize = 1;
    if (data.sepchar >= 0x80 && data.grouping.any_separator(data.digcount)) {
        auto sepsize = validate(data.sepchar);
        data.sepsize = static_cast<unsigned>(sepsize);
        if (sepsize == strf::invalid_char_len) {
            data.grouping = strf::digits_grouping{};
            data.sepsize = 0;
        }
    }
    data.sepcount = data.grouping.separators_count(data.digcount);
    unsigned prefix_size;
    STRF_IF_CONSTEXPR (Base == 10 ) {
        prefix_size = data.has_prefix;
    } else STRF_IF_CONSTEXPR (Base == 8 ) {
        if (ifmt.showbase && data.uvalue != 0) {
            data.has_prefix = true;
            prefix_size = 1;
            if (ifmt.precision > 0) {
                -- ifmt.precision;
            }
        } else {
            data.has_prefix = false;
            prefix_size = 0;
        }
    } else {
        data.has_prefix = ifmt.showbase;
        prefix_size = (unsigned)ifmt.showbase << 1;
    }
    unsigned content_width = data.digcount + data.sepcount + prefix_size;
    unsigned zeros_a = ifmt.precision > data.digcount ? ifmt.precision - data.digcount : 0;
    unsigned zeros_b = ifmt.pad0width > content_width ? ifmt.pad0width - content_width : 0;
    data.leading_zeros = (detail::max)(zeros_a, zeros_b);
    content_width += data.leading_zeros;
    auto fmt_width = afmt.width.round();
    data.fillchar = afmt.fill;
    bool fill_sign_space = Base == 10 && data.sign == ' ';
    if (fmt_width <= (int)content_width) {
        bool x = fill_sign_space && afmt.fill != ' ';
        data.left_fillcount = x;
        data.right_fillcount = 0;
        data.has_prefix &= !x;
        return {content_width - data.sepcount - x, x};
    }
    auto fillcount = static_cast<unsigned>(fmt_width - content_width);
    data.has_prefix &= !fill_sign_space;
    switch (afmt.alignment) {
        case strf::text_alignment::left:
            data.left_fillcount = fill_sign_space;
            data.right_fillcount = fillcount;
            break;
        case strf::text_alignment::right:
            data.left_fillcount = fillcount + fill_sign_space;
            data.right_fillcount = 0;
            break;
        default:
            auto halfcount = fillcount >> 1;
            data.left_fillcount = halfcount + fill_sign_space;
            data.right_fillcount = halfcount + (fillcount & 1);
    }
    return {content_width - data.sepcount - fill_sign_space, fillcount + fill_sign_space};
}
#endif // defined(STRF_OMIT_IMPL)

template <typename CharT, int Base>
class punct_fmt_int_printer: public printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit punct_fmt_int_printer
        ( const strf::usual_printer_input<T...>& i)
        : punct_fmt_int_printer
            ( i.arg.value()
            , i.arg.get_int_format()
            , i.arg.get_alignment_format()
            , i.preview
            , i.facets )
    {
    }

    template <typename IntT, typename Preview, typename FPack>
    STRF_HD punct_fmt_int_printer
        ( IntT ivalue
        , int_format_static_base<Base> ifmt
        , alignment_format afmt
        , Preview& preview
        , const FPack& facets )
        : punct_fmt_int_printer
            ( ivalue, ifmt, afmt, preview
            , strf::get_facet<lettercase_c, IntT>(facets)
            , strf::get_facet<numpunct_c<Base>, IntT>(facets).grouping()
            , strf::get_facet<numpunct_c<Base>, IntT>(facets).thousands_sep()
            , strf::get_facet<char_encoding_c<CharT>, IntT>(facets) )
    {
    }

    template <typename IntT, typename Preview, typename Encoding>
    STRF_HD punct_fmt_int_printer
        ( IntT ivalue
        , int_format_static_base<Base> ifmt
        , alignment_format afmt
        , Preview& preview
        , strf::lettercase lc
        , strf::digits_grouping grp
        , char32_t thousands_sep
        , Encoding enc )
        : encode_fill_{enc.encode_fill_func()}
        , encode_char_{enc.encode_char_func()}
        , lettercase_{lc}
    {
        data_.sepchar = thousands_sep;
        data_.grouping = grp;
        detail::init_1(data_, ifmt, ivalue);
        const auto w = detail::init_punct_fmt_int_printer_data<Base>
            (data_, enc.validate_func(), ifmt, afmt);
        preview.subtract_width(w.sub_width + w.fillcount + data_.sepcount);
        STRF_IF_CONSTEXPR (Preview::size_required) {
            preview.add_size(w.sub_width);
            if (w.fillcount) {
                preview.add_size(w.fillcount * enc.encoded_char_size(afmt.fill));
            }
            preview.add_size(data_.sepcount * data_.sepsize);
        }
    }

    STRF_HD ~punct_fmt_int_printer();

    STRF_HD void print_to( strf::basic_outbuff<CharT>& ob ) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_;
    strf::encode_char_f<CharT> encode_char_;
    strf::detail::punct_fmt_int_printer_data data_;
    strf::lettercase lettercase_;
};

template <typename CharT, int Base>
STRF_HD punct_fmt_int_printer<CharT, Base>::~punct_fmt_int_printer()
{
}

template <typename CharT, int Base>
STRF_HD void punct_fmt_int_printer<CharT, Base>::print_to
        ( strf::basic_outbuff<CharT>& ob ) const
{
    if (data_.left_fillcount > 0) {
        encode_fill_(ob, data_.left_fillcount, data_.fillchar);
    }
    if (data_.has_prefix) {
        STRF_IF_CONSTEXPR (Base == 10) {
            ob.ensure(1);
            * ob.pointer() = static_cast<CharT>(data_.sign);
            ob.advance();
        } else STRF_IF_CONSTEXPR (Base == 8) {
            ob.ensure(1);
            * ob.pointer() = static_cast<CharT>('0');
            ob.advance();
        } else {
            constexpr CharT xb = Base == 16 ? 'X' : 'B';
            ob.ensure(2);
            auto it = ob.pointer();
            it[0] = static_cast<CharT>('0');
            it[1] = static_cast<CharT>(xb | ((lettercase_ != strf::uppercase) << 5));
            ob.advance_to(it + 2);
        }
    }
    if (data_.leading_zeros > 0) {
        strf::detail::write_fill(ob, data_.leading_zeros, CharT('0'));
    }
    using dig_writer = detail::intdigits_writer<Base>;
    if (data_.sepcount == 0) {
        dig_writer::write(ob, data_.uvalue, data_.digcount, lettercase_);
    } else if (data_.sepsize == 1) {
        CharT sepchar = static_cast<CharT>(data_.sepchar);
        if (data_.sepchar >= 0x80) {
            encode_char_(&sepchar, data_.sepchar);
        }
        dig_writer::write_little_sep
            ( ob, data_.uvalue, data_.grouping, data_.digcount, data_.sepcount
            , sepchar, lettercase_ );
    } else {
        dig_writer::write_big_sep
            ( ob, encode_char_, data_.uvalue, data_.grouping
            , data_.sepchar, data_.sepsize, data_.digcount, lettercase_ );
    }
    if (data_.right_fillcount > 0) {
        encode_fill_(ob, data_.right_fillcount, data_.fillchar);
    }
}


template <typename CharT>
class int_printer_full_dynamic
{
public:

    template <typename... T>
    STRF_HD explicit int_printer_full_dynamic
        ( const strf::usual_printer_input<T...>& i )
    {
        auto ifmt = i.arg.get_int_format();
        auto afmt = i.arg.get_alignment_format();
        switch(ifmt.base) {
            case 16: {
                int_format_static_base<16> ifmt16
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase, ifmt.punctuate };
                new ((void*)&storage_) punct_fmt_int_printer<CharT, 16>
                    ( i.arg.value(), ifmt16, afmt, i.preview, i.facets );
                break;
            }
            case 8: {
                int_format_static_base<8> ifmt8
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase, ifmt.punctuate };
                new ((void*)&storage_) punct_fmt_int_printer<CharT, 8>
                    ( i.arg.value(), ifmt8, afmt, i.preview, i.facets );
                break;
            }
            case 2: {
                int_format_static_base<2> ifmt2
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase, ifmt.punctuate };
                new ((void*)&storage_) punct_fmt_int_printer<CharT, 2>
                    ( i.arg.value(), ifmt2, afmt, i.preview, i.facets );
                break;
            }
            default:  {
                int_format_static_base<10> ifmt10
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase, ifmt.punctuate };
                new ((void*)&storage_) punct_fmt_int_printer<CharT, 10>
                    ( i.arg.value(), ifmt10, afmt, i.preview, i.facets );
                break;
            }
        }
    }

    STRF_HD ~int_printer_full_dynamic()
    {
        const strf::printer<CharT>& p = *this;
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD operator const strf::printer<CharT>& () const
    {
        return * reinterpret_cast<const strf::printer<CharT>*>(&storage_);
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::punct_fmt_int_printer<CharT, 10>);

    using storage_type_ = typename std::aligned_storage_t
        < pool_size_, alignof(strf::printer<CharT>)>;

    storage_type_ storage_;
};

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE
STRF_HD punct_fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<2>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base<2> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD punct_fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<8>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base<8> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD punct_fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<10>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base<10> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD punct_fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<16>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base<16> ifmt
    , strf::alignment_format afmt ) noexcept;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class default_int_printer<char8_t>;
//STRF_EXPLICIT_TEMPLATE class punct_int_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char8_t,  8>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char8_t, 10>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char8_t, 16>;
#endif

STRF_EXPLICIT_TEMPLATE class default_int_printer<char>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<wchar_t>;

// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char16_t>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char32_t>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char,  8>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char, 10>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char, 16>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char16_t,  8>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char16_t, 10>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char16_t, 16>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char32_t,  8>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char32_t, 10>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<char32_t, 16>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<wchar_t,  8>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<wchar_t, 10>;
STRF_EXPLICIT_TEMPLATE class punct_fmt_int_printer<wchar_t, 16>;

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
