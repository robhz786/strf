#ifndef STRF_DETAIL_PRINTABLE_TYPES_INT_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_INT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/facets/charset.hpp>
#include <strf/detail/int_digits.hpp>

// todo: optimize as in:
// https://pvk.ca/Blog/2017/12/22/appnexus-common-framework-its-out-also-how-to-print-integers-faster/

namespace strf {

struct int_format
{
#if __cplusplus < 201402L
    constexpr STRF_HD int_format
        ( int base_ = 10
        , int precision_ = 0
        , int pad0width_ = 0
        , strf::showsign sign_ = strf::showsign::negative_only
        , bool showbase_ = false
        , bool punctuate_ = false ) noexcept
        : base(base_)
        , precision(precision_)
        , pad0width(pad0width_)
        , sign(sign_)
        , showbase(showbase_)
        , punctuate(punctuate_)
    {
    }
#endif // __cplusplus < 201402L

    int base = 10;
    int precision = 0;
    int pad0width = 0;
    strf::showsign sign = strf::showsign::negative_only;
    bool showbase = false;
    bool punctuate = false;
};

using int_format_full_dynamic = int_format;

template <int Base, bool Punctuate>
struct int_format_static_base_and_punct
{

#if __cplusplus < 201402L
    constexpr STRF_HD int_format_static_base_and_punct
        ( int precision_ = 0
        , int pad0width_ = 0
        , strf::showsign sign_ = strf::showsign::negative_only
        , bool showbase_ = false ) noexcept
        : precision(precision_)
        , pad0width(pad0width_)
        , sign(sign_)
        , showbase(showbase_)
    {
    }
#endif // __cplusplus < 201402L

    constexpr static int base = Base;
    int precision = 0;
    int pad0width = 0;
    strf::showsign sign = strf::showsign::negative_only;
    bool showbase = false;
    constexpr static bool punctuate = Punctuate;

    constexpr STRF_HD operator strf::int_format_full_dynamic () const
    {
        return {base, precision, pad0width, sign, showbase, punctuate};
    }
};

template <int Base>
struct int_format_no_pad0_nor_punct
{
    constexpr static int base = Base;
    constexpr static int precision = 0;
    constexpr static int pad0width = 0;
    strf::showsign sign = strf::showsign::negative_only;
    bool showbase = false;
    constexpr static bool punctuate = false;

    constexpr STRF_HD operator int_format_static_base_and_punct<Base, false>  () const
    {
        return {0, 0, sign, showbase};
    }
    constexpr STRF_HD operator strf::int_format_full_dynamic () const
    {
        return {base, precision, pad0width, sign, showbase, punctuate};
    }
};

template <int ToBase, int FromBase>
constexpr STRF_HD int_format_no_pad0_nor_punct<ToBase>
    change_base(int_format_no_pad0_nor_punct<FromBase> f) noexcept
{
    return {f.sign, f.showbase};
}

struct default_int_format
{
    constexpr static int base = 10;
    constexpr static int precision = 0;
    constexpr static int pad0width = 0;
    constexpr static strf::showsign sign = strf::showsign::negative_only;
    constexpr static bool showbase = false;
    constexpr static bool punctuate = false;

    constexpr STRF_HD operator strf::int_format_no_pad0_nor_punct<10> () const
    {
        return {};
    }
    constexpr STRF_HD operator strf::int_format_static_base_and_punct<10, false> () const
    {
        return {};
    }
    constexpr STRF_HD operator strf::int_format_full_dynamic () const
    {
        return {};
    }
};

template <int ToBase, int ToPunctuate, int FromBase, bool FromPunctuate>
constexpr STRF_HD int_format_static_base_and_punct<ToBase, ToPunctuate>
    change_static_params(int_format_static_base_and_punct<FromBase, FromPunctuate> f) noexcept
{
    return {f.precision, f.pad0width, f.sign, f.showbase};
}

template <class T>
class default_int_formatter_fn;

template <class T, int Base>
class int_formatter_no_pad0_nor_punct_fn;

template <class T, int Base, bool Punctuate>
class int_formatter_static_base_and_punct_fn;

template <class T>
class int_formatter_full_dynamic_fn;

struct default_int_formatter
{
    template <typename T>
    using fn = strf::default_int_formatter_fn<T>;
};

template <int Base>
struct int_formatter_no_pad0_nor_punct
{
    template <typename T>
    using fn = int_formatter_no_pad0_nor_punct_fn<T, Base>;
};

template <int Base, bool Punctuate>
struct int_formatter_static_base_and_punct
{
    template <typename T>
    using fn = int_formatter_static_base_and_punct_fn<T, Base, Punctuate>;
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
    template <int Base, bool Punctuate>
    using static_base_and_punct_t_ = strf::fmt_replace
        < T, default_int_formatter
        , int_formatter_static_base_and_punct<Base, Punctuate> >;

    template <int Base>
    using no_pad0_nor_punct_t_ = strf::fmt_replace
        < T, default_int_formatter
        , int_formatter_no_pad0_nor_punct<Base> >;

public:

    constexpr default_int_formatter_fn() noexcept = default;

    template <typename U>
    constexpr STRF_HD default_int_formatter_fn(const default_int_formatter_fn<U>&) noexcept
    {
    }
    constexpr STRF_HD explicit default_int_formatter_fn(default_int_format) noexcept
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& dec() && noexcept
    {
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD const T&& dec() const && noexcept
    {
        return static_cast<const T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& dec() & noexcept
    {
        return static_cast<T&>(*this);
    }
    constexpr STRF_HD const T& dec() const & noexcept
    {
        return *static_cast<const T*>(this);
    }
    constexpr STRF_HD no_pad0_nor_punct_t_<16> hex() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<16>>{}
               , int_format_no_pad0_nor_punct<16>{} };
    }
    constexpr STRF_HD no_pad0_nor_punct_t_<8> oct() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<8>>{}
               , int_format_no_pad0_nor_punct<8>{} };
    }
    constexpr STRF_HD no_pad0_nor_punct_t_<2> bin() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<2>>{}
               , int_format_no_pad0_nor_punct<2>{} };
    }
    STRF_CONSTEXPR_IN_CXX14
    STRF_HD static_base_and_punct_t_<10, false> p(int _) const noexcept
    {
        int_format_static_base_and_punct<10, false> data;
        data.precision = _;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base_and_punct<10, false>>{}
               , data };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD static_base_and_punct_t_<10, false> pad0(int w) && noexcept
    {
        int_format_static_base_and_punct<10, false> data;
        data.pad0width = w;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base_and_punct<10, false>>{}
               , data };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_pad0_nor_punct_t_<10> operator+() && noexcept
    {
        int_format_no_pad0_nor_punct<10> data;
        data.sign = strf::showsign::positive_also;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<10>>{}
               , data };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_pad0_nor_punct_t_<10> fill_sign() && noexcept
    {
        int_format_no_pad0_nor_punct<10> data;
        data.sign = strf::showsign::fill_instead_of_positive;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<10>>{}
               , data };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_pad0_nor_punct_t_<10> operator~() && noexcept
    {
        int_format_no_pad0_nor_punct<10> data;
        data.sign = strf::showsign::fill_instead_of_positive;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<10>>{}
               , data };
    }
    STRF_HD void operator*() && noexcept = delete;
    STRF_CONSTEXPR_IN_CXX14 STRF_HD static_base_and_punct_t_<10, true> punct() && noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base_and_punct<10, true>>{}
               , strf::tag<>{} };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD static_base_and_punct_t_<10, true> operator!() && noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base_and_punct<10, true>>{}
               , strf::tag<>{} };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
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
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_int_format(strf::default_int_format) && noexcept
    {
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& set_int_format(strf::default_int_format) & noexcept
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
    constexpr STRF_HD no_pad0_nor_punct_t_<Base> set_int_format
        (const strf::int_format_no_pad0_nor_punct<Base>& data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_no_pad0_nor_punct<Base>>{}
               , data };
    }
    template <int Base, bool Punctuate>
    constexpr STRF_HD static_base_and_punct_t_<Base, Punctuate> set_int_format
        (const strf::int_format_static_base_and_punct<Base, Punctuate>& data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_static_base_and_punct<Base, Punctuate>>{}
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

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
};

template <class T, int Base>
class int_formatter_no_pad0_nor_punct_fn
{
private:

    template <int OtherBase>
    using other_base_t_ = strf::fmt_replace
        < T , int_formatter_no_pad0_nor_punct<Base>
        , int_formatter_no_pad0_nor_punct<OtherBase> >;

    template <int OtherBase, bool Punctuate>
    using static_base_and_punct_t_ = strf::fmt_replace
        < T, int_formatter_no_pad0_nor_punct<Base>
        , int_formatter_static_base_and_punct<Base, Punctuate> >;

    using full_dynamic_t_ = strf::fmt_replace
        < T, int_formatter_no_pad0_nor_punct<Base>
        , int_formatter_full_dynamic >;

public:

    constexpr int_formatter_no_pad0_nor_punct_fn()  noexcept = default;

    constexpr STRF_HD explicit int_formatter_no_pad0_nor_punct_fn(strf::tag<>) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD int_formatter_no_pad0_nor_punct_fn
        ( const int_formatter_no_pad0_nor_punct_fn<U, Base> & u ) noexcept
        : data_(u.get_int_format())
    {
    }

    constexpr STRF_HD explicit int_formatter_no_pad0_nor_punct_fn
        ( int_format_no_pad0_nor_punct<Base> data ) noexcept
        : data_(data)
    {
    }

    template < int B = 16 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 16, T&&> hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 16, other_base_t_<B>>
    hex() const &
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_no_pad0_nor_punct<B>>{}
               , strf::change_base<B>(data_) };
    }
    template < int B = 10 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 10, T&&> dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 10, other_base_t_<B>>
    dec() const &
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_no_pad0_nor_punct<B>>{}
               , strf::change_base<B>(data_) };
    }
    template < int B = 8 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 8 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 8, other_base_t_<B>>
    oct() const &
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_no_pad0_nor_punct<B>>{}
               , strf::change_base<B>(data_) };
    }
    template < int B = 2 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 2, T&&>
    bin() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 2 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 2, other_base_t_<B>>
    bin() const &
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_no_pad0_nor_punct<B>>{}
               , strf::change_base<B>(data_) };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD static_base_and_punct_t_<Base, false>
    p(int _) && noexcept
    {
        int_format_static_base_and_punct<Base, false> new_data = data_;
        new_data.precision = _;
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_static_base_and_punct<Base, false>>{}
               , new_data };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD static_base_and_punct_t_<Base, false>
    pad0(int w) const noexcept
    {
        int_format_static_base_and_punct<Base, false> new_data = data_;
        new_data.pad0width = w;
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_static_base_and_punct<Base, false>>{}
               , new_data };
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator+() && noexcept
    {
        static_assert(DecimalBase, "operator+ only allowed in decimal base");
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill_sign() && noexcept
    {
        static_assert(DecimalBase, "fill_sign() only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator~() && noexcept
    {
        static_assert(DecimalBase, "operator~ only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator*() && noexcept
    {
        static_assert(!DecimalBase, "operator* not allowed in decimal base");
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD static_base_and_punct_t_<Base, true> punct() const noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_static_base_and_punct<Base, true>>{}
               , change_static_params<Base, true>
                   (static_cast<int_format_static_base_and_punct<Base, false>>(data_)) };
    }
    constexpr STRF_HD static_base_and_punct_t_<Base, true> operator!() const noexcept
    {
        return punct();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD full_dynamic_t_ base(int b) const
    {
        int_format_full_dynamic new_data = data_;
        new_data.base = b;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_full_dynamic>{}
               , new_data };
    }
    constexpr static STRF_HD int base() noexcept
    {
        return Base;
    }
    constexpr STRF_HD strf::int_format_no_pad0_nor_punct<Base> get_int_format() const noexcept
    {
        return data_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_int_format
        ( strf::int_format_no_pad0_nor_punct<Base> data ) && noexcept
    {
        data_ = data;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& set_int_format
        ( strf::int_format_no_pad0_nor_punct<Base> data ) & noexcept
    {
        data_ = data;
        return static_cast<T&>(*this);
    }
    template <int OtherBase>
    constexpr STRF_HD strf::detail::enable_if_t< Base != OtherBase, other_base_t_<OtherBase> >
    set_int_format(strf::int_format_no_pad0_nor_punct<OtherBase> new_data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_no_pad0_nor_punct<OtherBase>>{}
               , new_data };
    }
    template <int B, bool P>
    constexpr STRF_HD static_base_and_punct_t_<B, P>
    set_int_format( int_format_static_base_and_punct<B, P> new_data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_static_base_and_punct<B, P>>{}
               , new_data };
    }
    constexpr STRF_HD strf::fmt_replace
        < T, int_formatter_no_pad0_nor_punct<Base>
        , default_int_formatter >
    set_int_format(strf::default_int_format) const & noexcept
    {
        return { * static_cast<const T*>(this) };
    }
    constexpr STRF_HD full_dynamic_t_
    set_int_format(strf::int_format_full_dynamic new_data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_full_dynamic>{}
               , new_data };
    }

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }

    strf::int_format_no_pad0_nor_punct<Base> data_;
};

template <class T, int Base, bool Punctuate>
class int_formatter_static_base_and_punct_fn
{
private:

    template <int OtherBase, bool OtherPunctuate>
    using adapted_derived_type_ = strf::fmt_replace
        < T
        , int_formatter_static_base_and_punct<Base, Punctuate>
        , int_formatter_static_base_and_punct<OtherBase, OtherPunctuate> >;

public:

    constexpr int_formatter_static_base_and_punct_fn()  noexcept = default;

    constexpr STRF_HD explicit int_formatter_static_base_and_punct_fn(strf::tag<>) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD int_formatter_static_base_and_punct_fn
        ( const int_formatter_static_base_and_punct_fn<U, Base, Punctuate> & u ) noexcept
        : data_(u.get_int_format())
    {
    }

    constexpr STRF_HD explicit int_formatter_static_base_and_punct_fn
        ( int_format_static_base_and_punct<Base, Punctuate> data ) noexcept
        : data_(data)
    {
    }

    template < int B = 16 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 16, T&&> hex() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 16 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 16, adapted_derived_type_<B, Punctuate>>
    hex() const &
    {
        return adapted_derived_type_<B, Punctuate>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<B, Punctuate>>{}
            , strf::change_static_params<B, Punctuate>(data_) };
    }
    template < int B = 10 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 10, T&&> dec() &&
    {
        return static_cast<T&&>(*this);
    }

    template < int B = 10 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 10, adapted_derived_type_<B, Punctuate>>
    dec() const &
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<B, Punctuate>>{}
            , strf::change_static_params<B, Punctuate>(data_) };
    }
    template < int B = 8 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 8, T&&>
    oct() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 8 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 8, adapted_derived_type_<B, Punctuate>>
    oct() const &
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<B, Punctuate>>{}
            , strf::change_static_params<B, Punctuate>(data_) };
    }
    template < int B = 2 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<Base == B && B == 2, T&&>
    bin() &&
    {
        return static_cast<T&&>(*this);
    }
    template < int B = 2 >
    constexpr STRF_HD
    strf::detail::enable_if_t<Base != B && B == 2, adapted_derived_type_<B, Punctuate>>
    bin() const &
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<B, Punctuate>>{}
            , strf::change_static_params<B, Punctuate>(data_) };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& p(int _) && noexcept
    {
        data_.precision = _;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& pad0(int w) && noexcept
    {
        data_.pad0width = w;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator+() && noexcept
    {
        static_assert(DecimalBase, "operator+ only allowed in decimal base");
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill_sign() && noexcept
    {
        static_assert(DecimalBase, "fill_sign() only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator~() && noexcept
    {
        static_assert(DecimalBase, "operator~ only allowed in decimal base");
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    template <bool DecimalBase = (Base == 10)>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator*() && noexcept
    {
        static_assert(!DecimalBase, "operator* not allowed in decimal base");
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    template <bool P = true>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<P == Punctuate, T&&> punct() && noexcept
    {
        return static_cast<T&&>(*this);
    }
    template <bool P = true>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::detail::enable_if_t<P == Punctuate, T&&> operator!() && noexcept
    {
        return static_cast<T&&>(*this);
    }
    template <bool P = true>
    constexpr STRF_HD strf::detail::enable_if_t<P != Punctuate, adapted_derived_type_<Base, true>>
    punct() const & noexcept
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<Base, true>>{}
            , strf::change_static_params<Base, true>(data_) };
    }
    template <bool P = true>
    constexpr STRF_HD strf::detail::enable_if_t<P != Punctuate, adapted_derived_type_<Base, true>>
    operator!() const & noexcept
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<Base, true>>{}
            , strf::change_static_params<Base, true>(data_) };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::fmt_replace
        < T
        , int_formatter_static_base_and_punct<Base, Punctuate>
        , int_formatter_full_dynamic >
    base(int b) const
    {
        int_format_full_dynamic data;
        data.base = b;
        data.punctuate = Punctuate;
        return { *static_cast<const T*>(this)
               , strf::tag<int_formatter_full_dynamic>{}
               , data };
    }
    constexpr static STRF_HD int base() noexcept
    {
        return Base;
    }
    constexpr STRF_HD
    strf::int_format_static_base_and_punct<Base, Punctuate>
    get_int_format() const noexcept
    {
        return data_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_int_format
        ( strf::int_format_static_base_and_punct<Base, Punctuate> data ) && noexcept
    {
        data_ = data;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& set_int_format
        ( strf::int_format_static_base_and_punct<Base, Punctuate> data ) & noexcept
    {
        data_ = data;
        return static_cast<T&>(*this);
    }
    template <int OtherBase, bool OtherPunctuate>
    constexpr STRF_HD
    strf::detail::enable_if_t
        < Base != OtherBase || Punctuate != OtherPunctuate
        , adapted_derived_type_<OtherBase, OtherPunctuate> >
    set_int_format
        ( strf::int_format_static_base_and_punct<OtherBase, OtherPunctuate> data ) const & noexcept
    {
        return adapted_derived_type_<OtherBase, OtherPunctuate>
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<OtherBase, OtherPunctuate>>{}
            , data };
    }
    constexpr STRF_HD strf::fmt_replace
        < T
        , int_formatter_static_base_and_punct<Base, Punctuate>
        , default_int_formatter >
    set_int_format(strf::default_int_format) const & noexcept
    {
        return { * static_cast<const T*>(this) };
    }
    constexpr STRF_HD
    strf::fmt_replace
        < T
        , int_formatter_static_base_and_punct<Base, Punctuate>
        , int_formatter_full_dynamic >
    set_int_format(strf::int_format_full_dynamic data) const & noexcept
    {
        return { *static_cast<const T*>(this)
               , strf::tag<strf::int_formatter_full_dynamic>{}
               , data };
    }

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }

    strf::int_format_static_base_and_punct<Base, Punctuate> data_;
};

template <typename T>
class int_formatter_full_dynamic_fn
{
public:

    constexpr int_formatter_full_dynamic_fn() noexcept = default;

    constexpr STRF_HD explicit int_formatter_full_dynamic_fn
        ( int_format_full_dynamic data )  noexcept
        : data_(data)
    {
    }

    template <typename U>
    constexpr STRF_HD explicit int_formatter_full_dynamic_fn
        ( const int_formatter_full_dynamic_fn<U>& other)  noexcept
        : data_(other.data_)
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& hex() &&
    {
        data_.base = 16;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& hex() &
    {
        data_.base = 16;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& dec() &&
    {
        data_.base = 10;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& dec() &
    {
        data_.base = 10;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& oct() &&
    {
        data_.base = 8;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& oct() &
    {
        data_.base = 8;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& bin() &&
    {
        data_.base = 2;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& bin() &
    {
        data_.base = 2;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& p(int _) && noexcept
    {
        data_.precision = _;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& p(int _) & noexcept
    {
        data_.precision = _;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& pad0(int w) && noexcept
    {
        data_.pad0width = w;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& pad0(int w) & noexcept
    {
        data_.pad0width = w;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator+() && noexcept
    {
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& operator+() & noexcept
    {
        data_.sign = strf::showsign::positive_also;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill_sign() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& fill_sign() & noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator~() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& operator~() & noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator*() && noexcept
    {
        data_.showbase = true;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& operator*() & noexcept
    {
        data_.showbase = true;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& punct() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& punct() & noexcept
    {
        data_.punctuate = true;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator!() && noexcept
    {
        data_.punctuate = true;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& operator!() & noexcept
    {
        data_.punctuate = true;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& base(int b) && noexcept
    {
        data_.base = b;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& base(int b) & noexcept
    {
        data_.base = b;
        return static_cast<T&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&&
    set_int_format(strf::int_format_full_dynamic data) && noexcept
    {
        data_ = data;
        return static_cast<T&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&
    set_int_format(strf::int_format_full_dynamic data) & noexcept
    {
        data_ = data;
        return static_cast<T&>(*this);
    }
    template <int Base, bool Punctuate>
    constexpr STRF_HD
    strf::fmt_replace< T
                     , int_formatter_full_dynamic
                     , int_formatter_static_base_and_punct<Base, Punctuate> >
    set_int_format(strf::int_format_static_base_and_punct<Base, Punctuate> data) const & noexcept
    {
        return
            { *static_cast<const T*>(this)
            , strf::tag<strf::int_formatter_static_base_and_punct<Base, Punctuate>>{}
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

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }

    int_format_full_dynamic data_;
};

namespace detail {

template <typename> class default_int_printer;
template <typename> class aligned_default_int_printer;
template <typename, int Base> class int_printer_no_pad0_nor_punct;
template <typename, int Base, bool Punctuate> class int_printer_static_base_and_punct;
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

// template <typename FPack, typename IntT, int Base>
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
//             ( use_facet< strf::numpunct_c<Base>, IntT >(fp())) );
// public:
//
//     static constexpr bool value = has_numpunct_type::value;
// };
//
// template <typename FPack, typename IntT, int Base>
// constexpr STRF_HD bool has_intpunct()
// {
//     return has_intpunct_impl<FPack, IntT, Base>::value;
// }

template <typename IntT>
struct int_printing;

template <typename CharT, typename Pre, typename IntT>
struct default_int_printer_input
{
    using printer_type = strf::detail::default_int_printer<CharT>;

    template<typename FPack>
    constexpr STRF_HD default_int_printer_input
        ( Pre* pre_, const FPack&, IntT arg_) noexcept
        : pre(pre_)
        , value(arg_)
    {
    }

    template<typename FPack, typename PTraits>
    constexpr STRF_HD default_int_printer_input
        ( Pre* pre_
        , const FPack&
        , strf::printable_with_fmt
            < PTraits
            , strf::int_formatter
            , strf::alignment_formatter > arg_ ) noexcept
        : pre(pre_)
        , value(arg_.value())
    {
    }

    Pre* pre;
    IntT value;
};

template <typename IntT>
struct int_printing
{
private:

    template <typename P, bool HasAlignment>
    using vwf_ = printable_with_fmt
              < P, int_formatter
              , alignment_formatter_q<HasAlignment> >;

    template <typename P, int Base, bool HasAlignment>
    using vwf_nopp_ = printable_with_fmt
              < P, int_formatter_no_pad0_nor_punct<Base>
              , alignment_formatter_q<HasAlignment> >;

    template <typename P, int Base, bool Punctuate, bool HasAlignment>
    using vwf_bp_ = printable_with_fmt
              < P, int_formatter_static_base_and_punct<Base, Punctuate>
              , alignment_formatter_q<HasAlignment> >;

    template <typename P, bool HasAlignment>
    using vwf_full_dynamic_ = printable_with_fmt
              < P, int_formatter_full_dynamic
              , alignment_formatter_q<HasAlignment> >;
public:

    using representative_type = IntT;
    using forwarded_type = IntT;
    using formatters = strf::tag< strf::int_formatter
                                , strf::alignment_formatter >;
    using is_overridable = std::true_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , IntT x ) noexcept
        -> strf::detail::default_int_printer_input<CharT, PreMeasurements, IntT>
    {
        return {pre, facets, x};
    }

    template < typename CharT, typename PreMeasurements, typename FPack
             , typename PTraits, int Base, bool HasAlignment >
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , vwf_nopp_<PTraits, Base, HasAlignment> x ) noexcept
        -> strf::usual_printer_input
            < CharT, PreMeasurements, FPack, vwf_nopp_<PTraits, Base, HasAlignment>
            , strf::detail::conditional_t
                < HasAlignment
                , strf::detail::int_printer_static_base_and_punct<CharT, Base, false>
                , strf::detail::int_printer_no_pad0_nor_punct<CharT, Base> > >
    {
        return {pre, facets, x};
    }

    template < typename CharT, typename PreMeasurements, typename FPack
             , typename PTraits, int Base, bool Punctuate, bool HasAlignment >
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , vwf_bp_<PTraits, Base, Punctuate, HasAlignment> x )
        -> strf::usual_printer_input
            < CharT, PreMeasurements, FPack, vwf_bp_<PTraits, Base, Punctuate, HasAlignment>
            , strf::detail::int_printer_static_base_and_punct<CharT, Base, Punctuate> >
    {
        return {pre, facets, x};
    }

    template < typename CharT, typename PreMeasurements, typename FPack
             , typename PTraits, bool HasAlignment>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , vwf_<PTraits, HasAlignment> x )
        -> strf::detail::conditional_t
            < ! HasAlignment
            , strf::detail::default_int_printer_input<CharT, PreMeasurements, IntT>
            , strf::usual_printer_input
                < CharT, PreMeasurements, FPack, vwf_<PTraits, HasAlignment>
                , strf::detail::aligned_default_int_printer<CharT> > >
    {
        return {pre, facets, x};
    }

    template < typename CharT, typename PreMeasurements, typename FPack
             , typename PTraits, bool HasAlignment>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , vwf_full_dynamic_<PTraits, HasAlignment> x )
        -> strf::usual_printer_input
                < CharT, PreMeasurements, FPack, vwf_full_dynamic_<PTraits, HasAlignment>
                , strf::detail::int_printer_full_dynamic<CharT> >
    {
        return {pre, facets, x};
    }
};

} // namespace detail

template <> struct printable_traits<signed char>:
    public strf::detail::int_printing<signed char> {};
template <> struct printable_traits<short>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<short> {}; // NOLINT(google-runtime-int)
template <> struct printable_traits<int>:
    public strf::detail::int_printing<int> {};
template <> struct printable_traits<long>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<long> {}; // NOLINT(google-runtime-int)
template <> struct printable_traits<long long>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<long long> {}; // NOLINT(google-runtime-int)

template <> struct printable_traits<unsigned char>:
    public strf::detail::int_printing<unsigned char> {};
template <> struct printable_traits<unsigned short>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<unsigned short> {}; // NOLINT(google-runtime-int)
template <> struct printable_traits<unsigned int>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<unsigned int> {}; // NOLINT(google-runtime-int)
template <> struct printable_traits<unsigned long>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<unsigned long> {}; // NOLINT(google-runtime-int)
template <> struct printable_traits<unsigned long long>: // NOLINT(google-runtime-int)
    public strf::detail::int_printing<unsigned long long> {}; // NOLINT(google-runtime-int)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, signed char) noexcept
    -> strf::detail::int_printing<signed char>
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, short) noexcept
    -> strf::detail::int_printing<short>  // NOLINT(google-runtime-int)
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, int) noexcept
    -> strf::detail::int_printing<int>
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, long) noexcept
    -> strf::detail::int_printing<long> // NOLINT(google-runtime-int)
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, long long) noexcept
    -> strf::detail::int_printing<long long> // NOLINT(google-runtime-int)
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, unsigned char) noexcept
    -> strf::detail::int_printing<unsigned char>
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, unsigned short) noexcept
    -> strf::detail::int_printing<unsigned short> // NOLINT(google-runtime-int)
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, unsigned int) noexcept
    -> strf::detail::int_printing<unsigned int> // NOLINT(google-runtime-int)
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, unsigned long) noexcept
    -> strf::detail::int_printing<unsigned long> // NOLINT(google-runtime-int)
    { return {}; }

// NOLINTNEXTLINE(google-runtime-int)
constexpr STRF_HD auto tag_invoke(strf::printable_tag, unsigned long long) noexcept
    -> strf::detail::int_printing<unsigned long long> // NOLINT(google-runtime-int)
    { return {}; }

namespace detail {

struct voidptr_printing
{
    using representative_type = const void*;
    using forwarded_type = const void*;
    using formatters = strf::tag<strf::alignment_formatter>;
    using is_overridable = std::true_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , const void* x ) noexcept
    -> decltype( strf::make_default_printer<CharT>
                   ( pre
                   , strf::pack
                       ( strf::use_facet<strf::numpunct_c<16>, const void*>(facets)
                       , strf::lettercase::lower
                       , strf::use_facet<strf::charset_c<CharT>, const void*>(facets) )
                   , *strf::hex(strf::detail::bit_cast<std::size_t>(x)) ) )
    {
        return strf::make_default_printer<CharT>
            ( pre
            , strf::pack
                ( strf::use_facet<strf::numpunct_c<16>, const void*>(facets)
                , strf::use_facet<strf::lettercase_c, const void*>(facets)
                , strf::use_facet<strf::charset_c<CharT>, const void*>(facets) )
            , *strf::hex(strf::detail::bit_cast<std::size_t>(x)) );
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , strf::printable_with_fmt<T...> x ) noexcept
    -> decltype( strf::make_default_printer<CharT>
                   ( pre
                   , strf::pack
                       ( strf::use_facet<strf::numpunct_c<16>, const void*>(facets)
                       , strf::use_facet<strf::lettercase_c, const void*>(facets)
                       , strf::use_facet<strf::charset_c<CharT>, const void*>(facets) )
                   , *strf::hex(strf::detail::bit_cast<std::size_t>(x.value()))
                                   .set_alignment_format(x.get_alignment_format()) ) )
    {
        return strf::make_default_printer<CharT>
            ( pre
            , strf::pack
                ( strf::use_facet<strf::numpunct_c<16>, const void*>(facets)
                , strf::use_facet<strf::lettercase_c, const void*>(facets)
                , strf::use_facet<strf::charset_c<CharT>, const void*>(facets) )
            , *strf::hex(strf::detail::bit_cast<std::size_t>(x.value()))
                            .set_alignment_format(x.get_alignment_format()) );
    }
};

} // namespace detail

constexpr STRF_HD auto tag_invoke(strf::printable_tag, const void*) noexcept
    -> strf::detail::voidptr_printing
    { return {}; }

namespace detail {

template <typename CharT>
class default_int_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit default_int_printer(strf::detail::default_int_printer_input<T...> i)
    {
        init_(i.pre, i.value);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template < typename PreMeasurements
             , typename IntT
             , strf::detail::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
    STRF_HD void init_(PreMeasurements* pre, IntT value)
    {
        using uint = typename std::make_unsigned<IntT>::type;
        uint uvalue;
        if (value >= 0) {
            negative_ = 0;
            uvalue = static_cast<uint>(value);
        } else {
            negative_ = 1;
            uvalue = unsigned_abs(value);
        }
        uvalue_ = uvalue;
        digcount_ = strf::detail::count_digits<10>(uvalue);
        pre->subtract_width(static_cast<width_t>(digcount_ + negative_));
        pre->add_size(digcount_ + negative_);
    }

   template < typename PreMeasurements
            , typename UIntT
            , strf::detail::enable_if_t< ! std::is_signed<UIntT>::value, int> = 0 >
    STRF_HD void init_(PreMeasurements* pre, UIntT value)
    {
        uvalue_ = value;
        negative_ = false;
        digcount_ = strf::detail::count_digits<10>(value);
        pre->subtract_width(digcount_);
        pre->add_size(digcount_);
    }

    std::uint64_t uvalue_{};
    int digcount_{};
    bool negative_{};
};

template <typename CharT>
STRF_HD void default_int_printer<CharT>::print_to
    ( strf::destination<CharT>& dst ) const
{
    dst.ensure(digcount_ + negative_);
    auto* it = dst.buffer_ptr();
    if (negative_) {
        *it++ = '-';
    }
    it += digcount_;
    write_int_dec_txtdigits_backwards(uvalue_, it);
    dst.advance_to(it);
}

template <typename CharT>
class aligned_default_int_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit aligned_default_int_printer(strf::usual_printer_input<T...> i)
    {
        init_(i.arg.value());
        init_fill_(i.arg.get_alignment_format());

        auto encoding = strf::use_facet<strf::charset_c<CharT>, int>(i.facets);
        STRF_MAYBE_UNUSED(encoding);
        encode_fill_ = encoding.encode_fill_func();
        i.pre->subtract_width(fillcount_ + digcount_ + negative_);
        STRF_IF_CONSTEXPR(strf::usual_printer_input<T...>::premeasurements_type::size_demanded) {
            auto fillsize = fillcount_ * encoding.encoded_char_size(fillchar_);
            i.pre->add_size(fillsize + digcount_ + negative_);
        }
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template < typename IntT
             , strf::detail::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
    STRF_HD int init_(IntT value) noexcept
    {
        using uint = typename std::make_unsigned<IntT>::type;
        uint uvalue;
        if (value >= 0) {
            negative_ = 0;
            uvalue = static_cast<uint>(value);
        } else {
            negative_ = 1;
            uvalue = strf::detail::unsigned_abs(value);
        }
        uvalue_ = uvalue;
        digcount_ = strf::detail::count_digits<10>(uvalue);
        return digcount_ + negative_;
    }

    template < typename UIntT
             , strf::detail::enable_if_t< ! std::is_signed<UIntT>::value, int> = 0 >
    STRF_HD int init_(UIntT value) noexcept
    {
        uvalue_ = value;
        negative_ = false;
        digcount_ = strf::detail::count_digits<10>(value);
        return digcount_;
    }

    STRF_HD void init_fill_(strf::alignment_format afmt)
    {
        fillchar_ = afmt.fill;
        alignment_ = afmt.alignment;
        auto sub_width = digcount_ + negative_;
        auto width = afmt.width.round();
        fillcount_ = width > sub_width ? width - sub_width : 0;
    }

    strf::encode_fill_f<CharT> encode_fill_;
    std::uint64_t uvalue_{};
    int digcount_{};
    int fillcount_{};
    strf::text_alignment alignment_{strf::text_alignment::right};
    char32_t fillchar_{};
    bool negative_{};
};

template <typename CharT>
STRF_HD void aligned_default_int_printer<CharT>::print_to
    ( strf::destination<CharT>& dst ) const
{
    int right_fillcount = 0;
    if (fillcount_ > 0) {
        int left_fillcount = 0;
        switch(alignment_) {
            case strf::text_alignment::left:
                right_fillcount = fillcount_;
                goto print_number;
            case strf::text_alignment::right:
                left_fillcount = fillcount_;
                break;
            default:
                left_fillcount = static_cast<unsigned>(fillcount_) >> 1;
                right_fillcount = fillcount_ - left_fillcount;
        }
        encode_fill_(dst, left_fillcount, fillchar_);
    }
    print_number:
    dst.ensure(digcount_ + negative_);
    auto* it = dst.buffer_ptr();
    if (negative_) {
        *it++ = '-';
    }
    it += digcount_;
    write_int_dec_txtdigits_backwards(uvalue_, it);
    dst.advance_to(it);
    if (right_fillcount > 0) {
        encode_fill_(dst, right_fillcount, fillchar_);
    }
}

struct int_printer_no_pad0_nor_punct_data
{
    std::uint64_t uvalue;
    int digcount;
    int prefix;
};

template
    < typename UIntT
    , strf::detail::enable_if_t<std::is_unsigned<UIntT>::value, int> = 0 >
inline STRF_HD int init
    ( int_printer_no_pad0_nor_punct_data& data
    , int_format_no_pad0_nor_punct<10>
    , UIntT uvalue ) noexcept
{
    data.uvalue = uvalue;
    data.digcount = strf::detail::count_digits<10>(uvalue);
    data.prefix = 0;
    return data.digcount;
}

template < typename IntT
         , strf::detail::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
inline STRF_HD int init
    ( int_printer_no_pad0_nor_punct_data& data
    , int_format_no_pad0_nor_punct<10> ifmt
    , IntT value ) noexcept
{
    using unsigned_IntT = typename std::make_unsigned<IntT>::type;
    unsigned_IntT uvalue;
    if (value >= 0) {
        uvalue = detail::cast_unsigned(value);
        data.prefix = static_cast<int>(ifmt.sign);
    } else {
        uvalue = 1 + static_cast<unsigned_IntT>(-(value + 1));
        data.prefix = '-';
    }
    data.uvalue = uvalue;
    data.digcount = strf::detail::count_digits<10>(uvalue);
    return data.digcount + (data.prefix != 0);
}

template < int Base
         , typename IntT
         , strf::detail::enable_if_t<Base != 10, int> = 0 >
inline STRF_HD int init
    ( int_printer_no_pad0_nor_punct_data& data
    , int_format_no_pad0_nor_punct<Base> ifmt
    , IntT value ) noexcept
{
    STRF_IF_CONSTEXPR (Base == 8) {
        data.prefix = ifmt.showbase && value != 0;
    } else {
        data.prefix = ifmt.showbase << 1;
    }
    auto uvalue = static_cast<typename std::make_unsigned<IntT>::type>(value);
    data.uvalue = uvalue;
    data.digcount = strf::detail::count_digits<Base>(uvalue);
    return data.digcount + data.prefix;
}

template <typename CharT>
class int_printer_no_pad0_nor_punct<CharT, 10>: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD explicit int_printer_no_pad0_nor_punct(strf::usual_printer_input<T...> i) noexcept
    {
        auto w = strf::detail::init(data_, i.arg.get_int_format(), i.arg.value());
        i.pre->subtract_width(w);
        i.pre->add_size(w);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        dst.ensure(data_.digcount + data_.prefix != 0);
        auto *it = dst.buffer_ptr();
        if (data_.prefix != 0) {
            *it++ = static_cast<CharT>(data_.prefix);
        }
        it += data_.digcount;
        strf::detail::write_int_dec_txtdigits_backwards(data_.uvalue, it);
        dst.advance_to(it);
    }

private:
    int_printer_no_pad0_nor_punct_data data_{};
};

template <typename CharT>
class int_printer_no_pad0_nor_punct<CharT, 16>: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit int_printer_no_pad0_nor_punct(strf::usual_printer_input<T...> i) noexcept
    {
        auto value = i.arg.value();
        auto w = strf::detail::init(data_, i.arg.get_int_format(), value);
        lettercase_ = strf::use_facet<strf::lettercase_c, decltype(value)>(i.facets);
        i.pre->subtract_width(static_cast<strf::width_t>(w));
        i.pre->add_size(w);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        dst.ensure(data_.digcount + data_.prefix);
        auto *it = dst.buffer_ptr();
        if (data_.prefix != 0) {
            it[0] = static_cast<CharT>('0');
            it[1] = static_cast<CharT>('X' | ((lettercase_ != strf::uppercase) << 5));
            it += 2;
        }
        it += data_.digcount;
        strf::detail::write_int_hex_txtdigits_backwards(data_.uvalue, it, lettercase_);
        dst.advance_to(it);
    }

private:

    int_printer_no_pad0_nor_punct_data data_{};
    strf::lettercase lettercase_;
};


template <typename CharT>
class int_printer_no_pad0_nor_punct<CharT, 8>: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD explicit int_printer_no_pad0_nor_punct(strf::usual_printer_input<T...> i) noexcept
    {
        auto w = strf::detail::init(data_, i.arg.get_int_format(), i.arg.value());
        i.pre->subtract_width(w);
        i.pre->add_size(w);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        dst.ensure(data_.digcount + data_.prefix);
        auto *it = dst.buffer_ptr();
        if (data_.prefix != 0) {
            *it++ = '0';
        }
        it += data_.digcount;
        strf::detail::write_int_oct_txtdigits_backwards(data_.uvalue, it);
        dst.advance_to(it);
    }

private:
    int_printer_no_pad0_nor_punct_data data_{};
};

template <typename CharT>
class int_printer_no_pad0_nor_punct<CharT, 2>: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit int_printer_no_pad0_nor_punct(strf::usual_printer_input<T...> i) noexcept
    {
        auto value = i.arg.value();
        auto w = strf::detail::init(data_, i.arg.get_int_format(), value);
        lettercase_ = strf::use_facet<strf::lettercase_c, decltype(value)>(i.facets);
        i.pre->subtract_width(w);
        i.pre->add_size(w);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        if (data_.prefix != 0) {
            dst.ensure(2);
            auto *it = dst.buffer_ptr();
            it[0] = static_cast<CharT>('0');
            it[1] = static_cast<CharT>('B' | ((lettercase_ != strf::uppercase) << 5));
            dst.advance_to(it + 2);
        }
        strf::detail::intdigits_writer<2>::write(dst, data_.uvalue, data_.digcount);
    }

private:
    int_printer_no_pad0_nor_punct_data data_{};
    strf::lettercase lettercase_;
};

struct fmt_int_printer_data {
    std::uint64_t uvalue = 0;
    int digcount = 0;
    int leading_zeros = 0;
    int left_fillcount = 0;
    int right_fillcount = 0;
    char32_t fillchar{U' '};
    bool has_prefix{false};
    char sign{'\0'};
};

struct punct_fmt_int_printer_data: public fmt_int_printer_data {
    int sepcount = 0;
    int sepsize = 0;
    char32_t sepchar{U'\0'};
    strf::digits_grouping grouping;
};

template
    < typename IntT
    , strf::detail::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::default_int_format
    , IntT value ) noexcept
{
    data.sign = '-';
    if (value >= 0) {
       data.uvalue = value;
       data.has_prefix = false;
    } else {
        using uvalue_type = decltype(data.uvalue);
        STRF_IF_CONSTEXPR (sizeof(IntT) < sizeof(data.uvalue)) {
            strf::detail::make_signed_t<uvalue_type> wide_value = value;
            data.uvalue = static_cast<uvalue_type>(-wide_value);
        } else {
            data.uvalue = 1 + static_cast<uvalue_type>(-(value + 1));
        }
        data.has_prefix = true;
    }
}

template
    < typename UIntT
    , strf::detail::enable_if_t<std::is_unsigned<UIntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::default_int_format
    , UIntT uvalue ) noexcept
{
    data.sign = '\0';
    data.has_prefix = false;
    data.uvalue = uvalue;
}

template
    < typename IntT
    , bool Punctuate
    , strf::detail::enable_if_t<std::is_signed<IntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<10, Punctuate> ifmt
    , IntT value ) noexcept
{
    if (value >= 0) {
        data.uvalue = detail::cast_unsigned(value);
        data.sign = static_cast<char>(ifmt.sign);
        data.has_prefix = ifmt.sign != strf::showsign::negative_only;
    } else {
        using uvalue_type = decltype(data.uvalue);
        STRF_IF_CONSTEXPR (sizeof(IntT) < sizeof(data.uvalue)) {
            const strf::detail::make_signed_t<uvalue_type> wide_value = value;
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
    , bool Punctuate
    , strf::detail::enable_if_t<std::is_unsigned<UIntT>::value, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<10, Punctuate>
    , UIntT uvalue ) noexcept
{
    data.sign = '\0';
    data.has_prefix = false;
    data.uvalue = uvalue;
}

template < typename IntT, bool Punctuate, int Base
         , strf::detail::enable_if_t<Base != 10, int> = 0 >
inline STRF_HD void init_1
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<Base, Punctuate>
    , IntT value ) noexcept
{
    data.sign = '\0';
    data.uvalue = static_cast<typename std::make_unsigned<IntT>::type>(value);
}

struct fmt_int_printer_data_init_result {
    int sub_width;
    int fillcount;
};

template <int Base>
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<Base, false> ifmt
    , strf::default_alignment_format ) noexcept
{
    data.digcount = strf::detail::count_digits<Base>(data.uvalue);
    int prefix_size = 0;
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
        prefix_size = (int)((unsigned)ifmt.showbase << 1);
    }
    int content_width = data.digcount + prefix_size;
    const int zeros_a = ifmt.precision > data.digcount ? ifmt.precision - data.digcount : 0;
    const int zeros_b = ifmt.pad0width > content_width ? ifmt.pad0width - content_width : 0;
    data.leading_zeros = (detail::max)(zeros_a, zeros_b);
    content_width += data.leading_zeros;
    data.fillchar = ' ';
    data.left_fillcount = 0;
    data.right_fillcount = 0;
    return {content_width, 0};
}

template <int Base>
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<Base, false> ifmt
    , strf::alignment_format afmt ) noexcept
#if defined(STRF_OMIT_IMPL)
    ;
#else
{
    data.digcount = strf::detail::count_digits<Base>(data.uvalue);
    int prefix_size = 0;
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
        prefix_size = (int)((unsigned)ifmt.showbase << 1);
    }
    int content_width = data.digcount + prefix_size;
    const int zeros_a = ifmt.precision > data.digcount ? ifmt.precision - data.digcount : 0;
    const int zeros_b = ifmt.pad0width > content_width ? ifmt.pad0width - content_width : 0;
    data.leading_zeros = (detail::max)(zeros_a, zeros_b);
    content_width += data.leading_zeros;
    auto fmt_width = afmt.width.round();
    data.fillchar = afmt.fill;
    const bool fill_sign_space = Base == 10 && data.sign == ' ';
    if (fmt_width <= content_width) {
        const bool x = fill_sign_space && afmt.fill != ' ';
        data.left_fillcount = x;
        data.right_fillcount = 0;
        data.has_prefix &= !x;
        return {content_width - x, static_cast<int>(x)};
    }
    auto fillcount = fmt_width - content_width;
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
    return {content_width - fill_sign_space, fillcount + fill_sign_space};
}

#endif // defined(STRF_OMIT_IMPL)

template <typename CharT, int Base>
class int_printer_static_base_and_punct<CharT, Base, false>: public printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit int_printer_static_base_and_punct
        ( const strf::usual_printer_input<T...>& i )
        : lettercase_(strf::use_facet<lettercase_c, detail::remove_cvref_t<decltype(i.arg.value())> >(i.facets))
    {
        auto ivalue = i.arg.value();
        using int_type = decltype(ivalue);
        auto charset = strf::use_facet<charset_c<CharT>, int_type>(i.facets);
        encode_fill_ = charset.encode_fill_func();
        const int_format_static_base_and_punct<Base, false> ifmt = i.arg.get_int_format();
        auto afmt = i.arg.get_alignment_format();
        detail::init_1(data_, ifmt, ivalue);
        auto w = detail::init_fmt_int_printer_data<Base>(data_, ifmt, afmt);
        i.pre->subtract_width(w.sub_width + w.fillcount);
        using pre_t = typename strf::usual_printer_input<T...>::premeasurements_type;
        STRF_IF_CONSTEXPR (pre_t::size_demanded) {
            i.pre->add_size(w.sub_width);
            if (w.fillcount > 0) {
                i.pre->add_size(w.fillcount * charset.encoded_char_size(afmt.fill));
            }
        }
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_ = nullptr;
    strf::detail::fmt_int_printer_data data_{};
    strf::lettercase lettercase_;
};

template <typename CharT, int Base>
STRF_HD void int_printer_static_base_and_punct<CharT, Base, false>::print_to
    ( strf::destination<CharT>& dst ) const
{
    if (data_.left_fillcount > 0) {
        encode_fill_(dst, data_.left_fillcount, data_.fillchar);
    }
    if (data_.has_prefix) {
        STRF_IF_CONSTEXPR (Base == 10) {
            dst.ensure(1);
            * dst.buffer_ptr() = static_cast<CharT>(data_.sign);
            dst.advance();
        } else STRF_IF_CONSTEXPR (Base == 8) {
            dst.ensure(1);
            * dst.buffer_ptr() = static_cast<CharT>('0');
            dst.advance();
        } else {
            constexpr int xb = static_cast<unsigned char>(Base == 16 ? 'X' : 'B');
            dst.ensure(2);
            auto *it = dst.buffer_ptr();
            it[0] = static_cast<CharT>('0');
            it[1] = static_cast<CharT>(xb | ((lettercase_ != strf::uppercase) << 5));
            dst.advance_to(it + 2);
        }
    }
    strf::detail::write_fill(dst, data_.leading_zeros, CharT('0'));
    detail::intdigits_writer<Base>::write(dst, data_.uvalue, data_.digcount, lettercase_);
    if (data_.right_fillcount > 0) {
        encode_fill_(dst, data_.right_fillcount, data_.fillchar);
    }
}

template <int Base>
STRF_HD fmt_int_printer_data_init_result init_punct_fmt_int_printer_data
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base_and_punct<Base, true> ifmt
    , strf::alignment_format afmt ) noexcept
#if defined(STRF_OMIT_IMPL)
    ;
#else
{
    data.digcount = strf::detail::count_digits<Base>(data.uvalue);
    data.sepsize = 1;
    data.sepcount = data.grouping.separators_count(data.digcount);
    if (data.sepchar >= 0x80 && data.sepcount) {
        auto sepsize = validate(data.sepchar);
        if (sepsize == strf::invalid_char_len) {
            data.grouping = strf::digits_grouping{};
            data.sepsize = 0;
        } else {
            data.sepsize = static_cast<int>(sepsize);
        }
    }
    int prefix_size = 0;
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
        prefix_size = (int)ifmt.showbase << 1;
    }
    int content_width = data.digcount + data.sepcount + prefix_size;
    const int zeros_a = ifmt.precision > data.digcount ? ifmt.precision - data.digcount : 0;
    const int zeros_b = ifmt.pad0width > content_width ? ifmt.pad0width - content_width : 0;
    data.leading_zeros = (detail::max)(zeros_a, zeros_b);
    content_width += data.leading_zeros;
    int const fmt_width = afmt.width.round();
    data.fillchar = afmt.fill;
    const bool fill_sign_space = Base == 10 && data.sign == ' ';
    if (fmt_width <= content_width) {
        const bool x = fill_sign_space && afmt.fill != ' ';
        data.left_fillcount = x;
        data.right_fillcount = 0;
        data.has_prefix &= !x;
        return {content_width - data.sepcount - x, static_cast<int>(x)};
    }
    auto fillcount = fmt_width - content_width;
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
class int_printer_static_base_and_punct<CharT, Base, true>: public printer<CharT>
{
public:

    template <typename... T>
    STRF_HD explicit int_printer_static_base_and_punct
        ( const strf::usual_printer_input<T...>& i)
        : int_printer_static_base_and_punct
            ( i.arg.value()
            , i.arg.get_int_format()
            , i.arg.get_alignment_format()
            , i.pre
            , i.facets )
    {
    }

    template <typename IntT, typename PreMeasurements, typename FPack>
    STRF_HD int_printer_static_base_and_punct
        ( IntT ivalue
        , int_format_static_base_and_punct<Base, true> ifmt
        , alignment_format afmt
        , PreMeasurements* pre
        , const FPack& facets )
        : int_printer_static_base_and_punct
            ( ivalue, ifmt, afmt, pre
            , strf::use_facet<lettercase_c, IntT>(facets)
            , strf::use_facet<numpunct_c<Base>, IntT>(facets).grouping()
            , strf::use_facet<numpunct_c<Base>, IntT>(facets).thousands_sep()
            , strf::use_facet<charset_c<CharT>, IntT>(facets) )
    {
    }

    template <typename IntT, typename PreMeasurements, typename Charset>
    STRF_HD int_printer_static_base_and_punct
        ( IntT ivalue
        , int_format_static_base_and_punct<Base, true> ifmt
        , alignment_format afmt
        , PreMeasurements* pre
        , strf::lettercase lc
        , strf::digits_grouping grp
        , char32_t thousands_sep
        , Charset charset )
        : encode_fill_{charset.encode_fill_func()}
        , encode_char_{charset.encode_char_func()}
        , lettercase_{lc}
    {
        data_.sepchar = thousands_sep;
        data_.grouping = grp;
        detail::init_1(data_, ifmt, ivalue);
        const auto w = detail::init_punct_fmt_int_printer_data<Base>
            (data_, charset.validate_func(), ifmt, afmt);
        pre->subtract_width(w.sub_width + w.fillcount + data_.sepcount);
        STRF_IF_CONSTEXPR (PreMeasurements::size_demanded) {
            pre->add_size(w.sub_width);
            if (w.fillcount) {
                pre->add_size(w.fillcount * charset.encoded_char_size(afmt.fill));
            }
            pre->add_size(data_.sepcount * data_.sepsize);
        }
    }

    STRF_HD void print_to( strf::destination<CharT>& dst ) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_;
    strf::encode_char_f<CharT> encode_char_;
    strf::detail::punct_fmt_int_printer_data data_;
    strf::lettercase lettercase_{strf::lettercase::lower};
};

template <typename CharT, int Base>
STRF_HD void int_printer_static_base_and_punct<CharT, Base, true>::print_to
        ( strf::destination<CharT>& dst ) const
{
    if (data_.left_fillcount > 0) {
        encode_fill_(dst, data_.left_fillcount, data_.fillchar);
    }
    if (data_.has_prefix) {
        STRF_IF_CONSTEXPR (Base == 10) {
            dst.ensure(1);
            * dst.buffer_ptr() = static_cast<CharT>(data_.sign);
            dst.advance();
        } else STRF_IF_CONSTEXPR (Base == 8) {
            dst.ensure(1);
            * dst.buffer_ptr() = static_cast<CharT>('0');
            dst.advance();
        } else {
            constexpr int xb = static_cast<unsigned char>(Base == 16 ? 'X' : 'B');
            dst.ensure(2);
            auto *it = dst.buffer_ptr();
            it[0] = static_cast<CharT>('0');
            it[1] = static_cast<CharT>(xb | ((lettercase_ != strf::uppercase) << 5));
            dst.advance_to(it + 2);
        }
    }
    strf::detail::write_fill(dst, data_.leading_zeros, CharT('0'));
    using dig_writer = detail::intdigits_writer<Base>;
    if (data_.sepcount == 0) {
        dig_writer::write(dst, data_.uvalue, data_.digcount, lettercase_);
    } else if (data_.sepsize == 1) {
        auto sepchar = static_cast<CharT>(data_.sepchar);
        if (data_.sepchar >= 0x80) {
            encode_char_(&sepchar, data_.sepchar);
        }
        dig_writer::write_little_sep
            ( dst, data_.uvalue, data_.grouping, data_.digcount, data_.sepcount
            , sepchar, lettercase_ );
    } else {
        dig_writer::write_big_sep
            ( dst, encode_char_, data_.uvalue, data_.grouping
            , data_.sepchar, data_.sepsize, data_.digcount, lettercase_ );
    }
    if (data_.right_fillcount > 0) {
        encode_fill_(dst, data_.right_fillcount, data_.fillchar);
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
        auto ivalue = i.arg.value();
        using int_type = decltype(ivalue);
        auto lc = strf::use_facet<lettercase_c, int_type>(i.facets);
        auto charset = strf::use_facet<charset_c<CharT>, int_type>(i.facets);
        strf::digits_grouping grp;
        char32_t thousands_sep = ',';
        switch(ifmt.base) {
            case 16: {
                const int_format_static_base_and_punct<16, true> ifmt16
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase };
                if (ifmt.punctuate) {
                    auto numpunct = strf::use_facet<numpunct_c<16>, int_type>(i.facets);
                    grp = numpunct.grouping();
                    thousands_sep = numpunct.thousands_sep();
                }
                new ((void*)&storage_) int_printer_static_base_and_punct<CharT, 16, true>
                    ( ivalue, ifmt16, afmt, i.pre, lc, grp, thousands_sep, charset );
                break;
            }
            case 8: {
                const int_format_static_base_and_punct<8, true> ifmt8
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase };
                if (ifmt.punctuate) {
                    auto numpunct = strf::use_facet<numpunct_c<8>, int_type>(i.facets);
                    grp = numpunct.grouping();
                    thousands_sep = numpunct.thousands_sep();
                }
                new ((void*)&storage_) int_printer_static_base_and_punct<CharT, 8, true>
                    ( ivalue, ifmt8, afmt, i.pre, lc, grp, thousands_sep, charset );
                break;
            }
            case 2: {
                const int_format_static_base_and_punct<2, true> ifmt2
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase };
                if (ifmt.punctuate) {
                    auto numpunct = strf::use_facet<numpunct_c<2>, int_type>(i.facets);
                    grp = numpunct.grouping();
                    thousands_sep = numpunct.thousands_sep();
                }
                new ((void*)&storage_) int_printer_static_base_and_punct<CharT, 2, true>
                    ( ivalue, ifmt2, afmt, i.pre, lc, grp, thousands_sep, charset );
                break;
            }
            default:  {
                const int_format_static_base_and_punct<10, true> ifmt10
                    { ifmt.precision, ifmt.pad0width, ifmt.sign, ifmt.showbase };
                if (ifmt.punctuate) {
                    auto numpunct = strf::use_facet<numpunct_c<10>, int_type>(i.facets);
                    grp = numpunct.grouping();
                    thousands_sep = numpunct.thousands_sep();
                }
                new ((void*)&storage_) int_printer_static_base_and_punct<CharT, 10, true>
                    ( ivalue, ifmt10, afmt, i.pre, lc, grp, thousands_sep, charset );
                break;
            }
        }
    }

    int_printer_full_dynamic(const int_printer_full_dynamic&) = delete;
    int_printer_full_dynamic(int_printer_full_dynamic&&) = delete;
    int_printer_full_dynamic& operator=(const int_printer_full_dynamic&) = delete;
    int_printer_full_dynamic& operator=(int_printer_full_dynamic&&) = delete;

    STRF_HD ~int_printer_full_dynamic()
    {
        const auto& p = static_cast<const strf::printer<CharT>&>(*this);
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD explicit operator const strf::printer<CharT>& () const
    {
        return * reinterpret_cast<const strf::printer<CharT>*>(&storage_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::int_printer_static_base_and_punct<CharT, 10, true>);

    using storage_type_ = typename std::aligned_storage
        < pool_size_, alignof(strf::printer<CharT>)>
        :: type;

    storage_type_ storage_;
};

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data<2>
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<2, false> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data<8>
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<8, false> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data<10>
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<10, false> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_fmt_int_printer_data<16>
    ( fmt_int_printer_data& data
    , strf::int_format_static_base_and_punct<16, false> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<2>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base_and_punct<2, true> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<8>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base_and_punct<8, true> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<10>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base_and_punct<10, true> ifmt
    , strf::alignment_format afmt ) noexcept;

STRF_EXPLICIT_TEMPLATE
STRF_HD fmt_int_printer_data_init_result init_punct_fmt_int_printer_data<16>
    ( punct_fmt_int_printer_data& data
    , strf::validate_f validate
    , strf::int_format_static_base_and_punct<16, true> ifmt
    , strf::alignment_format afmt ) noexcept;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class default_int_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_default_int_printer<char8_t>;
//STRF_EXPLICIT_TEMPLATE class punct_int_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t,  8, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t, 10, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t, 16, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t,  8, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t, 10, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char8_t, 16, false>;

#endif

STRF_EXPLICIT_TEMPLATE class default_int_printer<char>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class default_int_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class aligned_default_int_printer<char>;
STRF_EXPLICIT_TEMPLATE class aligned_default_int_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_default_int_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_default_int_printer<wchar_t>;

// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char16_t>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<char32_t>;
// STRF_EXPLICIT_TEMPLATE class punct_int_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char,  8, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char, 10, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char, 16, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t,  8, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t, 10, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t, 16, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t,  8, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t, 10, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t, 16, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t,  8, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t, 10, true>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t, 16, true>;

STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char,  8, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char, 10, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char, 16, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t,  8, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t, 10, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char16_t, 16, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t,  8, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t, 10, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<char32_t, 16, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t,  8, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t, 10, false>;
STRF_EXPLICIT_TEMPLATE class int_printer_static_base_and_punct<wchar_t, 16, false>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <typename> struct is_int_number: public std::false_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<short>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<int>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<long>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<long long>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<unsigned short>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<unsigned int>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<unsigned long>: public std::true_type {};
// NOLINTNEXTLINE(google-runtime-int)
template <> struct is_int_number<unsigned long long>: public std::true_type {};

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_INT_HPP
