#ifndef STRF_DETAIL_VALUE_WITH_FORMATTERS_HPP
#define STRF_DETAIL_VALUE_WITH_FORMATTERS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {

namespace detail {

template
    < class From
    , class To
    , template <class...> class List
    , class... T >
struct fmt_replace_impl2
{
    template <class U>
    using f = strf::detail::conditional_t<std::is_same<From, U>::value, To, U>;

    using type = List<f<T>...>;
};

template <class From, class List>
struct fmt_replace_impl;

template
    < class From
    , template <class...> class List
    , class... T>
struct fmt_replace_impl<From, List<T...> >
{
    template <class To>
    using type_tmpl =
        typename strf::detail::fmt_replace_impl2
            < From, To, List, T...>::type;
};

template <typename FmtA, typename FmtB, typename ValueWithFormat>
struct fmt_forward_switcher
{
    template <typename FmtAInit>
    static STRF_HD const typename FmtB::template fn<ValueWithFormat>&
    f(const FmtAInit&, const ValueWithFormat& v)
    {
        return v;
    }

    template <typename FmtAInit>
    static STRF_HD typename FmtB::template fn<ValueWithFormat>&&
    f(const FmtAInit&, ValueWithFormat&& v)
    {
        return v;
    }
};

template <typename FmtA, typename ValueWithFormat>
struct fmt_forward_switcher<FmtA, FmtA, ValueWithFormat>
{
    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
    f(strf::detail::remove_reference_t<FmtAInit>& fa,  const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }

    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
    f(strf::detail::remove_reference_t<FmtAInit>&& fa, const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }
};

} // namespace detail

template <typename List, typename From, typename To>
using fmt_replace
    = typename strf::detail::fmt_replace_impl<From, List>
    ::template type_tmpl<To>;

template <typename PrintingTraits, class... Fmts>
class value_with_formatters;

namespace detail {

template <typename T>
struct is_value_with_formatters : std::false_type
{ };

template <typename... T>
struct is_value_with_formatters<strf::value_with_formatters<T...>>: std::true_type
{ };

template <typename T>
struct is_value_with_formatters<const T> : is_value_with_formatters<T>
{ };

template <typename T>
struct is_value_with_formatters<volatile T> : is_value_with_formatters<T>
{ };

template <typename T>
struct is_value_with_formatters<T&> : is_value_with_formatters<T>
{ };

template <typename T>
struct is_value_with_formatters<T&&> : is_value_with_formatters<T>
{ };

} // namespace detail

template <typename PrintingTraits, class... Fmts>
class value_with_formatters
    : public Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>> ...
{
public:
    using traits = PrintingTraits;
    using value_type = typename PrintingTraits::forwarded_type;

    template <typename... OtherFmts>
    using replace_fmts = strf::value_with_formatters<PrintingTraits, OtherFmts ...>;

    explicit constexpr STRF_HD value_with_formatters(const value_type& v)
        : value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD value_with_formatters
        ( const value_type& v
        , const strf::value_with_formatters<OtherPrintingTraits, Fmts...>& f )
        : Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<value_with_formatters<OtherPrintingTraits, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD value_with_formatters
        ( const value_type& v
        , strf::value_with_formatters<OtherPrintingTraits, Fmts...>&& f )
        : Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_formatters<OtherPrintingTraits, Fmts...>> &&>(f) )
        ...
        , value_(static_cast<value_type&&>(v))
    {
    }

    template <typename... F, typename... FInit>
    constexpr STRF_HD value_with_formatters
        ( const value_type& v
        , strf::tag<F...>
        , FInit&&... finit )
        : F::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename... OtherFmts>
    constexpr STRF_HD explicit value_with_formatters
        ( const strf::value_with_formatters<PrintingTraits, OtherFmts...>& f )
        : Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<value_with_formatters<PrintingTraits, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD explicit value_with_formatters
        ( strf::value_with_formatters<PrintingTraits, OtherFmts...>&& f )
        : Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_formatters<PrintingTraits, OtherFmts ...>>&& >(f) )
        ...
        , value_(static_cast<value_type&&>(f.value()))
    {
    }

    template <typename Fmt, typename FmtInit, typename ... OtherFmts>
    constexpr STRF_HD value_with_formatters
        ( const strf::value_with_formatters<PrintingTraits, OtherFmts...>& f
        , strf::tag<Fmt>
        , FmtInit&& fmt_init )
        : Fmts::template fn<value_with_formatters<PrintingTraits, Fmts...>>
            ( strf::detail::fmt_forward_switcher
                  < Fmt
                  , Fmts
                  , strf::value_with_formatters<PrintingTraits, OtherFmts...> >
              :: template f<FmtInit>(fmt_init, f) )
            ...
        , value_(f.value())
    {
    }

    constexpr STRF_HD const value_type& value() const
    {
        return value_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD value_type& value()
    {
        return value_;
    }

private:

    value_type value_;
};

} // namespace strf

#endif  // STRF_DETAIL_VALUE_WITH_FORMATTERS_HPP

