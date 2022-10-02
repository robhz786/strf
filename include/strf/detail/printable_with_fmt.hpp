#ifndef STRF_DETAIL_PRINTABLE_WITH_FMT_HPP
#define STRF_DETAIL_PRINTABLE_WITH_FMT_HPP

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
class printable_with_fmt;

template <typename PrintingTraits, class... Fmts>
using value_with_formatters
STRF_DEPRECATED_MSG("value_with_formatters renamed to printable_with_fmt")
= printable_with_fmt<PrintingTraits, Fmts...>;

namespace detail {

template <typename T>
struct is_printable_with_fmt : std::false_type
{ };

template <typename... T>
struct is_printable_with_fmt<strf::printable_with_fmt<T...>>: std::true_type
{ };

template <typename T>
struct is_printable_with_fmt<const T> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<volatile T> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<T&> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<T&&> : is_printable_with_fmt<T>
{ };

} // namespace detail

template <typename PrintingTraits, class... Fmts>
class printable_with_fmt
    : public Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>> ...
{
public:
    using traits = PrintingTraits;
    using value_type = typename PrintingTraits::forwarded_type;

    template <typename... OtherFmts>
    using replace_fmts = strf::printable_with_fmt<PrintingTraits, OtherFmts ...>;

    explicit constexpr STRF_HD printable_with_fmt(const value_type& v)
        : value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , const strf::printable_with_fmt<OtherPrintingTraits, Fmts...>& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<printable_with_fmt<OtherPrintingTraits, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , strf::printable_with_fmt<OtherPrintingTraits, Fmts...>&& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<printable_with_fmt<OtherPrintingTraits, Fmts...>> &&>(f) )
        ...
        , value_(static_cast<value_type&&>(v))
    {
    }

    template <typename... F, typename... FInit>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , strf::tag<F...>
        , FInit&&... finit )
        : F::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename... OtherFmts>
    constexpr STRF_HD explicit printable_with_fmt
        ( const strf::printable_with_fmt<PrintingTraits, OtherFmts...>& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<printable_with_fmt<PrintingTraits, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD explicit printable_with_fmt
        ( strf::printable_with_fmt<PrintingTraits, OtherFmts...>&& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<printable_with_fmt<PrintingTraits, OtherFmts ...>>&& >(f) )
        ...
        , value_(static_cast<value_type&&>(f.value()))
    {
    }

    template <typename Fmt, typename FmtInit, typename ... OtherFmts>
    constexpr STRF_HD printable_with_fmt
        ( const strf::printable_with_fmt<PrintingTraits, OtherFmts...>& f
        , strf::tag<Fmt>
        , FmtInit&& fmt_init )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( strf::detail::fmt_forward_switcher
                  < Fmt
                  , Fmts
                  , strf::printable_with_fmt<PrintingTraits, OtherFmts...> >
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

#endif  // STRF_DETAIL_PRINTABLE_WITH_FMT_HPP

