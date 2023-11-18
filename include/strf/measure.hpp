#ifndef STRF_MEASURE_HPP
#define STRF_MEASURE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>

namespace strf {

template < typename CharT
         , strf::precalc_size SizeRequired
         , strf::precalc_width WidthRequired
         , typename... FPE >
STRF_HD void measure
    ( strf::premeasurements<SizeRequired, WidthRequired>*
    , const strf::facets_pack<FPE...> &)
{
}

template < typename CharT
         , typename... FPE
         , typename Arg
         , typename... Args >
STRF_HD STRF_CONSTEXPR_IN_CXX14 void measure
    ( strf::premeasurements<strf::precalc_size::no, strf::precalc_width::no>*
    , const strf::facets_pack<FPE...>&
    , const Arg&
    , const Args&... ) noexcept
{
}

namespace detail {

template < typename CharT, typename... FPE >
STRF_HD STRF_CONSTEXPR_IN_CXX14 void measure_only_width
    ( strf::premeasurements<strf::precalc_size::no, strf::precalc_width::yes>*
    , const strf::facets_pack<FPE...>& ) noexcept
{
}

template < typename CharT
         , typename... FPE
         , typename Arg
         , typename... OtherArgs >
STRF_HD void measure_only_width
    ( strf::premeasurements<strf::precalc_size::no, strf::precalc_width::yes>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Arg& arg
    , const OtherArgs&... other_args )
{
    using pre_type = strf::premeasurements<strf::precalc_size::no, strf::precalc_width::yes>;

    (void) strf::printer_type<CharT, pre_type, strf::facets_pack<FPE...>, Arg>
        ( strf::make_printer_input<CharT>(pre, facets, arg) );

    if (pre->remaining_width() > 0) {
        strf::detail::measure_only_width<CharT>(pre, facets, other_args...);
    }
}

} // namespace detail

template <typename CharT, typename... FPE, typename... Args>
STRF_HD void measure
    ( strf::premeasurements<strf::precalc_size::no, strf::precalc_width::yes>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Args&... args )
{
    if (pre->remaining_width() > 0) {
        strf::detail::measure_only_width<CharT>(pre, facets, args...);
    }
}

namespace detail {

template <typename... Args>
STRF_HD STRF_CONSTEXPR_IN_CXX14 void do_nothing_with(const Args&...) noexcept
{
    // workaround for the lack of support for fold expressions
}

} // namespace detail

template < typename CharT
         , strf::precalc_width WidthRequired
         , typename... FPE
         , typename... Args >
STRF_HD void measure
    ( strf::premeasurements<strf::precalc_size::yes, WidthRequired>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Args&... args )
{
    STRF_MAYBE_UNUSED(pre);
    using pre_type = strf::premeasurements<strf::precalc_size::yes, WidthRequired>;
    strf::detail::do_nothing_with
        ( strf::printer_type<CharT, pre_type, strf::facets_pack<FPE...>, Args>
          ( strf::make_printer_input<CharT>(pre, facets, args) ) ... );
}


} // namespace strf

#endif  // STRF_MEASURE_HPP

