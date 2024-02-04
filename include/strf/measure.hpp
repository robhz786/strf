#ifndef STRF_MEASURE_HPP
#define STRF_MEASURE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>

namespace strf {

template < typename CharT
         , strf::size_presence SizePresence
         , strf::width_presence WidthPresence
         , typename... FPE >
STRF_HD void measure
    ( strf::premeasurements<SizePresence, WidthPresence>*
    , const strf::facets_pack<FPE...> &)
{
}

template < typename CharT
         , typename... FPE
         , typename Arg
         , typename... Args >
STRF_HD STRF_CONSTEXPR_IN_CXX14 void measure
    ( strf::premeasurements<strf::size_presence::no, strf::width_presence::no>*
    , const strf::facets_pack<FPE...>&
    , const Arg&
    , const Args&... ) noexcept
{
}

namespace detail {

template < typename CharT, typename... FPE >
STRF_HD STRF_CONSTEXPR_IN_CXX14 void measure_only_width
    ( strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>*
    , const strf::facets_pack<FPE...>& ) noexcept
{
}

template < typename CharT
         , typename... FPE
         , typename Arg
         , typename... OtherArgs >
STRF_HD void measure_only_width
    ( strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Arg& arg
    , const OtherArgs&... other_args )
{
    (void) strf::make_printer<CharT>(pre, facets, arg);

    if (pre->has_remaining_width()) {
        strf::detail::measure_only_width<CharT>(pre, facets, other_args...);
    }
}

} // namespace detail

template <typename CharT, typename... FPE, typename... Args>
STRF_HD void measure
    ( strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Args&... args )
{
    if (pre->has_remaining_width()) {
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
         , strf::width_presence WidthPresence
         , typename... FPE
         , typename... Args >
STRF_HD void measure
    ( strf::premeasurements<strf::size_presence::yes, WidthPresence>* pre
    , const strf::facets_pack<FPE...>& facets
    , const Args&... args )
{
    STRF_MAYBE_UNUSED(pre);
    strf::detail::do_nothing_with(strf::make_printer<CharT>(pre, facets, args)...);
}


} // namespace strf

#endif  // STRF_MEASURE_HPP

