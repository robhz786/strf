#ifndef STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/format_functions.hpp>

namespace strf {
namespace detail {

template <typename FPack, typename ... Args>
struct inner_pack_with_args
{
    FPack fp;
    strf::detail::simple_tuple<Args...> args;
};

template <typename FPack>
struct inner_pack
{
    template <typename ... T>
    STRF_HD constexpr explicit inner_pack(T&& ... args)
        : fp{std::forward<T>(args)...}
    {
    }

    FPack fp;

    template <typename... Args>
    constexpr strf::detail::inner_pack_with_args<FPack, strf::forwarded_printable_type<Args>...>
    STRF_HD operator()(const Args&... args) const
    {
        return { fp
               , strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                   { strf::detail::simple_tuple_from_args{}
                   , static_cast<strf::forwarded_printable_type<Args>>(args)... } };
    }
};


} // namespace detail

template <typename ChildFPack, typename... Args>
struct printable_def<strf::detail::inner_pack_with_args<ChildFPack, Args...>>
{
    using forwarded_type = strf::detail::inner_pack_with_args<ChildFPack, Args...>;
    using representative = strf::facets_pack<>;
    using is_overridable = std::false_type;

    template < typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& parentFacets
        , const forwarded_type& x )
        -> strf::detail::printers_tuple_from_args
            < CharT
            , PreMeasurements
            , strf::facets_pack<FPack, ChildFPack>
            , Args... >
    {
        return {x.args, pre, strf::pack(parentFacets, x.fp)};
    }
};

template <typename ... T>
STRF_HD auto with(T&& ... args)
    -> strf::detail::inner_pack<decltype(strf::pack(std::forward<T>(args)...))>
{
    using fp_type = decltype(strf::pack(std::forward<T>(args)...));
    static_assert
        ( strf::is_constrainable<fp_type>()
        , "All facet categories must be constrainable" );
    return strf::detail::inner_pack<fp_type>{std::forward<T>(args)...};
}

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP

