#ifndef STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/format_functions.hpp>

namespace strf {

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
    constexpr strf::inner_pack_with_args<FPack, strf::forwarded_printable_type<Args>...>
    STRF_HD operator()(const Args&... args) const
    {
        return { fp
               , strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                   { strf::detail::simple_tuple_from_args{}
                   , static_cast<strf::forwarded_printable_type<Args>>(args)... } };
    }
};

namespace detail {

template < typename, typename, typename, typename, typename ... >
class facets_pack_printer;

} // namespace detail

template <typename ChildFPack, typename... Args>
struct printable_traits<strf::inner_pack_with_args<ChildFPack, Args...>>
{
    using forwarded_type = strf::inner_pack_with_args<ChildFPack, Args...>;
    using representative_type = strf::facets_pack<>;
    using is_overridable = std::false_type;

    template < typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , const forwarded_type& x )
        -> strf::usual_printer_input
            < CharT, PreMeasurements, FPack, forwarded_type
            , strf::detail::facets_pack_printer
                < CharT, PreMeasurements, FPack, ChildFPack, Args... > >
    {
        return {pre, fp, x};
    }
};

namespace detail {

template < typename CharT
         , typename PreMeasurements
         , typename ParentFPack
         , typename ChildFPack
         , typename ... Args >
class facets_pack_printer
{
public:

    template <typename... T>
    STRF_HD explicit facets_pack_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_{input.facets, input.arg.fp}
        , printers_{input.arg.args, input.pre, fp_}
    {
    }

    facets_pack_printer() = delete;
    ~facets_pack_printer() = default;
    facets_pack_printer(const facets_pack_printer&) = delete;
    facets_pack_printer(facets_pack_printer&&) = delete;
    facets_pack_printer& operator=(const facets_pack_printer&) = delete;
    facets_pack_printer& operator=(facets_pack_printer&&) = delete;

    STRF_HD void print_to(strf::destination<CharT>& dst) const
    {
        strf::detail::write(dst, printers_);
    }

private:

    strf::facets_pack<ParentFPack, ChildFPack> fp_;

    strf::detail::printers_tuple_from_args
        < CharT
        , PreMeasurements
        , strf::facets_pack<ParentFPack, ChildFPack>
        , Args... >
    printers_;
};

} // namespace detail

template <typename ... T>
STRF_HD auto with(T&& ... args)
    -> strf::inner_pack<decltype(strf::pack(std::forward<T>(args)...))>
{
    using fp_type = decltype(strf::pack(std::forward<T>(args)...));
    static_assert
        ( strf::is_constrainable<fp_type>()
        , "All facet categories must be constrainable" );
    return strf::inner_pack<fp_type>{std::forward<T>(args)...};
}

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_FACETS_PACK_HPP

