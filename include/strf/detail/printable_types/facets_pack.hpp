#ifndef STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/stringifiers_tuple.hpp>

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
    STRF_HD constexpr inner_pack(T&& ... args)
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
class facets_pack_stringifier;

} // namespace detail

template <typename ChildFPack, typename... Args>
struct printable_traits<strf::inner_pack_with_args<ChildFPack, Args...>>
{
    using forwarded_type = strf::inner_pack_with_args<ChildFPack, Args...>;

    template < typename CharT, typename PrePrinting, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , const forwarded_type& x )
        -> strf::usual_stringifier_input
            < CharT, PrePrinting, FPack, forwarded_type
            , strf::detail::facets_pack_stringifier
                < CharT, PrePrinting, FPack, ChildFPack, Args... > >
    {
        return {pre, fp, x};
    }
};

namespace detail {

template < typename CharT
         , typename PrePrinting
         , typename ParentFPack
         , typename ChildFPack
         , typename ... Args >
class facets_pack_stringifier: public strf::stringifier<CharT>
{
public:

    template <typename... T>
    STRF_HD facets_pack_stringifier
        ( const strf::usual_stringifier_input<T...>& input )
        : fp_{input.facets, input.arg.fp}
        , printers_{input.arg.args, input.pre, fp_}
    {
    }

    facets_pack_stringifier(const facets_pack_stringifier&) = delete;
    facets_pack_stringifier(facets_pack_stringifier&&) = delete;

    STRF_HD void print_to(strf::destination<CharT>& dest) const override
    {
        strf::detail::write(dest, printers_);
    }

    STRF_HD virtual ~facets_pack_stringifier()
    {
    }

private:

    strf::facets_pack<ParentFPack, ChildFPack> fp_;

    strf::detail::stringifiers_tuple_from_args
        < CharT
        , PrePrinting
        , strf::facets_pack<ParentFPack, ChildFPack>
        , Args... >
    printers_;
};

} // namespace detail

template <typename ... T>
STRF_HD auto with(T&& ... args)
    -> strf::inner_pack<decltype(strf::pack(std::forward<T>(args)...))>
{
    static_assert
        ( strf::is_constrainable<decltype(strf::pack(std::forward<T>(args)...))>()
        , "All facet categories must be constrainable" );
    return {std::forward<T>(args)...};
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

