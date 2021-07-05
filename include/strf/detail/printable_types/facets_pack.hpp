#ifndef STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/printers_tuple.hpp>

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
class facets_pack_printer;

} // namespace detail

template <typename ChildFPack, typename... Args>
struct print_traits<strf::inner_pack_with_args<ChildFPack, Args...>>
{
    using forwarded_type = strf::inner_pack_with_args<ChildFPack, Args...>;

    template < typename CharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , const forwarded_type& x )
        -> strf::usual_printer_input
            < CharT, Preview, FPack, forwarded_type
            , strf::detail::facets_pack_printer
                < CharT, Preview, FPack, ChildFPack, Args... > >
    {
        return {preview, fp, x};
    }
};

namespace detail {

template < typename CharT
         , typename Preview
         , typename ParentFPack
         , typename ChildFPack
         , typename ... Args >
class facets_pack_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD facets_pack_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_{input.facets, input.arg.fp}
        , printers_{input.arg.args, input.preview, fp_}
    {
    }

    facets_pack_printer(const facets_pack_printer&) = delete;
    facets_pack_printer(facets_pack_printer&&) = delete;

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override
    {
        strf::detail::write(ob, printers_);
    }

    STRF_HD virtual ~facets_pack_printer()
    {
    }

private:

    strf::facets_pack<ParentFPack, ChildFPack> fp_;

    strf::detail::printers_tuple_from_args
        < CharT
        , Preview
        , strf::facets_pack<ParentFPack, ChildFPack>
        , Args... >
    printers_;
};

template <typename ... F>
constexpr STRF_HD bool are_constrainable_impl()
{
    constexpr std::size_t N = sizeof...(F);
    constexpr bool values[N] = {strf::is_constrainable<F>() ...};

    for (std::size_t i = 0; i < N; ++i) {
        if( ! values[i]) {
            return false;
        }
    }
    return true;
}

template <>
constexpr STRF_HD bool are_constrainable_impl<>()
{
    return true;
}

template <typename ... F>
struct all_are_constrainable
{
    constexpr static bool value
        = strf::detail::are_constrainable_impl<F...>();
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

