#ifndef STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

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

    template <typename ... Args>
    constexpr strf::inner_pack_with_args
        < FPack
        , strf::detail::opt_val_or_cref<Args>... >
    STRF_HD operator()(const Args& ... args) const
    {
        return { fp, strf::detail::make_simple_tuple(args ...) };
    }
};

namespace detail {

template < typename, typename, typename, typename, typename ... >
class facets_pack_printer;

} // namespace detail


template < typename CharT, typename FPack, typename Preview
         , typename ChildFPack, typename... Args >
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>
    , const strf::inner_pack_with_args<ChildFPack, Args...>& x
    , const FPack& fp
    , Preview& preview ) noexcept
    -> strf::usual_printer_input
        < CharT, FPack, Preview
        , strf::detail::facets_pack_printer
            < CharT, FPack, Preview, ChildFPack, Args... >
        , strf::inner_pack_with_args<ChildFPack, Args...> >
{
    return {fp, preview, x};
}

namespace detail {

template < typename CharT
         , typename ParentFPack
         , typename Preview
         , typename ChildFPack
         , typename ... Args >
class facets_pack_printer: public strf::printer<CharT>
{
public:

    template <typename... T>
    STRF_HD facets_pack_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_{input.fp, input.arg.fp}
        , printers_{fp_, input.preview, input.arg.args}
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
        , strf::facets_pack<ParentFPack, ChildFPack>
        , Preview
        , Args... >
    printers_;
};

template <typename ... F>
constexpr STRF_HD bool are_constrainable_impl()
{
    constexpr std::size_t N = sizeof...(F);
    constexpr bool values[N] = {strf::is_constrainable_v<F> ...};

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
        ( strf::is_constrainable_v
            < decltype(strf::pack(std::forward<T>(args)...)) >
        , "All facet categories must be constrainable" );
    return {std::forward<T>(args)...};
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

