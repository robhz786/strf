#ifndef STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP
#define STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/printers_tuple.hpp>

STRF_NAMESPACE_BEGIN

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

template < typename CharT
         , typename ParentFPack
         , typename ChildFPack
         , typename Preview
         , typename ... Args >
class facets_pack_printer: public strf::printer<CharT>
{
public:

	  STRF_HD facets_pack_printer
        ( const ParentFPack& parent_fp
        , Preview& preview
        , const strf::inner_pack_with_args<ChildFPack, Args...>& args )
        : _fp{parent_fp, args.fp}
        , _printers{_fp, preview, args.args}
    {
    }

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override
    {
        strf::detail::write(ob, _printers);
    }

    STRF_HD virtual ~facets_pack_printer()
    {
    }

private:

    strf::facets_pack<ParentFPack, ChildFPack> _fp;

    strf::detail::printers_tuple_from_args
        < CharT
        , strf::facets_pack<ParentFPack, ChildFPack>
        , Preview
        , Args... >
    _printers;
};

template <typename ... F>
constexpr STRF_HD bool are_constrainable_impl()
{
    constexpr std::size_t N = sizeof...(F);
    constexpr bool values[N] = {strf::is_constrainable_v<F> ...};

    for (std::size_t i = 0; i < N; ++i)
    {
        if( ! values[i])
        {
            return false;
        }
    }
    return true;;
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
    -> strf::inner_pack
           < decltype(strf::pack(std::forward<T>(args)...)) >
{
    static_assert
        ( strf::is_constrainable_v
            < decltype(strf::pack(std::forward<T>(args)...)) >
        , "All facet categories must be constrainable" );
    return {std::forward<T>(args)...};
}

template < typename CharT
         , typename FPack
         , typename Preview
         , typename InnerFPack
         , typename ... Args >
inline STRF_HD strf::detail::facets_pack_printer< CharT
                                        , FPack
                                        , InnerFPack
                                        , Preview
                                        , Args... >
STRF_HD make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::inner_pack_with_args<InnerFPack, Args...>& f )
{
    return {fp, preview, f};
}

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

