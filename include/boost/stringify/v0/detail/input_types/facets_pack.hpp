#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FACETS_PACK_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/printers_tuple.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename FPack, typename ... Args>
struct inner_pack_with_args
{
    FPack fp;
    stringify::v0::detail::simple_tuple<Args...> args;
};

template <typename FPack>
struct inner_pack
{
    template <typename ... T>
    constexpr inner_pack(T&& ... args)
        : fp{std::forward<T>(args)...}
    {
    }

    FPack fp;

    template <typename ... Args>
    constexpr stringify::v0::inner_pack_with_args
        < FPack
        , stringify::v0::detail::opt_val_or_cref<Args>... >
    operator()(const Args& ... args) const
    {
        return { fp, stringify::v0::detail::make_simple_tuple(args ...) };
    }
};

namespace detail {

template < typename CharT
         , typename ParentFPack
         , typename ChildFPack
         , typename ... Args >
class facets_pack_printer: public stringify::v0::printer<CharT>
{
public:

    facets_pack_printer
        ( const ParentFPack& parent_fp
        , const stringify::v0::inner_pack_with_args<ChildFPack, Args...>& args )
        : _fp{parent_fp, args.fp}
        , _printers{_fp, args.args}
    {
    }

    std::size_t necessary_size() const override
    {
        return stringify::v0::detail::necessary_size(_printers);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override
    {
        stringify::v0::detail::write(ob, _printers);
    }

    stringify::v0::width_t width(stringify::v0::width_t limit) const override
    {
        return stringify::v0::detail::width(_printers, limit);
    }

    virtual ~facets_pack_printer()
    {
    }

private:

    stringify::v0::facets_pack<ParentFPack, ChildFPack> _fp;

    stringify::v0::detail::printers_tuple_from_args
        < CharT
        , stringify::v0::facets_pack<ParentFPack, ChildFPack>
        , Args... >
    _printers;
};

template <typename ... F>
constexpr bool are_constrainable_impl()
{
    constexpr std::size_t N = sizeof...(F);
    constexpr bool values[N] = {stringify::v0::is_constrainable_v<F> ...};

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
constexpr bool are_constrainable_impl<>()
{
    return true;
}

template <typename ... F>
struct all_are_constrainable
{
    constexpr static bool value
        = stringify::v0::detail::are_constrainable_impl<F...>();
};

} // namespace detail

template <typename ... T>
auto facets(T&& ... args)
    -> stringify::v0::inner_pack
           < decltype(stringify::v0::pack(std::forward<T>(args)...)) >
{
    static_assert
        ( stringify::v0::is_constrainable_v
            < decltype(stringify::v0::pack(std::forward<T>(args)...)) >
        , "All facet categories must be constrainable" );
    return {std::forward<T>(args)...};
}

template < typename CharT
         , typename FPack
         , typename InnerFPack
         , typename ... Args >
inline stringify::v0::detail::facets_pack_printer< CharT
                                                 , FPack
                                                 , InnerFPack
                                                 , Args... >
make_printer( const FPack& fp
            , const stringify::v0::inner_pack_with_args<InnerFPack, Args...>& f )
{
    return {fp, f};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FACETS_PACK_HPP

