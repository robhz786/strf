#ifndef STRF_DETAIL_PRINTERS_TUPLE_HPP
#define STRF_DETAIL_PRINTERS_TUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>

namespace strf {
namespace detail {

template <typename Arg>
using opt_val_or_cref = std::conditional_t
    < std::is_array<Arg>::value, const Arg&, Arg > ;

template <std::size_t I, typename T>
struct indexed_obj
{
    constexpr STRF_HD indexed_obj(const T& cp)
        : obj(cp)
    {
    }

    T obj;
};

struct simple_tuple_from_args {};

template <typename ISeq, typename ... T>
class simple_tuple_impl;

template <std::size_t ... I, typename ... T>
class simple_tuple_impl<std::index_sequence<I...>, T...>
    : private indexed_obj<I, T> ...
{
    template <std::size_t J, typename U>
    static constexpr STRF_HD const indexed_obj<J, U>& get_(const indexed_obj<J, U>& r)
    {
        return r;
    }

public:

    static constexpr std::size_t size = sizeof...(T);

    template <typename ... Args>
    constexpr STRF_HD explicit simple_tuple_impl(simple_tuple_from_args, Args&& ... args)
        : indexed_obj<I, T>(args)...
    {
    }

    template <std::size_t J>
    constexpr STRF_HD const auto& get() const
    {
        return get_<J>(*this).obj;
    }
};

template <typename ... T>
class simple_tuple
    : public strf::detail::simple_tuple_impl
    < std::make_index_sequence<sizeof...(T)>, T...>
{
    using strf::detail::simple_tuple_impl
        < std::make_index_sequence<sizeof...(T)>, T...>
        ::simple_tuple_impl;
};

template <typename ... Args>
constexpr STRF_HD strf::detail::simple_tuple
    < strf::detail::opt_val_or_cref<Args>... >
make_simple_tuple(const Args& ... args)
{
    return strf::detail::simple_tuple
        < strf::detail::opt_val_or_cref<Args>... >
    { strf::detail::simple_tuple_from_args{}, args... };
}

template <std::size_t J, typename ... T>
constexpr STRF_HD const auto& get(const simple_tuple<T...>& tp)
{
    return tp.template get<J>();
}

template <std::size_t I, typename Printer>
struct indexed_printer
{
    template <typename FPack, typename Preview, typename Arg, typename CharT>
    STRF_HD indexed_printer( const FPack& fp
                           , Preview& preview
                           , const Arg& arg
                           , strf::tag<CharT> t)
        : printer(fp, preview, arg, t)
    {
    }

    Printer printer;
};


template < std::size_t CharSize
         , typename ISeq
         , typename ... Printers >
class printers_tuple_impl;

template < std::size_t CharSize
         , std::size_t ... I
         , typename ... Printers >
class printers_tuple_impl<CharSize, std::index_sequence<I...>, Printers...>
    : private detail::indexed_printer<I, Printers> ...
{
    template <std::size_t J, typename T>
    static STRF_HD const indexed_printer<J, T>& get_(const indexed_printer<J, T>& r)
    {
        return r;
    }

public:

    static constexpr std::size_t size = sizeof...(Printers);

    template < typename FPack, typename Preview, typename CharT, typename ... Args >
    STRF_HD printers_tuple_impl
        ( const FPack& fp
        , Preview& preview
        , const strf::detail::simple_tuple<Args...>& args
        , strf::tag<CharT> tag)
        : indexed_printer<I, Printers>(fp, preview, args.template get<I>(), tag) ...
    {
        (void)tag;
    }

    template <std::size_t J>
    STRF_HD const auto& get() const
    {
        return get_<J>(*this).printer;
    }
};


template< std::size_t CharSize, std::size_t ... I, typename ... Printers >
STRF_HD void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::detail::printers_tuple_impl
        < CharSize, std::index_sequence<I...>, Printers... >& printers )
{
    strf::detail::write_args<CharSize>(ob, printers.template get<I>()...);
}

template < std::size_t CharSize, typename ... Printers >
using printers_tuple = printers_tuple_impl
        < CharSize
        , std::make_index_sequence<sizeof...(Printers)>
        , Printers... >;

template < typename CharT, typename FPack, typename ISeq, typename ... Args >
class printers_tuple_alias
{
public:
    using type = printers_tuple_impl
        <sizeof(CharT), ISeq, strf::printer_impl<CharT, FPack, Args> ...>;
};

template < typename CharT, typename FPack, typename ... Args >
using printers_tuple_from_args
= typename printers_tuple_alias
    < CharT, FPack, std::make_index_sequence<sizeof...(Args)>, Args ...>
    :: type;

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_PRINTERS_TUPLE_HPP

