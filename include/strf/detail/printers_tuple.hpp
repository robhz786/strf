#ifndef STRF_DETAIL_PRINTERS_TUPLE_HPP
#define STRF_DETAIL_PRINTERS_TUPLE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>
#include <strf/detail/polymorphic_printer.hpp>

namespace strf {
namespace detail {

template <std::size_t I, typename T>
struct indexed_obj
{
    constexpr STRF_HD explicit indexed_obj(const T& cp)
        : obj(cp)
    {
    }

    T obj;
};

struct simple_tuple_from_args {};

template <typename ISeq, typename... T>
class simple_tuple_impl;

template <std::size_t... I, typename... T>
class simple_tuple_impl<strf::detail::index_sequence<I...>, T...>
    : public indexed_obj<I, T> ...
{
    template <std::size_t J, typename U>
    static constexpr STRF_HD const indexed_obj<J, U>& get_(const indexed_obj<J, U>* r) noexcept
    {
        return *r;
    }

    template <typename U>
    static constexpr STRF_HD const U& as_cref(const U& r) noexcept
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
    constexpr STRF_HD auto get() const noexcept
        -> decltype(as_cref(get_<J>(this).obj))
    {
        return get_<J>(this).obj;
    }
};

template <typename ... T>
class simple_tuple
    : public strf::detail::simple_tuple_impl
    < strf::detail::make_index_sequence<sizeof...(T)>, T...>
{
    using strf::detail::simple_tuple_impl
        < strf::detail::make_index_sequence<sizeof...(T)>, T...>
        ::simple_tuple_impl;
};

template <typename ... Args>
constexpr STRF_HD strf::detail::simple_tuple<Args...>
make_simple_tuple(const Args& ... args)
{
    return strf::detail::simple_tuple<Args...>
    { strf::detail::simple_tuple_from_args{}, args... };
}

template <std::size_t J, typename ... T>
constexpr STRF_HD auto get(const simple_tuple<T...>& tp)
    -> decltype(tp.template get<J>())
{
    return tp.template get<J>();
}

template <std::size_t I, typename Printer>
struct indexed_printer
{
    template <typename Arg>
    STRF_HD explicit indexed_printer(const Arg& arg)
        : printer(arg)
    {
    }

    Printer printer;
};

template < typename CharT
         , typename ISeq
         , typename ... Printers >
class printers_tuple_impl;

template < typename CharT
         , std::size_t ... I
         , typename ... Printers >
class printers_tuple_impl<CharT, strf::detail::index_sequence<I...>, Printers...>
    : public detail::indexed_printer<I, Printers> ...
{
    template <std::size_t J, typename T>
    static constexpr STRF_HD const indexed_printer<J, T>& get_(const indexed_printer<J, T>* r) noexcept
    {
        return *r;
    }

    template <typename U>
    static constexpr STRF_HD const U& as_cref(const U& r) noexcept
    {
        return r;
    }

public:

    static constexpr std::size_t size = sizeof...(Printers);

    template < typename... Args
             , typename... FPElems
             , strf::size_presence SizePresence
             , strf::width_presence WidthPresence >
    STRF_HD printers_tuple_impl
        ( const strf::detail::simple_tuple<Args...>& args
        , strf::premeasurements<SizePresence, WidthPresence>* pre
        , const strf::facets_pack<FPElems...>& fp )
        : indexed_printer<I, Printers>
            ( strf::make_printer<CharT>
              ( pre, fp, args.template get<I>() ) ) ...
    {
        STRF_MAYBE_UNUSED(pre);
    }

    template < typename... Args
             , typename... FPElems
             , strf::size_presence SizePresence
             , strf::width_presence WidthPresence >
    STRF_HD printers_tuple_impl
        ( const strf::detail::simple_tuple<Args...>& args
        , const strf::facets_pack<FPElems...>& fp
        , strf::premeasurements<SizePresence, WidthPresence>* pp_array )
        : indexed_printer<I, Printers>
            ( strf::make_printer<CharT>
              ( &pp_array[I], fp, args.template get<I>() ) ) ...
    {
    }

    template <std::size_t J>
    constexpr STRF_HD auto get() const noexcept
        -> decltype(as_cref(get_<J>(this).printer))
    {
        return get_<J>(this).printer;
    }
};


template<typename CharT, std::size_t ... I, typename ... Printers>
STRF_HD void write
    ( strf::destination<CharT>& dst
    , const strf::detail::printers_tuple_impl
        < CharT, strf::detail::index_sequence<I...>, Printers... >& printers )
{
    strf::detail::write_args<CharT>(dst, printers.template get<I>()...);
}

template <typename CharT, typename ... Printers>
using printers_tuple = printers_tuple_impl
        < CharT
        , strf::detail::make_index_sequence<sizeof...(Printers)>
        , Printers... >;

template < typename CharT, typename PreMeasurements, typename FPack
         , typename ISeq, typename... Args >
class printers_tuple_alias
{
public:
    using type = printers_tuple_impl
        <CharT, ISeq, strf::printer_type<CharT, PreMeasurements, FPack, Args> ...>;
};

template < typename CharT, typename PreMeasurements, typename FPack, typename ... Args >
using printers_tuple_from_args
= typename printers_tuple_alias
    < CharT, PreMeasurements, FPack, strf::detail::make_index_sequence<sizeof...(Args)>, Args ...>
    :: type;

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_PRINTERS_TUPLE_HPP

