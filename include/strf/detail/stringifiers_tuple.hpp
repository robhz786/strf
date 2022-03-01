#ifndef STRF_DETAIL_STRINGIFIERS_TUPLE_HPP
#define STRF_DETAIL_STRINGIFIERS_TUPLE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>

namespace strf {
namespace detail {

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
struct indexed_stringifier
{
    template <typename Arg>
    STRF_HD indexed_stringifier(const Arg& arg)
        : printer(arg)
    {
    }

    Printer printer;
};

template < typename CharT
         , typename ISeq
         , typename ... Printers >
class stringifiers_tuple_impl;

template < typename CharT
         , std::size_t ... I
         , typename ... Printers >
class stringifiers_tuple_impl<CharT, strf::detail::index_sequence<I...>, Printers...>
    : public detail::indexed_stringifier<I, Printers> ...
{
    template <std::size_t J, typename T>
    static constexpr STRF_HD const indexed_stringifier<J, T>& get_(const indexed_stringifier<J, T>* r) noexcept
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

    template <typename PrePrinting, typename FPack, typename... Args>
    STRF_HD stringifiers_tuple_impl
        ( const strf::detail::simple_tuple<Args...>& args
        , PrePrinting& pre
        , const FPack& fp )
        : indexed_stringifier<I, Printers>
            ( strf::make_stringifier_input<CharT>
              ( pre, fp, args.template get<I>() ) ) ...
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
    ( strf::destination<CharT>& dest
    , const strf::detail::stringifiers_tuple_impl
        < CharT, strf::detail::index_sequence<I...>, Printers... >& printers )
{
    strf::detail::write_args<CharT>
        (dest, static_cast<const strf::stringifier<CharT>&>(printers.template get<I>())...);
}

template <typename CharT, typename ... Printers>
using stringifiers_tuple = stringifiers_tuple_impl
        < CharT
        , strf::detail::make_index_sequence<sizeof...(Printers)>
        , Printers... >;

template < typename CharT, typename PrePrinting, typename FPack
         , typename ISeq, typename... Args >
class stringifiers_tuple_alias
{
public:
    using type = stringifiers_tuple_impl
        <CharT, ISeq, strf::stringifier_type<CharT, PrePrinting, FPack, Args> ...>;
};

template < typename CharT, typename PrePrinting, typename FPack, typename ... Args >
using stringifiers_tuple_from_args
= typename stringifiers_tuple_alias
    < CharT, PrePrinting, FPack, strf::detail::make_index_sequence<sizeof...(Args)>, Args ...>
    :: type;

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_STRINGIFIERS_TUPLE_HPP

