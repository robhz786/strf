#ifndef BOOST_STRINGIFY_V0_DETAIL_PRINTERS_TUPLE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_PRINTERS_TUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename Arg>
using opt_val_or_cref = std::conditional_t
    < ( std::is_trivially_copyable<Arg>::value
     && ! std::is_array<Arg>::value
     && sizeof(Arg) < 4 * sizeof(void*) )
    , Arg
    , const Arg& > ;

template <std::size_t I, typename T>
struct indexed_obj
{
    T obj;
};

struct simple_tuple_from_args {};

template <typename ISeq, typename ... T>
class simple_tuple_impl;

template <std::size_t ... I, typename ... T>
class simple_tuple_impl<std::index_sequence<I...>, T...>
    : private detail::indexed_obj<I, T> ...
{
    template <std::size_t J, typename U>
    constexpr static const indexed_obj<J, U>& _get(const indexed_obj<J, U>& r)
    {
        return r;
    }

public:

    static constexpr std::size_t size = sizeof...(T);

    template <typename ... Args>
    constexpr explicit simple_tuple_impl(simple_tuple_from_args, Args&& ... args)
        : detail::indexed_obj<I, T>{args}...
    {
    }

    constexpr simple_tuple_impl(const simple_tuple_impl&) = default;
    constexpr simple_tuple_impl(simple_tuple_impl&&) = default;

    template <std::size_t J>
    constexpr const auto& get() const
    {
        return _get<J>(*this).obj;
    }
};

template <typename ... T>
class simple_tuple
    : public stringify::v0::detail::simple_tuple_impl
    < std::make_index_sequence<sizeof...(T)>, T...>
{
    using stringify::v0::detail::simple_tuple_impl
        < std::make_index_sequence<sizeof...(T)>, T...>
        ::simple_tuple_impl;
};

template <typename ... Args>
constexpr stringify::v0::detail::simple_tuple
    < stringify::v0::detail::opt_val_or_cref<Args>... >
make_simple_tuple(const Args& ... args)
{
    return stringify::v0::detail::simple_tuple
        < stringify::v0::detail::opt_val_or_cref<Args>... >
    { stringify::v0::detail::simple_tuple_from_args{}, args... };
}


#ifdef __cpp_fold_expressions

template <typename ... Printers>
inline std::size_t sum_necessary_size(const Printers& ... printers)
{
    return (0 + ... + printers.necessary_size());
}

template <typename CharT, typename ... Printers>
inline void write_args( stringify::v0::basic_outbuf<CharT>& ob
                      , const Printers& ... printers )
{
    (... , printers.write(ob));
}

#else

inline std::size_t sum_necessary_size()
{
    return 0;
}

template <typename Printer, typename ... Printers>
inline std::size_t sum_necessary_size(const Printer& printer, const Printers& ... printers)
{
    return printer.necessary_size()
        + stringify::v0::detail::sum_necessary_size(printers...);
}


template <typename CharT>
inline void write_args(stringify::v0::basic_outbuf<CharT>&)
{
}

template <typename CharT, typename Printer, typename ... Printers>
inline void write_args
    ( stringify::v0::basic_outbuf<CharT>& ob
    , const Printer& printer
    , const Printers& ... printers )
{
    printer.write(ob);
    if (ob.good()) {
        write_args(ob, printers ...);
    }
}

#endif

constexpr int sum_width(int)
{
    return 0;
}

template <typename CharT, typename ... Printers>
int sum_width( int limit
             , const stringify::v0::printer<CharT>& p
             , const Printers& ... printers )
{
    auto w = p.width(limit);
    return w < limit ? (w + sum_width(limit - w, printers...)) : limit;
}

template <std::size_t J, typename ... T>
constexpr const auto& get(const simple_tuple<T...>& tp)
{
    return tp.template get<J>();
}

template < typename CharT
         , typename ISeq
         , typename ... Printers >
class printers_tuple_impl;

template < typename CharT
         , std::size_t ... I
         , typename ... Printers >
class printers_tuple_impl<CharT, std::index_sequence<I...>, Printers...>
    : private detail::indexed_obj<I, Printers> ...
{
    template <std::size_t J, typename T>
    static const indexed_obj<J, T>& _get(const indexed_obj<J, T>& r)
    {
        return r;
    }

public:

    using char_type = CharT;
    static constexpr std::size_t size = sizeof...(Printers);

    template < typename FPack, typename ... Args >
    printers_tuple_impl
        ( const FPack& fp
        , const stringify::v0::detail::simple_tuple<Args...>& args )
        : indexed_obj<I, Printers>
            { make_printer<CharT>(fp, args.template get<I>()) } ...
    {
    }

    printers_tuple_impl(const printers_tuple_impl& fp) = default;
    printers_tuple_impl(printers_tuple_impl&& fp) = default;

    template <std::size_t J>
    const auto& get() const
    {
        return _get<J>(*this).obj;
    }
};

template< typename CharT, std::size_t ... I, typename ... Printers >
std::size_t necessary_size
    ( const stringify::v0::detail::printers_tuple_impl
          < CharT, std::index_sequence<I...>, Printers... >& printers )
{
    return stringify::v0::detail::sum_necessary_size
        ( printers.template get<I>()... );
}

template< typename CharT, std::size_t ... I, typename ... Printers >
int width( const stringify::v0::detail::printers_tuple_impl
             < CharT, std::index_sequence<I...>, Printers... >& printers
         , int lim )
{
    return stringify::v0::detail::sum_width(lim, printers.template get<I>()...);
}

template< typename CharT, std::size_t ... I, typename ... Printers >
void write( stringify::v0::basic_outbuf<CharT>& ob
          , const stringify::v0::detail::printers_tuple_impl
             < CharT, std::index_sequence<I...>, Printers... >& printers )
{
    stringify::v0::detail::write_args<CharT>(ob, printers.template get<I>()...);
}

template < typename CharT, typename ... Printers >
using printers_tuple = printers_tuple_impl
        < CharT
        , std::make_index_sequence<sizeof...(Printers)>
        , Printers... >;

template < typename CharT, typename FPack, typename ISeq, typename ... Args >
class printers_tuple_alias
{
    template <typename Arg>
    using _printer
    = decltype(make_printer<CharT>( std::declval<const FPack&>()
                                  , std::declval<const Arg&>()));
public:

    using type = printers_tuple_impl<CharT, ISeq, _printer<Args>...>;
};

template < typename CharT, typename FPack, typename ... Args >
using printers_tuple_from_args
= typename printers_tuple_alias
    < CharT, FPack, std::make_index_sequence<sizeof...(Args)>, Args... >
    :: type;

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_PRINTERS_TUPLE_HPP

