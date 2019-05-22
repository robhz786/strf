#ifndef BOOST_STRINGIFY_V0_DISPATCHER_HPP
#define BOOST_STRINGIFY_V0_DISPATCHER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/tr_string.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename W>
static auto has_reserve_impl(W&& ow)
    -> decltype(ow.reserve(std::size_t{}), std::true_type{});

template <typename W>
static std::false_type has_reserve_impl(...);

template <typename W>
using has_reserve
= decltype(stringify::v0::detail::has_reserve_impl<W>(std::declval<W>()));

class output_size_reservation_dummy
{

public:

    constexpr output_size_reservation_dummy(...)
    {
    }

    constexpr void set_reserve_size(std::size_t)
    {
    }

    constexpr void set_reserve_calc()
    {
    }

    constexpr void set_no_reserve()
    {
    }

    constexpr std::false_type has_reserve() const
    {
        return {};
    }
};

struct reserve_calc_tag{};


class output_size_reservation_real
{
    std::size_t _size = 0;
    enum {calculate_size, predefined_size, no_reserve} _flag = no_reserve;

public:

    constexpr output_size_reservation_real(const output_size_reservation_real&)
    = default;

    constexpr output_size_reservation_real()
        : _flag(no_reserve)
    {
    }

    constexpr explicit output_size_reservation_real(std::size_t size)
        : _size(size)
        , _flag(predefined_size)
    {
    }

    constexpr explicit output_size_reservation_real(reserve_calc_tag)
        : _flag(calculate_size)
    {
    }

    constexpr void set_reserve_size(std::size_t size)
    {
        _flag = predefined_size;
        _size = size;
    }

    constexpr void set_reserve_calc()
    {
        _flag = calculate_size;
    }

    constexpr void set_no_reserve()
    {
        _flag = no_reserve;
    }

    constexpr bool must_calculate_size() const
    {
        return _flag == calculate_size;
    }
    constexpr std::size_t get_size_to_reserve() const
    {
        return _size;
    }

    constexpr std::true_type has_reserve() const
    {
        return {};
    }
};


template <typename OutputBuff>
using output_size_reservation
= typename std::conditional
    < stringify::v0::detail::has_reserve<OutputBuff>::value
    , output_size_reservation_real
    , output_size_reservation_dummy
    > :: type;

#ifdef __cpp_fold_expressions

template <typename ... Printers>
inline std::size_t sum_necessary_size(const Printers& ... printers)
{
    return (... + printers.necessary_size());
}

template <typename CharT, typename ... Printers>
inline bool write_args( stringify::v0::output_buffer<CharT>& ob
               , const Printers& ... printers )
{
    return (... && printers.write(ob));
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
inline bool write_args(stringify::v0::output_buffer<CharT>&)
{
    return true;
}

template <typename CharT, typename Printer, typename ... Printers>
inline bool write_args
    ( stringify::v0::output_buffer<CharT>& ob
    , const Printer& printer
    , const Printers& ... printers )
{
    return printer.write(ob) && write_args(ob, printers ...);
}

#endif

template <typename OutputBuff, typename ... Printers>
inline decltype(std::declval<OutputBuff>().finish()) reserve_and_write
    ( stringify::v0::detail::output_size_reservation_real reser
    , OutputBuff& ob
    , const Printers& ... printers )
{
    std::size_t s = ( reser.must_calculate_size()
                    ? stringify::v0::detail::sum_necessary_size(printers...)
                    : reser.get_size_to_reserve() );
    if (s != 0)
    {
        ob.reserve(s);
    }
    stringify::v0::detail::write_args(ob, printers...);
    return ob.finish();
}

template <typename OutputBuff, typename ... Printers>
inline decltype(std::declval<OutputBuff>().finish()) reserve_and_write
    ( stringify::v0::detail::output_size_reservation_dummy
    , OutputBuff& ob
    , const Printers& ... printers )
{
    stringify::v0::detail::write_args(ob, printers...);
    return ob.finish();
}

template <typename CharT>
inline const stringify::v0::printer<CharT>&
as_printer_cref(const stringify::v0::printer<CharT>& p)
{
    return p;
}

template <typename OutputBuff, typename CharT>
inline decltype(std::declval<OutputBuff>().finish()) reserve_and_tr_write
    ( stringify::v0::detail::output_size_reservation_real reser
    , OutputBuff& ob
    , const CharT* tr_str
    , const CharT* tr_str_end
    , std::initializer_list<const stringify::v0::printer<CharT>*> args
    , stringify::v0::encoding<CharT> enc
    , stringify::v0::tr_invalid_arg policy )
{
    if(reser.must_calculate_size())
    {
        auto invs = stringify::v0::detail::invalid_arg_size(enc, policy);
        auto size = stringify::v0::detail::tr_string_size( tr_str, tr_str_end
                                                         , args, invs );
        ob.reserve(size);
    }
    else if(reser.get_size_to_reserve() != 0)
    {
        ob.reserve(reser.get_size_to_reserve());
    }

    bool no_error = stringify::v0::detail::tr_string_write
        ( tr_str, tr_str_end, args, ob, enc, policy );

    BOOST_ASSERT(no_error == ! ob.has_error());
    (void) no_error;

    return ob.finish();
}

template <typename OutputBuff, typename CharT>
inline decltype(std::declval<OutputBuff>().finish()) reserve_and_tr_write
    ( stringify::v0::detail::output_size_reservation_dummy
    , OutputBuff& ob
    , const CharT* tr_str
    , const CharT* tr_str_end
    , std::initializer_list<const stringify::v0::printer<CharT>*> args
    , stringify::v0::encoding<CharT> enc
    , stringify::v0::tr_invalid_arg policy )
{
    bool no_error = stringify::v0::detail::tr_string_write
        ( tr_str, tr_str_end, args, ob, enc, policy );

    BOOST_ASSERT(no_error == ! ob.has_error());
    (void) no_error;

    return ob.finish();
}

template <std::size_t Index, typename T>
struct indexed_wrapper
{
    using value_type = T;

    template <typename U>
    constexpr explicit indexed_wrapper(U&& u)
        : value(std::forward<U>(u))
    {
    }

    constexpr indexed_wrapper(const indexed_wrapper&) = default;
    constexpr indexed_wrapper(indexed_wrapper&&) = default;

    T value;
};

template <typename ISequence, typename ... T>
class simple_tuple_impl;

template <typename T, typename... U>
constexpr bool is_constructible_v = std::is_constructible<T, U...>::value;

template <typename ... T>
struct not_a_simple_tuple_impl_impl
{
    using type = void;
};

template <typename ... T>
struct not_a_simple_tuple_impl_impl<const simple_tuple_impl<T...>&>
{
};

template <typename ... T>
using not_a_simple_tuple_impl =
    typename detail::not_a_simple_tuple_impl_impl<const T&...>::type;



template <std::size_t ... I, typename ... T>
class simple_tuple_impl<std::index_sequence<I...>, T...>
    : private indexed_wrapper<I, T> ...
{
    template <std::size_t N, typename U>
    static indexed_wrapper<N, U> f(indexed_wrapper<N, U>*);

    template <std::size_t N>
    using _wrapper_type_at = decltype(f<N>((simple_tuple_impl*)0));

    template <std::size_t N>
    using _value_type_at = typename _wrapper_type_at<N>::value_type;

public:

    template < typename ... U
             , typename = detail::not_a_simple_tuple_impl<U...> >
    constexpr explicit simple_tuple_impl(U&&...args)
        : indexed_wrapper<I, T>(std::forward<U>(args))...
    {
    }

    constexpr simple_tuple_impl(const simple_tuple_impl& ) = default;
    constexpr simple_tuple_impl(simple_tuple_impl&& ) = default;

    template <std::size_t N>
    constexpr const _value_type_at<N>& get() const &
    {
        using W = _wrapper_type_at<N>;
        return static_cast<const W*>(this)->value;
    }

    template <std::size_t N>
    constexpr _value_type_at<N>&& forward() &&
    {
        using W = _wrapper_type_at<N>;
        return static_cast<_value_type_at<N>&&>(static_cast<W&>(*this).value);
    }
};

template <typename ... T>
using simple_tuple = simple_tuple_impl<std::index_sequence_for<T...>, T...>;

} // namespace detail

template<typename FPack, typename OutputBuff, typename ... OutBuffArgs >
class dispatcher
    : private stringify::v0::detail::output_size_reservation<OutputBuff>
    , private stringify::v0::detail::simple_tuple<OutBuffArgs...>
{


    using _reservation
        = stringify::v0::detail::output_size_reservation<OutputBuff>;

    using _obargs_tuple = stringify::v0::detail::simple_tuple<OutBuffArgs...>;

    using _obargs_index_sequence = std::make_index_sequence<sizeof...(OutBuffArgs)>;

public:

    using return_type
        = decltype(std::declval<OutputBuff&>().finish());

    using char_type = typename OutputBuff::char_type;

    constexpr dispatcher(const dispatcher& d) = default;
    /*    : _reservation(d)
        , _obargs_tuple(static_cast<const _obargs_tuple&>(d))
        , _fpack(static_cast<const FPack&>(d._fpack))
    {
    }*/

    constexpr dispatcher(dispatcher&& d) = default;
    /*   : _reservation(d)
        , _obargs_tuple(static_cast<_obargs_tuple&&>(d))
        , _fpack(static_cast<FPack&&>(d._fpack))
    {
    }*/

    template
        < typename ... OBArgs
        , typename = std::enable_if_t<sizeof...(OBArgs) == sizeof...(OutBuffArgs)> >
           // && stringify::v0::detail::fold_and
           //      < detail::is_constructible_v<OutBuffArgs, OBArgs&&>... > > >
    constexpr explicit dispatcher(FPack&& fp, OBArgs&&... args)
        : _obargs_tuple(std::forward<OBArgs>(args)...)
        , _fpack(std::move(fp))
    {
    }

    template
        < typename ... OBArgs
        , typename = std::enable_if_t<sizeof...(OBArgs) == sizeof...(OutBuffArgs)> >
           // && stringify::v0::detail::fold_and
           //      < detail::is_constructible_v<OutBuffArgs, OBArgs&&>... > > >
    constexpr explicit dispatcher(const FPack& fp, OBArgs&&... args)
        : _obargs_tuple(std::forward<OBArgs>(args)...)
        , _fpack(fp)
    {
    }

    template <typename ... Fpe>
    BOOST_STRINGIFY_NODISCARD constexpr auto facets(Fpe&& ... fpe) const &
    {
        static_assert
            ( stringify::v0::detail::fold_and
                  <std::is_copy_constructible<OutBuffArgs>::value ...>
            , "OutBuffArgs... must all be copy constructible" );

        using NewFPack = decltype
            ( stringify::v0::pack(_fpack, std::forward<Fpe>(fpe) ...) );

        return dispatcher<NewFPack, OutputBuff, OutBuffArgs...>
            ( *this
            , stringify::v0::pack(_fpack, std::forward<Fpe>(fpe) ...) );
    }

    template <typename ... Fpe>
    BOOST_STRINGIFY_NODISCARD constexpr auto facets(Fpe&& ... fpe) &&
    {
        static_assert
            ( stringify::v0::detail::fold_and
                  <std::is_move_constructible<OutBuffArgs>::value ...>
            , "OutBuffArgs... must be move constructible" );

        using NewFPack = decltype
            ( stringify::v0::pack(_fpack, std::forward<Fpe>(fpe) ...) );

        return dispatcher<NewFPack, OutputBuff, OutBuffArgs...>
            ( std::move(*this)
            , stringify::v0::pack(_fpack, std::forward<Fpe>(fpe) ...) );
    }

    BOOST_STRINGIFY_NODISCARD constexpr dispatcher no_reserve() const &
    {
        return dispatcher(_reservation(), *this);
    }
    BOOST_STRINGIFY_NODISCARD constexpr dispatcher no_reserve() const &&
    {
        return dispatcher(_reservation(), *this);
    }
    constexpr dispatcher& no_reserve() &
    {
        this->set_no_reserve();
        return *this;
    }
    constexpr dispatcher&& no_reserve() &&
    {
        this->set_no_reserve();
        return static_cast<dispatcher&&>(*this);
    }

    BOOST_STRINGIFY_NODISCARD constexpr dispatcher reserve_calc() const &
    {
        return dispatcher(_reservation(detail::reserve_calc_tag{}), *this);
    }
    BOOST_STRINGIFY_NODISCARD constexpr dispatcher reserve_calc() const &&
    {
        return dispatcher(_reservation(detail::reserve_calc_tag{}), *this);
    }
    constexpr dispatcher& reserve_calc() &
    {
        this->set_reserve_calc();
        return *this;
    }
    constexpr dispatcher&& reserve_calc() &&
    {
        this->set_reserve_calc();
        return static_cast<dispatcher&&>(*this);
    }

    BOOST_STRINGIFY_NODISCARD
    constexpr dispatcher reserve(std::size_t size) const &
    {
        return dispatcher(_reservation(size), *this);
    }
    BOOST_STRINGIFY_NODISCARD
    constexpr dispatcher reserve(std::size_t size) const &&
    {
        return dispatcher(_reservation(size), *this);
    }
    constexpr dispatcher& reserve(std::size_t size) &
    {
        this->set_reserve_size(size);
        return *this;
    }
    constexpr dispatcher&& reserve(std::size_t size) &&
    {
        this->set_reserve_size(size);
        return static_cast<dispatcher&&>(*this);
    }

    static std::size_t calculate_size()
    {
        return 0;
    }

    template <typename Arg0, typename ... Args>
    static std::size_t calculate_size(const Arg0& arg, const Args ... args)
    {
        return arg.necessary_size() + calculate_size(args...);
    }

    template <typename ... Args>
    return_type operator()(const Args& ... args) const &
    {
        return _create_ob_and_write(_obargs_index_sequence(), args...);
    }

    template <typename ... Args>
    return_type operator()(const Args& ... args) &&
    {
        return static_cast<dispatcher&&>(*this)._create_ob_and_write
            ( _obargs_index_sequence(), args... );
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename ... Args>
    return_type tr
        ( const std::basic_string_view<char_type>& str
        , const Args& ... args ) const &
    {
        return _create_ob_and_tr_write( _obargs_index_sequence()
                                      , str.begin()
                                      , str.end()
                                      , args ... );
    }

    template <typename ... Args>
    return_type tr
        ( const std::basic_string_view<char_type>& str
        , const Args& ... args ) &&
    {
        return static_cast<dispatcher&&>(*this)._create_ob_and_tr_write
            ( _obargs_index_sequence()
            , str.begin()
            , str.end()
            , args ... );
    }

#else

    template <typename ... Args>
    return_type tr(const char_type* str, const Args& ... args) const &
    {
        return _create_ob_and_tr_write
            ( _obargs_index_sequence()
            , str
            , str + std::char_traits<char_type>::length(str)
            , args ... );
    }

    template <typename ... Args>
    return_type tr(const char_type* str, const Args& ... args ) &&
    {
        return std::move(*this)._create_ob_and_tr_write
            ( _obargs_index_sequence()
            , str
            , str + std::char_traits<char_type>::length(str)
            , args ... );
    }

    template <typename Traits, typename A, typename ... Args>
    return_type tr
        ( const std::basic_string<char_type, Traits, A>& str
        , const Args& ... args ) const &
    {
        return _create_ob_and_tr_write
            ( _obargs_index_sequence()
            , str.data()
            , str.data() + str.size()
            , args ... );
    }

    template <typename Traits, typename A, typename ... Args>
    return_type tr
    ( const std::basic_string<char_type, Traits, A>& str
        , const Args& ... args ) &&
    {
        return std::move(*this)._create_ob_and_tr_write
            ( _obargs_index_sequence()
            , str.begin()
            , str.end()
            , args ... );
    }

#endif

private:

    template <std::size_t ... I, typename ... Args>
    return_type _create_ob_and_write
        ( std::index_sequence<I...>
        , const Args& ... args ) const &
    {
        OutputBuff ob(this->template get<I>()...);
        return _do_write(ob, args...);
    }

    template <std::size_t ... I, typename ... Args>
    return_type _create_ob_and_write
        ( std::index_sequence<I...>
        , const Args& ... args ) &&
    {
        OutputBuff ob(std::move(*this).template forward<I>()...);
        return _do_write(ob, args...);
    }

    static inline const stringify::v0::printer<char_type>&
    _as_printer_cref(const stringify::v0::printer<char_type>& p)
    {
        return p;
    }

    template <typename ... Args>
    return_type _do_write(OutputBuff& ob, const Args& ... args) const
    {
        return detail::reserve_and_write
            ( static_cast<const _reservation&>(*this)
            , ob
            , _as_printer_cref(make_printer<char_type, FPack>(_fpack, args))... );
    }

    template <std::size_t ... I, typename ... Args>
    return_type _create_ob_and_tr_write
        ( std::index_sequence<I...>
        , const char_type* tr_str
        , const char_type* tr_str_end
        , const Args& ... args ) const &
    {
        OutputBuff ob(this->template get<I>()...);
        return _do_tr_write(ob, tr_str, tr_str_end, args...);
    }

    template <std::size_t ... I, typename ... Args>
    return_type _create_ob_and_tr_write
        ( std::index_sequence<I...>
        , const char_type* tr_str
        , const char_type* tr_str_end
        , const Args& ... args ) &&
    {
        OutputBuff ob(std::move(*this).template forward<I>()...);
        return _do_tr_write(ob, tr_str, tr_str_end, args...);
    }

    static inline const stringify::v0::printer<char_type>*
    _as_printer_cptr(const stringify::v0::printer<char_type>& p)
    {
         return &p;
    }

    template <typename ... Args>
    return_type _do_tr_write
        ( OutputBuff& ob
        , const char_type* tr_str
        , const char_type* tr_str_end
        , const Args& ... args ) const
    {
        using cat1 = stringify::v0::encoding_c<char_type>;
        using cat2 = stringify::v0::tr_invalid_arg_c;

        return stringify::v0::detail::reserve_and_tr_write
            ( static_cast<const _reservation&>(*this)
            , ob
            , tr_str
            , tr_str_end
            , { _as_printer_cptr(make_printer<char_type, FPack>(_fpack, args))... }
            , stringify::v0::get_facet<cat1, void>(_fpack)
            , stringify::v0::get_facet<cat2, void>(_fpack) );
    }

    template <class, class, class...>
    friend class dispatcher;

    template <class FP2>
    constexpr dispatcher
        ( const dispatcher<FP2, OutputBuff, OutBuffArgs...>& d
        , FPack&& fp )
        : _reservation(d)
        , _obargs_tuple(static_cast<const _obargs_tuple&>(d))
        , _fpack(static_cast<FPack&&>(fp))
    {
    }

    template <class FP2>
    constexpr dispatcher
        ( dispatcher<FP2, OutputBuff, OutBuffArgs...>&& d
        , FPack&& fp )
        : _reservation(d)
        , _obargs_tuple(static_cast<_obargs_tuple&&>(d))
        , _fpack(static_cast<FPack&&>(fp))
    {
    }

    constexpr dispatcher(const _reservation& r, const dispatcher& d)
        : _reservation(r)
        , _obargs_tuple(static_cast<const _obargs_tuple&>(d))
        , _fpack(d._fpack)
    {
    }

    FPack _fpack;
};


template <typename CharOut, typename FPack, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>( std::declval<FPack>()
                                       , std::declval<Arg>() ) );

class stringify_error: public std::system_error
{
public:
    using std::system_error::system_error;
};

class BOOST_STRINGIFY_NODISCARD nodiscard_error_code: public std::error_code
{
public:
    using std::error_code::error_code;

    nodiscard_error_code() = default;

    nodiscard_error_code(const std::error_code& ec) noexcept
        : std::error_code(ec)
    {
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DISPATCHER_HPP
