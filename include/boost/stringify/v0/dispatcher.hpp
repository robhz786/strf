#ifndef BOOST_STRINGIFY_V0_DISPATCHER_HPP
#define BOOST_STRINGIFY_V0_DISPATCHER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/asm_string.hpp>
#include <tuple>

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

    constexpr output_size_reservation_dummy() = default;

    constexpr output_size_reservation_dummy(const output_size_reservation_dummy&)
    = default;

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


class output_size_reservation_real
{
    std::size_t _size = 0;
    enum {calculate_size, predefined_size, no_reserve} _flag = no_reserve;

public:

    constexpr output_size_reservation_real() = default;

    constexpr output_size_reservation_real(const output_size_reservation_real&)
    = default;

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



template <typename OutputBuff>
class output_writer_from_tuple
{
public:

    template <typename ... Args>
    output_writer_from_tuple(const std::tuple<Args...>& tp)
        : output_writer_from_tuple(tp, std::make_index_sequence<sizeof...(Args)>{})
    {
    }

    OutputBuff& get()
    {
        return _out;
    }

private:

    OutputBuff _out;

    template <typename ArgsTuple, std::size_t ... I>
    output_writer_from_tuple
        ( const ArgsTuple& args
        , std::index_sequence<I...>)
        : _out(std::get<I>(args)...)
    {
    }
};

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


template < typename OutputBuffImpl
         , typename OutputBuffInitArgsTuple
         , typename ... Printers >
inline decltype(std::declval<OutputBuffImpl>().finish()) do_create_ob_and_write
    ( stringify::v0::detail::output_size_reservation_real reser
    , const OutputBuffInitArgsTuple& ob_args
    , const Printers& ... printers )
{
    stringify::v0::detail::output_writer_from_tuple<OutputBuffImpl> ob_wrapper(ob_args);

    std::size_t s = ( reser.must_calculate_size()
                    ? stringify::v0::detail::sum_necessary_size(printers...)
                    : reser.get_size_to_reserve() );
    if (s != 0)
    {
        ob_wrapper.get().reserve(s);
    }

    stringify::v0::detail::write_args(ob_wrapper.get(), printers...);
    return ob_wrapper.get().finish();
}

template < typename OutputBuffImpl
         , typename OutputBuffInitArgsTuple
         , typename ... Printers >
inline decltype(std::declval<OutputBuffImpl>().finish()) do_create_ob_and_write
    ( stringify::v0::detail::output_size_reservation_dummy
    , const OutputBuffInitArgsTuple& ob_args
    , const Printers& ... printers )
{
    stringify::v0::detail::output_writer_from_tuple<OutputBuffImpl> ob_wrapper(ob_args);

    stringify::v0::detail::write_args(ob_wrapper.get(), printers...);
    return ob_wrapper.get().finish();
}

template <typename CharT>
inline const stringify::v0::printer<CharT>&
as_printer_cref(const stringify::v0::printer<CharT>& p)
{
    return p;
}

template < typename OutputBuffImpl
         , typename ReservationType
         , typename OutputBuffInitArgsTuple
         , typename FPack
         , typename ... Args >
inline decltype(std::declval<OutputBuffImpl>().finish()) create_ob_and_write
    ( ReservationType reser
    , const OutputBuffInitArgsTuple& ob_args
    , const FPack& fp
    , const Args& ... args )
{
    using char_type = typename OutputBuffImpl::char_type;
    return stringify::v0::detail::do_create_ob_and_write<OutputBuffImpl>
        ( reser, ob_args
        , stringify::v0::detail::as_printer_cref
          ( make_printer<char_type, FPack>(fp, args) )... );
}

template
    < typename FPack
    , typename OutputBuff
    , typename OutputBuffInitArgsTuple >
class asm_dispatcher
    : private stringify::v0::detail::output_size_reservation<OutputBuff>
    , private OutputBuffInitArgsTuple
    , private FPack
{
public:    

    using char_type = typename OutputBuff::char_type;

private:
    
    using _output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputBuff>;

    using _return_type
        = decltype(std::declval<OutputBuff>().finish());

    using _arglist_type
        = std::initializer_list<const stringify::v0::printer<char_type>*>;

    using _reservation_type
        = stringify::v0::detail::output_size_reservation<OutputBuff>;

public:

    asm_dispatcher
        ( _reservation_type reservation
        , const char_type* asm_str
        , const char_type* asm_str_end
        , OutputBuffInitArgsTuple ob_args
        , const FPack& fp )
        : _reservation_type(reservation)
        , OutputBuffInitArgsTuple(ob_args)
        , FPack(fp)
        , _asm_str(asm_str)
        , _asm_str_end(asm_str_end)
    {
    }

    template <typename ... Args>
    _return_type operator() (const Args& ... args ) const
    {
        return _asm_write
            ( this->has_reserve()
            , _asm_str
            , _asm_str_end
            , {_as_printer_cptr(make_printer<char_type, FPack>(*this, args))... } );
    }

private:

    static const stringify::v0::printer<char_type>*
    _as_printer_cptr(const stringify::v0::printer<char_type>& p)
    {
        return &p;
    }

    _return_type _asm_write
        ( std::true_type
        , const char_type* str
        , const char_type* str_end
        , _arglist_type args) const
    {
        decltype(auto) enc
            = stringify::v0::get_facet<stringify::v0::encoding_category<char_type>, void>(*this);
        decltype(auto) policy
            = stringify::v0::get_facet<stringify::v0::asm_invalid_arg_category, void>(*this);

        _output_writer_wrapper writer{*this};

        std::size_t res_size = this->get_size_to_reserve();
        if (this->must_calculate_size())
        {
            auto invs = stringify::v0::detail::invalid_arg_size(enc, policy);
            res_size = stringify::v0::detail::asm_string_size(str, str_end, args, invs);
        }
        if (res_size != 0)
        {
            writer.get().reserve(res_size);
        }

        bool no_error = stringify::v0::detail::asm_string_write
            ( str, str_end, args, writer.get(), enc, policy );
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish();
    }

    _return_type _asm_write
        ( std::false_type
        , const char_type* str
        , const char_type* str_end
        , _arglist_type args ) const
    {
        _output_writer_wrapper writer{*this};

        decltype(auto) enc
            = stringify::v0::get_facet<stringify::v0::encoding_category<char_type>, void>(*this);
        decltype(auto) policy
            = stringify::v0::get_facet<stringify::v0::asm_invalid_arg_category, void>(*this);

        bool no_error = stringify::v0::detail::asm_string_write
            ( str, str_end, args, writer.get(), enc, policy );
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish();
    }

    const char_type* _asm_str;
    const char_type* _asm_str_end;
};


} // namespace detail

template<typename FPack, typename OutputBuff, typename ... OBInitArgs >
class dispatcher
    : private stringify::v0::detail::output_size_reservation<OutputBuff>
{
    static_assert
        ( std::is_constructible<OutputBuff, OBInitArgs...>::value
       && detail::fold_and<std::is_copy_constructible<OBInitArgs>::value...>  
        , "Invalid template parameters" );

    using _output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputBuff>;

    using _return_type
        = decltype(std::declval<OutputBuff>().finish());

    using _reservation
        = stringify::v0::detail::output_size_reservation<OutputBuff>;

    using _obargs_tuple = std::tuple<OBInitArgs...>;

    using _asm_dispatcher
        = stringify::v0::detail::asm_dispatcher< FPack
                                               , OutputBuff
                                               , _obargs_tuple>;
public:

    using char_type = typename OutputBuff::char_type;
    
    // template
    //     < typename ... OBArgs
    //     , typename = std::enable_if_t
    //           < sizeof...(OBArgs) != 0
    //          && sizeof...(OBArgs) == sizeof...(OBInitArgs) 
    //          && detail::fold_and
    //                 < std::is_constructible<OBInitArgs, OBArgs&&>::value... >>>
    constexpr explicit dispatcher(FPack&& fp, OBInitArgs... args)
        : _fpack(std::move(fp))
        , _ob_args(args...)
    {
    }

    template< typename T = _obargs_tuple
            , typename = std::enable_if_t
                  < std::is_constructible<T, const OBInitArgs&...>::value >>
    constexpr explicit dispatcher(const FPack& fp, const OBInitArgs& ... args)
        : _fpack(fp)
        , _ob_args(args...)
    {
    }

    constexpr dispatcher(const dispatcher&) = default;

    constexpr dispatcher(dispatcher&&) = default;

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) const &
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, facets ...))
                         , OutputBuff
                         , OBInitArgs... >
            ( *this, stringify::v0::pack(_fpack, facets ...), _ob_args );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::facets_pack<Facets...>& fp) const &
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, fp))
                         , OutputBuff
                         , OBInitArgs... >
            ( *this, stringify::v0::pack(_fpack, fp), _ob_args );
    }

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) &&
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, facets ...))
                         , OutputBuff
                         , OBInitArgs... >
            ( std::move(*this)
            , stringify::v0::pack(_fpack, facets ...)
            , std::move(_ob_args) );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::facets_pack<Facets...>& fp) &&
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, fp))
                         , OutputBuff
                         , OBInitArgs... >
            ( std::move(*this)
            , stringify::v0::pack(std::move(_fpack), fp)
            , std::move(_ob_args) );
    }

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) &
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, facets ...))
                         , OutputBuff
                         , OBInitArgs... >
            ( *this, stringify::v0::pack(_fpack, facets ...), _ob_args );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::facets_pack<Facets...>& fp) &
    {
        return dispatcher< decltype(stringify::v0::pack(_fpack, fp))
                         , OutputBuff
                         , OBInitArgs... >
            ( *this, stringify::v0::pack(_fpack, fp), _ob_args );
    }

    constexpr dispatcher facets() const &
    {
        return *this;
    }

    constexpr dispatcher& facets() &
    {
        return *this;
    }

    constexpr dispatcher&& facets() &&
    {
        return std::move(*this);
    }

    constexpr dispatcher facets(stringify::v0::facets_pack<>) const &
    {
        return *this;
    }

    constexpr dispatcher& facets(stringify::v0::facets_pack<>) &
    {
        return *this;
    }

    constexpr dispatcher&& facets(stringify::v0::facets_pack<>) &&
    {
        return std::move(*this);
    }

    constexpr dispatcher no_reserve() const &
    {
        dispatcher copy = *this;
        copy.set_no_reserve();
        return copy;
    }
    constexpr dispatcher no_reserve() const &&
    {
        dispatcher copy = *this;
        copy.set_no_reserve();
        return copy;
    }
    constexpr dispatcher no_reserve() &
    {
        this->set_no_reserve();
        return *this;
    }
    constexpr dispatcher no_reserve() &&
    {
        this->set_no_reserve();
        return *this;
    }

    constexpr dispatcher reserve_calc() const &
    {
        dispatcher copy = *this;
        copy.set_reserve_calc();
        return copy;
    }
    constexpr dispatcher reserve_calc() const &&
    {
        dispatcher copy = *this;
        copy.set_reserve_calc();
        return copy;
    }
    constexpr dispatcher reserve_calc() &
    {
        this->set_reserve_calc();
        return *this;
    }
    constexpr dispatcher&& reserve_calc() &&
    {
        this->set_reserve_calc();
        return std::move(*this);
    }

    constexpr dispatcher reserve(std::size_t size) const &
    {
        dispatcher copy = *this;
        copy.set_reserve_size(size);
        return copy;
    }
    constexpr dispatcher reserve(std::size_t size) const &&
    {
        dispatcher copy = *this;
        copy.set_reserve_size(size);
        return copy;
    }
    constexpr dispatcher reserve(std::size_t size) &
    {
        this->set_reserve_size(size);
        return *this;
    }
    constexpr dispatcher reserve(std::size_t size) &&
    {
        this->set_reserve_size(size);
        return *this;
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

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename Arg, typename ... Args>
    _return_type as
        ( const std::basic_string_view<char_type>& str
        , const Arg& ... arg
        , const Args& ... args ) const
    {
        return as(str)(arg, args...);
    }

    _asm_dispatcher as(const std::basic_string_view<char_type>& str) const
    {
        return {*this, str.begin(), str.end(), _ob_args, _fpack};
    }

#else

    template <typename Arg, typename ... Args>
    _return_type as
        ( const char_type* str
        , const Arg& arg
        , const Args& ... args) const
    {
        return as(str)(arg, args...);
    }

    template <typename Arg, typename Traits, typename ... Args>
    _return_type as
        ( const std::basic_string<char_type, Traits>& str
        , const Arg& arg
        , const Args& ... args ) const
    {
        return as(str)(arg, args...);
    }

    _asm_dispatcher as(const char_type* str) const
    {
        return { *this
               , str
               , str + std::char_traits<char_type>::length(str)
               , _ob_args
               , _fpack };
    }

    _asm_dispatcher as(const std::basic_string<char_type>& str) const
    {
        return {*this, str.data(), str.data() + str.size(), _ob_args, _fpack};
    }

#endif

    template <typename ... Args>
    _return_type operator()(const Args& ... args) const
    {
        return stringify::v0::detail::create_ob_and_write<OutputBuff>
            ( static_cast<const _reservation&>(*this)
            , _ob_args
            , _fpack
            , args... );
    }

private:

    template <class, class, class...>
    friend class dispatcher;

    constexpr dispatcher
        ( const _reservation& r
        , FPack&& fp
        , _obargs_tuple&& args )
        : _reservation(r)
        , _fpack(std::move(fp))
        , _ob_args(std::move(args))
    {
    }

    constexpr dispatcher
        ( const _reservation& r
        , const FPack& fp
        , const _obargs_tuple& args )
        : _reservation(r)
        , _fpack(fp)
        , _ob_args(args)
    {
    }

    FPack _fpack;
    std::tuple<OBInitArgs ...> _ob_args;
};


template <typename CharOut, typename FPack, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>( std::declval<FPack>()
                                       , std::declval<Arg>() ) );

// template <typename OutputBuff, typename ... Args>
// constexpr auto make_dispatcher(Args ... args)
// {
//     return stringify::v0::dispatcher< stringify::v0::facets_pack<>
//                                     , OutputBuff
//                                     , Args ... >
//         { stringify::v0::facets_pack<>{}
//         , std::forward<Args>(args)... };
// }

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
