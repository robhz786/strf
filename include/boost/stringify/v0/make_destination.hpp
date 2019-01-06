#ifndef BOOST_STRINGIFY_V0_MAKE_DESTINATION_HPP
#define BOOST_STRINGIFY_V0_MAKE_DESTINATION_HPP

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


template <typename OutputWriter>
using output_size_reservation
= typename std::conditional
    < stringify::v0::detail::has_reserve<OutputWriter>::value
    , output_size_reservation_real
    , output_size_reservation_dummy
    > :: type;



template <typename OutputWriter>
class output_writer_from_tuple
{
public:

    template <typename ... Args>
    output_writer_from_tuple(const std::tuple<Args...>& tp)
        : output_writer_from_tuple(tp, std::make_index_sequence<sizeof...(Args)>{})
    {
    }

    OutputWriter& get()
    {
        return _out;
    }

private:

    OutputWriter _out;

    template <typename ArgsTuple, std::size_t ... I>
    output_writer_from_tuple
        ( const ArgsTuple& args
        , std::index_sequence<I...>)
        : _out(std::get<I>(args)...)
    {
    }
};

template
    < typename FPack
    , typename OutputWriter
    , typename OutputWriterInitArgsTuple
    >
class syntax_after_leading_expr
    : private stringify::v0::detail::output_size_reservation<OutputWriter>

{
    using _char_type = typename OutputWriter::char_type;

    using _output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

    using _return_type
        = decltype(std::declval<OutputWriter>().finish((_char_type*)(0)));

    using _arglist_type
        = std::initializer_list<const stringify::v0::printer<_char_type>*>;

    using _reservation
        = stringify::v0::detail::output_size_reservation<OutputWriter>;

public:

    constexpr syntax_after_leading_expr
        ( FPack&& fp
        , OutputWriterInitArgsTuple&& args )
        : _fpack(std::move(fp))
        , _owinit(std::move(args))
    {
    }

    constexpr syntax_after_leading_expr
        ( const FPack& fp
        , const OutputWriterInitArgsTuple& args )
        : _fpack(fp)
        , _owinit(args)
    {
    }

    constexpr syntax_after_leading_expr(const syntax_after_leading_expr&)
        = default;

    constexpr syntax_after_leading_expr(syntax_after_leading_expr&&) = default;

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::pack(_fpack, facets ...))
            , OutputWriter
            , OutputWriterInitArgsTuple >
            ( *this
            , stringify::v0::pack(_fpack, facets ...)
            , _owinit );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::facets_pack<Facets...>& fp) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::pack(_fpack, fp))
            , OutputWriter
            , OutputWriterInitArgsTuple >
            ( *this
            , stringify::v0::pack(_fpack, fp)
            , _owinit );
    }

    constexpr syntax_after_leading_expr facets() const &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr& facets() &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr&& facets() &&
    {
        return std::move(*this);
    }

    constexpr syntax_after_leading_expr facets(stringify::v0::facets_pack<>) const &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr& facets(stringify::v0::facets_pack<>) &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr&& facets(stringify::v0::facets_pack<>) &&
    {
        return std::move(*this);
    }

    constexpr syntax_after_leading_expr no_reserve() const &
    {
        syntax_after_leading_expr copy = *this;
        copy.set_no_reserve();
        return copy;
    }
    constexpr syntax_after_leading_expr no_reserve() const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.set_no_reserve();
        return copy;
    }
    constexpr syntax_after_leading_expr no_reserve() &
    {
        this->set_no_reserve();
        return *this;
    }
    constexpr syntax_after_leading_expr no_reserve() &&
    {
        this->set_no_reserve();
        return *this;
    }

    constexpr syntax_after_leading_expr reserve_calc() const &
    {
        syntax_after_leading_expr copy = *this;
        copy.set_reserve_calc();
        return copy;
    }
    constexpr syntax_after_leading_expr reserve_calc() const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.set_reserve_calc();
        return copy;
    }
    constexpr syntax_after_leading_expr reserve_calc() &
    {
        this->set_reserve_calc();
        return *this;
    }
    constexpr syntax_after_leading_expr reserve_calc() &&
    {
        this->set_reserve_calc();
        return *this;
    }

    constexpr syntax_after_leading_expr reserve(std::size_t size) const &
    {
        syntax_after_leading_expr copy = *this;
        copy.set_reserve_size(size);
        return copy;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.set_reserve_size(size);
        return copy;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) &
    {
        this->set_reserve_size(size);
        return *this;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) &&
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

    template <typename ... Args>
    _return_type as
        ( const std::basic_string_view<_char_type>& str
        , const Args& ... args ) const
    {

        return asm_write
            ( this->has_reserve()
            , str.begin()
            , str.end()
            , {_as_pointer(make_printer<_char_type, FPack>(_fpack, args))... } );
    }

#else

    template <typename ... Args>
    _return_type as(const _char_type* str, const Args& ... args) const
    {
        return _asm_write
            ( this->has_reserve()
            , str
            , str + std::char_traits<_char_type>::length(str)
            , {_as_pointer(make_printer<_char_type, FPack>(_fpack, args))... } );
    }

    template <typename Traits, typename ... Args>
    _return_type as
        ( const std::basic_string<_char_type, Traits>& str
        , const Args& ... args ) const
    {
        return _asm_write
            ( this->has_reserve()
            , str.data()
            , str.data() + str.size()
            , {_as_pointer(make_printer<_char_type, FPack>(_fpack, args))... } );
    }

#endif

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) operator()(const Args& ... args) const
    {
        return _write
            ( this->has_reserve()
            , make_printer<_char_type, FPack>(_fpack, args) ... );
    }

private:

    template <class, class, class>
    friend class syntax_after_leading_expr;

    constexpr syntax_after_leading_expr
        ( const _reservation& r
        , FPack&& fp
        , OutputWriterInitArgsTuple&& args )
        : _reservation(r)
        , _fpack(std::move(fp))
        , _owinit(std::move(args))
    {
    }

    constexpr syntax_after_leading_expr
        ( const _reservation& r
        , const FPack& fp
        , const OutputWriterInitArgsTuple& args )
        : _reservation(r)
        , _fpack(fp)
        , _owinit(args)
    {
    }

    static const stringify::v0::printer<_char_type>*
    _as_pointer(const stringify::v0::printer<_char_type>& p)
    {
        return &p;
    }

    _return_type _asm_write
        ( std::true_type
        , const _char_type* str
        , const _char_type* str_end
        , _arglist_type args) const
    {
        const auto& enc
            = get_facet<stringify::v0::encoding_category<_char_type>, void>(_fpack);
        const auto policy
            = get_facet<stringify::v0::asm_invalid_arg_category, void>(_fpack);

        _output_writer_wrapper writer{_owinit};

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

        auto buff = writer.get().start();
        bool no_error = stringify::v0::detail::asm_string_write
            ( str, str_end, args, buff, writer.get(), enc, policy );
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish(buff.it);
    }

    _return_type _asm_write
        ( std::false_type
        , const _char_type* str
        , const _char_type* str_end
        , _arglist_type args ) const
    {
        _output_writer_wrapper writer{_owinit};
        auto buff = writer.get().start();

        const auto& enc
            = get_facet<stringify::v0::encoding_category<_char_type>, void>(_fpack);
        const auto policy
            = get_facet<stringify::v0::asm_invalid_arg_category, void>(_fpack);

        bool no_error = stringify::v0::detail::asm_string_write
            ( str, str_end, args, buff, writer.get(), enc, policy );
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish(buff.it);
    }


    template <typename ... Args>
    _return_type _write(std::true_type, const Args& ... args) const
    {
        _output_writer_wrapper writer{_owinit};

        auto res_size = this->must_calculate_size()
            ? this->calculate_size(args ...)
            : this->get_size_to_reserve();

        if (res_size != 0)
        {
            writer.get().reserve(res_size);
        }
        auto buff = writer.get().start();
        bool no_error = _write_args(buff, writer.get(), args...);
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish(buff.it);
    }


    template <typename ... Args>
    _return_type _write(std::false_type, const Args& ... args) const
    {
        _output_writer_wrapper writer{_owinit};

        auto buff = writer.get().start();
        bool no_error = _write_args(buff, writer.get(), args...);
        BOOST_ASSERT(no_error == ! writer.get().has_error());
        (void) no_error;
        return writer.get().finish(buff.it);
    }

    template <typename Arg>
    static bool _write_args
        ( stringify::v0::output_buffer<_char_type>& b
        , OutputWriter& writer
        , const Arg& arg )
    {
        return arg.write(b, writer);
    }

    template <typename Arg, typename ... Args>
    static bool _write_args
        ( stringify::v0::output_buffer<_char_type>& b
        , OutputWriter& writer
        , const Arg& arg
        , const Args& ... args )
    {
        return arg.write(b, writer) && _write_args(b, writer, args ...);
    }

    static bool _write_args
        ( stringify::v0::output_buffer<_char_type>& b
        , OutputWriter& )
    {
        return true;
    }

    FPack _fpack;
    OutputWriterInitArgsTuple _owinit;
};

} // namespace detail

template <typename CharOut, typename FPack, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>( std::declval<FPack>()
                                       , std::declval<Arg>() ) );

template <typename OutputWriter, typename ... Args>
constexpr auto make_destination(Args ... args)
    -> stringify::v0::detail::syntax_after_leading_expr
        < stringify::v0::facets_pack<>
        , OutputWriter
        , std::tuple<Args ...> >
{
    return {stringify::v0::facets_pack<>{}, std::tuple<Args ...>{args ...}};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_MAKE_DESTINATION_HPP
