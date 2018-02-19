#ifndef BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
#define BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/detail/assembly_string.hpp>
#include <tuple>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

class output_size_reservation_dummy
{

public:

    constexpr output_size_reservation_dummy() = default;

    constexpr output_size_reservation_dummy(const output_size_reservation_dummy&)
    = default;

    template <typename ... Args>
    constexpr void reserve(Args&& ...) const
    {
    }

    constexpr void set_reserve_size(std::size_t)
    {
    }

    constexpr void set_reserve_auto()
    {
    }

    constexpr void set_no_reserve()
    {
    }
};

class output_size_reservation_impl
{
    std::size_t m_size = 0;
    enum {calculate_size, predefined_size, no_reserve} m_flag = calculate_size;

public:

    constexpr output_size_reservation_impl() = default;

    constexpr output_size_reservation_impl(const output_size_reservation_impl&)
    = default;

    template <typename SizeCalculator, typename OutputWriter, typename ArgList>
    void reserve
        ( SizeCalculator& calc
        , OutputWriter& writer
        , const ArgList& args
        ) const
    {
        switch(m_flag)
        {
            case calculate_size:
                writer.reserve(calc.calculate_size(args));
                break;
            case predefined_size:
                writer.reserve(m_size);
                break;
            case no_reserve:
            default:
                ;
        }
    }

    constexpr void set_reserve_size(std::size_t size)
    {
        m_flag = predefined_size;
        m_size = size;
    }

    constexpr void set_reserve_auto()
    {
        m_flag = calculate_size;
    }

    constexpr void set_no_reserve()
    {
        m_flag = no_reserve;
    }
};


template <typename W>
static auto has_reserve_impl(W&& ow)
    -> decltype(ow.reserve(std::size_t{}), std::true_type{});

template <typename W>
static std::false_type has_reserve_impl(...);

template <typename W>
using has_reserve
= decltype(stringify::v0::detail::has_reserve_impl<W>(std::declval<W>()));

template <typename OutputWriter>
using output_size_reservation
= typename std::conditional
    < stringify::v0::detail::has_reserve<OutputWriter>::value
    , output_size_reservation_impl
    , output_size_reservation_dummy
    > :: type;


template <typename OutputWriter>
class output_writer_from_tuple: public OutputWriter
{
public:

    template <typename ... Args>
    output_writer_from_tuple(const std::tuple<Args...>& tp)
        : output_writer_from_tuple(tp, std::make_index_sequence<sizeof...(Args)>{})
    {
    }

private:

    template <typename ArgsTuple, std::size_t ... I>
    output_writer_from_tuple(const ArgsTuple& args, std::index_sequence<I...>)
        : OutputWriter(std::get<I>(args)...)
    {
    }

};


template
    < typename FTuple
    , typename OutputWriter
    , typename OutputWriterInitArgsTuple
    >
class assembly_string
{
    using char_type = typename OutputWriter::char_type;
    using arg_type = stringify::v0::input_arg<char_type, FTuple>;
    using arglist_type = std::initializer_list<arg_type>;
    using reservation_type
        = stringify::v0::detail::output_size_reservation<OutputWriter>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

public:

    assembly_string
        ( const FTuple& ft
        , const OutputWriterInitArgsTuple& owinit
        , const char_type* str
        , const reservation_type& res
        )
        : m_ftuple(ft)
        , m_owinit(owinit)
        , m_str(str)
        , m_end(str + std::char_traits<char_type>::length(str))
        , m_reservation(res)
    {
    }

    BOOST_STRINGIFY_NODISCARD decltype(auto) operator=(arglist_type args) &
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish();
    }

    BOOST_STRINGIFY_NODISCARD decltype(auto) operator=(arglist_type args) &&
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish();
    }

    decltype(auto) operator&=(arglist_type args) &
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish_throw();
    }

    decltype(auto) operator&=(arglist_type args) &&
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish_throw();
    }

    std::size_t calculate_size(arglist_type args) const
    {
        stringify::v0::detail::asm_string_measurer<char_type, FTuple>
            measurer{args, m_ftuple};
        stringify::v0::detail::parse_asm_string(m_str, m_end, measurer);
        return measurer.result();
    }

private:

    void write(OutputWriter& owriter, const arglist_type& args) const
    {
        m_reservation.reserve(*this, owriter, args);
        stringify::v0::detail::asm_string_writer<char_type, FTuple>
            writer{args, owriter, m_ftuple};
        stringify::v0::detail::parse_asm_string(m_str, m_end, writer);
    }

    const FTuple& m_ftuple;
    const OutputWriterInitArgsTuple& m_owinit;
    const char_type* const m_str;
    const char_type* const m_end;
    reservation_type m_reservation;
};


template
    < typename FTuple
    , typename OutputWriter
    , typename OutputWriterInitArgsTuple
    >
class args_handler
{
public:

    using char_type = typename OutputWriter::char_type;
    using arg_type =  stringify::v0::input_arg<char_type, FTuple>;
    using arglist_type = std::initializer_list<arg_type>;
    using asm_string = stringify::v0::detail::assembly_string
        <FTuple, OutputWriter, OutputWriterInitArgsTuple>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

public:

    constexpr args_handler(FTuple&& ft, OutputWriterInitArgsTuple&& args)
        : m_ftuple(std::move(ft))
        , m_owinit(std::move(args))
    {
    }

    constexpr args_handler(const FTuple& ft, const OutputWriterInitArgsTuple& args)
        : m_ftuple(ft)
        , m_owinit(args)
    {
    }

    constexpr args_handler(const args_handler&) = default;

    constexpr args_handler(args_handler&&) = default;

    template <typename ... Facets>
    constexpr auto with(const Facets& ... facets) const
    {
        return args_handler
            < decltype(stringify::v0::make_ftuple(m_ftuple, facets ...))
            , OutputWriter
            , OutputWriterInitArgsTuple
            >
            ( stringify::v0::make_ftuple(m_ftuple, facets ...)
            , m_owinit
            );
    }

    template <typename ... Facets>
    constexpr auto with(const stringify::v0::ftuple<Facets...>& ft) const
    {
        return args_handler
            < decltype(stringify::v0::make_ftuple(m_ftuple, ft))
            , OutputWriter
            , OutputWriterInitArgsTuple
            >
            ( stringify::v0::make_ftuple(m_ftuple, ft)
            , m_owinit
            );
    }

    constexpr args_handler with() const &
    {
        return *this;
    }

    constexpr args_handler& with() &
    {
        return *this;
    }

    constexpr args_handler&& with() &&
    {
        return std::move(*this);
    }

    constexpr args_handler with(stringify::v0::ftuple<>) const &
    {
        return *this;
    }

    constexpr args_handler& with(stringify::v0::ftuple<>) &
    {
        return *this;
    }

    constexpr args_handler&& with(stringify::v0::ftuple<>) &&
    {
        return std::move(*this);
    }

    constexpr args_handler no_reserve() const &
    {
        args_handler copy = *this;
        copy.m_reservation.set_no_reserve();
        return copy;
    }
    constexpr args_handler no_reserve() const &&
    {
        args_handler copy = *this;
        copy.m_reservation.set_no_reserve();
        return copy;
    }
    constexpr args_handler no_reserve() &
    {
        m_reservation.set_no_reserve();
        return *this;
    }
    constexpr args_handler no_reserve() &&
    {
        m_reservation.set_no_reserve();
        return *this;
    }

    constexpr args_handler reserve_auto() const &
    {
        args_handler copy = *this;
        copy.m_reservation.set_reserve_auto();
        return copy;
    }
    constexpr args_handler reserve_auto() const &&
    {
        args_handler copy = *this;
        copy.m_reservation.set_reserve_auto();
        return copy;
    }
    constexpr args_handler reserve_auto() &
    {
        m_reservation.set_reserve_auto();
        return *this;
    }
    constexpr args_handler reserve_auto() &&
    {
        m_reservation.set_reserve_auto();
        return *this;
    }

    constexpr args_handler reserve(std::size_t size) const &
    {
        args_handler copy = *this;
        copy.m_reservation.set_reserve_size(size);
        return copy;
    }
    constexpr args_handler reserve(std::size_t size) const &&
    {
        args_handler copy = *this;
        copy.m_reservation.set_reserve_size(size);
        return copy;
    }
    constexpr args_handler reserve(std::size_t size) &
    {
        m_reservation.set_reserve_size(size);
        return *this;
    }
    constexpr args_handler reserve(std::size_t size) &&
    {
        m_reservation.set_reserve_size(size);
        return *this;
    }

    constexpr args_handler&& operator()() &&
    {
        return std::move(*this);
    }

    constexpr args_handler operator()() const &&
    {
        return *this;
    }

    constexpr args_handler& operator()() &
    {
        return *this;
    }

    constexpr args_handler operator()() const &
    {
        return *this;
    }

    std::size_t calculate_size(arglist_type args) const
    {
        std::size_t len = 0;
        for(const auto& arg : args)
        {
            len += arg.length(m_ftuple);
        }
        return len;
    }

    asm_string operator[](const char_type* asm_str) &&
    {
        return {m_ftuple, m_owinit, asm_str, m_reservation};
    }

    asm_string operator[](const char_type* asm_str) const &&
    {
        return {m_ftuple, m_owinit, asm_str, m_reservation};
    }

    asm_string operator[](const char_type* asm_str) &
    {
        return {m_ftuple, m_owinit, asm_str, m_reservation};
    }

    asm_string operator[](const char_type* asm_str) const &
    {
        return {m_ftuple, m_owinit, asm_str, m_reservation};
    }

    BOOST_STRINGIFY_NODISCARD
    constexpr decltype(auto) operator=(arglist_type args) &&
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish();
    }

    BOOST_STRINGIFY_NODISCARD
    constexpr decltype(auto) operator=(arglist_type args) &
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish();
    }

    constexpr decltype(auto) operator&=(arglist_type args) &&
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish_throw();
    }

    constexpr decltype(auto) operator&=(arglist_type args) &
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, args);
        return writer.finish_throw();
    }

private:

    void write(OutputWriter& writer,arglist_type args) const
    {
        m_reservation.reserve(*this, writer, args);
        for(auto it = args.begin(); it != args.end() && writer.good(); ++it)
        {
            (*it).write(writer, m_ftuple);
        }
    }

    FTuple m_ftuple;
    OutputWriterInitArgsTuple m_owinit;
    stringify::v0::detail::output_size_reservation<OutputWriter>
    m_reservation;
};

} // namespace detail

template <typename OutputWriter, typename ... Args>
constexpr auto make_args_handler(Args ... args)
    -> stringify::v0::detail::args_handler
        < stringify::v0::ftuple<>
        , OutputWriter
        , std::tuple<Args ...>
        >
{
    return {stringify::v0::ftuple<>{}, std::tuple<Args ...>{args ...}};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
