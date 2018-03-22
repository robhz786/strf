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


class output_size_reservation_real
{
    std::size_t m_size = 0;
    enum {calculate_size, predefined_size, no_reserve} m_flag = calculate_size;

public:

    constexpr output_size_reservation_real() = default;

    constexpr output_size_reservation_real(const output_size_reservation_real&)
    = default;

    template <typename SizeCalculator, typename OutputWriter, typename ArgList>
    void reserve
        ( SizeCalculator& calc
        , OutputWriter& writer
        , const ArgList& args
        ) const
    {
        if (m_flag != no_reserve)
        {
            writer.reserve(m_flag == calculate_size ? calc.calculate_size(args): m_size);
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


template <typename OutputWriter>
using output_size_reservation
= typename std::conditional
    < stringify::v0::detail::has_reserve<OutputWriter>::value
    , output_size_reservation_real
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
class syntax_after_assembly_string
{
    using char_type = typename OutputWriter::char_type;
    using arglist_type = std::initializer_list<const stringify::v0::formatter<char_type>*>;

    using reservation_type
        = stringify::v0::detail::output_size_reservation<OutputWriter>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

public:

    syntax_after_assembly_string
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

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) error_code(const Args& ... args) const
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, {as_pointer(make_formatter(args)) ...});
        return writer.finish_error_code();
    }

    template <typename ... Args>
    decltype(auto) exception(const Args& ... args) const
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, {as_pointer(make_formatter(args)) ...});
        return writer.finish_exception();
    }

    std::size_t calculate_size(arglist_type args) const
    {
        stringify::v0::detail::asm_string_measurer<char_type, char_type> measurer{args};
        stringify::v0::detail::parse_asm_string(m_str, m_end, measurer);
        return measurer.result();
    }

private:

    template <typename Arg>
    auto make_formatter(const Arg& arg) const
    {
        return boost_stringify_make_formatter<char_type, FTuple>(m_ftuple, arg);
    }

    static const stringify::v0::formatter<char_type>*
    as_pointer(const stringify::v0::formatter<char_type>& fmt)
    {
        return &fmt;
    }

    void write(OutputWriter& owriter, const arglist_type& args) const
    {
        m_reservation.reserve(*this, owriter, args);
        stringify::v0::detail::asm_string_writer<char_type, char_type> asm_writer
        {m_ftuple, owriter, args};

        stringify::v0::detail::parse_asm_string(m_str, m_end, asm_writer);
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
class syntax_after_leading_expr
{
public:

    using char_type = typename OutputWriter::char_type;
    using arglist_type = std::initializer_list<const stringify::v0::formatter<char_type>*>;
    using asm_string = stringify::v0::detail::syntax_after_assembly_string
        <FTuple, OutputWriter, OutputWriterInitArgsTuple>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

public:

    constexpr syntax_after_leading_expr(FTuple&& ft, OutputWriterInitArgsTuple&& args)
        : m_ftuple(std::move(ft))
        , m_owinit(std::move(args))
    {
    }

    constexpr syntax_after_leading_expr(const FTuple& ft, const OutputWriterInitArgsTuple& args)
        : m_ftuple(ft)
        , m_owinit(args)
    {
    }

    constexpr syntax_after_leading_expr(const syntax_after_leading_expr&) = default;

    constexpr syntax_after_leading_expr(syntax_after_leading_expr&&) = default;

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::make_ftuple(m_ftuple, facets ...))
            , OutputWriter
            , OutputWriterInitArgsTuple
            >
            ( stringify::v0::make_ftuple(m_ftuple, facets ...)
            , m_owinit
            );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::ftuple<Facets...>& ft) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::make_ftuple(m_ftuple, ft))
            , OutputWriter
            , OutputWriterInitArgsTuple
            >
            ( stringify::v0::make_ftuple(m_ftuple, ft)
            , m_owinit
            );
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

    constexpr syntax_after_leading_expr facets(stringify::v0::ftuple<>) const &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr& facets(stringify::v0::ftuple<>) &
    {
        return *this;
    }

    constexpr syntax_after_leading_expr&& facets(stringify::v0::ftuple<>) &&
    {
        return std::move(*this);
    }

    constexpr syntax_after_leading_expr no_reserve() const &
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_no_reserve();
        return copy;
    }
    constexpr syntax_after_leading_expr no_reserve() const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_no_reserve();
        return copy;
    }
    constexpr syntax_after_leading_expr no_reserve() &
    {
        m_reservation.set_no_reserve();
        return *this;
    }
    constexpr syntax_after_leading_expr no_reserve() &&
    {
        m_reservation.set_no_reserve();
        return *this;
    }

    constexpr syntax_after_leading_expr reserve_auto() const &
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_reserve_auto();
        return copy;
    }
    constexpr syntax_after_leading_expr reserve_auto() const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_reserve_auto();
        return copy;
    }
    constexpr syntax_after_leading_expr reserve_auto() &
    {
        m_reservation.set_reserve_auto();
        return *this;
    }
    constexpr syntax_after_leading_expr reserve_auto() &&
    {
        m_reservation.set_reserve_auto();
        return *this;
    }

    constexpr syntax_after_leading_expr reserve(std::size_t size) const &
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_reserve_size(size);
        return copy;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) const &&
    {
        syntax_after_leading_expr copy = *this;
        copy.m_reservation.set_reserve_size(size);
        return copy;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) &
    {
        m_reservation.set_reserve_size(size);
        return *this;
    }
    constexpr syntax_after_leading_expr reserve(std::size_t size) &&
    {
        m_reservation.set_reserve_size(size);
        return *this;
    }

    std::size_t calculate_size(arglist_type args) const
    {
        std::size_t len = 0;
        for(const auto& arg : args)
        {
            len += arg->length();
        }
        return len;
    }

    asm_string operator()(const char_type* asm_str) const
    {
        return {m_ftuple, m_owinit, asm_str, m_reservation};
    }

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) error_code(const Args& ... args) const
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, {as_pointer(make_formatter(args)) ...});
        return writer.finish_error_code();
    }

    template <typename ... Args>
    decltype(auto) exception(const Args& ... args) const
    {
        output_writer_wrapper writer{m_owinit};
        write(writer, {as_pointer(make_formatter(args)) ...});
        return writer.finish_exception();
    }

private:

    static const stringify::v0::formatter<char_type>*
    as_pointer(const stringify::v0::formatter<char_type>& fmt)
    {
        return &fmt;
    }

    template <typename Arg>
    auto make_formatter(const Arg& arg) const
    {
        return boost_stringify_make_formatter<char_type, FTuple>(m_ftuple, arg);
    }

    void write(OutputWriter& writer, arglist_type args) const
    {
        m_reservation.reserve(*this, writer, args);
        for(auto it = args.begin(); it != args.end() && writer.good(); ++it)
        {
            (*it)->write(writer);
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
    -> stringify::v0::detail::syntax_after_leading_expr
        < stringify::v0::ftuple<>
        , OutputWriter
        , std::tuple<Args ...>
        >
{
    return {stringify::v0::ftuple<>{}, std::tuple<Args ...>{args ...}};
}

template <typename T>
constexpr auto fmt(const T& value)
{
    return boost_stringify_fmt(value);
}

template <typename T>
constexpr auto uphex(const T& value)
{
    return fmt(value).uphex();
}

template <typename T>
constexpr auto hex(const T& value)
{
    return fmt(value).hex();
}

template <typename T>
constexpr auto dec(const T& value)
{
    return fmt(value).dec();
}

template <typename T>
constexpr auto oct(const T& value)
{
    return fmt(value).oct();
}

template <typename T>
constexpr auto left(const T& value, int width)
{
    return fmt(value) < width;
}

template <typename T>
constexpr auto right(const T& value, int width)
{
    return fmt(value) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width)
{
    return fmt(value) % width;
}

template <typename T>
constexpr auto center(const T& value, int width)
{
    return fmt(value) ^ width;
}

template <typename T>
constexpr auto left(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) < width;
}

template <typename T>
constexpr auto right(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) % width;
}

template <typename T>
constexpr auto center(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) ^ width;
}

template <typename T, typename I>
constexpr auto multi(const T& value, I count)
{
    return fmt(value).multi(count);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
