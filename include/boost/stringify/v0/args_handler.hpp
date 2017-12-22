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

public:

    assembly_string
        ( const FTuple& ft
        , const OutputWriterInitArgsTuple& owinit
        , const char_type* str
        )
        : m_ftuple(ft)
        , m_owinit(owinit)
        , m_str(str)
        , m_end(str + std::char_traits<char_type>::length(str))
    {
    }

    decltype(auto) operator()(arglist_type args) const
    {
        stringify::v0::detail::output_writer_from_tuple<OutputWriter> writer{m_owinit};
        reserve(writer, args);
        write(writer, args);
        return writer.finish();
    }

    decltype(auto) operator=(arglist_type args) &
    {
        return operator() (args);
    }

    decltype(auto) operator=(arglist_type args) &&
    {
        return operator() (args);
    }

private:

    template <typename W>
    void reserve(std::true_type, W& writer, const arglist_type& args) const
    {
        stringify::v0::detail::asm_string_measurer<char_type, FTuple>
            measurer{args, m_ftuple};
        stringify::v0::detail::parse_asm_string(m_str, m_end, measurer);
        writer.reserve(measurer.result());
    }

    template <typename W>
    void reserve(std::false_type, W&, const arglist_type&) const
    {
    }

    void reserve(OutputWriter& owriter, const arglist_type& args) const
    {
        stringify::v0::detail::has_reserve<OutputWriter> has_reserve;
        reserve(has_reserve, owriter, args);
    }

    void write(OutputWriter& owriter, const arglist_type& args) const
    {
        stringify::v0::detail::asm_string_writer<char_type, FTuple>
            writer{args, owriter, m_ftuple};
        stringify::v0::detail::parse_asm_string(m_str, m_end, writer);
    }

    const FTuple& m_ftuple;
    const OutputWriterInitArgsTuple& m_owinit;
    const char_type* const m_str;
    const char_type* const m_end;
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

    decltype(auto) operator()(arglist_type lst) const
    {
        return process_arg_list(lst);
    }

    constexpr decltype(auto) operator=(arglist_type lst) &&
    {
        return process_arg_list(lst);
    }

    constexpr decltype(auto) operator=(arglist_type lst) &
    {
        return process_arg_list(lst);
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

    asm_string operator[](const char_type* asm_str) &&
    {
        return {m_ftuple, m_owinit, asm_str};
    }

    asm_string operator[](const char_type* asm_str) const &&
    {
        return {m_ftuple, m_owinit, asm_str};
    }

    asm_string operator[](const char_type* asm_str) &
    {
        return {m_ftuple, m_owinit, asm_str};
    }

    asm_string operator[](const char_type* asm_str) const &
    {
        return {m_ftuple, m_owinit, asm_str};
    }

private:

    template <typename W>
    void reserve(std::false_type, W&, const arglist_type&) const
    {
    }

    template <typename W>
    void reserve(std::true_type, W& writer, const arglist_type& lst) const
    {
        std::size_t len = 0;
        for(const auto& arg : lst)
        {
            len += arg.length(m_ftuple);
        }
        writer.reserve(len);
    }

    decltype(auto) process_arg_list(arglist_type lst) const
    {
        stringify::v0::detail::output_writer_from_tuple<OutputWriter> writer{m_owinit};
        stringify::v0::detail::has_reserve<OutputWriter> has_reserve;
        reserve(has_reserve, writer, lst);
        for(auto it = lst.begin(); it != lst.end() && writer.good(); ++it)
        {
            (*it).write(writer, m_ftuple);
        }
        return writer.finish();
    }

    FTuple m_ftuple;
    OutputWriterInitArgsTuple m_owinit;
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

