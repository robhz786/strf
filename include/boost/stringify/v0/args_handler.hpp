#ifndef BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
#define BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_arg.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <tuple>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

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
    < typename output_writer
    , typename ftuple_type
    , typename init_args_tuple_type
    >
class args_handler
{
    using char_type = typename output_writer::char_type;
    using arg_type =  stringify::v0::input_arg<char_type, ftuple_type>;
    using arg_list_type = std::initializer_list<arg_type>;

    template <typename W>
    static auto has_reserve_impl(W&& ow)
        -> decltype(ow.reserve(std::size_t{}), std::true_type{});

    template <typename W>
    static std::false_type has_reserve_impl(...);

    using has_reserve
    = decltype(has_reserve_impl<output_writer>(std::declval<output_writer>()));

    template <typename W>
    void reserve_lst(std::true_type, W& writer, const arg_list_type& lst) const
    {
        std::size_t len = 1;
        for(const auto& arg : lst)
        {
            len += arg.length(m_ftuple);
        }
        writer.reserve(len);
    }

    template <typename W>
    void reserve_lst(std::false_type, W&, const arg_list_type&) const
    {
    }

    template <typename Arg1, typename ... Args>
    std::size_t length(Arg1 && arg1, Args && ... args) const
    {
        return arg1.length(m_ftuple) + length(args...);
    }

    std::size_t length() const
    {
        return 0;
    }

    template <typename W, typename ... Args>
    auto reserve(std::true_type, W& writer, const Args & ... args) const
    {
        writer.reserve(1 + length(args...));
    }

    template <typename W, typename ... Args>
    void reserve(std::false_type, W&, const Args & ...) const
    {
    }

    void do_write(output_writer&) const
    {

    }

    template <typename Arg, typename ... Args>
    void do_write(output_writer& out, const Arg& a1, const Args& ... args) const
    {
        a1.write(out, m_ftuple);
        do_write(out, args ...);
    }

    template <typename ... Args>
    decltype(auto) write(const Args & ... args) const
    {
        output_writer_from_tuple<output_writer> writer(this->m_args);
        reserve(has_reserve{}, writer, args ...);
        do_write(writer, args ...);
        return writer.finish();
    }

public:

    constexpr args_handler(ftuple_type&& ft, init_args_tuple_type&& args)
        : m_ftuple(std::move(ft))
        , m_args(std::move(args))
    {
    }

    constexpr args_handler(const ftuple_type& ft, const init_args_tuple_type& a)
        : m_ftuple(ft)
        , m_args(a)
    {
    }

    constexpr const args_handler& with() const
    {
        return *this;
    }

    constexpr const args_handler& with(stringify::v0::ftuple<>) const
    {
        return *this;
    }

    template <typename ... Facets>
    constexpr auto with(const Facets& ... facets) const
    {
        return args_handler
            < output_writer
            , decltype(stringify::v0::make_ftuple(m_ftuple, facets ...))
            , init_args_tuple_type
            >
            ( stringify::v0::make_ftuple(m_ftuple, facets ...)
            , m_args
            );
    }

    template <typename ... Facets>
    constexpr auto with(const stringify::v0::ftuple<Facets...>& ft) const
    {
        return args_handler
            < output_writer
            , decltype(stringify::v0::make_ftuple(m_ftuple, ft))
            , init_args_tuple_type
            >
            ( stringify::v0::make_ftuple(m_ftuple, ft)
            , m_args
            );
    }

    decltype(auto) operator[](const std::initializer_list<arg_type>& lst) const
    {
        output_writer_from_tuple<output_writer> writer(this->m_args);
        reserve_lst(has_reserve{}, writer, lst);
        for(auto it = lst.begin(); it != lst.end() && writer.good(); ++it)
        {
            (*it).write(writer, m_ftuple);
        }
        return writer.finish();
    }

    decltype(auto) operator()() const
    {
        return write();
    }

    decltype(auto) operator()(arg_type a1) const
    {
        return write(a1);
    }

    decltype(auto) operator() (arg_type a1, arg_type a2) const
    {
        return write(a1, a2);
    }

    decltype(auto) operator() (arg_type a1, arg_type a2, arg_type a3) const
    {
        return write(a1, a2, a3);
    }

    decltype(auto) operator()
        (arg_type a1, arg_type a2, arg_type a3, arg_type a4) const
    {
        return write(a1, a2, a3, a4);
    }

    decltype(auto) operator()
        (arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5) const
    {
        return write(a1, a2, a3, a4, a5);
    }

    decltype(auto) operator()
        ( arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5
        , arg_type a6
        ) const
    {
        return write(a1, a2, a3, a4, a5, a6);
    }

    decltype(auto) operator()
        ( arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5
        , arg_type a6, arg_type a7
        ) const
    {
        return write(a1, a2, a3, a4, a5, a6, a7);
    }

    decltype(auto) operator()
        ( arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5
        , arg_type a6, arg_type a7, arg_type a8
        ) const
    {
        return write(a1, a2, a3, a4, a5, a6, a7, a8);
    }

    decltype(auto) operator()
        ( arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5
        , arg_type a6, arg_type a7, arg_type a8, arg_type a9
        ) const
    {
        return write(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }

    decltype(auto) operator()
        ( arg_type a1, arg_type a2, arg_type a3, arg_type a4, arg_type a5
        , arg_type a6, arg_type a7, arg_type a8, arg_type a9, arg_type a10
        ) const
    {
        return write(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }

private:

    ftuple_type m_ftuple;
    init_args_tuple_type m_args;
};

} // namespace detail

template <typename OutputWriter, typename ... Args>
constexpr auto make_args_handler(Args ... args)
    -> stringify::v0::detail::args_handler
            < OutputWriter
            , stringify::v0::ftuple<>
            , std::tuple<Args ...>
            >
{
    return {stringify::v0::ftuple<>{}, std::tuple<Args ...>{args ...}};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP

