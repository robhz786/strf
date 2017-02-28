#ifndef BOOST_STRINGIFY_V1_ARGS_HANDLER_HPP
#define BOOST_STRINGIFY_V1_ARGS_HANDLER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/input_arg.hpp>
#include <boost/stringify/v1/ftuple.hpp>
#include <tuple>

namespace boost {
namespace stringify {
inline namespace v1 {
namespace detail {

template <typename ArgsHandlerImpl, typename ftuple_type, typename output_writer>
class args_handler_base
{
    
    // using output_writer = typename ArgsHandlerImpl::output_writer;
    // using ftuple_type = typename ArgsHandlerImpl::ftuple_type;
    using char_type = typename output_writer::char_type;
    using arg_type =  boost::stringify::v1::input_arg<char_type, output_writer, ftuple_type>;
  
    template <typename Arg1, typename ... Args>
    std::size_t length(Arg1 && arg1, Args && ... args) const
    {
        return arg1.length(get_ftuple()) + length(args...);
    }

    std::size_t length() const
    {
        return 0;
    }

    struct matching_strength_1 {};
    struct matching_strength_2 : matching_strength_1 {};
    
    template <typename output_writer2, typename ... Args>
    auto reserve
        ( const matching_strength_2&
        , output_writer2& writer
        , const Args & ... args
        ) const
        -> decltype(writer.reserve(std::size_t()), void())
    {
        writer.reserve(1 + length(args...));
    }

    
    template <typename output_writer2, typename ... Args>
    void reserve
        ( const matching_strength_1&
        , output_writer2&
        , const Args & ... args
        ) const
    {
    }

    template <typename ... Args>
    void reserve(output_writer& writer, const Args & ... args) const
    {
        reserve(matching_strength_2(), writer, args...); 
    }
    
    template <typename OW>
    auto write_inilist
        ( const matching_strength_2&
        , const std::initializer_list<arg_type>& lst
        , OW& writer
        ) const
       -> decltype(writer.reserve(std::size_t()), void())
    {
        std::size_t len = 0;
        for(auto arg : lst)
        {
            len += arg.length();
        }
        writer.reserve(len + 1);
        decltype(auto) fmt = get_ftuple();
        for(auto arg : lst)
        {
            arg.write(writer, fmt);
        }
    }

    template <typename OW>
    auto write_inilist
        ( const matching_strength_1&
        , const std::initializer_list<arg_type>& lst
        , OW& writer  
        ) const
    {
        decltype(auto) fmt = get_ftuple();
        for(auto arg : lst)
        {
            arg.write(writer, fmt);
        }
    }

    void do_write(output_writer&, const ftuple_type) const
    {

    }

    template <typename Arg1, typename ... Args>
    void do_write
        ( output_writer& out
        , const ftuple_type& fmt
        , const Arg1& arg1
        , const Args& ... args
        ) const
    {
        arg1.write(out, fmt);
        do_write(out, fmt, args ...);
    }

    template <typename ... Args>
    decltype(auto) write(const Args & ... args) const
    {
        decltype(auto) owriter
            = static_cast<const ArgsHandlerImpl&&>(*this).get_writer();
        reserve(owriter, args ...);
        do_write(owriter, get_ftuple(), args ...);
        return owriter.finish();
    }

    
public:

    decltype(auto) operator[](const std::initializer_list<arg_type>& lst) const
    {
        decltype(auto) owriter
            = static_cast<const ArgsHandlerImpl&&>(*this).get_writer();
        write_inilist(matching_strength_2(), lst, owriter);
        return owriter.finish();
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

    decltype(auto) get_ftuple() const
    {
        return static_cast<const ArgsHandlerImpl&>(*this).get_ftuple();
    }
    
};


template <typename OutputWriter, typename ... Args>
class output_writer_instantiator
{
public:
    constexpr output_writer_instantiator(output_writer_instantiator&& ) = default;

    constexpr output_writer_instantiator(const output_writer_instantiator& ) = default;
    
    constexpr output_writer_instantiator(Args ... args)
        : m_args(std::forward<Args>(args)...)
    {
    }

    OutputWriter get_writer() const
    {
        using index_sequence = std::make_index_sequence<sizeof...(Args)>;
        return std::move(do_instantiate(index_sequence()));
    }
    
private:

    template<std::size_t ... I>
    OutputWriter do_instantiate(std::index_sequence<I...>) const
    {
        return std::move(OutputWriter(get<I>(m_args) ...));
    }
            
    std::tuple<Args...> m_args;
};

template <typename OutputWriter>
class output_writer_instantiator<OutputWriter>
{
public:
    constexpr output_writer_instantiator(output_writer_instantiator&& ) = default;

    constexpr output_writer_instantiator(const output_writer_instantiator& ) = default;
    
    constexpr output_writer_instantiator() = default;

    OutputWriter get_writer() const
    {
        return std::move(OutputWriter());
    }
};


template <typename FTuple, typename OutputWriter, typename ... OutputWriterArgs>
class args_handler
    : public boost::stringify::v1::detail::args_handler_base
        < args_handler<FTuple, OutputWriter, OutputWriterArgs...>
        , FTuple
        , OutputWriter
        >  
    , private boost::stringify::v1::detail::output_writer_instantiator
        <OutputWriter, OutputWriterArgs...>
{
    friend class boost::stringify::v1::detail::args_handler_base
        < args_handler, FTuple, OutputWriter>;

    using output_writer_instantiator
        = boost::stringify::v1::detail::output_writer_instantiator
        <OutputWriter, OutputWriterArgs...>;    
  
public:

    using ftuple_type = FTuple;
    using output_writer = OutputWriter;

    constexpr args_handler(args_handler&& x) = default;
    
    constexpr args_handler(FTuple&& ft, const output_writer_instantiator& owi)
        : output_writer_instantiator(owi)
        , m_ftuple(std::move(ft))
    {
    }
 
    constexpr const args_handler& with() const
    {
        return *this;
    }

    constexpr const args_handler& with(boost::stringify::v1::ftuple<>)
    {
        return *this;
    }       

    template <typename ... Formaters>
    constexpr auto with(const Formaters& ... formaters) const
    {
        return args_handler
            < decltype(boost::stringify::v1::make_ftuple(m_ftuple, formaters ...))
            , OutputWriter
            , OutputWriterArgs ...
            >
            (boost::stringify::v1::make_ftuple(m_ftuple, formaters ...), *this);
    }
    
    template <typename ... Formaters>
    constexpr auto with(const boost::stringify::v1::ftuple<Formaters...>& ft) const
    {
        return args_handler
            < boost::stringify::v1::ftuple<Formaters...>
            , OutputWriter
            , OutputWriterArgs ...
            >
            (boost::stringify::v1::make_ftuple(m_ftuple, ft), *this);
    }

    const FTuple& get_ftuple() const
    {
        return m_ftuple;
    }

private:

    const FTuple m_ftuple;
    
};


template <typename OutputWriter, typename ... OutputWriterArgs>
class args_handler
    < boost::stringify::v1::ftuple<>
    , OutputWriter
    , OutputWriterArgs...
    >
    : public boost::stringify::v1::detail::args_handler_base
        < args_handler
             < boost::stringify::v1::ftuple<>
             , OutputWriter
             , OutputWriterArgs...
             >
        , boost::stringify::v1::ftuple<>
        , OutputWriter
        >
    , private boost::stringify::v1::detail::output_writer_instantiator
        <OutputWriter, OutputWriterArgs...>
    , private boost::stringify::v1::ftuple<>
{
    friend class boost::stringify::v1::detail::args_handler_base
        < args_handler
        , boost::stringify::v1::ftuple<>
        , OutputWriter
        >;

    using output_writer_instantiator
        = boost::stringify::v1::detail::output_writer_instantiator
        <OutputWriter, OutputWriterArgs...>;    

public:

    using ftuple_type = boost::stringify::v1::ftuple<>;
    using output_writer = OutputWriter;

    constexpr args_handler(args_handler&& x) = default;
    
    constexpr args_handler(OutputWriterArgs ... args)
        : output_writer_instantiator(std::forward<OutputWriterArgs>(args)...)
    {
    }

    constexpr const args_handler& with() const
    {
        return *this;
    }

    constexpr const args_handler& with(boost::stringify::v1::ftuple<>) const
    {
        return *this;
    }
    
    template <typename ... Formaters>
    constexpr auto with(const Formaters& ... formaters) const
    {
        return args_handler
            < decltype(boost::stringify::v1::make_ftuple(formaters ...))
            , OutputWriter
            , OutputWriterArgs ...
            >
            ( boost::stringify::v1::make_ftuple(formaters ...)
            , *this
            );
    }
    
    template <typename ... Formaters>
    constexpr auto with(const boost::stringify::v1::ftuple<Formaters...>& ft) const
    {
        return args_handler
            < boost::stringify::v1::ftuple<Formaters...>
            , OutputWriter
            , OutputWriterArgs ...
            >
            ( std::move(ft)
            , *this
            );
    }

    constexpr const boost::stringify::v1::ftuple<>& get_ftuple() const
    {
        return *this;
    }
};


} // namespace detail

template <typename OutputWriter, typename ... Args>
constexpr auto make_args_handler(Args ... args)
{
    using args_handler_type
        = boost::stringify::v1::detail::args_handler
            < boost::stringify::v1::ftuple<>
            , OutputWriter
            , Args ...
            >;

    return std::move(args_handler_type(args...));
}

} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_ARGS_HANDLER_HPP

