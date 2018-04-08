#ifndef BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
#define BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/assembly_string.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <tuple>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

// template
//     < typename CharT
//     , typename FTuple
//     , typename InputArg
//     >
// auto get_input_traits_helper(int)
//     -> decltype(stringify_get_input_traits(std::declval<const InputArg&>()));

// template
//     < typename CharT
//     , typename FTuple
//     , typename InputArg
//     >
// auto get_input_traits_helper(...)
//     -> typename stringify::v0::get_input_traits<InputArg>::type;

// template
//     < typename CharT
//     , typename FTuple
//     , typename InputArg
//     >
// using input_traits = decltype(get_input_traits_helper<CharT, FTuple, InputArg>(0));


// template < typename CharT, typename FTuple, typename InputArg>
// inline auto make_printer(const FTuple& ft, const InputArg& arg)
// {
//     using input_traits = decltype(stringify_get_input_traits(arg));
//     return input_traits::template make_printer<CharT, FTuple>(ft, arg);
// }

// template < typename CharT, typename FTuple, typename InputArg>
// inline auto make_printer(const FTuple& ft, const InputArg& arg)
// {

//     using traits = stringify::v0::input_traits<CharT, FTuple, InputArg>;
//     return input_traits::template make_printer<CharT, FTuple>(ft, arg);
// }


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
            writer.reserve
                ( m_flag == calculate_size
                ? calc.calculate_size(writer, args)
                : m_size );
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
class output_writer_from_tuple
{
public:

    template <typename FTuple, typename ... Args>
    output_writer_from_tuple
        ( const FTuple& ft
        , const std::tuple<Args...>& tp
        )
        : output_writer_from_tuple(ft, tp, std::make_index_sequence<sizeof...(Args)>{})
    {
    }

    OutputWriter& get()
    {
        return m_out;
    }
    
private:

    OutputWriter m_out;
    
    using CharOut = typename OutputWriter::char_type;
    using Init = typename stringify::v0::output_writer_init<CharOut>;
    
    template <typename FTuple, typename ArgsTuple, std::size_t ... I>
    output_writer_from_tuple
        ( const FTuple& ft
        , const ArgsTuple& args
        , std::index_sequence<I...>)
        : m_out(Init{ft}, std::get<I>(args)...)
    {
    }

};

template
    < typename CharIn
    , typename FTuple
    , typename OutputWriter
    , typename OutputWriterInitArgsTuple
    >
class syntax_after_assembly_string
{
    using CharOut = typename OutputWriter::char_type;
    using arglist_type = std::initializer_list<const stringify::v0::printer<CharOut>*>;

    using reservation_type
        = stringify::v0::detail::output_size_reservation<OutputWriter>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

    using input_tag = stringify::v0::asm_string_input_tag<CharIn>;

public:

    syntax_after_assembly_string
        ( const FTuple& ft
        , const OutputWriterInitArgsTuple& owinit
        , const CharIn* str
        , const CharIn* str_end
        , const reservation_type& res
        , bool sanitise = false
        )
        : m_ftuple(ft)
        , m_owinit(owinit)
        , m_str(str)
        , m_end(str_end)
        , m_reservation(res)
        , m_sanitise(sanitise)
    {
    }

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) error_code(const Args& ... args) const
    {
        output_writer_wrapper dest{m_ftuple, m_owinit};
        write(dest.get(), {as_pointer(mk_printer<Args>(dest.get(), args)) ...});
        return dest.get().finish_error_code();
    }

    template <typename ... Args>
    decltype(auto) exception(const Args& ... args) const
    {
        output_writer_wrapper dest{m_ftuple, m_owinit};
        write(dest.get(), {as_pointer(mk_printer<Args>(dest.get(), args)) ...});
        return dest.get().finish_exception();
    }

    std::size_t calculate_size(OutputWriter& dest, arglist_type args) const
    {
        stringify::v0::detail::asm_string_measurer<CharIn, CharOut> measurer
        { dest, m_ftuple, args, m_sanitise };
        stringify::v0::detail::parse_asm_string(m_str, m_end, measurer);
        return measurer.result();
    }

private:

    template <typename FCategory>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<FCategory, input_tag>();
    }

    template <typename Arg>
    auto mk_printer
        ( stringify::v0::output_writer<CharOut>& dest
        , const Arg& arg
        ) const
    {
        return stringify_make_printer<CharOut, FTuple>(dest, m_ftuple, arg);
    }

    static const stringify::v0::printer<CharOut>*
    as_pointer(const stringify::v0::printer<CharOut>& p)
    {
        return &p;
    }

    void write(OutputWriter& owriter, const arglist_type& args) const
    {
        m_reservation.reserve(*this, owriter, args);
        stringify::v0::detail::asm_string_writer<CharIn, CharOut> asm_writer
        { owriter, m_ftuple, args, m_sanitise };

        stringify::v0::detail::parse_asm_string(m_str, m_end, asm_writer);
    }

    const FTuple& m_ftuple;
    const OutputWriterInitArgsTuple& m_owinit;
    //output_writer_wrapper m_writer;
    const CharIn* const m_str;
    const CharIn* const m_end;
    reservation_type m_reservation;
    bool m_sanitise;
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
    using arglist_type = std::initializer_list<const stringify::v0::printer<char_type>*>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

    template <typename CharIn>
    using asm_string = stringify::v0::detail::syntax_after_assembly_string
        <CharIn, FTuple, OutputWriter, OutputWriterInitArgsTuple>;

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

    std::size_t calculate_size(OutputWriter&, arglist_type args) const
    {
        std::size_t len = 0;
        for(const auto& arg : args)
        {
            len += arg->length();
        }
        return len;
    }

    template <typename CharIn>
    asm_string<CharIn> operator()(const CharIn* str) const
    {
        return asm_str(str, str + std::char_traits<CharIn>::length(str));
    }

    template <typename CharIn, typename Traits, typename Allocator>
    asm_string<CharIn> operator()
        (const std::basic_string<CharIn, Traits, Allocator>& str) const
    {
        return asm_str(str.data(), str.data() + str.size());
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename CharIn, typename Traits>
    asm_string<CharIn> operator()
        (const std::basic_string_view<CharIn, Traits>& str) const
    {
        return asm_str(str.begin(), str.end());
    }

#endif

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) error_code(const Args& ... args) const
    {
        output_writer_wrapper writer{m_ftuple, m_owinit};
        write(writer.get(), {as_pointer(mk_printer<Args>(writer.get(), args)) ...});
        return writer.get().finish_error_code();
    }

    template <typename ... Args>
    decltype(auto) exception(const Args& ... args) const
    {
        output_writer_wrapper writer{m_ftuple, m_owinit};
        write(writer.get(), {as_pointer(mk_printer<Args>(writer.get(), args)) ...});
        return writer.get().finish_exception();
    }

private:

    template <typename CharIn>
    asm_string<CharIn> asm_str(const CharIn* begin, const CharIn* end) const
    {
        return {m_ftuple, m_owinit, begin, end, m_reservation};
    }

    
    static const stringify::v0::printer<char_type>*
    as_pointer(const stringify::v0::printer<char_type>& p)
    {
        return &p;
    }

    template <typename Arg>
    auto mk_printer
        ( stringify::v0::output_writer<char_type>& ow
        , const Arg& arg )
        const
    {
        return stringify_make_printer<char_type, FTuple>(ow, m_ftuple, arg);
    }

    void write(OutputWriter& writer, arglist_type args) const
    {
        m_reservation.reserve(*this, writer, args);
        for(auto it = args.begin(); it != args.end() && writer.good(); ++it)
        {
            (*it)->write();
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
    //using input_traits = decltype(stringify_get_input_traits(value));
    //return input_traits::fmt(value);
    return stringify_fmt(value);
}

template <typename T, typename ... Args>
constexpr auto sani(const T& value, const Args& ... args)
{
    return stringify_fmt(value).sani(args...);
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
