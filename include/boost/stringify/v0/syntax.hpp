#ifndef BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
#define BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
//#include <boost/stringify/v0/detail/assembly_string.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
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

    template <typename SizeCalculator, typename OutputWriter, typename ... ArgList>
    void reserve
        ( SizeCalculator& calc
        , OutputWriter& writer
        , const ArgList& ... args ) const
    {
        if (m_flag != no_reserve)
        {
            writer.reserve
                ( m_flag == calculate_size
                ? calc.calculate_size(args...)
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

private:

    static std::size_t do_calculate_size()
    {
        return 0;
    }

    template <typename Arg0, typename ... Args>
    static std::size_t do_calculate_size(const Arg0& arg, const Args ... args)
    {
        return arg.necessary_size() + do_calculate_size(args...);
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
        return m_out;
    }

private:

    OutputWriter m_out;

    using CharOut = typename OutputWriter::char_type;

    template <typename ArgsTuple, std::size_t ... I>
    output_writer_from_tuple
        ( const ArgsTuple& args
        , std::index_sequence<I...>)
        : m_out(std::get<I>(args)...)
    {
    }

};

// template
//     < typename CharIn
//     , typename FPack
//     , typename OutputWriter
//     , typename OutputWriterInitArgsTuple
//     >
// class syntax_after_assembly_string
// {
//     using CharOut = typename OutputWriter::char_type;
//     using arglist_type = std::initializer_list<const stringify::v0::printer<CharOut>*>;

//     using reservation_type
//         = stringify::v0::detail::output_size_reservation<OutputWriter>;
//     using output_writer_wrapper
//         = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

//     using input_tag = stringify::v0::asm_string_input_tag<CharIn>;

// public:

//     syntax_after_assembly_string
//         ( const FPack& fp
//         , const OutputWriterInitArgsTuple& owinit
//         , const CharIn* str
//         , const CharIn* str_end
//         , const reservation_type& res
//         , bool sanitise = false
//         )
//         : m_fpack(fp)
//         , m_owinit(owinit)
//         , m_str(str)
//         , m_end(str_end)
//         , m_reservation(res)
//         , m_sanitise(sanitise)
//     {
//     }

//     template <typename ... Args>
//     BOOST_STRINGIFY_NODISCARD
//     decltype(auto) operator()(const Args& ... args) const
//     {
//         output_writer_wrapper dest{m_fpack, m_owinit};
//         return write(dest.get(), {as_pointer(make_printer<CharOut, FPack>(m_fpack, args)) ...});
//     }

//     std::size_t calculate_size(OutputWriter& dest, arglist_type args) const
//     {
//         stringify::v0::detail::asm_string_measurer<CharIn, CharOut> measurer
//         { dest, m_fpack, args, m_sanitise };
//         stringify::v0::detail::parse_asm_string(m_str, m_end, measurer);
//         return measurer.result();
//     }

// private:

//     template <typename FCategory>
//     const auto& get_facet(const FPack& fp) const
//     {
//         return fp.template get_facet<FCategory, input_tag>();
//     }

//     static const stringify::v0::printer<CharOut>*
//     as_pointer(const stringify::v0::printer<CharOut>& p)
//     {
//         return &p;
//     }

//     void write(OutputWriter& owriter, const arglist_type& args) const
//     {
//         m_reservation.reserve(*this, owriter, args);
//         stringify::v0::detail::asm_string_writer<CharIn, CharOut> asm_writer
//         { owriter, m_fpack, args, m_sanitise };

//         stringify::v0::detail::parse_asm_string(m_str, m_end, asm_writer);
//     }

//     const FPack& m_fpack;
//     const OutputWriterInitArgsTuple& m_owinit;
//     const CharIn* const m_str;
//     const CharIn* const m_end;
//     reservation_type m_reservation;
//     bool m_sanitise;
// };

template
    < typename FPack
    , typename OutputWriter
    , typename OutputWriterInitArgsTuple
    >
class syntax_after_leading_expr
{
public:

    using char_type = typename OutputWriter::char_type;
    using arglist_type
        = std::initializer_list<const stringify::v0::printer<char_type>*>;
    using output_writer_wrapper
        = stringify::v0::detail::output_writer_from_tuple<OutputWriter>;

    // template <typename CharIn>
    // using asm_string = stringify::v0::detail::syntax_after_assembly_string
    //     <CharIn, FPack, OutputWriter, OutputWriterInitArgsTuple>;

public:

    constexpr syntax_after_leading_expr
        ( FPack&& fp
        , OutputWriterInitArgsTuple&& args)
        : m_fpack(std::move(fp))
        , m_owinit(std::move(args))
    {
    }

    constexpr syntax_after_leading_expr
        ( const FPack& fp
        , const OutputWriterInitArgsTuple& args )
        : m_fpack(fp)
        , m_owinit(args)
    {
    }

    constexpr syntax_after_leading_expr(const syntax_after_leading_expr&)
        = default;

    constexpr syntax_after_leading_expr(syntax_after_leading_expr&&) = default;

    template <typename ... Facets>
    constexpr auto facets(const Facets& ... facets) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::pack(m_fpack, facets ...))
            , OutputWriter
            , OutputWriterInitArgsTuple >
            ( stringify::v0::pack(m_fpack, facets ...)
            , m_owinit );
    }

    template <typename ... Facets>
    constexpr auto facets(const stringify::v0::facets_pack<Facets...>& fp) const
    {
        return syntax_after_leading_expr
            < decltype(stringify::v0::pack(m_fpack, fp))
            , OutputWriter
            , OutputWriterInitArgsTuple >
            ( stringify::v0::pack(m_fpack, fp)
            , m_owinit );
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

    static std::size_t calculate_size()
    {
        return 0;
    }

    template <typename Arg0, typename ... Args>
    static std::size_t calculate_size(const Arg0& arg, const Args ... args)
    {
        return arg.necessary_size() + calculate_size(args...);
    }

//     template <typename CharIn>
//     asm_string<CharIn> as(const CharIn* str) const
//     {
//         return asm_str(str, str + std::char_traits<CharIn>::length(str));
//     }

//     template <typename CharIn, typename Traits, typename Allocator>
//     asm_string<CharIn> as
//         (const std::basic_string<CharIn, Traits, Allocator>& str) const
//     {
//         return asm_str(str.data(), str.data() + str.size());
//     }

// #if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

//     template <typename CharIn, typename Traits>
//     asm_string<CharIn> as
//         (const std::basic_string_view<CharIn, Traits>& str) const
//     {
//         return asm_str(&*str.begin(), &*str.end());
//     }

// #endif

    template <typename ... Args>
    BOOST_STRINGIFY_NODISCARD
    decltype(auto) operator()(const Args& ... args) const
    {
        return write(make_printer<char_type, FPack>(m_fpack, args) ...);
    }

private:

    template <typename ... Args>
    auto write(Args ... args) const
        -> decltype(std::declval<OutputWriter>().finish((char_type*)(0)))
    {
        output_writer_wrapper writer{m_owinit};

        m_reservation.reserve(*this, writer.get(), args...);
        auto x = writer.get().start();
        if (x)
        {
            x = write_args(*x, writer.get(), args...);
            if (x)
            {
                return writer.get().finish((*x).it);
            }
        }
        return { stringify::v0::unexpect_t{}, x.error() };
    }

    template <typename Arg>
    static stringify::v0::expected_buff_it<char_type> write_args
        ( stringify::v0::buff_it<char_type> b
        , OutputWriter& writer
        , const Arg& arg )
    {
        return arg.write(b, writer);
    }

    template <typename Arg, typename ... Args>
    static stringify::v0::expected_buff_it<char_type> write_args
        ( stringify::v0::buff_it<char_type> b
        , OutputWriter& writer
        , const Arg& arg
        , const Args& ... args )
    {
        auto x = arg.write(b, writer);
        return x ? write_args(*x, writer, args ...) : x;
    }

    static stringify::v0::expected_buff_it<char_type> write_args
        ( stringify::v0::buff_it<char_type> b
        , OutputWriter& writer )
    {
        return { stringify::v0::in_place_t{}, b};
    }

    FPack m_fpack;
    OutputWriterInitArgsTuple m_owinit;
    stringify::v0::detail::output_size_reservation<OutputWriter>
    m_reservation;
};

} // namespace detail

template <typename CharOut, typename FPack, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>
             ( std::declval<FPack>()
             , std::declval<Arg>() ) );

template <typename OutputWriter, typename ... Args>
constexpr auto make_destination(Args ... args)
    -> stringify::v0::detail::syntax_after_leading_expr
        < stringify::v0::facets_pack<>
        , OutputWriter
        , std::tuple<Args ...>
        >
{
    return {stringify::v0::facets_pack<>{}, std::tuple<Args ...>{args ...}};
}

template <typename T>
constexpr auto fmt(const T& value)
{
    return make_fmt(stringify::v0::tag{}, value);
}

template <typename T, typename ... Args>
constexpr auto sani(const T& value, const Args& ... args)
{
    return make_fmt(stringify::v0::tag{}, value).sani(args...);
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

// template <typename T>
// inline auto ascii(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::ascii());
// }
// template <typename T>
// inline auto iso_8859_1(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::iso_8859_1());
// }
// template <typename T>
// inline auto iso_8859_15(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::iso_8859_15());
// }
// template <typename T>
// inline auto windows_1252(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::windows_1252());
// }
template <typename T>
inline auto utf8(const T& x)
{
    return fmt(x).encoding(stringify::v0::utf8());
}
// template <typename T>
// inline auto mutf8(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::mutf8());
// }
// template <typename T>
// inline auto utfw(const T& x)
// {
//     return fmt(x).encoding(stringify::v0::utfw());
// }

template <typename T>
inline auto utf16(const T& x)
{
    return fmt(x).encoding(stringify::v0::utf16());
}
template <typename T>
inline auto utf32(const T& x)
{
    return fmt(x).encoding(stringify::v0::utf32());
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARGS_HANDLER_HPP
