#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_FTUPLE_HPP
#define BOOST_STRINGIFY_V0_INPUT_TYPES_FTUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/input_types/join.hpp>
#include <initializer_list>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename FTuple, typename ... Args>
struct inner_ftuple_with_args
{
    const FTuple& ft;
    stringify::v0::detail::args_tuple<Args...> args;
};


template <typename FTuple>
struct inner_ftuple
{
    FTuple ft;

    template <typename ... Args>
    stringify::v0::detail::inner_ftuple_with_args<FTuple, Args...>
    operator()(const Args& ... args)
    {
        return stringify::v0::detail::inner_ftuple_with_args<FTuple, Args...>
            { ft
            , stringify::v0::detail::args_tuple<Args...>{args ...}
            };
    }
};

template <typename FTuple>
struct inner_ftuple_ref
{
    const FTuple& ft;

    template <typename ... Args>
    stringify::v0::detail::inner_ftuple_with_args<FTuple, Args...>
    operator()(const Args& ... args)
    {
        return stringify::v0::detail::inner_ftuple_with_args<FTuple, Args...>
            { ft
            , stringify::v0::detail::args_tuple<Args...>{args ...}
            };
    }
};

template <typename CharT>
class pp_range_printer: public stringify::v0::printer<CharT>
{

    using printer_type = stringify::v0::printer<CharT>;
    using pp_range = stringify::v0::detail::printer_ptr_range<CharT>;

public:

    pp_range_printer(const pp_range& args)
        : m_args(args)
    {
    }

    virtual ~pp_range_printer()
    {
    }

    std::size_t length() const override
    {
        std::size_t sum = 0;
        for(const auto* arg : m_args)
        {
            sum += arg->length();
        }
        return sum;
    }

    void write(stringify::v0::output_writer<CharT>& out) const override
    {
        for(const auto& arg : m_args)
        {
            arg->write(out);
        }
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_args.begin(); w > 0 && it != m_args.end(); ++it)
        {
            w = (*it) -> remaining_width(w);
        }
        return w;
    }

private:

    pp_range m_args;
};


template <typename CharT, typename ParentFTuple, typename ChildFTuple, typename ... Args>
class ftuple_printer
    : private stringify::v0::ftuple<ParentFTuple, ChildFTuple>
    , private stringify::v0::detail::printers_group
          < CharT
          , stringify::v0::ftuple<ParentFTuple, ChildFTuple>
          , Args...
          >
    , public stringify::v0::detail::pp_range_printer<CharT>
{
    using fmt_group
    = stringify::v0::detail::printers_group
          < CharT
          , stringify::v0::ftuple<ParentFTuple, ChildFTuple>
          , Args...
          >;

public:

    ftuple_printer
        ( const ParentFTuple& parent_ft
        , const stringify::v0::detail::inner_ftuple_with_args<ChildFTuple, Args...>& args
        )
        : stringify::v0::ftuple<ParentFTuple, ChildFTuple>{parent_ft, args.ft}
        , fmt_group{*this, args.args}
        , stringify::v0::detail::pp_range_printer<CharT>{fmt_group::range()}
    {
    }

    virtual ~ftuple_printer()
    {
    }
};

// struct ftuple_input_traits
// {
//     template
//         < typename CharT
//         , typename ParentFTuple
//         , typename ChildFTuple
//         , typename ... Args
//         >
//     static inline stringify::v0::detail::ftuple_printer
//         < CharT
//         , ParentFTuple
//         , ChildFTuple
//         , Args...
//         >
//     make_printer
//         ( const ParentFTuple& ft
//         , const detail::inner_ftuple_with_args<ChildFTuple, Args...>& x
//         )
//     {
//         return {ft, x};
//     }
// };

} // namespace detail

// template <typename ChildFTuple, typename ... Args>
// stringify::v0::detail::ftuple_input_traits
// stringify_get_input_traits
// ( const stringify::v0::detail::inner_ftuple_with_args<ChildFTuple, Args...>& fmt );

template <typename ... Facets>
stringify::v0::detail::inner_ftuple<stringify::v0::ftuple<Facets ...>>
facets(const Facets& ... facets)
{
    return {stringify::v0::make_ftuple(facets ...)};
}

template <typename ... Facets>
stringify::v0::detail::inner_ftuple_ref<stringify::v0::ftuple<Facets ...>>
facets(const stringify::v0::ftuple<Facets...>& ft)
{
    return {ft};
}

template
   < typename CharT
   , typename FTuple
   , typename ChildFTuple
   , typename ... Args
   >
inline stringify::v0::detail::ftuple_printer<CharT, FTuple, ChildFTuple, Args...>
stringify_make_printer
   ( const FTuple& ft
   , const stringify::v0::detail::inner_ftuple_with_args<ChildFTuple, Args...>& fmt
   )
{
    return {ft, fmt};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_INPUT_TYPES_FTUPLE_HPP

