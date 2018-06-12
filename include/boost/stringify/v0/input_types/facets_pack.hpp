#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_FACETS_PACK_HPP
#define BOOST_STRINGIFY_V0_INPUT_TYPES_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/input_types/join.hpp>
#include <initializer_list>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename FPack, typename ... Args>
struct inner_pack_with_args
{
    const FPack& ft;
    stringify::v0::detail::args_tuple<Args...> args;
};


template <typename FPack>
struct inner_pack
{
    FPack ft;

    template <typename ... Args>
    stringify::v0::detail::inner_pack_with_args<FPack, Args...>
    operator()(const Args& ... args)
    {
        return stringify::v0::detail::inner_pack_with_args<FPack, Args...>
            { ft
            , stringify::v0::detail::args_tuple<Args...>{args ...}
            };
    }
};

template <typename FPack>
struct inner_pack_ref
{
    const FPack& ft;

    template <typename ... Args>
    stringify::v0::detail::inner_pack_with_args<FPack, Args...>
    operator()(const Args& ... args)
    {
        return stringify::v0::detail::inner_pack_with_args<FPack, Args...>
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

    void write() const override
    {
        for(const auto& arg : m_args)
        {
            arg->write();
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


template <typename CharT, typename ParentFPack, typename ChildFPack, typename ... Args>
class facets_pack_printer
    : private stringify::v0::facets_pack<ParentFPack, ChildFPack>
    , private stringify::v0::detail::printers_group
          < CharT
          , stringify::v0::facets_pack<ParentFPack, ChildFPack>
          , Args...
          >
    , public stringify::v0::detail::pp_range_printer<CharT>
{
    using fmt_group
    = stringify::v0::detail::printers_group
          < CharT
          , stringify::v0::facets_pack<ParentFPack, ChildFPack>
          , Args...
          >;

public:

    facets_pack_printer
        ( stringify::v0::output_writer<CharT>& out
        , const ParentFPack& parent_ft
        , const stringify::v0::detail::inner_pack_with_args<ChildFPack, Args...>& args
        )
        : stringify::v0::facets_pack<ParentFPack, ChildFPack>{parent_ft, args.ft}
        , fmt_group{out, *this, args.args}
        , stringify::v0::detail::pp_range_printer<CharT>{fmt_group::range()}
    {
    }

    virtual ~facets_pack_printer()
    {
    }
};

// struct facets_pack_input_traits
// {
//     template
//         < typename CharT
//         , typename ParentFPack
//         , typename ChildFPack
//         , typename ... Args
//         >
//     static inline stringify::v0::detail::facets_pack_printer
//         < CharT
//         , ParentFPack
//         , ChildFPack
//         , Args...
//         >
//     make_printer
//         ( const ParentFPack& ft
//         , const detail::inner_pack_with_args<ChildFPack, Args...>& x
//         )
//     {
//         return {ft, x};
//     }
// };



template <typename F>
struct is_constrainable
{
    using category = typename F::category;
    constexpr static bool value = category::constrainable;
};

template <typename ... F>
constexpr bool are_constrainable_impl()
{
    constexpr std::size_t N = sizeof...(F);
    constexpr bool values[N]
        = {stringify::v0::detail::is_constrainable<F>::value ...};

    for (std::size_t i = 0; i < N; ++i)
    {
        if( ! values[i])
        {
            return false;
        }
    }
    return true;;
}

template <>
constexpr bool are_constrainable_impl<>()
{
    return true;
}

template <typename ... F>
struct is_constrainable<stringify::v0::facets_pack<F...>>
{
    constexpr static bool value
        = stringify::v0::detail::are_constrainable_impl<F...>();
};

template <typename ... F>
struct all_are_constrainable
{
    constexpr static bool value
        = stringify::v0::detail::are_constrainable_impl<F...>();
};



} // namespace detail

// template <typename ChildFPack, typename ... Args>
// stringify::v0::detail::facets_pack_input_traits
// stringify_get_input_traits
// ( const stringify::v0::detail::inner_pack_with_args<ChildFPack, Args...>& fmt );

template <typename ... Facets>
stringify::v0::detail::inner_pack<stringify::v0::facets_pack<Facets ...>>
facets(const Facets& ... facets)
{
    static_assert
        ( stringify::v0::detail::all_are_constrainable<Facets...>::value
        , "All facet categories must be constrainable" );
    return {stringify::v0::pack(facets ...)};
}

template <typename ... Facets>
stringify::v0::detail::inner_pack_ref<stringify::v0::facets_pack<Facets ...>>
facets(const stringify::v0::facets_pack<Facets...>& ft)
{
    static_assert
        ( stringify::v0::detail::all_are_constrainable<Facets...>::value
        , "All facet categories must be constrainable" );
    return {ft};
}

template
   < typename CharT
   , typename FPack
   , typename ChildFPack
   , typename ... Args
   >
inline stringify::v0::detail::facets_pack_printer<CharT, FPack, ChildFPack, Args...>
stringify_make_printer
   ( stringify::v0::output_writer<CharT>& out
   , const FPack& ft
   , const stringify::v0::detail::inner_pack_with_args<ChildFPack, Args...>& fmt
   )
{
    return {out, ft, fmt};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_INPUT_TYPES_FACETS_PACK_HPP
