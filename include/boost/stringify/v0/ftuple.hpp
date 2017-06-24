#ifndef BOOST_STRINGIFY_V0_FTUPLE_HPP
#define BOOST_STRINGIFY_V0_FTUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/type_traits.hpp>

namespace boost {
namespace stringify {
inline namespace v0 {

template <typename ... F> class ftuple;

namespace detail {

template <typename Tag>  struct increment_tag: Tag {};

struct absolute_lowest_tag {};

using base_tag = increment_tag<absolute_lowest_tag>;


template <typename LowestTag>
struct ftuple_end
{
    void do_get_facet();
};

template <>
struct ftuple_end<base_tag>
{
    template <typename, typename FacetCategory>
    constexpr decltype(auto)
    do_get_facet (const absolute_lowest_tag&, FacetCategory) const
    {
        return typename FacetCategory::default_impl();
    }
};


template <typename LowestTag>
class empty_ftuple: public ftuple_end<LowestTag>
{
public:

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    constexpr empty_ftuple() = default;

    constexpr empty_ftuple(const empty_ftuple& ) = default;

    template <typename OtherLowestTag>
    using rebind = empty_ftuple<OtherLowestTag>;

    using ftuple_end<LowestTag>::do_get_facet;
};


template <typename LowestTag, typename Facet>
class single_facet_ftuple: private Facet, public ftuple_end<LowestTag>
{
public:

    constexpr single_facet_ftuple(const Facet& f) : Facet(f) {};

    constexpr single_facet_ftuple(const single_facet_ftuple& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_ftuple<OtherLowestTag, Facet>;

    using ftuple_end<LowestTag>::do_get_facet;

    template
        < typename InputType
        , typename = typename std::enable_if_t
          <Facet::template accept_input_type<InputType>::value>
        >
    constexpr const Facet& do_get_facet
        (const highest_tag&, typename Facet::category) const
    {
        return *this;
    }

};


template< typename LowerFTuple, typename HigherFTuple>
class ftuple_join : private LowerFTuple, private HigherFTuple
{
public:

    static_assert
        ( std::is_base_of
              < typename LowerFTuple::highest_tag
              , typename HigherFTuple::lowest_tag
              >::value
        , "inconsistent tags"
        );

    using highest_tag = typename HigherFTuple::highest_tag;
    using lowest_tag = typename LowerFTuple::lowest_tag;

    constexpr ftuple_join(const LowerFTuple lf, const HigherFTuple& hf)
        : LowerFTuple(lf)
        , HigherFTuple(hf)
    {
    }

    constexpr ftuple_join(const ftuple_join&) = default;

    template <typename OtherLowestTag>
    struct rebind_helper
    {
        using new_lower_ftuple
            = typename LowerFTuple::template rebind<OtherLowestTag>;
        using tag = increment_tag<typename new_lower_ftuple::highest_tag>;
        using new_higher_ftuple
            = typename HigherFTuple::template rebind<tag>;

        using type = ftuple_join<new_lower_ftuple, new_higher_ftuple>;
    };

    template <typename OtherLowestTag>
    using rebind = typename rebind_helper<OtherLowestTag>::type;

    using HigherFTuple::do_get_facet;
    using LowerFTuple::do_get_facet;

};

struct ftuple_helper
{

    template <typename LowestTag>
    static constexpr empty_ftuple<LowestTag> join_multi_ftuples()
    {
        return empty_ftuple<LowestTag>();
    }

    template <typename LowestTag, typename FTuple>
    static constexpr auto join_multi_ftuples(const FTuple& f)
    {
        using rebinded_ftuple = typename FTuple::template rebind<LowestTag>;
        return reinterpret_cast<const rebinded_ftuple&>(f);
    }


    template
        < typename LowestTag
        , typename FTuple1
        , typename FTuple2
        , typename ... HigherFTuples
        >
    static constexpr auto join_multi_ftuples
        ( const FTuple1& f1
        , const FTuple2& f2
        , const HigherFTuples& ... hfs
        )
    {
        using lower_type = typename FTuple1::template rebind<LowestTag>;
        using middle_tag = increment_tag<typename lower_type::highest_tag>;
        using higher_type = decltype(join_multi_ftuples<middle_tag>(f2, hfs ...));

        return ftuple_join<lower_type, higher_type>
            { reinterpret_cast<const lower_type&>(f1)
            , join_multi_ftuples<middle_tag>(f2, hfs ...)
            };
    }

    template <typename Facet, typename = typename Facet::category>
    static constexpr const auto& as_ftuple(const Facet& f)
    {
        using ftuple_type = single_facet_ftuple<base_tag, Facet>;
        return reinterpret_cast<const ftuple_type&>(f);
    }

    template <typename ... F>
    static constexpr const boost::stringify::ftuple<F...>&
    as_ftuple(const boost::stringify::ftuple<F...>& f)
    {
        return f;
    }

}; // struct ftuple_helper

template <typename ... F>
static constexpr auto make_ftuple_impl(const F& ... f)
{
    using base_tag = boost::stringify::v0::detail::base_tag;
    using helper = boost::stringify::v0::detail::ftuple_helper;
    return helper::join_multi_ftuples<base_tag>(helper::as_ftuple(f)...);
}

template <typename ... F>
using ftuple_impl
= decltype(boost::stringify::v0::detail::make_ftuple_impl(std::declval<F>()...));

} // namespace detail


template <typename ... F>
class ftuple: private boost::stringify::v0::detail::ftuple_impl<F...>
{
    using impl = boost::stringify::v0::detail::ftuple_impl<F...>;

    friend struct boost::stringify::v0::detail::ftuple_helper;

    template <typename, typename>
    friend class boost::stringify::v0::detail::ftuple_join;

public:

    constexpr ftuple(const F& ... f)
        : impl(boost::stringify::v0::detail::make_ftuple_impl(f...))
    {
    }
   
    template <typename FacetCategory, typename InputType>
    constexpr decltype(auto) get_facet() const
    {
        return this->template do_get_facet<InputType>
            (typename impl::highest_tag(), FacetCategory());
    }
};


template <typename ... F>
auto make_ftuple(const F& ... f)
{
    return boost::stringify::v0::ftuple<F...>(f...);
}


template
    < typename FacetCategory
    , typename InputType
    , typename FTuple
    >
constexpr decltype(auto) get_facet(const FTuple& f)
{
    return f.template get_facet<FacetCategory, InputType>();
}


} // namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_FTUPLE2_HPP

