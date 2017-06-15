#ifndef BOOST_STRINGIFY_V1_FTUPLE_HPP
#define BOOST_STRINGIFY_V1_FTUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/type_traits.hpp>

namespace boost {
namespace stringify {
inline namespace v1 {
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



template<typename HigherFTuple, typename LowerFTuple>  
class ftuple_join : private HigherFTuple, private LowerFTuple
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
    
    constexpr ftuple_join(const HigherFTuple& hf, const LowerFTuple lf)
        : HigherFTuple(hf)
        , LowerFTuple(lf)
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

        using type = ftuple_join<new_higher_ftuple, new_lower_ftuple>;
    };
    
    template <typename OtherLowestTag>
    using rebind = typename rebind_helper<OtherLowestTag>::type;

    using HigherFTuple::do_get_facet;
    using LowerFTuple::do_get_facet;

};


template <typename LowestTag, typename FTuple>
typename FTuple::template rebind<LowestTag>
rebinded_ftuple_value(const FTuple& f);

template <typename LowestTag, typename Facet, typename = typename Facet::category>
single_facet_ftuple<LowestTag, Facet>
rebinded_ftuple_value(const Facet& f);

constexpr empty_ftuple<base_tag> join_multi_ftuples()
{
    return empty_ftuple<base_tag>();
}

template <typename FTuple>
constexpr FTuple join_multi_ftuples(const FTuple& f)
{
    return f;
}


template <typename FTuple1, typename FTuple2, typename ... LowerFTuples>
constexpr auto join_multi_ftuples
    ( const FTuple1& f1
    , const FTuple2& f2  
    , const LowerFTuples& ... lfs
    )
{
    //auto lower = join_multi_ftuples(f2, lfs ...);
    using lower_type = decltype(join_multi_ftuples(f2, lfs ...));
    using tag = increment_tag<typename lower_type::highest_tag>;
    using rebinded_higher_type = decltype(rebinded_ftuple_value<tag>(f1));
    const auto& higher = *reinterpret_cast<const rebinded_higher_type*>(&f1);

    return ftuple_join<rebinded_higher_type, lower_type>
        { higher, join_multi_ftuples(f2, lfs ...) };
}

template <typename Facet, typename = typename Facet::category>
constexpr const auto& as_ftuple(const Facet& f)
{
    using ftuple_type = single_facet_ftuple<base_tag, Facet>;
    return * reinterpret_cast<const ftuple_type*>(&f);
}

template <typename FTuple, typename = typename FTuple::highest_tag>
constexpr const FTuple& as_ftuple(const FTuple& f)
{
    return f;
}

} // namespace detail


template <typename ... F>
constexpr auto make_ftuple(const F& ... f)
{
    return boost::stringify::v1::detail::join_multi_ftuples
        (boost::stringify::v1::detail::as_ftuple(f)...);
}


template <typename ... F>
using ftuple
= decltype(boost::stringify::v1::make_ftuple<F...>(std::declval<F>()...));


template
    < typename FacetCategory
    , typename InputType
    , typename FTuple  
    >
constexpr decltype(auto) get_facet(const FTuple& f)
{
    using highest_tag = typename FTuple::highest_tag;
    return f.template do_get_facet<InputType>
        (highest_tag(), FacetCategory());
}

} // namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_FTUPLE2_HPP

