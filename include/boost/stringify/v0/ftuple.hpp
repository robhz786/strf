#ifndef BOOST_STRINGIFY_V0_FTUPLE_HPP
#define BOOST_STRINGIFY_V0_FTUPLE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/constrained_facet.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename ... F> class ftuple;

namespace detail {

template <typename Tag>
struct increment_tag: Tag
{
};


struct absolute_lowest_tag
{
};


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
    constexpr const auto&
    do_get_facet (const absolute_lowest_tag&, FacetCategory) const
    {
        return FacetCategory::get_default();
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
class single_facet_ftuple: public ftuple_end<LowestTag>
{
public:

    constexpr single_facet_ftuple(const Facet& f) : m_facet(f) {}

    constexpr single_facet_ftuple(const single_facet_ftuple& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_ftuple<OtherLowestTag, Facet>;

    using ftuple_end<LowestTag>::do_get_facet;

    template <typename>
    constexpr const Facet& do_get_facet
        (const highest_tag&, typename Facet::category) const
    {
        return m_facet;
    }

private:

    Facet m_facet;
};


template
    < typename LowestTag
    , template <class> class Filter
    , typename Facet
    >
class single_facet_ftuple
        < LowestTag
        , boost::stringify::v0::constrained_facet<Filter, Facet>
        >
    : public ftuple_end<LowestTag>
{

    using ConstrainedFacet
        = boost::stringify::v0::constrained_facet<Filter, Facet>;

public:

    constexpr single_facet_ftuple
        (const ConstrainedFacet& f) : m_constrained_facet(f) {};

    constexpr single_facet_ftuple
        (const single_facet_ftuple& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_ftuple<OtherLowestTag, ConstrainedFacet>;

    using ftuple_end<LowestTag>::do_get_facet;

    template
        < typename InputType
        , typename = typename std::enable_if
            <ConstrainedFacet::template matches<InputType>::value>
            ::type
        >
    constexpr const auto& do_get_facet
        ( const highest_tag&
        , typename ConstrainedFacet::category
        ) const
    {
        return m_constrained_facet.get_facet();
    }

private:

    ConstrainedFacet m_constrained_facet;
};


template <typename LowestTag, typename Facet>
class ref_facet_ftuple: public ftuple_end<LowestTag>
{
public:

    constexpr ref_facet_ftuple(std::reference_wrapper<Facet> facet_ref)
        : m_facet_ref(facet_ref)
    {
    }

    constexpr ref_facet_ftuple(const ref_facet_ftuple& copy) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = ref_facet_ftuple<OtherLowestTag, Facet>;

    using ftuple_end<LowestTag>::do_get_facet;

    template <typename>
    constexpr const Facet& do_get_facet
        (const highest_tag&, typename Facet::category) const
    {
        return m_facet_ref.get();
    }

private:

    std::reference_wrapper<Facet> m_facet_ref;
};


template
    < typename LowestTag
    , template <class> class Filter
    , typename Facet
    >
class ref_facet_ftuple
        < LowestTag
        , const boost::stringify::v0::constrained_facet<Filter, Facet > >
    : public ftuple_end<LowestTag>
{

    using ConstrainedFacet
        = boost::stringify::v0::constrained_facet<Filter, Facet>;
    using RefConstrainedFacet = std::reference_wrapper<const ConstrainedFacet>;

public:

    constexpr ref_facet_ftuple(RefConstrainedFacet facet_ref)
        : m_facet_ref(facet_ref)
    {
    }

    constexpr ref_facet_ftuple(const ref_facet_ftuple&) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = ref_facet_ftuple<OtherLowestTag, const ConstrainedFacet>;

    using ftuple_end<LowestTag>::do_get_facet;

    template
        < typename InputType
        , typename = typename std::enable_if
            <ConstrainedFacet::template matches<InputType>::value>::type
        >
    constexpr const auto& do_get_facet
        ( const highest_tag&
        , typename ConstrainedFacet::category
        ) const
    {
        return m_facet_ref.get().get_facet();
    }

private:

    RefConstrainedFacet m_facet_ref;
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

    template <typename Facet>
    static constexpr auto as_ftuple(std::reference_wrapper<Facet> f)
    {
        return stringify::v0::detail::ref_facet_ftuple<base_tag, const Facet>
        {f.get()};
    }

    template <typename ... F>
    static constexpr const stringify::v0::ftuple<F...>&
    as_ftuple(const stringify::v0::ftuple<F...>& f)
    {
        return f;
    }

}; // struct ftuple_helper

template <typename ... F>
static constexpr auto make_ftuple_impl(const F& ... f)
{
    using base_tag = stringify::v0::detail::base_tag;
    using helper = stringify::v0::detail::ftuple_helper;
    return helper::join_multi_ftuples<base_tag>(helper::as_ftuple(f)...);
}

template <typename ... F>
using ftuple_impl
= decltype(stringify::v0::detail::make_ftuple_impl(std::declval<F>()...));

} // namespace detail


template <typename ... Facets>
class ftuple: private stringify::v0::detail::ftuple_impl<Facets...>
{
    using impl = stringify::v0::detail::ftuple_impl<Facets...>;

    friend struct stringify::v0::detail::ftuple_helper;

    template <typename, typename>
    friend class stringify::v0::detail::ftuple_join;

public:

    constexpr ftuple(const Facets& ... f)
        : impl(stringify::v0::detail::make_ftuple_impl(f...))
    {
    }

    template <typename FacetCategory, typename InputType>
    constexpr const auto& get_facet() const
    {
        return this->template do_get_facet<InputType>
            (typename impl::highest_tag(), FacetCategory());
    }
};


template <typename ... Facets>
auto make_ftuple(const Facets& ... f)
{
    return stringify::v0::ftuple<Facets...>(f...);
}


template
    < typename FacetCategory
    , typename InputType
    , typename ... Facets
    >
constexpr const auto& get_facet(const stringify::v0::ftuple<Facets...>& ft)
{
    return ft.template get_facet<FacetCategory, InputType>();
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FTUPLE2_HPP

