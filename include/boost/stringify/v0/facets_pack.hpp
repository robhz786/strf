#ifndef BOOST_STRINGIFY_V0_FACETS_PACK_HPP
#define BOOST_STRINGIFY_V0_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <type_traits>
#include <functional>


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <template <class> class Filter, typename Facet>
class constrained_facet;

namespace detail
{

template <template <class> class ParentFilter, typename F>
struct constrained_facet_helper
{
    constexpr static const auto& get(const F& f)
    {
        return f;
    }

    template <typename InputType>
    using matches = ParentFilter<InputType>;

    using category = typename F::category;
};


template <template <class> class ParentFilter, typename F>
struct constrained_facet_helper<ParentFilter, std::reference_wrapper<F>>
{
    using helper
        = stringify::v0::detail::constrained_facet_helper<ParentFilter, F>;

    constexpr static const auto& get(std::reference_wrapper<F> f)
    {
        return helper::get(f.get());
    }

    template <typename InputType>
    using matches = typename helper::template matches<InputType>;

    using category = typename helper::category;
};


template
    < template <class> class ParentFilter
    , template <class> class Filter
    , typename F
    >
struct constrained_facet_helper
    < ParentFilter
    , stringify::v0::constrained_facet<Filter, F>
    >
{
    using constrained_f = stringify::v0::constrained_facet<Filter, F>;
    using helper
        = stringify::v0::detail::constrained_facet_helper<ParentFilter, F>;

    constexpr static const auto& get(const constrained_f& f)
    {
        return helper::get(f.m_facet);
    }

    template <typename T>
    using matches = std::integral_constant
        < bool
        , Filter<T>::value && helper::template matches<T>::value
        >;

    using category = typename helper::category;
};


} // namespace detail


template <template <class> class Filter, typename Facet>
class constrained_facet
{
    using helper =
        stringify::v0::detail::constrained_facet_helper<Filter, Facet>;

public:

    template <typename InputType>
    using matches = typename helper::template matches<InputType>;

    using category = typename helper::category;

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;

    constrained_facet(constrained_facet&&) = default;

    constrained_facet(const Facet& f)
        : m_facet(f)
    {
    }

    constexpr const auto& get_facet() const
    {
        return helper::get(m_facet);
    }

private:

    template <template <class> class, class>
    friend struct stringify::v0::detail::constrained_facet_helper;

    Facet m_facet;
};

template <template <class> class Filter, typename Facet>
auto constrain(const Facet& facet)
{
    return constrained_facet<Filter, Facet>(facet);
}

template <typename ... F> class facets_pack;

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
struct facets_pack_end
{
    void do_get_facet();
};


template <>
struct facets_pack_end<base_tag>
{
    template <typename, typename FacetCategory>
    constexpr const auto&
    do_get_facet (const absolute_lowest_tag&, FacetCategory) const
    {
        return FacetCategory::get_default();
    }
};


template <typename LowestTag>
class empty_fpack: public facets_pack_end<LowestTag>
{
public:

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    constexpr empty_fpack() = default;

    constexpr empty_fpack(const empty_fpack& ) = default;

    template <typename OtherLowestTag>
    using rebind = empty_fpack<OtherLowestTag>;

    using facets_pack_end<LowestTag>::do_get_facet;
};


template <typename LowestTag, typename Facet>
class single_facet_pack: public facets_pack_end<LowestTag>
{
public:

    constexpr single_facet_pack(const Facet& f) : m_facet(f) {}

    constexpr single_facet_pack(const single_facet_pack& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_pack<OtherLowestTag, Facet>;

    using facets_pack_end<LowestTag>::do_get_facet;

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
class single_facet_pack
        < LowestTag
        , boost::stringify::v0::constrained_facet<Filter, Facet>
        >
    : public facets_pack_end<LowestTag>
{

    using ConstrainedFacet
        = boost::stringify::v0::constrained_facet<Filter, Facet>;

public:

    constexpr single_facet_pack
        (const ConstrainedFacet& f) : m_constrained_facet(f) {};

    constexpr single_facet_pack
        (const single_facet_pack& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_pack<OtherLowestTag, ConstrainedFacet>;

    using facets_pack_end<LowestTag>::do_get_facet;

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
class ref_facet_pack: public facets_pack_end<LowestTag>
{
public:

    constexpr ref_facet_pack(std::reference_wrapper<Facet> facet_ref)
        : m_facet_ref(facet_ref)
    {
    }

    constexpr ref_facet_pack(const ref_facet_pack& copy) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = ref_facet_pack<OtherLowestTag, Facet>;

    using facets_pack_end<LowestTag>::do_get_facet;

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
class ref_facet_pack
        < LowestTag
        , const boost::stringify::v0::constrained_facet<Filter, Facet > >
    : public facets_pack_end<LowestTag>
{

    using ConstrainedFacet
        = boost::stringify::v0::constrained_facet<Filter, Facet>;
    using RefConstrainedFacet = std::reference_wrapper<const ConstrainedFacet>;

public:

    constexpr ref_facet_pack(RefConstrainedFacet facet_ref)
        : m_facet_ref(facet_ref)
    {
    }

    constexpr ref_facet_pack(const ref_facet_pack&) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = ref_facet_pack<OtherLowestTag, const ConstrainedFacet>;

    using facets_pack_end<LowestTag>::do_get_facet;

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


template< typename LowerFPack, typename HigherFPack>
class facets_pack_join : private LowerFPack, private HigherFPack
{
public:

    static_assert
        ( std::is_base_of
              < typename LowerFPack::highest_tag
              , typename HigherFPack::lowest_tag
              >::value
        , "inconsistent tags"
        );

    using highest_tag = typename HigherFPack::highest_tag;
    using lowest_tag = typename LowerFPack::lowest_tag;

    constexpr facets_pack_join(const LowerFPack lf, const HigherFPack& hf)
        : LowerFPack(lf)
        , HigherFPack(hf)
    {
    }

    constexpr facets_pack_join(const facets_pack_join&) = default;

    template <typename OtherLowestTag>
    struct rebind_helper
    {
        using new_lower_fpack
            = typename LowerFPack::template rebind<OtherLowestTag>;
        using tag = increment_tag<typename new_lower_fpack::highest_tag>;
        using new_higher_fpack
            = typename HigherFPack::template rebind<tag>;

        using type = facets_pack_join<new_lower_fpack, new_higher_fpack>;
    };

    template <typename OtherLowestTag>
    using rebind = typename rebind_helper<OtherLowestTag>::type;

    using HigherFPack::do_get_facet;
    using LowerFPack::do_get_facet;

};

struct facets_pack_helper
{

    template <typename LowestTag>
    static constexpr empty_fpack<LowestTag> join_multi_fpacks()
    {
        return empty_fpack<LowestTag>();
    }

    template <typename LowestTag, typename FPack>
    static constexpr auto join_multi_fpacks(const FPack& f)
    {
        using rebinded_fpack = typename FPack::template rebind<LowestTag>;
        return reinterpret_cast<const rebinded_fpack&>(f);
    }


    template
        < typename LowestTag
        , typename FPack1
        , typename FPack2
        , typename ... HigherFPacks
        >
    static constexpr auto join_multi_fpacks
        ( const FPack1& f1
        , const FPack2& f2
        , const HigherFPacks& ... hfs
        )
    {
        using lower_type = typename FPack1::template rebind<LowestTag>;
        using middle_tag = increment_tag<typename lower_type::highest_tag>;
        using higher_type = decltype(join_multi_fpacks<middle_tag>(f2, hfs ...));

        return facets_pack_join<lower_type, higher_type>
            { reinterpret_cast<const lower_type&>(f1)
            , join_multi_fpacks<middle_tag>(f2, hfs ...)
            };
    }

    template <typename Facet, typename = typename Facet::category>
    static constexpr const auto& as_pack(const Facet& f)
    {
        using fpack_type = single_facet_pack<base_tag, Facet>;
        return reinterpret_cast<const fpack_type&>(f);
    }

    template <typename Facet>
    static constexpr auto as_pack(std::reference_wrapper<Facet> f)
    {
        return stringify::v0::detail::ref_facet_pack<base_tag, const Facet>
        {f.get()};
    }

    template <typename ... F>
    static constexpr const stringify::v0::facets_pack<F...>&
    as_pack(const stringify::v0::facets_pack<F...>& f)
    {
        return f;
    }

}; // struct facets_pack_helper

template <typename ... F>
static constexpr auto pack_impl(const F& ... f)
{
    using base_tag = stringify::v0::detail::base_tag;
    using helper = stringify::v0::detail::facets_pack_helper;
    return helper::join_multi_fpacks<base_tag>(helper::as_pack(f)...);
}

template <typename ... F>
using facets_pack_impl
= decltype(stringify::v0::detail::pack_impl(std::declval<F>()...));

} // namespace detail


template <typename ... Facets>
class facets_pack: private stringify::v0::detail::facets_pack_impl<Facets...>
{
    using impl = stringify::v0::detail::facets_pack_impl<Facets...>;

    friend struct stringify::v0::detail::facets_pack_helper;

    template <typename, typename>
    friend class stringify::v0::detail::facets_pack_join;

public:

    constexpr facets_pack(const Facets& ... f)
        : impl(stringify::v0::detail::pack_impl(f...))
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
auto pack(const Facets& ... f)
{
    return stringify::v0::facets_pack<Facets...>(f...);
}


template
    < typename FacetCategory
    , typename InputType
    , typename ... Facets
    >
constexpr const auto& get_facet(const stringify::v0::facets_pack<Facets...>& ft)
{
    return ft.template get_facet<FacetCategory, InputType>();
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_PACK_HPP

