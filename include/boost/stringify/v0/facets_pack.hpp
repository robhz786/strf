#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <type_traits>
#include <functional>


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail
{

template <typename F, typename = typename F::category>
constexpr bool has_category_member_type(F*)
{
    return true;
}
constexpr bool has_category_member_type(...)
{
    return false;
}

template <typename F, bool has_it_as_member_type>
struct category_member_type_or_void_2;

template <typename F>
struct category_member_type_or_void_2<F, true>
{
    using type = typename F::category;
};

template <typename F>
struct category_member_type_or_void_2<F, false>
{
    using type = void;
};

template <typename F>
using category_member_type_or_void
= stringify::v0::detail::category_member_type_or_void_2
    < F, stringify::v0::detail::has_category_member_type((F*)0) >;

} // namespace detail


template <typename F>
class facet_trait
{
    using _helper = stringify::v0::detail::category_member_type_or_void<F>;

public:

    using category = typename _helper::type;
};

template <typename F>
class facet_trait<const F>
{
public:
    using category = typename facet_trait<F>::category;
};

template <typename F>
using facet_category_t
= typename stringify::v0::facet_trait<F>::category;

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

    using category = stringify::v0::facet_category_t<F>;
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
    , typename F >
struct constrained_facet_helper
    < ParentFilter
    , stringify::v0::constrained_facet<Filter, F> >
{
    using constrained_f = stringify::v0::constrained_facet<Filter, F>;
    using helper
        = stringify::v0::detail::constrained_facet_helper<ParentFilter, F>;

    constexpr static const auto& get(const constrained_f& f)
    {
        return helper::get(f._facet);
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
    using _helper =
        stringify::v0::detail::constrained_facet_helper<Filter, Facet>;

public:

    template <typename InputType>
    using matches = typename _helper::template matches<InputType>;

    using category = typename _helper::category;

    static_assert(category::constrainable, "This facet is not constrainable");

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;

    constrained_facet(constrained_facet&&) = default;

    constrained_facet(const Facet& f)
        : _facet(f)
    {
    }

    constexpr const auto& get_facet() const
    {
        return _helper::get(_facet);
    }

private:

    template <template <class> class, class>
    friend struct stringify::v0::detail::constrained_facet_helper;

    Facet _facet;
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
    constexpr decltype(auto)
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

    constexpr single_facet_pack(const Facet& f) : _facet(f) {}

    constexpr single_facet_pack(const single_facet_pack& r) = default;

    using lowest_tag = LowestTag;
    using highest_tag = LowestTag;

    template <typename OtherLowestTag>
    using rebind = single_facet_pack<OtherLowestTag, Facet>;

    using facets_pack_end<LowestTag>::do_get_facet;

    template <typename>
    constexpr const Facet& do_get_facet
        (const highest_tag&, stringify::v0::facet_category_t<Facet>) const
    {
        return _facet;
    }

private:

    Facet _facet;
};


template
    < typename LowestTag
    , template <class> class Filter
    , typename Facet >
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
        (const ConstrainedFacet& f) : _constrained_facet(f) {};

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
        return _constrained_facet.get_facet();
    }

private:

    ConstrainedFacet _constrained_facet;
};


template <typename LowestTag, typename Facet>
class ref_facet_pack: public facets_pack_end<LowestTag>
{
public:

    constexpr ref_facet_pack(std::reference_wrapper<Facet> facet_ref)
        : _facet_ref(facet_ref)
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
        (const highest_tag&, stringify::v0::facet_category_t<Facet>) const
    {
        return _facet_ref.get();
    }

private:

    std::reference_wrapper<const Facet> _facet_ref;
};


template
    < typename LowestTag
    , template <class> class Filter
    , typename Facet >
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
        : _facet_ref(facet_ref)
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
        return _facet_ref.get().get_facet();
    }

private:

    RefConstrainedFacet _facet_ref;
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

class facets_pack_helper
{
public:

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

private:

    template < typename Facet>
    static constexpr const auto& _as_pack(const Facet& f, std::true_type)
    {
        using fpack_type
            = stringify::v0::detail::single_facet_pack<base_tag, Facet>;
        return reinterpret_cast<const fpack_type&>(f);
    }


    template < typename Facet>
    static constexpr auto _as_pack(const Facet& f, std::false_type)
    {
        using fpack_type
            = stringify::v0::detail::ref_facet_pack<base_tag, const Facet>;
        return fpack_type(f);
    }

public:

    template < typename Facet
             , typename category = stringify::v0::facet_category_t<Facet>
             , typename = std::enable_if_t< ! std::is_void<category>::value> >
    static constexpr decltype(auto) as_pack(const Facet& f)
    {
        return _as_pack<Facet>(f, std::integral_constant<bool, category::by_value>{});
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

}; // class facets_pack_helper

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
    using _impl = stringify::v0::detail::facets_pack_impl<Facets...>;

    friend class stringify::v0::detail::facets_pack_helper;

    template <typename, typename>
    friend class stringify::v0::detail::facets_pack_join;

public:

    constexpr facets_pack(const Facets& ... f)
        : _impl(stringify::v0::detail::pack_impl(f...))
    {
    }

    template <typename FacetCategory, typename InputType>
    constexpr decltype(auto) get_facet() const
    {
        return this->template do_get_facet<InputType>
            (typename _impl::highest_tag(), FacetCategory());
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
    , typename ... Facets >
constexpr decltype(auto) get_facet(const stringify::v0::facets_pack<Facets...>& fp)
{
    return fp.template get_facet<FacetCategory, InputType>();
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP

