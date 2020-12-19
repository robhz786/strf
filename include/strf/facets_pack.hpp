#ifndef STRF_DETAIL_FACETS_PACK_HPP
#define STRF_DETAIL_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp>

namespace strf {

template <typename ... F> class facets_pack;

template <template <class> class Filter, typename FPE>
class constrained_fpe;

namespace detail {

template <typename A, typename... B>
struct cvref_is_same: std::false_type {};

template <typename A, typename B>
struct cvref_is_same<A, B>
    : std::is_same< strf::detail::remove_cvref_t<A>
                  , strf::detail::remove_cvref_t<B> >
{
};

template <typename A, typename... B>
constexpr bool cvref_is_same_v = cvref_is_same<A, B...>::value;

template <typename F, typename = typename F::category>
constexpr STRF_HD bool has_category_member_type(F*)
{
    return true;
}
constexpr STRF_HD bool has_category_member_type(...)
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
= strf::detail::category_member_type_or_void_2
    < F, strf::detail::has_category_member_type((F*)0) >;

template <typename T, bool v = T::constrainable>
static constexpr bool get_constrainable(T*)
{
    return v;
}

static constexpr bool get_constrainable(...)
{
    return true;
}

template <typename FPE> class fpe_traits;
template <typename Rank, typename FPE> class fpe_wrapper;

} // namespace detail

template <typename F>
struct facet_traits
{
    using helper_ = strf::detail::category_member_type_or_void<F>;

    using category = typename helper_::type;
};

template <typename F>
struct facet_traits<const F>
{
    using category = typename strf::facet_traits<F>::category;
};

template <typename F>
using facet_category
= typename strf::facet_traits<F>::category;

template <typename F>
constexpr bool facet_constrainable
= strf::detail::get_constrainable
    ((strf::facet_category<F>*)0);

template <typename FPE>
constexpr bool is_constrainable_v
= strf::detail::fpe_traits<FPE>::is_constrainable;

namespace detail{

template <typename Cat, typename Tag, typename FPE>
constexpr bool has_facet_v
= strf::detail::fpe_traits<FPE>
    ::template has_facet_v<Cat, Tag>;

template <typename Cat, typename Tag, typename FPE>
constexpr STRF_HD decltype(auto) do_get_facet(const FPE& elm)
{
    using traits = strf::detail::fpe_traits<FPE>;
    return traits::template get_facet<Cat, Tag>(elm);
}


template <typename ... FPE>
class fpe_traits<strf::facets_pack<FPE...>>
{
    using FP_ = strf::facets_pack<FPE...>;

public:

    template <typename Category, typename Tag>
    static constexpr bool has_facet_v
    = strf::detail::fold_or
        <strf::detail::has_facet_v<Category, Tag, FPE>...>;

    template <typename Cat, typename Tag>
    constexpr STRF_HD static decltype(auto) get_facet(const FP_& fp)
    {
        return fp.template get_facet<Cat, Tag>();
    }

    constexpr static bool is_constrainable
      = strf::detail::fold_and
            <strf::is_constrainable_v<FPE>...>;
};

template <typename FPE>
class fpe_traits<const FPE&>
{
public:

    template < typename Cat, typename Tag>
    constexpr static bool has_facet_v
        = strf::detail::has_facet_v<Cat, Tag, FPE>;

    template < typename Cat, typename Tag>
    constexpr STRF_HD static decltype(auto) get_facet(const FPE& r)
    {
        return strf::detail::fpe_traits<FPE>
            ::template get_facet<Cat, Tag>(r);
    }

    constexpr static bool is_constrainable
    = strf::is_constrainable_v<FPE>;
};


template <template <class> class Filter, typename FPE>
class fpe_traits<strf::constrained_fpe<Filter, FPE>>
{
public:

    template <typename Cat, typename Tag>
    constexpr static bool has_facet_v
        = Filter<Tag>::value
       && strf::detail::has_facet_v<Cat, Tag, FPE>;

    template <typename Cat, typename Tag>
    constexpr STRF_HD static decltype(auto) get_facet
        ( const strf::constrained_fpe<Filter, FPE>& cf )
    {
        return strf::detail::fpe_traits<FPE>
            ::template get_facet<Cat, Tag>(cf.get());
    }

    constexpr static bool is_constrainable = true;
};

template <typename F>
class fpe_traits
{
public:

    template <typename Cat, typename>
    constexpr static bool has_facet_v
         = std::is_same<Cat, strf::facet_category<F>>::value;

    template <typename, typename>
    constexpr STRF_HD static const F& get_facet(const F& f)
    {
        return f;
    }

    constexpr static bool is_constrainable
        = strf::facet_constrainable<F>;
};

template <typename Rank, typename Facet>
class fpe_wrapper
{
    using category_ = strf::facet_category<Facet>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;
    constexpr fpe_wrapper() = default;

    template
        < typename F = Facet
        , std::enable_if_t<std::is_copy_constructible<F>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper(const Facet& facet)
        : facet_(facet)
    {
    }

    template
        < typename F = Facet
        , std::enable_if_t<std::is_constructible<Facet, F&&>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper(F&& facet)
        : facet_(std::forward<F>(facet))
    {
    }

    template <typename Tag>
    constexpr STRF_HD const auto& do_get_facet
        ( const Rank&
        , strf::tag<category_>
        , std::true_type ) const
    {
        return facet_;
    }

private:
    Facet facet_;
};

template < typename Rank
         , template <class> class Filter
         , typename FPE >
class fpe_wrapper<Rank, strf::constrained_fpe<Filter, FPE>>
{

    template <typename Category, typename Tag>
    static constexpr bool has_facet_v_
         = Filter<Tag>::value
        && strf::detail::has_facet_v<Category, Tag, FPE>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;
    constexpr fpe_wrapper() = default;

    template
        < typename F = FPE
        , std::enable_if_t<std::is_copy_constructible<F>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper
        ( const strf::constrained_fpe<Filter, FPE>& cfpe )
        : fpe_(cfpe.get())
    {
    }

    template
        < typename F = FPE
        , std::enable_if_t<std::is_move_constructible<F>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper
        ( strf::constrained_fpe<Filter, FPE>&& cfpe )
        : fpe_(std::move(cfpe.get()))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, has_facet_v_<Category, Tag>> ) const
    {
        return strf::detail::do_get_facet<Category, Tag>(fpe_);
    }

private:

    FPE fpe_;
};

template <typename Rank, typename FPE>
class fpe_wrapper<Rank, const FPE&>
{
public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;

    constexpr STRF_HD fpe_wrapper(const FPE& fpe)
        : fpe_(fpe)
    {
    }

    constexpr STRF_HD fpe_wrapper(FPE&& fpe) = delete;

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant
            < bool
            , strf::detail::has_facet_v<Category, Tag, FPE> > ) const
    {
        return strf::detail::do_get_facet<Category, Tag>(fpe_);
    }

private:

    const FPE& fpe_;
};

template <typename Rank, typename ... FPE>
class fpe_wrapper<Rank, strf::facets_pack<FPE...>>
{
    using fp_type_ = strf::facets_pack<FPE...>;

    template <typename Category, typename Tag>
    static constexpr bool has_facet_v_
    = strf::detail::fold_or
        <strf::detail::has_facet_v<Category, Tag, FPE>...>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;
    constexpr fpe_wrapper() = default;

    template
        < typename FP = fp_type_
        , std::enable_if_t<std::is_copy_constructible<FP>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper(const fp_type_& fp)
        : fp_(fp)
    {
    }

    template
        < typename FP = fp_type_
        , std::enable_if_t<std::is_move_constructible<FP>::value, int> = 0 >
    constexpr STRF_HD fpe_wrapper(fp_type_&& fp)
        : fp_(std::move(fp))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, has_facet_v_<Category, Tag>> ) const
    {
        return fp_.template get_facet<Category, Tag>();
    }

private:

    fp_type_ fp_;
};

#if defined (__cpp_variadic_using)

template <typename FPEWrappersList, typename... FPE>
class facets_pack_base;

template <typename... FPEWrappers, typename... FPE>
class facets_pack_base<strf::tag<FPEWrappers...>, FPE...>
    : private FPEWrappers ...
{
    static_assert(sizeof...(FPE) != 0, "");

public:

    constexpr facets_pack_base(const facets_pack_base&) = default;
    constexpr facets_pack_base(facets_pack_base&&) = default;
    constexpr facets_pack_base() = default;

    template
        < typename... U
        , std::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && strf::detail::fold_and<std::is_constructible<FPE, U>::value...>
           && ! strf::detail::cvref_is_same_v<facets_pack_base, U...>
            , int > = 0 >
    constexpr STRF_HD facets_pack_base(U&& ... fpe)
        : FPEWrappers(std::forward<U>(fpe))...
    {
    }

    using FPEWrappers::do_get_facet ...;
};

template <typename RankNumSeq, typename ... FPE>
struct facets_pack_base_trait;

template <std::size_t ... RankNum, typename ... FPE>
struct facets_pack_base_trait<std::index_sequence<RankNum...>, FPE...>
{
    using type = facets_pack_base
        < strf::tag
            < strf::detail::fpe_wrapper
                 < strf::rank<RankNum>
                 , FPE >
            ... >
        , FPE ... >;
};

template <typename ... FPE>
using facets_pack_base_t
 = typename strf::detail::facets_pack_base_trait
    < std::make_index_sequence<sizeof...(FPE)>
    , FPE ... >
   :: type;

#else  // defined (__cpp_variadic_using)

template <std::size_t RankN, typename ... FPE>
class facets_pack_base;

template <std::size_t RankN>
class facets_pack_base<RankN>
{
public:

    constexpr STRF_HD void do_get_facet() const
    {
    }
};

template <std::size_t RankN, typename TipFPE, typename ... OthersFPE>
class facets_pack_base<RankN, TipFPE, OthersFPE...>
    : public strf::detail::fpe_wrapper
        < strf::rank<RankN>, TipFPE >
    , public strf::detail::facets_pack_base
        < RankN + 1, OthersFPE...>
{
    using base_tip_fpe_ = strf::detail::fpe_wrapper
        < strf::rank<RankN>, TipFPE >;

    using base_others_fpe_ = strf::detail::facets_pack_base
        < RankN + 1, OthersFPE...>;

public:

    constexpr facets_pack_base(const facets_pack_base&) = default;
    constexpr facets_pack_base(facets_pack_base&&) = default;
    constexpr facets_pack_base() = default;

    template< typename T = TipFPE
            , typename B = base_others_fpe_
            , std::enable_if_t
                < std::is_copy_constructible<T>::value
               && std::is_constructible<B, const OthersFPE&...>::value
                , int > = 0 >
    constexpr STRF_HD facets_pack_base(const TipFPE& tip, const OthersFPE& ... others)
        : base_tip_fpe_(tip)
        , base_others_fpe_(others...)
    {
    }

    template< typename Tip
            , typename ... Others
            , std::enable_if_t
                < sizeof...(Others) == sizeof...(OthersFPE)
               && std::is_constructible<TipFPE, Tip&&>::value
               && std::is_constructible<base_others_fpe_, Others&&...>::value
                , int > = 0 >
    constexpr STRF_HD facets_pack_base(Tip&& tip, Others&& ... others)
        : base_tip_fpe_(std::forward<Tip>(tip))
        , base_others_fpe_(std::forward<Others>(others)...)
    {
    }

    using base_tip_fpe_::do_get_facet;
    using base_others_fpe_::do_get_facet;
};


template <typename ... FPE>
using facets_pack_base_t = strf::detail::facets_pack_base<0, FPE...>;

#endif // defined (__cpp_variadic_using) #else

template <typename T>
struct pack_arg // rvalue reference
{
    using elem_type = std::remove_cv_t<T>;

    static STRF_HD constexpr T&& forward(T& arg)
    {
        return static_cast<T&&>(arg);
    }
};

template <typename T>
struct pack_arg<T&>
{
    using elem_type = std::remove_cv_t<T>;

    static STRF_HD constexpr const T& forward(const T& arg)
    {
        return arg;
    }
};

} // namespace detail

template <template <class> class Filter, typename FPE>
class constrained_fpe
{
public:

    constexpr constrained_fpe(const constrained_fpe&) = default;
    constexpr constrained_fpe(constrained_fpe&&) = default;
    constexpr constrained_fpe() = default;

    static_assert(strf::is_constrainable_v<FPE>, "FPE not constrainable");

    template
        < typename U
        , std::enable_if_t
            < ! std::is_same<constrained_fpe, strf::detail::remove_cvref_t<U>>::value
           && std::is_constructible<FPE, U>::value
            , int > = 0 >
    constexpr STRF_HD explicit constrained_fpe(U&& arg)
        : fpe_(std::forward<U>(arg))
    {
    }

    constexpr STRF_HD const FPE& get() const
    {
        return fpe_;
    }

    constexpr STRF_HD FPE& get()
    {
        return fpe_;
    }

private:

    FPE fpe_;
};


template <>
class facets_pack<>
{
public:

    template <typename Category, typename Tag>
    constexpr STRF_HD decltype(auto) get_facet() const
    {
        return Category::get_default();
    }
};

template <typename ... FPE>
class facets_pack: private strf::detail::facets_pack_base_t<FPE...>
{
    using base_type_ = strf::detail::facets_pack_base_t<FPE...>;
    using base_type_::do_get_facet;

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const strf::absolute_lowest_rank&
        , strf::tag<Category>
        , ... ) const
    {
        return Category::get_default();
    }

public:

    constexpr facets_pack(const facets_pack&) = default;
    constexpr facets_pack(facets_pack&&) = default;
    constexpr facets_pack() = default;

    template
        < typename... U
        , std::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && strf::detail::fold_and<std::is_constructible<FPE, U>::value...>
           && ! strf::detail::cvref_is_same_v<facets_pack, U...>
            , int > = 0 >
    constexpr STRF_HD explicit facets_pack(U&&... fpe)
        : base_type_(std::forward<U>(fpe)...)
    {
    }

    template <typename Category, typename Tag>
    constexpr STRF_HD decltype(auto) get_facet() const
    {
        return this->template do_get_facet<Tag>
            ( strf::rank<sizeof...(FPE)>()
            , strf::tag<Category>()
            , std::true_type() );
    }
};

template <typename ... T>
constexpr STRF_HD strf::facets_pack<strf::detail::remove_cvref_t<T>...>
    pack(T&& ... args)
{
    return strf::facets_pack<strf::detail::remove_cvref_t<T>...>
        { std::forward<T>(args)... };
}

template <template <class> class Filter, typename T>
constexpr STRF_HD strf::constrained_fpe<Filter, strf::detail::remove_cvref_t<T>>
    constrain(T&& x)
{
    return strf::constrained_fpe<Filter, strf::detail::remove_cvref_t<T>>
        { std::forward<T>(x) };
}

template
    < typename FacetCategory
    , typename Tag
    , typename ... FPE >
constexpr STRF_HD decltype(auto) get_facet(const strf::facets_pack<FPE...>& fp)
{
    return fp.template get_facet<FacetCategory, Tag>();
}

} // namespace strf

#endif  // STRF_DETAIL_FACETS_PACK_HPP

