#ifndef STRF_DETAIL_FACETS_PACK_HPP
#define STRF_DETAIL_FACETS_PACK_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
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

template <typename T>
struct has_category_member_type
{
    template <typename U, typename = typename U::category>
    static STRF_HD std::true_type test(U&)
    {
        return {};
    }
    template <typename U>
    static STRF_HD std::false_type test(...)
    {
        return {};
    }

    constexpr static bool value = decltype(test<T>(std::declval<T&>()))::value;
};

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
    < F, has_category_member_type<F>::value >;

template <typename T, bool v = T::constrainable>
constexpr STRF_HD bool get_constrainable(T*)
{
    return v;
}

constexpr STRF_HD bool get_constrainable(...)
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

namespace detail {

template <typename T>
struct is_facet_helper
{
    template < typename U
             , typename C = strf::facet_category<U>
             , typename F = strf::detail::remove_cvref_t<decltype(C::get_default())>
             , typename C2 = strf::facet_category<F> >
    static STRF_HD std::is_same<C, C2> test_(U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using result = decltype(test_<T>((T*)0));
};

template <typename T>
using is_facet = typename is_facet_helper<strf::detail::remove_cvref_t<T>>::result;

template <typename T>
struct is_fpe: is_facet<T>
{
};

template <typename... T>
struct is_fpe<strf::facets_pack<T...>> : std::true_type
{
};

template <template <class> class Filter, typename FPE>
struct is_fpe<strf::constrained_fpe<Filter, FPE>> : std::true_type
{
};

} // namespace detail

template <typename FPE>
constexpr STRF_HD bool is_constrainable() noexcept
{
    return strf::detail::fpe_traits<FPE>::is_constrainable;
}
template <typename Facet>
constexpr STRF_HD bool is_constrainable_facet() noexcept
{
    return strf::detail::get_constrainable((strf::facet_category<Facet>*)0);
}

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename FPE>
constexpr bool is_constrainable_v = is_constrainable<FPE>();

template <typename F>
constexpr bool facet_constrainable = is_constrainable_facet<F>();

#endif

namespace detail {

template <typename Cat, typename Tag, typename FPE>
constexpr STRF_HD bool has_facet() noexcept
{
    return strf::detail::fpe_traits<FPE>::template has_facet<Cat, Tag>();
}

template < typename Cat
         , typename Tag
         , typename FPE
         , typename FPETraits = strf::detail::fpe_traits<FPE> >
constexpr STRF_HD auto do_get_facet(const FPE& elm)
    -> decltype(FPETraits::template get_facet<Cat, Tag>(elm))
{
    return FPETraits::template get_facet<Cat, Tag>(elm);
}

template <typename... FPE>
class fpe_traits<strf::facets_pack<FPE...>>
{
    using fp_ = strf::facets_pack<FPE...>;

public:

    template <typename Category, typename Tag>
    static constexpr STRF_HD bool has_facet() noexcept
    {
        return strf::detail::fold_or<strf::detail::has_facet<Category, Tag, FPE>()...>::value;
    }

    template <typename Cat, typename Tag>
    constexpr STRF_HD static auto get_facet(const fp_& fp)
        -> decltype(std::declval<fp_>().template get_facet<Cat, Tag>())
    {
        return fp.template get_facet<Cat, Tag>();
    }

    constexpr static bool is_constrainable
      = strf::detail::fold_and<strf::is_constrainable<FPE>()...>::value;
};

template <typename FPE>
class fpe_traits<const FPE&>
{
    using fpe_traits_ = strf::detail::fpe_traits<FPE>;
public:

    template < typename Cat, typename Tag>
    constexpr static STRF_HD bool has_facet() noexcept
    {
        return strf::detail::has_facet<Cat, Tag, FPE>();
    }

    template <typename Cat, typename Tag>
    constexpr STRF_HD static auto get_facet(const FPE& r)
        -> decltype(fpe_traits_::template get_facet<Cat, Tag>(std::declval<const FPE&>()))
    {
        return fpe_traits_::template get_facet<Cat, Tag>(r);
    }

    constexpr static bool is_constrainable
    = strf::is_constrainable<FPE>();
};


template <template <class> class Filter, typename FPE>
class fpe_traits<strf::constrained_fpe<Filter, FPE>>
{
    using unfiltered_traits_ = strf::detail::fpe_traits<FPE>;
public:

    template <typename Cat, typename Tag>
    constexpr static STRF_HD bool has_facet() noexcept
    {
        return Filter<Tag>::value && strf::detail::has_facet<Cat, Tag, FPE>();
    }

    template <typename Cat, typename Tag>
    constexpr STRF_HD static auto
    get_facet( const strf::constrained_fpe<Filter, FPE>& cf )
        -> decltype(unfiltered_traits_::template get_facet<Cat, Tag>(cf.get()))
    {
        return unfiltered_traits_::template get_facet<Cat, Tag>(cf.get());
    }

    constexpr static bool is_constrainable = true;
};

template <typename F>
class fpe_traits
{
public:

    template <typename Cat, typename>
    constexpr static STRF_HD bool has_facet() noexcept
    {
        return std::is_same<Cat, strf::facet_category<F>>::value;
    }

    template <typename, typename>
    constexpr STRF_HD static const F& get_facet(const F& f)
    {
        return f;
    }

    constexpr static bool is_constrainable = strf::is_constrainable_facet<F>();
};

template <typename Rank, typename Facet>
class fpe_wrapper
{
    using category_ = strf::facet_category<Facet>;

public:

    constexpr fpe_wrapper() = default;

    template
        < typename F = Facet
        , strf::detail::enable_if_t<std::is_copy_constructible<F>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper(const Facet& facet)
        noexcept(noexcept(Facet(std::declval<const F&>())))
        : facet_(facet)
    {
    }

    template
        < typename F = Facet
        , strf::detail::enable_if_t<std::is_constructible<Facet, F&&>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper(F&& facet)
        noexcept(noexcept(Facet(std::declval<F>())))
        : facet_(std::forward<F>(facet))
    {
    }

    template <typename Tag>
    constexpr STRF_HD const Facet& do_get_facet
        ( const Rank&
        , strf::tag<category_>
        , std::true_type ) const
    {
        return facet_;
    }

    constexpr STRF_HD const Facet& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> ) const
    {
        return facet_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD Facet& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> )
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
    static constexpr STRF_HD bool has_facet_() noexcept
    {
        return Filter<Tag>::value && strf::detail::has_facet<Category, Tag, FPE>();
    }

public:

    constexpr fpe_wrapper() = default;

    template
        < typename F = FPE
        , strf::detail::enable_if_t<std::is_copy_constructible<F>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper
        ( const strf::constrained_fpe<Filter, FPE>& cfpe )
        noexcept(noexcept(F(std::declval<const F&>())))
        : fpe_(cfpe.get())
    {
    }

    template
        < typename F = FPE
        , strf::detail::enable_if_t<std::is_move_constructible<F>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper
        ( strf::constrained_fpe<Filter, FPE>&& cfpe )
        noexcept(noexcept(F(std::declval<F>())))
        : fpe_(std::move(cfpe.get()))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD
    STRF_DECLTYPE_AUTO((strf::detail::do_get_facet<Category, Tag>(std::declval<const FPE&>())))
    do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, has_facet_<Category, Tag>()> ) const
    {
        return strf::detail::do_get_facet<Category, Tag>(fpe_);
    }

    constexpr STRF_HD const FPE& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> ) const
    {
        return fpe_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD FPE& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> )
    {
        return fpe_;
    }

private:

    FPE fpe_;
};

template <typename Rank, typename FPE>
class fpe_wrapper<Rank, const FPE&>
{
public:

    constexpr STRF_HD explicit fpe_wrapper(const FPE& fpe) noexcept
        : fpe_(fpe)
    {
    }

    constexpr STRF_HD fpe_wrapper(FPE&& fpe) = delete;

    template <typename Tag, typename Category>
    constexpr STRF_HD
    STRF_DECLTYPE_AUTO((strf::detail::do_get_facet<Category, Tag>(*(const FPE*)0)))
    do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant
            < bool
            , strf::detail::has_facet<Category, Tag, FPE>() > ) const
    {
        return strf::detail::do_get_facet<Category, Tag>(fpe_);
    }

    constexpr STRF_HD const FPE& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> ) const
    {
        return fpe_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD FPE& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> )
    {
        return fpe_;
    }

private:

    const FPE& fpe_;
};

template <typename Rank, typename ... FPE>
class fpe_wrapper<Rank, strf::facets_pack<FPE...>>
{
    using fp_type_ = strf::facets_pack<FPE...>;

    template <typename Category, typename Tag>
    static constexpr STRF_HD bool has_facet_() noexcept
    {
        return strf::detail::fold_or<strf::detail::has_facet<Category, Tag, FPE>()...>::value;
    }

public:

    constexpr fpe_wrapper() = default;

    template
        < typename FP = fp_type_
        , strf::detail::enable_if_t<std::is_copy_constructible<FP>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper(const fp_type_& fp)
        noexcept(noexcept(fp_type_(std::declval<const fp_type_&>())))
        : fp_(fp)
    {
    }

    template
        < typename FP = fp_type_
        , strf::detail::enable_if_t<std::is_move_constructible<FP>::value, int> = 0 >
    constexpr STRF_HD explicit fpe_wrapper(fp_type_&& fp)
        noexcept(noexcept(fp_type_(std::declval<fp_type_>())))
        : fp_(std::move(fp))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD
    STRF_DECLTYPE_AUTO((((const fp_type_*)0)->template get_facet<Category, Tag>()))
    do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, has_facet_<Category, Tag>()> ) const
    {
        return fp_.template get_facet<Category, Tag>();
    }

    constexpr STRF_HD const fp_type_& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> ) const
    {
        return fp_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD fp_type_& do_get_element
        ( std::integral_constant<std::size_t, Rank::number> )
    {
        return fp_;
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

    constexpr facets_pack_base() = default;

    template
        < typename... U
        , strf::detail::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && strf::detail::fold_and<std::is_constructible<FPE, U>::value...>::value
           && ! strf::detail::cvref_is_same<facets_pack_base, U...>::value
            , int > = 0 >
    constexpr STRF_HD explicit facets_pack_base(U&& ... fpe)
        noexcept(strf::detail::fold_and
                 < noexcept(FPEWrappers(std::declval<U>()))... >::value)
        : FPEWrappers(std::forward<U>(fpe))...
    {
    }

    using FPEWrappers::do_get_facet... ;
    using FPEWrappers::do_get_element... ;
};

template <typename RankNumSeq, typename... FPE>
struct facets_pack_base_trait;

template <std::size_t... RankNum, typename... FPE>
struct facets_pack_base_trait<strf::detail::index_sequence<RankNum...>, FPE...>
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
    < strf::detail::make_index_sequence<sizeof...(FPE)>
    , FPE ... >
   :: type;

#else  // defined (__cpp_variadic_using)

template <std::size_t RankN, typename ... FPE>
class facets_pack_base;

template <std::size_t RankN>
class facets_pack_base<RankN>
{
public:

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void do_get_facet() const
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void do_get_element() const
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

    constexpr facets_pack_base() = default;

    template< typename T = TipFPE
            , typename B = base_others_fpe_
            , strf::detail::enable_if_t
                < std::is_copy_constructible<T>::value
               && std::is_constructible<B, const OthersFPE&...>::value
                , int > = 0 >
    constexpr STRF_HD facets_pack_base(const TipFPE& tip, const OthersFPE& ... others)
        noexcept( noexcept(base_tip_fpe_(std::declval<const TipFPE&>()))
               && noexcept(base_others_fpe_(std::declval<const OthersFPE&>()...)) )
        : base_tip_fpe_(tip)
        , base_others_fpe_(others...)
    {
    }

    template< typename Tip
            , typename ... Others
            , strf::detail::enable_if_t
                < sizeof...(Others) == sizeof...(OthersFPE)
               && std::is_constructible<TipFPE, Tip&&>::value
               && std::is_constructible<base_others_fpe_, Others&&...>::value
                , int > = 0 >
    constexpr STRF_HD facets_pack_base(Tip&& tip, Others&& ... others)
        noexcept( noexcept(base_tip_fpe_(std::declval<Tip>()))
               && noexcept(base_others_fpe_(std::declval<Others>()...)) )
        : base_tip_fpe_(std::forward<Tip>(tip))
        , base_others_fpe_(std::forward<Others>(others)...)
    {
    }

    using base_tip_fpe_::do_get_facet;
    using base_tip_fpe_::do_get_element;
    using base_others_fpe_::do_get_facet;
    using base_others_fpe_::do_get_element;
};


template <typename ... FPE>
using facets_pack_base_t = strf::detail::facets_pack_base<0, FPE...>;

#endif // defined (__cpp_variadic_using) #else

template <typename T>
struct pack_arg // rvalue reference
{
    using elem_type = strf::detail::remove_cv_t<T>;

    static STRF_HD constexpr T&& forward(T& arg)
    {
        return static_cast<T&&>(arg);
    }
};

template <typename T>
struct pack_arg<T&>
{
    using elem_type = strf::detail::remove_cv_t<T>;

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

    constexpr constrained_fpe() = default;

    static_assert(strf::is_constrainable<FPE>(), "FPE not constrainable");

    template
        < typename U
        , strf::detail::enable_if_t
            < ! std::is_same<constrained_fpe, strf::detail::remove_cvref_t<U>>::value
           && std::is_constructible<FPE, U>::value
            , int > = 0 >
    constexpr STRF_HD explicit constrained_fpe(U&& arg)
        noexcept(noexcept(FPE(std::declval<U>())))
        : fpe_(std::forward<U>(arg))
    {
    }

    constexpr STRF_HD const FPE& get() const
    {
        return fpe_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD FPE& get()
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
    constexpr STRF_HD STRF_DECLTYPE_AUTO(Category::get_default()) get_facet() const
    {
        return Category::get_default();
    }
    template <typename Category, typename Tag>
    constexpr STRF_HD STRF_DECLTYPE_AUTO(Category::get_default()) use_facet() const
    {
        return Category::get_default();
    }
};


struct merge_1st {}; // tag type

template <typename ... FPE>
class facets_pack: private strf::detail::facets_pack_base_t<FPE...>
{
    using base_type_ = strf::detail::facets_pack_base_t<FPE...>;
    using base_type_::do_get_facet;
    using base_type_::do_get_element;

    template <typename Tag, typename Category>
    constexpr STRF_HD STRF_DECLTYPE_AUTO(Category::get_default()) do_get_facet
        ( const strf::absolute_lowest_rank&
        , strf::tag<Category>
        , ... ) const
    {
        return Category::get_default();
    }

    template <typename...>
    friend class facets_pack;

public:

    constexpr facets_pack() = default;

    template
        < typename... U
        , strf::detail::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && strf::detail::fold_and<std::is_constructible<FPE, U>::value...>::value
           && ! strf::detail::cvref_is_same<facets_pack, U...>::value
            , int > = 0 >
    constexpr STRF_HD explicit facets_pack(U&&... fpe)
        noexcept(noexcept(base_type_(std::declval<U>()...)))
        : base_type_(std::forward<U>(fpe)...)
    {
    }

    template <typename Category, typename Tag>
    constexpr STRF_HD auto get_facet() const
        -> decltype( this->template do_get_facet<Tag>
                      ( strf::rank<sizeof...(FPE)>()
                      , strf::tag<Category>()
                      , std::true_type() ) )
    {
        return this->template do_get_facet<Tag>
            ( strf::rank<sizeof...(FPE)>()
            , strf::tag<Category>()
            , std::true_type() );
    }
    template <typename Category, typename Tag>
    constexpr STRF_HD auto use_facet() const
        -> decltype( this->template do_get_facet<Tag>
                       ( strf::rank<sizeof...(FPE)>()
                       , strf::tag<Category>()
                       , std::true_type() ) )
    {
        return this->template do_get_facet<Tag>
            ( strf::rank<sizeof...(FPE)>()
            , strf::tag<Category>()
            , std::true_type() );
    }

    template <std::size_t Index>
    constexpr STRF_HD auto get_element() const
        -> decltype(this->do_get_element(std::integral_constant<std::size_t, Index>{}))
    {
        return this->do_get_element(std::integral_constant<std::size_t, Index>{});
    }

    template <std::size_t Index>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD auto get_element()
        -> decltype(this->do_get_element(std::integral_constant<std::size_t, Index>{}))
    {
        return this->do_get_element(std::integral_constant<std::size_t, Index>{});
    }

private:

    template < typename... PackedFpes
             , std::size_t... I
             , typename... OtherFpes
             , strf::detail::enable_if_t<sizeof...(I) != 0, int> = 0 >
    constexpr STRF_HD facets_pack
        ( strf::detail::index_sequence<I...>
        , const facets_pack<PackedFpes...>& fp
        , OtherFpes&&... fpes )
        : facets_pack(fp.template get_element<I>()..., (OtherFpes&&)fpes...)
    {
    }

    template < typename... PackedFpes
             , std::size_t... I
             , typename... OtherFpes
             , strf::detail::enable_if_t<sizeof...(I) != 0, int> = 0 >
    constexpr STRF_HD facets_pack
        ( strf::detail::index_sequence<I...>
        , facets_pack<PackedFpes...>&& fp
        , OtherFpes&&... fpes )
        : facets_pack(std::move(fp.template get_element<I>())..., (OtherFpes&&)fpes...)
    {
    }

    template <typename... Fpes>
    constexpr STRF_HD facets_pack
        ( strf::detail::index_sequence<>
        , facets_pack<>
        , Fpes&&... fpes )
        : base_type_((Fpes&&)fpes...)
    {
    }

public:

    template <typename... PackedFpes, typename... OtherFpes>
    constexpr STRF_HD facets_pack
        ( strf::merge_1st
        , const facets_pack<PackedFpes...>& fp
        , OtherFpes&&... fpes )
        : facets_pack( strf::detail::make_index_sequence<sizeof...(PackedFpes)>()
                     , fp
                     , (OtherFpes&&)fpes...)
    {
    }

    template <typename... PackedFpes, typename... OtherFpes>
    constexpr STRF_HD facets_pack
        ( strf::merge_1st
        , facets_pack<PackedFpes...>&& fp
        , OtherFpes&&... fpes)
        : facets_pack( strf::detail::make_index_sequence<sizeof...(PackedFpes)>()
                     , std::move(fp)
                     , (OtherFpes&&)fpes... )
    {
    }
};


template <typename... Args>
constexpr STRF_HD strf::facets_pack<strf::detail::remove_cvref_t<Args>...>
    pack(Args&&... args)
    noexcept(noexcept(strf::facets_pack<strf::detail::remove_cvref_t<Args>...>
                      {std::declval<Args>()...}))
{
    return strf::facets_pack<strf::detail::remove_cvref_t<Args>...>
        { std::forward<Args>(args)... };
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
constexpr STRF_HD auto use_facet(const strf::facets_pack<FPE...>& fp)
    -> decltype(fp.template use_facet<FacetCategory, Tag>())
{
    return fp.template use_facet<FacetCategory, Tag>();
}

template
    < typename FacetCategory
    , typename Tag
    , typename ... FPE >
STRF_DEPRECATED_MSG("Use strf::use_facet instead")
constexpr STRF_HD auto get_facet(const strf::facets_pack<FPE...>& fp)
    -> decltype(fp.template use_facet<FacetCategory, Tag>())
{
    return fp.template use_facet<FacetCategory, Tag>();
}

template
    < typename FacetCategory
    , typename Tag
    , typename FPack >
using facet_type_in_pack = strf::detail::remove_cv_t
    < decltype(std::declval<FPack>(). template use_facet<FacetCategory, Tag>()) >;


} // namespace strf

#endif  // STRF_DETAIL_FACETS_PACK_HPP

