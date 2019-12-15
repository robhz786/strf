#ifndef STRF_DETAIL_FACETS_PACK_HPP
#define STRF_DETAIL_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>
#include <utility>
#include <type_traits>
#include <functional>

STRF_NAMESPACE_BEGIN

template <typename ... F> class facets_pack;

template <template <class> class Filter, typename FPE>
class constrained_fpe;

namespace detail {

template <typename ... T>
struct tmp_list
{
    template <typename E>
    using push_front = strf::detail::tmp_list<E, T...>;

    template <typename E>
    using push_back = strf::detail::tmp_list<T..., E>;
};

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

template <typename T, bool v = T::store_by_value>
static constexpr bool by_value_getter(T*)
{
    return v;
}

static constexpr bool by_value_getter(...)
{
    return true;
}

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
class facet_trait
{
    using _helper = strf::detail::category_member_type_or_void<F>;

public:

    using category = typename _helper::type;
    constexpr static bool store_by_value
    = strf::detail::by_value_getter((F*)0);
};

template <typename F>
constexpr bool facet_stored_by_value
= strf::detail::by_value_getter((strf::facet_trait<F>*)0);

template <typename F>
class facet_trait<const F>
{
public:
    using category = typename strf::facet_trait<F>::category;
    const static bool store_by_value = strf::facet_trait<F>::store_by_value;
};

template <typename F>
using facet_category
= typename strf::facet_trait<F>::category;

template <typename F>
constexpr bool facet_constrainable
= strf::detail::get_constrainable
    ((strf::facet_category<F>*)0);

template <typename FPE>
constexpr bool is_constrainable_v
= strf::detail::fpe_traits<FPE>::is_constrainable;

namespace detail{

template <bool enable> struct fp_args
{
};

template <> struct fp_args<true>
{
    template <typename ... T>
    static constexpr bool copy_constructible_v
    = detail::fold_and<std::is_copy_constructible<T>::value...>;
};

template <typename Cat, typename Tag, typename FPE>
constexpr bool has_facet_v
= strf::detail::fpe_traits<FPE>
    ::template has_facet_v<Cat, Tag>;

template <typename Cat, typename Tag, typename FPE>
constexpr STRF_HD STRF_HD decltype(auto) do_get_facet(const FPE& elm)
{
    using traits = strf::detail::fpe_traits<FPE>;
    return traits::template get_facet<Cat, Tag>(elm);
}


template <typename ... FPE>
class fpe_traits<strf::facets_pack<FPE...>>
{
    using _FP = strf::facets_pack<FPE...>;

public:

    template <typename Category, typename Tag>
    static constexpr bool has_facet_v
    = strf::detail::fold_or
        <strf::detail::has_facet_v<Category, Tag, FPE>...>;

    template <typename Cat, typename Tag>
    constexpr STRF_HD static decltype(auto) get_facet(const _FP& fp)
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
    using _category = strf::facet_category<Facet>;

public:

    constexpr STRF_HD fpe_wrapper(const fpe_wrapper&) = default;

    constexpr STRF_HD fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename F = Facet
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value> >
    constexpr STRF_HD fpe_wrapper(const Facet& facet)
        : _facet(facet)
    {
    }

    template
        < typename F = Facet
        , typename = std::enable_if_t
            < std::is_constructible<Facet, F&&>::value > >
    constexpr STRF_HD fpe_wrapper(F&& facet)
        : _facet(std::forward<F>(facet))
    {
    }

    template <typename Tag>
    constexpr STRF_HD const auto& do_get_facet
        ( const Rank&
        , strf::tag<_category>
        , std::true_type ) const
    {
        return _facet;
    }

private:
    static_assert( facet_stored_by_value<Facet>
                 , "This facet must not be stored by value" );
    const Facet _facet;
};

template < typename Rank
         , template <class> class Filter
         , typename FPE >
class fpe_wrapper<Rank, strf::constrained_fpe<Filter, FPE>>
{

    template <typename Category, typename Tag>
    static constexpr bool _has_facet_v
         = Filter<Tag>::value
        && strf::detail::has_facet_v<Category, Tag, FPE>;

public:

    constexpr STRF_HD fpe_wrapper(const fpe_wrapper&) = default;
    constexpr STRF_HD fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value > >
    constexpr STRF_HD fpe_wrapper
        ( const strf::constrained_fpe<Filter, FPE>& cfpe )
        : _fpe(cfpe.get())
    {
    }

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_move_constructible<F>::value > >
    constexpr STRF_HD fpe_wrapper
        ( strf::constrained_fpe<Filter, FPE>&& cfpe )
        : _fpe(std::move(cfpe.get()))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, _has_facet_v<Category, Tag>> ) const
    {
        return strf::detail::do_get_facet<Category, Tag>(_fpe);
    }

private:

    FPE _fpe;
};

template <typename Rank, typename FPE>
class fpe_wrapper<Rank, const FPE&>
{
public:

    constexpr STRF_HD fpe_wrapper(const fpe_wrapper&) = default;
    constexpr STRF_HD fpe_wrapper(fpe_wrapper&&) = default;

    constexpr STRF_HD fpe_wrapper(const FPE& fpe)
        : _fpe(fpe)
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
        return strf::detail::do_get_facet<Category, Tag>(_fpe);
    }

private:

    const FPE& _fpe;
};

template <typename Rank, typename ... FPE>
class fpe_wrapper<Rank, strf::facets_pack<FPE...>>
{
    using _fp_type = strf::facets_pack<FPE...>;

    template <typename Category, typename Tag>
    static constexpr bool _has_facet_v
    = strf::detail::fold_or
        <strf::detail::has_facet_v<Category, Tag, FPE>...>;

public:

    constexpr STRF_HD fpe_wrapper(const fpe_wrapper&) = default;
    constexpr STRF_HD fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename FP = _fp_type
        , typename = std::enable_if_t<std::is_copy_constructible<FP>::value> >
    constexpr STRF_HD fpe_wrapper(const _fp_type& fp)
        : _fp(fp)
    {
    }

    template
        < typename FP = _fp_type
        , typename = std::enable_if_t<std::is_move_constructible<FP>::value> >
    constexpr STRF_HD fpe_wrapper(_fp_type&& fp)
        : _fp(std::move(fp))
    {
    }

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const Rank&
        , strf::tag<Category>
        , std::integral_constant<bool, _has_facet_v<Category, Tag>> ) const
    {
        return _fp.template get_facet<Category, Tag>();
    }

private:

    _fp_type _fp;
};

#if defined (__cpp_variadic_using)

template <typename FPEWrappersList, typename ... FPE>
class facets_pack_base;

template <typename ... FPEWrappers, typename ... FPE>
class facets_pack_base< strf::detail::tmp_list<FPEWrappers ...>
                      , FPE ... >
    : private FPEWrappers ...
{
    template <typename... T, typename... From>
    constexpr STRF_HD static bool all_constructible_f
        (detail::tmp_list<T...>, detail::tmp_list<From...>)
    {
        return detail::fold_and
            < std::is_constructible<T, From>::value... >;
    }

public:

    constexpr STRF_HD facets_pack_base(const facets_pack_base&) = default;
    constexpr STRF_HD facets_pack_base(facets_pack_base&&) = default;

    template
        < typename WL = detail::tmp_list<FPE...>
        , typename FL = detail::tmp_list<const FPE&...>
        , typename = std::enable_if_t<all_constructible_f(WL{}, FL{})> >
    constexpr STRF_HD facets_pack_base(const FPE& ... fpe)
        : FPEWrappers(fpe) ...
    {
    }

    template< typename... U
            , typename WL = detail::tmp_list<FPE...>
            , typename UL = detail::tmp_list<U&&...>
            , typename = std::enable_if_t
                 < sizeof...(U) == sizeof...(FPE)
                && sizeof...(U) >= 1 >
            , typename = std::enable_if_t<all_constructible_f(WL{}, UL{})> >
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
        < strf::detail::tmp_list
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
    constexpr STRF_HD facets_pack_base(const facets_pack_base&) = default;
    constexpr STRF_HD facets_pack_base(facets_pack_base&&) = default;
    constexpr STRF_HD facets_pack_base() = default;

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
    using _base_tip_fpe = strf::detail::fpe_wrapper
        < strf::rank<RankN>, TipFPE >;

    using _base_others_fpe = strf::detail::facets_pack_base
        < RankN + 1, OthersFPE...>;

public:

    constexpr STRF_HD facets_pack_base(const facets_pack_base&) = default;

    constexpr STRF_HD facets_pack_base(facets_pack_base&& other) = default;

    template< typename T = TipFPE
            , typename B = _base_others_fpe
            , typename = std::enable_if_t
                < std::is_copy_constructible<T>::value
               && std::is_constructible<B, const OthersFPE&...>::value >>
    constexpr STRF_HD facets_pack_base(const TipFPE& tip, const OthersFPE& ... others)
        : _base_tip_fpe(tip)
        , _base_others_fpe(others...)
    {
    }

    template< typename Tip
            , typename ... Others
            , typename = std::enable_if_t
                < sizeof...(Others) == sizeof...(OthersFPE)
               && std::is_constructible<TipFPE, Tip&&>::value
               && std::is_constructible<_base_others_fpe, Others&&...>::value >>
    constexpr STRF_HD facets_pack_base(Tip&& tip, Others&& ... others)
        : _base_tip_fpe(std::forward<Tip>(tip))
        , _base_others_fpe(std::forward<Others>(others)...)
    {
    }

    using _base_tip_fpe::do_get_facet;
    using _base_others_fpe::do_get_facet;
};


template <typename ... FPE>
using facets_pack_base_t = strf::detail::facets_pack_base<0, FPE...>;

#endif // defined (__cpp_variadic_using) #else

template <typename T>
struct pack_arg // rvalue reference
{
    static_assert( strf::facet_stored_by_value<T>
                 , "can't bind lvalue reference to rvalue reference" );

    using elem_type = std::remove_cv_t<T>;

    static constexpr T&& forward(T& arg)
    {
        return static_cast<T&&>(arg);
    }
};

template <typename T>
struct pack_arg<T&>
{
    using elem_type = std::conditional_t
        < strf::facet_stored_by_value<T>
        , std::remove_cv_t<T>
        , const T& >;

    static constexpr const T& forward(const T& arg)
    {
        return arg;
    }
};

template <typename T>
struct pack_arg_ref
{
    using elem_type = const T&;

    static constexpr const T& forward(const std::reference_wrapper<T>& arg)
    {
        return arg.get();
    }
};

template <typename T>
struct pack_arg<std::reference_wrapper<T>>
    : public strf::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<const std::reference_wrapper<T>>
    : public strf::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<std::reference_wrapper<T>&>
    : public strf::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<const std::reference_wrapper<T>&>
    : public strf::detail::pack_arg_ref<T>
{
};

} // namespace detail

template <template <class> class Filter, typename FPE>
class constrained_fpe
{
public:

    static_assert( strf::is_constrainable_v<FPE>
                 , "Not constrainable");

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_default_constructible<F>::value> >
    constexpr STRF_HD explicit constrained_fpe()
    {
    }

    constexpr STRF_HD constrained_fpe(const constrained_fpe&) = default;

    constexpr STRF_HD constrained_fpe(constrained_fpe&& other) = default;

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value> >
    constexpr STRF_HD explicit constrained_fpe(const FPE& f)
        : _fpe(f)
    {
    }

    template
        < typename U
        , typename = std::enable_if_t<std::is_constructible<FPE, U&&>::value> >
    constexpr STRF_HD explicit constrained_fpe(U&& arg)
        : _fpe(std::forward<U>(arg))
    {
    }

    constexpr STRF_HD constrained_fpe& operator=(const constrained_fpe&) = delete;
    constexpr STRF_HD constrained_fpe& operator=(constrained_fpe&&) = delete;

    constexpr STRF_HD const FPE& get() const
    {
        return _fpe;
    }

    constexpr STRF_HD FPE& get()
    {
        return _fpe;
    }

private:
    static_assert
        ( std::is_lvalue_reference<FPE>::value
       || strf::facet_stored_by_value<std::remove_reference_t<FPE>>
        , "This facet must not be stored by value" );

    FPE _fpe; // todo check of facet_stored_by_value
};


template <>
class facets_pack<>
{
public:
    constexpr STRF_HD facets_pack(const facets_pack&) = default;
    constexpr STRF_HD facets_pack(facets_pack&&) = default;
    constexpr STRF_HD facets_pack() = default;

    STRF_HD facets_pack& operator=(const facets_pack&) = delete;
    STRF_HD facets_pack& operator=(facets_pack&&) = delete;

    template <typename Category, typename Tag>
    STRF_HD decltype(auto) get_facet() const
    {
        return Category::get_default();
    }
};

template <typename ... FPE>
class facets_pack: private strf::detail::facets_pack_base_t<FPE...>
{
    using _base_type = strf::detail::facets_pack_base_t<FPE...>;
    using _base_type::do_get_facet;

    template <typename Tag, typename Category>
    constexpr STRF_HD decltype(auto) do_get_facet
        ( const strf::absolute_lowest_rank&
        , strf::tag<Category>
        , ... ) const
    {
        return Category::get_default();
    }

public:

    constexpr STRF_HD facets_pack(const facets_pack&) = default;
    constexpr STRF_HD facets_pack(facets_pack&& other) = default;

    template
        < bool Dummy = true
        , typename = std::enable_if_t
            < detail::fp_args<Dummy>::template copy_constructible_v<FPE...> > >
    constexpr STRF_HD explicit facets_pack(const FPE& ... fpe)
        : _base_type(fpe...)
    {
    }

    template
        < typename ... U
        , typename = std::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && sizeof...(U) == 1
           && detail::fold_and<std::is_constructible<FPE, U&&>::value...> > >
    constexpr STRF_HD explicit facets_pack(U&& ... fpe)
        : _base_type(std::forward<U>(fpe)...)
    {
    }

    STRF_HD facets_pack& operator=(const facets_pack&) = delete;

    STRF_HD facets_pack& operator=(facets_pack&&) = delete;

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
constexpr STRF_HD auto pack(T&& ... args)
{
    return strf::facets_pack
        < typename detail::pack_arg<T>::elem_type ... >
        { detail::pack_arg<T>::forward(args)... };
}

template <template <class> class Filter, typename T>
constexpr STRF_HD auto constrain(T&& x)
{
    return strf::constrained_fpe
        < Filter, typename detail::pack_arg<T>::elem_type >
        ( detail::pack_arg<T>::forward(x) );
}

template
    < typename FacetCategory
    , typename Tag
    , typename ... FPE >
constexpr STRF_HD decltype(auto) get_facet(const strf::facets_pack<FPE...>& fp)
{
    return fp.template get_facet<FacetCategory, Tag>();
}

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_FACETS_PACK_HPP

