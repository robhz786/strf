#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <utility>
#include <type_traits>
#include <functional>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename ... F> class facets_pack;

template <template <class> class Filter, typename FPE>
class constrained_fpe;

namespace detail
{

#if defined(__cpp_fold_expressions)

template <bool ... C> constexpr bool fold_and = (C && ...);
template <bool ... C> constexpr bool fold_or  = (C || ...);

#else

template <bool ... > struct fold_and_impl;
template <bool ... > struct fold_or_impl;

template <> struct fold_and_impl<>
{
    constexpr static bool value = true;
};

template <> struct fold_or_impl<>
{
    constexpr static bool value = false;
};

template <bool C0, bool ... C>
struct fold_and_impl<C0, C...>
{
     constexpr static bool value = fold_and_impl<C...>::value && C0;
};

template <bool C0, bool ... C>
struct fold_or_impl<C0, C...>
{
     constexpr static bool value = fold_or_impl<C...>::value || C0;
};

template <bool ... C> constexpr bool fold_and = fold_and_impl<C...>::value;
template <bool ... C> constexpr bool fold_or = fold_or_impl<C...>::value;

#endif // defined(__cpp_fold_expressions) else

template <typename T>
struct identity
{
    using type = T;
};

template <typename ... T>
struct tmp_list
{
    template <typename E>
    using push_front = stringify::v0::detail::tmp_list<E, T...>;

    template <typename E>
    using push_back = stringify::v0::detail::tmp_list<T..., E>;
};

struct absolute_lowest_rank
{
};

template <std::size_t N>
struct rank: rank<N - 1>
{
};

template <>
struct rank<0>: absolute_lowest_rank
{
};

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
    using _helper = stringify::v0::detail::category_member_type_or_void<F>;

public:

    using category = typename _helper::type;
    constexpr static bool store_by_value
    = stringify::v0::detail::by_value_getter((F*)0);
};

template <typename F>
constexpr bool facet_stored_by_value
= stringify::v0::detail::by_value_getter((stringify::v0::facet_trait<F>*)0);

template <typename F>
class facet_trait<const F>
{
public:
    using category = typename stringify::v0::facet_trait<F>::category;
    const static bool store_by_value = stringify::v0::facet_trait<F>::store_by_value;
};

template <typename F>
using facet_category
= typename stringify::v0::facet_trait<F>::category;

template <typename F>
constexpr bool facet_constrainable
= stringify::v0::detail::get_constrainable
    ((stringify::v0::facet_category<F>*)0);

template <typename FPE>
constexpr bool is_constrainable_v
= stringify::v0::detail::fpe_traits<FPE>::is_constrainable;

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
= stringify::v0::detail::fpe_traits<FPE>
    ::template has_facet_v<Cat, Tag>;

template <typename Cat, typename Tag, typename FPE>
constexpr decltype(auto) do_get_facet(const FPE& elm)
{
    using traits = stringify::v0::detail::fpe_traits<FPE>;
    return traits::template get_facet<Cat, Tag>(elm);
}


template <typename ... FPE>
class fpe_traits<stringify::v0::facets_pack<FPE...>>
{
    using _FP = stringify::v0::facets_pack<FPE...>;

public:

    template <typename Category, typename Tag>
    static constexpr bool has_facet_v
    = stringify::v0::detail::fold_or
        <stringify::v0::detail::has_facet_v<Category, Tag, FPE>...>;

    template <typename Cat, typename Tag>
    constexpr static decltype(auto) get_facet(const _FP& fp)
    {
        return fp.template get_facet<Cat, Tag>();
    }

    constexpr static bool is_constrainable
      = stringify::v0::detail::fold_and
            <stringify::v0::is_constrainable_v<FPE>...>;
};

template <typename FPE>
class fpe_traits<const FPE&>
{
public:

    template < typename Cat, typename Tag>
    constexpr static bool has_facet_v
        = stringify::v0::detail::has_facet_v<Cat, Tag, FPE>;

    template < typename Cat, typename Tag>
    constexpr static decltype(auto) get_facet(const FPE& r)
    {
        return stringify::v0::detail::fpe_traits<FPE>
            ::template get_facet<Cat, Tag>(r);
    }

    constexpr static bool is_constrainable
    = stringify::v0::is_constrainable_v<FPE>;
};


template <template <class> class Filter, typename FPE>
class fpe_traits<stringify::v0::constrained_fpe<Filter, FPE>>
{
public:

    template <typename Cat, typename Tag>
    constexpr static bool has_facet_v
        = Filter<Tag>::value
       && stringify::v0::detail::has_facet_v<Cat, Tag, FPE>;

    template <typename Cat, typename Tag>
    constexpr static decltype(auto) get_facet
        ( const stringify::v0::constrained_fpe<Filter, FPE>& cf )
    {
        return stringify::v0::detail::fpe_traits<FPE>
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
         = std::is_same<Cat, stringify::v0::facet_category<F>>::value;

    template <typename, typename>
    constexpr static const F& get_facet(const F& f)
    {
        return f;
    }

    constexpr static bool is_constrainable
        = stringify::v0::facet_constrainable<F>;
};

template <typename Rank, typename Facet>
class fpe_wrapper
{
    using _category = stringify::v0::facet_category<Facet>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;

    constexpr fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename F = Facet
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value> >
    constexpr fpe_wrapper(const Facet& facet)
        : _facet(facet)
    {
    }

    template
        < typename F = Facet
        , typename = std::enable_if_t
            < std::is_constructible<Facet, F&&>::value > >
    constexpr fpe_wrapper(F&& facet)
        : _facet(std::forward<F>(facet))
    {
    }

    template <typename Tag>
    constexpr const auto& do_get_facet
        ( const Rank&
        , stringify::v0::detail::identity<_category>
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
class fpe_wrapper<Rank, stringify::v0::constrained_fpe<Filter, FPE>>
{

    template <typename Category, typename Tag>
    static constexpr bool _has_facet_v
         = Filter<Tag>::value
        && stringify::v0::detail::has_facet_v<Category, Tag, FPE>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value > >
    constexpr fpe_wrapper
        ( const stringify::v0::constrained_fpe<Filter, FPE>& cfpe )
        : _fpe(cfpe.get())
    {
    }

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_move_constructible<F>::value > >
    constexpr fpe_wrapper
        ( stringify::v0::constrained_fpe<Filter, FPE>&& cfpe )
        : _fpe(std::move(cfpe.get()))
    {
    }

    template <typename Tag, typename Category>
    constexpr decltype(auto) do_get_facet
        ( const Rank&
        , stringify::v0::detail::identity<Category>
        , std::integral_constant<bool, _has_facet_v<Category, Tag>> ) const
    {
        return stringify::v0::detail::do_get_facet<Category, Tag>(_fpe);
    }

private:

    FPE _fpe;
};

template <typename Rank, typename FPE>
class fpe_wrapper<Rank, const FPE&>
{
public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;

    constexpr fpe_wrapper(const FPE& fpe)
        : _fpe(fpe)
    {
    }

    constexpr fpe_wrapper(FPE&& fpe) = delete;

    template <typename Tag, typename Category>
    constexpr decltype(auto) do_get_facet
        ( const Rank&
        , stringify::v0::detail::identity<Category>
        , std::integral_constant
            < bool
            , stringify::v0::detail::has_facet_v<Category, Tag, FPE> > ) const
    {
        return stringify::v0::detail::do_get_facet<Category, Tag>(_fpe);
    }

private:

    const FPE& _fpe;
};

template <typename Rank, typename ... FPE>
class fpe_wrapper<Rank, stringify::v0::facets_pack<FPE...>>
{
    using _fp_type = stringify::v0::facets_pack<FPE...>;

    template <typename Category, typename Tag>
    static constexpr bool _has_facet_v
    = stringify::v0::detail::fold_or
        <stringify::v0::detail::has_facet_v<Category, Tag, FPE>...>;

public:

    constexpr fpe_wrapper(const fpe_wrapper&) = default;
    constexpr fpe_wrapper(fpe_wrapper&&) = default;

    template
        < typename FP = _fp_type
        , typename = std::enable_if_t<std::is_copy_constructible<FP>::value> >
    constexpr fpe_wrapper(const _fp_type& fp)
        : _fp(fp)
    {
    }

    template
        < typename FP = _fp_type
        , typename = std::enable_if_t<std::is_move_constructible<FP>::value> >
    constexpr fpe_wrapper(_fp_type&& fp)
        : _fp(std::move(fp))
    {
    }

    template <typename Tag, typename Category>
    constexpr decltype(auto) do_get_facet
        ( const Rank&
        , stringify::v0::detail::identity<Category>
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
class facets_pack_base< stringify::v0::detail::tmp_list<FPEWrappers ...>
                      , FPE ... >
    : private FPEWrappers ...
{
    template <typename... T, typename... From>
    constexpr static bool all_constructible_f
        (detail::tmp_list<T...>, detail::tmp_list<From...>)
    {
        return detail::fold_and
            < std::is_constructible<T, From>::value... >;
    }

public:

    constexpr facets_pack_base(const facets_pack_base&) = default;
    constexpr facets_pack_base(facets_pack_base&&) = default;

    template
        < typename WL = detail::tmp_list<FPE...>
        , typename FL = detail::tmp_list<const FPE&...>
        , typename = std::enable_if_t<all_constructible_f(WL{}, FL{})> >
    constexpr facets_pack_base(const FPE& ... fpe)
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
    constexpr facets_pack_base(U&& ... fpe)
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
        < stringify::v0::detail::tmp_list
            < stringify::v0::detail::fpe_wrapper
                 < stringify::v0::detail::rank<RankNum>
                 , FPE >
            ... >
        , FPE ... >;
};

template <typename ... FPE>
using facets_pack_base_t
 = typename stringify::v0::detail::facets_pack_base_trait
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
    constexpr facets_pack_base(const facets_pack_base&) = default;
    constexpr facets_pack_base(facets_pack_base&&) = default;
    constexpr facets_pack_base() = default;

    constexpr void do_get_facet() const
    {
    }
};

template <std::size_t RankN, typename TipFPE, typename ... OthersFPE>
class facets_pack_base<RankN, TipFPE, OthersFPE...>
    : public stringify::v0::detail::fpe_wrapper
        < stringify::v0::detail::rank<RankN>, TipFPE >
    , public stringify::v0::detail::facets_pack_base
        < RankN + 1, OthersFPE...>
{
    using _base_tip_fpe = stringify::v0::detail::fpe_wrapper
        < stringify::v0::detail::rank<RankN>, TipFPE >;

    using _base_others_fpe = stringify::v0::detail::facets_pack_base
        < RankN + 1, OthersFPE...>;

public:

    constexpr facets_pack_base(const facets_pack_base&) = default;

    constexpr facets_pack_base(facets_pack_base&& other) = default;

    template< typename T = TipFPE
            , typename B = _base_others_fpe
            , typename = std::enable_if_t
                < std::is_copy_constructible<T>::value
               && std::is_constructible<B, const OthersFPE&...>::value >>
    constexpr facets_pack_base(const TipFPE& tip, const OthersFPE& ... others)
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
    constexpr facets_pack_base(Tip&& tip, Others&& ... others)
        : _base_tip_fpe(std::forward<Tip>(tip))
        , _base_others_fpe(std::forward<Others>(others)...)
    {
    }

    using _base_tip_fpe::do_get_facet;
    using _base_others_fpe::do_get_facet;
};


template <typename ... FPE>
using facets_pack_base_t = stringify::v0::detail::facets_pack_base<0, FPE...>;

#endif // defined (__cpp_variadic_using) #else

template <typename T>
struct pack_arg // rvalue reference
{
    static_assert( stringify::v0::facet_stored_by_value<T>
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
        < stringify::v0::facet_stored_by_value<T>
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
    : public stringify::v0::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<const std::reference_wrapper<T>>
    : public stringify::v0::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<std::reference_wrapper<T>&>
    : public stringify::v0::detail::pack_arg_ref<T>
{
};

template <typename T>
struct pack_arg<const std::reference_wrapper<T>&>
    : public stringify::v0::detail::pack_arg_ref<T>
{
};

} // namespace detail

template <template <class> class Filter, typename FPE>
class constrained_fpe
{
public:

    static_assert( stringify::v0::is_constrainable_v<FPE>
                 , "Not constrainable");

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_default_constructible<F>::value> >
    constexpr explicit constrained_fpe()
    {
    }

    constexpr constrained_fpe(const constrained_fpe&) = default;

    constexpr constrained_fpe(constrained_fpe&& other) = default;

    template
        < typename F = FPE
        , typename = std::enable_if_t<std::is_copy_constructible<F>::value> >
    constexpr explicit constrained_fpe(const FPE& f)
        : _fpe(f)
    {
    }

    template
        < typename U
        , typename = std::enable_if_t<std::is_constructible<FPE, U&&>::value> >
    constexpr explicit constrained_fpe(U&& arg)
        : _fpe(std::forward<U>(arg))
    {
    }

    constexpr constrained_fpe& operator=(const constrained_fpe&) = delete;
    constexpr constrained_fpe& operator=(constrained_fpe&&) = delete;

    constexpr const FPE& get() const
    {
        return _fpe;
    }

    constexpr FPE& get()
    {
        return _fpe;
    }

private:
    static_assert
        ( std::is_lvalue_reference<FPE>::value
       || stringify::v0::facet_stored_by_value<std::remove_reference_t<FPE>>
        , "This facet must not be stored by value" );

    FPE _fpe; // todo check of facet_stored_by_value
};


template <>
class facets_pack<>
{
public:
    constexpr facets_pack(const facets_pack&) = default;
    constexpr facets_pack(facets_pack&&) = default;
    constexpr facets_pack() = default;

    facets_pack& operator=(const facets_pack&) = delete;
    facets_pack& operator=(facets_pack&&) = delete;

    template <typename Category, typename Tag>
    decltype(auto) get_facet() const
    {
        return Category::get_default();
    }
};

template <typename ... FPE>
class facets_pack: private stringify::v0::detail::facets_pack_base_t<FPE...>
{
    using _base_type = stringify::v0::detail::facets_pack_base_t<FPE...>;
    using _base_type::do_get_facet;

    template <typename Tag, typename Category>
    constexpr decltype(auto) do_get_facet
        ( const stringify::v0::detail::absolute_lowest_rank&
        , stringify::v0::detail::identity<Category>
        , ... ) const
    {
        return Category::get_default();
    }

public:

    constexpr facets_pack(const facets_pack&) = default;
    constexpr facets_pack(facets_pack&& other) = default;

    template
        < bool Dummy = true
        , typename = std::enable_if_t
            < detail::fp_args<Dummy>::template copy_constructible_v<FPE...> > >
    constexpr explicit facets_pack(const FPE& ... fpe)
        : _base_type(fpe...)
    {
    }

    template
        < typename ... U
        , typename = std::enable_if_t
            < sizeof...(U) == sizeof...(FPE)
           && sizeof...(U) == 1
           && detail::fold_and<std::is_constructible<FPE, U&&>::value...> > >
    constexpr explicit facets_pack(U&& ... fpe)
        : _base_type(std::forward<U>(fpe)...)
    {
    }

    facets_pack& operator=(const facets_pack&) = delete;

    facets_pack& operator=(facets_pack&&) = delete;

    template <typename Category, typename Tag>
    constexpr decltype(auto) get_facet() const
    {
        return this->template do_get_facet<Tag>
            ( stringify::v0::detail::rank<sizeof...(FPE)>()
            , stringify::v0::detail::identity<Category>()
            , std::true_type() );
    }
};

template <typename ... T>
constexpr auto pack(T&& ... args)
{
    return stringify::v0::facets_pack
        < typename detail::pack_arg<T>::elem_type ... >
        { detail::pack_arg<T>::forward(args)... };
}

template <template <class> class Filter, typename T>
constexpr auto constrain(T&& x)
{
    return stringify::v0::constrained_fpe
        < Filter, typename detail::pack_arg<T>::elem_type >
        ( detail::pack_arg<T>::forward(x) );
}

template
    < typename FacetCategory
    , typename Tag
    , typename ... FPE >
constexpr decltype(auto) get_facet(const stringify::v0::facets_pack<FPE...>& fp)
{
    return fp.template get_facet<FacetCategory, Tag>();
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_PACK_HPP

