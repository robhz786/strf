#ifndef STRF_PRINTING_SYNTAX_HPP
#define STRF_PRINTING_SYNTAX_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/separate_args_and_print.hpp>

namespace strf {

template <typename DestCreator, typename Policy, typename... Facets>
class printing_syntax;

namespace detail {

template <typename DestCreator>
struct can_create_dst_with_size
{
    template < typename T
             , typename Dst = typename T::sized_destination_type
             , typename = decltype(Dst(std::declval<T&>().create((std::size_t)0))) >
    static STRF_HD std::true_type test(const T&);

    template <typename>
    static STRF_HD std::false_type test(...);

    using result = decltype(test<DestCreator>(std::declval<DestCreator>()));

    static constexpr bool value = result::value;
};


template <typename DestCreator>
struct can_create_dst_without_size
{
    template < typename T
             , typename Dst = typename T::destination_type
             , typename = decltype(Dst(std::declval<T>().create())) >
    static STRF_HD std::true_type test(const T&);

    template <typename>
    static STRF_HD std::false_type test(...);

    using result = decltype(test<DestCreator>(std::declval<DestCreator>()));

    static constexpr bool value = result::value;
};



template <typename DestCreator, typename Policy, typename... Facets>
class reserve_funcs;

template <typename DestCreator, typename Policy, typename... FPEs>
class no_reserve_return_new_pss
{
private:
    using new_poli_ = strf::no_reserve;

    using new_pss_ = printing_syntax
        <DestCreator, new_poli_, FPEs...>;

    using this_pss_ = printing_syntax<DestCreator, Policy, FPEs...>;

public:

    template < typename FP = strf::facets_pack<FPEs...>
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<FP>::value
                && std::is_copy_constructible<DC>::value
                 , int > = 0 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ no_reserve() const &
    {
        const auto& self = static_cast<const this_pss_&>(*this);
        return { self.dest_creator_
               , self.fpack_};
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ no_reserve() &&
    {
        auto& self = static_cast<this_pss_&>(*this);
        return { (DestCreator&&) self.dest_creator_
               , new_poli_{}
               , std::move(self.fpack_) };
    }
};


template <typename DestCreator, typename Policy, typename... FPEs>
class reserve_calc_return_new_pss
{
private:
    using new_poli_ = strf::reserve_calc;

    using new_pss_ = printing_syntax
        <DestCreator, new_poli_, FPEs...>;

    using this_pss_ = printing_syntax<DestCreator, Policy, FPEs...>;

public:

    template < typename FP = strf::facets_pack<FPEs...>
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<FP>::value
                && std::is_copy_constructible<DC>::value
                 , int > = 0 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ reserve_calc() const &
    {
        const auto& self = static_cast<const this_pss_&>(*this);
        return {self.dest_creator_, new_poli_{}, self.fpack_};
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ reserve_calc() &&
    {
        auto& self = static_cast<this_pss_&>(*this);
        return { (DestCreator&&) self.dest_creator_
               , new_poli_{}
               , std::move(self.fpack_) };
    }
};


template <typename DestCreator, typename Policy, typename... FPEs>
class reserve_given_space_return_new_pss
{
private:
    using new_poli_ = strf::reserve_given_space;

    using new_pss_ = printing_syntax
        <DestCreator, new_poli_, FPEs...>;

    using this_pss_ = printing_syntax<DestCreator, Policy, FPEs...>;

public:

    template < typename IntT
             , typename FP = strf::facets_pack<FPEs...>
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_integral<IntT>::value
                && std::is_copy_constructible<FP>::value
                && std::is_copy_constructible<DC>::value
                 , int > = 0 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ reserve(IntT space) const &
    {
        const auto& self = static_cast<const this_pss_&>(*this);
        return { self.dest_creator_
               , new_poli_{detail::safe_cast_size_t(space)}
               , self.fpack_ };
    }

    template < typename IntT
             , strf::detail::enable_if_t<std::is_integral<IntT>::value, int> =0>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD new_pss_ reserve(IntT space) &&
    {
        auto& self = static_cast<this_pss_&>(*this);
        return { (DestCreator&&) self.dest_creator_
               , new_poli_{detail::safe_cast_size_t(space)}
               , std::move(self.fpack_) };
    }
};


template <typename DestCreator, typename... FPEs>
class reserve_funcs<DestCreator, strf::no_reserve, FPEs...>
    : public reserve_calc_return_new_pss<DestCreator, strf::no_reserve, FPEs...>
    , public reserve_given_space_return_new_pss<DestCreator, strf::no_reserve, FPEs...>
{
    using this_pss_ = printing_syntax<DestCreator, strf::no_reserve, FPEs...>;

    static_assert( can_create_dst_without_size<DestCreator>::value
                 , "no_reserve policy not applicable to this DestinationCreator" );

public:
    constexpr reserve_funcs() = default;

    constexpr STRF_HD explicit reserve_funcs(strf::no_reserve)
    {
    }
    constexpr STRF_HD strf::no_reserve get_reserve_policy() const
    {
        return {};
    }
    constexpr STRF_HD const this_pss_& no_reserve() const &
    {
        return static_cast<const this_pss_&>(*this);
    }
    constexpr STRF_HD const this_pss_&& no_reserve() const &&
    {
        return static_cast<const this_pss_&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_& no_reserve() &
    {
        return static_cast<this_pss_&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_&& no_reserve() &&
    {
        return static_cast<this_pss_&&>(*this);
    }
};


template <typename DestCreator, typename... FPEs>
class reserve_funcs<DestCreator, strf::reserve_calc, FPEs...>
    : public no_reserve_return_new_pss<DestCreator, strf::reserve_calc, FPEs...>
    , public reserve_given_space_return_new_pss<DestCreator, strf::reserve_calc, FPEs...>
{
    using this_pss_ = printing_syntax
        <DestCreator, strf::reserve_calc, FPEs...>;

    static_assert( can_create_dst_with_size<DestCreator>::value
                 , "reserve_calc policy not applicable to this DestinationCreator" );

public:
    constexpr reserve_funcs() = default;

    constexpr STRF_HD explicit reserve_funcs(strf::reserve_calc)
    {
    }
    constexpr STRF_HD strf::reserve_calc get_reserve_policy() const
    {
        return {};
    }
    constexpr STRF_HD const this_pss_& reserve_calc() const &
    {
        return static_cast<const this_pss_&>(*this);
    }
    constexpr STRF_HD const this_pss_&& reserve_calc() const &&
    {
        return static_cast<const this_pss_&&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_& reserve_calc() &
    {
        return static_cast<this_pss_&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_&& reserve_calc() &&
    {
        return static_cast<this_pss_&&>(*this);
    }
};


template <typename DestCreator, typename... FPEs>
class reserve_funcs<DestCreator, strf::reserve_given_space, FPEs...>
    : public no_reserve_return_new_pss  <DestCreator, strf::reserve_given_space, FPEs...>
    , public reserve_calc_return_new_pss<DestCreator, strf::reserve_given_space, FPEs...>
{
    static_assert( can_create_dst_with_size<DestCreator>::value
                 , "reserve_given_space policy not applicable to this DestinationCreator" );
public:
    using reserve_policy_t = strf::reserve_given_space;

private:
    template <typename Poli>
    using pss_ = printing_syntax<DestCreator, Poli, FPEs...>;

    using this_pss_ = pss_<reserve_policy_t>;

    reserve_policy_t poli_;

public:

    constexpr reserve_funcs() = default;

    constexpr STRF_HD explicit reserve_funcs(strf::reserve_given_space poli)
        : poli_(poli)
    {
    }
    constexpr STRF_HD strf::reserve_given_space get_reserve_policy() const
    {
        return poli_;
    }

    template < typename FP = strf::facets_pack<FPEs...>
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<FP>::value
                && std::is_copy_constructible<DC>::value
                 , int > = 0 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_ reserve_given_space(std::size_t space) const &
    {
        auto& self = static_cast<const this_pss_&>(*this);
        return {self.dest_creator_, reserve_policy_t{space}, self.fpack_};
    }

    template < typename FP = strf::facets_pack<FPEs...>
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<FP>::value
                && std::is_copy_constructible<DC>::value
                 , int > = 0 >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_ reserve_given_space(std::size_t space) const &&
    {
        auto& self = static_cast<const this_pss_&>(*this);
        return {self.dest_creator_, reserve_policy_t{space}, self.fpack_};
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_& reserve_given_space(std::size_t space) &
    {
        poli_.space = space;
        return static_cast<pss_<reserve_policy_t>&>(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_&& reserve_given_space(std::size_t space) &&
    {
        poli_.space = space;
        return static_cast<pss_<reserve_policy_t>&&>(*this);
    }
};

template <typename DestCreator, typename... FPEs>
struct pss_is_move_constructible
{
    static constexpr bool value =
        ( std::is_move_constructible<DestCreator>::value
       && strf::detail::fold_and
          < std::is_move_constructible<FPEs>::value ... >
          :: value );
};

template <typename DestCreator, typename... FPEs>
struct pss_is_copy_constructible
{
    static constexpr bool value =
        ( std::is_copy_constructible<DestCreator>::value
       && strf::detail::fold_and
          < std::is_copy_constructible<FPEs>::value ... >
          :: value );
};


template <bool CanMove, typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_move_impl;

template <bool CanCopy, typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_copy_impl;

template < typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_move_impl <false, DestCreator, ReservePolicy, FPEs...>
{
};

template < typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_copy_impl <false, DestCreator, ReservePolicy, FPEs...>
{
};

template < typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_move_impl <true, DestCreator, ReservePolicy, FPEs...>
{
    template <typename... FPEs_>
    using pss_ = printing_syntax<DestCreator, ReservePolicy, FPEs_...>;
    using this_pss_ = pss_<FPEs...>;
    using fpack_t_ = strf::facets_pack<FPEs...>;

public:

    template <typename NewFPE, typename... OtherNewFPEs>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    pss_< FPEs...
        , strf::detail::remove_cvref_t<NewFPE>
        , strf::detail::remove_cvref_t<OtherNewFPEs>...>
    with(NewFPE&& NewFpe, OtherNewFPEs&&... otherNewFpes) &&
    {
        auto& self = static_cast<this_pss_&>(*this);
        return { (DestCreator&&) self.dest_creator_
               , self.get_reserve_policy()
               , strf::merge_1st{}
               , (fpack_t_&&)self.fpack_
               , (NewFPE&&) NewFpe
               , (OtherNewFPEs&&)otherNewFpes... };
    }
};

template <typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs_copy_impl <true, DestCreator, ReservePolicy, FPEs...>
{
    template <typename... F>
    using pss_ = printing_syntax<DestCreator, ReservePolicy, F...>;

    using this_pss_ = pss_<FPEs...>;
    using fpack_t_ = strf::facets_pack<FPEs...>;

public:

    template <typename NewFPE, typename... OtherNewFPEs>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    pss_< FPEs...
        , strf::detail::remove_cvref_t<NewFPE>
        , strf::detail::remove_cvref_t<OtherNewFPEs>... >
    with(NewFPE&& NewFpe, OtherNewFPEs&&... otherNewFpes) const &
    {
        const auto& self = static_cast<const this_pss_&>(*this);
        return { self.dest_creator_
               , self.get_reserve_policy()
               , strf::merge_1st{}
               , self.fpack_
               , (NewFPE&&) NewFpe
               , (OtherNewFPEs&&)otherNewFpes... };
    }
};

template <typename DestCreator, typename ReservePolicy, typename... FPEs>
using fpe_funcs_copy = fpe_funcs_copy_impl
    < pss_is_copy_constructible<DestCreator, FPEs...>::value
    , DestCreator
    , ReservePolicy
    , FPEs...>;

template <typename DestCreator, typename ReservePolicy, typename... FPEs>
using fpe_funcs_move = fpe_funcs_move_impl
    < pss_is_move_constructible<DestCreator, FPEs...>::value
    , DestCreator
    , ReservePolicy
    , FPEs...>;

template <typename DestCreator, typename ReservePolicy, typename... FPEs>
class fpe_funcs
    : public fpe_funcs_copy<DestCreator, ReservePolicy, FPEs...>
    , public fpe_funcs_move<DestCreator, ReservePolicy, FPEs...>
{

    using this_pss_ = printing_syntax<DestCreator, ReservePolicy, FPEs...>;

public:
    using fpe_funcs_copy<DestCreator, ReservePolicy, FPEs...>::with;
    using fpe_funcs_move<DestCreator, ReservePolicy, FPEs...>::with;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_&& with() &&
    {
        return static_cast<this_pss_&&>(*this);
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD this_pss_& with() &
    {
        return static_cast<this_pss_&&>(*this);
    }

    constexpr STRF_HD const this_pss_&& with() const &&
    {
        return static_cast<const this_pss_&&>(*this);
    }

    constexpr STRF_HD const this_pss_& with() const &
    {
        return static_cast<const this_pss_&&>(*this);
    }
};

} // namespace detail

template < typename DestCreator
         , typename ReservePolicy = strf::no_reserve
         , typename... FPEs>
class printing_syntax
    : public detail::reserve_funcs<DestCreator, ReservePolicy, FPEs...>
    , public detail::fpe_funcs<DestCreator, ReservePolicy, FPEs...>
{
private:
    DestCreator dest_creator_;
    strf::facets_pack<FPEs...> fpack_;

    template <class, class, class...>
    friend class ::strf::detail::reserve_funcs;

    template <class, class, class...>
    friend class ::strf::detail::no_reserve_return_new_pss;

    template <class, class, class...>
    friend class ::strf::detail::reserve_calc_return_new_pss;

    template <class, class, class...>
    friend class ::strf::detail::reserve_given_space_return_new_pss;

    template <bool, class, class, class...>
    friend class ::strf::detail::fpe_funcs_copy_impl;

    template <bool, class, class, class...>
    friend class ::strf::detail::fpe_funcs_move_impl;

    template <class, class, class...>
    friend class ::strf::detail::fpe_funcs;

    using reserve_funcs_t_ = ::strf::detail::reserve_funcs<DestCreator, ReservePolicy, FPEs...>;
    using fpe_funcs_t_ = ::strf::detail::fpe_funcs<DestCreator, ReservePolicy, FPEs...>;

public:

    using char_type = typename DestCreator::char_type;

    constexpr printing_syntax() = default;

    template < typename DC = DestCreator
             , typename Poly = ReservePolicy
             , std::ptrdiff_t NumFpes = sizeof...(FPEs)
             , strf::detail::enable_if_t
                 < NumFpes == 0
                && std::is_copy_constructible<DC>::value
                && std::is_default_constructible<Poly>::value
                 , int > = 0 >
    constexpr STRF_HD explicit printing_syntax(const DestCreator& destCreator)
        : dest_creator_(destCreator)
    {
    }

    template < typename DC = DestCreator
             , typename Poly = ReservePolicy
             , std::ptrdiff_t NumFpes = sizeof...(FPEs)
             , strf::detail::enable_if_t
                 < NumFpes == 0
                && std::is_move_constructible<DC>::value
                && std::is_default_constructible<Poly>::value
                 , int > = 0 >
    constexpr STRF_HD explicit printing_syntax(DestCreator&& destCreator)
        : dest_creator_(std::move(destCreator))
    {
    }

    template < typename... otherFPEs
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<DC>::value
                && std::is_constructible<strf::facets_pack<FPEs...>, otherFPEs...>::value
                 , int > = 0 >
    constexpr STRF_HD printing_syntax
        ( const DestCreator& destCreator
        , ReservePolicy poli
        , otherFPEs&&... fpes )
        : reserve_funcs_t_(poli)
        , dest_creator_(destCreator)
        , fpack_((otherFPEs&&)fpes...)
    {
    }

    template < typename... otherFPEs
             , typename DC = DestCreator
             , strf::detail::enable_if_t
                 < std::is_move_constructible<DC>::value
                && std::is_constructible<strf::facets_pack<FPEs...>, otherFPEs...>::value
                 , int > = 0 >
    constexpr STRF_HD printing_syntax
        ( DestCreator&& destCreator
        , ReservePolicy poli
        , otherFPEs&&... fpes )
        : reserve_funcs_t_(poli)
        , dest_creator_(std::move(destCreator))
        , fpack_((otherFPEs&&)fpes...)
    {
    }

    using return_type = typename ReservePolicy::template return_type<DestCreator>;

    template <typename... Args>
    inline return_type STRF_HD operator()(Args&& ... args) const &
    {
        return detail::separate_args_and_print<false>
            ( this->get_reserve_policy(), dest_creator_, fpack_, (Args&&)args...);
    }

    template <typename... Args>
    inline return_type STRF_HD line(Args&& ... args) const &
    {
        return detail::separate_args_and_print<true>
            ( this->get_reserve_policy(), dest_creator_, fpack_, (Args&&)args...);
    }

    template <typename... Args>
    inline return_type STRF_HD tr(Args&& ... args) const &
    {
        return detail::separate_tr_args_and_print<false>
            ( this->get_reserve_policy(), dest_creator_, fpack_, (Args&&)args...);
    }

    template <typename... Args>
    inline return_type STRF_HD trline(Args&& ... args) const &
    {
        return detail::separate_tr_args_and_print<true>
            ( this->get_reserve_policy(), dest_creator_, fpack_, (Args&&)args...);
    }
};

template <typename DestCreator>
constexpr STRF_HD
printing_syntax<strf::detail::remove_cvref_t<DestCreator>>
make_printing_syntax(DestCreator&& dc)
{
    return {dc, strf::no_reserve{}};
}

template <typename DestCreator, typename ReservePolicy, typename... FPEs>
constexpr STRF_HD printing_syntax
    < strf::detail::remove_cvref_t<DestCreator>
    , ReservePolicy
    , strf::detail::remove_cvref_t<FPEs>... >
make_printing_syntax(DestCreator&& dc, ReservePolicy poli, FPEs&&... fpe)
{
    return {(DestCreator&&)dc, poli, (FPEs&&)fpe...};
}

namespace detail {

template <typename CharT>
class destination_reference
{
public:

    using char_type = CharT;
    using destination_type = strf::destination<CharT>&;

    explicit STRF_HD destination_reference(strf::destination<CharT>& dst) noexcept
        : dst_(&dst)
    {
    }

    STRF_HD strf::destination<CharT>& create() const noexcept
    {
        return *dst_;
    }

private:
    strf::destination<CharT>* dst_;
};


} // namespace detail

template <typename CharT>
strf::printing_syntax<strf::detail::destination_reference<CharT>>
STRF_HD to(strf::destination<CharT>& dst) noexcept
{
    return strf::make_printing_syntax
        (strf::detail::destination_reference<CharT>{dst});
}

namespace detail {

template <typename CharT>
class basic_cstr_destination_creator
{
public:

    using char_type = CharT;
    using finish_type = typename basic_cstr_destination<CharT>::result;
    using destination_type = basic_cstr_destination<CharT>;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    basic_cstr_destination_creator(CharT* dst, CharT* dst_end) noexcept
        : dst_(dst)
        , dst_end_(dst_end)
    {
        STRF_ASSERT(dst < dst_end);
    }

    STRF_HD typename basic_cstr_destination<CharT>::range create() const noexcept
    {
        return typename basic_cstr_destination<CharT>::range{dst_, dst_end_};
    }

private:

    CharT* dst_;
    CharT* dst_end_;
};

template <typename CharT>
class array_destination_creator
{
public:

    using char_type = CharT;
    using finish_type = typename array_destination<CharT>::result;
    using destination_type = array_destination<CharT>;

    constexpr STRF_HD
    array_destination_creator(CharT* dst, CharT* dst_end) noexcept
        : dst_(dst)
        , dst_end_(dst_end)
    {
        STRF_ASSERT_IN_CONSTEXPR(dst < dst_end);
    }

    STRF_HD typename array_destination<CharT>::range create() const noexcept
    {
        return typename array_destination<CharT>::range{dst_, dst_end_};
    }

private:

    CharT* dst_;
    CharT* dst_end_;
};

}  // namespace detail

#if defined(__cpp_char8_t)

template<std::size_t N>
inline STRF_HD auto to(char8_t (&dst)[N]) noexcept
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        ({dst, dst + N});
}

inline STRF_HD auto to(char8_t* dst, char8_t* end) noexcept
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        ({dst, end});
}

inline STRF_HD auto to(char8_t* dst, std::size_t count) noexcept
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        ({dst, dst + count});
}

#endif

template<std::size_t N>
inline STRF_HD auto to(char (&dst)[N]) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
        ({dst, dst + N});
}

inline STRF_HD auto to(char* dst, char* end) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
        ({dst, end});
}

inline STRF_HD auto to(char* dst, std::size_t count) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char> >
        ({dst, dst + count});
}

template<std::size_t N>
inline STRF_HD auto to(char16_t (&dst)[N]) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        ({dst, dst + N});
}

inline STRF_HD auto to(char16_t* dst, char16_t* end) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        ({dst, end});
}

inline STRF_HD auto to(char16_t* dst, std::size_t count) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        ({dst, dst + count});
}

template<std::size_t N>
inline STRF_HD auto to(char32_t (&dst)[N]) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        ({dst, dst + N});
}

inline STRF_HD auto to(char32_t* dst, char32_t* end) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        ({dst, end});
}

inline STRF_HD auto to(char32_t* dst, std::size_t count) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        ({dst, dst + count});
}

template<std::size_t N>
inline STRF_HD auto to(wchar_t (&dst)[N]) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        ({dst, dst + N});
}

inline STRF_HD auto to(wchar_t* dst, wchar_t* end) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        ({dst, end});
}

inline STRF_HD auto to(wchar_t* dst, std::size_t count) noexcept
    -> strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printing_syntax
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        ({dst, dst + count});
}

template<typename CharT, std::size_t N>
inline STRF_HD auto to_range(CharT (&dst)[N]) noexcept
    -> strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
        ({dst, dst + N});
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dst, CharT* end) noexcept
    -> strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
        ({dst, end});
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dst, std::size_t count) noexcept
    -> strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printing_syntax
        < strf::detail::array_destination_creator<CharT> >
        ({dst, dst + count});
}

} // namespace strf

#endif  // STRF_PRINTING_SYNTAX_HPP

