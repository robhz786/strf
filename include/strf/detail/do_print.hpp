#ifndef STRF_DETAIL_DO_PRINT_HPP
#define STRF_DETAIL_DO_PRINT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

namespace do_print_ {

using strf::detail::mp_type_list;

template <typename Arg>
struct print_arg_validator
{
    static constexpr bool is_fpe       = strf::detail::is_fpe<Arg>::value;
    static constexpr bool is_printable = strf::detail::is_printable<Arg>::value;

    static_assert( ! (is_fpe && is_printable)
                 , "type is both Printable and FacetPackElement");

    static_assert( ! (! is_fpe && ! is_printable)
                 , "type is not Printable nor FacetPackElement");
};

template <typename Arg>
struct assert_is_printable
{
    static_assert(strf::detail::is_printable<Arg>::value, "Type is not Printable");
    using type = Arg;
};

template <typename Arg>
using assert_is_printable_t = typename assert_is_printable<Arg>::type;

template <bool FirstArgIsFpe, typename... Args>
struct separate_args_2;

template <typename... Args>
struct separate_args_2<false, Args...>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<assert_is_printable_t<Args>...>;
};

template <typename Arg>
struct separate_args_2<true, Arg>
{
    using fpes = mp_type_list<Arg>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename Arg1, typename... OtherArgs>
struct separate_args_2<true, Arg0, Arg1, OtherArgs...>
{
    using validator = print_arg_validator<strf::detail::remove_cvref_t<Arg1>>;
    static constexpr bool arg1_is_fpe = validator::is_fpe;

    using continuation = separate_args_2<arg1_is_fpe, Arg1, OtherArgs...>;

    using continuation_fpes = typename continuation::fpes;
    using fpes = typename continuation_fpes::template add_front<Arg0>;

    using printables = typename continuation::printables;
};


template <typename... Args>
struct separate_args;

template <>
struct separate_args<>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename... OtherArgs>
struct separate_args<Arg0, OtherArgs...>
{
    using validator = print_arg_validator<strf::detail::remove_cvref_t<Arg0>>;
    static constexpr bool arg0_is_fpe = validator::is_fpe;

    using helper = separate_args_2<arg0_is_fpe, Arg0, OtherArgs...>;

    using fpes = typename helper::fpes;
    using printables = typename helper::printables;
};

template <typename FpesList, typename PrintablesList>
class print_impl;

template <typename... Fpes, typename... Printables>
class print_impl<mp_type_list<Fpes...>, mp_type_list<Printables...>>
{
public:

    template <typename ReservePolicy, typename DestCreator>
    using return_type =
        typename ReservePolicy::template return_type<DestCreator>;

private:

    template <typename CharT>
    static inline STRF_HD const strf::stringifier<CharT>&
    as_stringifier_cref_(const strf::stringifier<CharT>& p) noexcept
    {
        return p;
    }

public:

    template <bool Ln, typename ReservePolicy, typename DestCreator>
    static return_type<ReservePolicy, DestCreator> print
        ( ReservePolicy reserve_policy
        , const DestCreator& dest_creator
        , Fpes... fpes
        , Printables... printables )
    {
        using preprinting_t = typename ReservePolicy::preprinting_type;
        using char_type = typename DestCreator::char_type;
        preprinting_t pre;
        auto fp = strf::pack((Fpes&&)fpes...);
        return reserve_policy.template print<Ln>
            ( dest_creator, pre
            , as_stringifier_cref_<char_type>
                ( strf::stringifier_type
                  < char_type, preprinting_t, decltype(fp), Printables >
                    ( strf::make_stringifier_input<char_type>
                        ( pre, fp, (Printables&&)printables ) ) )... );
    }
};

template <bool Ln, typename ReservePolicy, typename DestCreator, typename... Args>
inline typename ReservePolicy::template return_type<DestCreator> do_print
    ( ReservePolicy reserve_policy
    , const DestCreator& dest_creator
    , Args&&... args )
{
    using separated_arg_types = strf::do_print_::separate_args<Args...>;
    using fpes_type_list = typename separated_arg_types::fpes;
    using printables_type_list = typename separated_arg_types::printables;
    using impl = strf::do_print_::print_impl<fpes_type_list, printables_type_list>;
    return impl::template print<Ln>(reserve_policy, dest_creator, (Args&&)args...);
}

} // namespace do_print_
} // namespace strf

#endif  // STRF_DETAIL_DO_PRINT_HPP

